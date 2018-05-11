#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides helper functions to georegister (add geo-referenced \
coordinates) to raster-based (image) data products. This module makes \
extensive use of osgeo.gdal.Dataset and osgeo.gdal.Dataset.GetGeoTransform
"""

from osgeo import osr, gdal
import numpy as np
import os
from . import features as feat
import rasterio


def geo_from_pix(pix, gt):
    """
    This function returns a georeference coordinate for an (x,y) pixel pair \
    given the affine geometric transform provided.

    Typically image coordinate systems (at least in GDAL, and OpenCV) are \
    defined with (x,y) = pixels_right, pixels_down. However images are stored\
    as Matrix/Array types where (x,y) = (row,col) = (down,right), so indexing\
    into either needs to be done with caution of how the functions operate.\
    In this case, we're working in the raster/image domain, so (x,y) refers to\
    all points right,down.

    :param nd.array pix: Nx2 matrix of (x,y) raster pixel coordinates to \
        transform to the coordinate system referenced by gt
    :param tuple gt: Length 6 tuple consisting of coefficients of the affine \
        transformation which converts coordinates in raster space to a \
        geometric coordinates system. Units are not defined, but are typically\
        meters, feet, or degrees. The affine transformation is defined by:

        .. math::

                x_{geo} = gt[0] + x_{raster}gt[1] + y_{raster}gt[2]

                y_{geo} = gt[3] + x_{raster}gt[4] + y_{raster}gt[5]

        If using gdal, this transforms is returned by \
        osgeo.gdal.Dataset.GetGeoTransform()

    :returns: numpy.ndarray -- Nx2 array of georeferenced coordinates of \
        raster(x,y) points provided in pix

    """
    if len(gt) != 6:
        raise ValueError("gt must be a 6 element tuple as defined by GDAL")
    if pix.ndim == 1 and pix.shape[0] == 2:
        pix = pix.reshape(1, 2)
    if pix.shape[1] != 2:
        raise ValueError("pix must be an Nx2 length numpy.ndarray")

    # Need a leading column of ones for the matrix multiply to work
    pix = np.hstack([np.ones(pix.shape[0]).reshape(pix.shape[0], 1), pix])
    gt_x = np.array(gt).reshape(2, 3).transpose()
    return np.dot(pix, gt_x)


def coord_transform_from_wkt(proj_ref_wkt, new_cs_wkt):
    """
    Returns an osr.CoordinateTransformation object which transforms from the \
    coordinate system described by proj_ref_wkt to the system described in\
    new_cs_wkt

    :param string proj_ref_wkt: WKT formatted string consisting of the \
        description of the reference projection of the Orhtophoto that\
        This object is typically called from \
        osgeo.gdal.Dataset.GetProjectionRef(), and further description \
        examples of WKT can be found at spatialreference.org
    :param string new_cs_wkt: WKT formatted string consisting of the \
        description of the new/targeted coordinate system that the returned \
        osr.CoordinateTransformation() object will transform points to
    :returns: osr.CoordinateTransformation() -- Object which transforms 2D/3D \
        coordinates from the system described in proj_ref_wkt to the \
        coordinate system described by new_cs_wkt
    """
    # Transform the features into WGS-84
    # What is the NITF/ophoto Referenced in?
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(proj_ref_wkt)

    # How about going to WGS-84?
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(new_cs_wkt)

    # The actual Tranfromation class/object
    transform = osr.CoordinateTransformation(old_cs, new_cs)
    return transform


def pix_from_geo(lon_lat, gt):
    """
    This is a function that pulls in a coordinate pair of georeferenced \
    coordinates along with an affine geometric transformation and performs \
    a quick least-squares search to determine the raster pixel location.

    :param nd.array(double) lon_lat: Nx2 matrix of "Longitude" or in GDAL the \
        geo-referenced x coordinate of the pixel to find in the first column \
        (units correspond to units in gt), and  "Latitude" or in GDAL the \
        geo-referenced y coordinate of the pixel to find (units correspond to \
        units in gt)
    :param tuple gt: Length 6 tuple consisting of coefficients of the affine \
        transformation which converts coordinates in raster space to a \
        geometric coordinates system. Units are not defined, but are typically\
        meters, feet, or degrees. The affine transformation is defined by:

        .. math::

                x_{geo} = gt[0] + x_{raster}gt[1] + y_{raster}gt[2]

                y_{geo} = gt[3] + x_{raster}gt[4] + y_{raster}gt[5]

        If using gdal, this transforms is returned by \
        osgeo.gdal.Dataset.GetGeoTransform()
    :returns: numpy.ndarray(double) -- Returns an Nx2 numpy.ndarray of Lon/Lat\
         coordinates corresponding to raster(x,y) points
    """
    if len(gt) != 6:
        raise ValueError("gt must be a 6 element tuple as defined by GDAL")
    if lon_lat.ndim == 1 and lon_lat.shape[0] == 2:
        lon_lat = lon_lat.reshape(1, 2)
    if lon_lat.shape[1] != 2:
        raise ValueError("pix must be an Nx2 length numpy.ndarray")

    # Here we set up a standard linear system Ax = b, and assuming A is
    # invertible (which based on this affine transform, it *should* be)
    # we go ahead and solve for X using x = inv(A)*b
    # TODO: use moore pseudoinverse if we come into non invertible A matrices

    A = np.array([[gt[1], gt[2]], [gt[4], gt[5]]])
    b = lon_lat.transpose() - np.tile(np.array([[gt[0]], [gt[3]]]),
                                      (1, lon_lat.shape[0]))
    return np.dot(np.linalg.inv(A), b).transpose()


class SRTM(object):

    """
    Class for using SRTM height files
    """

    def __init__(self, srtm_root_dir, geoid):
        self.root = srtm_root_dir
        self.geoid = geoid

    def find_srtm_tiles(self, lon_lat):
        """
        Give me a lon, lat WGS-84 (deg) Nx2 numpy.ndarray, retrun a list \
        of resolvable filepaths to SRTM tiles
        """
        file_names = []
        for ll in lon_lat:
            ns = 'N' if ll[1] > 0 else 'S'
            ew = 'E' if ll[0] > 0 else 'W'
            fname = ns + '%02d' % np.abs(np.floor(ll[1]))
            fname += ew + '%03d' % np.abs(np.floor(ll[0]))
            fname += '.hgt'
            file_names.append(os.path.join(self.root, fname))
        return file_names

    def msl_nn_srtm_interp(self, lon_lat):
        """
        Given an Nx2 numpy.ndarray of WGS-84 lon, lat, return the MSL values
        from SRTM. Returns np.nan for values that contain no-data
        """
        tiles = self.find_srtm_tiles(lon_lat)
        lon_lat_msl = np.zeros((lon_lat.shape[0], 3))
        lon_lat_msl[:, 0:2] = lon_lat
        for tile in set(tiles):
            otile = rasterio.open(tile, 'r')
            oimg = otile.read(1)
            idx = np.where(np.array(tiles) == tile)[0]
            pix = feat.geo_to_pix(
                otile.affine, lon_lat[idx, 0], lon_lat[idx, 1])
            pix = np.round(pix).astype(np.int)
            lon_lat_msl[idx, 2] = oimg[pix[:, 1], pix[:, 0]]
            otile.close()
        nan_mask = lon_lat_msl[:, 2] == -32768
        lon_lat_msl[nan_mask, 2] = np.NaN
        return lon_lat_msl

    def msl_to_wgs84(self, lon_lat_msl):
        """
        This function calculates the height of a point specified by latitude, \
        longitude, height above msl, and the Mean Sea Level is defined as:

        .. math::
          h_{wgs84}(lat,lon) = h_{msl}(lat,lon) + h_{geoid}(lat,lon)

            the geo-referenced x coordinate of the pixel to find in the first \
            column (units correspond to units in geoid's GetGeoTransform), \
            and "Latitude" or in GDAL the geo-referenced y coordinate of the \
            pixel to find where the 3rd column is height above MSL in m
        :returns: numpy.ndarray(double) -- Nx3 numpy array of [longitude, \
            latitude, height_above_wgs84_ellipsoid(m) ]
        """
        geoid = rasterio.open(self.geoid)
        lon_lat_hae = np.zeros_like(lon_lat_msl)
        lon_lat_hae[:, 0:2] = lon_lat_msl[:, 0:2]
        gimg = geoid.read(1)
        geo = feat.wgs84_to_geo(
            geoid.crs, lon_lat_msl[:, 0], lon_lat_msl[:, 1])
        pix = feat.geo_to_pix(geoid.affine, geo[:, 0], geo[:, 1])
        pix = np.round(pix).astype(np.int)
        hae = lon_lat_msl[:, 2] + gimg[pix[:, 1], pix[:, 0]]
        lon_lat_hae[:, 2] = hae
        return lon_lat_hae

    def add_heights(self, lon_lat):
        """
        Given an Nx2 numpy.ndarray of (Lon, Lat (deg)), return an Nx3 array \
        where the extra column is WGS-84 height above ellipsoid from both \
        SRTM data and the specified geoid shift file.
        """
        lon_lat_msl = self.msl_nn_srtm_interp(lon_lat)
        return self.msl_to_wgs84(lon_lat_msl)


class DTED(object):

    """
    This class provides a convenience wrapper for the DTED-related functions
    """

    def __init__(self, dted_directory, dted_extension, geoid):
        self.dted_dir = dted_directory
        self.dted_ext = dted_extension
        self.geoid = geoid

    def add_heights(self, lon_lat):
        pts_msl = nearest_neighbor_dted_interp(
            lon_lat, self.dted_dir, self.dted_ext)
        pts_hae = msl_to_wgs84(pts_msl, self.geoid)
        return pts_hae


def msl_to_wgs84(lon_lat_msl, geoid):
    """
    This function calculates the height of a point specified by latitude, \
    longitude, height above msl, and the Mean Sea Level is defined as:

    .. math::
      h_{wgs84}(lat,lon) = h_{msl}(lat,lon) + h_{geoid}(lat,lon)

        the geo-referenced x coordinate of the pixel to find in the first \
        column (units correspond to units in geoid's GetGeoTransform), and  \
        "Latitude" or in GDAL the geo-referenced y coordinate of the pixel to \
        find (units correspond to units in geoid's GetGeoTransform), where \
        the 3rd column is height above MSL in m
    :param osgeo.gdal.Dataset geoid: Raster of geoid shift file (.gtx) loaded \
        into osgeo.gdal.Dataset using osgeo.gdal.Open(...)
    :returns: numpy.ndarray(double) -- Nx3 numpy array of [longitude, \
        latitude, height_above_wgs84_ellipsoid(m) ]
    """

    if lon_lat_msl.shape[1] != 3:
        raise ValueError("pix must be an Nx3 length numpy.ndarray")

    # Get the geometric transform for the Geoid
    gt_geoid = geoid.GetGeoTransform()

    # Load the geoid-shift as an "image"
    band = geoid.GetRasterBand(1)
    image = band.ReadAsArray(0, 0, band.XSize, band.YSize)
    pix = pix_from_geo(lon_lat_msl[:, 0:2], gt_geoid)

    # Lookup, pixel(x,y) is in image coordinate frame, so we need to reverse
    # when we do a numpy lookup into the array, as they're rotated/reversed
    # e.g. image frames run x right, y down and numpy array corresponding to
    # the mean pixel x is a column (y) index, and vice versa.
    lon_lat_hae = np.hstack((lon_lat_msl[:, 0:2], np.zeros((pix.shape[0], 1))))
    for ii in np.arange(pix.shape[0]):
        lon_lat_hae[ii, 2] = lon_lat_msl[ii, 2] + image[pix[ii, 1], pix[ii, 0]]
    return lon_lat_hae


def nearest_neighbor_dted_interp(lon_lat, dted_directory, dted_extension):
    """
    This function returns DTED heights (m, MSL) for the points specified in\
    the Nx2 numpy.ndarray lon_lat (degrees) by performing a lookup to the\
    nearest raster-based location (integer pixel) to the DTED grid.

    :param nd.array(double) lon_lat: Nx2 numpy.ndarray of longitude and \
        latitude coordinates (degrees)
    :param string dted_directory: Full path to the base directory of DTED\
        file structure
    :param string dted_extension: Extension of the DTED tiles to use, e.g.\
        ('DT0', 'DT1')
    :returns: numpy.ndarray(double) -- Nx1 numpy arrary of height above \
        Mean Sea Level (m) for each row/coordinate pair in lon_lat
    """

    # Set these to NAN for the moment
    heights = np.empty((lon_lat.shape[0], 1))
    heights[:] = np.NAN

    # First find the corresponding dted files
    dted_files = find_containing_dted_file(lon_lat, dted_directory,
                                           dted_extension)

    # Go through each file
    for tile in set(dted_files):

        # Open DTED, get the image, and affine transform
        dted = gdal.Open(tile)
        gt_dted = dted.GetGeoTransform()
        band = dted.GetRasterBand(1)
        img = band.ReadAsArray(0, 0, dted.RasterXSize, dted.RasterYSize)

        # Get raster space idx for the points in this tile
        idx = np.nonzero(np.array(dted_files) == tile)[0]
        d_pix = pix_from_geo(lon_lat[idx, :], gt_dted)

        # Grab nearest neighbor
        d_pix = np.round(d_pix).astype('int')

        # Index into image to grab heights, remember that because the
        # coordinate systems between raster space and ndarray are rotated we
        # need to transpose x,y
        heights[idx] = img[d_pix[:, 1], d_pix[:, 0]].reshape((idx.shape[0], 1))

    # Put heights into msl
    lon_lat_msl = np.hstack((lon_lat, heights))
    return lon_lat_msl


def find_containing_dted_file(lon_lat, dted_directory, dted_extension):
    """
    This function returns a tuple of strings to the path of the DTED tile \
    containing the points in lon_lat

    :param nd.array(double) lon_lat: Nx2 matrix of Longitude in degrees West,\
        and Latitude in Degrees North
    :param string dted_directory: base-path for the dted directory
    :param string dted_extension: extension of the DTED files in the directory\
        structure pointed to by dted_directory, e.g. ('DT0',DT1')
    :returns: tuple -- N-length tuple of DTED files containing the points in \
        lon_lat
    """
    if lon_lat.shape[1] != 2:
        raise ValueError("pix must be an Nx2 length numpy.ndarray")
    dted_out = []
    for point in lon_lat:
        dted_out.append(os.path.abspath(dted_directory)
                        +
                        '/W%03d/n%d.' % (
                            np.abs(np.floor(point[0])), np.floor(point[1]))
                        + dted_extension)
    return tuple(dted_out)
