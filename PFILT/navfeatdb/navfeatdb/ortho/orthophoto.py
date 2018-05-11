#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides a command line function to create an interact with a
hdf5 stored mosiac created from various tiled orthophotos
"""

from osgeo import osr, gdal
import numpy as np
import os
import copy
import h5py
import yaml
from ..frames import coverage2d as georeg
from ..frames import rasterio2d
import mercantile
import rasterio
import rasterio.crs
import dask.array as da


class OrthoPhoto(object):

    def __init__(self, ophoto_file, **kwargs):
        raise NotImplementedError("This is an abstract class")

    def read(self, x_vals, y_vals):
        raise NotImplementedError("Use a child class")

    def get_img_from_lon_lat(self, lon, lat, m_north_south, m_east_west):
        """
        Grab a subimage from the HDF5 image, whose center is the nearest \
        pixel to the coordinate pair given by lon, lat. And the size of the \
        subimage is determined by the m_north_south, m_east_west. Subimages \
        will be truncated at the boundaries of the HDF5 file

        :param double lon: Longitude (deg) of desired center point of image, \
            Longitude should be in range of +/- 180 deg
        :param double lat: Latitude (deg) of desired center point of image, \
            Latitude in range +/- 90 deg
        :param double m_north_south: Desired north-south footprint of image
        :param double m_east_west: Desired east-west footprint of image
        :returns: hdf_orthophoto.OrthoImage() -- Class containing an \
            orthophoto whose center pixel is the nearest pixel to [Lon, Lat] \
            and contains the metadata needed to retrieve geographic info \
            for each pixel in the returned subimage
        """
        if self.is_closed:
            raise ValueError("HDF File is closed")
        else:
            geo = np.array([self.tf_wgs_to_geo.TransformPoint(lon, lat)])
            pix = np.round(georeg.pix_from_geo(geo[:, 0:2],
                                               self.gt)).astype(np.int)

            m_ns = np.round(
                (m_north_south / self.m_per_pix) / 2.0).astype('int')
            m_ew = np.round(
                (m_east_west / self.m_per_pix) / 2.0).astype('int')

            # Bound pixel corners
            pix_corners = np.tile(pix, (2, 1)) - np.array([[m_ew, m_ns],
                                                           [-1 * m_ew,
                                                            -1 * m_ns]])
            pix_corners = self.bound_pixel_corners(pix_corners)

            xv = (pix_corners[0, 1], pix_corners[1, 1] + 1)
            yv = (pix_corners[0, 0], pix_corners[1, 0] + 1)
            img = self.read(xv, yv)

            # Modify GT
            gt = copy.copy(self.gt)
            ul_corner_geo = georeg.geo_from_pix(pix_corners[0, :], self.gt)
            gt[0] = ul_corner_geo[0, 0]
            gt[3] = ul_corner_geo[0, 1]
            return OrthoImage(gt, self.cs, img)

    def bound_pixel_corners(self, pix_corners):
        """
        Bounds the corners of the 2x2 nd_array corner array which contains \
        the upper left corner on the first row, and image(x,y) of the \
        bottom right corner in the second row

        :param numpy.ndarray pix_corners: A 2x2 numpy.ndarray whose first row \
            represents the image x,y pixel locations of the upper left corner \
            of the desired subimage, and the second row representing the \
            image x,y pixel locations of the lower-right corner
        :returns: numpy.ndarray -- 2x2 pix_corners matrix with array indices \
            modified if the requested indices are out of bounds of the image \
            stored in the HDF5 file
        """
        pix_corners[np.where(pix_corners < 0)] = 0
        over_x = np.where(pix_corners[:, 0] > self.x_size)
        pix_corners[over_x, 0] = self.x_size - 1
        over_y = np.where(pix_corners[:, 1] > self.y_size)
        pix_corners[over_y, 1] = self.y_size - 1
        return pix_corners

    def get_img_from_lon_lat_by_pix(self, lon, lat, half_width, half_height):
        """
        Grab a subimage from the HDF5 image, whose center is the nearest \
        pixel to the coordinate pair given by lon, lat. And the size of the \
        subimage is determined by the pixel values half_height, half_width. \
        Subimages will be truncated at the boundaries of the HDF5 file

        :param double lon: Longitude (deg) of desired center point of image, \
            Longitude should be in range of +/- 180 deg
        :param double lat: Latitude (deg) of desired center point of image, \
            Latitude in range +/- 90 deg
        :param double half_width: One Half of the Desired east-west pixel \
            footprint of image
        :param double half_height: One half of the Desired north-south pixel \
            footprint of image
        :returns: hdf_orthophoto.OrthoImage() -- Class containing an \
            orthophoto whose center pixel is the nearest pixel to [Lon, Lat] \
            and contains the metadata needed to retrieve geographic info \
            for each pixel in the returned subimage
        """
        if self.is_closed:
            raise ValueError("HDF File is closed")
        else:
            geo = np.array([self.tf_wgs_to_geo.TransformPoint(lon, lat)])
            pix = np.round(georeg.pix_from_geo(geo[:, 0:2],
                                               self.gt)).astype(np.int)
        pix = pix.flatten()
        pix_corners = np.array([[pix[0] - half_width, pix[1] - half_height],
                                [pix[0] + half_width, pix[1] + half_height]])
        pix_corners = pix_corners.astype(int)
        pix_corners = self.bound_pixel_corners(pix_corners)

        xv = (pix_corners[0, 1], pix_corners[1, 1])
        yv = (pix_corners[0, 0], pix_corners[1, 0])
        img = self.read(xv, yv)

        gt = copy.copy(self.gt)
        ul_corner_geo = georeg.geo_from_pix(pix_corners[0, :], self.gt)

        gt[0] = ul_corner_geo[0, 0]
        gt[3] = ul_corner_geo[0, 1]
        return OrthoImage(gt, self.cs, img)

    def get_img_from_pix_by_pix(self, pix_x, pix_y, x_size, y_size):
        """
        Grab a subimage from the HDF5 image, whose upper left corner is the \
        pixel value specified by pix_x and pix_y, and whose x_size and y_size \
        are the sizes of the sub image

        :param int pix_x: Upper Left pixel X value of the desired subimage
        :param int pix_y: Upper Left pixel Y value of the desired subimage
        :param int x_size: Size of the X value (width) of the subimage
        :param int y_size: Size of the Y value (height) of the subimage
        :returns: hdf_orthophoto.OrthoImage() -- Class containing an \
            orthophoto whose upper left corner is at the Pixel(pix_x, pix_y) \
            and contains the metadata needed to retrieve geographic info \
            for each pixel in the returned subimage
        """
        if self.is_closed:
            raise ValueError("HDF File is closed")
        else:
            pix = np.array([pix_x, pix_y])
        pix_corners = np.array([[pix[0], pix[1]],
                                [pix[0] + x_size, pix[1] + y_size]])
        pix_corners = pix_corners.astype(int)
        pix_corners = self.bound_pixel_corners(pix_corners)

        xv = (pix_corners[0, 1], pix_corners[1, 1])
        yv = (pix_corners[0, 0], pix_corners[1, 0])
        img = self.read(xv, yv)

        gt = copy.copy(self.gt)
        ul_corner_geo = georeg.geo_from_pix(pix_corners[0, :], self.gt)

        gt[0] = ul_corner_geo[0, 0]
        gt[3] = ul_corner_geo[0, 1]
        return OrthoImage(gt, self.cs, img)

    def get_img_from_slice(self, slc):
        """
        Grab a subimage from the HDF5 image, whose pixel bounds are given
        by the slice tuple

        :param tuple slc: Tuple of the form ((rstart, rstop), (cstart, cstop))
        :returns: hdf_orthophoto.OrthoImage() -- Class containing an \
            orthophoto whose upper left corner is at the Pixel(cstart, rstart) \
            and contains the metadata needed to retrieve geographic info \
            for each pixel in the returned subimage
        """
        if self.is_closed:
            raise ValueError("HDF File is closed")
        else:
            pix = np.array([slc[0][0], slc[1][0]])


        img = self.read(slc[0], slc[1])
        gt = copy.copy(self.gt)
        ul_corner_geo = georeg.geo_from_pix(pix[[1, 0]], self.gt)

        gt[0] = ul_corner_geo[0, 0]
        gt[3] = ul_corner_geo[0, 1]
        return OrthoImage(gt, self.cs, img)


    def get_img_from_tile(self, tile):
        """
        Grab a subimage from the HDF5 image, whose bounds are calculated \
        from the mercantile tile object

        :param mercantile.Tile: named tuple of spherical mercator z/x/y \
            coordinates whose bounds will be used to return the subimage
        :returns: hdf_orthophoto.OrthoImage() -- Class containing an \
            orthophoto whose bounds are the calculated from tile
        """
        bounds = mercantile.bounds(tile.x, tile.y, tile.z)
        return self.get_img_from_bounds(bounds)

    def get_img_from_bounds(self, bounds):
        """
        Grab a subimage from the HDF5 image, whose bounds are given

        :param mercantile.bounds: named tuple of spherical mercator bounding \
            box that will be used to return the subimage
        :returns: hdf_orthophoto.OrthoImage() -- Class containing an \
            orthophoto whose bounds are the calculated from tile
        """
        if self.is_closed:
            raise ValueError("HDF File is closed")
        geo = np.vstack((self.tf_wgs_to_geo.TransformPoint(bounds.west,
                                                           bounds.north),
                         self.tf_wgs_to_geo.TransformPoint(bounds.east,
                                                           bounds.south)))
        pix = georeg.pix_from_geo(geo[:, 0:2], self.gt)
        pix[0, :] = np.floor(pix[0, :])
        pix[1, :] = np.ceil(pix[1, :])
        pix = pix.astype(np.int)
        pix_corners = self.bound_pixel_corners(pix)

        xv = (pix_corners[0, 1], pix_corners[1, 1])
        yv = (pix_corners[0, 0], pix_corners[1, 0])
        img = self.read(xv, yv)

        gt = copy.copy(self.gt)
        ul_corner_geo = georeg.geo_from_pix(pix_corners[0, :], self.gt)

        gt[0] = ul_corner_geo[0, 0]
        gt[3] = ul_corner_geo[0, 1]
        return OrthoImage(gt, self.cs, img)


class VRTOrthophoto(OrthoPhoto):
    """
    This class implements a VRT Driver for OrthoPhotos
    """
    def __init__(self, vrt_file, **kwargs):
        self.is_closed = False
        self.vrt = rasterio.open(vrt_file, 'r')
        self.x_size = self.vrt.width
        self.y_size = self.vrt.height
        self.shape = self.vrt.shape
        self.num_bands = self.vrt.meta['count']

        base_cs = self.vrt.crs
        osr_ref = osr.SpatialReference()
        osr_ref.ImportFromProj4(rasterio.crs.CRS.to_string(base_cs))
        base_wkt = osr_ref.ExportToWkt()

        # Now grab the GeoRegistration data, both the geo transform and the
        # CS Wkt stored as attributes in /ophoto
        self.gt = self.vrt.get_transform()
        self.cs = osr.SpatialReference(yaml.dump(base_wkt))
        if self.cs.IsProjected():
            self.m_per_pix = self.cs.GetLinearUnits()
        else:
            self.m_per_pix = None

        if not self.cs.IsGeographic():
            self.m_per_pix = self.m_per_pix * self.gt[1]

        # Set Corner Lat-Lon
        new_cs_wkt = osr.GetWellKnownGeogCSAsWKT('wgs84')
        self.tf_to_wgs84 = georeg.coord_transform_from_wkt(
            self.cs.ExportToWkt(), new_cs_wkt)
        self.tf_wgs_to_geo = georeg.coord_transform_from_wkt(
            new_cs_wkt, self.cs.ExportToWkt())

        c_g = np.zeros((4, 2))
        # Upper left, upper right, lower left, lower right
        # Image frame = +x right, +y down... Numpy = +x down, +y right
        c_g[0, :] = georeg.geo_from_pix(np.array([0, 0]), self.gt)
        c_g[1, :] = georeg.geo_from_pix(np.array([self.x_size, 0]), self.gt)
        c_g[2, :] = georeg.geo_from_pix(np.array([0, self.y_size]), self.gt)
        c_g[3, :] = georeg.geo_from_pix(np.array([self.x_size, self.y_size]),
                                        self.gt)
        self.corners_geo = c_g
        try:
            self.corners_wgs = np.array(
                self.tf_to_wgs84.TransformPoints(c_g))[:, 0:2]
        except:
            self.hdf.close()
            raise ValueError("HDF5 WKT for reference system was most \
                              likely inusfficent to perform transform")

    def close(self):
        """
        Closes the vrt file
        """
        self.vrt.close()

    def read(self, xv, yv):
        img = self.vrt.read(window=(xv, yv))
        if img.ndim > 2:
            img = np.rollaxis(img, 0, 3)
        return img


class HDFOrthophoto(OrthoPhoto):

    """
    This class provides an object capable of interacting with an orthophoto\
    stored in the HDF5 format. Methods are available for retrieving\
    georeferenced sub images based on Lat/Lon queries

    :param string hdf5_file: Full path to the HDF5 file where the orthophoto \
        and associated geodata is stored
    """

    def __init__(self, hdf5_file, **kwargs):
        """
        Class constructor, attempts to open the HDF based orthoPhoto at\
        hdf5_file, checks to make sure that the Image has valid metadata\
        """
        img_path = '/ophoto/gray'
        if 'img_path' in kwargs.keys():
            img_path = kwargs['img_path']
        self.img_path = img_path
        self.hdf = self.open_hdf_file(hdf5_file)
        self.check_hdf_metadata()
        self.is_closed = False

    def close(self):
        """
        Closes the HDF File
        """
        self.is_closed = True
        self.hdf.close()

    def open_hdf_file(self, hdf_full_path):
        """
        This function opens an hdf5 file for read.

        :param string hdf_full_path: Path to hdf5 file that will be opened
        :returns: file -- file object pointing to hdf file
        """
        if os.path.exists(hdf_full_path):
            f_hdf = h5py.File(hdf_full_path, "r")
        else:
            raise ValueError("File %s does not exist" % hdf_full_path)
        return f_hdf

    def check_hdf_metadata(self):
        """
        This function makes sure that the HDF5 file has a valid image, and
        geometric coordinate system data, and appropriate transformation data
        """
        # Check to see if the image is stored
        try:
            self.img = self.hdf[self.img_path]
        except KeyError:
            print('Could not find Image at: %s', self.img_path)
        self.x_size = self.img.shape[1]
        self.y_size = self.img.shape[0]
        self.shape = self.img.shape
        # if len(self.img.shape) > 2:
        #     raise ValueError('Pointing to Image path with > 2 Dimensions')

        # Now grab the GeoRegistration data, both the geo transform and the
        # CS Wkt stored as attributes in /ophoto
        self.gt = self.hdf['/ophoto'].attrs['upper_left_corner_geo_transform']
        self.cs = osr.SpatialReference(
            str(self.hdf['/ophoto'].attrs['coordinate_system']))
        if self.cs.GetAttrValue('AUTHORITY', 1) is not None:
            self.epsg = self.cs.GetAttrValue('AUTHORITY', 1)
            crs = rasterio.crs.CRS.from_epsg(self.epsg)
            self.cs = osr.SpatialReference(crs.wkt)
        if self.cs.IsProjected():
            self.m_per_pix = self.cs.GetLinearUnits()
        else:
            self.m_per_pix = 1.0
        if not self.cs.IsGeographic():
            self.m_per_pix = self.m_per_pix * self.gt[1]

        # Set Corner Lat-Lon
        new_cs_wkt = osr.GetWellKnownGeogCSAsWKT('wgs84')
        self.tf_to_wgs84 = georeg.coord_transform_from_wkt(
            self.cs.ExportToWkt(), new_cs_wkt)
        self.tf_wgs_to_geo = georeg.coord_transform_from_wkt(
            new_cs_wkt, self.cs.ExportToWkt())

        c_g = np.zeros((4, 2))
        # Upper left, upper right, lower left, lower right
        # Image frame = +x right, +y down... Numpy = +x down, +y right
        c_g[0, :] = georeg.geo_from_pix(np.array([0, 0]), self.gt)
        c_g[1, :] = georeg.geo_from_pix(np.array([self.x_size, 0]), self.gt)
        c_g[2, :] = georeg.geo_from_pix(np.array([0, self.y_size]), self.gt)
        c_g[3, :] = georeg.geo_from_pix(np.array([self.x_size, self.y_size]),
                                        self.gt)
        self.corners_geo = c_g
        try:
            self.corners_wgs = np.array(
                self.tf_to_wgs84.TransformPoints(c_g))[:, 0:2]
        except:
            self.hdf.close()
            raise ValueError("HDF5 WKT for reference system was most \
                              likely inusfficent to perform transform")

    def read(self, xv, yv):
        return self.img[xv[0]:xv[1], yv[0]:yv[1]]


class OrthoImage(object):

    """
    This class represents an orthophoto, and provides methods for querying\
    geographic coordinate information for various parts of the image.

    :param numpy.ndarray gt: Length 6 ndarray consisting of coefficients of \
        the affine transformation which converts coordinates in raster space \
        to a geometric coordinate system. Units are not defined. \
        The affine transformation is defined by:

        .. math::

                x_{geo} = gt[0] + x_{raster}gt[1] + y_{raster}gt[2]

                y_{geo} = gt[3] + x_{raster}gt[4] + y_{raster}gt[5]

        If using gdal, this transforms is returned by \
        osgeo.gdal.Dataset.GetGeoTransform()
    :param osgeo.SpatialReference() cs: osgeo Object representing the \
        coordinate system used to for the reference of the orthophoto. \
        Usually initialized from WKT formatted description stored with the \
        orthophoto or from SpatialReference.org
    :param numpy.ndarray img: N x M x num_bands sized numpy.ndarray \
        representing the orthophoto itself. The gt parameter specifies the \
        affine transform whose origin is the upper left corner of img. Also \
        remember that the image coordinate system is: +x = right, +y = down \
        which is the reverse of how the image is indexed via numpy
    """

    def __init__(self, gt, cs, img):
        # Set Image Metadata
        self.img = img
        self.gt = gt
        self.x_size = self.img.shape[1]
        self.y_size = self.img.shape[0]

        # Set up the coordinate systems and transformation objects to/from
        # wgs84
        self.cs = cs
        self.m_per_pix = self.cs.GetLinearUnits()
        if not self.cs.IsGeographic():
            self.m_per_pix = self.m_per_pix * self.gt[1]
        self.m_north_south = self.y_size * self.m_per_pix
        self.m_east_west = self.x_size * self.m_per_pix
        new_cs_wkt = osr.GetWellKnownGeogCSAsWKT('wgs84')
        self.tf_to_wgs84 = georeg.coord_transform_from_wkt(
            self.cs.ExportToWkt(), new_cs_wkt)
        self.tf_wgs_to_geo = georeg.coord_transform_from_wkt(
            new_cs_wkt, self.cs.ExportToWkt())

        c_g = np.zeros((4, 2))
        # Upper left, upper right, lower left, lower right
        # Image frame = +x right, +y down... Numpy = +x down, +y right
        c_g[0, :] = georeg.geo_from_pix(np.array([0, 0]), self.gt)
        c_g[1, :] = georeg.geo_from_pix(np.array([self.x_size, 0]), self.gt)
        c_g[2, :] = georeg.geo_from_pix(np.array([0, self.y_size]), self.gt)
        c_g[3, :] = georeg.geo_from_pix(np.array([self.x_size, self.y_size]),
                                        self.gt)
        self.corners_geo = c_g
        self.corners_wgs = np.array(
            self.tf_to_wgs84.TransformPoints(c_g))[:, 0:2]

    def pix_to_wgs84(self, pix):
        """
        Returns an n by 2 numpy.ndarray of Longitude,Latitude in degrees of \
        the pixel locations specified in the n by 2 numpy.ndarray pix

        :param numpy.ndarray pix: N x 2 numpy.ndarray of image frame (x,y) \
            pixel values where WGS-84 Longitude, Latitude coordinates are \
            desired
        :returns: -- numpy.ndarray n x 2 numpy.ndarray of [Lon(deg), \
            Lat(deg)] per pixel x,y location
        """
        geo = georeg.geo_from_pix(pix, self.gt)
        out_wgs = np.array(self.tf_to_wgs84.TransformPoints(geo))[:, 0:2]
        return out_wgs


    def wgs84_to_pix(self, lon_lat):
        """
        Returns an n by 2 numpy.ndarray of Longitude,Latitude in degrees of \
        the pixel locations specified in the n by 2 numpy.ndarray pix

        :param numpy.ndarray pix: N x 2 numpy.ndarray of image frame (x,y) \
            pixel values where WGS-84 Longitude, Latitude coordinates are \
            desired
        :returns: -- numpy.ndarray n x 2 numpy.ndarray of [Lon(deg), \
            Lat(deg)] per pixel x,y location
        """
        geo = np.array(self.tf_wgs_to_geo.TransformPoints(lon_lat))[:, 0:2]
        pix = georeg.pix_from_geo(geo, self.gt)
        return pix


def open_hdf_file(hdf_full_path):
    """
    This function opens an hdf5 file for either write or append. If the file \
    exists, it will be opened for append, else it will be created and opened \
    for write.

    :param string hdf_full_path: Path to hdf5 file that will be created or \
        appended
    :returns: file -- file object pointing to hdf file
    """

    if os.path.exists(hdf_full_path):
        f_hdf = h5py.File(hdf_full_path, "a")
    else:
        f_hdf = h5py.File(hdf_full_path, "w")
    return f_hdf


def create_hdf_image_from_vrt(hdf_file, vrt_path,
                              proj4_string=None, band_names=None):
    """
    This function creates an HDF5 Image instance using the information from \
    the tiles in the GDAL Virtual DataSet

    :param string hdf_file: Path to HDF5 file where the HDF5 orthophoto will \
        be created
    :param string vrt_path: Path to GDAL Virtual Dataset consiting of the \
        tiles uses to create the mosiac
    :param string proj4_string: Optional CRS overload, in case (as in USGS \
        and other) tiles provide insufficient information to translate \
        from the projection reference frame into WGS-84
    :param int comp_lvl: Sets compression options for blosc compression
        for these files default = 7, h5py defaults to 4
    """
    # Grab the first tile info to grab info on dtype and NumBands to create
    # HDF5 array size
    base_o = rasterio.open(vrt_path, 'r')
    base_cs = base_o.crs
    if proj4_string is not None:
        base_cs = rasterio.crs.from_string(proj4_string)
    osr_ref = osr.SpatialReference()
    osr_ref.ImportFromProj4(rasterio.crs.to_string(base_cs))
    base_wkt = osr_ref.ExportToWkt()

    num_bands = base_o.meta['count']
    img_type = np.dtype(base_o.dtypes[0])
    img_shape = (base_o.shape[0], base_o.shape[1])
    chunkshape = (
        base_o.block_shapes[0][0], base_o.block_shapes[0][1])

    # Build the HDF Structure
    hdf = h5py.File(hdf_file, 'w')
    g = hdf.create_group('/ophoto')

    # Write the metadata to the group
    g.attrs.create('coordinate_system', yaml.dump(base_wkt))
    g.attrs.create('crs', yaml.dump(base_cs))
    g.attrs.create('upper_left_corner_geo_transform', base_o.get_transform())

    if band_names is None:
        band_names = [repr(bidx) for bidx in np.arange(num_bands)]

    # Write the Image
    for band in np.arange(num_bands):
        img = g.create_dataset(band_names[band],
                               img_shape,
                               chunks=chunkshape,
                               dtype=img_type,
                               compression='lzf')
        for ji, win in base_o.block_windows():
            img[win[0][0]:win[0][1], win[1][0]:win[1][1]] = base_o.read(
                band + 1, window=win)
    hdf.close()


def create_thumbnail_from_vrt(vrt_file_name):
    """
    Build a thumbnail from VRT
    """
    band = 2
    vrt_file = rasterio.open(vrt_file_name)
    block_shape = vrt_file.block_shapes[0]
    x_size = int(np.ceil(vrt_file.shape[0] / float(block_shape[0])))
    y_size = int(np.ceil(vrt_file.shape[1] / float(block_shape[1])))
    out_img = np.zeros((x_size, y_size), dtype=np.uint8)
    for ji, win in vrt_file.block_windows():
        out_img[ji] = np.uint8(vrt_file.read(band, window=win).mean())
    return out_img


def overwrite_wkt(hdf_file, wkt_yaml_file):
    """
    This function creates an HDF5 Image instance using the information from \
    the tiles provided in grid_structure

    :param string hdf_file: Path to HDF5 file where the HDF5 orthophoto will \
        be created
    :param string wkt_yaml_file: Path to YAML formatted WKT structure used \
        to overwrite the WKT field in hdf_file
    """
    hdf = h5py.File(hdf_file)
    with open(wkt_yaml_file, 'r') as stream:
        wkt = yaml.load(stream)
    hdf['/ophoto'].attrs['coordinate_system'] = wkt['wkt']
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wkt['wkt'])
    crs = yaml.dump(rasterio.crs.from_string(new_cs.ExportToProj4()))
    hdf['/ophoto'].attrs['crs'] = crs
    hdf.close()


def overwrite_wkt_from_crs(hdf_file, crs):
    """
    This function overwrites the reference frame in the HDF5 file using
    a rasterio.crs formatted dictionary of a Proj4 style string
    """
    hdf = h5py.File(hdf_file, 'a')
    new_cs = osr.SpatialReference()
    new_cs.ImportFromProj4(rasterio.chdf_rs.to_string(crs))
    wkt = new_cs.ExportToWkt()
    hdf['/ophoto'].attrs['coordinate_system'] = yaml.dump(wkt)
    hdf['/ophoto'].attrs['crs'] = crs
    hdf.close()


def search_subfolders_for_files(root_folder, extension='.tif'):
    """
    This is a quick function that searches all sub-folders under root-folder
    for files with extensions == '.extension'.

    :param string root_folder: path to root folder where to start searching
    :param string extension: Extension to search for
    """
    result = [os.path.join(dp, f) for dp, dn, filenames in
              os.walk(root_folder) for f in filenames if
              os.path.splitext(f)[1] == extension]
    return result


def dask_rgb2gray(hdf_file, chunkshape=(1000, 1000), group_name='/ophoto',
                  rgb_names=None, gray_channel='gray'):
    """
    Open RGB channels in the orthophoto and convert to grayscale
    """
    hdf = h5py.File(hdf_file, 'a')
    group = hdf[group_name]
    if rgb_names is None:
        rgb_names = ['0', '1', '2']

    r = 0.2125 * \
        da.from_array(group[:, :, 0], chunks=chunkshape).astype(np.float)
    g = 0.7154 * \
        da.from_array(group[:, :, 1], chunks=chunkshape).astype(np.float)
    b = 0.0721 * \
        da.from_array(group[:, :, 2], chunks=chunkshape).astype(np.float)
    gray = da.stack([r, g, b], axis=2).sum(axis=2)
    gray_int = gray.map_blocks(np.rint).astype(np.uint8)
    gray_int.to_hdf5(hdf_file, '/ophoto/gray', compression='lzf')
    hdf.close()


def find_chunk_size(image_shape, largest_size=5000):
    s0 = 0
    s1 = 1
    for ii in np.arange(1, largest_size + 1):
        if np.mod(image_shape[0], ii) == 0.0:
            s0 = ii
        if np.mod(image_shape[1], ii) == 0.0:
            s1 = ii
    return (s0, s1)


def slices_from_ophoto(hdfo, max_pix_in_dim=5000):
    """
    :param hdfo: Valid neogeodb.hdf_orthophoto to create pixel slices from
    :param max_pix_in_dim: Integer value of the maximum number of pixels
        to be used in any dimension of a subslice
    :return list of tuples of the x-y slices of the image
    Given an neogeodb.hdf_orthophoto object, return list of slice tuples e.g. \
    ((row_start, row_stop), (col_start, col_stop))
    """

    bshape = find_chunk_size(hdfo.shape, max_pix_in_dim)
    blocks = np.array(hdfo.shape)[0:2] / np.array(bshape)
    out_tuples = []
    for row in np.arange(blocks[0]):
        rstart = row * bshape[0]
        rstop = rstart + bshape[0]
        for col in np.arange(blocks[1]):
            cstart = col * bshape[1]
            cstop = cstart + bshape[1]
            out_tuples.append(((int(rstart), int(rstop)), (int(cstart), int(cstop))))
    return out_tuples

