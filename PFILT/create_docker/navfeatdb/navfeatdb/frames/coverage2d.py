#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides helper functions to georegister (add geo-referenced \
coordinates) to raster-based (image) data products. This module makes \
extensive use of osgeo.gdal.Dataset and osgeo.gdal.Dataset.GetGeoTransform
"""

from osgeo import osr
import numpy as np


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


