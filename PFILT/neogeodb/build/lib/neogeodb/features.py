#!/usr/bin/env python
#
# A module that implements feature manipulation into the database

"""
This module is designed for manipulating neogeodb feature documents in \
mongoDB. Each feature will be geolocated, and contain a numpy array of
SIFT descriptors, along with
"""


import rasterio.warp
import numpy as np


def geo_to_wgs84(crs, geo_x, geo_y):
    """
    Takes in geo X and geo Y coordinates represented in the rasterio crs \
    format and returns an Nx2 numpy.ndarray of Longitude, Latitude in Degrees
    """
    lng, lat = rasterio.warp.transform(crs, {'init': 'epsg:4326'},
                                       geo_x, geo_y)
    return np.array([lng, lat]).T


def wgs84_to_geo(crs, lng, lat):
    """
    Takes in WGS-84 Lng, Lat (deg) an returns an Nx2 numpy.ndarray of \
    coordinates represented in the rasterio crs \
    """
    gx, gy = rasterio.warp.transform({'init': 'epsg:4326'}, crs, lng, lat)
    return np.array([gx, gy]).T


def pix_to_geo(src_affine, pix_x, pix_y):
    """
    Converts pixel x,y vectors to geo coordinates given a rasterio.Affine \
    object representing the affine transformation between raster and \
    2D geographic space
    """
    pix_x = np.array(pix_x)
    pix_y = np.array(pix_y)
    pix_x = pix_x.flatten()
    pix_y = pix_y.flatten()
    if pix_x.shape[0] != pix_y.shape[0]:
        raise ValueError("pix_x and pix_y are different lengths")
    A = np.array(src_affine).reshape(3, 3)
    xy1 = np.vstack((pix_x, pix_y, np.ones(pix_x.shape[0])))
    return np.dot(A, xy1).T[:, 0:2]


def geo_to_pix(src_affine, geo_x, geo_y):
    """
    Converts geo x,y vectors to pixel coordinates given a rasterio.Affine \
    object representing the affine transformation between raster and \
    2D geographic space
    """
    geo_x = np.array(geo_x)
    geo_y = np.array(geo_y)
    geo_x = geo_x.flatten()
    geo_y = geo_y.flatten()
    if geo_x.shape[0] != geo_y.shape[0]:
        raise ValueError("geo_x and geo_y are different lengths")
    A = np.array(src_affine).reshape(3, 3)
    xy1 = np.vstack((geo_x, geo_y, np.ones(geo_x.shape[0])))
    return np.dot(np.linalg.inv(A), xy1).T[:, 0:2]
