#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides utility functions for doing vision aided navigation
"""

import numpy as np
import numpy.linalg as nl
from ..frames import coverage2d as c2d
import cv2


def draw_2D3DCorrespondences(lon_lat_h, obs_kp, obs_img, tile_ophoto,
                             use_h = True, r_thresh=5.0, mask=None):
    """
    Takes in a pnpnav.matching.FeatureCorrespondence2D3D object, the observed \
    image, and the hdf_orthophoto to draw out the matches
    """
    tile_geo = np.array(tile_ophoto.tf_wgs_to_geo.TransformPoints(lon_lat_h))
    tile_pix = c2d.pix_from_geo(tile_geo[:, 0:2], tile_ophoto.gt)
    tile_img = np.copy(tile_ophoto.img)
    if tile_img.ndim > 2:
        tile_img = cv2.cvtColor(tile_img, cv2.COLOR_RGB2GRAY)
    if use_h and (mask is None):
        H, mask = cv2.findHomography(
            obs_kp, tile_pix, cv2.RANSAC, r_thresh)
    img3 = draw_match(obs_img, tile_img, obs_kp, tile_pix, mask)
    return img3, mask


def draw_match(img1, img2, p1, p2, status=None, H=None):
    """
    This function is stolen from some older versions of find_obj.py in the
    OpenCV 2.4.x example code (for the cv2 bindings). This function is
    implemented in C++ but there's no binding as of OpenCV 2.4.9
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(
            cv2.perspectiveTransform(
                corners.reshape(
                    1,
                    -1,
                    2),
                H).reshape(
                -1,
                    2) + (
                    w1,
                0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(p1), np.bool_)
    green = (0, 255, 0)
    red = (0, 0, 255)
    for (x1, y1), (x2, y2), inlier in zip(np.int32(p1), np.int32(p2), status):
        col = [red, green][int(inlier)]
        if inlier:
            cv2.line(vis, (x1, y1), (x2 + w1, y2), col, 7)
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2 + w1, y2), 2, col, -1)
        else:
            r = 2
            thickness = 7
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 + w1 - r, y2 - r),
                     (x2 + w1 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 + w1 - r, y2 + r),
                     (x2 + w1 + r, y2 - r), col, thickness)
    return vis


def dot_product_match(train, query):
    """
    This function returns the index of the nearest neighbor in train for every
    feature vector in query, using the L2-Distance of the two vectors, computed
     by normalizing each feature set, and computing the square root of
     2 - 2*dot product
    :param train: (NxD) np.array of the training features, where N is the number
        of features, and D is the dimensionality of the descriptor vector
    :param query: (MxD) np.array of M query features, whose descriptors are
        D dimensional
    :return: (M,) sized np.array of indices into train, where train[idx] was
        the feature in train that has the smallest L2 distance to the ith query
        feature
    """
    t_norm = (train.astype(np.double).T / nl.norm(train.astype(np.double), axis=1)).T
    q_norm = (query.astype(np.double).T / nl.norm(query.astype(np.double), axis=1)).T
    return np.argmin((2 - 2*np.dot(t_norm, q_norm.T))**0.5, axis=0)
