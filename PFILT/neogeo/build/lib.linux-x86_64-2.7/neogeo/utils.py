#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module is built for pnp utilities
"""

import cv2
import numpy as np
import datetime


def plot_on_grid(tx, ty, val, xbounds, ybounds, fill=None):
    grid_shape = (ybounds[1] - ybounds[0], xbounds[1] - xbounds[0])
    grid_shape = np.abs(grid_shape) + 1
    img = np.zeros(grid_shape)
    if fill is not None:
        img[:, :] = fill
    img[ty - ybounds[0], tx - xbounds[0]] = val
    return img


def img_to_kmz(image_path, folder, lon_lat, title=None, t_start=None, t_stop=None):
    """
    Writes a new document in kml_folder pointing to image_path, and Writes
    using the list lon_lat which is a list of 4 tuples of (lon_lat) starting
    in the lower left and going counter-clockwise
    """

    ground = folder.newgroundoverlay(name=title)
    ground.icon.href = image_path
    ground.color = '7fffffff'
    ground.gxlatlonquad.coords = lon_lat
    if t_start is not None:
        ground.timespan.begin = datetime.datetime.fromtimestamp(
            t_start).strftime('%Y-%m-%dT%H:%M:%SZ')
        ground.timespan.end = datetime.datetime.fromtimestamp(
            t_stop).strftime('%Y-%m-%dT%H:%M:%SZ')


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
                corners.reshape(1, -1, 2), H).reshape(1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(p1), np.bool_)
    green = (0, 255, 0)
    red = (0, 0, 255)
    for (x1, y1), (x2, y2), inlier in zip(np.int32(p1), np.int32(p2), status):
        col = [red, green][inlier]
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
