#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module handles interfacing with feature databases for the purposes of
creating a search structure for searching a local (1km) area of a database
in order to perform PnP-based navigation
"""

import numpy as np
import neogeodb.pytables_db as pdb
import mercantile
import navpy


def pad_grid(xbounds, ybounds):
    # Get Database Quadrants
    xsize = xbounds[1] - xbounds[0] + 1
    ysize = ybounds[1] - ybounds[0] + 1
    grid_base_2 = 2 ** np.ceil(np.log2(np.max([xsize, ysize])))
    to_add = grid_base_2 - (xsize, ysize)
    xb = np.copy(xbounds)
    yb = np.copy(ybounds)
    for ii in np.arange(np.floor(to_add[0] / 2)):
        xb += np.array([-1, 1])
    if np.mod(to_add[0], 2) == 1:
        xb[0] += -1
    for ii in np.arange(np.floor(to_add[1] / 2)):
        yb += np.array([-1, 1])
    if np.mod(to_add[1], 2) == 1:
        yb[0] += -1
    return xb, yb


def bbox_from_extent(extent, z=15):
    """
    Returns a list of (lng, lat) tuples for each corner of the search extent, \
    starting with the lower left and pushing counter-clockwise (KML style).
    """
    ul = mercantile.bounds(extent.xb[0], extent.yb[0], z)
    ur = mercantile.bounds(extent.xb[1], extent.yb[0], z)
    ll = mercantile.bounds(extent.xb[0], extent.yb[1], z)
    lr = mercantile.bounds(extent.xb[1], extent.yb[1], z)
    lon_lat = [(ll.west, ll.south),
               (lr.east, lr.south),
               (ur.east, ur.north),
               (ul.west, ul.north)]
    return (lon_lat)


def get_tiles_in_extent(extent):
    for xid in np.arange(extent.xb[0], extent.xb[1] + 1):
        for yid in np.arange(extent.yb[0], extent.yb[1] + 1):
            yield pdb.elegant_pair_xy(xid, yid)


def get_children_at_zoom(extent, zoom):
    if extent.zoom == zoom:
        yield extent
    else:
        for child in extent.children:
            for cgen in get_children_at_zoom(child, zoom):
                yield cgen


def get_average_tile_size(extent):
    ii = 0
    box_size = np.zeros((extent.grid_size ** 2, 2))
    for xid in np.arange(extent.xb[0], extent.xb[1] + 1):
        for yid in np.arange(extent.yb[0], extent.yb[1] + 1):
            bbox = mercantile.bounds(xid, yid, 15)
            ned_box = navpy.lla2ned(
                bbox.north, bbox.west, 0.0, bbox.south, bbox.east, 0.0)
            box_size[ii, :] = np.copy(ned_box[0:2])
            ii += 1
    return np.abs(box_size.mean(0)).mean()


def calc_num_feat_per_child(extent, zoom, num_feat):
    """
    Given that you want to load num_feat from the extent, return \
    the number of features to load per the sub-extents @ zoom.
    This function does some checks to account for empty sub-extents.
    """
    leaf_gen = get_children_at_zoom(extent, zoom)
    leaf_extents = [mb for mb in leaf_gen]
    N = np.floor(float(num_feat) / len(leaf_extents))
    feat_extent = np.array([bb.num_feat_in_extent for bb in leaf_extents])
    good_cells = np.where(feat_extent >= N)[0].shape[0]
    num_loaded = (good_cells * N) + feat_extent[feat_extent < N].sum()
    N2 = np.floor((float(num_feat) - num_loaded) / good_cells) + N
    return int(np.ceil(N2))


class SearchExtent(object):
    """
    This class defines a quadtree structure that represents a search extent
    """

    def __init__(self, base_zoom, xb, yb, tid, tidcount):

        grid_size = xb[1] - xb[0] + 1
        self.children = []
        self.xb = xb
        self.yb = yb
        self.grid_size = grid_size
        self.zoom = base_zoom - int(np.log2(grid_size))
        stiles = np.array([tiles for tiles in get_tiles_in_extent(self)])
        gmask = np.in1d(tid, stiles)
        self.num_feat_in_extent = tidcount[gmask].sum()
        self.num_feat_per_tile = tidcount[gmask]
        self.tiles = tid[gmask]

        # If we're not at the bottom, keep pushing
        if self.zoom < base_zoom:

            xb0 = np.array([xb[0], xb[0] + (grid_size / 2) - 1])
            xb1 = np.array([xb[0] + (grid_size / 2), xb[1]])

            yb0 = np.array([yb[0], yb[0] + (grid_size / 2) - 1])
            yb1 = np.array([yb[0] + (grid_size / 2), yb[1]])

            quad_bounds = np.array([[xb0, yb0], [xb1, yb0],
                                    [xb0, yb1], [xb1, yb1]])

            for quad in quad_bounds:
                self.children.append(SearchExtent(base_zoom, quad[0], quad[1],
                                                  tid, tidcount))
