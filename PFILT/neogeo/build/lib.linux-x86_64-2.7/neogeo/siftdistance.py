#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements neogeo for SIFT feature distance as the feature
metric. Uses a database of SIFT features
"""

import h5py
import cv2
from . import core
import neogeodb.hdf_orthophoto as himg
import tables as tb
import neogeodb.grid_from_hdf_ophoto as neo_grid
import numpy as np
from scipy.interpolate import griddata
import time


class NeoGeo(object):

    """
    This class is used to perform inference over a 3D point cloud database \
    consisting of georegistered visual features, arranged in a grid, given \
    a series of observations (airborne imagery), and prior pose estimates.
    """

    def __init__(self):
        """
        Class constructor, right now just does some basic book_keeping
        """
        self._db = None
        self.matcher = cv2.BFMatcher()

    def load_database_file(self, db_hdf):
        """
        Supports loading a feature database from an HDF file. Right now
        the file path is hard coded to the UVAN defaults
        """
        # Load the subsampled DB
        self._hdf = h5py.File(db_hdf, 'r')
        self._db = self._hdf['/georegistered_features']
        self.db_wgs = np.array(self._db['feat_wgs84_lon_lat_hae'])
        self.db_desc = self._db['feature_descriptors/desc_SIFT']

    def load_grid(self, grid_hdf_file):
        """
        Loads the grid that we're doing Bayesian inference over
        """
        # Set up basic grid info, and indices into feature db per grid cell
        self.grid_info = neo_grid.get_grid_info_from_hdf(grid_hdf_file)
        self._grid_hdf = tb.openFile(grid_hdf_file)
        self.img_index = self._grid_hdf.getNode('/img_index')
        self.img_len = np.array([xx.shape[0] for xx in self.img_index])
        self.img_prob = self.img_len / float(self.img_len.sum())

    def calc_cell_distance_bf(self, obs_desc, cell_idx):
        """
        Uses an OpenCV BruteForce L2 Distance metric to calculate the
        distance between the obs_desc and the features in cell_idx. Returns
        the distance and the time it took to load and eval
        """
        t0 = time.time()
        ref_idx = self.img_index[cell_idx]
        t_load = np.nan
        t_match = np.nan
        t_bf = np.nan
        d = np.nan
        if ref_idx.shape[0] > 0:
            ref_desc = self.db_desc[ref_idx, :]
            t_load = time.time() - t0
            t0 = time.time()
            matches = self.matcher.knnMatch(obs_desc, ref_desc, k=2)
            t_bf = time.time() - t0
            out_mat = np.array([(match[0].trainIdx,
                                 match[0].distance / match[1].distance) for
                                match in matches])
            idx = np.where(out_mat[:, 1] < 0.7)[0]
            d = np.linalg.norm(out_mat[idx, 1])
            t_match = time.time() - t0
        return (d, t_load, t_bf, t_match)

    def calc_likelihood(self, obs_desc):
        """
        Do work
        """
        print obs_desc
