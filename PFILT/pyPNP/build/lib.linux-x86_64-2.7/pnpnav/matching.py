#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides feature matching routines and structures for
pnpnav
"""

import pyflann
import cv2
import numpy as np
from . import database as db


class FeatureCorrespondence2D3D(object):

    """
    This class is derived from the UVAN ROS message used to hold
    2D/3D Correspondences.
    """

    def __init__(self):
        """
        Class init
        """
        self.metadata = ''
        self.num_correspondences = 0  # N matches
        self.img_idx = np.array([])
        self.db_idx = np.array([])
        self.keypoints = np.array([])  # Will be Nx2 for keypoints in image 1
        self.world_coordinate_frame = 'WGS-84'  # Should do a proj.4 frame
        self.world_coordinates = np.array([])  # Nx3 for Lon Lat Height
        self.match_ratio = np.array([])


class BFMatcher(object):

    """
    This class instantiates a Brute Force matcher
    """

    def __init__(self, nn_ratio=0.7, norm_type=cv2.NORM_L2):
        self.bf = cv2.BFMatcher(normType=norm_type)
        self.__nn_ratio = nn_ratio
        self.__db = None
        self.__loaded = False
        self.__tiles = None
        self.__norm = norm_type

    def load_db(self, db_path, db_class=db.SplitPandasDatabase):
        """
        Right now just use PyTablesDatabase
        """
        self.__db = db_class(db_path)

    def load_features_from_tiles(self, tiles, N=None):
        """
        Internal PnP method that loads features from the database into \
        local memory for further matching.
        """
        if self.__tiles != tiles:
            print("Loading db from tile: %s" % repr(tiles))
            self.__tiles = tiles
            self.__db_geo, self.__db_desc, self.__db_meta, self.__db_idx = self.__db.load_features(
                tiles, N)
            print("Loading %d features" % self.__db_geo.shape[0])
            if self.__db_geo.shape[0] > 0:
                self.__loaded = True
            else:
                self.__loaded = False
        return self.__loaded

    def load_features_from_tiles(self, tiles, N=None):
        """
        Internal PnP method that loads features from the database into \
        local memory for further matching.
        """
        if self.__tiles != tiles:
            print("Loading db from tile: %s" % repr(tiles))
            self.__tiles = tiles
            self.__db_geo, self.__db_desc, self.__db_meta, self.__db_idx = self.__db.load_features(
                tiles, N)
            print("Loading %d features" % self.__db_geo.shape[0])
            if self.__db_geo.shape[0] > 0:
                self.__loaded = True
            else:
                self.__loaded = False
        return self.__loaded

    def load_features_from_bbox(self, bbox, N=None):
        """
        Internal PnP method that loads features from the database into \
        local memory for further matching.
        """
        print("Loading db from bbox: %s" % repr(bbox))
        self.__db_geo, self.__db_desc, self.__db_meta, self.__db_idx = self.__db.load_features_from_bbox(
            bbox, N)
        print("Loading %d features" % self.__db_geo.shape[0])
        if self.__db_geo.shape[0] > 0:
            self.__loaded = True
        else:
            self.__loaded = False
        return self.__loaded

    def get_loaded_db_idx(self):
        """
        Returns the loaded db idx
        :return: Database indices
        """
        return self.__db_idx

    def trim_duplicates(self, img_idx, db_idx):
        u, u_idx = np.unique(img_idx, return_index=True)
        img_idx = img_idx[u_idx]
        db_idx = db_idx[u_idx]
        u, u_idx = np.unique(db_idx, return_index=True)
        img_idx = img_idx[u_idx]
        db_idx = db_idx[u_idx]
        return img_idx, db_idx

    def match(self, obs_kp, obs_desc, prune=True):
        """
        Let's match
        """
        if self.__loaded and len(obs_kp) > 0:
            m1 = self.bf.knnMatch(obs_desc, self.__db_desc, k=2)
            m2 = np.array([(m0[0].trainIdx,
                            m0[0].distance / m0[1].distance) for m0 in m1])
            img_idx = np.where(m2[:, 1] < self.__nn_ratio)[0]
            db_idx = m2[img_idx, 0].astype(np.int)

            if prune:
                img_idx, db_idx = self.trim_duplicates(img_idx, db_idx)

            matches = FeatureCorrespondence2D3D()
            matches.num_correspondences = db_idx.shape[0]
            matches.keypoints = obs_kp[img_idx, :]
            matches.world_coordinates = self.__db_geo[db_idx, :]
            matches.img_idx = img_idx
            matches.db_idx = self.__db_idx[db_idx]
            matches.match_ratio = m2[img_idx, 1]
            return matches
        else:
            return FeatureCorrespondence2D3D()


class FlannMatcher(object):

    """
    This class instantiates a FLANN based k-NN matcher. You can tell it to
    load new features into the index.
    """

    def __init__(self, nn_ratio=0.7):
        self.flann = pyflann.FLANN()
        self.flann_params = {'algorithm': 'kdtree', 'trees': 8}
        self.__nn_ratio = nn_ratio
        self.__db = None
        self.__loaded = False
        self.__tiles = None

    def load_db(self, db_path):
        """
        Right now just use PyTablesDatabase
        """
        self.__db = db.PyTablesDatabase(db_path)

    def load_features_from_tiles(self, tiles, N=None):
        """
        Internal PnP method that loads features from the database into \
        local memory for further matching.
        """
        if self.__tiles != tiles:
            print("Loading db from tile: %s" % repr(tiles))
            self.__tiles = tiles
            self.__db_geo, self.__db_desc, self.__db_meta, self.__db_idx = self.__db.load_features(
                tiles, N)
            print("Loading %d features" % self.__db_geo.shape[0])
            if self.__db_geo.shape[0] > 0:
                self.__loaded = True
                self.flann.delete_index()
                self._fp = self.flann.build_index(self.__db_desc,
                                                  **self.flann_params)
            else:
                self.__loaded = False
        return self.__loaded

    def trim_duplicates(self, img_idx, db_idx):
        u, u_idx = np.unique(img_idx, return_index=True)
        img_idx = img_idx[u_idx]
        db_idx = db_idx[u_idx]
        u, u_idx = np.unique(db_idx, return_index=True)
        img_idx = img_idx[u_idx]
        db_idx = db_idx[u_idx]
        return img_idx, db_idx

    def match(self, obs_kp, obs_desc, prune=True):
        """
        Let's match
        """
        if self.__loaded:
            fidx, dists = self.flann.nn_index(obs_desc, 2)
            d_vec = dists[:, 0] / dists[:, 1]

            img_idx = np.where(d_vec < self.__nn_ratio)[0]
            db_idx = fidx[img_idx, 0].astype(np.int)

            if prune:
                img_idx, db_idx = self.trim_duplicates(img_idx, db_idx)

            matches = FeatureCorrespondence2D3D()
            matches.num_correspondences = db_idx.shape[0]
            matches.keypoints = obs_kp[img_idx, :]
            matches.world_coordinates = self.__db_geo[db_idx, :]
            matches.img_idx = img_idx
            matches.db_idx = db_idx
            return matches
        else:
            return FeatureCorrespondence2D3D()
