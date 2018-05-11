#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module handles interfacing with feature databases for the purposes of
creating a search structure for searching a local (1km) area of a database
in order to perform PnP-based navigation
"""

import tables as tb
import numpy as np
import neogeodb.pytables_db as pdb
from . import utils


class FeatureDatabase(object):

    """
    This is an abstract base class for implementing a FeatureDatabase
    """

    def load_features(self, bbox):
        """
        Takes in a mercantile tile object and loads features from the
        data store into a local search object
        """
        raise NotImplementedError

    def search_db(self, query_kp, query_descriptors):
        """
        Takes in an NxM numpy.ndarray of N features with an M length
        descriptor. Returns 2D / 3D correspondences.
        """
        raise NotImplementedError


class PyTablesDatabase(FeatureDatabase):

    """
    This class implements a feature database that operates on and HDF5
    formatted store of database features, and uses pyflann to build a local
    search tree of features to be searched.
    """

    def __init__(self, hdf_path):
        """
        Load the HDF database into a pandas DataFrame
        """
        self.zoom = 15  # Hard coded but you could check feat_geo
        self.matcher = None  # Don't build the FLANN Index if we don't need to

        # HDF5 path names are coded for the moment. Sorry
        self.table_hdf = tb.open_file(hdf_path, mode='r')
        self.dbt = self.table_hdf.root.sift_db.sift_features_sorted
        self.uid = np.array(self.table_hdf.root.sift_db.unique_tiles)
        self.uid = self.uid.astype(np.uint32)

        # Set up the boundary info
        txmin = self.dbt.cols.x[self.dbt.colindexes['x'][0]]
        txmax = self.dbt.cols.x[self.dbt.colindexes['x'][-1]]
        tymin = self.dbt.cols.y[self.dbt.colindexes['y'][0]]
        tymax = self.dbt.cols.y[self.dbt.colindexes['y'][-1]]
        xbounds = (txmin, txmax)
        ybounds = (tymin, tymax)
        xb, yb = neoextent.pad_grid(xbounds, ybounds)
        self.xb = xb
        self.yb = yb

        # Load the aggregation by tile
        self.tid, self.tidcount = np.unique(self.dbt.cols.pair_id,
                                            return_counts=True)

    def load_features_by_extent(self, leaf, N):
        """
        Called with a lon (deg), lat(deg), and radius (tiles) to return N \
        feature descriptors from PyTablesDatabase. Uses the octave value \
        to return the N largest features. If N > the number of features in \
        that tile, then it returns all features. Returns None \
        if nothing was found.
        """
        rows = []
        for ii in np.arange(leaf.tiles.shape[0]):
            tileid = leaf.tiles[ii]
            if leaf.num_feat_per_tile[ii] > 0:
                rows.append(self.dbt.read_where('pair_id == tileid'))
        if len(rows) > 0:
            feat_in_tile = np.hstack(rows)
            ff = np.argsort(feat_in_tile['response'])
            best_feat = feat_in_tile[ff[-N:]]
            return best_feat
