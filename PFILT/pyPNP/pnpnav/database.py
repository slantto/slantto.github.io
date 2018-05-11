#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module handles interfacing with feature databases for the purposes of
creating a search structure for searching a local (1km) area of a database
in order to perform PnP-based navigation
"""

import tables as tb
import numpy as np
import pandas as pd
import navfeatdb.db.pytables as tbdb


class FeatureDatabase(object):

    """
    This is an abstract base class for implementing a FeatureDatabase
    """
    def query_db_by_tid(self, tid, N, sort_metric='response'):
        """
        Abstract function for querying the database by tile ID
        :param tid: TID which to return indices from
        :param N: Number of features to Load by best metric
        :param sort_metric: Column of table by which to sort landmarks (descending order)
        :return: table rows of landmarks matching query, and indices into the
            descriptor matrix
        """
        raise NotImplementedError('Calling this function on base class')

    def query_db_by_bbox(self, bbox, N, sort_metric='response'):
        """
        Retrive rows of table, and indices, by bounding box
        :param bbox: Bounding box to load landmarks
        :param N: Number of landmarks to load
        :param sort_metric: Column of table by which to sort landmarks (descending order)
        :return: pandas.Dataframe of landmarks, indices into descriptors
        """
        raise NotImplementedError('Calling this function on base class')

    def get_descriptors(self, idx):
        """
        Return a numpy.ndarray of descriptors, based on their indices
        :param idx: Indices of descriptors to return
        :return: numpy.ndarray of descriptors
        """
        raise NotImplementedError('Calling this function on base class')


    def _format_landmarks(self, ta, idx):
           feat_wgs = []
           feat_desc = []
           feat_meta = []
           feat_idx = []

           feat_wgs.append(np.vstack((ta['lon'],
                                      ta['lat'],
                                      ta['height'])).T)
           feat_desc.append(self.get_descriptors(idx))
           feat_meta.append(np.vstack((np.int16(ta['octave']),
                                                ta['layer'],
                                                ta['scale'],
                                                ta['angle'],
                                                ta['response'],
                                                ta['size'])).T)
           feat_idx.append(idx)
           return (np.vstack(feat_wgs),
                   np.vstack(feat_desc),
                   np.vstack(feat_meta),
                   np.vstack(feat_idx).flatten())


    def load_features(self, tiles, N=10000):
        """
        Given a list of tiles, loads the features from each tile.
        """
        ret_val = np.array([]), np.array([]), np.array([]), np.array([])
        ids = np.array([tbdb.elegant_pair_xy(t.x, t.y) for t in tiles])
        if ids.shape[0] > 0:
            results = [self.query_db_by_tid(tid, N) for tid in ids]
            ta = pd.concat([r[0] for r in results])
            if ta.empty:
                return ret_val
            else:
                idx = np.concatenate([r[1] for r in results])
                return self._format_landmarks(ta, idx)
        else:
            return ret_val

    def load_features_from_bbox(self, bbox, N=10000):
        west = bbox.west
        east = bbox.east
        north = bbox.north
        south = bbox.south
        ta, idx = self.query_db_by_bbox(bbox, N)

        if idx.shape[0] > 10:
            return self._format_landmarks(ta, idx)
        else:
            return np.array([]), np.array([]), np.array([]), np.array([])


class SplitPyTablesDatabase(FeatureDatabase):
    """
    This class implements a feature database that operates on and HDF5
    formatted store of database features, and uses pyflann to build a local
    search tree of features to be searched.
    """

    def __init__(self, hdf_path, db_path='/db'):
        """
        Load the HDF database into a pandas DataFrame
        """
        self.x = None
        self.tile = None
        self.zoom = 15  # Hard coded but you could check feat_geo

        # HDF5 relative path names are coded for the moment. Sorry
        self.table_hdf = tb.open_file(hdf_path, mode='r')
        self.table = self.table_hdf.get_node(db_path + '/landmarks')
        self.desc = self.table_hdf.get_node(db_path + '/descriptors')

    def query_db_by_tid(self, tid, N, sort_metric='response'):
        idx = self.table.get_where_list('pair_id == tid')
        ta = self.table[idx]
        if N is not None:
            resp_sort = np.argsort(ta[sort_metric])
            ta = ta[resp_sort[-N:]]
            idx = idx[resp_sort[-N:]]
        return pd.DataFrame(ta), idx

    def query_db_by_bbox(self, bbox, N, sort_metric='response'):
        west = bbox.west
        east = bbox.east
        north = bbox.north
        south = bbox.south
        idx = self.table.get_where_list('(lat <= north) & (lat >= south) & (lon <= east) & (lon >= west)')
        ta = self.table[idx]
        if N is not None:
            resp_sort = np.argsort(ta[sort_metric])
            ta = ta[resp_sort[-N:]]
            idx = idx[resp_sort[-N:]]
        return ta, idx

    def get_descriptors(self, idx):
        return self.desc[idx, :]


class SplitPandasDatabase(FeatureDatabase):
    """
    This class implements a feature database that operates on and HDF5
    formatted store of database features, and uses pyflann to build a local
    search tree of features to be searched.
    """

    def __init__(self, hdf_path, db_path='/db'):
        """
        Load the HDF database into a pandas DataFrame
        """
        self.x = None
        self.tile = None
        self.zoom = 15  # Hard coded but you could check feat_geo

        # HDF5 relative path names are coded for the moment. Sorry
        self.table_hdf = tb.open_file(hdf_path, mode='r')
        self.table_key = db_path + '/landmarks'
        self.store = pd.HDFStore(hdf_path, key=self.table_key, mode='r')
        self.desc = self.table_hdf.get_node(db_path + '/descriptors')

    def query_db_by_tid(self, tid, N, sort_metric='response'):
        ta = self.store.select(key=self.table_key, where='pair_id == tid')
        idx = ta.index
        if N is not None:
            ta.sort_values(by=sort_metric, inplace=True, ascending=False)
            idx = ta.index.values
        return ta, idx

    def query_db_by_bbox(self, bbox, N, sort_metric='response'):
        west = bbox.west
        east = bbox.east
        north = bbox.north
        south = bbox.south
        ta = self.store.select(key=self.table_key, where='(lat <= north) & (lat >= south) & (lon <= east) & (lon >= west)')
        idx = ta.index
        if N is not None:
            ta.sort_values(by=sort_metric, inplace=True, ascending=False)
            idx = ta.index.values
        return ta, idx

    def get_descriptors(self, idx):
        return self.desc[idx, :]