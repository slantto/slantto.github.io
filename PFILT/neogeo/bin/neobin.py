#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script runs neogeo from an HDF5 formatted dataset, varies the numbers
of features loaded from the database and number of features searched per
image in order to make some meaningful results for the journal article
"""

import pnpnav.utils as pnputils
import neogeodb.pytables_db as pdb
import neogeo.extent as neoextent
import numpy as np
import tables as tb
import time
import pyflann
import neogeo.utils as utils
from blessings import Terminal
from progressive.bar import Bar
from progressive.tree import ProgressTree, Value, BarDescriptor


def load_features_by_extent(dbt, leaf, N):
    """
    Called with a lon (deg), lat(deg), and radius (tiles) to return N \
    feature descriptors from PyTablesDatabase. Uses the octave value \
    to return the N largest features. If N > the number of features in \
    that tile, then it returns all features. Returns empty array \
    if nothing was found.
    """
    rows = []
    for ii in np.arange(leaf.tiles.shape[0]):
        tileid = leaf.tiles[ii]
        if leaf.num_feat_per_tile[ii] > 0:
            rows.append(dbt.read_where('pair_id == tileid'))
    if len(rows) > 0:
        feat_in_tile = np.hstack(rows)
        ff = np.argsort(feat_in_tile['response'])
        best_feat = feat_in_tile[ff[-N:]]
        return best_feat


def build_progress(f2l_mat, n_img_feat, img_times):
    # Setup Progress Bar Garbage
    leaf_values = [Value(0)
                   for i in range(f2l_mat.shape[0] * n_img_feat.shape[0])]
    max_val = img_times.nrows
    leaf_dict = {}
    bd_defaults = dict(type=Bar, kwargs=dict(max_value=max_val))

    ii = 0
    for f2l in f2l_mat:
        out_dict = {}
        for n_img in n_img_feat:
            out_dict[repr(n_img)] = BarDescriptor(
                value=leaf_values[ii], **bd_defaults)
            ii += 1
        leaf_dict[repr(f2l)] = out_dict

    test_d = {'Graduate': leaf_dict}
    return leaf_values, test_d


def load_all_features(dbt, extent, zoom_depth, f2l):
    leaf_gen = neoextent.get_children_at_zoom(extent, zoom_depth)
    leaves = [leaf for leaf in leaf_gen]
    N = neoextent.calc_num_feat_per_child(extent, zoom_depth, f2l)

    rows = []
    for leaf in leaves:
        rows.append(load_features_by_extent(dbt, leaf, N))
    rows = np.hstack([gr for gr in rows if gr is not None])
    return rows, N


def build_flann_db(rows):
    flann = pyflann.FLANN()
    flann_params = {'algorithm': 'kdtree', 'trees': 8}
    db_desc = rows['descriptor'].astype(np.float32)
    t = time.time()
    flann.build_index(db_desc, **flann_params)
    build_time = time.time() - t
    return flann, build_time


def generate_obs(img_num, n_img, db_rows, obs_out, neogeo_out, flann_time, flann):
    t0 = time.time()
    ft = f5.get_node('/images/sift_features/img_%d' % img_num)
    tret = time.time() - t0
    t1 = time.time()
    windowed = ft.read_where('size > 0.5')
    obs_desc = windowed['descriptor'][-n_img:, :]

    obs_desc = obs_desc.astype(np.float32)
    tsort = time.time() - t1
    t2 = time.time()
    idx, dist = flann.nn_index(obs_desc, 2)
    d_vec = dist[:, 0] / dist[:, 1]
    img_idx = np.where(d_vec < 0.7)[0]
    db_idx = idx[img_idx, 0].astype(np.int)
    flann_time[img_num] = time.time() - t2

    if db_idx.shape[0] > 0:
        pid = db_rows[db_idx]['pair_id']
        flannid, fidx, flanncount = np.unique(
            pid, return_counts=True, return_index=True)
        flannxy = np.array([pdb.unpair(tt)
                            for tt in flannid]).astype(np.int)
        img = utils.plot_on_grid(
            flannxy[:, 0], flannxy[:, 1], flanncount, xb, yb)
        neogeo_out[img_num] = np.copy(img)
        obs_likelihood = np.zeros_like(flanncount, dtype=np.float64)
        for fii in np.arange(flannid.shape[0]):
            obs_likelihood[fii] = (
                1 - d_vec[img_idx][pid == flannid[fii]]).sum()
        img2 = utils.plot_on_grid(
            flannxy[:, 0], flannxy[:, 1], obs_likelihood, xb, yb)
        obs_out[img_num] = np.copy(img2)


if __name__ == '__main__':
    f5 = tb.open_file('/Users/venabled/catkin_ws/data/fc2/sorted_fc2_f5.hdf', 'r')
    img_times = f5.root.images.t_valid
    feat_tables = f5.list_nodes(f5.root.images.sift_features)

    # Get a truth finder
    dted_path = '/Users/venabled/catkin_ws/data/dted'
    geoid_file = '/Users/venabled/catkin_ws/data/geoid/egm96_15.tiff'
    frame_yaml = '/Users/venabled/pnpnav/data/fc2_pod_frames.yaml'

    finder = pnputils.DTEDTileFinder(dted_path, geoid_file)
    finder.load_cam_and_vehicle_frames(frame_yaml)
    finder.load_camera_cal('/Users/venabled/pnpnav/data/fc2_cam_model.yaml')

    # Get the Feature Database
    dbf = tb.open_file(
        '/Users/venabled/catkin_ws/data/neogeo/pytables_db.hdf', 'r')
    dbt = dbf.get_node('/sift_db/sift_features_sorted')

    # Set up the output file
    out_tb = tb.open_file(
        '/Users/venabled/catkin_ws/data/neogeo/obs_out_mat.hdf', 'w')

    # You need boundaries
    txmin = dbt.cols.x[dbt.colindexes['x'][0]].astype(np.int64)
    txmax = dbt.cols.x[dbt.colindexes['x'][-1]].astype(np.int64)
    tymin = dbt.cols.y[dbt.colindexes['y'][0]].astype(np.int64)
    tymax = dbt.cols.y[dbt.colindexes['y'][-1]].astype(np.int64)
    xbounds = (txmin, txmax)
    ybounds = (tymin, tymax)
    print(xbounds)
    print(ybounds)
    xb, yb = neoextent.pad_grid(xbounds, ybounds)

    f2l_mat = np.array([125E3, 250E3, 500E3, 1E6])
    n_img_feat = np.array([1E3, 5E3, 10E3, 20E3])

    # Create blessings.Terminal instance
    leaf_values, test_d = build_progress(f2l_mat, n_img_feat, img_times)
    tt = Terminal()
    prog_tree = ProgressTree(term=tt)
    prog_tree.make_room(test_d)

    # Full aggregate tile id
    tid, tidcount = np.unique(dbt.cols.pair_id, return_counts=True)
    extent = neoextent.SearchExtent(15, xb, yb, tid, tidcount)
    zoom_depth = 14

    # Ok do whatever
    filters = tb.Filters(complevel=5, complib='blosc')
    atom = tb.Float64Atom()
    neogeo_shape = (img_times.nrows, 64, 64)
    times_shape = (img_times.nrows, )
    leaf_ii = 0
    for f2l in f2l_mat:

        f2l_group = out_tb.createGroup(out_tb.root, 'loaded_%d' % int(f2l))
        f2l_group.num_feat_loaded = f2l
        db_rows, N = load_all_features(dbt, extent, zoom_depth, f2l)
        flann, t_build_db = build_flann_db(db_rows)
        f2l_group.time_to_build_flann = t_build_db

        for n_img in n_img_feat:
            n_group = out_tb.createGroup(f2l_group, 'feat_%d' % int(n_img))
            n_group.img_feat = n_img
            neogeo_out = out_tb.create_carray(n_group, 'neogeo_out', atom=atom, shape=neogeo_shape, filters=filters)
            obs_out = out_tb.create_carray(n_group, 'obs_out', atom=atom, shape=neogeo_shape, filters=filters)
            flann_time = out_tb.create_carray(n_group, 'flann_time', atom=atom, shape=times_shape, filters=filters)
            for img_num in np.arange(img_times.nrows):
                generate_obs(img_num, n_img, db_rows, obs_out, neogeo_out, flann_time, flann)
                leaf_values[leaf_ii].value += 1
                if np.mod(leaf_values[leaf_ii].value, 5) == 0:
                    prog_tree.cursor.restore()
                    prog_tree.draw(test_d)
            leaf_ii += 1
            out_tb.flush()

    out_tb.close()
    dbf.close()
    f5.close()
