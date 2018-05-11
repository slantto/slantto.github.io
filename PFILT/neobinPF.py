#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script runs neogeo from an HDF5 formatted dataset, varies the numbers
of features loaded from the database and number of features searched per
image in order to make some meaningful results for the journal article
"""
# import sys
import bcolz
import pnpnav.utils as pnputils
import neogeodb.pytables_db as pdb
import neogeo.extent as neoextent
import numpy as np
import tables as tb
import time
import pyflann
import pandas as pd
import neogeo.utils as utils
from blessings import Terminal
from progressive.bar import Bar
from progressive.tree import ProgressTree, Value, BarDescriptor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation


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
    max_val = img_times.size
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
    # flann_params = {'algorithm': 'kdtree', 'trees': 8}
    db_desc = rows['descriptor'].astype(np.float32)
    t = time.time()
    # flann.build_index(db_desc, **flann_params)
    flann.build_index(db_desc, target_precision=0.95)

    build_time = time.time() - t
    return flann, build_time


def generate_obs(img_num, n_img, db_rows, obs_out, neogeo_out, flann_time, flann):
    t0 = time.time()
    ft = pd.read_hdf('/media/sean/D2F2E7B2F2E798CD/Users/student/AIEoutput2/feat/' + feat_path.iloc[img_num])
    ftdesc = bcolz.open('/media/sean/D2F2E7B2F2E798CD/Users/student/AIEoutput2/feat/' + descripath.iloc[img_num], 'r')
    tret = time.time() - t0
    t1 = time.time()

    # Here, we need to make sure that we sort the values by response
    n_img = n_img.astype(np.int)
    ft = ft.sort_values(by='response', ascending=False).iloc[:n_img]

    # Then pull the descriptors by the index
    obs_desc = ftdesc[ft.index, :]

    #    windowedindex = ft[ft['size'] > 0.5].index
    #    windowed = ftdesc[windowedindex,:]
    #   #obs_desc = windowed['descriptor'][-n_img:, :]
    #    n_img = n_img.astype(np.int)
    #    obs_desc = windowed[-n_img:, :]
    ##    print(n_img)
    #    print(windowed)
    #    print(obs_desc)

    obs_desc = obs_desc.astype(np.float32)
    tsort = time.time() - t1

    t2 = time.time()
    idx, dist = flann.nn_index(obs_desc, 2)
    d_vec = dist[:, 0] / dist[:, 1]

    img_idx = np.where(d_vec < 0.7)[0]
    # print[img_idx.size]
    db_idx = idx[img_idx, 0].astype(np.int)
    # print[db_idx.size]
    flann_time[img_num] = time.time() - t2
    # print(db_idx.shape[0])brisk

    if db_idx.shape[0] > 0:
        pid = db_rows[db_idx]['pair_id']
        # print(1-d_vec[img_idx])
        dbfeatwlla = np.array(
            (1 - d_vec[img_idx], db_rows[db_idx]['lat'], db_rows[db_idx]['lon'], db_rows[db_idx]['height'])).transpose()
        # print(dbfeatwlla)
        obs_out[img_num, 0:(dbfeatwlla.shape[0]), :] = np.copy(dbfeatwlla)
        print(obs_out[img_num, 0:(dbfeatwlla.shape[0] + 1), :])
        # dbloc[img_num][0:dbfeatwlla.shape[0]] = dbfeatwlla
        flannid, fidx, flanncount = np.unique(
            pid, return_counts=True, return_index=True)
        flannxy = np.array([pdb.unpair(tt)
                            for tt in flannid]).astype(np.int)
        img = utils.plot_on_grid(
            flannxy[:, 0], flannxy[:, 1], flanncount, xb, yb)
        #        print(flannid)
        #        print(flanncount)
        neogeo_out[img_num] = np.copy(img)
        # obs_likelihood = np.zeros_like(flanncount, dtype=np.object)
        # for fii in np.arange(flannid.shape[0]):
        #     obs_likelihood[fii] = 1 - d_vec[img_idx][pid == flannid[fii]]
        # img2 = utils.plot_on_grid(flannxy[:, 0], flannxy[:, 1], obs_likelihood, xb, yb)
        # obs_out[img_num] = np.copy(img2)
        # obs_out[img_num] = np.copy(obs_likelihood)


if __name__ == '__main__':
    f5 = tb.open_file('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/fc2_f5.hdf', 'r')
    f5featmeta = pd.read_hdf('/media/sean/D2F2E7B2F2E798CD/Users/student/AIEoutput2/feat/feat_meta.hdf')
    img_times = f5.root.camera.image_raw.compressed.metadata.col('t_valid')
    feat_path = f5featmeta.iloc[:, 3]
    descripath = f5featmeta.iloc[:, 4]

    # Get a truth finder
    dted_path = '/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/srtm'
    geoid_file = '/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/egm96-15.tif'
    frame_yaml = '/home/sean/ImageAidedNav/pypnp/data/fc2_pod_frames.yaml'

    finder = pnputils.DEMTileFinder(dted_path, geoid_file)
    finder.load_cam_and_vehicle_frames(frame_yaml)
    finder.load_camera_cal('/home/sean/ImageAidedNav/pypnp/data/fc2_cam_model.yaml')

    # Get the Feature Database
    dbf = tb.open_file('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/pytables_db.hdf', 'r')
    dbt = dbf.get_node('/sift_db/sift_features_sorted')

    # Set up the output file
    out_tb = tb.open_file('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/obs_out_mat7.hdf', 'w')

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

    f2l_mat = np.array([1E6])
    n_img_feat = np.array([10E3])

    # Create blessings.Terminal instance
    leaf_values, test_d = build_progress(f2l_mat, n_img_feat, img_times)
    tt = Terminal()
    prog_tree = ProgressTree(term=tt)
    prog_tree.make_room(test_d)

    # Full aggregate tile id
    tid, tidcount = np.unique(dbt.cols.pair_id, return_counts=True)
    extent = neoextent.SearchExtent(15, xb, yb, tid, tidcount)
    zoom_depth = 15  # orginal value was 14

    # Ok do whatever
    filters = tb.Filters(complevel=5, complib='blosc')
    atom = tb.Float64Atom()
    neogeo_shape = (img_times.size, 64, 64)
    times_shape = (img_times.size,)
    leaf_ii = 0
    for f2l in f2l_mat:

        f2l_group = out_tb.create_group(out_tb.root, 'loaded_%d' % int(f2l))
        f2l_group.num_feat_loaded = f2l
        db_rows, N = load_all_features(dbt, extent, zoom_depth, f2l)
        flann, t_build_db = build_flann_db(db_rows)
        f2l_group.time_to_build_flann = t_build_db
        obs_shape = (img_times.size, db_rows.shape[0], 4)
        for n_img in n_img_feat:
            n_group = out_tb.create_group(f2l_group, 'feat_%d' % int(n_img))
            n_group.img_feat = n_img
            neogeo_out = out_tb.create_carray(n_group, 'neogeo_out', atom=atom, shape=neogeo_shape, filters=filters)
            obs_out = out_tb.create_carray(n_group, 'obs_out', atom=atom, shape=obs_shape, filters=filters)
            # print(neogeo_out)
            flann_time = out_tb.create_carray(n_group, 'flann_time', atom=atom, shape=times_shape, filters=filters)
            # dbloc = out_tb.create_carray(n_group,'dbloc',atom=atom, shape=dbloc_shape, filters=filters)
            for img_num in np.arange(img_times.size):
                if f5featmeta.num_feat[img_num] > 0:
                    generate_obs(img_num, n_img, db_rows, obs_out, neogeo_out, flann_time, flann)
                    leaf_values[leaf_ii].value += 1
                    # if np.mod(leaf_values[leaf_ii].value, 5) == 0:
                    #     prog_tree.cursor.restore()
                    #     prog_tree.draw(test_d)

            leaf_ii += 1
            out_tb.flush()

    out_tb.close()
    dbf.close()
    f5.close()
