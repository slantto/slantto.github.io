#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script builds the neogeo tf-idf / FAB-MAP Comparison DB
"""

import pnpnav.utils as pnputils
import neogeo.utils as neoutils
import neogeodb.pytables_db as pdb
import neogeo.extent as neoextent
import numpy as np
import tables as tb
import mercantile
import time
import navpy
import pyflann
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import neogeo.core as core
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Get the Feature Database
dbf = tb.open_file(
    '/Users/venabled/catkin_ws/data/neogeo/pytables_db.hdf', 'r')
dbt = dbf.get_node('/sift_db/sift_features_sorted')

# You need boundaries
txmin = dbt.cols.x[int(dbt.colindexes['x'][0])]
txmax = dbt.cols.x[int(dbt.colindexes['x'][-1])]
tymin = dbt.cols.y[int(dbt.colindexes['y'][0])]
tymax = dbt.cols.y[int(dbt.colindexes['y'][-1])]
xbounds = (txmin, txmax)
ybounds = (tymin, tymax)
print(xbounds)
print(ybounds)
xb, yb = neoextent.pad_grid(xbounds, ybounds)

# Full aggregate tile id
tid, tidcount = np.unique(dbt.cols.pair_id, return_counts=True)
tidxy = np.array([pdb.unpair(tt) for tt in tid]).astype(np.int)


cluster_centers = np.load('~/catkin_ws/data/neogeo/100k_vocab_centers.npy')
flann = pyflann.FLANN()
flann.build_index(cluster_centers)

inverse_index = {}
ii = 0
for tileid in tid: 
    rows = dbt.read_where('pair_id == tileid')
    ff = np.argsort(rows['response'])
    best_feat = rows[ff[-N:]]
    feat_desc = best_feat['descriptor'].astype(np.float32)
    result, dist = flann.nn_index(feat_desc, 1)
    inverse_index[tileid] = np.unique(result, return_counts=True)
    print("%d / %d" % (ii, tid.shape[0]))
    ii += 1

word_mat = np.zeros((tid.shape[0], cluster_centers.shape[0]), dtype=np.int)
for ii in np.arange(tid.shape[0]):
    tf = tid[ii]
    word_mat[ii, inverse_index[tf][0]] = inverse_index[tf][1]


idf_term = np.log(float(tid.shape[0]) / ((word_mat > 0).sum(0)).astype(np.float))
tf_mat = (word_mat.T / word_mat.sum(1).astype(np.float)).T
tf_idf = tf_mat * idf_term
tf_idf_norm = (tf_idf.T / np.linalg.norm(tf_idf, axis=1)).T


#Mask out 5% of common features
num_mask = np.int(idf_term.shape[0] * 0.10)
mask_val = np.sort(idf_term)[num_mask]
mask = idf_term > mask_val

#Recompute tf term / matrix
tf_mat_mask = (word_mat[:, mask].T / word_mat[:,  mask].sum(1).astype(np.float)).T
tf_idf_mask = tf_mat_mask * idf_term[mask]
tf_idf_mask = (tf_idf_mask.T / np.linalg.norm(tf_idf_mask, axis=1)).T


# Run TF-IDF for Flight 5
f5 = tb.open_file('/Users/venabled/catkin_ws/data/fc2/fc2_f5.hdf', 'r')
imgs = f5.root.images.image_data
img_times = f5.root.images.t_valid
feat_tables = f5.list_nodes(f5.root.images.sift_features)
pva = f5.root.pva

# Get a truth finder
dted_path = '/Users/venabled/catkin_ws/data/dted'
geoid_file = '/Users/venabled/catkin_ws/data/geoid/egm96_15.tiff'
frame_yaml = '/Users/venabled/pnpnav/data/fc2_pod_frames.yaml'

finder = pnputils.DTEDTileFinder(dted_path, geoid_file)
finder.load_cam_and_vehicle_frames(frame_yaml)
finder.load_camera_cal('/Users/venabled/pnpnav/data/fc2_cam_model.yaml')

vocab_out = np.zeros((img_times.nrows, 64, 64))
for img_num in np.arange(img_times.nrows):
        t0 = time.time()
        ft = f5.get_node('/images/sift_features/img_%d' % img_num)
        tret = time.time() - t0
        t1 = time.time()
        windowed = ft.read_where('size > 0.5')
        img_resp= windowed['response']
        img_sort = np.argsort(img_resp)
        obs_desc = windowed['descriptor'][img_sort[-10000:], :]
        # obs_desc = windowed['descriptor']
        obs_desc = obs_desc.astype(np.float32)
        result, dist = flann.nn_index(obs_desc, 1)
        obs_words = np.unique(result, return_counts=True)
        obs_tf_vec = np.zeros(cluster_centers.shape[0])
        obs_tf_vec[obs_words[0]] = obs_words[1]
        obs_tf_vec = obs_tf_vec[mask] / obs_tf_vec[mask].sum()
        obs_tf_idf = obs_tf_vec * idf_term[mask]
        obs_tf_idf = obs_tf_idf / np.linalg.norm(obs_tf_idf)
        d2 = 2 - 2*np.dot(tf_idf_mask, obs_tf_idf)
        d3 = np.zeros_like(d2)
        d3[np.argsort(d2)[:25]] = 1.0
        img = neoutils.plot_on_grid(tidxy[:, 0], tidxy[:, 1], d2, xb, yb)
        vocab_out[img_num] = np.copy(img)
        print('%d / %d' % (img_num, img_times.nrows))

backup = np.copy(vocab_out)
vocab_out[np.isnan(vocab_out)] = 2.0
vocab_out[vocab_out == 0.0] = 2.0
vocab_out = 2.0 - vocab_out
vocab_out[vocab_out == 0.0] = vocab_out[vocab_out > 0.0].min()


import neogeo.core as core
reload(core)
neo = core.NeoGeo()
neo.init_px_from_extent(extent)
neo.pix_size_m = neoextent.get_average_tile_size(neo.extent)
posterior = np.zeros_like(obs_out)

for ii in np.arange(110, vocab_out.shape[0]):
    print(ii)
    dpos = ned_vel[ii, 0:2]
    dpos_sig = 0.01 * np.linalg.norm(dpos) * np.ones(2)
    neo.motion_model(dpos, dpos_sig)
    if not np.isnan(vocab_out[ii][0, 0]):
        neo.vocab_update(vocab_out[ii])
    posterior[ii] = np.copy(neo.p_x)


def animate(img_num):
    """ Get whatever """
    global t_loc, posterior, vocab_out, neo
    img_num = img_num + 110
    oimg.set_array(posterior[img_num])
    rect = patches.Rectangle(t_loc[img_num], 2, 2, edgecolor='b', fc='none', lw=4.5)
    oimg.get_axes().patches = []
    oimg.get_axes().add_patch(rect)
    oimg.get_axes().title.set_text('Geohistogram of Posterior for Img %d @ Zoom=15' % img_num)
    return oimg, title


fig = plt.figure(figsize=(10,10))
ax = plt.gca()
oimg = ax.imshow(neo.p_x, cmap='hot', interpolation='Nearest')
oimg.set_clim(0.0, posterior.max())
oimg.autoscale()
title = plt.title('Geohistogram of Posterior for Img %d @ Zoom=15' % 110)

num_frames = neogeo_out.shape[0] - 110
ani = animation.FuncAnimation(fig, animate, interval=1, frames=num_frames)
plt.show()