#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script runs neogeo from an HDF5 formatted dataset
"""

import neogeo.extent as neoextent
import neogeo.utils as nutils
import numpy as np
import tables as tb
import neogeodb.pytables_db as pdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import neogeo.core as core

import bcolz

import neogeo.core as core

def animate(img_num):
    """ Get whatever """
    global t_loc, posterior, obs_out, neo
    img_num = img_num + 110
    oimg.set_array(posterior[img_num])
    rect = patches.Rectangle(t_loc[img_num], 2, 2, edgecolor='b', fc='none', lw=4.5)
    oimg.get_axes().patches = []
    oimg.get_axes().add_patch(rect)
    oimg.get_axes().title.set_text('Geohistogram of Posterior for Img %d @ Zoom=15' % img_num)
    return oimg, title


if __name__ == '__main__':
    t_loc =   np.load('/Users/venabled/data/neogeo/t_loc.npy')
    ned_vel = np.load('/Users/venabled/data/neogeo/ned_vel.npy')

    # Get the Feature Database
    dbf = tb.open_file(
        '/Users/venabled/data/neogeo/pytables_db.hdf', 'r')
    dbt = dbf.get_node('/sift_db/sift_features_sorted')

    # Open the flat observations
    obs = bcolz.open('/Users/venabled/data/uvan/neogeo/f5_10k_tfidf_obs', mode='r')
    obs_tid = bcolz.open('/Users/venabled/data/uvan/vocab/10k_vocabpydb_pair_id', mode='r')[0]
    obs_xy = np.array([pdb.unpair(z) for z in obs_tid]).astype(int)

    # You need boundaries
    txmin = dbt.cols.x[int(dbt.colindexes['x'][0])]
    txmax = dbt.cols.x[int(dbt.colindexes['x'][-1])]
    tymin = dbt.cols.y[int(dbt.colindexes['y'][0])]
    tymax = dbt.cols.y[int(dbt.colindexes['y'][-1])]
    xbounds = np.array((txmin, txmax), dtype=np.int64)
    ybounds = np.array((tymin, tymax), dtype=np.int64)
    xb, yb = neoextent.pad_grid(xbounds, ybounds)

    # Full aggregate tile id
    tid, tidcount = np.unique(dbf.root.sift_db.unique_tiles[:], return_counts=True)
    extent = neoextent.SearchExtent(6, xb, yb, tid, tidcount)

    # Reshape the observations to fit the grid.
    obs_out = np.zeros((obs.shape[0], extent.grid_size, extent.grid_size))
    for ii, i_obs in enumerate(obs):
        obs_out[ii] = nutils.plot_on_grid(obs_xy[:, 0], obs_xy[:, 1], i_obs, xb, yb)

    neo = core.NeoGeo()
    neo.init_px_from_extent(extent)
    posterior = np.zeros_like(obs_out)

    for ii in np.arange(0, obs_out.shape[0]):
        dpos = np.copy(ned_vel[ii, 0:2])
        dpos_sig = 0.10 * np.linalg.norm(dpos) * np.ones(2)
        neo.motion_model(dpos, dpos_sig)
        neo.update(obs_out[ii])
        posterior[ii] = np.copy(neo.p_x)


    fig = plt.figure(figsize=(10,10))
    ax = plt.gca()
    oimg = ax.imshow(neo.p_x, cmap='hot', interpolation='Nearest')
    oimg.set_clim(0.0, posterior.max())
    oimg.autoscale()
    title = plt.title('Geohistogram of Posterior for Img %d @ Zoom=15' % 110)

    num_frames = obs_out.shape[0] - 110
    ani = animation.FuncAnimation(fig, animate, interval=1, frames=num_frames)