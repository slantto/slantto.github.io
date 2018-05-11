#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script runs neogeo from an HDF5 formatted dataset
"""

import pnpnav.utils as pnputils
import neogeodb.pytables_db as pdb
import neogeo.extent as neoextent
import numpy as np
import tables as tb
import mercantile
import time
import navpy
import pnpnav.utils as pnputils

import pyflann
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import neogeo.core as core
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid


def run_neogeo(neo, num_images, extent, obs_out, ned_vel):
    neo.reinit()
    op = np.zeros((num_images, 64, 64))
    for img_num in np.arange(num_images):
        dpos = np.copy(ned_vel[img_num, 0:2])
        dpos_sig = 0.10 * np.linalg.norm(dpos) * np.ones(2)
        neo.motion_model(dpos, dpos_sig)
        neo.update(obs_out[img_num])
        op[img_num] = np.copy(neo.p_x)
    return op


def truth_neogeo(tiles, tid, post, xb, yb):
    tidindb = np.array([np.intersect1d(tt, tid).shape[0] for tt in tiles])
    gidx = np.where(tidindb > 0)[0]
    img_in_db = gidx.shape[0]
    gcnt = 0
    gmatch = 0
    matched_idx = False
    for ii in gidx:
        amax = np.unravel_index(np.argmax(post[ii]), post[ii].shape)
        amax = np.array(amax)
        amax = amax + np.array([yb[0], xb[0]])
        tile_id = pdb.elegant_pair_xy(amax[1], amax[0])
        if tile_id in tiles[ii]:
            gcnt += 1
            if not matched_idx:
                gmatch = ii
                matched_idx = True
    return gcnt, gmatch


def namespace_test(whatever):
    print img_times[0]


t_loc = np.load('/Users/venabled/data/neogeo/t_loc.npy')
ned_vel = np.load('/Users/venabled/data/neogeo/ned_vel.npy')
obsdf = tb.open_file('/Users/venabled/data/neogeo/obs_out_mat.hdf', 'r')

# Get the Feature Database
dbf = tb.open_file(
    '/Users/venabled/data/neogeo/pytables_db.hdf', 'r')
dbt = dbf.get_node('/sift_db/sift_features_sorted')

# You need boundaries
txmin = dbt.cols.x[int(dbt.colindexes['x'][0])].astype(np.int64)
txmax = dbt.cols.x[int(dbt.colindexes['x'][-1])].astype(np.int64)
tymin = dbt.cols.y[int(dbt.colindexes['y'][0])].astype(np.int64)
tymax = dbt.cols.y[int(dbt.colindexes['y'][-1])].astype(np.int64)
xbounds = (txmin, txmax)
ybounds = (tymin, tymax)
print(xbounds)
print(ybounds)
xb, yb = neoextent.pad_grid(xbounds, ybounds)

# Full aggregate tile id
tid, tidcount = np.unique(dbt.cols.pair_id, return_counts=True)
extent = neoextent.SearchExtent(15, xb, yb, tid, tidcount)

reload(core)
neo = core.NeoGeo()
neo.init_px_from_extent(extent)
neo.reinit()
neo.pix_size_m = neoextent.get_average_tile_size(neo.extent)

posterior = np.zeros((16, 1349, 64, 64))

f2l_mat = np.array([125E3, 250E3, 500E3, 1E6])
n_img_feat = np.array([1E3, 5E3, 10E3, 20E3])

num_images = 1349

ii = 0
for f2l in f2l_mat: 
    for n_img in n_img_feat:
        obs_out = obsdf.get_node('/loaded_%d/feat_%d/obs_out' % (f2l, n_img))
        posterior[ii] = run_neogeo(neo, num_images, extent, obs_out, ned_vel)
        ii += 1
        print('%d / 16 : %d :: %d' % (ii, f2l, n_img))

ii = 0
plt.figure()
plt.subplot(4,4,1)

for ii in np.arange(16):
    plt.subplot(4, 4, ii+1)
    plt.imshow(posterior[ii].sum(0), cmap='cubehelix', interpolation='nearest')
plt.subplots_adjust(wspace=0.01, hspace=0.01)

plt.show()


fig = plt.figure()
grid = ImageGrid(fig, 111, # similar to subplot(111)
                nrows_ncols = (4, 4), # creates 2x2 grid of axes
                axes_pad=0.2, # pad between axes in inch.
                label_mode="L"
                )

for i in range(16):
    grid[i].imshow(posterior[i].sum(0), cmap='binary', interpolation='nearest')
    grid[i].tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', labelleft='off')

grid[0].set_ylabel('db = 125K')
grid[4].set_ylabel('db = 250K')
grid[8].set_ylabel('db = 500K')
grid[12].set_ylabel('db = 1M')
grid[12].set_xlabel('feat = 1K')
grid[13].set_xlabel('feat = 5K')
grid[14].set_xlabel('feat = 10K')
grid[15].set_xlabel('feat = 20K')
plt.savefig('/Users/venabled/doc/journal1/img/neogeo_grid_out.png', dpi=600)


# Get a truth finder
f5 = tb.open_file('/Users/venabled/data/uvan/fc2_f5.hdf', 'r')
imgs = f5.root.camera.image_raw.compressed.images
img_times = imgs = f5.root.camera.image_raw.compressed.metadata.cols.t_valid
pva = f5.root.nov_span.pva

srtm_path = '/Users/venabled/data/srtm/SRTM1/Region_01'
geoid_file = '/Users/venabled/data/geoid/egm96_15.tiff'
frame_yaml = '/Users/venabled/pysrc/pnpnav/data/fc2_pod_frames.yaml'

finder = pnputils.DEMTileFinder(srtm_path, geoid_file)
finder.load_cam_and_vehicle_frames(frame_yaml)
finder.load_camera_cal('/Users/venabled/pysrc/pnpnav/data/fc2_cam_model.yaml')


corners = np.zeros((img_times.shape[0], 4, 3))
for img_num in np.arange(img_times.shape[0]):
    dt = pva.cols.t_valid - img_times[img_num]
    ii = np.abs(dt).argmin()
    lon_lat_h = np.array([pva[ii]['lon'], pva[ii]['lat'], pva[ii]['height']])
    att = pva[ii]['c_nav_veh'].reshape(3, 3)
    corners[img_num] = finder.get_corners(lon_lat_h, att)[1]
    print('image: %d / %d' % (img_num, img_times.shape[0]))

centers = np.array([corner[:,:2].mean(0) for corner in corners])
bbox = neoextent.bbox_from_extent(extent)
c_wgs = np.hstack((np.array(bbox)[:, [1, 0]], np.zeros((4,1))))
c_ned = navpy.lla2ned(c_wgs[:, 0], c_wgs[:, 1], c_wgs[:, 2],
                      c_wgs[3, 0], c_wgs[3, 1], c_wgs[3, 2])
pix_width = np.abs(c_ned[1,:2]).mean() / 64.0
cent_wgs = np.hstack((centers[:, [1, 0]], np.zeros((centers.shape[0], 1))))
cent_ned = navpy.lla2ned(cent_wgs[:, 0], cent_wgs[:, 1], cent_wgs[:, 2],
                         c_wgs[3, 0], c_wgs[3, 1], c_wgs[3, 2])
cent_pix = cent_ned / pix_width


plt.figure()
ax = plt.gca()
plt.title('Center Point of Observation Overlaid on Posterior Aggregation')
im = ax.imshow(posterior[-1].sum(0), cmap='binary', interpolation='nearest')
ax.set_xlim(20, 63)
ax.set_ylim(63, 26)
ax.autoscale(False)
line = ax.plot(cent_pix[:, 1] - 0.5, -1*cent_pix[:, 0] - 0.5, 'k')
# ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
plt.xlabel('Spherical Mercator Tiles (Z=15)')
plt.ylabel('Spherical Mercator Tiles (Z=15)')
plt.savefig('/Users/venabled/doc/journal1/img/truth_posterior.png', bbox_inches='tight', dpi=600)

plt.figure()
plt.imshow(posterior[-1].sum(0), cmap='cubehelix', interpolation='nearest')
plt.plot(cent_pix[:, 1] - 0.5, -1*cent_pix[:, 0] - 0.5, 'w')

tiles = []
for corner in corners:
    west = corner[:, 0].min()
    east = corner[:, 0].max()
    north = corner[:, 1].max()
    south = corner[:, 1].min()
    tiles.append([ndb.elegant_pair_xy(t.x, t.y) for t in mercantile.tiles(west, south, east, north, [15])])


gcnt = np.zeros(16)
first_id = np.zeros(16)
for ii in np.arange(16):
    gcnt[ii], first_id[ii] = truth_neogeo(tiles, tid, posterior[ii], xb, yb)



f_times = np.zeros(16)
ii = 0
for f2l in f2l_mat: 
    for n_img in n_img_feat:
        f_time = obsdf.get_node('/loaded_%d/feat_%d/flann_time' % (f2l, n_img))
        f_times[ii] = f_time[:].mean()
        ii += 1

num_features = np.zeros(1349)
for ii in np.arange(1349):
    array = f5.get_node('/images/sift_features/img_%d' % ii)
    num_features[ii] = array.nrows


