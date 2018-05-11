#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script runs neogeo from an HDF5 formatted dataset
"""
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import navpy
import neogeo.extent as neoextent
import numpy as np
import pandas as pd
import pnpnav.utils as pnputils
import tables as tb

from neogeo import particle_core as core

if __name__ == '__main__':
    out_tb = tb.open_file('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/obs_out_mat6.hdf', 'r')

    neogeo_out = out_tb.root.loaded_1000000.feat_10000.neogeo_out
    obs_out = out_tb.root.loaded_1000000.feat_10000.obs_out
    t_loc =      np.load('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/t_loc.npy')
    ned_vel =    np.load('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/ned_vel.npy')

    # Get the Feature Database
    dbf = tb.open_file(
        '/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/pytables_db.hdf', 'r')
    dbt = dbf.get_node('/sift_db/sift_features_sorted')
    f5 = tb.open_file('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/fc2_f5.hdf', 'r')
    f5featmeta = pd.read_hdf('/media/sean/D2F2E7B2F2E798CD/Users/student/AIEoutput2/feat/feat_meta.hdf')
    imgs = f5.root.camera.image_raw.compressed.images
    img_times = f5.root.camera.image_raw.compressed.metadata.col('t_valid')
    feat_path = f5featmeta.iloc[:, 3]
    descripath = f5featmeta.iloc[:, 4]
    # feat_tables = f5.list_nodes(f5.root.images.sift_features)
    pva = f5.root.nov_span.pva

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
    particles = neo.create_uniform_particles(xb,yb,0.5,10000)

    neo = core.NeoGeo()
    neo.init_px_from_extent(extent)
    posterior = np.zeros_like(obs_out)

    for ii in np.arange(0, obs_out.shape[0]):
        dpos = np.copy(ned_vel[ii, 0:2])
        dpos_sig = 0.10 * np.linalg.norm(dpos) * np.ones(2)
        neo.motion_model(dpos, dpos_sig)
        neo.update(obs_out[ii])
        posterior[ii] = np.copy(neo.p_x)

    plt.figure()
    plt.imshow(posterior.sum(0), cmap='cubehelix', interpolation='nearest')
    plt.show()
    plt.savefig('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/posterior2.png')


#def animate(ii):
#    global posterior, p0
#    oimg.set_array(posterior[ii] - p0[ii])
#    oimg.get_axes().title.set_text(repr(ii))
#    oimg.autoscale()
#    return oimg
#
#fig = plt.figure(figsize=(10,10))
#ax = plt.gca()
#oimg = ax.imshow(neo.p_x, cmap='hot', interpolation='Nearest')
#oimg.autoscale()
#title = plt.title('Geohistogram of Posterior for Img %d @ Zoom=15' % 0)
#num_frames = neogeo_out.shape[0]
#ani = animation.FuncAnimation(fig, animate, interval=1, frames=num_frames)


 #Need to truth this thing out
 #Get me center points, tiles, and corners of the image for every image
#import pnpnav.utils as pnputils
f5 = tb.open_file('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/fc2_f5.hdf', 'r')
f5featmeta = pd.read_hdf('/media/sean/D2F2E7B2F2E798CD/Users/student/AIEoutput2/feat/feat_meta.hdf')
imgs =f5.root.camera.image_raw.compressed.images
img_times = f5.root.camera.image_raw.compressed.metadata.col('t_valid')
feat_path = f5featmeta.iloc[:,3]
descripath = f5featmeta.iloc[:,4]
#feat_tables = f5.list_nodes(f5.root.images.sift_features)
pva = f5.root.nov_span.pva

 # Get a truth finder
dted_path = '/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/srtm'
geoid_file = '/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/egm96-15.tif'
frame_yaml = '/home/sean/ImageAidedNav/navfeatdb/data/fc2_pod_frames.yaml'

finder = pnputils.DEMTileFinder(dted_path, geoid_file)
finder.load_cam_and_vehicle_frames(frame_yaml)
finder.load_camera_cal('/home/sean/ImageAidedNav/navfeatdb/data/fc2_cam_model.yaml')

corners = np.zeros((img_times.shape[0], 4, 3))
for img_num in np.arange(img_times.shape[0]):
    dt = pva.cols.t_valid - img_times[img_num]
    ii = np.abs(dt).argmin()
    lon_lat_h = np.array([pva[ii]['lon'], pva[ii]['lat'], pva[ii]['height']])
    #lon_lat_h = pva.pos[ii, :]
    att = pva[ii]['c_nav_veh'].reshape(3, 3)
    #att = pva.c_nav_veh[ii, :].reshape(3, 3)
    corners[img_num] = finder.get_corners(lon_lat_h, att)[1]
    print("imagani.save('whatever.mp4', fps=4, extra_args=['-vcodec', 'libx264'])e: %d / %d" % (img_num, img_times.shape[0]))

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

im = ax.imshow(posterior.sum(0), cmap='cubehelix', interpolation='nearest')
ax.set_xlim(20, 63)
ax.set_ylim(63, 26)
ax.autoscale(False)#10
line = ax.plot(cent_pix[:, 1] - 0.5, -1*cent_pix[:, 0] - 0.5, 'w')
 # ax.xaxis.set_visible(False)
 # ax.yaxis.set_visible(False)
plt.xlabel('Spherical Mercator Tiles (Z=15)')
plt.ylabel('Spherical Mercator Tiles (Z=15)')
plt.savefig('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/truth_posterior2.png', bbox_inches='tight', dpi=400)


def animate(img_num):
    """ Get whatever """
    global t_loc, posterior, obs_out, neo
    img_num = img_num + 110
    #oimg.set_array(posterior[img_num])
    oimg.set_array(obs_out[img_num])
    rect = patches.Rectangle(t_loc[img_num], 2, 2, edgecolor='b', fc='none', lw=4.5)
    oimg.get_axes().patches = []
    oimg.get_axes().add_patch(rect)
    oimg.get_axes().title.set_text('Geohistogram of Posterior for Img %d @ Zoom=15' % img_num)
    return oimg, title


fig = plt.figure(figsize=(10,10))
ax = plt.gca()
print(neo.p_x.shape)
print(obs_out.shape)
oimg = ax.imshow(obs_out[350], cmap='hot', interpolation='Nearest')
#oimg = ax.imshow(neo.p_x, cmap='hot', interpolation='Nearest')
#oimg.set_clim(0.0, posterior.max())
#oimg.set_clim(0.0, obs_out.max())
oimg.autoscale()
title = plt.title('Geohistogram of Posterior for Img %d @ Zoom=15' % 110)

num_frames = neogeo_out.shape[0] - 110
ani = animation.FuncAnimation(fig, animate, interval=1, frames=num_frames)

#ani.save('neo_posterior.gif', writer='imagemagick', fps=8.0)
ani.save('whateverseanrefobs.mp4', fps=4, extra_args=['-vcodec', 'libx264'])


import simplekml
import datetime

img_times = f5.root.camera.image_raw.compressed.metadata.cols.t_valid
out_path = '/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data'
kml = simplekml.Kml()
img_folder = kml.newfolder(name='UVAN Acquisition Imagery')
ecorners = neoextent.bbox_from_extent(extent)

for ii in np.arange(300, 1349):
    img_path = out_path + '/acq_%d.png' % ii
    rel_path = './acq_%d.png' % ii
    k_img = kml.addfile(rel_path)
    img = st.resize(posterior[ii], (1024, 1024), order=0)
    img2 = cm.hot(img)
    img2_min = img2[:, :, 0:3].sum(2).min()
    img_mask = img2[:, :, 0:3].sum(2) == img2_min

    img3 = np.zeros((1024, 1024))
    cv2.rectangle(img3, tuple((t_loc[ii]) * 16 + 8 ) , tuple((t_loc[ii] + 2) * 16 + 8), 1.0, 5)

    img2[:, :, 3] = 0.90
    img2[img_mask, 3] = 0.80
    img2[:, :, 2] = img3
    img2[img3 > 0, 3] = 1.0
    plt.imsave(img_path, img2)

    ground = img_folder.newgroundoverlay(name='acq_%d' % ii)
    ground.icon.href = img_path
    ground.color = 'ffffffff'
    ground.gxlatlonquad.coords = ecorners

    ground.timespan.begin = datetime.datetime.fromtimestamp(
        img_times[ii] - 1.0).strftime('%Y-%m-%dT%H:%M:%SZ')
    ground.timespan.end = datetime.datetime.fromtimestamp(
        img_times[ii] + 1.0).strftime('%Y-%m-%dT%H:%M:%SZ')

kml.save('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/acq.kml')



flight = tb.open_file('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/fc2_f5.hdf','r')
pva = flight.root.nov_span.pva
image_times = flight.root.camera.image_raw.compressed.metadata
images = flight.root.camera.image_raw.compressed.images

img0 = image_times.cols.t_valid[0]
imgN = image_times.cols.t_valid[-1]
good_pva = pva.read_where('(t_valid >= img0) & (t_valid <= imgN)')
lon_lat_h = np.array(
    [good_pva[:]['lon'], good_pva[:]['lat'], good_pva[:]['height']])

kml = simplekml.Kml()
doc = kml.newdocument
track_folder = kml.newfolder(name='UVAN Track')
track = track_folder.newgxtrack()
track.altitudemode = simplekml.AltitudeMode.absolute
track.whens = [ datetime.datetime.fromtimestamp(row['t_valid']).strftime('%Y-%m-%dT%H:%M:%SZ') for row in good_pva ]
track.newgxcoord([(row['lon'], row['lat'], row['height']) for row in good_pva])
#track.newgxangle([(r[2], r[1], r[0]) for r in rpy])
track.model = simplekml.Model()
track.model.link = simplekml.Link('/Users/venabled/data/models/Aircraft_General_Atomics_RQ_1A_Predator.kmz')
track.model.scale.x = 10
track.model.scale.y = 10
track.model.scale.z = 10

#Save the KML
kml.save('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/uvan_track.kml')




# #Plots for journal article
# neo.obs_0 = np.ones((64,64))
# obs = np.maximum(obs_out[705], neo.obs_0)
# obs = obs / obs.sum()

# nmap = 'Blues'
# fig, (ax1, ax2, ax3) = plt.subplots(1,3)

# #First Subplot
# ax1.set_title('Prior')
# im1 = ax1.imshow(posterior[704][32:, 15:47], interpolation='nearest', cmap=nmap, aspect='equal')
# divider1 = make_axes_locatable(ax1)
# cax1 = divider1.append_axes("right", size="5%", pad=0.05)
# cbar1 = plt.colorbar(im1, cax=cax1)
# ax1.xaxis.set_visible(False)
# ax1.yaxis.set_visible(False)


# #Second Subplot
# ax2.set_title('Observation')
# im2 = ax2.imshow(obs[32:, 15:47], interpolation='nearest', cmap=nmap, aspect='equal')
# divider2 = make_axes_locatable(ax2)
# cax2 = divider2.append_axes("right", size="5%", pad=0.05)
# cbar2 = plt.colorbar(im2, cax=cax2)
# ax2.xaxis.set_visible(False)
# ax2.yaxis.set_visible(False)

# #Third Subplot
# ax3.set_title('Posterior')
# im3 = ax3.imshow(posterior[706][32:, 15:47], interpolation='nearest', cmap=nmap, aspect='equal')
# divider3 = make_axes_locatable(ax3)
# cax3 = divider3.append_axes("right", size="5%", pad=0.05)
# cbar3 = plt.colorbar(im3, cax=cax3)
# ax3.xaxis.set_visible(False)
# ax3.yaxis.set_visible(False)
# plt.tight_layout()

# plt.savefig('/Users/venabled/doc/journal1/img/point_mass_filter.png', bbox_inches='tight', dpi=400)



# #aggregations ....
# flights = os.listdir('./')
# inlier_tiles = []
# pnp_err = []
# inliers_idx = []
# outliers_tiles = []
# inliers_response = []
# outliers_response = []
# outliers_size = []
# inliers_size = []
# cam_response = []
# cam_size = []
# cam_height = []

# for flight in flights:
#     ofile = whatever
#     pnp = ofile.get_node('/pnp')
#     inliers = ofile.get_node('/inliers')
#     pnp_err.append(pnp.cols.ned_error[:])
#     inliers_idx.append(inliers.cols.db_idx[:])
#     inliers_response.append(inliers.cols.db_response[:])
#     inliers_size.append(inliers.cols.db_size[:])
#     inlier_tiles.append(inliers.cols.pair_id[:])
#     cam_response.append(inliers.cols.img_response[:])
#     cam_size.append(inliers.cols.img_size[:])
#     cam_height.append(inliers.cols.cam_height[:])
#     outliers_response.append(outliers.cols.db_response[:])
#     outliers_size.append(outliers.cols.db_size[:])
#     outliers_tiles.append(outliers.cols.pair_id[:])
#     ofile.close()


# pnp_err = np.vstack(pnp_err)
# inlier_tiles = np.hstack(inlier_tiles)
# inliers_idx = np.hstack(inliers_idx)
# outliers_tiles = np.hstack(outliers_tiles)
# inliers_response = np.hstack(inliers_response)
# outliers_response = np.hstack(outliers_response)
# outliers_size = np.hstack(outliers_size)
# inliers_size = np.hstack(inliers_size)
# cam_response = np.hstack(cam_response)
# cam_size = np.hstack(cam_size)
# cam_height = np.hstack(cam_height)

