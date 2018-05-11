#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module subscribes to a topic providing\
vision_nav_msgs/FeatureCorrespondence2D3D messages, and uses the included 2D \
image pixel locations along with the 3D WGS-84 Lon,Lat,HAE measurements that \
were corresponded in some type of "appearance-space" matching to calculate \
a global pose (Position and Attitude). Currently this version of PnP does \
not provide a rigorous Covariance matrix for these measurements of global \
pose
"""


import pnpnav.noattitude as ptude
from scipy.interpolate import interp1d
import navfeatdb.projection.localprojection as llproj

import pnpnav.utils as pnputils
import pnpnav.matching as pmatch

import bcolz
import yaml
import os
import numpy as np
import tables as tb
import cv2
import navpy

import pandas as pd


pair_cols = ['feat_lon', 'feat_lat', 'feat_height', 'feat_img_x', 'feat_img_y',
             'veh_lon', 'veh_lat', 'veh_height',
             'veh_roll', 'veh_pitch', 'veh_yaw',
             'north_error', 'east_error', 'down_error',
             'rss_error', 'db_idx', 'img_num', 'ransac_status']


def populate_pairs(matches, status, ned_error, lon_lat_h, rpy, img_num):
    pairs = pd.DataFrame(columns=pair_cols)
    pairs['feat_lon'] = matches.world_coordinates[:, 0]
    pairs['feat_lat'] = matches.world_coordinates[:, 1]
    pairs['feat_height'] = matches.world_coordinates[:, 2]
    pairs['feat_img_x'] = matches.keypoints[:, 0]
    pairs['feat_img_y'] = matches.keypoints[:, 1]
    pairs['veh_lon'] = lon_lat_h[0]
    pairs['veh_lat'] = lon_lat_h[1]
    pairs['veh_height'] = lon_lat_h[2]
    pairs['veh_roll'] = rpy[0]
    pairs['veh_pitch'] = rpy[1]
    pairs['veh_yaw'] = rpy[2]
    pairs['north_error'] = ned_error[0]
    pairs['east_error'] = ned_error[1]
    pairs['down_error'] = ned_error[2]
    pairs['rss_error'] = np.linalg.norm(ned_error)
    pairs['db_idx'] = matches.db_idx
    pairs['img_num'] = img_num
    f_status = np.zeros(matches.world_coordinates.shape[0], dtype=np.int)
    f_status[status] = 1
    pairs['ransac_status'] = f_status
    return pairs



if __name__ == "__main__":

    flight = tb.open_file('/Users/venabled/data/uvan/fc2_f5.hdf', 'r')
    pva = flight.root.nov_span.pva
    pva_times = flight.root.nov_span.pva.cols.t_valid
    img_times = flight.root.camera.image_raw.compressed.metadata.cols.t_valid
    images = flight.root.camera.image_raw.compressed.images

    img0 = img_times[0] - 1.0
    imgN = img_times[-1] + 1.0
    good_pva = pva.read_where('(t_valid >= img0) & (t_valid <= imgN)')
    lon_lat_h = np.array(
        [good_pva[:]['lon'], good_pva[:]['lat'], good_pva[:]['height']]).T

    rpy = np.array([pnputils.DcmToRpy(dcm.reshape(3, 3)) for dcm in good_pva['c_nav_veh']])
    pva_interp = interp1d(good_pva['t_valid'], np.hstack((lon_lat_h, rpy)).T)

    srtm_path = '/Users/venabled/data/srtm/SRTM1/Region_01'
    cam_path = '/Users/venabled/pysrc/pnpnav/data/nom_autocamcal2.yaml'
    uvan_frames = '/Users/venabled/pysrc/pnpnav/data/nom_autoframes2.yaml'

    geoid_file = '/Users/venabled/data/geoid/egm96_15.tiff'

    tf = llproj.LocalLevelProjector(srtm_path, geoid_file)
    tf.load_cam_and_vehicle_frames(uvan_frames)
    tf.load_camera_cal(cam_path)
    # Boot up that PNP class
    brisk_matcher = pmatch.BFMatcher(nn_ratio=0.8,
                                     norm_type=cv2.HAMMING_NORM_TYPE)
    pnp2 = ptude.NoAttitudePnP(matcher=brisk_matcher)

    pnp2.load_pytables_db('/Users/venabled/data/neogeo/briskdb.hdf')
    pnp2.load_camera_parameters(cam_path)
    pnp2.use_homography_constraint(25.0)
    pnp2.load_frames(uvan_frames)

    flight_path = '/Users/venabled/data/uvan/fc2brisk'
    feat_meta = pd.read_hdf(os.path.join(flight_path, 'feat_meta.hdf'))

    # img_range = pd.read_hdf('/Users/venabled/data/uvan/fc2_sampled_image_nums_for_autocal.hdf').sort_values().tolist()
    img_range = np.arange(350, images.shape[0])
    # img_range = feat_meta.index

    pnp_out = []
    opairs = []
    num_feat = 5000

    img_iter = 0
    for ii in img_range:
        print('Img %d :: %d / %d' % (ii, img_iter,len(img_range)))
        img_iter += 1
        llh_rpy = pva_interp(img_times[ii])
        lon_lat_h = llh_rpy[0:3]
        att = pnputils.rpy_to_cnb(*llh_rpy[3:])

        # tile = tf.find_tile_from_pose(lon_lat_h, att)
        # pnp2.set_db_location_from_tiles([tile])
        corners_wgs = tf.project_corners(lon_lat_h, att)
        bbox = pnputils.bbox_from_pts(corners_wgs[:, 0], corners_wgs[:, 1])
        pnp2.set_db_location_from_bbox(bbox, N=10000)

        if pnp2.good_tile and feat_meta.loc[ii].num_feat > 0:

            # ft = pd.read_hdf(os.path.join(flight_path, feat_meta.iloc[ii].df_path))
            # odesc_idx = ft[ft.octave >= 1][:num_feat].index
            # obs_kp = ft[['pix_x', 'pix_y']].as_matrix()[odesc_idx]
            # obs_desc = bcolz.open(os.path.join(flight_path, feat_meta.iloc[ii].desc_path), 'r')[odesc_idx, :]

            ft = pd.read_hdf(os.path.join(flight_path, feat_meta.loc[ii].df_path))
            odesc_idx = ft[ft.octave > 0][:num_feat].index
            # odesc_idx = np.arange(num_feat)
            obs_kp = ft[['pix_x', 'pix_y']].as_matrix()[odesc_idx]
            obs_desc = bcolz.open(os.path.join(flight_path, feat_meta.loc[ii].desc_path), 'r')[odesc_idx, :].astype(np.uint8)

            center = tf.project_center(lon_lat_h, att)
            twgs, cov_n, niter, matches, status, cv_wgs = pnp2.do_pnp(obs_kp, obs_desc, att, return_matches=True, return_cv=True)
            if status.shape[0] > 0 and twgs.shape[0] > 0:
                print ("PnPIter: %d" % niter)
                cv_error = navpy.lla2ned(cv_wgs[0], cv_wgs[1], cv_wgs[2],
                                         lon_lat_h[1], lon_lat_h[0],
                                         lon_lat_h[2])
                print("OCV Error: %s :: %f " % (repr(cv_error), np.linalg.norm(cv_error)))

                ned_error = navpy.lla2ned(twgs[0], twgs[1], twgs[2],
                                          lon_lat_h[1], lon_lat_h[0],
                                          lon_lat_h[2])
                print("NED Error: %s :: %f " % (repr(ned_error), np.linalg.norm(ned_error)))

                pnp_out.append(np.hstack((ii, matches.num_correspondences, status.shape[0],
                                          np.linalg.norm(cv_error), cv_error[0], cv_error[1], cv_error[2],
                                          np.linalg.norm(ned_error), ned_error[0], ned_error[1], ned_error[2],
                                          cov_n[0], cov_n[1], cov_n[2])))

                # if np.linalg.norm(ned_error) < 100.0:
                opairs.append(populate_pairs(matches, status, ned_error,
                                             lon_lat_h, llh_rpy[3:], ii))

    img_data = pd.DataFrame(np.array(pnp_out),  columns=['img_num',
                                                         'num_correspondences',
                                                         'num_inliers',
                                                         'rss_6dof',
                                                         '6dof_n_error',
                                                         '6dof_e_error',
                                                         '6dof_d_error',
                                                         'rss_3dof',
                                                         '3dof_n_error',
                                                         '3dof_e_error',
                                                         '3dof_d_error',
                                                         'n_var',
                                                         'e_var',
                                                         'd_var'])

    img_data.to_hdf('/Users/venabled/data/pnpresults/covariance_fix.hdf', key='pnp_meta')
    pd.concat(opairs).to_hdf('/Users/venabled/data/uvan/covariance_fix_pairs.hdf', key='pnp_pairs')

    flight.close()