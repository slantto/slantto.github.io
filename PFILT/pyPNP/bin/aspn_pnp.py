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
import navfeatdb.db.features as ndbfeat

import pnpnav.utils as pnputils
import pnpnav.matching as pmatch

import yaml
import os
import tables as tb
import cv2
import navpy

import numpy as np
import pandas as pd
import glob
import datautils.lcmlog as ll
from datasources.lcm.messages.aspn import positionvelocityattitude, opticalcameraimage
import dask.multiprocessing as dm
import datautils.aspn_lcmlog as dal
import mercantile
from navfeatdb.frames import rasterio2d
import neogeodb.hdf_orthophoto as hdfo
import cv2
import navfeatdb.frames as ndbframes
import navfeatdb.utils.cvfeat2d as f2d
import navfeatdb.utils.matching as matchutil
from navfeatdb.projection import localprojection as llp
import navfeatdb.utils.nav as ndu
import skimage.transform as skt
import matplotlib.pyplot as plt



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

    athens_imgs = pd.read_hdf('/data/aspn/s3/june2016/athens_imgs.h5', 'df')

    ccm = {'aspn://vikingnav/novatel/1010.0/0': positionvelocityattitude,
           'aspn://vikingnav/image/20040/0': opticalcameraimage}

    cam_path = '/data/aspn/s3/june2016/camcal/intrinsics.yaml'
    frame_path = '/data/aspn/s3/june2016/camcal/extrinsics.yaml'
    srtm_path = '/home/RYWN_Data/srtm/SRTM1/combined'
    geoid_file = '/data/geoid/egm08_25.gtx'
    db_path = '/data/osip/db/athens_fubar.h5'

    tf = llproj.LocalLevelProjector(srtm_path, geoid_file)
    tf.load_cam_and_vehicle_frames(frame_path)
    tf.load_camera_cal(cam_path)
    # Boot up that PNP class
    brisk_matcher = pmatch.BFMatcher(nn_ratio=0.8,
                                     norm_type=cv2.HAMMING_NORM_TYPE)
    pnp2 = ptude.NoAttitudePnP(matcher=brisk_matcher)

    pnp2.load_pytables_db(db_path)
    pnp2.load_camera_parameters(cam_path)
    pnp2.use_homography_constraint(25.0)
    pnp2.load_frames(frame_path)

    img_iter = 0
    sample_num = 1000

    brisk = cv2.BRISK_create()

    pnp_out = []
    opairs = []

    for ii, row in athens_imgs.sample(sample_num).iterrows():
        print('Img %d :: %d / %d' % (ii, img_iter, sample_num))
        img_iter += 1

        lon_lat_h = np.array(row[['longitude', 'latitude', 'altitude']])
        att = ndu.rpy_to_cnb(row['roll'], row['pitch'], row['yaw'], units='rad')
        rpy = np.array([row['roll'], row['pitch'], row['yaw']]) * 180.0 / np.pi

        corners_wgs = tf.project_corners(lon_lat_h, att)
        bbox = pnputils.bbox_from_pts(corners_wgs[:, 0], corners_wgs[:, 1])
        # pnp2.set_db_location_from_bbox(bbox, N=10000)
        tile = tf.find_tile_from_pose(lon_lat_h, att)
        pnp2.set_db_location_from_tiles([tile], N=10000)


        print(bbox)

        aspn_img = dal.log_mapper(row, ccm)
        img = np.frombuffer(aspn_img.data, np.uint8).reshape(aspn_img.height, aspn_img.width)

        obs_cv_kp, obs_desc = brisk.detectAndCompute(img, mask=None)
        air_resp = [kp.response for kp in obs_cv_kp]
        air_resp_idx = np.argsort(air_resp)[-5000:]
        obs_kp = f2d.keypoint_pixels(obs_cv_kp)[air_resp_idx]
        obs_desc = obs_desc[air_resp_idx]


        if pnp2.good_tile and len(air_resp_idx) > 0:

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

                if np.linalg.norm(ned_error) < 500.0:
                    opairs.append(populate_pairs(matches, status, ned_error, lon_lat_h, rpy, ii))


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

    img_data.to_hdf('/data/aspn/s3/june2016/pnp/pnp_newdb_results.hdf', key='pnp_meta')
    pd.concat(opairs).to_hdf('/data/aspn/s3/june2016/pnp/pnp_newdb_pairs.hdf', key='pnp_pairs')

