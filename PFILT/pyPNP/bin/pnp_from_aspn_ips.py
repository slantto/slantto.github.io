import argparse
import time
import mercantile
from datasources.lcm.messages.aspn import geodeticposition3d
from datasources.lcm.messages.aspn import positionvelocityattitude
from datasources.lcm.messages.aspn import uncorrespondedopticalcamerafeatures
from datasources.lcm.messages.aspn import gnss
from datasources.lcm.messages.aspn import opticalcameraimage
import lcm
import pnpnav.noattitude as ptude
import navfeatdb.projection.localprojection as llproj
import cv2
import navpy

import pnpnav.matching as pmatch
import pnpnav.utils as pnputils
import numpy as np
import pandas as pd

import datautils.lcmlog as ll
import dask.multiprocessing as dm
import datautils.aspn_lcmlog as dal
import glob

from scipy.interpolate import interp1d
import fiona
from geopandas import GeoDataFrame
import shapely.geometry as sg
from shapely.geometry import Point as sp
import cartopy
from cartopy import crs as ccrs


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

    log_dir = '/Volumes/Samsung USB/aspn-vlf/flight_4_04apr2017/engb/'
    ofiles = glob.glob(log_dir + '*.log.*', recursive=True)

    ofiles = glob.glob(log_dir + '/*.log.*', recursive=True)
    truth_meta = [ll.log_metadata(of) for of in ofiles]
    truth_df = pd.concat(truth_meta)

    ccm = {'aspn://vikingnav/novatel/1010.0/0': positionvelocityattitude,
           'aspn://vikingnav/image/20040/0': opticalcameraimage,
           'aspn://vikingnav/novatel/2003.0/0': gnss,
           'aspn://ips/frontend/20050/0': uncorrespondedopticalcamerafeatures,
           'aspn://ips/tracking/10013/0': geodeticposition3d}

    pva_df = dal.build_lcm_df(truth_df, 'aspn://vikingnav/novatel/1010.0/0', ccm, dm.get)
    pva_df = pva_df.join(truth_df[truth_df.channel == 'aspn://vikingnav/novatel/1010.0/0'])
    dal.add_t_valid(pva_df)

    # Get times of Feature Messages
    img_meta = dal.build_header_df(truth_df[truth_df.channel == 'aspn://ips/frontend/20050/0'], ccm, dm.get, npartitions=8)
    dal.add_t_valid(img_meta)
    img_meta = img_meta[img_meta.t_valid > 0]

    interpva = interp1d(pva_df.t_valid, pva_df[['longitude', 'latitude', 'altitude', 'roll', 'pitch', 'yaw']].T,
                        bounds_error=False)

    llh_at_img = interpva(img_meta.t_valid).T
    llh_df = pd.DataFrame(llh_at_img,
                          columns=['longitude', 'latitude', 'altitude', 'roll',
                                   'pitch', 'yaw'], index=img_meta.index)
    img_meta = pd.concat([img_meta, llh_df], axis=1)
    img_meta = pd.concat([img_meta.reset_index(drop=True), truth_df[truth_df.channel == 'aspn://ips/frontend/20050/0'].reset_index(drop=True)], axis=1)
    img_meta = img_meta[img_meta['timestamp_valid.sec'] > 0].sort_values(by='t_valid').reset_index()

    # Store off in case I break everything
    img_meta.to_hdf('/Volumes/Samsung USB/aspn-vlf/flight_4_04apr2017/kp_meta.h5', key='img_meta')


    # Ok Try to set up PNP
    # Setup PnP
    srtm_path = '/Volumes/Samsung USB/ips/srtm/Region_06'
    cam_path = '/Volumes/Samsung USB/ips/camcal_vlf/intrinsics.yaml'
    uvan_frames = '/Volumes/Samsung USB/ips/camcal_vlf/extrinsics.yaml'
    geoid_file = '/Volumes/Samsung USB/ips/geoid/egm08_25.gtx'
    db_file = '/Volumes/Samsung USB/ips/db/ohio_db.h5'

    tf = llproj.LocalLevelProjector(srtm_path, geoid_file)
    tf.load_cam_and_vehicle_frames(uvan_frames)
    tf.load_camera_cal(cam_path)
    # Boot up that PNP class
    brisk_matcher = pmatch.BFMatcher(nn_ratio=0.85,
                                     norm_type=cv2.HAMMING_NORM_TYPE)

    brisk_matcher.load_db(db_file)
    pnp2 = ptude.NoAttitudePnP(matcher=brisk_matcher)
    pnp2.load_camera_parameters(cam_path)
    pnp2.use_homography_constraint(10.0)
    pnp2.load_frames(uvan_frames)

    img_meta.longitude = np.rad2deg(img_meta.longitude)
    img_meta.latitude = np.rad2deg(img_meta.latitude)


    # Get Athens Images
    gjson = fiona.open('/Volumes/Samsung USB/ips/athens_bounds.geojson')
    athens_bounds = sg.box(*gjson.bounds)

    geometry = [sp(x, y) for x, y in zip(img_meta.longitude, img_meta.latitude)]
    img_geo_df = GeoDataFrame(img_meta, geometry=geometry,
                              crs=ccrs.PlateCarree())
    athens_imgs = img_geo_df[img_geo_df.intersects(athens_bounds)]
    athens_imgs.reset_index(inplace=True, drop=True)

    pnp_out = []
    opairs = []

    for ii, kp_row in img_meta.iterrows():
        print(ii)
        lon_lat_h = np.array([kp_row.longitude, kp_row.latitude, kp_row.altitude])
        rpy = kp_row[['roll', 'pitch', 'yaw']]
        att = pnputils.rpy_to_cnb(*rpy, units='rad')

        tile = tf.find_tile_from_pose(lon_lat_h, att)
        # tile = mercantile.tile(kp_row.longitude, kp_row.latitude, 15)
        pnp2.set_db_location_from_tiles([tile], N=10000)

        if pnp2.good_tile:

            kp = dal.log_mapper(kp_row, ccm)
            obs_kp = np.vstack((kp.x, kp.y)).T
            obs_desc = np.array([np.frombuffer(dx, dtype=np.uint8) for dx in kp.descriptor_vector])
            pnpout = pnp2.do_pnp(obs_kp,
                                     obs_desc,
                                     att,
                                     return_matches=True,
                                     return_cv=True)
            twgs, cov_n, niter, matches, status, cv_wgs = pnpout
            if status.shape[0] > 0 and twgs.shape[0] > 0:
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

                opairs.append(populate_pairs(matches, status, ned_error,
                                             lon_lat_h, rpy * 180.0 / np.pi, ii))

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

    img_data.to_hdf('/Users/venabled/data/pnpresults/aspn_vlf_pnp.hdf', key='pnp_meta')
    pd.concat(opairs).to_hdf('/Users/venabled/data/pnpresults/aspn_vlf_pairs.hdf', key='pnp_pairs')
