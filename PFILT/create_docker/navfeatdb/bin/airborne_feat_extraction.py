#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import tables as tb
import bcolz
import navfeatdb.utils.cvfeat2d as f2d
import cv2
import navfeatdb.db.features as fdb
from scipy.interpolate import interp1d

import navfeatdb.projection.localprojection as llproj
import navfeatdb.utils.nav as nu
import pandas as pd
import shutil

import argparse

import odo

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='airborne_feat_extraction',
                                     description='Extracts and GeoLocates Features From a UVAN Flight')
    parser.add_argument('-o', '--output',
                        help='Directory where to create test datasets')
    parser.add_argument('-g', '--geoid', help='Path to Geoid File')
    parser.add_argument('-s', '--srtm', help='Path to SRTM Root')
    parser.add_argument('-m', '--meta', help='Path to UVAN Metadata')
    parser.add_argument('-u', '--uvan', help='Path to UVAN HDF5 Flight')
    parser.add_argument('-a', '--append', default=True)
    args = parser.parse_args()


    flight = tb.open_file(args.uvan, 'r')
    pva = flight.root.nov_span.pva
    pva_times = flight.root.nov_span.pva.cols.t_valid
    img_times = flight.root.camera.image_raw.compressed.metadata.cols.t_valid
    images = flight.root.camera.image_raw.compressed.images

    out_path = args.output
    if (not args.append) and (os.path.exists(out_path)):
        shutil.rmtree(out_path)

    if not os.path.exists(out_path):
        os.makedirs(os.path.join(out_path, 'feat/df'))
        os.makedirs(os.path.join(out_path, 'feat/desc'))

    img0 = img_times[0] - 1.0
    imgN = img_times[-1] + 1.0
    good_pva = pva.read_where('(t_valid >= img0) & (t_valid <= imgN)')
    lon_lat_h = np.array(
        [good_pva[:]['lon'], good_pva[:]['lat'], good_pva[:]['height']]).T

    rpy = np.array([nu.DcmToRpy(dcm.reshape(3, 3)) for dcm in good_pva['c_nav_veh']])
    pva_interp = interp1d(good_pva['t_valid'], np.hstack((lon_lat_h, rpy)).T)

    srtm_path = args.srtm
    cam_path = os.path.join(args.meta, 'fc2_cam_model.yaml')
    uvan_frames = os.path.join(args.meta, 'fc2_pod_frames.yaml')
    geoid_file = args.geoid

    tf = llproj.LocalLevelProjector(srtm_path, geoid_file)
    tf.load_cam_and_vehicle_frames(uvan_frames)
    tf.load_camera_cal(cam_path)

    # det_f = os.path.join(args.meta, 'default_detector_SIFT.yaml')
    # des_f = os.path.join(args.meta, 'default_descriptor_SIFT.yaml')
    # detector, descriptor = f2d.load_feature_operators(det_f, des_f)
    # detector = cv2.BRISK_create()
    # descriptor = cv2.BRISK_create()

    detector = cv2.xfeatures2d.SIFT_create()
    descriptor = cv2.xfeatures2d.SIFT_create()

    aie = fdb.AirborneImageExtractor(detector, descriptor, tf)

    img_range = np.arange(img_times.shape[0])

    # TODO: Make sure to remove this line
    # img_range = np.arange(340, 345)
    # img_range = np.array([343])

    feat_meta = pd.DataFrame(index=img_range,
                             columns=['num_feat', 'center_lon', 'center_lat', 'df_path', 'desc_path'])

    for ii in img_range:
        print('%d / %d' % (ii, img_range.shape[0]))
        llh_rpy = pva_interp(img_times[ii])
        lon_lat_h = llh_rpy[0:3]
        c_n_v = nu.rpy_to_cnb(*llh_rpy[3:])
        feat_df, desc = aie.extract_features(images[ii], lon_lat_h, c_n_v)
        center_wgs = tf.project_center(lon_lat_h, c_n_v).flatten()
        df_path = 'feat/df/feat_%d.hdf' % ii
        desc_path = 'feat/desc/desc_%d' % ii



        if feat_df is None:
            feat_meta.loc[ii] = [0, center_wgs[0], center_wgs[1], df_path, desc_path]
        else:
            print("%d :: %d Feat" % (ii, desc.shape[0]))
            feat_meta.loc[ii] = [desc.shape[0], center_wgs[0], center_wgs[1], df_path, desc_path]
            feat_df.to_hdf(os.path.join(out_path, df_path), 'feat_df', mode='w', format='table', complib='zlib', complevel=7)
            bcolz.carray(desc.astype(np.float32), rootdir=os.path.join(out_path, desc_path), mode='w').flush()


    feat_meta.to_hdf(os.path.join(out_path, 'feat_meta.hdf'), key='feat_meta')
    flight.close()