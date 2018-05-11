#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime

import cv2
import numpy as np
import simplekml
import tables as tb
import pandas as pd
import os
import navpy

import navfeatdb.projection.localprojection as llproj
import navfeatdb.utils.nav as nu
from scipy.interpolate import interp1d


if __name__ == '__main__':


    out_path = '/data/aspn/s3/june2016'
    athens_images = pd.read_hdf('/data/aspn/s3/june2016/athens_imgs.h5')
    proj_imgs = athens_images.loc[225:275]

    lon_lat_h = proj_imgs[['longitude', 'latitude', 'altitude']]
    rpy = proj_imgs[['roll', 'pitch', 'yaw']] * 180.0 / np.pi

    tf = llproj.LocalLevelProjector('/home/RYWN_Data/srtm/SRTM1/Region_06', '/data/geoid/egm08_25.gtx')
    tf.load_camera_cal('/home/venabldt/pysrc/navfeatdb/data/aspn_s3_cam_model.yaml')
    tf.load_cam_and_vehicle_frames('/home/venabldt/pysrc/navfeatdb/data/aspn_s3_frames.yaml')


    kml = simplekml.Kml()
    img_folder = kml.newfolder(name='Athens Aspn Projection')
    mf = kml.newfolder(name='Images')

    # Dump image
    for ii, row in proj_imgs.iterrows():

        lon_lat_h = np.array(row[['longitude', 'latitude', 'altitude']])
        c_n_v = nu.rpy_to_cnb(row['roll'], row['pitch'], row['yaw'], units='rad')

        img_path = os.path.join(out_path, 'proj_imgs/img_%d.png' % ii)
        k_img = kml.addfile(img_path)

        corners_wgs = tf.project_corners(lon_lat_h, c_n_v)

        ground = img_folder.newgroundoverlay(name='ASPN_%d' % ii)
        ground.icon.href = img_path
        ground.color = 'ffffffff'
        ground.gxlatlonquad.coords = [tuple(c[:2]) for c in corners_wgs]

        ground.timespan.begin = datetime.datetime.fromtimestamp(
            row.t_valid_est).strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time = row.t_valid_est + 0.10
        ground.timespan.end = datetime.datetime.fromtimestamp(
            end_time).strftime('%Y-%m-%dT%H:%M:%SZ')

    kml.save(out_path + '/images.kml')
