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

    flight = tb.open_file('/Users/venabled/data/uvan/fc2_f5.hdf', 'r')
    pva = flight.root.nov_span.pva
    pva_times = flight.root.nov_span.pva.cols.t_valid
    img_times = flight.root.camera.image_raw.compressed.metadata.cols.t_valid
    images = flight.root.camera.image_raw.compressed.images

    out_path = '/Users/venabled/data/uvan/fc2/f5'

    img0 = img_times[0] - 1.0
    imgN = img_times[-1] + 1.0
    good_pva = pva.read_where('(t_valid >= img0) & (t_valid <= imgN)')
    lon_lat_h = np.array(
        [good_pva[:]['lon'], good_pva[:]['lat'], good_pva[:]['height']]).T

    rpy = np.array([nu.DcmToRpy(dcm.reshape(3, 3)) for dcm in good_pva['c_nav_veh']])

    pva_interp = interp1d(good_pva['t_valid'], np.hstack((lon_lat_h, rpy)).T)

    srtm_path = '/Users/venabled/data/srtm/SRTM1/Region_01'
    cam_path = '/Users/venabled/pysrc/pnpnav/data/fc2_cam_model.yaml'
    uvan_frames = '/Users/venabled/pysrc/navfeatdb/data/fc2_pod_frames.yaml'
    geoid_file = '/Users/venabled/data/geoid/egm96_15.tiff'

    tf = llproj.LocalLevelProjector(srtm_path, geoid_file)
    tf.load_cam_and_vehicle_frames(uvan_frames)
    tf.load_camera_cal(cam_path)

    kml = simplekml.Kml()
    img_folder = kml.newfolder(name='UVAN Acquisition Imagery')
    mf = kml.newfolder(name='Matches')

    img_range = [342,  343,  344,  345,  346,  347,  348,  349,  350,  419,  755,
       1239, 1241, 1242, 1243, 1244, 1245, 1246]

    # Dump image
    for ii in img_range:

        llh_rpy = pva_interp(img_times[ii])
        lon_lat_h = llh_rpy[0:3]
        c_n_v = nu.rpy_to_cnb(*llh_rpy[3:])

        img_path = os.path.join(out_path, 'imgs/uvan_%d.jpg' % ii)
        k_img = kml.addfile(img_path)
        undistort = cv2.undistort(images[ii], tf.K, tf.distortion)
        cv2.imwrite(img_path, undistort)

        corners_wgs = tf.project_corners(lon_lat_h, c_n_v)

        ground = img_folder.newgroundoverlay(name='UVAN_%d' % ii)
        ground.icon.href = img_path
        ground.color = 'ffffffff'
        ground.gxlatlonquad.coords = [tuple(c[:2]) for c in corners_wgs]

        ground.timespan.begin = datetime.datetime.fromtimestamp(
            img_times[ii]).strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time = img_times[ii + 1]
        ground.timespan.end = datetime.datetime.fromtimestamp(
            end_time).strftime('%Y-%m-%dT%H:%M:%SZ')

    kml.save(out_path + '/images.kml')

