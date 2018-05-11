#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime

import numpy as np
import simplekml
import pandas as pd
import os

import navfeatdb.projection.localprojection as llproj
import navfeatdb.utils.nav as nu
from scipy.interpolate import interp1d
import argparse

if __name__ == '__main__':


    parser = argparse.ArgumentParser(prog='cland_clif_projector.py',
                                     description='Creates a KML file of projected CLAND imagery')
    parser.add_argument('-c', '--clif',
                        help='CLIF Folder', required=True)
    parser.add_argument('-f', '--frames', required=True,
                        help='Path to extrinsic calibration YAML file')
    parser.add_argument('-i', '--intrinsic', required=True,
                        help='Path to intrinsic calibration YAML file')
    parser.add_argument('-s', '--srtm', required=True,
                        help='Root directory of SRTM HGT terrain files')
    parser.add_argument('-o', '--out', required=True,
                        help='Directory where you want the output KML file')
    parser.add_argument('-w', '--csv',
                        help='Which camera.csv file to use, default=camera_nadir.csv',
                        default='camera_nadir.csv')
    parser.add_argument('-g', '--geoid', required=True,
                        help='Path to the geoid-shift file to be used')
    args = parser.parse_args()

    pva = pd.read_csv(os.path.join(args.clif, 'novatel_INSPositionVelocityAttitudeShort.csv'))
    lon_lat_h = pva[['longitude', 'latitude', 'height']]
    rpy = pva[['roll', 'pitch', 'azimuth']]

    pva_interp = interp1d(pva.gps_ms, np.hstack((lon_lat_h, rpy)).T)

    srtm_path = args.srtm
    cam_path = args.intrinsic
    cland_frames = args.frames
    geoid_file = args.geoid

    tf = llproj.LocalLevelProjector(srtm_path, geoid_file)
    tf.load_cam_and_vehicle_frames(cland_frames)
    tf.load_camera_cal(cam_path)

    kml = simplekml.Kml()
    img_folder = kml.newfolder(name='CLIF Imagery')
    mf = kml.newfolder(name='Projection')

    clif_dir = os.path.realpath(args.clif)
    ndf = pd.read_csv(os.path.join(clif_dir, args.csv))
    ndf = ndf[(ndf.gps_ms > pva.iloc[0].gps_ms) & (ndf.gps_ms < pva.iloc[-1].gps_ms)]

    for ii, ndf_row in ndf.iterrows():

        llh_rpy = pva_interp(ndf_row.gps_ms)
        lon_lat_h = llh_rpy[0:3]
        c_n_v = nu.rpy_to_cnb(*llh_rpy[3:])

        img_path = os.path.realpath(os.path.join(clif_dir, ndf_row.ImageBuffer))
        k_img = kml.addfile(img_path)

        corners_wgs = tf.project_corners(lon_lat_h, c_n_v)

        ground = img_folder.newgroundoverlay(name='cland_%d' % ii)
        ground.icon.href = img_path
        ground.color = 'ffffffff'
        ground.gxlatlonquad.coords = [tuple(c[:2]) for c in corners_wgs]

        ground.timespan.begin = datetime.datetime.fromtimestamp(
            ndf_row.system_time).strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time = ndf_row.system_time + (1/5.0)
        ground.timespan.end = datetime.datetime.fromtimestamp(
            end_time).strftime('%Y-%m-%dT%H:%M:%SZ')

    folder_name = os.path.split(clif_dir)[1]
    kml.save(args.out + '%s.kml' % folder_name)

