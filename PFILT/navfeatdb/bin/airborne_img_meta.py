#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime

import cv2
import numpy as np
import simplekml
import tables as tb
import pandas as pd
import os

import navfeatdb.projection.localprojection as llproj
import navfeatdb.utils.nav as nu
from scipy.interpolate import interp1d


if __name__ == '__main__':

    srtm_path = '/Users/venabled/data/srtm/SRTM1/Region_01'
    cam_path = '/Users/venabled/pysrc/pnpnav/data/fc2_cam_model.yaml'
    uvan_frames = '/Users/venabled/pysrc/pnpnav/data/fc2_pod_frames.yaml'
    geoid_file = '/Users/venabled/data/geoid/egm96_15.tiff'

    tf = llproj.LocalLevelProjector(srtm_path, geoid_file)
    tf.load_cam_and_vehicle_frames(uvan_frames)
    tf.load_camera_cal(cam_path)

    out_path = '/Users/venabled/data/uvan/meta'
    in_path = '/Users/venabled/data/uvan/flights_noimg'

    flights = [os.path.join(in_path, x) for x in os.listdir(in_path) if os.path.splitext(x)[1] == '.hdf']
    flights = ['/Users/venabled/data/uvan/flights_noimg/fc2_f2_noimgs.hdf']

    for fname in flights:

        flight = tb.open_file(fname, 'r')
        pva = flight.root.pva
        pva_times = flight.root.pva.cols.t_valid
        img_times = flight.root.metadata.cols.t_valid

        img0 = img_times[0] - 1.0
        imgN = img_times[-1] + 1.0
        good_pva = pva.read_where('(t_valid >= img0) & (t_valid <= imgN)')
        lon_lat_h = np.array(
            [good_pva[:]['lon'], good_pva[:]['lat'], good_pva[:]['height']]).T

        rpy = np.array([nu.DcmToRpy(dcm.reshape(3, 3)) for dcm in good_pva['c_nav_veh']])
        pva_interp = interp1d(good_pva['t_valid'], np.hstack((lon_lat_h, rpy)).T)

        img_range = np.arange(img_times.shape[0])
        img_meta = pd.DataFrame(index=img_range,
                                 columns=['center_lon', 'center_lat', 'center_height',
                                          'bound_x', 'bound_y', 'bound_z', 'gsd'])

        # Dump image
        for ii in img_range:
            print('Img %d / %d' % (ii, img_range[-1]))
            llh_rpy = pva_interp(img_times[ii])
            lon_lat_h = llh_rpy[0:3]
            c_n_v = nu.rpy_to_cnb(*llh_rpy[3:])

            center_wgs = tf.project_center(lon_lat_h, c_n_v).flatten()
            gsd = tf.get_pix_size(lon_lat_h, c_n_v).mean()
            b_tile = tf.get_bounding_tile(lon_lat_h, c_n_v)
            img_meta.loc[ii] = np.hstack((center_wgs, np.array(b_tile), gsd))

        img_meta.to_hdf(os.path.join(out_path, 'meta_' + os.path.split(fname)[1]), key='img_geo_meta', mode='w')
        flight.close()

    # Dump Center Points to KML
    kml = simplekml.Kml()
    doc = kml.newdocument
    track_folder = kml.newfolder(name='UVAN Track')
    track = track_folder.newgxtrack()
    track.altitudemode = simplekml.AltitudeMode.clamptoground
    track.newgxcoord([(row[1]['center_lon'], row[1]['center_lat'], row[1]['center_height']) for row in img_meta.iterrows()])
    kml.save(os.path.join(out_path, 'cam_centerpoint.kml'))


