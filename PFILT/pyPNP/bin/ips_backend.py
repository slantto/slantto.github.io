#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script implements the IPS Fine Tracking Interface, using the pyPnP
algorithm. This script uses LCM as a transport.
"""

import argparse
import time
import mercantile
from datasources.lcm.messages.aspn import geodeticposition3d
from datasources.lcm.messages.aspn import positionvelocityattitude
from datasources.lcm.messages.aspn import uncorrespondedopticalcamerafeatures
import lcm
import pnpnav.noattitude as ptude
import navfeatdb.projection.localprojection as llproj
import cv2
import navpy

import pnpnav.matching as pmatch
import pnpnav.utils as pnputils
import numpy as np

import datetime
import logging


r2d = 180.0 / np.pi


def aspn_geo_pos_from_row(m_time, seq, lon_lat_h):
    pos = geodeticposition3d()
    tai = m_time
    t_sec = np.int(tai)
    t_nsec = np.round((tai - t_sec) * 1E9).astype(np.int)
    pos.header.timestamp_arrival.sec = t_sec
    pos.header.timestamp_arrival.nsec = t_nsec
    pos.header.timestamp_valid.sec = t_sec
    pos.header.timestamp_valid.nsec = t_nsec
    pos.covariance = (1000.0 * np.eye(3)).tolist()
    pos.header.device_id = 'aspn://ips/tracking/10013/0'
    pos.position.latitude = lon_lat_h[1] * (np.pi / 180.0)
    pos.position.longitude = lon_lat_h[0] * (np.pi / 180.0)
    pos.position.altitude = lon_lat_h[2]
    pos.header.seq_num = seq
    return pos


class PnPRunner():
    def __init__(self, pnp, lc, wait_time=0.9):
        self.seq = 0
        self.pnp = pnp
        self.rpy = None
        self.att = None
        self.llh = None
        self.lch = lc
        self.wait_time = wait_time
        self.current_time = time.time()

    def pos_callback(self, channel, lcm_pos):
        pos = geodeticposition3d.decode(lcm_pos)
        tile = mercantile.tile(pos.position.longitude * r2d, pos.position.latitude * r2d, 15)
        self.pnp.set_db_location_from_tiles([tile], N=5000)

    def pva_callback(self, channel, lcm_pva):
        print("PVA Callback")
        pva = positionvelocityattitude().decode(lcm_pva)
        rpy = np.array(pva.attitude).reshape(3) * r2d
        self.att = pnputils.rpy_to_cnb(*rpy)
        self.llh = np.array([pva.position.longitude * r2d,
                             pva.position.latitude * r2d, 
                             pva.position.altitude])

    def keypoint_callback(self, channel, lcm_kp):
        kp = uncorrespondedopticalcamerafeatures().decode(lcm_kp)
        logger.info("Got KP with: %d Features" % kp.num_features)
        if (self.att is not None) and ((time.time() - self.current_time) > self.wait_time):
            self.current_time = time.time()
            obs_kp = np.vstack((kp.x, kp.y)).T
            obs_desc = np.array([np.frombuffer(dx, dtype=np.uint8) for dx in kp.descriptor_vector])
            pnpout = self.pnp.do_pnp(obs_kp,
                                     obs_desc,
                                     self.att,
                                     return_matches=True,
                                     return_cv=True)
            twgs, cov_n, niter, matches, status, cv_wgs = pnpout
            if status.shape[0] > 0 and twgs.shape[0] > 0:
                lon_lat_h = self.llh
                cv_error = navpy.lla2ned(cv_wgs[0], cv_wgs[1], cv_wgs[2],
                                         lon_lat_h[1], lon_lat_h[0],
                                         lon_lat_h[2])
                logger.info("OCV Error: %s :: %f " % (repr(cv_error), np.linalg.norm(cv_error)))

                ned_error = navpy.lla2ned(twgs[0], twgs[1], twgs[2],
                                          lon_lat_h[1], lon_lat_h[0],
                                          lon_lat_h[2])
                logger.info("NED Error: %s :: %f " % (repr(ned_error), np.linalg.norm(ned_error)))

                tstamp = kp.header.timestamp_valid
                kp_time = tstamp.sec + (tstamp.nsec / 1E9)

                aspn_pos = aspn_geo_pos_from_row(kp_time,
                                                 self.seq,
                                                 twgs)
                self.lch.publish(aspn_pos.header.device_id, aspn_pos.encode())
                self.seq += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='ips_backend',
                                     description='Uses PnP to compute platform position')
    parser.add_argument('-d', '--db', help='Path to Database HDF5 File', required=True)
    parser.add_argument('-t', '--terrain', help='Path to Database HDF5 File', required=True)
    parser.add_argument('-c', '--cam', help='Path to Database HDF5 File', required=True)
    parser.add_argument('-f', '--frames', help='Path to Database HDF5 File', required=True)
    parser.add_argument('-g', '--geoid', help='Path to Database HDF5 File', required=True)

    args = parser.parse_args()

    lcm_h = lcm.LCM()

    dt = datetime.datetime.now().isoformat()
    logger = pnputils.get_log(filename='ips_backend-%s.log' % dt)


    # Setup PnP
    srtm_path = args.terrain
    cam_path = args.cam
    uvan_frames = args.frames
    geoid_file = args.geoid

    tf = llproj.LocalLevelProjector(srtm_path, geoid_file)
    tf.load_cam_and_vehicle_frames(uvan_frames)
    tf.load_camera_cal(cam_path)
    # Boot up that PNP class
    brisk_matcher = pmatch.BFMatcher(nn_ratio=0.8,
                                     norm_type=cv2.HAMMING_NORM_TYPE)

    brisk_matcher.load_db(args.db)
    pnp2 = ptude.NoAttitudePnP(matcher=brisk_matcher)
    pnp2.load_camera_parameters(cam_path)
    pnp2.use_homography_constraint(5.0)
    pnp2.load_frames(uvan_frames)

    pnpr = PnPRunner(pnp2, lcm_h)
    lcm_h.subscribe('aspn://vikingnav/novatel/1001.3/0', pnpr.pos_callback)
    lcm_h.subscribe('aspn://vikingnav/novatel/1010.0/0', pnpr.pva_callback)
    lcm_h.subscribe('aspn://ips/frontend/20050/0', pnpr.keypoint_callback)

    try:
        timeout = 0.10  # amount of time to wait, in seconds
        while True:
            lcm_h.handle()

    except KeyboardInterrupt:
        pass