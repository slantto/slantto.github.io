#!/usr/bin/env python
# -*- coding: utf-8 -*-import numpy as np

import lcm
import pandas as pd
import cv2
import os
import navfeatdb.utils.nav as ndu
import argparse
import numpy as np
import time
from datasources.lcm.messages.aspn import geodeticposition3d
from datasources.lcm.messages.aspn import positionvelocityattitude
from datasources.lcm.messages.aspn import uncorrespondedopticalcamerafeatures
from datasources.lcm.messages.aspn import opticalcameraimage
import datautils.aspn_lcmlog as dal
import navfeatdb.utils.cvfeat2d as f2d


def aspn_navsln_from_row(m_time, seq, lon_lat_h, rpy, leap_sec=35):
    pva = positionvelocityattitude()
    tai = m_time + leap_sec
    t_sec = np.int(tai)
    t_nsec = np.round((tai - t_sec) * 1E9).astype(np.int)
    pva.header.timestamp_arrival.sec = t_sec
    pva.header.timestamp_arrival.nsec = t_nsec
    pva.header.timestamp_valid.sec = t_sec
    pva.header.timestamp_valid.nsec = t_nsec
    pva.covariance = (1000.0 * np.eye(9)).tolist()
    pva.header.device_id = 'aspn://vikingnav/novatel/1010.0/0'
    pva.position.latitude = lon_lat_h[1] * (np.pi / 180.0)
    pva.position.longitude = lon_lat_h[0] * (np.pi / 180.0)
    pva.position.altitude = lon_lat_h[2]
    pva.attitude = (rpy * np.pi / 180.0).flatten().tolist()
    pva.header.seq_num = seq
    return pva


def aspn_kp_from_row(m_time, seq, df, cv_desc, leap_sec=35, N=5000):
    kp = uncorrespondedopticalcamerafeatures()
    tai = m_time + leap_sec
    t_sec = np.int(tai)
    t_nsec = np.round((tai - t_sec) * 1E9).astype(np.int)
    kp.header.timestamp_arrival.sec = t_sec
    kp.header.timestamp_arrival.nsec = t_nsec
    kp.header.timestamp_valid.sec = t_sec
    kp.header.timestamp_valid.nsec = t_nsec
    kp.descriptor_vector = cv_desc[:N].tolist()
    kp.descriptor_array_len = cv_desc.shape[1]
    kp.header.device_id = 'aspn://ips/frontend/20050/0'
    kp.num_features = len(df[:N])
    kp.header.seq_num = seq
    kp.x = df.pix_x.iloc[:N].tolist()
    kp.y = df.pix_y.iloc[:N].tolist()
    kp.orientation = (df.angle.iloc[:N] * np.pi / 180.0).tolist()
    kp.response = df.response.iloc[:N].tolist()
    kp.size = df['size'].iloc[:N].tolist()
    print(kp.num_features)
    return kp


def aspn_geo_pos_from_row(m_time, seq, lon_lat_h, leap_sec=35):
    pos = geodeticposition3d()
    tai = m_time + leap_sec
    t_sec = np.int(tai)
    t_nsec = np.round((tai - t_sec) * 1E9).astype(np.int)
    pos.header.timestamp_arrival.sec = t_sec
    pos.header.timestamp_arrival.nsec = t_nsec
    pos.header.timestamp_valid.sec = t_sec
    pos.header.timestamp_valid.nsec = t_nsec
    pos.covariance = (1000.0 * np.eye(3)).tolist()
    pos.header.device_id = 'aspn://vikingnav/novatel/1001.3/0'
    pos.position.latitude = lon_lat_h[1] * (np.pi / 180.0)
    pos.position.longitude = lon_lat_h[0] * (np.pi / 180.0)
    pos.position.altitude = lon_lat_h[2]
    pos.header.seq_num = seq
    return pos


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='sim_frontend',
                                     description='Extracts and GeoLocates Features From a UVAN Flight')
    parser.add_argument('-a', '--aspn', help='Path to ASPN_IMGS Flight')
    parser.add_argument('-i', '--imgs', help='Path to aspn log files containing images')
    args = parser.parse_args()

    athens_imgs = pd.read_hdf(args.aspn, 'df')
    athens_imgs.logfile = athens_imgs.logfile.apply(lambda x: os.path.join(args.imgs, os.path.split(x)[1]))

    detector = cv2.BRISK_create()
    descriptor = cv2.BRISK_create()

    ccm = {'aspn://vikingnav/novatel/1010.0/0': positionvelocityattitude,
           'aspn://vikingnav/image/20040/0': opticalcameraimage}

    lc = lcm.LCM()

    seq = 0
    sample_num = 1000
    num_features = 5000

    for ii, row in athens_imgs.iterrows():
        seq += 1

        lon_lat_h = np.array(row[['longitude', 'latitude', 'altitude']])
        att = ndu.rpy_to_cnb(row['roll'], row['pitch'], row['yaw'], units='rad')
        rpy = np.array([row['roll'], row['pitch'], row['yaw']]) * 180.0 / np.pi

        aspn_img = dal.log_mapper(row, ccm)
        img = np.frombuffer(aspn_img.data, np.uint8).reshape(aspn_img.height, aspn_img.width)

        t0 = time.time()
        obs_cv_kp, obs_desc = detector.detectAndCompute(img, mask=None)
        print(obs_desc.shape)
        pix = f2d.keypoint_pixels(obs_cv_kp)
        meta = np.array([(pt.angle, pt.response, pt.size) for pt in obs_cv_kp])
        packed = f2d.numpySIFTScale(obs_cv_kp)
        img_df = pd.DataFrame(np.hstack((pix, meta, packed)),
                              columns=['pix_x', 'pix_y', 'angle',
                                       'response', 'size', 'octave',
                                       'layer', 'scale'])
        img_df = img_df.sort_values(by='response', ascending=False).iloc[:num_features]
        obs_desc = obs_desc[img_df.index]

        t1 = time.time()
        if obs_desc is not None:
            pos = aspn_geo_pos_from_row(row.t_valid_est, seq, lon_lat_h)
            pva = aspn_navsln_from_row(row.t_valid_est, seq, lon_lat_h, rpy)
            akp = aspn_kp_from_row(row.t_valid_est, seq, img_df, obs_desc)
            lc.publish(pos.header.device_id, pos.encode())
            lc.publish(pva.header.device_id, pva.encode())
            lc.publish(akp.header.device_id, akp.encode())
        ttsleep = 2.0 - (t1 - t0)
        ttsleep = min(0.0, abs(ttsleep))
        time.sleep(ttsleep)
        print('%d / %d - %f' % (seq, sample_num, time.time() - t0))
        seq += 1

