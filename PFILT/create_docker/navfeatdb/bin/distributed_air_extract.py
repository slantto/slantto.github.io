#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cv2
import os
from distributed import Executor
import tables as tb
import shutil
import pandas as pd


def extract_kp_from_frame(flight_dict):
    """
    This function extracts Keypoints
    :param flight_dict:
    :param img_num:
    :param o_path:
    :param kp_det_func:
    :param kp_desc_func:
    :param p_meta:
    :return:
    """
    import navfeatdb.db.features as fdb
    import bcolz
    import os
    import numpy as np
    import tables as tb
    from scipy.interpolate import interp1d
    import navfeatdb.utils.nav as nu
    import navfeatdb.projection.localprojection as llproj


    flight_hdf = flight_dict['flight_hdf']
    img_num = flight_dict['img_num']
    o_path = flight_dict['o_path']
    kp_det_func = flight_dict['kp_det_func']
    kp_desc_func = flight_dict['kp_desc_func']
    p_meta = flight_dict['p_meta']

    flight = tb.open_file(flight_hdf, 'r')
    pva = flight.root.nov_span.pva
    pva_times = flight.root.nov_span.pva.cols.t_valid
    img_times = flight.root.camera.image_raw.compressed.metadata.cols.t_valid
    images = flight.root.camera.image_raw.compressed.images

    img0 = img_times[0] - 1.0
    imgN = img_times[-1] + 1.0
    good_pva = pva.read_where('(t_valid >= img0) & (t_valid <= imgN)')
    lon_lat_h = np.array(
        [good_pva[:]['lon'], good_pva[:]['lat'], good_pva[:]['height']]).T

    rpy = np.array([nu.DcmToRpy(dcm.reshape(3, 3)) for dcm in good_pva['c_nav_veh']])
    pva_interp = interp1d(good_pva['t_valid'], np.hstack((lon_lat_h, rpy)).T)

    llh_rpy = pva_interp(img_times[img_num])
    lon_lat_h = llh_rpy[0:3]
    c_n_v = nu.rpy_to_cnb(*llh_rpy[3:])

    proj = llproj.LocalLevelProjector(p_meta['srtm_path'], p_meta['geoid_file'])
    proj.load_cam_and_vehicle_frames(p_meta['uvan_frames'])
    proj.load_camera_cal(p_meta['cam_path'])

    aie = fdb.AirborneImageExtractor(kp_det_func(), kp_desc_func(), proj)

    feat_df, desc = aie.extract_features(images[img_num], lon_lat_h, c_n_v)
    center_wgs = proj.project_center(lon_lat_h, c_n_v).flatten()
    df_path = os.path.join(o_path, 'feat/df/feat_%d.hdf' % img_num)
    desc_path = os.path.join(o_path, 'feat/desc/desc_%d' % img_num)

    if feat_df is None:
        kp_meta = [0, center_wgs[0], center_wgs[1], df_path, desc_path, flight_hdf, img_num]
    else:
        kp_meta = [desc.shape[0], center_wgs[0], center_wgs[1], df_path, desc_path, flight_hdf, img_num]
        feat_df.to_hdf(os.path.join(o_path, df_path), 'feat_df', mode='w', format='table', complib='zlib', complevel=7)
        bcolz.carray(desc, rootdir=os.path.join(o_path, desc_path), mode='w').flush()

    flight.close()
    return kp_meta


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='airborne_feat_extraction',
                                     description='Extracts and GeoLocates Features From a UVAN Flight')
    parser.add_argument('-o', '--output',
                        help='Directory where to create test datasets')
    parser.add_argument('-g', '--geoid', help='Path to Geoid File')
    parser.add_argument('-s', '--srtm', help='Path to SRTM Root')
    parser.add_argument('-m', '--meta', help='Path to UVAN Metadata')
    parser.add_argument('-u', '--uvan', help='Path to UVAN HDF5 Flight')
    parser.add_argument('-d', '--scheduler', help='Node hosting the scheduler')
    parser.add_argument('-a', '--append', default=True)
    args = parser.parse_args()

    flight = tb.open_file(args.uvan, 'r')
    images = flight.root.camera.image_raw.compressed.images
    num_imgs = images.shape[0]
    print(num_imgs)
    flight.close()

    srtm_path = args.srtm
    cam_path = os.path.join(args.meta, 'fc2_cam_model.yaml')
    uvan_frames = os.path.join(args.meta, 'fc2_pod_frames.yaml')
    geoid_file = args.geoid


    # kp_type_dict = {'sift': cv2.xfeatures2d.SIFT_create,
    #                 'surf': cv2.xfeatures2d.SURF_create,
    #                 'brisk': cv2.BRISK_create}

    kp_type_dict = {'brisk': cv2.BRISK_create}

    tf_meta = {'srtm_path': srtm_path,
               'cam_path': cam_path,
               'uvan_frames': uvan_frames,
               'geoid_file': geoid_file}

    executor = Executor(args.scheduler + ':8786')

    for kp_type in kp_type_dict.keys():
        out_path = os.path.join(args.output, kp_type)

        if (not args.append) and (os.path.exists(out_path)):
            shutil.rmtree(out_path)

        if not os.path.exists(out_path):
            os.makedirs(os.path.join(out_path, 'feat/df'))
            os.makedirs(os.path.join(out_path, 'feat/desc'))

        odDicts = [{'flight_hdf': args.uvan,
                    'img_num': ii,
                    'kp_det_func': kp_type_dict[kp_type],
                    'kp_desc_func': kp_type_dict[kp_type],
                    'p_meta': tf_meta,
                    'o_path': out_path} for ii in range(num_imgs)]

        r = executor.map(extract_kp_from_frame, odDicts, pure=False)

        kp_list = executor.gather(r)
        kp_meta = pd.DataFrame(kp_list, columns=['num_feat',
                                                 'center_lon',
                                                 'center_lat',
                                                 'df_path',
                                                 'desc_path',
                                                 'flight',
                                                 'img_num'])

        kp_meta.to_hdf(os.path.join(out_path, 'feat_meta.hdf'),
                       key='feat_meta')

    print('what')

