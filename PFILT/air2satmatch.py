import bcolz
import navfeatdb.utils.matching as mutils
import pandas as pd
import numpy as np
import os
import tables as tb
import cv2
import navfeatdb.projection.localprojection as llproj
import navfeatdb.utils.nav as nu
from scipy.interpolate import interp1d
import skimage.transform as skit
import matplotlib.pyplot as plt
from navfeatdb.utils import cvfeat2d as f2d
import yaml
import neogeodb.hdf_orthophoto as hdfortho

detectors = ['sdetectors_default_detector_Dense.yaml',
             'sdetectors_default_detector_FAST.yaml',
             'sdetectors_default_detector_BRISK.yaml',
             'sdetectors_default_detector_GFTT.yaml',
             'sdetectors_default_detector_HARRIS.yaml',
             'sdetectors_default_detector_MSER.yaml',
             'sdetectors_default_detector_ORB.yaml',
             'default_detector_SIFT.yaml',
             'sdetectors_default_detector_STAR.yaml',
             'sdetectors_default_detector_SURF.yaml',
             'sdetectors_default_detector_SimpleBlob.yaml']

binary_descriptors = ['sdesc_default_descriptor_BRIEF.yaml',
                      'sdesc_default_descriptor_BRISK.yaml',
                      'sdesc_default_descriptor_FREAK.yaml',
                      'sdesc_default_descriptor_ORB.yaml']

hog_descriptors = ['default_descriptor_SIFT.yaml']#,
                   #'sdesc_default_descriptor_SURF.yaml']

def bf_match(target_desc, query_desc, normType, nn_thresh=0.7):

    bfm = cv2.BFMatcher(normType=normType)

    m1 = bfm.knnMatch(query_desc, target_desc, k=2)
    m2 = np.array([(m0[0].trainIdx,
                    m0[0].distance / m0[1].distance) if m0[1].distance != 0.0 else
                   (m0[0].trainIdx,
                    1.0) for m0 in m1])
    kp2_idx = np.where(m2[:, 1] < nn_thresh)[0]
    kp1_idx = m2[kp2_idx, 0].astype(np.int)
    return kp1_idx, kp2_idx


def air_2_air_case(t_desc, q_desc, t_kp, q_kp, t_img, q_img, normType, thresh=0.7):
    t_idx, q_idx = bf_match(t_desc, q_desc, normType, nn_thresh=thresh)

    kp_t = t_kp[t_idx]
    kp_q = q_kp[q_idx]

    if kp_t.shape[0] > 10 and kp_q.shape[0] > 10:

        nn_H, nn_mask = cv2.findHomography(kp_t, kp_q,
                                           method=cv2.RANSAC,
                                           ransacReprojThreshold=25.0)

        feat_match_img = mutils.draw_match(t_img, q_img, kp_t, kp_q, status=nn_mask)
        return (nn_mask, feat_match_img)

    else:
        return (None, None)

if __name__ == "__main__":

   # flight_path = '/Users/venabled/data/uvan/fc2/f5'
    flight = tb.open_file('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/fc2_f5.hdf', 'r')
    pva = flight.root.nov_span.pva
    pva_times = flight.root.nov_span.pva.cols.t_valid
    img_times = flight.root.camera.image_raw.compressed.metadata.cols.t_valid
    img_array = flight.root.camera.image_raw.compressed.images

    # Orthophoto Metadata
    ophoto = hdfortho.HDFOrthophoto('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/dugway_ciortho.hdf')
                                    #'/ophoto/image')


    # Find the Lat Lon of the center point of the image
    img0 = img_times[0] - 1.0
    imgN = img_times[-1] + 1.0
    good_pva = pva.read_where('(t_valid >= img0) & (t_valid <= imgN)')
    lon_lat_h = np.array(
        [good_pva[:]['lon'], good_pva[:]['lat'], good_pva[:]['height']]).T

    rpy = np.array([nu.DcmToRpy(dcm.reshape(3, 3)) for dcm in good_pva['c_nav_veh']])

    pva_interp = interp1d(good_pva['t_valid'], np.hstack((lon_lat_h, rpy)).T)

    srtm_path = '/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/srtm'
    cam_path = '/home/sean/PFVisNavLibBranches/navfeatdb/data/fc2_cam_model.yaml'
    uvan_frames = '/home/sean/PFVisNavLibBranches/navfeatdb/data/fc2_pod_frames.yaml'
    geoid_file = '/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/egm96-15.tif'

    tf = llproj.LocalLevelProjector(srtm_path, geoid_file)
    tf.load_cam_and_vehicle_frames(uvan_frames)
    tf.load_camera_cal(cam_path)

    idx1 = 10
    img1 = img_array[idx1]
    llh_rpy = pva_interp(img_times[idx1])
    lon_lat_h = llh_rpy[0:3]
    c_n_v = nu.rpy_to_cnb(*llh_rpy[3:])

    R_w_c = np.dot(c_n_v, np.dot(tf.C_b_v.T, tf.C_b_cam))
    cam_rpy = nu.DcmToRpy(R_w_c)
    img1 = skit.rotate(img1, -1.0*(cam_rpy[2] - 90.0))
    img1 = (255.0 * img1).round().astype(np.uint8)

    img2 = img_array[idx1 + 1]
    img2 = skit.rotate(img2, -1.0*(cam_rpy[2] - 90.0))
    img2 = (255.0 * img2).round().astype(np.uint8)




    n_feat = 500

    dpath = '/home/sean/PFVisNavLibBranches/navfeatdb/data/'
    opath = '/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/'

    descriptor = cv2.xfeatures2d.SIFT_create()
    detector = cv2.xfeatures2d.SIFT_create()

    kp1 = detector.detect(img1)
    kp2 = detector.detect(img2)

    # kp1, desc1 = f2d.extract_features(img1, detector, descriptor)
    # kp2, desc2 = f2d.extract_features(img2, detector, descriptor)

    # Sort by response when downsampling
    r1 = np.array([kp.response for kp in kp1])
    r2 = np.array([kp.response for kp in kp2])
    if r1.max() != 0.0 and r2.max() != 0.0:

        gidx1 = np.argsort(r1)[-n_feat:]
        gidx2 = np.argsort(r2)[-n_feat:]

        fkp1 = [kp1[ii] for ii in gidx1]
        fkp2 = [kp2[ii] for ii in gidx2]

        fkp1, desc1 = descriptor.compute(img1, fkp1)
        fkp2, desc2 = descriptor.compute(img2, fkp2)

        kpts1 = f2d.keypoint_pixels(fkp1)
        kpts2 = f2d.keypoint_pixels(fkp2)

        masks, match_img = air_2_air_case(desc1, desc2,
                                      kpts1, kpts2,
                                      img1, img2,
                                      normType=cv2.NORM_L2,
                                      thresh=0.8)

        if masks is not None:
            plt.figure()
            plt.imshow(match_img)
            plt.title('SIFT descriptor&detector - %d RANSAC Matches' % masks.sum())
            plt.savefig(os.path.join(opath, 'siftair2satmatch'))
            plt.close()

    # for desc_file in hog_descriptors:
    #     desc_yaml = yaml.load(file(os.path.join(dpath, desc_file), 'r'))
    #     descriptor = f2d.create_feature_descriptor_from_dict(desc_yaml)
    #     for det_file in detectors:
    #         det_yaml = yaml.load(file(os.path.join(dpath, det_file), 'r'))
    #         detector = f2d.create_feature_detector_from_dict(det_yaml)

            # kp1 = detector.detect(img1)
            # kp2 = detector.detect(img2)
            #
            # # kp1, desc1 = f2d.extract_features(img1, detector, descriptor)
            # # kp2, desc2 = f2d.extract_features(img2, detector, descriptor)
            #
            # # Sort by response when downsampling
            # r1 = np.array([kp.response for kp in kp1])
            # r2 = np.array([kp.response for kp in kp2])
            # if r1.max() != 0.0 and r2.max() != 0.0:
            #
            #     gidx1 = np.argsort(r1)[-n_feat:]
            #     gidx2 = np.argsort(r2)[-n_feat:]
            #
            #     fkp1 = [kp1[ii] for ii in gidx1]
            #     fkp2 = [kp2[ii] for ii in gidx2]
            #
            #     fkp1, desc1 = descriptor.compute(img1, fkp1)
            #     fkp2, desc2 = descriptor.compute(img2, fkp2)
            #
            #     kpts1 = f2d.keypoint_pixels(fkp1)
            #     kpts2 = f2d.keypoint_pixels(fkp2)
            #
            #     masks, match_img = air_2_air_case(desc1, desc2,
            #                                       kpts1, kpts2,
            #                                       img1, img2,
            #                                       normType=cv2.NORM_L2,
            #                                       thresh=0.8)
            #
            #     if masks is not None:
            #
            #         plt.figure()
            #         plt.imshow(match_img)
            #         plt.title('%s Detector - %s Descriptor - %d RANSAC Matches' % (det_yaml['detector']['name'],
            #                                                                        desc_yaml['descriptor']['name'],
            #                                                                        masks.sum()))
            #         plt.savefig(os.path.join(opath, '%s_desc_%s_det.png' % (desc_yaml['descriptor']['name'],
            #                                                                 det_yaml['detector']['name'])))
            #         plt.close()

    # for desc_file in binary_descriptors:
    #     desc_yaml = yaml.load(file(os.path.join(dpath, desc_file), 'r'))
    #     descriptor = f2d.create_feature_descriptor_from_dict(desc_yaml)
    #     for det_file in detectors:
    #         det_yaml = yaml.load(file(os.path.join(dpath, det_file), 'r'))
    #         detector = f2d.create_feature_detector_from_dict(det_yaml)
    #         try:
    #             kp1, desc1 = f2d.extract_features(img1, detector, descriptor)
    #             kp2, desc2 = f2d.extract_features(img2, detector, descriptor)
    #
    #             # Sort by response when downsampling
    #             r1 = np.array([kp.response for kp in kp1])
    #             r2 = np.array([kp.response for kp in kp2])
    #             if r1.max() != 0.0 and r2.max() != 0.0:
    #
    #                 idx1 = np.argsort(r1)[-n_feat:]
    #                 idx2 = np.argsort(r2)[-n_feat:]
    #
    #                 kpts1 = f2d.keypoint_pixels(kp1)[idx1]
    #                 kpts2 = f2d.keypoint_pixels(kp2)[idx2]
    #
    #                 masks, match_img = air_2_air_case(desc1[idx1], desc2[idx2],
    #                                                   kpts1, kpts2,
    #                                                   img1, img2,
    #                                                   normType=cv2.NORM_HAMMING,
    #                                                   thresh=0.8)
    #
    #                 if masks is not None:
    #                     plt.figure()
    #                     plt.imshow(match_img)
    #                     plt.title('%s Detector - %s Descriptor - %d RANSAC Matches' % (det_yaml['detector']['name'],
    #                                                                                    desc_yaml['descriptor']['name'],
    #                                                                                    masks.sum()))
    #
    #                     plt.savefig(os.path.join(opath, '%s_desc_%s_det.png' % (desc_yaml['descriptor']['name'],
    #                                                                             det_yaml['detector']['name'])))
    #
    #                     plt.close()
    #         except:
    #             print("Ya blew it")

