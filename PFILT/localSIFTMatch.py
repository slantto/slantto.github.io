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
import neogeodb.pytables_db as pdb
from neogeodb import hdf_orthophoto as hdo
import mercantile


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


def air_2_db(t_desc, q_desc, q_kp, db_meta, q_img,
             db_ophoto, normType, thresh=0.7, use_h=True):

    db_idx, q_idx = bf_match(t_desc, q_desc, normType, nn_thresh=thresh)

    kp_q = q_kp[q_idx]
    wgs_db = np.vstack((db_meta['lon'], db_meta['lat'], db_meta['height'])).T[db_idx]

    if  kp_q.shape[0] > 10:
        (match_img, nn_mask) = mutils.draw_2D3DCorrespondences(wgs_db, kp_q,
                                                               q_img, db_ophoto,
                                                               use_h=use_h,
                                                               r_thresh=15.0)
        return nn_mask, match_img

    else:
        return None, None


if __name__ == "__main__":

    flight_path = '/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/'
    flight = tb.open_file('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/fc2_f5.hdf', 'r')
    pva = flight.root.nov_span.pva
    pva_times = flight.root.nov_span.pva.cols.t_valid
    img_times = flight.root.camera.image_raw.compressed.metadata.cols.t_valid
    img_array = flight.root.camera.image_raw.compressed.images

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

    idx1 = 343
    num_feat = 4500
    img1 = img_array[idx1]

    llh_rpy = pva_interp(img_times[idx1])
    lon_lat_h = llh_rpy[0:3]
    c_n_v = nu.rpy_to_cnb(*llh_rpy[3:])

    R_w_c = np.dot(c_n_v, np.dot(tf.C_b_v.T, tf.C_b_cam))
    cam_rpy = nu.DcmToRpy(R_w_c)
    # img1 = skit.rotate(img1, -1.0*(cam_rpy[2] - 90.0))
    # img1 = (255.0 * img1).round().astype(np.uint8)

    corners_wgs = tf.project_corners(lon_lat_h, c_n_v)

    descriptor = cv2.xfeatures2d.SIFT_create()
    detector = cv2.xfeatures2d.SIFT_create()

    kp1, desc1 = f2d.extract_features(img1, detector, descriptor)

    # Sort by response when downsampling
    r1 = np.array([kp.response for kp in kp1])

    kidx1 = np.argsort(r1)[-num_feat:]
    kpts1 = f2d.keypoint_pixels(kp1)[kidx1]
    ddesc1 = desc1[kidx1]

    # Load Database Info
    tbdb = tb.open_file('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/pytables_db.hdf', 'r')
    db = tbdb.root.sift_db.sift_features_sorted
    bbox = mercantile.LngLatBbox(corners_wgs[:, 0].min(),
                                 corners_wgs[:, 1].min(),
                                 corners_wgs[:, 0].max(),
                                 corners_wgs[:, 1].max())
    tiles = [t for t in mercantile.tiles(*bbox, zooms=[15])]
    print(tiles)

    tuids = [pdb.elegant_pair_xy(t.x, t.y) for t in tiles]
    rows = [db.read_where('pair_id == uid') for uid in tuids]

    # Downsample by response
    rows_per_tile = 10000
    rows = [np.sort(f, order='response')[-rows_per_tile:] for f in rows]

    tile1_desc = rows[0]['descriptor']
    tile2_desc = rows[1]['descriptor']

    ophoto = hdo.HDFOrthophoto('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/dugway_ciortho.hdf')#, '/ophoto/gray')
    t_photo = ophoto.get_img_from_bounds(bbox)
    ddesc1 = ddesc1.astype(np.float32)
    tile1_desc = tile1_desc.astype(np.float32)
    mmask, mimg = air_2_db(tile1_desc, ddesc1, kpts1, rows[0], img1, t_photo, cv2.NORM_L2, thresh=0.7)


    plt.imshow(mimg)

    # Try those precomputed ones
    flight_path = '/media/sean/D2F2E7B2F2E798CD/Users/student/AIEoutput2/feat/'
    feat_meta = pd.read_hdf(os.path.join(flight_path, 'feat_meta.hdf'))

    ft = pd.read_hdf(os.path.join(flight_path, feat_meta.loc[idx1].df_path))
    obs_kp = ft[['pix_x', 'pix_y']].as_matrix()[:num_feat]
    obs_desc = bcolz.open(os.path.join(flight_path, feat_meta.loc[idx1].desc_path), 'r')[:num_feat, :]

    tile1_desc = tile1_desc.astype(np.float32)
    obs_desc = obs_desc.astype(np.float32)

    mmask2, mimg2 = air_2_db(tile1_desc, obs_desc, obs_kp, rows[0],
                           img1, t_photo, cv2.NORM_L2, thresh=0.7)

    plt.figure()
    plt.imshow(mimg2)
    plt.show()