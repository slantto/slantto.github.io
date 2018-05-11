#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bcolz
import navfeatdb.utils.matching as mutils
import pandas as pd
import numpy as np
import os
import tables as tb
import cv2
import neogeodb.pytables_db as pdb
from neogeodb import hdf_orthophoto as hdo
import mercantile
import matplotlib.pyplot as plt


def bfmatch(target_desc, query_desc, nn_thresh=0.7):

    bfm = cv2.BFMatcher()

    m1 = bfm.knnMatch(query_desc, target_desc, k=2)
    m2 = np.array([(m0[0].trainIdx,
                    m0[0].distance / m0[1].distance) for m0 in m1])
    kp2_idx = np.where(m2[:, 1] < nn_thresh)[0]
    kp1_idx = m2[kp2_idx, 0].astype(np.int)
    return kp1_idx, kp2_idx


def air_2_air_case(t_desc, q_desc, t_meta, q_meta, t_img, q_img, v_centers):
    t_idx, q_idx = bfmatch(t_desc, q_desc)

    kp_t = np.array(t_meta[['pix_x', 'pix_y']])[t_idx]
    kp_q = np.array(q_meta[['pix_x', 'pix_y']])[q_idx]

    nn_H, nn_mask = cv2.findHomography(kp_t, kp_q,
                                       method=cv2.cv.CV_RANSAC,
                                       ransacReprojThreshold=15.0)

    feat_match_img = mutils.draw_match(t_img, q_img, kp_t, kp_q, status=nn_mask)

    # Try Matching to the Vocabs
    c_norm = (v_centers.T / np.linalg.norm(v_centers, axis=1)).T
    v1 = cv2.BFMatcher().match(t_desc.astype(np.float32), c_norm.astype(np.float32))
    t_vocab = np.array([v.trainIdx for v in v1])

    v2 = cv2.BFMatcher().match(q_desc.astype(np.float32), c_norm.astype(np.float32))
    q_vocab = np.array([v.trainIdx for v in v2])

    # See how many of the good feature matches were consistent with vocab
    num_matches = t_idx.shape[0]
    good_match_vocab = q_vocab[q_idx] == t_vocab[t_idx]
    vocab_percent = np.where(good_match_vocab)[0].shape[0] / float(num_matches)

    common_vocab = np.intersect1d(t_vocab, q_vocab)
    vocab_match = np.array([(np.where(t_vocab == xs)[0][0], np.where(q_vocab == xs)[0][0]) for xs in common_vocab])

    num_vocab_matches = vocab_match.shape[0]

    vm_t = np.array(t_meta[['pix_x', 'pix_y']])[vocab_match[:, 0]]
    vm_q = np.array(q_meta[['pix_x', 'pix_y']])[vocab_match[:, 1]]

    vH, vmask = cv2.findHomography(vm_t, vm_q,
                                   method=cv2.cv.CV_RANSAC,
                                   ransacReprojThreshold=15.0)
    vocab_match_img = mutils.draw_match(img1, img2, vm_t, vm_q, status=vmask)

    return (num_matches, nn_mask,
            num_vocab_matches, vmask,
            feat_match_img, vocab_match_img,
            vocab_percent)


def air_2_db_case(db_desc, q_desc, db_meta, q_meta, db_ophoto, q_img,
                  v_centers, use_h=True):

    db_idx, q_idx = bfmatch(db_desc, q_desc)

    kp_q = np.array(q_meta[['pix_x', 'pix_y']])[q_idx]
    wgs_db = np.vstack((db_meta['lon'], db_meta['lat'], db_meta['height'])).T[db_idx]

    (nn_match_img, nn_mask) = mutils.draw_2D3DCorrespondences(wgs_db, kp_q,
                                                              q_img, db_ophoto,
                                                              use_h=use_h,
                                                              r_thresh=15.0)

    # Now Try Again With the Vocab
    c_norm = (v_centers.T / np.linalg.norm(v_centers, axis=1)).T
    t1 = cv2.BFMatcher().match(db_desc.astype(np.float32), c_norm.astype(np.float32))
    t_vocab = np.array([v.trainIdx for v in t1])

    v2 = cv2.BFMatcher().match(q_desc.astype(np.float32), c_norm.astype(np.float32))
    q_vocab = np.array([v.trainIdx for v in v2])

    num_matches = db_idx.shape[0]
    good_match_vocab = q_vocab[q_idx] == t_vocab[db_idx]
    vocab_percent = np.where(good_match_vocab)[0].shape[0] / float(num_matches)

    common_vocab = np.intersect1d(t_vocab, q_vocab)
    vocab_match = np.array([(np.where(t_vocab == xs)[0][0], np.where(q_vocab == xs)[0][0]) for xs in common_vocab])

    vm_q = np.array(q_meta[['pix_x', 'pix_y']])[vocab_match[:, 1]]
    vm_wgs = np.vstack((db_meta['lon'], db_meta['lat'], db_meta['height'])).T[vocab_match[:, 0]]

    num_vocab_matches = vocab_match.shape[0]

    (v_match_img, v_mask) = mutils.draw_2D3DCorrespondences(vm_wgs, vm_q,
                                                            q_img, db_ophoto,
                                                            use_h=use_h,
                                                            r_thresh=15.0)
    return (num_matches, nn_mask,
            num_vocab_matches, v_mask,
            nn_match_img, v_match_img,
            vocab_percent)



if __name__ == "__main__":

    flight_path = '/Users/venabled/data/uvan/fc2/f5'
    flight = tb.open_file('/Users/venabled/data/uvan/fc2_f5.hdf', 'r')
    img_array = flight.root.camera.image_raw.compressed.images
    feat_meta = pd.read_hdf(os.path.join(flight_path, 'feat_meta.hdf'))

    idx1 = 342
    idx2 = 343

    img1_meta = pd.read_hdf(os.path.join(flight_path, feat_meta['df_path'].loc[idx1]))
    img2_meta = pd.read_hdf(os.path.join(flight_path, feat_meta['df_path'].loc[idx2]))

    # Reindex
    # img1_meta.index = np.arange(img1_meta.shape[0])
    # img2_meta.index = np.arange(img2_meta.shape[0])

    num_feat=4500
    img1_desc = bcolz.open(os.path.join(flight_path, feat_meta['desc_path'].loc[idx1]))[:num_feat, :].astype(np.float32)
    img2_desc = bcolz.open(os.path.join(flight_path, feat_meta['desc_path'].loc[idx2]))[:num_feat, :].astype(np.float32)

    img1 = img_array[idx1]
    img2 = img_array[idx2]

    img1_kp = img1_meta.iloc[:num_feat]
    img2_kp = img2_meta.iloc[:num_feat]

    # Load Database Info
    tbdb = tb.open_file('/Users/venabled/data/neogeo/pytables_db.hdf', 'r')
    db = tbdb.root.sift_db.sift_features_sorted
    bbox = mercantile.LngLatBbox(min(img1_kp.lon.min(), img2_kp.lon.min()),
                                 min(img1_kp.lat.min(), img2_kp.lat.min()),
                                 max(img1_kp.lon.max(), img2_kp.lon.max()),
                                 max(img1_kp.lat.max(), img2_kp.lat.max()))
    tiles = [t for t in mercantile.tiles(*bbox, zooms=[15])]
    tuids = [pdb.elegant_pair_xy(t.x, t.y) for t in tiles]
    rows = [db.read_where('pair_id == uid') for uid in tuids]

    # Downsample by response
    rows_per_tile = 10000
    rows = [np.sort(f, order='response')[-rows_per_tile:] for f in rows]

    tile1_desc = rows[0]['descriptor'].astype(np.float32)
    tile2_desc = rows[1]['descriptor'].astype(np.float32)

    # Can we make this match more features if we pull features from a different
    # ocatve range? EG the average GSD from our
    img1_oct_idx = img1_meta.loc[img1_meta['octave'] > 0].index[:num_feat]
    img2_oct_idx = img2_meta.loc[img2_meta['octave'] > 0].index[:num_feat]


    img1_oct_desc = bcolz.open(os.path.join(flight_path, feat_meta['desc_path'].loc[idx1]))[img1_oct_idx, :].astype(np.float32)
    img2_oct_desc = bcolz.open(os.path.join(flight_path, feat_meta['desc_path'].loc[idx2]))[img2_oct_idx, :].astype(np.float32)

    img1_oct_kp = img1_meta.iloc[img1_oct_idx]
    img2_oct_kp = img2_meta.iloc[img2_oct_idx]


    ophoto = hdo.HDFOrthophoto('/Users/venabled/data/dugway/dugway_ciortho.hdf', '/ophoto/image')
    t_photo = ophoto.get_img_from_bounds(bbox)

    flight.close()

    # Load Vision Vocab for all cases
    vocab_cases = ['100_vocab', '500_vocab', '1000_vocab', '10k_vocab']
    for v_case in vocab_cases:

        out_path = os.path.join("/Users/venabled/doc/presentations", v_case)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        cpath = os.path.join('/Users/venabled/data/uvan/fc2/f2', v_case)
        centers = np.array(bcolz.open(cpath))

        # Generate an Air 2 Air Matching Case
        a2a = air_2_air_case(img1_desc, img2_desc, img1_kp, img2_kp, img1, img2, centers)
        plt.figure()
        plt.imshow(a2a[4])
        plt.title('SIFT Feature Matching Between FC2-F5 Imgs %d and %d' % (idx1, idx2))
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'case1-feat.png'))

        plt.figure()
        plt.imshow(a2a[5])
        plt.title('Vision Vocab Matching Between FC2-F5 Imgs %d and %d' % (idx1, idx2))
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'case1-vocab.png'))


        # Do the Air To Database Matching Case
        a2db = air_2_db_case(tile1_desc, img2_desc, rows[0], img2_kp, t_photo, img2, centers, use_h=True)
        plt.figure()
        plt.imshow(a2db[4])
        plt.title('SIFT Feature Matching Between FC2-F5 Imgs %d Satellite Image' % idx2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'case2-feat.png'))

        plt.figure()
        plt.imshow(a2db[5])
        plt.title('Vision Vocab Matching Between FC2-F5 Imgs %d and Satellite Image' % idx2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'case2-vocab.png'))

        a2a_oct = air_2_air_case(img1_oct_desc, img2_oct_desc,
                                 img1_oct_kp, img2_oct_kp,
                                 img1, img2, centers)
        plt.figure()
        plt.imshow(a2a_oct[4])
        plt.title('OF - SIFT Feature Matching Between FC2-F5 Imgs %d and %d' % (idx1, idx2))
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'case3-feat.png'))

        plt.figure()
        plt.imshow(a2a_oct[5])
        plt.title('OF - Vision Vocab Matching Between FC2-F5 Imgs %d and %d' % (idx1, idx2))
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'case3-vocab.png'))


        # Do the Air To Database Matching Case
        a2db_oct = air_2_db_case(tile1_desc, img2_oct_desc,
                                 rows[0], img2_oct_kp,
                                 t_photo, img2, centers, use_h=True)
        plt.figure()
        plt.imshow(a2db_oct[4])
        plt.title('OF - SIFT Feature Matching Between FC2-F5 Imgs %d Satellite Image' % idx2)
        plt.savefig(os.path.join(out_path, 'case4-feat.png'))

        plt.figure()
        plt.imshow(a2db_oct[5])
        plt.title('OF - Vision Vocab Matching Between FC2-F5 Imgs %d and Satellite Image' % idx2)
        plt.savefig(os.path.join(out_path, 'case4-vocab.png'))

        olines = list()
        olines.append("Case 1:")
        olines.append("Num Feat Matches %d" % a2a[0])
        olines.append("Num Feat Inliers %d" % a2a[1].sum())
        olines.append("Percent Inliers of Matches %f" % (a2a[1].sum()/float(a2a[0])))
        olines.append("Number of Vocab Matches %d" % a2a[2])
        olines.append("Number of Vocab Inliers %d" % a2a[3].sum())
        olines.append("Percent Inliers of Vocab Matches %f" % (a2a[3].sum()/float(a2a[2])))
        olines.append("Percent of Feature Matches quantizing to same Vocab Word: %f" % a2a[6])
        olines.append("\n")
        olines.append("Case 2:")
        olines.append("Num Feat Matches %d" % a2db[0])
        olines.append("Num Feat Inliers %d" % a2db[1].sum())
        olines.append("Percent Inliers of Matches %f" % (a2db[1].sum()/float(a2db[0])))
        olines.append("Number of Vocab Matches %d" % a2db[2])
        olines.append("Number of Vocab Inliers %d" % a2db[3].sum())
        olines.append("Percent Inliers of Vocab Matches %f" % (a2db[3].sum()/float(a2db[2])))
        olines.append("Percent of Feature Matches quantizing to same Vocab Word: %f" % a2db[6])
        olines.append("\n")
        olines.append("Case 3:")
        olines.append("Num Feat Matches %d" % a2a_oct[0])
        olines.append("Num Feat Inliers %d" % a2a_oct[1].sum())
        olines.append("Percent Inliers of Matches %f" % (a2a_oct[1].sum()/float(a2a_oct[0])))
        olines.append("Number of Vocab Matches %d" % a2a_oct[2])
        olines.append("Number of Vocab Inliers %d" % a2a_oct[3].sum())
        olines.append("Percent Inliers of Vocab Matches %f" % (a2a_oct[3].sum()/float(a2a_oct[2])))
        olines.append("Percent of Feature Matches quantizing to same Vocab Word: %f" % a2a_oct[6])
        olines.append("\n")
        olines.append("Case 4:")
        olines.append("Num Feat Matches %d" % a2db_oct[0])
        olines.append("Num Feat Inliers %d" % a2db_oct[1].sum())
        olines.append("Percent Inliers of Matches %f" % (a2db_oct[1].sum()/float(a2db_oct[0])))
        olines.append("Number of Vocab Matches %d" % a2db_oct[2])
        olines.append("Number of Vocab Inliers %d" % a2db_oct[3].sum())
        olines.append("Percent Inliers of Vocab Matches %f" % (a2db_oct[3].sum()/float(a2db_oct[2])))
        olines.append("Percent of Feature Matches quantizing to same Vocab Word: %f" % a2db_oct[6])
        with open(os.path.join(out_path, 'stats.txt'), 'w') as ofile:
            ofile.writelines("%s\n" % it for it in olines)
