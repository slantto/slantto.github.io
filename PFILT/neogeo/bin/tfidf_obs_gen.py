
import neogeo.extent as neoextent
from collections import namedtuple
import numpy as np
import tables as tb
import navfeatdb.projection.localprojection as llproj
import bcolz
import pandas as pd
import os
import cv2

from scipy.spatial.distance import cosine as cdist


def tf_idf_match(img_desc, centers_norm, sat_v, sat_idf, sat_mask):
    """
    Match img_desc against vision vocab, return N-length cosine distance func
    :param img_desc: num_feat x 128 length descriptor vector
    :param centers_norm: k x 128 length normalized cluster centers
    :param sat_v: N x k satellite derived tf-idf vectors
    :param sat_idf: k, length idf term derived from sat_hist
    :param sat_mask: Mask of words to use
    :return: (N,) shaped double array of cosine distance to each db location
    """
    v1 = cv2.BFMatcher().match(img_desc.astype(np.float32), centers_norm.astype(np.float32))
    img_vocab = np.array([v.trainIdx for v in v1])
    img_vid, img_vidcount = np.unique(img_vocab, return_counts=True)
    img_hist = np.zeros(sat_mask.shape[0])
    img_hist[img_vid] = img_vidcount

    # Get a TF-IDF Vector for each case
    s_n_id = img_hist[sat_mask]
    s_tf_d = s_n_id / float(np.where(img_hist[sat_mask] > 0)[0].shape[0])
    img1_s_v = s_tf_d * s_idf

    s_dist = np.apply_along_axis(lambda x: cdist(x, img1_s_v),
                                 axis=1, arr=sat_v)

    return s_dist


if __name__ == '__main__':

    # Grab the features from the flight
    flight_path = '/Users/venabled/data/uvan/fc2/f5'
    feat_meta = pd.read_hdf(os.path.join(flight_path, 'feat_meta.hdf'))

    # Get the Histograms and TileIDs from the Database
    hpath = '/Users/venabled/data/uvan/vocab/10k_vocab_pydb_sat_db_hist'
    sdb_hist = np.array(bcolz.open(hpath, 'r')).astype(np.uint16)
    hist_uid = bcolz.open('/Users/venabled/data/uvan/vocab/10k_vocabpydb_pair_id', 'r')

    num_feat = 10000
    N = sdb_hist.shape[0]
    k = sdb_hist.shape[1]
    sdb_sum = sdb_hist.sum(0)
    s_mask = sdb_sum > 0

    s_bin = np.zeros_like(sdb_hist)
    s_bin[sdb_hist > 0] = 1

    s_idf = np.log( float(N) / s_bin.sum(0)[s_mask])
    s_v = np.zeros((N, sdb_hist[:, s_mask].shape[1]))

    for d in np.arange(N):
        s_n_id = sdb_hist[d, s_mask]
        s_tf_d = s_n_id / float(np.where(sdb_hist[d, s_mask] > 0)[0].shape[0])
        s_v[d] = s_tf_d * s_idf

    # Also a vocab
    v_case = '1000_vocab'
    cpath = os.path.join('/Users/venabled/data/uvan/fc2/f2', v_case)
    v_centers = np.array(bcolz.open(cpath))
    c_norm = (v_centers.T / np.linalg.norm(v_centers, axis=1)).T

    # Create observations from flight imagery
    obs = np.zeros((feat_meta.shape[0], sdb_hist.shape[0]))
    dist = np.zeros_like(obs)

    for imgZ in feat_meta.itertuples():
        print(imgZ.Index, feat_meta.shape[0])
        if imgZ.num_feat > 0:
            feat_desc = bcolz.open(os.path.join(flight_path, imgZ.desc_path), 'r')[:num_feat, :].astype(np.float32)
            dist[imgZ.Index] = tf_idf_match(feat_desc, c_norm, s_v, s_idf, s_mask)
            obs[imgZ.Index] = (1.0 - dist[imgZ.Index]) / np.linalg.norm(1.0 - dist[imgZ.Index])

    neopath = '/Users/venabled/data/uvan/neogeo'
    bcolz.carray(dist, rootdir=os.path.join(neopath, 'f5_10k_tfidf_dist'), mode='w').flush()
    bcolz.carray(obs, rootdir=os.path.join(neopath, 'f5_10k_tfidf_obs'), mode='w').flush()
