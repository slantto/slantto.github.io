import sklearn.cluster as skc
import bcolz
import os
import numpy as np


def create_vocab(feat_meta, path_prefix, k, feat_per_img=5000, sample_factor=3):
    """
    Create a feature vocab from sampling featuers in feat_meta
    :param feat_meta:
    :param sample_factor:
    :param k:
    :return: (KxD) numpy array of cluster centers
    """
    feat_per_img = int(feat_per_img)
    sample_factor = int(sample_factor)

    # Setup K-Means
    km = skc.MiniBatchKMeans()
    km.n_clusters = k

    # Load and sample feature descriptors
    sampled_feat = np.vstack(
            [bcolz.open(os.path.join(path_prefix, db))[:feat_per_img, :] for db
             in feat_meta['desc_path'].iloc[::sample_factor]])
    print("Fitting %d centers from %d Features using MiniBatchKMeans()" % (
                k, sampled_feat.shape[0]))
    km.fit(sampled_feat.astype(np.double))
    return km.cluster_centers_


