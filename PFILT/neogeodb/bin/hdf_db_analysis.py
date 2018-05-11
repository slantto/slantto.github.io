import numpy as np
import neogeodb.features2d as f2d
import neogeodb.hdf_orthophoto as himg
import neogeodb.georeg as georeg
import h5py
import cv2
import mercantile
from pandas import DataFrame
import neogeo.vis as vis


def trim_duplicate_matches(img_pts, db_pts):
    t, idx0 = np.unique(img_pts[:, 0], return_index=True)
    img_pts = img_pts[idx0, :]
    db_pts = db_pts[idx0, :]
    t, idx0 = np.unique(img_pts[:, 1], return_index=True)
    img_pts = img_pts[idx0, :]
    db_pts = db_pts[idx0, :]
    t, idx0 = np.unique(db_pts[:, 0], return_index=True)
    img_pts = img_pts[idx0, :]
    db_pts = db_pts[idx0, :]
    t, idx0 = np.unique(db_pts[:, 1], return_index=True)
    img_pts = img_pts[idx0, :]
    db_pts = db_pts[idx0, :]
    return img_pts, db_pts


def brute_force_matcher(mat1, mat2):
    """
    Finds the two closest matches in mat2 for each row in mat1
    """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(mat1, mat2, k=2)
    out_mat = np.array([(match[0].trainIdx,
                         match[0].distance / match[1].distance) for
                        match in matches])
    return out_mat

hdf_path = '/Users/venabled/catkin_ws/data/neogeo/neogeo_db.hdf'
ophoto = himg.HDFOrthophoto(
    '/Users/venabled/catkin_ws/data/dugway/dugway_ciortho.hdf')
hdf = h5py.File(hdf_path)
feat_geo = hdf['/sift/feat_geo']
pdf = DataFrame({'x': feat_geo[:, 0],
                 'y': feat_geo[:, 1],
                 'z': feat_geo[:, 2],
                 'lon': feat_geo[:, 3],
                 'lat': feat_geo[:, 4],
                 'height': feat_geo[:, 5],
                 'octave': feat_geo[:, 6],
                 'layer': feat_geo[:, 7],
                 'scale': feat_geo[:, 8],
                 'angle': feat_geo[:, 9],
                 'response': feat_geo[:, 10],
                 'size': feat_geo[:, 11]})

aaf_lon_lat = np.array([-112.9255, 40.180630])
aaf_img = ophoto.get_img_from_lon_lat(-112.9255, 40.180630, 1500, 1500)
cv_color_code = cv2.COLOR_RGB2GRAY
aaf_bw = cv2.cvtColor(aaf_img.img, cv_color_code)

aaf_tile = mercantile.tile(aaf_lon_lat[0], aaf_lon_lat[1], 15)

aaf = pdf.query('x==6105 & y==12383')
hdf_desc = hdf['/sift/feature_descriptors']
ref_desc = hdf_desc[aaf.index, :]
ref_octave = np.array(pdf.octave)[aaf.index]

f5 = h5py.File('/Users/venabled/catkin_ws/data/fc2/fc2_f5.hdf', 'r')

# Find a good image in there
imgs = f5['/images/image_data']
img_times = f5['/images/t_valid']
pva = f5['/pva']
ii = 350


# Ok let's start feature matching etc
obs = imgs[ii]
obs = np.rot90(obs, 1)
det_f = '/Users/venabled/neogeo/data/features/default_detector_SIFT.yaml'
des_f = '/Users/venabled/neogeo/data/features/default_descriptor_SIFT.yaml'
detector, descriptor = f2d.load_feature_operators(det_f, des_f)


obs_kp, obs_desc = f2d.extract_features(obs, detector, descriptor)
obs_scale = f2d.numpySIFTScale(obs_kp)

lon_lat = np.array([aaf.lon, aaf.lat]).T
pts_geo = np.array(aaf_img.tf_wgs_to_geo.TransformPoints(lon_lat))
ref_kp = georeg.pix_from_geo(pts_geo[:, 0:2], aaf_img.gt)
d_vec = brute_force_matcher(obs_desc, ref_desc)

# Visualize matches, and #see what we get
idx = np.where(d_vec[:, 1] < 0.7)[0]
idx_2 = d_vec[idx, 0].astype(np.int)
p1 = f2d.keypoint_pixels(obs_kp)
p2 = ref_kp
p1_t = p1[idx, :]
p2_t = p2[idx_2, :]

# Are higher octave features any good?
scale_idx = np.where(ref_octave[idx_2] == 1)
p1_t = p1[idx[scale_idx], :]
p2_t = p2[idx_2[scale_idx], :]


# H, mask = cv2.findHomography(p1_t, p2_t, cv2.cv.CV_RANSAC, 7.0)
img3 = vis.draw_match(obs, aaf_bw, p1_t, p2_t)
plt.imshow(img3)
plt.show()
