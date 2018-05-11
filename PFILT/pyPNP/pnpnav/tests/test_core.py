import pnpnav.core as core
import numpy as np
from pnpnav._shared.testing import *

assert_almost_equal = np.testing.assert_almost_equal
assert_array_equal = np.testing.assert_array_equal


class TestCore(object):

    def setup(self):
        self.num_pts = 1000
        self.zoom = 15
        self.image_x = 1024
        self.image_y = 1024

        lon_lat_h, gp_ned, t_center = generate_ground_points(self.zoom,
                                                             self.num_pts)
        K, distortion = generate_cam_data()
        alt = find_optimal_altitude(gp_ned, K, self.image_x)
        t_w, R_c_w = generate_nadir_data(alt)

        gp_enu = np.hstack((gp_ned[:, [1, 0]], -1 * gp_ned[:, [2]]))
        img_pts = project_ground_points(gp_enu, K, R_c_w, t_w)

        t_ned = np.hstack((t_w[[1, 0]], -1 * t_w[2]))
        t_wgs = navpy.ned2lla(t_ned, t_center[1], t_center[0], t_center[2])

        self.lon_lat_h = lon_lat_h
        self.gp_ned = gp_ned
        self.gp_enu = gp_enu
        self.img_pts = img_pts
        self.K = K
        self.distortion = distortion
        self.t_center = t_center
        self.t_wgs = t_wgs

    def test_core_pnp(self):
        """
        In this test we set up a nominal case with no outliers / noise
        """
        pnp = core.PnP()
        pnp._PnP__camera_matrix = self.K
        pnp._PnP__distortion = self.distortion

        matches = build_matches(self.lon_lat_h, self.img_pts)
        pnp_wgs = pnp.do_opencv_pnp(matches)[0]
        assert_almost_equal(pnp_wgs, self.t_wgs, decimal=3)

    def test_homography_10percent(self):
        """
        In this test we set up a case with 10 percentage outliers
        """
        pnp = core.PnP()
        pnp._PnP__camera_matrix = self.K
        pnp._PnP__distortion = self.distortion
        r_thresh = 7.0
        pnp.use_homography_constraint(r_thresh)
        num_outliers = np.int(0.10 * self.num_pts)
        all_idx = np.arange(self.num_pts)
        out_idx = np.random.randint(0, self.num_pts, num_outliers)
        out_idx = np.unique(out_idx)
        good_idx = np.setdiff1d(all_idx, out_idx)
        num_outliers = out_idx.shape[0]

        ang = np.random.uniform(2 * np.pi)
        R = np.array([[np.cos(ang), -1 * np.sin(ang)],
                      [np.sin(ang), -1 * np.cos(ang)]])
        img_2 = np.dot(R, self.img_pts.T).T
        # perturb the img2 points
        img_err = np.random.uniform(r_thresh, 2 * r_thresh, (num_outliers, 2))
        img_2[out_idx, :] = img_2[out_idx, :] + img_err

        h_x = pnp._PnP__apply_geometric_constraint(self.img_pts, img_2)
        assert_array_equal(h_x, good_idx)
