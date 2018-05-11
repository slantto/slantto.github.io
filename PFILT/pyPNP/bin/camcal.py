#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module subscribes to a topic providing\
vision_nav_msgs/FeatureCorrespondence2D3D messages, and uses the included 2D \
image pixel locations along with the 3D WGS-84 Lon,Lat,HAE measurements that \
were corresponded in some type of "appearance-space" matching to calculate \
a global pose (Position and Attitude). Currently this version of PnP does \
not provide a rigorous Covariance matrix for these measurements of global \
pose
"""

import numpy as np
import cv2
import navpy
import pandas as pd
import pnpnav.utils as pnputils
import numdifftools as nd
import yaml
import matplotlib.pyplot as plt
import os


def skew(in_vec):
    """Input 3x1 vector, output 3x3 skew symmetric Matrix"""
    return np.array([[0, -1 * in_vec[2], in_vec[1]],
                     [in_vec[2], 0, -1 * in_vec[0]],
                     [-1 * in_vec[1], in_vec[0], 0]])


def undistort_keypoints(K, distort, keypoints):
    nc = keypoints.shape[0]
    dist_img_pts = keypoints.reshape(nc, 1, 2).astype(np.float32)
    norm_pts = cv2.undistortPoints(dist_img_pts, K, distort)
    norm_pts = np.hstack((norm_pts.squeeze(), np.ones((nc, 1))))
    img_pts = np.dot(K, norm_pts.T).transpose()[:, 0:2]
    return img_pts


def project_points(K, C_c_b, body_points):
    x_cam = np.dot(C_c_b, body_points.T)
    x_hat = np.dot(K, x_cam).T
    return x_hat[:, 0:2] / np.tile(x_hat[:, 2], (2, 1)).T


class CamCalBlock(object):
    def __init__(self, C_b_nav, C_c_b_nom, trans_w, world_pt, img_pt, distortion):
        self.world_point = np.copy(world_pt)
        self.t_w = np.copy(trans_w)
        self.C_b_w = np.copy(C_b_nav)
        self.C_c_b_nom = np.copy(C_c_b_nom)
        self.C_c_b_nom_buf = np.copy(C_c_b_nom)
        self.img_pt = np.copy(img_pt)
        self.distortion = distortion
        self.jacobian = self.hand_jacobian
        # self.jacobian = self.numerical_jacobian

    def project_point(self, state_vec):
        """
        This function generates a measured image point given the currently \
        estimated focal length in state-vec, and the metadata (world point, \
        C_nav_v, C_b_cam_nom, etc.)
        """
        K = np.array([state_vec[0], 0, state_vec[1],
                      0, state_vec[0], state_vec[2],
                      0, 0, 1]).reshape(3, 3)

        bias_w = state_vec[6:9]

        C_c_b_perturb = np.dot(np.eye(3) - skew(state_vec[3:6]), self.C_c_b_nom)
        C_c_w = np.dot(C_c_b_perturb, self.C_b_w)
        x_hat = np.dot(K, np.dot(C_c_w, self.world_point - bias_w - self.t_w))
        x_hat = x_hat[0:2] / x_hat[2]
        return x_hat

    def numerical_jacobian(self, state_vec):
        jac = nd.Jacobian(self.project_point)
        return -1*jac(state_vec)

    def calc_residual(self, state_vec):
        return (self.img_pt - self.project_point(state_vec)).reshape(2, 1)

    def update_rotation(self, state_vec):
        new_rot_mat = np.dot(np.eye(3) - skew(state_vec[3:6]), self.C_c_b_nom)
        self.C_c_b_nom_buf = np.copy(self.C_c_b_nom)
        self.C_c_b_nom = pnputils.OrthoDcm(new_rot_mat)

    def revert_rotation(self):
        self.C_c_b_nom = np.copy(self.C_c_b_nom_buf)

    def hand_jacobian(self, state_vec):
        """
        This function generates a measured image point given the currently \
        estimated focal length in state-vec, and the metadata (world point, \
        C_nav_v, C_b_cam_nom, etc.)
        """
        K = np.array([state_vec[0], 0, state_vec[1],
                      0, state_vec[0], state_vec[2],
                      0, 0, 1]).reshape(3, 3)

        bias_w = state_vec[6:9]
        C_c_b_perturb = np.dot(np.eye(3) - skew(state_vec[3:6]), self.C_c_b_nom)
        C_c_w = np.dot(C_c_b_perturb, self.C_b_w)
        x_cam = np.dot(C_c_w, self.world_point - bias_w - self.t_w)

        x_i = np.dot(K, x_cam).reshape(3, 1)

        df_dx = -1 * np.dot(K, C_c_w)
        g = x_i[0:2].reshape(2, 1)
        h = x_i[2]
        dg = df_dx[0:2, :]
        dh = df_dx[2, :].reshape(1, 3)

        jac_block = -1*(
            (dg*h - np.tile(dh, (2, 1))*np.tile(g, (1, 3)))/h**2)

        J = np.zeros((2, 9))
        J[0, 0] = -1 * x_cam[0] / x_cam[2]
        J[1, 0] = -1 * x_cam[1] / x_cam[2]
        J[0, 1] = -1.0
        J[1, 2] = -1.0

        df_dpsi = np.dot(K, skew(x_cam))
        J[:, 3:6] = -1*(df_dpsi[0:2,:]*(1/x_i[2]) - (np.tile(x_i[0:2], (1, 3))*np.tile(df_dpsi[2,:], (2, 1))) / x_i[2]**2)

        J[:, 6:9] = jac_block
        return J


class CamCalProblem(object):
    """
    Whatever, its a non-linear least squares problem
    """
    def __init__(self, K):
        self.states = {}
        self.residuals = []
        self.x = np.array([tf.K[0, 0], tf.K[0, 2], tf.K[1, 2],
                           0, 0, 0, 0, 0, 0])
        self.z = np.array([])
        self.J = np.array([])

    def add_residual_block(self, residual_block):
        self.residuals.append(residual_block)

    def build_problem_matrices(self):
        self.z = np.vstack(
            [block.calc_residual(self.x) for block in self.residuals])
        self.J = np.vstack(
            [block.jacobian(self.x) for block in self.residuals])

    def solve(self):
        """
        Try LM
        """
        imgs, pcount =  np.unique(pairs['img_num'], return_counts=True)
        self.build_problem_matrices()
        self.lam = 10
        cost = (self.z**2).sum()**(0.5)
        max_iter = 40
        trial = 0
        run_LM = True

        fig = plt.figure()
        idx = 0
        for count in pcount:

            z0 = self.z[idx:idx + 2*count]
            plt.plot(z0[::2], z0[1::2], 'x')
            idx = idx + 2*count
        ax = plt.gca()
        lim = np.max(np.abs((ax.get_xlim(), ax.get_ylim())))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')
        plt.grid('on')
        plt.title('Original Resid :: x: (%f, %f) y: (%f, %f)' % (self.z[::2].mean(), self.z[::2].var(),
                                                                         self.z[1::2].mean(), self.z[1::2].var()))
        plt.savefig(os.path.join(plot_dir, 'alpha.png'))

        while (run_LM and trial < max_iter):

            self.build_problem_matrices()
            g = np.dot(self.J.T, self.z)
            H = np.dot(self.J.T, self.J)
            D = np.diag(np.sqrt(np.diag(H)))
            H_aug = H + self.lam*np.dot(D.transpose(), D)
            del_x = np.dot(np.linalg.inv(H_aug), -1*g)
            self.x = self.x + del_x.flatten()

            for block in self.residuals:
                block.update_rotation(self.x)

            # Nuke the del_rotations
            self.x[3:6] = 0.0
            self.build_problem_matrices()
            new_cost = (self.z**2).sum()**(0.5)

            print "%f :: %f :: %f" % (cost, new_cost, self.lam)
            print(self.x)

            fig = plt.figure()
            idx = 0
            for count in pcount:

                z0 = self.z[idx:idx + 2*count]
                plt.plot(z0[::2], z0[1::2], 'x')
                idx = idx + 2*count
            ax = plt.gca()
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.spines['bottom'].set_position('zero')
            ax.spines['left'].set_position('zero')
            plt.grid('on')
            plt.title('Residual Iter %d :: x: (%f, %f) y: (%f, %f)' % (trial, self.z[::2].mean(), self.z[::2].var(),
                                                                               self.z[1::2].mean(), self.z[1::2].var()))
            plt.savefig(os.path.join(plot_dir, 'iter_%d.png' % trial))
            plt.close('all')

            # Adjust Lambda for how bad we did
            if new_cost <= cost:
                self.lam = self.lam / 2.1

                # Check to see if we're done iterating...
                del_c = np.abs(new_cost - cost)
                if (del_c) < 10*np.finfo(float).eps:
                    run_LM = False
                cost = new_cost
            else:
                self.x = self.x - del_x.flatten()
                for block in self.residuals:
                    block.revert_rotation()
                # Nuke the del_rotations
                self.x[3:6] = 0.0
                self.lam = self.lam * 4

            trial += 1
        return trial

srtm_path = '/Users/venabled/data/srtm/SRTM1/Region_04'
# cam_path = '/Users/venabled/pysrc/pnpnav/data/fc2_nom_cam.yaml'
# uvan_frames = '/Users/venabled/pysrc/pnpnav/data/fc2_nom_frames.yaml'
# cam_path = '/Users/venabled/pysrc/pnpnav/data/fc2_cam_model.yaml'
# uvan_frames = '/Users/venabled/pysrc/pnpnav/data/fc2_pod_frames.yaml'
# cam_path = '/Users/venabled/pysrc/pnpnav/data/nom_autocamcal2.yaml'
# uvan_frames = '/Users/venabled/pysrc/pnpnav/data/nom_autoframes2.yaml'
cam_path = '/Users/venabled/data/c5/config/nadir_camera_XXX.yaml'
uvan_frames = '/Users/venabled/data/c5/config/wsmr_frames.yaml'
geoid_file = '/Users/venabled/data/geoid/egm96_15.tiff'
plot_dir = '/Users/venabled/data/cland_cal_plots'

# Find some tiles
tf = pnputils.DEMTileFinder(srtm_path, geoid_file)
tf.load_cam_and_vehicle_frames(uvan_frames)
tf.load_camera_cal(cam_path)

pairs = pd.read_hdf('/Users/venabled/data/cland_wsmr/f1_sampled_pairs.hdf')
pairs = pairs[pairs.rss_error < 15.0]
pairs = pairs[pairs.ransac_status == 1]

imgs, counts = np.unique(pairs['img_num'], return_counts=True)
problem = CamCalProblem(tf.K)

# Get local level frame
ref = np.array(pairs[['veh_lon', 'veh_lat']].mean())
ref = np.append(ref, pairs['veh_height'].min() - 200.0)

# Get local level frame
ref = np.array(pairs[['veh_lon', 'veh_lat']].mean())
ref = np.append(ref, pairs['veh_height'].min() - 200.0)

for img_num in np.unique(pairs['img_num']):

    # img_num = imgs[counts.argmax()]
    pair = pairs[pairs['img_num'] == img_num]
    lon_lat_h = np.array([pair['veh_lon'].iloc[0], pair['veh_lat'].iloc[0],
                          pair['veh_height'].iloc[0]])
    att = pnputils.rpy_to_cnb(pair['veh_roll'].iloc[0], pair['veh_pitch'].iloc[0],
                              pair['veh_yaw'].iloc[0])
    C_n_v = att
    C_cam_ned = C_n_v.dot(np.dot(tf.C_b_v.T, tf.C_b_cam))

    feat_wgs = np.array([pair['feat_lon'], pair['feat_lat'],
                         pair['feat_height']]).T

    world_points = navpy.lla2ned(feat_wgs[:, 1], feat_wgs[:, 0],
                                 feat_wgs[:, 2], ref[1], ref[0], ref[2])

    t_nav = navpy.lla2ned(lon_lat_h[1], lon_lat_h[0], lon_lat_h[2],
                          ref[1], ref[0], ref[2])
    t_vec = np.dot(C_cam_ned, t_nav)
    C_b_nav = tf.C_b_v.dot(C_n_v.T)
    body_points = np.dot(C_b_nav, world_points.T).T - np.dot(C_b_nav, t_nav)
    C_c_nav = np.dot(tf.C_b_cam.T, C_b_nav)

    # img_meas = project_points(tf.K, tf.C_b_cam.T, body_points)

    # Undistort the measurements outside of the opimization loop
    img_meas = undistort_keypoints(tf.K, tf.distortion,
                                    np.array([pair['feat_img_x'],
                                              pair['feat_img_y']]).T)

    for ii in np.arange(world_points.shape[0]):
        blk = CamCalBlock(C_b_nav, tf.C_b_cam.T, t_nav, world_points[ii, :],
                    img_meas[ii, :], tf.distortion)
        blk_rss = np.linalg.norm(blk.project_point(problem.x))
        if (blk_rss < 25000.0):
            problem.add_residual_block(blk)

problem.solve()

state_vec = problem.x
cam_nom = problem.residuals[0].C_c_b_nom
cam_cal_yaml = yaml.load(file(cam_path))
cal_K = np.array([state_vec[0], 0, state_vec[1],
                  0, state_vec[0], state_vec[2],
                  0, 0, 1])
cal_P = np.zeros((3,4))
cal_P[:, 0:3] = cal_K.reshape(3,3)
cam_cal_yaml['camera_matrix']['data'] = cal_K.flatten().tolist()
# cam_cal_yaml['projection_matrix']['data'] = cal_P.flatten().tolist()
# cam_cal_yaml['cam_to_body_dcm'] = cam_nom.T.flatten().tolist()
yaml.dump(cam_cal_yaml, file('/Users/venabled/data/c5/config/cland_autocamcal2.yaml', 'w'))

frames = yaml.load(file(uvan_frames))
frames['/camera/image_raw'] = cam_nom.T.flatten().tolist()
yaml.dump(frames, file('/Users/venabled/data/c5/config/cland_autoframes2.yaml', 'w'))

bias = state_vec[6:9]
yaml.dump({'bias': bias.tolist()}, file('/Users/venabled/data/c5/config/cland_autocal_bias.yaml', 'w'))