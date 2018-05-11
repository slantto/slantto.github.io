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
import navpy
import yaml
from .import core
import cv2


def skew(in_vec):
    """Input 3x1 vector, output 3x3 skew symmetric Matrix"""
    return np.array([[0, -1*in_vec[2], in_vec[1]],
                    [in_vec[2], 0, -1*in_vec[0]],
                    [-1*in_vec[1], in_vec[0], 0]])


def image_point_translation_worldpt_meas_func(state_vec, metadata):
    """
    The actual measurement function h(x, metadata) for calculating the \
    predicted image measurement based on two parameter blocks (translation \
    vector, the world point to be projected into the image frame) and \
    metadata consisting of rotation from world to camera frame (from IMU), \
    and the camera matrix K
    :param numpy.ndarray state_vec: (6,) vector built from stacking 2x(3,) \
        vectors of:
        - state_vec[0:3] = translation of the camera as represented in the \
        world frame
        - state_vec[3:6] = (3,) vector of the point represented in \
        world coordinate points (x,y,z)
    :param dictionary metadata: Dictionary of metadata with keys as follows:
        - K - numpy.ndarray - 3x3 camera matrix
        - R_c_w - numpy.ndarray - 3x3 DCM rotating from world to cam frame
    """
    K = metadata['K'].reshape(3, 3)
    R_c_w = metadata['R_c_w'].reshape(3, 3)
    t_vec = state_vec[0:3].reshape(3, 1)
    world_point = state_vec[3:6].reshape(3, 1)
    # local_bias = state_vec[6:9].reshape(3, 1)
    # world_point = world_point - local_bias

    x_hat = np.dot(K, np.dot(R_c_w, world_point) - np.dot(R_c_w, t_vec))
    x_hat = x_hat[0:2, :] / x_hat[2, :]
    return x_hat.flatten()


def image_point_trans_worldpt_jacobian_func(state_vec, metadata):
    """
    Calculate block Jacobians, returns a list of Numpy.NDarrays corresponding
    to state_bock
    """
    K = metadata['K'].reshape(3, 3)
    R_c_w = metadata['R_c_w'].reshape(3, 3)
    t_w = state_vec[0:3].reshape(3, 1)
    world_point = state_vec[3:6].reshape(3, 1)
    # local_bias = state_vec[6:9].reshape(3, 1)
    # world_point = world_point - local_bias

    x_i = np.dot(K, np.dot(R_c_w, world_point) - np.dot(R_c_w, t_w))
    df_dx = -1 * np.dot(K, R_c_w)
    g = x_i[0:2].reshape(2, 1)
    h = x_i[2]
    dg = df_dx[0:2, :]
    dh = df_dx[2, :].reshape(1, 3)

    jac_block = -1*(
        (dg*h - np.tile(dh, (2, 1))*np.tile(g, (1, 3)))/h**2)
    jacobian = [jac_block, -1*jac_block, -1*jac_block]
    return jacobian


def worldpt_meas_func(state_vec, metadata=None):
    """
    This is a really simple pseudo-measurement function for direct \
    measurements of world points
    """
    x_hat = state_vec[0:3]
    local_bias = state_vec[3:6]
    return x_hat - local_bias


def worldpt_jac_func(state_vec, metadata=None):
    """
    Jacobian here is identity
    """
    return [-1*np.eye(3), np.eye(3)]


class StateBlock(object):
    """
    This class represents StateBlocks, or states to be estimated. Constructor
    asks for a name and initial values for the states (even if not known)
    """
    def __init__(self, block_name, init_values):
        """
        Class Constructor, takes in:
        :param string block_name: unique name of block
        :param numpy.ndarray init_values: (N,) sized array of initial values
        """
        self.name = block_name
        self.__init_values = init_values
        self.size = init_values.shape[0]
        self.__problem = None

    def now_its_your_problem(self, problem, idx):
        """
        Add problem info to this state block
        """
        self.__prob_idx = idx
        self.__problem = problem
        self.__problem.x[self.__prob_idx:self.__prob_idx+self.size] = np.copy(
            self.__init_values)

    def get_current_val(self):
        """
        Maybe you want the current value of this state?
        """
        if self.__problem is None:
            raise ValueError('Problem not initialized')
        return self.__problem.x[self.__prob_idx:self.__prob_idx+self.size]


class ResidualBlock(object):
    """
    This measurement / residual block represents a measurement of information \
    being estimated by the Problem
    """
    def __init__(self, states_measured, meas_func, jacobian_func):
        """
        Class constructor, at construction time need to know which states \
        you're providing information on, how they're related to the meas_vec \
        via the meas_function, and the function that calcs jacobians \
        :param list states_measured: List of strings of StateBlock.names \
            ordered by how they're passed into
        """
        self.states_measured = states_measured
        self.meas_func = meas_func
        self.jacobian_func = jacobian_func
        self.metadata = None
        self.z = np.array([])
        self.R = np.array([])
        self.good_meas = False

    def now_its_your_problem(self, problem):
        """
        Add problem info to this state block
        """
        self.problem = problem

    def set_meas(self, z, R):
        self.z = z
        self.R = R
        self.good_meas = True
        self.size = z.shape[0]

    def update_metadata(self, new_metadata):
        self.metadata = new_metadata

    def calc_residual(self):
        """
        This function calculates the residual vector to be used in the \
        non-linear optimization
        """
        state_vec = np.array(
            [self.problem.states[state].get_current_val()
             for state in self.states_measured]).flatten()
        self.jacobian = self.jacobian_func(state_vec, self.metadata)
        r = self.z - self.meas_func(state_vec, self.metadata)
        # r = np.dot(r.T, np.dot(np.linalg.inv(self.R), r))
        self.residual = r


class Problem(object):
    """
    Whatever, its a non-linear least squares problem
    """
    def __init__(self):
        self.states = {}
        self.residuals = []
        self.x = np.array([])
        self.z = np.array([])
        self.J = np.array([])

    def add_state_block(self, state_block):
        block_idx = self.x.shape[0]
        self.x = np.resize(self.x, self.x.shape[0] + state_block.size)
        state_block.now_its_your_problem(self, block_idx)
        self.states[state_block.name] = state_block

    def add_residual_block(self, residual_block):
        residual_block.now_its_your_problem(self)
        self.residuals.append(residual_block)

    def build_problem_matrices(self):
        self.z = np.array([])
        for resid_block in self.residuals:
            if resid_block.good_meas:
                z_idx = self.z.shape[0]
                resid_block.calc_residual()
                self.z = np.append(self.z, resid_block.residual)
        self.J = np.zeros((self.z.shape[0], self.x.shape[0]))

        z_idx = 0
        for resid_block in self.residuals:
            stop_z = z_idx + resid_block.z.shape[0]
            for ii in np.arange(len(resid_block.states_measured)):
                sm = resid_block.states_measured[ii]
                jac_block = resid_block.jacobian[ii]
                x_idx = self.states[sm]._StateBlock__prob_idx
                x_stop = x_idx + self.states[sm].size
                self.J[z_idx:stop_z, x_idx:x_stop] = np.copy(jac_block)
            z_idx = stop_z

    def solve(self):
        """
        Try LM
        """
        self.build_problem_matrices()
        self.lam = 10.0
        cost = (self.z**2).sum()**(0.5)
        max_iter = 1000
        trial = 0
        run_LM = True
        try:
            while (run_LM and trial < max_iter):
                # print "%f :: %f" % ((self.z**2).sum()**(0.5), self.lam)
                self.build_problem_matrices()
                g = np.dot(self.J.T, self.z)
                H = np.dot(self.J.T, self.J)
                D = np.diag(np.sqrt(np.diag(H)))
                H_aug = H + self.lam*np.dot(D.transpose(), D)
                del_x = np.dot(np.linalg.inv(H_aug), (-1*g.reshape(g.shape[0], 1)))
                self.x = self.x + del_x.flatten()
                self.build_problem_matrices()
                new_cost = (self.z**2).sum()**(0.5)

                # Adjust Lambda for how bad we did
                if new_cost <= cost:
                    self.lam = self.lam * .10
                else:
                    self.lam = self.lam * 2

                # Check to see if we're done iterating...
                del_c = np.abs(new_cost - cost)
                if (del_c) < 10*np.finfo(float).eps:
                    run_LM = False
                cost = new_cost
                trial += 1
            return trial
        except:
            return trial


class NoAttitudeAndBiasPnP(core.PnP):

    def load_frames(self, frame_yaml):
        """
        Set up the internal coordinate systems from a yaml file that
        stores DCMs from the camera frame to body and from vehicle
        frame to body as (9,) flattened row-major arrays.
        """
        vehicle_frames = yaml.load(file(frame_yaml, 'r'))
        self.C_b_cam = np.array(
            vehicle_frames['/camera/image_raw']).reshape(3, 3)
        self.C_b_v = np.array(vehicle_frames['/vehicle']).reshape(3, 3)

    def undistort_keypoints(self, keypoints):
        K = self._PnP__camera_matrix
        distort = self._PnP__distortion
        nc = keypoints.shape[0]
        dist_img_pts = keypoints.reshape(nc, 1, 2).astype(np.float32)
        norm_pts = cv2.undistortPoints(dist_img_pts, K, distort)
        norm_pts = np.hstack((norm_pts.squeeze(), np.ones((nc, 1))))
        img_pts = np.dot(K, norm_pts.T).transpose()[:, 0:2]
        return img_pts

    def setup_pnp_problem(self, matches, idx, init_wgs, C_nav_v):
        """
        If you've got 99 problems, give yourself one more
        """
        K = self._PnP__camera_matrix

        keypoints = matches.keypoints[idx, :]
        img_pts = self.undistort_keypoints(keypoints)

        # First we need to create a local level Cartesian frame (NED),
        lon_lat_h = matches.world_coordinates[idx, :]
        ref = lon_lat_h.mean(0)
        ref[2] = ref[2] - 150.0
        world_n = navpy.lla2ned(
            lon_lat_h[:, 1], lon_lat_h[:, 0], lon_lat_h[:, 2],
            ref[1], ref[0], ref[2])

        if self._PnP__db_bias is not None:
            world_n = world_n - self._PnP__db_bias

        t_n = navpy.lla2ned(init_wgs[0], init_wgs[1], init_wgs[2],
                            ref[1], ref[0], ref[2])

        C_nav_b = np.dot(self.C_b_v, C_nav_v.T).T
        R_c_w = np.dot(self.C_b_cam.T, C_nav_b.T)

        X_hat = world_n.T
        x_img = img_pts.T

        #Setup Constrained PnP
        metadata = {'K': K, 'R_c_w': R_c_w}
        prob = Problem()
        trans_state = StateBlock('translation', t_n.flatten())
        world_pt_states = [StateBlock('pt_%d'%ii, X_hat[:,ii].flatten()) for ii in np.arange(X_hat.shape[1])]
        prob.add_state_block(trans_state)

        bias_state = StateBlock('local_bias', np.zeros(3))
        prob.add_state_block(bias_state)

        for state in world_pt_states:
            prob.add_state_block(state)
        resids = []

        for ii in np.arange(keypoints.shape[0]):
            x_i = x_img[:, ii].T
            states_measured = ['translation', 'pt_%d' % ii]
            resid = ResidualBlock(states_measured,
                                  image_point_translation_worldpt_meas_func,
                                  image_point_trans_worldpt_jacobian_func)
            resid.set_meas(x_i, np.eye(2))
            resid.update_metadata(metadata)
            resids.append(resid)
            prob.add_residual_block(resid)

        # Add in Ground Control Meas....
        for ii in np.arange(keypoints.shape[0]):
            X_w = X_hat[:, ii].T
            states_measured = ['pt_%d' % ii, 'local_bias']
            resid = ResidualBlock(states_measured,
                                  worldpt_meas_func,
                                  worldpt_jac_func)
            resid.set_meas(X_w, np.diag([5.0, 5.0, 10.0]))
            resid.update_metadata(None)
            resids.append(resid)
            prob.add_residual_block(resid)

        niter = prob.solve()

        R = np.array([])
        for r in resids:
            R = np.append(R,np.diag(r.R))
        R = np.diag(R)
        try:
            cov = np.linalg.pinv(np.dot(prob.J.T, np.dot(np.linalg.inv(R), prob.J)))
            pnp_n = prob.states['translation'].get_current_val()
            cov_n = np.abs(np.diag(cov)[0:3])
            pnp_wgs = np.array(navpy.ned2lla(pnp_n, ref[1], ref[0], ref[2]))
            lbias = prob.states['local_bias'].get_current_val()
            print("Local Bias: %s" % lbias)
            return pnp_wgs, cov_n, niter
        except:
            return (np.array([]), np.array([]), 0)

    def do_no_tude_pnp(self, matches, pnp_idx, c_wgs, C_nav_v, return_matches=False, return_cv=False):
        if pnp_idx.shape[0] > 0:
            c_pnp, cov_n, niter = self.setup_pnp_problem(matches, pnp_idx,
                                                         c_wgs, C_nav_v)
        else:
            c_pnp, cov_n, niter = (np.array([]), np.array([]), 0)
        out_tuple = (c_pnp, cov_n, niter)
        if return_matches:
            out_tuple = out_tuple + (matches, pnp_idx)
        if return_cv:
            out_tuple = out_tuple + (c_wgs,)
        return out_tuple

    def do_pnp(self, query_kp, query_desc, C_nav_v, return_matches=False, return_cv=False):
        """
        Compute the constrained attitude. RPY is assumed to be from the \
        INSPVA, which expresses a rotation from vehicle frame to NED. \
        We assume a no-lever arm between camera, IMU, body, vehicle. \
        This function can only be called after PnP is successful, and will \
        overwrite self.__cartesian_position
        :param numpy.ndarray C_nav_v: 3x3 vehicle to nav rotation matrix\
            to NED
        :param numpy.ndarray C_b_v: 3x3 vehicle to body rotation matrix
        :param numpy.ndarray C_b_c: 3x3 camera to body rotation matrix
        """

        matches = self._PnP__matcher.match(query_kp, query_desc)
        c_wgs, pnp_idx = self._PnP__opencv_pnp(matches)
        if pnp_idx.shape[0] > 0:
            c_pnp, cov_n, niter = self.setup_pnp_problem(matches, pnp_idx,
                                                         c_wgs, C_nav_v)
        else:
            c_pnp, cov_n, niter = (np.array([]), np.array([]), 0)
        out_tuple = (c_pnp, cov_n, niter)
        if return_matches:
            out_tuple = out_tuple + (matches, pnp_idx)
        if return_cv:
            out_tuple = out_tuple + (c_wgs,)
        return out_tuple
