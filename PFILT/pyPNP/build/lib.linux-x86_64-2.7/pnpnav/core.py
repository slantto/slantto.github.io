#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides helper functions for computing pose from 2D/3D
correspondences using the OpenCV solvePnPRansac() function. This module
will compute both Cartesian (ECEF) and WGS-84 pose as requested.
"""

import cv2
import yaml
import numpy as np
import navpy
from .import matching
from .import utils
from .import database as pnpdb


class PnP(object):

    """
    This class implements the Perspective-N-Point "bundle adjustment", or \
    pose from 2D-3D correspondences given a calibrated camera matrix
    """

    def __init__(self, matcher=matching.BFMatcher(), db_bias=None):
        """
        Initialization of the MatchVis object
        """
        self.good_tile = False
        self.__camera_matrix = np.zeros((3, 3))
        self.__db_bias = db_bias
        self.__distortion = np.zeros(5)
        self.__geometric_constraint = None
        self.__constraint_parameters = None
        self.__current_tiles = None
        self.__matcher = matcher
        self.logger = utils.get_log()

    def use_fundamental_matrix_constraint(self,
                                          ransac_threshold=5.0,
                                          ransac_confidence=0.99):
        """
        This function sets the PnP class to use a fundamental matrix\
        geometric constraint + RANSAC to try to remove outliers from the\
        2D/3D correspondences.

        :param double ransac_threshold: Parameter used for RANSAC. It is the\
            maximum distance from a point to an epipolar line in pixels,\
            beyond which the point is considered an outlier and is not used\
            for computing the final fundamental matrix. It can be set to\
            something like 1-3, depending on the accuracy of the point\
            localization, image resolution, and the image noise. Default = 3.0\
        :param double ransac_confidence: Parameter used to specify a desirable\
            level of confidence (probability) that the estimated matrix is\
            correct. Default = 0.99
        :return None
        """
        self.__geometric_constraint = cv2.findFundamentalMat
        self.__constraint_parameters = {'method': cv2.FM_RANSAC,
                                        'param1': ransac_threshold,
                                        'param2': ransac_confidence}

    def use_homography_constraint(self, ransac_threshold=5.0):
        """
        This function sets the PnP class to use a homography as a geometric\
        constraint, using RANSAC to identify outliers from the 2D/3D\
        correspondences

        :param double ransac_threshold: Maximum allowed reprojection error to\
            treat a point pair as an inlier, if this is in pixels then OpenCV\
            suggests something between 1-10 for a constraint. Maybe more if\
            you're using something geometric for one constraint
        :return None
        """
        self.__geometric_constraint = cv2.findHomography
        self.__constraint_parameters = \
            {'method': cv2.RANSAC,
             'ransacReprojThreshold': ransac_threshold}

    def __apply_geometric_constraint(self, img_1_pts, img_2_pts):
        """
        This function takes in two numpy.ndarrays of the same size (Nx2)\
        and applies the RANSAC based geometric constraint stored in\
        self.__geometric_constraint to identify inliers, and return the\
        indicies of the inliers as a numpy.ndarray named idx, sized Mx1 s.t.\
        img_1_pts[idx,:] and img_2_pts[idx,:] represent the inliers. If\
        no Geometric constraint was setup using the use_XX_constrating()\
        functions, this function will return idx = np.arange(N)

        :param numpy.ndarray img_1_pts: Nx2 numpy.ndarray of image points\
            from the first image
        :param numpy.ndarray img_2_pts: Nx2 numpy.ndarray of image points\
            from the second image
        :return numpy.ndarray: Indicies into img_X_pts of inliners based\
            on geometric constraint. If self.__geometric_constraint is None\
            then idx == np.arange(img_X_pts.shape[0])
        """
        if not (img_1_pts.shape == img_2_pts.shape):
            raise ValueError("img_1_pts.shape not equal to img_2_pts.shape")
        if self.__geometric_constraint is None:
            idx = np.arange(img_1_pts.shape[0])
        else:
            Z, mask = \
                self.__geometric_constraint(img_1_pts,
                                            img_2_pts,
                                            **self.__constraint_parameters)
            idx = np.where(mask == 1)[0]
        return idx

    def load_camera_parameters(self, camera_yaml):
        """
        This function reads in a YAML formatted OpenCV camera calibration\
        file and stores the members internally into self.__camera_matrix\
        and self.__distortion for use in the PnP process

        :param string camera_yaml: Path to yaml-formatted OpenCV-style camera\
            calibration file. YAML file needs to have fields "camera_matrix"\
            and "distortion_coefficients", which should be sized 3x3 and\
            1x5 respectively
        """
        with open(camera_yaml, 'r') as stream:
            camera_cal = yaml.load(stream)

        # Check that the camera matrix exists and is the correct size
        if 'camera_matrix' not in camera_cal.keys():
            raise ValueError('Field "camera_matrix" does not exist in\
                %s' % camera_yaml)
        elif ((camera_cal['camera_matrix']['rows'] != 3) or
              (camera_cal['camera_matrix']['cols'] != 3)):
            raise ValueError('Camera matrix should be 3x3, provided matrix\
                is %d x %d' % (camera_cal['camera_matrix']['rows'],
                               camera_cal['camera_matrix']['cols']))
        self.__camera_matrix = \
            np.array(camera_cal['camera_matrix']['data']).reshape(3, 3)

        # Save off camera distortion
        if 'distortion_coefficients' not in camera_cal.keys():
            raise ValueError('Field "distortion_coefficients"  not in\
                %s' % camera_yaml)
        elif ((camera_cal['distortion_coefficients']['rows'] != 1) or
              (camera_cal['distortion_coefficients']['cols'] != 5)):
            raise ValueError('Camera matrix should be 1x5, provided matrix\
                is %d x %d' % (camera_cal['distortion_coefficients']['rows'],
                               camera_cal['distortion_coefficients']['cols']))
        self.__distortion = \
            np.array(camera_cal['distortion_coefficients']['data'])

    def load_pytables_db(self, db_path, db_class=pnpdb.SplitPandasDatabase):
        """
        Instantiates a pnpnav.PyTablesDatabase object and assigns it to \
        self.__db.
        """
        self.__matcher.load_db(db_path, db_class)

    def set_db_location_from_tiles(self, tiles, N=None):
        """
        PnP needs a local area to search from. This method sets the current \
        center of the search location. Depending on other internal parameters \
        it will load features from the current tile, and possibly adjacent
        ones. This function checks to make sure that if we've already loaded \
        the current tile that we don't bother loading new features.
        """
        if tiles != self.__current_tiles:
            self.__current_tiles = tiles
            self.good_tile = self.__matcher.load_features_from_tiles(tiles, N)

    def set_db_location_from_bbox(self, bbox, N=None):
        """
        PnP needs a local area to search from. This method sets the current \
        center of the search location. Depending on other internal parameters \
        it will load features from the current tile, and possibly adjacent
        ones. This function checks to make sure that if we've already loaded \
        the current tile that we don't bother loading new features.
        """
        self.good_tile = self.__matcher.load_features_from_bbox(bbox, N)


    def get_loaded_db_idx(self):
        """
        Query the currently loaded DB landmarks, and return their indices into
        the full DB
        :return: Numpy Array of loaded database landmark indices
        """
        return self.__matcher.get_loaded_db_idx()

    def do_pnp(self, query_kp, query_desc, return_matches=False):
        """
        This is the main interface into PnP. Right now it will use \
        self.__matcher to match the query features to the currently loaded \
        features, and then pick a method to do PnP
        """
        matches = self.__matcher.match(query_kp, query_desc)
        if return_matches:
            t_wgs, idx = self.__opencv_pnp(matches)
            return t_wgs, matches, idx
        else:
            return self.__opencv_pnp(matches)[0]

    def do_opencv_pnp(self, new_matches, ref=None):
        return self.__opencv_pnp(new_matches, ref)

    def __opencv_pnp(self, new_matches, ref=None):
        """
        This function is a callback which listens for new\
        vision_nav_msgs/FeatureCorrespondence2D3D messages, and computes pose \
        (3D position and orientation) of the camera in the world frame \
        associated with new_matches.world_coordinate_frame. For now \
        we're assuming this is WGS-84, [Longitude (deg), Latitude (deg), \
        Height Above Ellipsoid (meters)]. Need to add logic in to transform\
        these points to some arbitrary frame later on.
        """

        if new_matches.num_correspondences > 5:
            keypoints = new_matches.keypoints
            lon_lat_h = new_matches.world_coordinates

            # First we need to create a local level Cartesian frame (NED),
            # and get the world_coordinates into that frame
            # Use the first world point as the reference for NED frame
            if ref is None:
                ref = lon_lat_h.mean(0)
                ref[2] = ref[2] - 150.0
            world_pts_ned = navpy.lla2ned(
                lon_lat_h[:, 1], lon_lat_h[:, 0], lon_lat_h[:, 2],
                ref[1], ref[0], ref[2])

            if self.__db_bias is not None:
                world_pts_ned = world_pts_ned - self.__db_bias

            # Apply the 2D geometric constraint
            idx = self.__apply_geometric_constraint(
                world_pts_ned[:, 0:2].astype(np.float32),
                keypoints.astype(np.float32))

            print("%d kp from Matcher :: %d passed geometric constraint" %
                  (new_matches.num_correspondences, idx.shape[0]))

            # Do PnP
            if idx.shape[0] > 6:
                success, rvec, tvec, pnp_status = \
                    cv2.solvePnPRansac(
                        world_pts_ned[idx, :].astype(np.float32).reshape(idx.shape[0], 1, 3),
                        keypoints[idx, :].astype(np.float32).reshape(idx.shape[0], 1, 2),
                        self.__camera_matrix,
                        self.__distortion,
                        reprojectionError=2.0)

                if (success) and (pnp_status.shape[0] >= 6):
                    # Rvec is a rodrigues vector from world to cam, so need to
                    # transpose
                    C_n_b = (cv2.Rodrigues(rvec)[0]).transpose()

                    # Then tvec is from world to cam, so need to rotate into
                    # world frame and negate
                    t_nav = -1 * np.dot(C_n_b, tvec.reshape(3, 1)).flatten()

                    t_wgs = navpy.ned2lla(t_nav, ref[1], ref[0], ref[2])
                    t_wgs = np.array(t_wgs)
                    return t_wgs, idx[pnp_status].flatten()
            else:
                print("Bailing Not Enough Matches")
        return np.array([]), np.array([])
