#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides utility functions for doing vision aided navigation
"""

import yaml
import numpy as np
import neogeodb.georeg as georeg
import mercantile
import logging
import cv2
import sys
import navpy


LOGGING_LUT = dict(debug=logging.DEBUG,
                   info=logging.INFO,
                   warn=logging.WARN,
                   error=logging.ERROR,
                   critical=logging.CRITICAL)


def bbox_from_pts(lon, lat):
    west = lon.min()
    east = lon.max()
    north = lat.max()
    south = lat.min()
    return mercantile.LngLatBbox(west, south, east, north)


def get_log(name=None, level=None, filename=None):
    """Return a file Logger if FileName is not none, else return console log
    :param name:  Name of process grabbing log
    :param level: Currently not really used?
    :param filename: Name of File to Write log, if None writes to console
    """
    if filename is None:
        log = get_console_log(name, level)
        _setup_log()

    else:
        if name is None:
            name = 'pnpnav'
        else:
            name = 'pnpnav.' + name

        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")

        # Add file and console handlers
        fhandler = logging.FileHandler(filename)
        fhandler.setLevel(logging.DEBUG)
        fhandler.setFormatter(formatter)
        try:
            chandler = logging.StreamHandler(stream=sys.stdout)
        except TypeError:  # pragma: no cover
            chandler = logging.StreamHandler(strm=sys.stdout)

        chandler.setLevel(logging.INFO)
        chandler.setFormatter(formatter)

        log.addHandler(fhandler)
        log.addHandler(chandler)

    return log



def get_console_log(name=None, level=None):
    """Return a console logger.

    Output may be sent to the logger using the `debug`, `info`, `warning`,
    `error` and `critical` methods.

    Parameters
    ----------
    name : str
        Name of the log.

    References
    ----------
    .. [1] Logging facility for Python,
           http://docs.python.org/library/logging.html

    """
    import logging

    if name is None:
        name = 'pnpnav'
    else:
        name = 'pnpnav.' + name

    log = logging.getLogger(name)
    log.setLevel(LOGGING_LUT.get(level, logging.DEBUG))

    return log


def _setup_log():
    """Configure root logger.

    """
    import logging
    import sys

    try:
        handler = logging.StreamHandler(stream=sys.stdout)
    except TypeError:  # pragma: no cover
        handler = logging.StreamHandler(strm=sys.stdout)

    handler.setLevel(logging.INFO)

    log = get_console_log()
    log.addHandler(handler)

    log.propagate = False
    log.setLevel(logging.INFO)


def find_bbox_from_tiles(tiles):
    """
    Returns a dictionary of mercantile.LngLatBbox named tuples for each
    zoom level in tiles
    """
    xyz = np.array([(t.x, t.y, t.z) for t in tiles])
    zooms = np.unique(xyz[:, 2])
    out_dict = {}
    for z in zooms:
        idx = np.where(xyz[:, 2] == z)[0]
        ul = xyz[idx, 0:2].min(0)
        ll = xyz[idx, 0:2].max(0) + 1
        a = mercantile.ul(ul[0], ul[1], z)
        b = mercantile.ul(ll[0], ll[1], z)
        out_dict[z] = mercantile.LngLatBbox(a[0], b[1], b[0], a[1])
    return out_dict


def draw_2D3DCorrespondences(matches, obs_img, tile_ophoto, r_thresh=5.0,
                             mask=None):
    """
    Takes in a pnpnav.matching.FeatureCorrespondence2D3D object, the observed \
    image, and the hdf_orthophoto to draw out the matches
    """
    lon_lat_h = matches.world_coordinates
    obs_kp = matches.keypoints
    tile_geo = np.array(tile_ophoto.tf_wgs_to_geo.TransformPoints(lon_lat_h))
    tile_pix = georeg.pix_from_geo(tile_geo[:, 0:2], tile_ophoto.gt)
    tile_img = np.copy(tile_ophoto.img)
    if tile_img.ndim > 2:
        tile_img = cv2.cvtColor(tile_img, cv2.COLOR_RGB2GRAY)
    if mask is None:
        H, mask = cv2.findHomography(
            obs_kp, tile_pix, cv2.cv.CV_RANSAC, r_thresh)
    img3 = draw_match(obs_img, tile_img, obs_kp, tile_pix, mask)
    return img3, mask


def draw_match(img1, img2, p1, p2, status=None, H=None):
    """
    This function is stolen from some older versions of find_obj.py in the
    OpenCV 2.4.x example code (for the cv2 bindings). This function is
    implemented in C++ but there's no binding as of OpenCV 2.4.9
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(
            cv2.perspectiveTransform(
                corners.reshape(
                    1,
                    -1,
                    2),
                H).reshape(
                -1,
                    2) + (
                    w1,
                0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(p1), np.bool_)
    green = (0, 255, 0)
    red = (0, 0, 255)
    for (x1, y1), (x2, y2), inlier in zip(np.int32(p1), np.int32(p2), status):
        col = [red, green][inlier]
        if inlier:
            cv2.line(vis, (x1, y1), (x2 + w1, y2), col, 7)
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2 + w1, y2), 2, col, -1)
        else:
            r = 2
            thickness = 7
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 + w1 - r, y2 - r),
                     (x2 + w1 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 + w1 - r, y2 + r),
                     (x2 + w1 + r, y2 - r), col, thickness)
    return vis


class DEMTileFinder(object):

    """
    This class uses a DEM and a Geoid shift file in order to find the terrain
    point of a local level frame under the aircraft at the terrain height.
    The camera model, position, and attitude is then used to project a ray
    from the center of the camera in this local level frame. Then the class
    solves for the point of intersection of the ray and the local level frame.
    """

    def __init__(self, terrain_path, geoid_file):
        """
        Passes in a resolvable path to the camera model, and dted path
        """
        self.terrain_handler = georeg.SRTM(terrain_path, geoid_file)

        # These are both set to identity in case you pass in camera
        # attitude. If you set these to
        self.C_b_v = np.eye(3)
        self.C_b_cam = np.eye(3)
        self.zoom = 15  # Hard coded for now

    def load_cam_and_vehicle_frames(self, frame_yaml):
        """
        Set up the internal coordinate systems from a yaml file that
        stores DCMs from the camera frame to body and from vehicle
        frame to body as (9,) flattened row-major arrays.
        """
        vehicle_frames = yaml.load(file(frame_yaml, 'r'))
        self.C_b_cam = np.array(
            vehicle_frames['/camera/image_raw']).reshape(3, 3)
        self.C_b_v = np.array(vehicle_frames['/vehicle']).reshape(3, 3)

    def load_camera_cal(self, cam_cal_path):
        """
        Loads the camera calibration file into self.K
        :param string cam_cal_path: Resolvable path to camera calibration yaml
        """
        cam_cal = yaml.load(file(cam_cal_path, 'r'))
        self.K = np.array(cam_cal['camera_matrix']['data']).reshape(3, 3)
        self.distortion = np.array(cam_cal['distortion_coefficients']['data'])
        self.cam_width = cam_cal['image_width']
        self.cam_height = cam_cal['image_height']
        self.corners_pix = np.array([[0, 0.0], [0, self.cam_height],
                                     [self.cam_width, self.cam_height],
                                     [self.cam_width, 0.0]])

    def find_center_point(self, lon_lat_h, C_n_v):
        """
        Get the world points of the 4 corners of the image and return them
        :param np.ndarray lon_lat_h: Lon, Lat, Height in (3,) in degrees
        :param np.ndarray att: (3,3) of DCM representing C_n_v
        """
        # Find point on terrain directly below lon_lat_h
        ref_lla = self.terrain_handler.add_heights(lon_lat_h[:2].reshape(1, 2))
        ref_lla = ref_lla.flatten()

        c_w_0 = navpy.lla2ned(lon_lat_h[1], lon_lat_h[0], lon_lat_h[2],
                              ref_lla[1], ref_lla[0], ref_lla[2])
        R_w_c = np.dot(C_n_v, np.dot(self.C_b_v.T, self.C_b_cam))
        n = np.array([0.0, 0.0, 1.0])
        K = self.K
        Ki = np.linalg.inv(K)
        center_pix = np.array([self.cam_width / 2.0,
                               self.cam_height / 2.0,
                               1.0])
        cvec = np.dot(Ki, center_pix)
        pclos = cvec / np.linalg.norm(cvec, axis=0)
        pwlos = np.dot(R_w_c, pclos)
        dd = (np.dot(n, c_w_0) / np.dot(n, pwlos))
        center_local = ((-1 * dd * pwlos) + c_w_0)

        # Return these for KML Style (clockwise, starting with Lower Left)
        c_wgs = navpy.ned2lla(center_local, ref_lla[1], ref_lla[0], ref_lla[2])
        c_wgs = np.array(c_wgs)
        return c_wgs[[1, 0, 2]]

    def get_corners(self, lon_lat_h, C_n_v):
        """
        Returns the corners of the image in WGS Coordinates and NED
        """
        # Find point on terrain directly below lon_lat_h
        ref_lla = self.terrain_handler.add_heights(lon_lat_h[:2].reshape(1, 2))
        ref_lla = ref_lla.flatten()
        c_w_0 = navpy.lla2ned(lon_lat_h[1], lon_lat_h[0], lon_lat_h[2],
                              ref_lla[1], ref_lla[0], ref_lla[2])
        R_w_c = np.dot(C_n_v, np.dot(self.C_b_v.T, self.C_b_cam))
        n = np.array([0.0, 0.0, 1.0])
        K = self.K
        Ki = np.linalg.inv(K)
        corners_pix = np.hstack((self.corners_pix, np.ones((4, 1)))).T
        cvec = np.dot(Ki, corners_pix)
        pclos = cvec / np.linalg.norm(cvec, axis=0)
        pwlos = np.dot(R_w_c, pclos)
        dd = (np.dot(n, c_w_0) / np.dot(n, pwlos))
        corners_local = ((-1 * dd * pwlos) + c_w_0.reshape(3, 1)).T

        # Return these for KML Style (clockwise, starting with Lower Left)
        corners_local = corners_local[[1, 2, 3, 0], :]
        out_pix = (corners_pix.T)[[1, 2, 3, 0], :]
        c_wgs = np.vstack(navpy.ned2lla(corners_local,
                                        ref_lla[1], ref_lla[0], ref_lla[2])).T
        return corners_local, c_wgs[:, [1, 0, 2]], out_pix

    def find_tile_from_pose(self, lon_lat_h, C_n_v):
        """
        This calls find_center_point and then finds the corresponding tile
        """
        t_wgs = self.find_center_point(lon_lat_h, C_n_v)
        return mercantile.tile(t_wgs[0], t_wgs[1], self.zoom)


def rpy_to_cnb(roll, pitch, yaw, units='deg'):
    if units == 'deg':
        roll = roll * (np.pi / 180.0)
        pitch = pitch * (np.pi / 180.0)
        yaw = yaw * (np.pi / 180.0)
    cph = np.cos(roll)
    sph = np.sin(roll)
    cth = np.cos(pitch)
    sth = np.sin(pitch)
    cps = np.cos(yaw)
    sps = np.sin(yaw)
    C1T = np.array([cps, -1 * sps, 0, sps, cps, 0, 0, 0, 1]).reshape(3, 3)
    C2T = np.array([cth, 0, sth, 0, 1, 0, -1 * sth, 0, cth]).reshape(3, 3)
    C3T = np.array([1, 0, 0, 0, cph, -1 * sph, 0, sph, cph]).reshape(3, 3)
    return (C1T.dot(C2T.dot(C3T)))


def DcmToRpy(dcm, units='deg'):
    """
    Converts the direction cosine matrix that rotates the reference frame to a
    new frame into roll, pitch and yaw angles through which the new frame would
    be rotated to obtain the reference frame.

    :param dcm 3x3 numpy array that rotates a vector from the body frame into
        the local level navigation (NED) frame

    :units string, if =='deg' we multiply by 180/pi to convert from radians,
        default = 'deg'

    :return rpy 3x1 numpy array containing the roll, pitch and yaw angles
        that represent the rotation of the body frame about the NED frame

    See Also: rpy_to_cnb
    """
    rpy = np.array([0.0, 0.0, 0.0])
    rpy[0] = np.arctan2(dcm[2, 1], dcm[2, 2])
    rpy[1] = np.arcsin(-1 * dcm[2, 0])
    rpy[2] = np.arctan2(dcm[1, 0], dcm[0, 0])
    if units == 'deg':
        rpy *= (180.0 / np.pi)
    return rpy


def DcmToQuat(dcm):
    """
    Converts a direction cosine matrix to its equivalent quaternion.

    :param dcm 3x3 numpy array that rotates the frame a to frame b. (unitless)

    :return q 4x1 numpy array quaternion rotation vector. (radians)

    See Also: QuatToDcm
    """
    q = np.zeros(4, )

    # Find which term will be the most numerically stable for the denominator
    a2 = (1 + dcm[0, 0] + dcm[1, 1] + dcm[2, 2])/4.0
    b2 = (1 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2])/4.0
    c2 = (1 - dcm[0, 0] + dcm[1, 1] - dcm[2, 2])/4.0
    d2 = (1 - dcm[0, 0] - dcm[1, 1] + dcm[2, 2])/4.0

    ind = np.argmax([a2, b2, c2, d2])
    if ind == 0:
        q[0] = np.sqrt(a2)
        q[1] = (dcm[2, 1] - dcm[1, 2])/(4.0 * q[0])
        q[2] = (dcm[0, 2] - dcm[2, 0])/(4.0 * q[0])
        q[3] = (dcm[1, 0] - dcm[0, 1])/(4.0 * q[0])
    elif ind == 1:
        q[1] = np.sqrt(b2)
        q[0] = (dcm[2, 1] - dcm[1, 2])/(4.0 * q[1])
        q[2] = (dcm[1, 0] + dcm[0, 1])/(4.0 * q[1])
        q[3] = (dcm[0, 2] + dcm[2, 0])/(4.0 * q[1])
    elif ind == 2:
        q[2] = np.sqrt(c2)
        q[0] = (dcm[0, 2] - dcm[2, 0])/(4.0 * q[2])
        q[1] = (dcm[1, 0] + dcm[0, 1])/(4.0 * q[2])
        q[3] = (dcm[2, 1] + dcm[1, 2])/(4.0 * q[2])
    elif ind == 3:
        q[3] = np.sqrt(d2)
        q[0] = (dcm[1, 0] - dcm[0, 1])/(4.0 * q[3])
        q[1] = (dcm[0, 2] + dcm[2, 0])/(4.0 * q[3])
        q[2] = (dcm[2, 1] + dcm[1, 2])/(4.0 * q[3])

    return q


def QuatMultiply(q1, q2):
    """
    Multiply 2 quaternions. Follows T+W equation 3.55.
    """
    q3 = np.zeros((4, ))
    q3[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    q3[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    q3[2] = q1[0]*q2[2] + q1[2]*q2[0] - q1[1]*q2[3] + q1[3]*q2[1]
    q3[3] = q1[0]*q2[3] + q1[3]*q2[0] + q1[1]*q2[2] - q1[2]*q2[1]
    return q3


def OrthoDcm(dcm, method=2):
    """
    Orthogonalize a direction cosine matrix to ensure that it is both normalized
    and symmetric about the diagonal.

    :param dcm 3x3 numpy array that rotates from frame 1 to frame 2. (float, unitless)

    :param method(optional) 0 for accurate method(default), 1 for 'fast', which
        may not actually orthogonalize.

    :return dcm 3x3 numpy array that rotates from frame 1 to frame 2, orthonormalized.
         (float, unitless)
    """
    if method == 1:
        new_dcm = np.eye(3)
        d = dcm[0, :] * dcm[1, :].T
        new_dcm[0, :] = dcm[0, :] - 0.5*d*dcm[1, :]
        new_dcm[1, :] = dcm[1, :] - 0.5*d*dcm[0, :]
        d = dcm[2, :] * dcm[1, :].T
        new_dcm[2, :] = dcm[2, :] - 0.5*d*dcm[1, :]
        new_dcm[1, :] = dcm[1, :] - 0.5*d*dcm[2, :]
        new_dcm[0, :] = new_dcm[0, :]/np.linalg.norm(new_dcm[0, :])
        new_dcm[1, :] = new_dcm[1, :]/np.linalg.norm(new_dcm[1, :])
        new_dcm[2, :] = new_dcm[2, :]/np.linalg.norm(new_dcm[2, :])
    if method == 2:
        new_dcm = np.eye(3)
        new_dcm[:, 0] = dcm[:, 0]*(1./np.linalg.norm(dcm[:, 0]))
        new_dcm[:, 1] = dcm[:, 1] - np.dot(new_dcm[:, 0], dcm[:, 1])*new_dcm[:, 0]
        new_dcm[:, 1] *= 1./np.linalg.norm(new_dcm[:, 1])
        new_dcm[:, 2] = dcm[:, 2] - np.dot(new_dcm[:, 1], dcm[:, 2])*new_dcm[:, 1] \
            - np.dot(new_dcm[:, 0], dcm[:, 2])*new_dcm[:, 0]
        new_dcm[:, 2] *= 1./np.linalg.norm(new_dcm[:, 2])
    else:
        [new_dcm, r] = np.linalg.qr(dcm)
        r = np.diag(r)
        for ind in range(3):
            if r[ind] < 0:
                new_dcm[:, ind] *= -1
    return new_dcm


def llh_to_cen(llh):
    """
    Calculates the direction cosine matrix that rotates the local-level
    navigation frame (North, East, Down axes) into the Earth-centered,
    Earth-fixed frame. The ECEF frame is defined as having the x-axis
    passing through the intersection of the equator and the Greenwich
    meridian, the z axis through the ITRF North pole, and the y-axis
    completing the right-handed frame.

    :param llh 3x1 numpy array of geodetic latitude, longitude and altitude
         with respect to the WGS-84 ellipsoid. (radians, radians, meters)

    :return Cen 3x3 numpy array which rotates local-level navigation frame
        (North, East, Down axes) into the ECEF frame. (unitless)

    See Also: EcefToCen
    """
    c_lat = np.cos(llh[0])
    s_lat = np.sin(llh[0])
    c_lon = np.cos(llh[1])
    s_lon = np.sin(llh[1])

    c1 = np.array([[0, 0, 1],
                [-s_lon, c_lon, 0],
                [-c_lon, -s_lon, 0]])

    c2 = np.array([[c_lat, 0, s_lat],
                [0, 1, 0],
                [-s_lat, 0, c_lat]])

    return np.dot(c2, c1).T
