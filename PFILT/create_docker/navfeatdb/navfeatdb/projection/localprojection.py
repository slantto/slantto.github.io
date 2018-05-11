import numpy as np
import yaml
from ..frames import terrain
import navpy
import mercantile


class LocalLevelProjector(object):
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
        :param terrain_path: Resolvable path to a directory containing
            GDAL readable SRTM tiles (*.hgt files)
        :param geoid_file: string, path to a GDAL readable geoid-shift file.
            This file (e.g. EGM96.gtx) stores the undulation, or difference
            between the geoid and ellipsoid heights
        :return None. Sets the internal self.terrain_handler and initial
            extrinsics and intrinsics.
        """
        self.terrain_handler = terrain.SRTM(terrain_path, geoid_file)

        # These are both set to identity in case you pass in camera
        # attitude. If you set these to
        self.C_b_v = np.eye(3)
        self.C_b_cam = np.eye(3)
        self.zoom = 15  # Hard coded for now
        self.K = None
        self.distortion = None
        self.cam_width = None
        self.cam_height = None
        self.corners_pix = None

    def load_cam_and_vehicle_frames(self, frame_yaml):
        """
        Set up the internal coordinate systems from a yaml file that
        stores DCMs from the camera frame to body and from vehicle
        frame to body as (9,) flattened row-major arrays.
        :param frame_yaml: Path to a yaml file that describes the relationship
            of the camera and vehicle frames to the body frame
        """
        with open(frame_yaml, 'r') as stream:
            vehicle_frames = yaml.load(stream)
            self.C_b_cam = np.array(
                vehicle_frames['/camera/image_raw']).reshape(3, 3)
            self.C_b_v = np.array(vehicle_frames['/vehicle']).reshape(3, 3)

    def load_camera_cal(self, cam_cal_path):
        """
        Loads the camera calibration file into self.K
        :param string cam_cal_path: Resolvable path to camera calibration yaml
        """
        with open(cam_cal_path, 'r') as stream: 
            cam_cal = yaml.load(stream)
        self.K = np.array(cam_cal['camera_matrix']['data']).reshape(3, 3)
        self.distortion = np.array(cam_cal['distortion_coefficients']['data'])
        self.cam_width = cam_cal['image_width']
        self.cam_height = cam_cal['image_height']
        self.corners_pix = np.array([[0, 0.0], [0, self.cam_height],
                                     [self.cam_width, self.cam_height],
                                     [self.cam_width, 0.0]])

    def project_points(self, lon_lat_h, C_n_v, points, return_local=False):
        """
        This is the key Function to project points from the image frame into
        the Local Level (NED) frame, and then convert into WGS-84. The origin
        of the local level (NED) frame is located as the (Lon, Lat) of the
        aircraft, with the height being located at h = Dem(lon, lat) (eg,
        terrain height). This function computes the points by solving for
        the ray-local level plane intersection
        :param lon_lat_h: [3,] Position of the camera in Lon, Lat (deg),
            height (m)
        :param C_n_v: [3x3] np.ndarray that translates a vector in the vehicle
            frame into the local-level NED frame
        :param points: [Nx2] [np.ndarray] of points to project
        :return: Returns an [Nx3] np.ndarray of Lon, Lat, Height per point in
            [points]
        """
        if self.K is None:
            raise ValueError("Camera Model Not Initialized")

        if points.ndim == 1:
            points = points.reshape(1, 2)

        ref_lla = self.terrain_handler.add_heights(lon_lat_h[:2].reshape(1, 2))
        ref_lla = ref_lla.flatten()
        c_w_0 = navpy.lla2ned(lon_lat_h[1], lon_lat_h[0], lon_lat_h[2],
                              ref_lla[1], ref_lla[0], ref_lla[2])
        R_w_c = np.dot(C_n_v, np.dot(self.C_b_v.T, self.C_b_cam))
        n = np.array([0.0, 0.0, 1.0])
        K = self.K
        Ki = np.linalg.inv(K)

        pix = np.hstack((points, np.ones((points.shape[0], 1)))).T
        cvec = np.dot(Ki, pix)
        pclos = cvec / np.linalg.norm(cvec, axis=0)
        pwlos = np.dot(R_w_c, pclos)
        dd = (np.dot(n, c_w_0) / np.dot(n, pwlos))
        points_local = ((-1 * dd * pwlos) + c_w_0.reshape(3, 1)).T

        c_wgs = np.vstack(navpy.ned2lla(points_local,
                                        ref_lla[1], ref_lla[0], ref_lla[2])).T
        if return_local:
            return c_wgs[:, [1, 0, 2]], points_local
        else:
            return c_wgs[:, [1, 0, 2]]

    def project_center(self, lon_lat_h, C_n_v, return_local=False):
        """
        Convenience function to project the center point of the image into
        WGS-84 frame

        :param lon_lat_h: [3,] Position of the camera: Lon, Lat (deg),
            height (m)
        :param C_n_v: [3x3] np.ndarray that translates a vector in the vehicle
            frame into the local-level NED frame
        :return: Returns an [1x3] np.ndarray of Lon, Lat, Height for the center
        """
        center_pix = np.array([self.cam_width / 2.0, self.cam_height / 2.0])
        return self.project_points(lon_lat_h, C_n_v, center_pix, return_local)

    def project_corners(self, lon_lat_h, C_n_v, return_local=False):
        """
        This function projects the corners of the image, calculated by the
        camera model, into the world frame, using self.project_points

        :param lon_lat_h: [3,] Position of the camera: Lon, Lat (deg), \
            height (m)
        :param C_n_v: [3x3] np.ndarray that translates a vector in the vehicle
            frame into the local-level NED frame
        :return: Returns an [4x3] np.ndarray of Lon, Lat, Height for the corners
            of the image. Going counter clockwise starting from the lower-left
        """
        if return_local:
            c_wgs, c_local = self.project_points(lon_lat_h, C_n_v, self.corners_pix, return_local=True)
            return c_wgs[[1, 2, 3, 0], :], c_local[[1, 2, 3, 0], :]
        else:
            c_wgs = self.project_points(lon_lat_h, C_n_v, self.corners_pix)
            return c_wgs[[1, 2, 3, 0], :]

    def find_tile_from_pose(self, lon_lat_h, C_n_v):
        """
        This calls find_center_point and then finds the corresponding tile
        """
        t_wgs = self.project_center(lon_lat_h, C_n_v)[0]
        return mercantile.tile(t_wgs[0], t_wgs[1], self.zoom)

    def get_pix_size(self, lon_lat_h, C_n_v):
        """
        Given the current pose of the aircraft, calculate the GSD for each edge
        of the image
        :param lon_lat_h: [3,] Position of the camera: Lon, Lat (deg), \
            height (m)
        :param C_n_v: [3x3] np.ndarray that translates a vector in the vehicle
            frame into the local-level NED frame
        :return: Returns an [4,] np.ndarray of the pixel size (m/pixel) for each
            of the 4 edges of the image. (bottom, right, top, left)
        """
        cwgs = self.project_corners(lon_lat_h, C_n_v)
        cref = cwgs[0, :]
        corners_ned = navpy.lla2ned(cwgs[:, 1], cwgs[:, 0], cwgs[:, 2],
                                    cref[1], cref[0], cref[2])
        corners_ned = np.vstack((corners_ned, corners_ned[0, :]))
        dist = np.linalg.norm(np.diff(corners_ned[:, 0:2], axis=0), axis=1)
        pix = np.tile([self.cam_width, self.cam_height],  2)
        return dist / pix

    def get_bounding_tile(self, lon_lat_h, C_n_v):
        """
        Calculates the corner points of the image in WGS-84, and returns the
        WMS tile z/x/y that completely bounds the image
        :param lon_lat_h: [3,] Position of the camera: Lon, Lat (deg), \
            height (m)
        :param C_n_v: [3x3] np.ndarray that translates a vector in the vehicle
            frame into the local-level NED frame
        :return: mercantile.Tile that completely bounds the projected image
        """
        cwgs = self.project_corners(lon_lat_h, C_n_v)
        b_tile = mercantile.bounding_tile(cwgs.min(0)[0], cwgs.min(0)[1],
                                          cwgs.max(0)[0], cwgs.max(0)[1])
        return b_tile
