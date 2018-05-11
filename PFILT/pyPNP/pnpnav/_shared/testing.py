"""Testing Utilities."""
import numpy as np
import navpy
import mercantile
import matplotlib.pyplot as plt
import pnpnav.matching as matching


def generate_cam_data(fm=0.08, image_x=1024, image_y=1024, pix_size=4.65e-06):
    """
    Builds a camera matrix using image_x and image_y as the camera width and \
    height. Focal len in mm, and pixel size in meters.
    """
    alpha = fm / pix_size  # focal_len_pix

    # Camera Matrix (pinhole)
    K = np.array([[alpha, 0.0, image_x / 2],
                  [0.0, alpha, image_y / 2],
                  [0.0, 0.0, 1.0]])
    distortion = np.zeros(5)
    return K, distortion


def find_optimal_altitude(ground_points, K, max_image_dim):
    """
    Assuming that the camera is looking nadir toward a local level plane, \
    finds the altitude that contains the points located within ground_points
    """
    mean_gp = (ground_points[:, 0:2]).mean(0)
    mg = np.abs(ground_points[:, 0:2] - mean_gp).max()
    angle = np.arctan(max_image_dim / 2.0 / K[0, 0])
    alt = np.ceil(mg / np.tan(angle))

    # Go to the nearest 10m multiple just to be safe
    return alt + (10 - np.mod(alt, 10))


def generate_nadir_data(alt, num_samples=0, att_sigma=0.0):
    """
    Given a numpy ndarray of Nx3 local level ground points, returns
    the camera metadata, true pose, index into ground_points for the
    M ground_points used, and an Mx2 numpy.ndarray of feature locations
    in the image. Returns the translation of the camera center in
    ENU coordinates, and the rotation from World to Cam
    """

    # Generate a nominal camera pose
    R_nadir = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    roll = att_sigma * np.random.standard_normal() * (np.pi / 180.0)
    pitch = att_sigma * np.random.standard_normal() * (np.pi / 180.0)
    yaw = att_sigma * np.random.standard_normal() * (np.pi / 180.0)
    R_c_nadir = navpy.angle2dcm(yaw, pitch, roll, rotation_sequence='ZYX')
    R_c_w = np.dot(R_c_nadir, R_nadir)

    # t_w - translation between world and camera frame center,
    # expressed in world coordinates
    east_m = 0.0
    north_m = 0.0
    t_w = np.array([east_m, north_m, alt])
    return t_w, R_c_w


def subsample_ground_points(ground_points):
    """
    This returns indices into ground points that create a subsampling
    of a circle centered @ circle_center, and the North East Quadrant
    (assuming you pass in NED, this will be the lower right).
    """

    idx = np.where(ground_points[:, 0] > 0)[0]
    idx = np.intersect1d(idx, np.where(ground_points[:, 1] < 0)[0])

    f2 = ((ground_points[:, 0] + 220) **
          2 + (ground_points[:, 1] - 220.0)**2)**(0.5)
    idx = np.union1d(idx, np.where(f2 < 150.0)[0])
    return idx


def generate_ground_points(zoom, num_pts, vert_sigma=10.0):
    """
    This function creates a uniformly sampled grid of points across a single
    Mercantile tile. Uses numpy.random.uniform to pick the 2D locations and
    then assigns a height value based on vert_sigma. Returns 2 Nx3 ndarrays,
    one of lon_lat_height, and the second is the NED locations in the tile.
    Finally we return the Lon, Lat (deg), and HAE of the center point of
    the tile used as the origin of the local level NED frame
    """
    wpafb_lon = -84.049954
    wpafb_lat = 39.8179055
    tile = mercantile.tile(wpafb_lon, wpafb_lat, zoom)
    bounds = mercantile.bounds(tile.x, tile.y, tile.z)
    lons = np.random.uniform(bounds.west, bounds.east, num_pts)
    lats = np.random.uniform(bounds.south, bounds.north, num_pts)
    heights = np.random.standard_normal(num_pts) * vert_sigma
    center = np.array([(bounds.west + bounds.east) / 2,
                       (bounds.north + bounds.south) / 2,
                       0.0])
    pts_ned = navpy.lla2ned(lats, lons, heights, center[1], center[0], 0.0)
    lon_lat_height = np.vstack((lons, lats, heights)).T
    return lon_lat_height, pts_ned, center


def project_ground_points(ground_points, K, R_c_w, t_w):
    """
    Takes an Nx3 numpy.ndarray of East, North, UP ground points,
    a camera calibration matrix, the image_y size, Rotation from ENU to cam
    and the camera position in the ENU world frame
    """
    # Rotate and translate these points
    x0 = np.dot(R_c_w, ground_points.T)
    tc = np.dot(R_c_w, t_w).T
    x_img = np.dot(K, (x0.T - tc).T)
    x_img = (x_img[0:2, :] / x_img[2]).T
    return x_img

def build_matches(lon_lat_h, img_pts):
    """
    Builds the match Class that PnP uses as input
    """
    matches = matching.FeatureCorrespondence2D3D()
    matches.keypoints = img_pts
    matches.world_coordinates = lon_lat_h
    matches.num_correspondences = img_pts.shape[0]
    return matches


def plot_output(ground_points, img_pts, idx, image_x, image_y):
    """
    Ground points are NED \
    img_pts are in opencv camera frame (x = right, y = down)
    """
    plt.plot(ground_points[:, 1], ground_points[:, 0], 'gx')
    plt.plot(ground_points[idx, 1], ground_points[idx, 0], 'bo')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.axis('equal')
    plt.title('Ground Sample')

    plt.figure()
    plt.plot(img_pts[:, 0], img_pts[:, 1], 'rx')
    plt.plot(img_pts[idx, 0], img_pts[idx, 1], 'bo')
    plt.xlim(0, image_x)
    plt.ylim(0, image_y)
    plt.title('Camera Image of Points')
    plt.axis('equal')
    plt.show()