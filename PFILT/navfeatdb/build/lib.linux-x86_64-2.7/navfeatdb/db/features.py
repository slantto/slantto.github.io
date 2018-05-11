import pandas as pd
import numpy as np
from ..utils import cvfeat2d as f2d
import cv2
import navpy


class ImageFeatureExtractor(object):
    """
    This is the base class for objects that take a descriptor and extractor
    and return geo-located features from an image. Expected subclasses will
    be AirborneImageExtractor and OrthophotoImageExtractor
    """

    def __init__(self, feature_detector, descriptor_extractor):
        """
        Base class constructor. Sets the internal member variables to use
        the base feature_detector and the descriptor_extractor internally
        :param feature_detector: utils.cvfeat2d feature detector object
        :param descriptor_extractor: utils.cvfeat2d feature_descriptor object
        :return: None
        """
        self._detector = feature_detector
        self._desc_extractor = descriptor_extractor

    def extract_features(self):
        raise NotImplementedError("This class should not be called directly")


class AirborneImageExtractor(ImageFeatureExtractor):
    """
    This class implements ImageFeatureExtractor for the airborne image case.
    Airborne images need a terrain projector along with the current pose
    of the aircraft in order to geolocate features
    """
    
    def __init__(self, feature_detector, descriptor_extractor, projector):
        """
        Class constructor
        :param feature_detector: utils.cvfeat2d feature detector object
        :param descriptor_extractor: utils.cvfeat2d feature_descriptor object
        :param ll_projector: navfeatdb.projection projector 
        :return: None
        """
        super(AirborneImageExtractor, self).__init__(feature_detector,
                                                     descriptor_extractor)
        self._projector = projector

    def extract_features(self, image, lon_lat_h, C_n_v):
        """
        Returns a pandas dataframe of features, along with an array of
        descriptors, whos row indices are correlated.
        :param image: Image suitable for feature extraction
        :param lon_lat_h: Lon, Lat (deg) and height (m) of the camera
        :param C_n_v: 3x3 DCM that rotates a vector expressed in the vehicle
            frame into the NED local level frame
        :return: Pandas Dataframe of feature data, a numpy array of the \
            descriptors
        """
        kp, desc = f2d.extract_features(image, self._detector, self._desc_extractor)
        if len(kp) > 0:
            pix = f2d.keypoint_pixels(kp)
            meta = np.array([(pt.angle, pt.response, pt.size) for pt in kp])
            packed = f2d.numpySIFTScale(kp)

            img_df = pd.DataFrame(np.hstack((pix, meta, packed)),
                                  columns=['pix_x', 'pix_y', 'angle',
                                           'response', 'size', 'octave',
                                           'layer', 'scale'])

            pts_wgs = self._projector.project_points(lon_lat_h, C_n_v, pix)
            b_tile = self._projector.get_bounding_tile(lon_lat_h, C_n_v)
            gsd = self._projector.get_pix_size(lon_lat_h, C_n_v)
            df = pd.concat([img_df, pd.DataFrame(pts_wgs, columns=['lon', 'lat', 'height'])], axis=1)
            df['bound_x'] = b_tile.x
            df['bound_y'] = b_tile.y
            df['bound_z'] = b_tile.z
            df['gsd'] = gsd.mean()

            # Sort by feature response
            df.sort_values('response', ascending=False, inplace=True)
            pp = np.copy(df.index)
            df.index = np.arange(df.shape[0])

            return df, desc[pp, :]
        else:
            return None, None


class OrthoPhotoExtractor(ImageFeatureExtractor):
    """
    This class implements ImageFeatureExtractor for OrthoPhotos
    """
    def __init__(self, feature_detector, descriptor_extractor, orthophoto, terrain_handler):
        """
        Class constructor
        :param feature_detector: utils.cvfeat2d feature detector object
        :param descriptor_extractor: utils.cvfeat2d feature_descriptor object
        :param orthophoto: navfeatdb.orthophoto OrthoPhoto Object
        :param terrain_handler: navfeatdb.frames TerrainHandler object used to put WGS-84 heights on orhtophoto points
        :return: None
        """
        super(OrthoPhotoExtractor, self).__init__(feature_detector, descriptor_extractor)
        self._ophoto = orthophoto
        self._terrain_handler = terrain_handler

    def features_from_slice(self, slc, bias=None, convert_to_gray=True, color_code=cv2.COLOR_RGBA2GRAY):
        """
        Takes in an orthophoto, slice tuple, detector, and extractor
        """
        sub_photo = self._ophoto.get_img_from_slice(slc)

        if (sub_photo.img.ndim > 2) and convert_to_gray:
            sub_img = cv2.cvtColor(sub_photo.img, color_code)
        elif (sub_photo.img.ndim > 2):
            raise TypeError("Underlying image is greater than 2 dimensions and convert_to_gray is set to false")
        else:
            sub_img = sub_photo.img

        kp, desc = f2d.extract_features(sub_img, self._detector, self._desc_extractor)
        if len(kp) > 0:
            pix = f2d.keypoint_pixels(kp)
            meta = np.array([(pt.angle, pt.response, pt.size) for pt in kp])
            packed = f2d.numpySIFTScale(kp)

            img_df = pd.DataFrame(np.hstack((pix, meta, packed)),
                                  columns=['pix_x', 'pix_y', 'angle',
                                           'response', 'size', 'octave',
                                           'layer', 'scale'])

            pts_lon_lat = sub_photo.pix_to_wgs84(pix)
            pts_wgs = self._terrain_handler.add_heights(pts_lon_lat)
            if bias is not None:
                print(bias)
                ref = pts_wgs[0, [1, 0, 2]]
                pts_ned = navpy.lla2ned(pts_wgs[:, 1], pts_wgs[:, 0], pts_wgs[:, 2], *ref)
                pts_ned -= bias
                pts_wgs = np.vstack(navpy.ned2lla(pts_ned, *ref)).T
                pts_wgs = pts_wgs[:, [1, 0, 2]]

            df = pd.concat([img_df, pd.DataFrame(pts_wgs, columns=['lon', 'lat', 'height'])], axis=1)

            # Sort by feature response
            df.sort_values('response', ascending=False, inplace=True)
            pp = np.copy(df.index)
            df.index = np.arange(df.shape[0])
            return df, desc[pp, :]
        else:
            return None, None