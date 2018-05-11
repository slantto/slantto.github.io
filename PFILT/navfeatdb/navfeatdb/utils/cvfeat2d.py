#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides wrapper/helper functions to interact with OpenCV's \
features2d module, providing commonly used functionality image feature
processing.
"""

import numpy as np
import cv2
import yaml
import os

__CV2_FT_DESCRIPTORS__ = ["SIFT", "SURF", "BRIEF", "BRISK", "ORB", "FREAK"]

__CV2_FT_DETECTORS__ = ["FAST", "STAR", "SIFT", "SURF", "ORB", "BRISK", "MSER",
                        "GFTT", "HARRIS", "Dense", "SimpleBlob"]


def set_feat2d_param(feat2d_obj, param_key, value):
    """
    This function uses the C++ wrapper code to determine which setter \
    function is appropriate to use for setting the parameter specified by \
    param_key in the feat2d_obj. If the type of the parameter is not \
    detectable or supported by the python wrappers, a ValueError is raised

    :param FeatureDetector/Descriptor feat2d_obj: features2d object to set \
        parameter
    :param string param_key: key of parameter to set
    :param any value: value to set feat2d_obj['param_key']
    """
    fd_params = feat2d_obj.getParams()
    if not param_key in fd_params:
        raise ValueError("%s is not a valid parameter in feat2d_obj"
                         % param_key)

    param_type = feat2d_obj.paramType(param_key)
    if param_type == 0:
        feat2d_obj.setInt(param_key, value)
    elif param_type == 1:
        feat2d_obj.setBool(param_key, value)
    elif param_type == 2:
        feat2d_obj.setDouble(param_key, value)


def get_feat2d_param(feat2d_obj, param_key):
    """
    This function decodes which getter function to use for accessing the \
    parameter of feat2d_obj associated with param_key. The python wrappers\
    require knowledge of the parameter data-type in order to call the correct\
    getter functions implemented in OpenCV

    :param FeatureDetector/Descriptor feat2d_obj: features2d object to get \
        parameter
    :param string param_key: key of parameter to set
    :returns: returns the value of the parameter pointed to by param_key
    """
    fd_params = feat2d_obj.getParams()
    if not param_key in fd_params:
        raise ValueError(
            "%s is not a valid parameter in feat2d_obj" %
            param_key)
    param_type = feat2d_obj.paramType(param_key)
    if param_type == 0:
        return feat2d_obj.getInt(param_key)
    elif param_type == 1:
        return feat2d_obj.getBool(param_key)
    elif param_type == 2:
        return feat2d_obj.getDouble(param_key)


def create_feature_detector_from_file(feature_file):
    """
    This function creates a feature detector and configures it based on the \
    parameters defined in the file (yaml)

    :param string feature_file: full path to a yaml formatted feature \
        extractor parameter file
    :returns: cv2.FeatureDetector -- OpenCV features2d Common Interface \
        Feature Detector
    """
    # Load in the feature detector description
    detector_yaml = yaml.load(file(feature_file))
    return create_feature_detector_from_dict(detector_yaml)


def create_feature_descriptor_from_file(feature_file):
    """
    This function creates a feature descriptor and configures it based on the \
    parameters defined in the file (yaml)

    :param string feature_file: full path to a yaml formatted feature \
        extractor parameter file
    :returns: cv2.DescriptorExtractor -- OpenCV features2d Common Interface \
        Feature descriptor
    """

    # Load in the feature descriptor description
    descriptor_yaml = yaml.load(file(feature_file))
    return create_feature_descriptor_from_dict(descriptor_yaml)


def create_feature_detector_from_dict(detector_yaml):
    """
    This function creates a feature detector and configures it based on the \
    parameters defined in the yaml-derived dictionary

    :param dict detector_yaml: dictionary derived from a yaml formatted \
        feature extractor parameter file
    :returns: cv2.FeatureDetector -- OpenCV features2d Common Interface \
        Feature Detector
    """

    global __CV2_FT_DETECTORS__

    # First check to see if we have a name defined, and that it matches one
    # of OpenCV's supported feature detectors
    if not 'detector' in detector_yaml:
        raise ValueError("feature_file does not contain key 'detector'")

    if not 'name' in detector_yaml['detector']:
        raise ValueError("Name of feature detector is not provided in \
            feature_file")

    if not detector_yaml['detector']['name'] in __CV2_FT_DETECTORS__:
        raise ValueError("Feature detection algorithm specified in \
            feature_file is not supported by OpenCV's feature2d module")

    # Now go ahead and create it
    feature_detector = cv2.FeatureDetector_create(
        detector_yaml['detector']['name'])

    # Do error checking on the parameter list:
    if not 'parameters' in detector_yaml['detector']:
        print( "No parameters specified for detector in feature_file, \
                    using defaults")

    # Check to make sure you don't have an empty parameter list, then go for it
    if not detector_yaml['detector']['parameters'] is None:
        for file_param in detector_yaml['detector']['parameters']:
            set_feat2d_param(feature_detector,
                             file_param.keys()[0],
                             file_param[file_param.keys()[0]])

    return feature_detector


def create_feature_descriptor_from_dict(descriptor_yaml):
    """
    This function creates a feature descriptor and configures it based on the \
    parameters defined in the yaml-derived dictionary

    :param dict descriptor_yaml: dictionary derived from a yaml formatted \
        feature extractor parameter file
    :returns: cv2.DescriptorExtractor -- OpenCV features2d Common Interface \
        Feature descriptor
    """

    global __CV2_FT_DESCRIPTORS__

    # First check to see if we have a name defined, and that it matches one
    # of OpenCV's supported feature descriptors
    if not 'descriptor' in descriptor_yaml:
        raise ValueError("feature_file does not contain key 'descriptor'")

    if not 'name' in descriptor_yaml['descriptor']:
        raise ValueError("Name of feature descriptor is not provided in \
            feature_file")

    if not descriptor_yaml['descriptor']['name'] in __CV2_FT_DESCRIPTORS__:
        raise ValueError("Feature descriptor algorithm specified in \
            feature_file is not supported by OpenCV's feature2d module")

    # Now go ahead and create it
    feature_descriptor = cv2.DescriptorExtractor_create(
        descriptor_yaml['descriptor']['name'])

    # Do error checking on the parameter list:
    if not 'parameters' in descriptor_yaml['descriptor']:
        print( "No parameters specified for descriptor in feature_file, \
                    using defaults")

    # Check to make sure you don't have an empty parameter list, then go for it
    if not descriptor_yaml['descriptor']['parameters'] is None:
        for file_param in descriptor_yaml['descriptor']['parameters']:
            set_feat2d_param(feature_descriptor,
                             file_param.keys()[0],
                             file_param[file_param.keys()[0]])

    return feature_descriptor


def extract_features(image, detector, desc_extract):
    """
    Takes in the image, and the feature detector and descriptor extractor \
    files, returns a tuple of keypoints and descriptors

    :param numpy.ndarray image: NxMx1 uint8 image in which to extract \
        features and compute feature descriptors
    :param cv2.FeatureDetector detector: openCV FeatureDetector object created\
        by feature2d helper functions
    :param cv2.DescriptorExtractor desc_extract: openCV DescriptorExtractor \
        object created by feature2d helper functions
    :returns: tuple -- (keypoints, descriptors), keypoints is a list of \
        cv2.KeyPoint objects with len() == N, and descriptors is a \
        numpy.ndarray of shape (N,M), whose nth row corresponds to the \
        the nth keypoint in keypoints
    """
    # Check for 8 bit
    if not image.dtype == np.dtype('uint8'):
        raise ValueError('image must be uint8 for most feature functionality')
    # And Grayscale
    if len(image.shape) > 2:
        raise ValueError('image must be grayscale, or 2-dimensional, current \
                         image has dimensions %s ' % (image.shape,))

    kp = detector.detect(image)
    kp, desc = desc_extract.compute(image, kp)
    return (kp, desc)


def keypoint_pixels(kp):
    """
    Returns an nd array of feature pixel (xy) locations in raster space given \
    a list of OpenCV Keypoint objects

    :param list kp: List of OpenCV cv2.Keypoint objects
    :returns: numpy.ndarray -- numpy.ndarray of size Nx2, where N = len(kp) \
        such that each row of the returned array contains the point.pt (x,y) \
        pixel coordinates for each point in kp
    """
    # Get the features
    kp_xy = np.zeros((len(kp), 2))
    for ii in np.arange(len(kp)):
        kp_xy[ii, :] = np.array(kp[ii].pt)
    return kp_xy


def load_feature_operators(detector_file, descriptor_file):
    """
    This function loads \
    a pair of commonly coupled image feature operators \
    consisting of a feature detector object and a keypoint descriptor object \
    through providing full paths to their corresponding parameter yaml files

    :param string detector_file: path to yaml file used to define the desired \
        feature detector and associated parameters
    :param string descriptor_file: path to yaml file used to define the \
        desired feature descriptor and associated parameters
    :returns: tuple -- (cv2.FeatureDetector, cv2.DescriptorExtractor) created \
        by the feature2d create_feature_XX_from_file() functions
    """
    detector = create_feature_detector_from_file(detector_file)
    desc_extract = create_feature_descriptor_from_file(descriptor_file)
    return(detector, desc_extract)


def build_parameter_dictionary(feat2d_obj):
    """
    This function builds a list of dictionaries of the same format used in \
    the feature_files used in this module, by self introspection of a \
    cv2.FeatureDetector() or cv2.FeatureDescriptor() object. As of now there's\
    no introspection available except to look at parameters.

    :param cv2.Feature2D Object feat2d: feat2d object whose parameters will \
        be output to the list of dictionaries
    :returns: list(dict{}) -- List of key/value pairs of feat2d_obj parameters\
        embeddable into feature_file type yaml descriptions
    """
    fd_params = feat2d_obj.getParams()
    return [{key: get_feat2d_param(feat2d_obj, key)} for key in fd_params]


def get_subimages_by_max_dimension(max_dim, image_x, image_y):
    """
    Takes max dimension and image_dim, and returns the image coordinates \
    of the upper-left corner or each sub-image, along with the run for each \
    individual axis.

    :param int max_dim: Maximum size of either dimension of the image
    :param int image_x: Size of the Image X dimension (numpy Y/col dimension)
    :param int image_y: Size of the Image Y dimension (numpy X/row dimension)
    :returns: tuple -- tuple of dictionaries, each element corresponding to\
        a sub image, and including x/y offset and run such that each image \
        can be addressed by (in numpy): \
        img[y_offset:y_offset+run, x_offset:x_offset+run], transpose for raster
    """
    num_x_blocks = int(np.ceil(image_x / float(max_dim)))
    x_block_size = int(np.floor(image_x / num_x_blocks))

    num_y_blocks = int(np.ceil(image_y / float(max_dim)))
    y_block_size = int(np.floor(image_y / num_y_blocks))

    out_list = []
    for n_x in np.arange(num_x_blocks):
        for n_y in np.arange(num_y_blocks):
            x_start = n_x * x_block_size
            if n_x == (num_x_blocks - 1):
                x_run = image_x - x_start
            else:
                x_run = x_block_size
            y_start = n_y * y_block_size
            if n_y == (num_y_blocks - 1):
                y_run = image_y - y_start
            else:
                y_run = y_block_size
            out_list.append({'x_start': x_start, 'x_run': x_run,
                             'y_start': y_start, 'y_run': y_run})
    return out_list


def create_sample_default_detectors(yaml_dir):
    """
    This function creates sample feature yaml files in the directory \
    specified by yaml_dir to provide coverage of all the supported feature \
    detectors in opencv's feat2d module and their parameters

    :param string yaml_dir: full path to write sample feature yaml files
    :returns: None
    """

    global __CV2_FT_DETECTORS__
    for detector in __CV2_FT_DETECTORS__:
        fd = cv2.FeatureDetector_create(detector)
        out_yaml = {'detector':
                    {'name': detector,
                     'parameters': build_parameter_dictionary(fd)}}
        out_name = os.path.join(yaml_dir +
                                'default_detector_%s.yaml' % detector)
        outfile = file(out_name, 'w')
        yaml.dump(out_yaml, stream=outfile)
        outfile.close()


def create_sample_default_descriptors(yaml_dir):
    """
    This function creates sample feature yaml files in the directory \
    specified by yaml_dir to provide coverage of all the supported feature \
    descriptor extractors in opencv's feat2d module and their parameters

    :param string yaml_dir: full path to write sample feature yaml files
    :returns: None
    """

    global __CV2_FT_DESCRIPTORS__
    for extractor in __CV2_FT_DESCRIPTORS__:
        fd = cv2.DescriptorExtractor_create(extractor)
        out_yaml = {'descriptor':
                    {'name': extractor,
                     'parameters': build_parameter_dictionary(fd)}}
        out_name = os.path.join(yaml_dir +
                                'default_descriptor_%s.yaml' % extractor)
        outfile = file(out_name, 'w')
        yaml.dump(out_yaml, stream=outfile)
        outfile.close()

def unpackSIFTOctave(kpt):
    """
    For any SIFT Keypoint in KPT, unpack the scale parameters from
    the packed 16 bit field
    """
    octave = kpt.octave & 255
    layer = (kpt.octave >> 8) & 255
    if octave >= 128:
        octave = (-128 | octave)
    if octave >= 0:
        scale = 1.0 / (1 << octave)
    else:
        scale = (float)(1 << -octave)
    return np.array([octave, layer, scale])

def numpySIFTScale(kpts):
    return np.array([unpackSIFTOctave(kp) for kp in kpts])





