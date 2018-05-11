import numpy as np


def rpy_to_cnb(roll, pitch, yaw, units='deg'):
    """
    Returns a 3x3 DCM that tranforms a vector from the body frame into the
    Local Level NED frame
    :param roll: Right handed angle between about X and NED
    :param pitch: Right handed angle between about Y and NED
    :param yaw: Right handed angle between about Z and NED
    :param units: 'deg' or else
    :return: 3x3 Numpy DCM
    """
    if units == 'deg':
        roll  *= (np.pi / 180.0)
        pitch *= (np.pi / 180.0)
        yaw   *= (np.pi / 180.0)

    cph = np.cos(roll)
    sph = np.sin(roll)
    cth = np.cos(pitch)
    sth = np.sin(pitch)
    cps = np.cos(yaw)
    sps = np.sin(yaw)
    C1T = np.array([cps, -1 * sps, 0, sps, cps, 0, 0, 0, 1]).reshape(3, 3)
    C2T = np.array([cth, 0, sth, 0, 1, 0, -1 * sth, 0, cth]).reshape(3, 3)
    C3T = np.array([1, 0, 0, 0, cph, -1 * sph, 0, sph, cph]).reshape(3, 3)
    return C1T.dot(C2T.dot(C3T))


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