#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This program calculates delta position between images in order to implement
the motion model for neogeo
"""

import pnpnav.utils as pnputils
import neogeodb.pytables_db as pdb
import neogeo.extent as neoextent
import numpy as np
import tables as tb
import mercantile
import time
import navpy
from scipy.interpolate import interp1d


if __name__ == '__main__':
    f5 = tb.open_file('/Users/venabled/catkin_ws/data/fc2/fc2_f5.hdf', 'r')
    imgs = f5.root.images.image_data
    img_times = f5.root.images.t_valid
    feat_tables = f5.list_nodes(f5.root.images.sift_features)
    pva = f5.root.pva

    pos_interp = interp1d(pva.t_valid[:], pva.pos[:].T)
    pos_img = pos_interp(img_times).T
    ned_at_img = navpy.lla2ned(pos_img[:, 1], pos_img[:, 0], pos_img[:, 2],
                               pos_img[0, 1], pos_img[0, 0], pos_img[0, 2])
    ned_vel = np.vstack((np.zeros((1, 3)), np.diff(ned_at_img, axis=0)))

    ii = 0
    box_size = np.zeros((4096, 2))
    for xid in np.arange(xb[0], xb[1] + 1):
        for yid in np.arange(yb[0], yb[1] + 1):
            bbox = mercantile.bounds(xid, yid, 15)
            ned_box = navpy.lla2ned(
                bbox.north, bbox.west, 1270.0, bbox.south, bbox.east, 1270.0)
            box_size[ii, :] = np.copy(ned_box[0:2])
            ii += 1
