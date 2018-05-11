#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:08:50 2017

@author: sean
"""

import pnpnav.utils as pnputils
import neogeodb.pytables_db as pdb
import neogeo.extent as neoextent
import numpy as np
import tables as tb
import mercantile
import time
import navpy
import pyflann
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import neogeo.core as core
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd


    f2l_mat = np.array([125E3, 250E3, 500E3, 1E6])
    n_img_feat = np.array([1E3, 5E3, 10E3, 20E3])
out_tb = tb.open_file('/media/sean/D2F2E7B2F2E798CD/Users/student/neo_data/obs_out_mat.hdf','r')
