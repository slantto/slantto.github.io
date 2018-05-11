#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script opens a flight with features, windows the rows and then
sorts the features, storing in a new pytables db
"""

import numpy as np
import tables as tb


if __name__ == '__main__':
    f5 = tb.open_file('/Users/venabled/catkin_ws/data/fc2/fc2_f5.hdf', 'a')
    imgs = f5.root.images.image_data
    img_times = f5.root.images.t_valid
    feat_tables = f5.list_nodes(f5.root.images.sift_features)
    pva = f5.root.pva

    f5_out = tb.open_file('/Users/venabled/catkin_ws/data/fc2/sorted_fc2_f5.hdf', 'a')
    img_group = f5_out.create_group(f5_out.root, 'images')
    feat_group = f5_out.create_group(img_group, 'sift_features')

    for img_num in np.arange(img_times.nrows):
        ft = f5.get_node('/images/sift_features/img_%d' % img_num)
        if not ft.cols.response.is_indexed:
            ft.cols.response.create_csindex()
        ft_new = ft.copy(feat_group, sortby='response', overwrite=True, propindexes=True)
        if not ft_new.cols.size.is_indexed:
            ft_new.cols.size.create_csindex()
        print img_num
