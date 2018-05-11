#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 09:10:48 2017
Script to extract SIFT features from image
Version 2.0: Is same as airborn_feat_extraction but without parser frome terminal
    all sources, and output hardcoded.
@author: Sean Lantto
"""
import os
import numpy as np
import tables as tb
import cv2
import bcolz
import pandas as pd
import navfeatdb.utils.cvfeat2d as f2d

out_path = ('/home/sean/ImageAidedNav/')
os.makedirs(os.path.join(out_path, 'feat/df'))
os.makedirs(os.path.join(out_path, 'feat/desc'))

flight = tb.open_file('/home/sean/ImageAidedNav/neo_data/fc2_f5.hdf', 'a')
images = flight.root.camera.image_raw.compressed.images
img_times = flight.root.camera.image_raw.compressed.metadata.cols.t_valid
#flight.create_group(flight.root, 'sift_feat')



img_range = np.arange(img_times.shape[0])

sift = cv2.xfeatures2d.SIFT_create()

for ii in img_range:
    print('%d / %d' % (ii, img_range.shape[0]))
    kp = sift.detect(images[ii])
    kp, desc = sift.compute(images[ii],kp)
    firstimg=images[ii]
    
    if len(kp) > 0:
        pix = f2d.keypoint_pixels(kp)
        meta = np.array([(pt.angle, pt.response, pt.size) for pt in kp])
        packed = f2d.numpySIFTScale(kp)
        #firstimg = cv2.drawKeypoints(images[ii],kp,firstimg)
        #cv2.imwrite("sift_keypoints" + str(ii) +".jpg", firstimg)
        feat_df = pd.DataFrame(np.hstack((pix, meta, packed)),
                                  columns=['pix_x', 'pix_y', 'angle',
                                           'response', 'size', 'octave',
                                           'layer', 'scale'])
        feat_df.sort_values('response', ascending=False, inplace=True)
        
    else:
        feat_df = None
    
    
    if feat_df is None:
        print('none')
    else:
        df_path = 'feat/df/feat_%d.hdf' % ii
        desc_path = 'feat/desc/desc_%d' % ii
        print("%d :: %d Feat" % (ii, desc.shape[0]))
        feat_df.to_hdf(os.path.join(out_path, df_path), 'feat_df', mode='w', format='table', complib='zlib', complevel=7)
        bcolz.carray(desc.astype(np.float32), rootdir=os.path.join(out_path, desc_path), mode='w').flush()
    

    

    
flight.close()
    