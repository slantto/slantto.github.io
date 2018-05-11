#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This executable takes in an orthophoto HDF, finds slices, and writes the
output to a pytables file. Then creates an index by pair_id, and copies
it to a new table/file
"""

import neogeodb.pytables_db as tbdb
import neogeodb.features2d as f2d
import numpy as np
import cv2
from neogeodb.georeg import SRTM
import yaml
import os
import argparse
import time


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='build_pytables_db',
        description='Creates a database of goeregistered landmarks from \
                    orthophotos')
    parser.add_argument('-x', '--hdf',
                        help='Path to output pytables file')
    parser.add_argument('-o', '--ophoto',
                        help='Path to HDF Orthophoto')
    # parser.add_argument('-f', '--detector',
    #                     help='Path to detector YAML')
    # parser.add_argument('-d', '--descriptor',
    #                     help='Path to detector YAML')
    parser.add_argument('-g', '--geoid',
                        help='Geoid .gtx file')
    parser.add_argument('-t', '--terrain',
                        help='Terrain ROOT')
    parser.add_argument('-m', '--dim', default=5000,
                        help='Maximum Dimension to Try and Extract Features')
    parser.add_argument('-i', '--img', default='/ophoto/gray',
                        help='Path within ophoto hdf5 structure for image')
    parser.add_argument('-b', '--bias')

    args = parser.parse_args()

    # Set up terrain handlers, feature detectors/extractors
    srtm = SRTM(args.terrain, args.geoid)

    detector = cv2.BRISK_create()
    desc_extract = cv2.BRISK_create()

    ned_bias = np.array(yaml.load(file(args.bias, 'r'))['bias'])

    # Create the temp table HDF that we will store unsorted features in
    # as they're extracted
    temp_table_hdf = args.hdf + '.temp'
    out_file, group, table = tbdb.create_pytables_db(temp_table_hdf)
    feature = table.row

    # Compute windows into the HDF Orthophoto
    slc_cnt = 0
    t0 = time.time()
    slices = tbdb.slices_from_ophoto(args.ophoto, args.img, args.dim)
    for slc in slices:
        t1 = time.time()
        print('Working on slice %s: %d / %d' % (slc, slc_cnt, len(slices)))
        df, dg = tbdb.features_from_slice(args.ophoto, args.img, slc,
                                          detector, desc_extract, srtm,
                                          bias=ned_bias)
        if df is not None:
            tbdb.add_rows_to_table(feature, df, dg)
        print('Slice %d took: %f' % (slc_cnt, time.time() - t1))
        slc_cnt += 1
        table.flush()

    print('DB Creation took: %f' % (time.time() - t0))

    # Create Index, Copy To New file
    print('Creating Index on PairId')
    table.cols.pair_id.create_csindex()

    print('Copying to New Table')
    tbdb.create_sorted_db(table, args.hdf)

    # Delete old table
    out_file.close()
    os.remove(temp_table_hdf)
