#!/usr/bin/env python
# -*- coding: utf-8 -*-

from navfeatdb.db import features as dbfeat
from navfeatdb.db import pytables as tbdb
from navfeatdb.frames import terrain
from navfeatdb.ortho import orthophoto
import cv2
import os
import pandas as pd
import dask.dataframe as dd
import numpy as np
from distributed import Client
import h5py
import argparse


def db_from_slice(row):
    print(row)
    vrto = orthophoto.VRTOrthophoto(row['ophoto'])
    srtm_handler = terrain.SRTM(row['terrain_path'], row['geoid'])
    brisk = cv2.BRISK_create()
    oextractor = dbfeat.OrthoPhotoExtractor(brisk, brisk, vrto, srtm_handler)
    out_path = os.path.join(row['out_path'], 'slice_%d.h5' % row.idx)

    fdf, desc = oextractor.features_from_slice((row['x_slice'], row['y_slice']))

    if not os.path.exists(out_path):
        fdf.to_hdf(out_path, key='/db/landmarks', format='table')
        f = h5py.File(out_path, 'a')
        f.create_dataset('db/descriptors', data=desc)
        f.close()

    return pd.Series({'file_path': out_path,
                      'num_landmarks': fdf.shape[0]})
    vrto.close()
    return pd.Series({'landmarks': fdf, 'descriptors': desc})


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='orthophoto_feat_extraction',
                                     description='Extracts and GeoLocates Features From a UVAN Flight')
    parser.add_argument('-o', '--output',
                        help='Directory where to create test datasets')
    parser.add_argument('-g', '--geoid', help='Path to Geoid File')
    parser.add_argument('-s', '--srtm', help='Path to SRTM Root')
    parser.add_argument('-d', '--scheduler', help='Node hosting the scheduler')
    parser.add_argument('-i', '--orthophoto', help='Path to OrthoPhoto')
    args = parser.parse_args()

    in_ophoto = orthophoto.VRTOrthophoto(args.orthophoto)
    slices = tbdb.slices_from_ophoto(in_ophoto, 5000)
    in_ophoto.close()

    slice_df = pd.DataFrame(slices, columns=['x_slice', 'y_slice'])
    slice_df['out_path'] = args.output
    slice_df['terrain_path'] = args.srtm
    slice_df['geoid'] = args.geoid
    slice_df['ophoto'] = args.orthophoto
    slice_df['idx'] = slice_df.index

    client = Client(args.scheduler)

    meta = {'file_path': np.str,
            'num_landmarks': np.int}

    ddf = dd.from_pandas(slice_df, npartitions=len(client.ncores()))
    ask_df = ddf.apply(db_from_slice, axis=1, meta=meta)
    ask_df.compute()
