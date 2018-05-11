#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import argparse
from skimage.transform import resize, pyramids
import h5py


def find_chunk_size(image_shape, largest_size=512):
    s0 = 0
    s1 = 1
    for ii in np.arange(1, largest_size+1):
        if np.mod(image_shape[0], ii) == 0.0:
            s0 = ii
        if np.mod(image_shape[1], ii) == 0.0:
            s1 = ii
    return (s0, s1)


def pyramid_reduce_hdf5(hdf5_file, image_path, out_file, level):
    hdf = h5py.File(hdf5_file)
    max_size = 1024
    blockshape = find_chunk_size(hdf[image_path].shape, max_size)
    dimg = da.from_array(hdf[image_path], chunks=blockshape)
    blurred = dimg.map_overlap(wrapped_smooth, depth=6, boundary='reflect')
    d2 = (np.ceil(np.array(blockshape)/2.0)).astype(int)
    down2 = blurred.map_blocks(wrapped_resize, chunks=(d2[0], d2[1]))
    down2.to_hdf5(out_file, '/level_%d' % level, compression='lzf')
    hdf.close()


def wrapped_smooth(image):
    return pyramids._smooth(image, sigma=(4/6.0), mode='reflect', cval=0)


def wrapped_resize(image):
    rows = image.shape[0]
    cols = image.shape[1]
    out_rows = np.ceil(rows / 2.0)
    out_cols = np.ceil(cols / 2.0)
    return resize(image, (out_rows, out_cols), order=1, mode='reflect', cval=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='pyramid_dask.py',
        description='Creates a pyramid of images using dask, stores the \
                     output layers to HDF5')
    parser.add_argument('-i', '--ophoto', required=True,
                        help='Path to HDF Orthophoto')
    parser.add_argument('-p', '--pyramid', required=True,
                        help='Path to create output HDF5 pyramid')
    parser.add_argument('-b', '--layer', default='/ophoto/gray',
                        help='Path within input Orthophoto for pyramid \
                              base layer')
    args = parser.parse_args()

    hdf_file = args.ophoto
    img_path = args.layer
    out_file = args.pyramid

    in_hdf = h5py.File(hdf_file, 'r')
    hdfimg = in_hdf[img_path]
    levels_to_render = np.int(
        np.ceil(np.log2(np.min(hdfimg.shape))) - np.log2(512))

    if levels_to_render > 0:
        pyramid_reduce_hdf5(hdf_file, img_path, out_file, 1)

        for ii in np.arange(1, levels_to_render):
            opath = '/level_%d' % ii
            pyramid_reduce_hdf5(out_file, opath, out_file, ii + 1)
