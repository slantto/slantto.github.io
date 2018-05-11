#!/usr/bin/env python
# -*- coding: utf-8 -*-

from navfeatdb.db import features as dbfeat
from navfeatdb.db import mongo
from navfeatdb.frames import terrain
from navfeatdb.ortho import orthophoto
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial


import cv2
import pymongo


def map_slice(slc, dbname, collectionname, srtm, geoid, ophoto):
    client = pymongo.MongoClient()
    db = client[dbname]
    collection = db[collectionname]

    print(slc)
    max_dim = 20000
    max_octaves = int(np.floor(np.log2(max_dim))) - 1
    detbrisk = cv2.BRISK_create(octaves=max_octaves)
    descbrisk = cv2.BRISK_create()
    vrto = orthophoto.VRTOrthophoto(ophoto)
    srtm_handler = terrain.SRTM(srtm, geoid)
    oextractor = dbfeat.OrthoPhotoExtractor(detbrisk, descbrisk, vrto, srtm_handler)
    fdf, desc = oextractor.features_from_slice(slc)
    if fdf is not None:
        print("Adding %d Features" % fdf.shape[0])
        mongo.add_rows_to_table(collection, fdf, desc)
    client.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='build_mongo_db',
                                     description='Extracts and GeoLocates Features From GDAL Readable Image')
    parser.add_argument('-g', '--geoid', help='Path to Geoid File')
    parser.add_argument('-s', '--srtm', help='Path to SRTM Root')
    parser.add_argument('-i', '--orthophoto', help='Path to OrthoPhoto')
    parser.add_argument('-d', '--db', help='MongoDB Database Name',
                        default='navfeatdb')
    parser.add_argument('-c', '--collection',
                        help='MongoDB Collection Name',
                        default='ophotocollection')
    args = parser.parse_args()



    # Get usable chunks
    vrto = orthophoto.VRTOrthophoto(args.orthophoto)
    slices = orthophoto.slices_from_ophoto(vrto, max_pix_in_dim=20000)

    pmap = partial(map_slice, dbname=args.db, collectionname=args.collection, srtm=args.srtm, geoid=args.geoid, ophoto=args.orthophoto)

    with Pool(8) as pool:
        pool.map(pmap, slices)

    client = pymongo.MongoClient()
    db = client[args.db]
    collection = db[args.collection]
    collection.create_index([("loc", pymongo.GEOSPHERE)])
    client.close()