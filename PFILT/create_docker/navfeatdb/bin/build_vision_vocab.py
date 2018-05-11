#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bcolz
import navfeatdb.utils.matching as mutils
import pandas as pd
import numpy as np
import navfeatdb.visvocab.core as vocabcore
import os


if __name__ == '__main__':
    path_prefix = '/Users/venabled/data/uvan/fc2/f2'
    feat_meta = pd.read_hdf(os.path.join(path_prefix, 'feat_meta.hdf'))

    for ki in [500, 1000]:
        vocab = vocabcore.create_vocab(feat_meta, path_prefix, k=ki)
        out_dir = os.path.join(path_prefix, '%d_vocab' % ki)
        bcolz.carray(vocab, rootdir=out_dir, mode='w')
