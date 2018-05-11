#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql.types import *

feat_schema = StructType(
    [StructField('angle', DoubleType(), False),
     StructField('descriptor', ArrayType(ShortType(), False), False),
     StructField('height', DoubleType(), False),
     StructField('lat', DoubleType(), False),
     StructField('layer', ShortType(), False),
     StructField('lon', DoubleType(), False),
     StructField('octave', ShortType(), False),
     StructField('pair_id', LongType(), False),
     StructField('response', DoubleType(), False),
     StructField('scale', DoubleType(), False),
     StructField('size', DoubleType(), False),
     StructField('x', IntegerType(), False),
     StructField('y', IntegerType(), False),
     StructField('z', IntegerType(), False)
    ])


def row_to_list(row):
    """
    Takes in a row from featuredb and dumps it to a nested list (thanks to
    descriptors for making this hard
    :param row: Nested numpy.ndarray
    :return: list of elements, and another list of integers for the descriptor
    """
    import numpy as np
    out = []
    for ii in np.arange(len(row)):
        if ii == 1:
            out.append(row[ii].tolist())
        else:
            out.append(row[ii].item())
    return out


def rows_from_tb(file_tb_tuple):
    import tables as tb
    import numpy as np
    filename = file_tb_tuple[0]
    uid = file_tb_tuple[1]
    pytbdb = tb.openFile(filename, 'r')
    table = pytbdb.root.sift_db.sift_features_sorted
    rows = table.read_where('pair_id == uid')
    listofrows = [row_to_list(row) for row in rows]
    return listofrows


if __name__ == "__main__":

    import tables as tb

    filepath = '/Users/venabled/data/neogeo/pytables_db.hdf'
    pytbdb = tb.open_file(filepath, 'r')
    uid_rdd = sc.parallelize(pytbdb.root.sift_db.unique_tiles[:])
    file_uid_rdd = uid_rdd.map(lambda uid: (filepath, uid))
    rows_rdd = file_uid_rdd.flatMap(rows_from_tb)
    sqlContext.createDataFrame(rows_rdd, feat_schema).write.parquet('/Users/venabled/data/featdb.parquet', mode='append', partitionBy='pair_id')



