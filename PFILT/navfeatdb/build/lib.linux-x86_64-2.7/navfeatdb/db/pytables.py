import numpy as np
import tables as tb

class Landmark(tb.IsDescription):
    """
    This class describes a row in a PyTables table that represents
    a georegistered feature in a feature database
    """
    x = tb.UInt32Col()
    y = tb.UInt32Col()
    z = tb.UInt32Col()
    pair_id = tb.UInt32Col()
    lon = tb.Float64Col()
    lat = tb.Float64Col()
    height = tb.Float64Col()
    octave = tb.Int16Col()
    layer = tb.Int16Col()
    scale = tb.Float64Col()
    angle = tb.Float64Col()
    response = tb.Float64Col()
    size = tb.Float64Col()
    rank = tb.UInt8Col()


def elegant_pair_xy(x, y):
    """
    http://szudzik.com/ElegantPairing.pdf
    """
    if x > y:
        return x**2 + x + y
    else:
        return y**2 + x


def unpair(z):
    fsz = np.floor(np.sqrt(z))
    if z - fsz**2 < fsz:
        return z - fsz**2, fsz
    else:
        return np.int(fsz), np.int(z - fsz**2 - fsz)


def find_chunk_size(image_shape, largest_size=5000):
    s0 = 0
    s1 = 1
    for ii in np.arange(1, largest_size + 1):
        if np.mod(image_shape[0], ii) == 0.0:
            s0 = ii
        if np.mod(image_shape[1], ii) == 0.0:
            s1 = ii
    return (s0, s1)


def slices_from_ophoto(hdfo, max_pix_in_dim=5000):
    """
    :param hdfo: Valid neogeodb.hdf_orthophoto to create pixel slices from
    :param max_pix_in_dim: Integer value of the maximum number of pixels
        to be used in any dimension of a subslice
    :return list of tuples of the x-y slices of the image
    Given an neogeodb.hdf_orthophoto object, return list of slice tuples e.g. \
    ((row_start, row_stop), (col_start, col_stop))
    """

    bshape = find_chunk_size(hdfo.shape, max_pix_in_dim)
    blocks = np.array(hdfo.shape)[0:2] / np.array(bshape)
    out_tuples = []
    for row in np.arange(blocks[0]):
        rstart = row * bshape[0]
        rstop = rstart + bshape[0]
        for col in np.arange(blocks[1]):
            cstart = col * bshape[1]
            cstop = cstart + bshape[1]
            out_tuples.append(((rstart, rstop), (cstart, cstop)))
    return out_tuples


def create_pytables_db(table_hdf, descriptor_len):
    """
    For a set of features in the flat_hdf, write them to a table in HDF5
    """
    out_file = tb.open_file(table_hdf, mode='w', tilte='Landmark Database')
    group = out_file.create_group('/', 'db', 'Landmark Database Group')
    filters = tb.Filters(complib='blosc', complevel=5)
    table = out_file.create_table(group, 'landmarks', Landmark, filters=filters)
    desc = out_file.create_earray(group, 'descriptors', atom=tb.UInt8Atom(), shape=(0, descriptor_len))

    return out_file, group, table, desc


def create_sorted_db(old_table, old_array, sorted_hdf):
    """
    For a set of features in the flat_hdf, write them to a table in HDF5
    """
    out_file = tb.open_file(sorted_hdffinger, mode='w', tilte='Landmark Database')
    group = out_file.create_group('/', 'db', 'Landmark Database')
    old_table.copy(newparent=group, newname='landmarks', sortby='pair_id', checkCSI=True)

    filters = tb.Filters(complib='blosc', complevel=5)
    new_array = out_file.create_carray(group, name='descriptors', atom=tb.UInt8Atom(), shape=old_array.shape, filters=filters)
    idx = old_table.cols.pair_id.index
    new_array[:, :] = old_array[idx[:], :]

    new_table = out_file.root.db.landmarks
    new_table.cols.pair_id.create_csindex()
    new_table.cols.x.create_csindex()
    new_table.cols.y.create_csindex()
    new_table.cols.octave.create_csindex()

    unique_tiles, uidcount = np.unique(old_table.cols.pair_id[:], return_counts=True)
    atom = tb.UInt32Atom()
    filters = tb.Filters(complevel=5, complib='blosc')
    uidcount_a = out_file.create_carray(group, 'unique_tile_count', atom,
                                unique_tiles.shape, filters=filters)

    uid_a = out_file.create_carray(group, 'unique_tiles', atom,
                                unique_tiles.shape, filters=filters)

    uidcount_a[:] = uidcount
    uid_a[:] = unique_tiles

    out_file.close()


def add_rows_to_table(landmark, df):
    """
    Taking in a feature, which is a table.row of Class Feature, feed in \
    df and dg, which are 2 np.arrays returned from features_from_slice
    """
    for idx, row in df.iterrows():
        landmark['x'] = row['x']
        landmark['y'] = row['y']
        landmark['z'] = row['z']
        landmark['lon'] = row['lon']
        landmark['lat'] = row['lat']
        landmark['height'] = row['height']
        landmark['octave'] = row['octave']
        landmark['layer'] = row['layer']
        landmark['scale'] = row['scale']
        landmark['angle'] = row['angle']
        landmark['response'] = row['response']
        landmark['size'] = row['size']
        landmark['pair_id'] = elegant_pair_xy(row.x, row.y)
        landmark.append()


def add_unique_tiles_table(out_file, feature_table, group):
    unique_tiles = np.unique(feature_table.cols.pair_id[:])
    atom = tb.UInt32Atom()
    filters = tb.Filters(complevel=5, complib='blosc')
    ca = out_file.create_carray(group, 'unique_tiles', atom,
                                unique_tiles.shape, filters=filters)
    ca.flush()


def create_pair_index(table):
    """
    Create a Completely Sorted Index (CSI) on pair_id
    """
    table.cols.pair_id.create_csindex()


def create_xy_octave_index(table):
    """
    Create CSI for X, Y, and octave
    """
    table.cols.x.create_csindex()
    table.cols.y.create_csindex()
    table.cols.octave.create_csindex()