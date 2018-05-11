import numpy as np
import tables as tb
from . import georeg
import h5py
from . import hdf_orthophoto as hdo
from . import features as feat
from . import features2d as f2d
import mercantile
import yaml
from affine import Affine
import osr
import navpy



class Feature(tb.IsDescription):

    """
    This class describes a row in a PyTables table that represents
    a SIFT feature in a feature database
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
    descriptor = tb.UInt8Col(64)


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


def create_pytables_db(table_hdf):
    """
    For a set of features in the flat_hdf, write them to a table in HDF5
    """
    out_file = tb.open_file(table_hdf, mode='w', tilte='Feature Database')
    group = out_file.create_group('/', 'sift_db', 'Sift Feature Database')
    filters = tb.Filters(complib='blosc', complevel=5)
    table = out_file.create_table(
        group, 'sift_features', Feature, filters=filters)
    return out_file, group, table


def create_sorted_db(old_table, sorted_hdf):
    """
    For a set of features in the flat_hdf, write them to a table in HDF5
    """
    out_file = tb.open_file(sorted_hdf, mode='w', tilte='Feature Database')
    group = out_file.create_group('/', 'sift_db', 'Sift Feature Database')
    old_table.copy(newparent=group, newname='sift_features_sorted',
                   sortby='pair_id', checkCSI=True)

    new_table = out_file.root.sift_db.sift_features_sorted
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


def add_rows_to_table(feature, df, dg):
    """
    Taking in a feature, which is a table.row of Class Feature, feed in \
    df and dg, which are 2 np.arrays returned from features_from_slice
    """
    for ii in np.arange(df.shape[0]):
        feature['x'] = df[ii, 0]
        feature['y'] = df[ii, 1]
        feature['z'] = df[ii, 2]
        feature['lon'] = df[ii, 3]
        feature['lat'] = df[ii, 4]
        feature['height'] = df[ii, 5]
        feature['octave'] = df[ii, 6]
        feature['layer'] = df[ii, 7]
        feature['scale'] = df[ii, 8]
        feature['angle'] = df[ii, 9]
        feature['response'] = df[ii, 10]
        feature['size'] = df[ii, 11]
        feature['pair_id'] = elegant_pair_xy(df[ii, 0], df[ii, 1])
        feature['descriptor'] = dg[ii, :]
        feature.append()


def add_unique_tiles_table(out_file, feature_table, group):
    unique_tiles = np.unique(feature_table.cols.pair_id[:])
    atom = tb.UInt32Atom()
    filters = tb.Filters(complevel=5, complib='blosc')
    ca = out_file.create_carray(group, 'unique_tiles', atom,
                                unique_tiles.shape, filters=filters)
    ca.flush()


def features_from_slice(ophoto, slc, detector, desc_extract,
                        terrain_handler, zoom=15, bias=None):
    """
    Takes in an orthophoto, slice tuple, detector, and extractor
    """
    sub_photo = ophoto.get_img_from_slice(slc)
    kp1, desc1 = f2d.extract_features(sub_photo.img, detector, desc_extract)
    if len(kp1) > 0:
        pix = f2d.keypoint_pixels(kp1)
        meta = np.array([(kp.angle, kp.response, kp.size) for kp in kp1])
        packed = f2d.numpySIFTScale(kp1)
        pts_wgs84 = sub_photo.pix_to_wgs84(pix)
        pts_hae = terrain_handler.add_heights(pts_wgs84)
        if bias is not None:
            print(bias)
            ref = pts_hae[0, [1, 0, 2]]
            pts_ned = navpy.lla2ned(pts_hae[:, 1], pts_hae[:, 0], pts_hae[:, 2],
                                    *ref)
            pts_ned -= bias
            pts_hae = np.vstack(navpy.ned2lla(pts_ned, *ref)).T
            pts_hae = pts_hae[:, [1, 0, 2]]

        tile_coords = np.array([np.array(mercantile.tile(pt[0], pt[1], zoom))
                                for pt in pts_hae])
        return (np.hstack((tile_coords, pts_hae, packed, meta)), desc1)
    else:
        return (None, None)


def create_pair_index(table):
    """
    Create a Completely Sorted Index (CSI) on pair_id
    """
    table.cols.pair_id.create_csindex()
    table.copy(newname='sift_features_sorted', sortby='pair_id', checkCSI=True)


def create_xy_octave_index(table):
    """
    Create CSI for X, Y, and octave
    """
    table.cols.x.create_csindex()
    table.cols.y.create_csindex()
    table.cols.octave.create_csindex()
