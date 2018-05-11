import tables as tb


class Landmark(tb.IsDescription):
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
    rank = tb.UInt8Col()


old_db = tb.open_file('/Users/venabled/data/neogeo/brisk_new_unbiased_db.hdf', 'r')
new_db = tb.open_file('/Users/venabled/data/neogeo/briskdb.hdf', 'w')

dbt = old_db.root.sift_db.sift_features_sorted

filters = tb.Filters(complib='blosc', complevel=5)
group = new_db.create_group('/', 'landmarkdb', title='Landmark Database')
table = new_db.create_table(group, 'landmarks', Landmark, filters=filters,
                            expectedrows=dbt.shape[0], chunkshape=500)

desc = new_db.create_carray(group, 'descriptors', atom=tb.UInt8Atom(), shape=(dbt.shape[0], 64))



for row in dbt.iterrows():
    desc[row.nrow] = row['descriptor']
    nrow = (row["angle"], row["height"], row["lat"], row["layer"], row["lon"],
            row["octave"], row["pair_id"], 1, row["response"], row["scale"],
            row["size"], row["x"], row["y"], row["z"])
    table.append([nrow])


old_db.close()
new_db.close()

old_db = tb.open_file('/Users/venabled/data/neogeo/brisk_new_unbiased_db.hdf', 'r')
oldt = old_db.root.sift_db.sift_features_sorted

new_db = tb.open_file('/Users/venabled/data/neogeo/briskdb.hdf', 'a')
newt = new_db.root.landmarkdb.landmarks
desc = new_db.root.landmarkdb.descriptors