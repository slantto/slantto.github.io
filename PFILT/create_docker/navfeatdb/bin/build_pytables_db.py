from navfeatdb.db import features as dbfeat
from navfeatdb.db import pytables as tbdb
from navfeatdb.frames import terrain
from navfeatdb.ortho import orthophoto
import cv2


# vrto- abstracted orthophoto object that adds on to GDAL
vrto = orthophoto.VRTOrthophoto('/data/osip/athens_tiles/athens_mosaic.vrt')
# Terrain object thing
srtm_handler = terrain.SRTM('/home/RYWN_Data/srtm/SRTM1/Region_06', '/data/geoid/egm08_25.gtx')

# Get usable chunks
brisk = cv2.BRISK_create()
oextractor = dbfeat.OrthoPhotoExtractor(brisk, brisk, vrto, srtm_handler)
slices = tbdb.slices_from_ophoto(vrto)

# Set up output pytables file
out_hdf5, h5_group, h5_table, h5_desc = tbdb.create_pytables_db('/data/osip/athens_tiles/athens_db.h5', 64)
landmark = h5_table.row

for idx, slice in enumerate(slices):
    print(slice, idx, len(slices))
    fdf, desc = oextractor.features_from_slice(slice)
    tbdb.add_rows_to_table(landmark, fdf)
    h5_desc.append(desc)

tbdb.create_pair_index(h5_table)
tbdb.add_unique_tiles_table(out_hdf5, h5_table, h5_group)
out_hdf5.close()
