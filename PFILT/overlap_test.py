from shapely.geometry import Polygon
import numpy as np
import matplotlib.pyplot as plt
import navpy
import pandas as pd
import mercantile
import neogeodb.pytables_db as ndb
from functools import partial
from descartes import PolygonPatch

cmap = {'f1': '#fff4f3',
        'f2': '#b7d3e3',
        'f3': '#fd9b0f',
        'f5': '#f7ff71',
        'tid': '#1cd046'}


def plot_coords(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, 'o', color='#999999', zorder=1)


def build_poly(img_num, geo_df, ref_pt):
    f_llh = geo_df.loc[img_num][10:-1].reshape(4, 3).astype(np.float)
    f_llh[:, 2] = 0
    f_ned = navpy.lla2ned(f_llh[:, 1], f_llh[:, 0], f_llh[:, 2],
                          ref_pt[0], ref_pt[1], ref_pt[2])

    f_ned = np.vstack((f_ned, f_ned[0, :]))
    f_xy = [tuple(l) for l in f_ned[:, 0:2].tolist()]
    return Polygon(f_xy)


def polygon_overlap(img_num, geo_df, base_p, ref_pt):
    pf = build_poly(img_num, geo_df, ref_pt)
    poly_intersect = base_p.intersection(pf)
    return poly_intersect.area


geo_map = {'f1': pd.read_hdf('/Users/venabled/data/uvan/geo_f1.hdf5'),
           'f2': pd.read_hdf('/Users/venabled/data/uvan/geo_f2.hdf5'),
           'f3': pd.read_hdf('/Users/venabled/data/uvan/geo_f3.hdf5'),
           'f5': pd.read_hdf('/Users/venabled/data/uvan/geo_f5.hdf5')}

tid = 153369548
tid = 153344794
tx, ty = ndb.unpair(tid)


tid_bbox = mercantile.bounds(tx, ty, 15)
tid_lon_lat = np.array([[tid_bbox.west, tid_bbox.south, 0.0],
                        [tid_bbox.east, tid_bbox.south, 0.0],
                        [tid_bbox.east, tid_bbox.north, 0.0],
                        [tid_bbox.west, tid_bbox.north, 0.0]])
tid_ul = mercantile.ul(tx, ty, 15)
ref = np.array([tid_ul.lat, tid_ul.lng, 0.0])
bbox_ned = navpy.lla2ned(tid_lon_lat[:, 1], tid_lon_lat[:, 0], tid_lon_lat[:, 2],
                         ref[0], ref[1], ref[2])
bbox_ned = np.vstack((bbox_ned, bbox_ned[0, :]))
bbox_xy = [tuple(l) for l in bbox_ned[:, 0:2].tolist()]

pbb = Polygon(bbox_xy)

for fg in air_meta[air_meta.tid == tid].groupby('flight'):
    plt.figure()
    ax = plt.gca()
    plot_coords(ax, pbb.exterior)

    patch = PolygonPatch(pbb, facecolor=cmap['tid'], edgecolor='#000000', alpha=0.5, zorder=2)
    ax.add_patch(patch)

    # Grab an image from f5 to figure this out
    for of in fg[1].iterrows():
        f5_img_num = of[1].img_num
        f5i_llh = geo_map[of[1].flight].loc[f5_img_num][10:-1].reshape(4,3).astype(np.float)
        f5i_llh[:, 2] = 0
        f5i_ned = navpy.lla2ned(f5i_llh[:, 1], f5i_llh[:, 0], f5i_llh[:, 2],
                                ref[0], ref[1], ref[2])

        f5i_ned = np.vstack((f5i_ned, f5i_ned[0, :]))
        f5i_xy = [tuple(l) for l in f5i_ned[:, 0:2].tolist()]
        pf5 = Polygon(f5i_xy)

        plot_coords(ax, pf5.exterior)

        patch = PolygonPatch(pf5,
                             facecolor=cmap[of[1].flight],
                             edgecolor='#000000',
                             alpha=0.1, zorder=2)
        ax.add_patch(patch)

        z = pf5.intersection(pbb)
        df = pd.read_hdf(of[1].df_path)
        print((of[1].flight, f5_img_num, z.area, df.gsd[0]))
        plt.title(of[1].flight)


# First Fix F5 Img
f1_obs = air_meta[(air_meta.flight == 'f1') & (air_meta.tid == tid)]
f2_obs = air_meta[(air_meta.flight == 'f2') & (air_meta.tid == tid)]
f3_obs = air_meta[(air_meta.flight == 'f3') & (air_meta.tid == tid)]
f5_obs = air_meta[(air_meta.flight == 'f5') & (air_meta.tid == tid)]
x5_obs = air_meta[(air_meta.flight != 'f5') & (air_meta.tid == tid)]

f5a = f5_obs.apply(lambda x: polygon_overlap(x.img_num, geo_map['f5'], pbb, ref), axis=1)
f5img = f5_obs.loc[f5a.argmax()]
f5p = build_poly(f5img.img_num, geo_map['f5'], ref)

def df_area(df, geo_map, pbb, ref):
    dfa = df.apply(lambda x: polygon_overlap(x.img_num, geo_map[x.flight], pbb, ref), axis=1)
    df['overlap_area'] = dfa
    return df.loc[dfa.argmax()]

# Find Imgs from other flights that overlap the most
most_overlap = x5_obs.groupby('flight').apply(df_area, geo_map=geo_map, pbb=f5p, ref=ref)

plt.figure()
ax = plt.gca()
plot_coords(ax, pbb.exterior)

patch = PolygonPatch(pbb, facecolor=cmap['tid'], edgecolor='#000000', alpha=0.5, zorder=2)
ax.add_patch(patch)

patch = PolygonPatch(f5p,
                     facecolor=cmap['f5'],
                     edgecolor='#000000',
                     alpha=0.6, zorder=2)
ax.add_patch(patch)

for of in most_overlap.iterrows():
    pf5 = build_poly(of[1].img_num, geo_map[of[1].flight], ref)
    plot_coords(ax, pf5.exterior)

    patch = PolygonPatch(pf5,
                         facecolor=cmap[of[1].flight],
                         edgecolor='#000000',
                         alpha=0.3, zorder=2)
    ax.add_patch(patch)



z = pf5.intersection(pbb)
plt.figure()
ax = plt.gca()
plot_coords(ax, z.exterior)
patch = PolygonPatch(z, facecolor=COLOR[True], edgecolor=COLOR[True], alpha=0.1, zorder=2)
ax.add_patch(patch)

plt.show()