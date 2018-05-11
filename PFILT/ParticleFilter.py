#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script processes the output from neogeoPF, which provides the feature matches for the images. This script uses a
particle filter to estimate the position of the aircraft, using the feature matches as the measurement likelihood, and
integrated IMU data as the prediction of motion. This script assumes no known initial position.
"""

import neogeo.extent as neoextent
import numpy as np
import pandas as pd
import pnpnav.utils as pnputils
import tables as tb
import particle_core as core
import navpy
import matplotlib
# matplotlib.use('Agg') #needed to run headless. If using the "GUI" to display in runtime, may need to remove this line
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import navfeatdb.utils.matching as mutils
import mercantile
from neogeodb import hdf_orthophoto as hdo
import multiprocessing as mp
from numpy.linalg import norm
import time
import pyflann

# def upd(neo,imgobs):
#     weights = neo.update(imgobs)
#     return weights

if __name__ == '__main__':
    f5 = tb.open_file('/media/sean/My Passport/data_files/fc2_f5.hdf', 'r')
    f5featmeta = pd.read_hdf('/media/sean/My Passport/data_files/AIEoutput2/feat/feat_meta.hdf')
    imgs = f5.root.camera.image_raw.compressed.images
    img_times = f5.root.camera.image_raw.compressed.metadata.col('t_valid')
    feat_path = f5featmeta.iloc[:, 3]
    descripath = f5featmeta.iloc[:, 4]
    pva = f5.root.nov_span.pva

    tmatchidx = []
    existmatch = 0
    for pp in np.arange(0, pva.cols.t_valid.shape[0]):
        pvatround = round(pva.cols.t_valid[pp])
        # print(pp)
        for tt in np.arange(0, img_times.shape[0]):
            imgtround = round(img_times[tt])
            if pvatround == imgtround:
                if existmatch == 0:
                    # print(pvatround, imgtround)
                    tmatchidx.append(pp)
                    existmatch = 1
                    break
                if existmatch == 1:
                    # print('duplicate')
                    existmatch = 0
                    break

    tmatchidx = np.array(tmatchidx)
    # print(tmatchidx.shape)

    # Get a truth finder
    dted_path = '/media/sean/My Passport/data_files/srtm'
    geoid_file = '/media/sean/My Passport/data_files/egm96-15.tif'
    frame_yaml = '/home/sean/AFRL/ParticleFilterVisNav/pyPNP/data/fc2_pod_frames.yaml'

    finder = pnputils.DEMTileFinder(dted_path, geoid_file)
    finder.load_cam_and_vehicle_frames(frame_yaml)
    finder.load_camera_cal('/home/sean/AFRL/ParticleFilterVisNav/pyPNP/data/fc2_cam_model.yaml')
    out_tb = tb.open_file('/media/sean/My Passport/data_files/obs_out_forPF.hdf', 'r')

    neogeo_out = out_tb.root.loaded_1000000.feat_10000.neogeo_out
    obs_out = out_tb.root.loaded_1000000.feat_10000.obs_out
    t_loc = np.load('/media/sean/My Passport/data_files/t_loc.npy')
    ned_vel = np.load('/media/sean/My Passport/data_files/ned_vel.npy')

    # Get the Feature Database
    dbf = tb.open_file(
        '/media/sean/My Passport/data_files/pytables_db.hdf', 'a')
    dbt = dbf.get_node('/sift_db/sift_features_sorted')

    # You need boundaries
    txmin = dbt.cols.x[int(dbt.colindexes['x'][0])].astype(np.int64)
    txmax = dbt.cols.x[int(dbt.colindexes['x'][-1])].astype(np.int64)
    tymin = dbt.cols.y[int(dbt.colindexes['y'][0])].astype(np.int64)
    tymax = dbt.cols.y[int(dbt.colindexes['y'][-1])].astype(np.int64)
    xbounds = (txmin, txmax)
    ybounds = (tymin, tymax)
    print(xbounds)
    print(ybounds)
    xb, yb = neoextent.pad_grid(xbounds, ybounds)
    neo = core.PartycleFilt()
    tid, tidcount = np.unique(dbt.cols.pair_id, return_counts=True)

    extent = neoextent.SearchExtent(15, xb, yb, tid, tidcount)

    # get corners of the search area
    bbox = neoextent.bbox_from_extent(extent)

    # print('bbox')
    # print(bbox)

    c_wgs = np.hstack((np.array(bbox)[:, [1, 0]], np.zeros((4, 1))))

    # print('wgscent')
    # print(c_wgs)
    c_ned = np.zeros((c_wgs.shape[0], 3))
    c_ned = navpy.lla2ned(c_wgs[:, 0], c_wgs[:, 1], c_wgs[:, 2], c_wgs[3, 0], c_wgs[3, 1], c_wgs[3, 2])
    # print(c_ned)
    # print(c_ned[:,0].max())

    nrange = (c_ned[:, 0].min(), c_ned[:, 0].max())
    # print(nrange)
    erange = (c_ned[:, 1].min(), c_ned[:, 1].max())
    # print(erange)
    drange = (c_ned[:, 2].min(), c_ned[:, 2].max())

    truthLLA = np.zeros((tmatchidx.shape[0], 3))
    for kk in np.arange(0, tmatchidx.shape[0]):
        truthLLA[kk, :] = np.array(
            (pva[tmatchidx[kk]]['lat'], pva[tmatchidx[kk]]['lon'], pva[tmatchidx[kk]]['height'])).transpose()
        # print(truthLLA[kk])
    # truthLLA = np.array(truthLLA).transpose()
    truthNED = navpy.lla2ned(truthLLA[:, 0], truthLLA[:, 1], truthLLA[:, 2], c_wgs[3, 0], c_wgs[3, 1], c_wgs[3, 2])

    # Camera has a FOV half angle fo 8.3 degrees
    halfang = 8.3 * np.pi / 180
    FOV = truthNED[:, 2] * np.tan(halfang)

    # initialize particles within bounding box
    rad_range = (25, 1000)
    particles, weights, N = neo.create_uniform_particles(nrange, erange, rad_range, 25000)

    posmean = []
    posvar = []
    radmean = []
    radvar = []
    poserr = []
    raderr = []
    experr = []
    picsave = 10
    elapimg = 0
    resamp = 1
    for ii in np.arange(0, obs_out.shape[0]):
        print(ii)

        imgobs = obs_out[ii, :, :]
        # print("precut")
        # print(imgobs.shape[0])
        imgobs = imgobs[~(imgobs == 0).all(1)]
        # print("cut")
        # print(imgobs.shape[0])
        if imgobs.shape[0] > 0:
            imgobs[:, 1:] = navpy.lla2ned(imgobs[:, 1], imgobs[:, 2], imgobs[:, 3], c_wgs[3, 0], c_wgs[3, 1],
                                          c_wgs[3, 2])
        # print("ned")
        # print(imgobs)
        dpos = np.copy(ned_vel[ii, 0:2])
        dpos_sig = 0.75 * np.linalg.norm(dpos) * np.ones(2)  # previously 0.1
        rad_sig = 1
        particles = neo.motion_model(dpos, dpos_sig, rad_sig)
        weights = neo.update(imgobs)
        # pool = mp.Pool(processes=4)
        # weights = pool.apply(upd, args = (neo,imgobs))
        # print(weights)
        # pool.close()

        pos_mean, pos_var, partrad_mean, partrad_var, covariance = neo.estimate_pos
        pos_err = pos_mean - truthNED[ii, [0, 1]]
        rad_err = partrad_mean - np.abs(FOV[ii])
        exp_err = 3 * np.array(
            (np.sqrt(covariance[0, 0]), np.sqrt(covariance[1, 1]), np.sqrt(covariance[2, 2]))).transpose()

        posmean.append(pos_mean)
        posvar.append(pos_var)
        radmean.append(partrad_mean)
        radvar.append(partrad_var)
        raderr.append(rad_err)
        poserr.append(pos_err)
        experr.append(exp_err)

        neff = neo.neff(N)
        if ii == 343:
            print("pause")

        elapimg += 1  # count number of elapsed images since last re-sample

        if neff < (N / 2):
            print("resampling")

            resamp = 1 - (elapimg * 0.001)  # the constant 0.001 has been decided upon, because it will only scatter 10
            # particles for every image elapsed, the need for a new constant will be determined empirically.
            if resamp < 0:
                resamp = 1  # if too many images have elapsed since last re-sample, don't scatter. This will likely only
                # happen at the beginning of the data set

            particles, weights = neo.simple_resample(N, nrange, erange, rad_range, resamp)

            elapimg = 0
            resamp = 1

        if picsave == 10:
            plt.figure()
            ax = plt.gca()
            ax.set_xlim(erange[0], erange[1])
            ax.set_ylim(nrange[0], nrange[1])
            ax.autoscale(False)
            obslocs = ax.scatter(imgobs[:, 1], imgobs[:, 2], s=5, marker='x', facecolor='k', label='Features')
            parts = ax.scatter(particles[:, 1], particles[:, 0], s=10, c=particles[:, 2], cmap='plasma',
                               label='Particles')
            ax.scatter(truthNED[ii, 1], truthNED[ii, 0], s=100, marker='^', facecolor='none', edgecolors='g',
                       label='True Pos.')
            ax.scatter(posmean[ii][1], posmean[ii][0], s=100, marker='s', facecolor='none', edgecolors='r',
                       label='Est. Pos.')
            lambda_, v = np.linalg.eig(covariance)
            lambda_ = np.sqrt(lambda_)
            ell = Ellipse(xy=[posmean[ii][1], posmean[ii][0]], width=lambda_[1] * 3* 2, height=lambda_[0] * 3 * 2,
                          angle=np.rad2deg(np.arccos(v[0, 0])), linewidth=2, facecolor='none', edgecolor='b')
            ax.add_patch(ell)

            plt.legend(loc='upper left')
            colbar = plt.colorbar(parts)
            colbar.set_label('Particle Radius(meters)')
            plt.xlabel('E(meters)')
            plt.ylabel('N(meters)')
            plt.title('Particle Distribuiton at Image %d' % int(ii))

            plt.savefig('/home/sean/AFRL/pfresults/PF1/party_img_pres_%d' % int(ii), bbox_inches='tight', dpi=400)

            plt.figure()
            ax = plt.gca()
            ax.set_xlim(erange[0], erange[1])
            ax.set_ylim(nrange[0], nrange[1])
            ax.autoscale(False)
            obslocs = ax.scatter(imgobs[:, 2], imgobs[:, 1], s=10, marker='x', c=imgobs[:, 0], cmap='viridis',
                                 label='Features')
            ax.scatter(truthNED[ii, 1], truthNED[ii, 0], s=100, marker='^', facecolor='none', edgecolors='g',
                       label='True Pos.')
            ax.scatter(posmean[ii][1], posmean[ii][0], s=100, marker='s', facecolor='none', edgecolors='r',
                       label='Est. Pos.')
            plt.legend(loc='upper left')
            colbar = plt.colorbar(parts)
            colbar.set_label('Feature Match Weights')
            plt.xlabel('E(meters)')
            plt.ylabel('N(meters)')
            plt.title('Matched Database Feature Distribuiton at Image %d' % int(ii))
            plt.savefig('/home/sean/AFRL/pfresults/PF1/party_feat_img_pres_%d' % int(ii), bbox_inches='tight', dpi=400)

            print("pics saved")
            picsave = 0
            plt.close()
        picsave += 1

    # save estimated position and plot it
    posmean = np.array(posmean)
    posvar = np.array(posvar)
    radmean = np.array(radmean)
    radvar = np.array(radvar)
    radmuvar = np.stack((radmean, radvar)).transpose()
    poserr = np.array(poserr)
    experr = np.array(experr)
    FOV = np.abs(FOV).reshape(1349, 1)
    raderr = np.array(raderr).reshape(1349, 1)
    FOVerr = np.concatenate((FOV, raderr), axis=1)
    print(FOVerr.shape)

    # colorvari = np.sum(posvar,axis = 1)
    colorvari = np.sum(poserr, axis=1)
    posmuvar = np.hstack((posmean, posvar, poserr, radmuvar, FOVerr, experr))
    Position = np.hstack((posmean, posvar))
    Errors = np.concatenate((poserr, raderr, experr), axis=1)
    np.savetxt('/home/sean/AFRL/pfresults/PF1/PositionEst.txt', Position)
    np.savetxt('/home/sean/AFRL/pfresults/PF1/FOVEst.txt', radmuvar)
    np.savetxt('/home/sean/AFRL/pfresults/PF1/Errors.txt', Errors)
    np.save('/home/sean/AFRL/pfresults/PF1/partestpos.npy', posmuvar)
    poserr2d = (poserr[:, 0] ** 2) + (poserr[:, 1] ** 2)
    poserr2d = np.sqrt(poserr2d)
    Time = 2 * (np.arange(0, poserr2d.shape[0]))
    plt.figure()
    plt.plot(Time, poserr2d, marker='o')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('2D Position Error (Meters)')
    plt.title('2D Position Error')
    plt.savefig('/home/sean/AFRL/pfresults/PF1/2DPosErr')
    plt.figure()
    plt.plot(Time, raderr, marker='o')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('FOV est. error (Meters)')
    plt.title('Field of View estimation error')
    plt.savefig('/home/sean/AFRL/pfresults/PF1/FOVErr')
    plt.figure()
    plt.plot(Time, FOV, marker='o', label='True FOV')
    plt.plot(Time, radmean, marker='x', label='Est. FOV')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('FOV (Meters)')
    plt.title('Field of View')
    plt.legend(loc='upper right')
    plt.savefig('/home/sean/AFRL/pfresults/PF1/FOV')
    plt.figure()
    plt.xlim(erange[0], erange[1])
    plt.ylim(nrange[0], nrange[1])
    plt.plot(truthNED[:, 1], truthNED[:, 0], label='True Pos.')
    estm = plt.scatter(posmean[:, 1], posmean[:, 0], s=10, c=colorvari, cmap='plasma', label='Est. Pos.')
    colbar2 = plt.colorbar(estm)
    plt.legend(loc='lower right')
    colbar2.set_label('Estimated Position Error')
    plt.xlabel('E(meters)')
    plt.ylabel('N(meters)')
    plt.title('Estimated Position')
    plt.savefig('/home/sean/AFRL/pfresults/PF1/partyposinf', bbox_inches='tight', dpi=400)
    plt.figure()
    plt.plot(Time, poserr[:, 1], marker='o')
    plt.plot(Time, experr[:, 1], marker='x')
    plt.legend(loc='upper left')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('East Position Error (Meters)')
    plt.title('East Position Error')
    plt.savefig('/home/sean/AFRL/pfresults/PF1/EPosErr')
    plt.figure()
    plt.plot(Time, poserr[:, 0], marker='o')
    plt.plot(Time, experr[:, 0], marker='x')
    plt.legend(loc='upper left')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('North Position Error (Meters)')
    plt.title('North Position Error')
    plt.savefig('/home/sean/AFRL/pfresults/PF1/NPosErr')
    # plt.show()
