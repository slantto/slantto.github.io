#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script processes the output from neogeoPF, which provides the feature matches for the images. This script uses a
particle filter to estimate the position of the aircraft, using the feature matches as the measurement likelihood, and
integrated IMU data as the prediction of motion. This script assumes no known initial position.

Includes a Down position state
"""

import neogeo.extent as neoextent
import numpy as np
import pandas as pd
import pnpnav.utils as pnputils
import tables as tb
import particle_core2 as core
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

if __name__ == '__main__':
    # Load Flight data
    f5 = tb.open_file('/media/sean/My Passport/data_files/fc2_f5.hdf', 'r')  # flight data
    f5featmeta = pd.read_hdf(
        '/media/sean/My Passport/data_files/AIEoutput2/feat/feat_meta.hdf')  # fight image feature data
    imgs = f5.root.camera.image_raw.compressed.images  # flight images
    img_times = f5.root.camera.image_raw.compressed.metadata.col('t_valid')  # flight image times
    feat_path = f5featmeta.iloc[:, 3]  # flight image sift features
    descripath = f5featmeta.iloc[:, 4]  # flight image sift descriptors
    pva = f5.root.nov_span.pva  # flight navigation data
    # Load Camera and terrain data
    dted_path = '/media/sean/My Passport/data_files/srtm'  # digital terrain elevation data
    geoid_file = '/media/sean/My Passport/data_files/egm96-15.tif'  # geoid file
    frame_yaml = '/home/sean/AFRL/ParticleFilterVisNav/pyPNP/data/fc2_pod_frames.yaml'  # camera frame
    # Load in Camera to body frame DCMs and Camera Calibrations
    finder = pnputils.DEMTileFinder(dted_path, geoid_file)
    finder.load_cam_and_vehicle_frames(frame_yaml)
    finder.load_camera_cal('/home/sean/AFRL/ParticleFilterVisNav/pyPNP/data/fc2_cam_model.yaml')
    # Load in precomputed observabations from neobinPF
    out_tb = tb.open_file('/media/sean/My Passport/data_files/obs_out_forPF.hdf', 'r')
    neogeo_out = out_tb.root.loaded_1000000.feat_10000.neogeo_out  # Number of observations
    obs_out = out_tb.root.loaded_1000000.feat_10000.obs_out  # Observation weight
    # Load truth location and Velocity
    t_loc = np.load('/media/sean/My Passport/data_files/t_loc.npy')
    ned_vel = np.load('/media/sean/My Passport/data_files/ned_vel.npy')
    # Get the Feature Database
    dbf = tb.open_file(
        '/media/sean/My Passport/data_files/pytables_db.hdf', 'a')  # Entire database
    dbt = dbf.get_node('/sift_db/sift_features_sorted')  # Sorted sift features

    # Find times that match between IMU and images
    tmatchidx = []  # Empty time match array
    existmatch = 0  # Match exists or doesnt, helps prevent duplicates that result from rounding
    for pp in np.arange(0, pva.cols.t_valid.shape[0]):
        pvatround = round(pva.cols.t_valid[pp])  # Nav System times need rounded for matching
        for tt in np.arange(0, img_times.shape[0]):
            imgtround = round(img_times[tt])  # Image times also need rounded for matching
            if pvatround == imgtround:
                if existmatch == 0:  # If a match is found and there is no other match like it
                    tmatchidx.append(pp)  # add time to match array
                    existmatch = 1  # make sure no duplicate will be made
                    break
                if existmatch == 1:  # If match is found and a duplicate exists
                    existmatch = 0  # Reset duplicate flag
                    break
    tmatchidx = np.array(tmatchidx)  # time match index as a numpy array

    # Set boundaries for the search area
    txmin = dbt.cols.x[int(dbt.colindexes['x'][0])].astype(np.int64)
    txmax = dbt.cols.x[int(dbt.colindexes['x'][-1])].astype(np.int64)
    tymin = dbt.cols.y[int(dbt.colindexes['y'][0])].astype(np.int64)
    tymax = dbt.cols.y[int(dbt.colindexes['y'][-1])].astype(np.int64)
    xbounds = (txmin, txmax)
    ybounds = (tymin, tymax)
    # print(xbounds)
    # print(ybounds)
    xb, yb = neoextent.pad_grid(xbounds, ybounds)
    neo = core.PartycleFilt()
    tid, tidcount = np.unique(dbt.cols.pair_id, return_counts=True)

    extent = neoextent.SearchExtent(15, xb, yb, tid, tidcount)  # Defines quadtree structure to represent search area

    # get LLA corners of the search area
    bbox = neoextent.bbox_from_extent(extent)  # long, lat tuples for each corner
    c_wgs = np.hstack((np.array(bbox)[:, [1, 0]], np.zeros((4, 1))))  # arranges corner positions into numpy array
    # Convert LLA corners to NED
    c_ned = np.zeros((c_wgs.shape[0], 3))
    c_ned = navpy.lla2ned(c_wgs[:, 0], c_wgs[:, 1], c_wgs[:, 2], c_wgs[3, 0], c_wgs[3, 1], c_wgs[3, 2])  # NED corners
    # NED range used for particle distribution and plotting
    nrange = (c_ned[:, 0].min(), c_ned[:, 0].max())
    erange = (c_ned[:, 1].min(), c_ned[:, 1].max())
    drange = (c_ned[:, 2].min(), c_ned[:, 2].max())

    MotionTime = np.arange(tmatchidx[0], (tmatchidx[-1] + 1))  # Idx to ensure mes update occurs at proper time,
    # MotionTime is just the index not the actual time
    # Get the truth position from Nova_span
    truthLLA = np.zeros((MotionTime.shape[0], 3))
    VelNED = np.zeros((MotionTime.shape[0], 3))
    for kk in np.arange(0, MotionTime.shape[0]):
        # Truth position in LLA
        truthLLA[kk, :] = np.array(
            (pva[MotionTime[kk]]['lat'], pva[MotionTime[kk]]['lon'], pva[MotionTime[kk]]['height'])).transpose()
        VelNED[kk, :] = np.array(
            (pva[MotionTime[kk]]['vel_n'], pva[MotionTime[kk]]['vel_e'],
             pva[MotionTime[kk]]['vel_d'])).transpose()  # Velocities to propagate particles in predict step
    truthNED = navpy.lla2ned(truthLLA[:, 0], truthLLA[:, 1], truthLLA[:, 2], c_wgs[3, 0], c_wgs[3, 1],
                             c_wgs[3, 2])  # Truth position in NED

    # Camera has a FOV half angle fo 8.3 degrees
    halfang = 8.3 * np.pi / 180
    FOV = truthNED[:, 2] * np.tan(halfang)

    # initialize particles within bounding box
    rad_range = (25, 1000)  # Partical field of view (FOV) range
    particles, weights, N = neo.create_uniform_particles(nrange, erange, drange, 25000)  # Uniform Distribution

    # Make some empty arrays and set some variables to desired initial value
    posmean = []
    posvar = []
    radmean = []
    radvar = []
    poserr = []
    raderr = []
    experr = []
    ResampIdx = np.zeros((5621, 1))
    PFtime = np.zeros((5621, 1))
    picsave = 10
    elapimg = 0
    resamp = 1
    ImgTime = 0
    dpos = np.array((0, 0, 0))
    for ii in np.arange(0, truthNED.shape[0]):
        # print(ii)

        # Prediction Step
        if ii > 0:
            timer = (round(pva.cols.t_valid[MotionTime[ii]], 2)) - (round(pva.cols.t_valid[MotionTime[ii - 1]], 2))
            PFtime[ii] = PFtime[ii-1] + timer
            print(float(timer))
            dpos = (VelNED[ii, 0:2] * timer)
        dpos_sig = 0.75 * np.linalg.norm(dpos) * np.ones(3)  # previously 0.1
        rad_sig = 1
        particles = neo.motion_model(dpos, dpos_sig, rad_sig)

        # Measurement Step

        if MotionTime[ii] == tmatchidx[ImgTime]:
            print('MesUp')

            imgobs = obs_out[ImgTime, :, :]
            imgobs = imgobs[~(imgobs == 0).all(1)]
            if imgobs.shape[0] > 0:
                imgobs[:, 1:] = navpy.lla2ned(imgobs[:, 1], imgobs[:, 2], imgobs[:, 3], c_wgs[3, 0], c_wgs[3, 1],
                                              c_wgs[3, 2])
            weights = neo.update(imgobs)
            ImgTime += 1

            neff = neo.neff(N)
            # print(neff)
            elapimg += 1  # count number of elapsed images since last re-sample

            if neff < (N / 2):
                print("resampling")

                resamp = 1 - (elapimg * 0.001)  # the constant 0.001 has been decided upon, because it will only
                # scatter 10 particles for every image elapsed, the need for a new constant will be determined
                # empirically.
                if resamp < 0:
                    resamp = 1  # if too many images have elapsed since last re-sample, don't scatter. This will
                    # likely only happen at the beginning of the data set

                particles, weights = neo.simple_resample(N, nrange, erange, rad_range, resamp)
                ResampIdx[ii] = 1  # Resampling Flag

                elapimg = 0
                resamp = 1

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
        # print(posvar)

        if picsave == 10:
            plt.figure()
            ax = plt.gca()
            ax.set_xlim(erange[0], erange[1])
            ax.set_ylim(nrange[0], nrange[1])
            ax.autoscale(False)
            # obslocs = ax.scatter(imgobs[:, 1], imgobs[:, 2], s=5, marker='x', facecolor='k', label='Features')
            parts = ax.scatter(particles[:, 1], particles[:, 0], s=5, c=particles[:, 2], cmap='plasma',
                               label='Particles')
            ax.scatter(truthNED[ii, 1], truthNED[ii, 0], s=15, marker='^', facecolor='g', edgecolors='g',
                       label='True Pos.')
            ax.scatter(posmean[ii][1], posmean[ii][0], s=15, marker='x', facecolor='r', edgecolors='r',
                       label='Est. Pos.')
            lambda_, v = np.linalg.eig(covariance)
            lambda_ = np.sqrt(lambda_)
            ell = Ellipse(xy=[posmean[ii][1], posmean[ii][0]], width=lambda_[1] * 3 * 2, height=lambda_[0] * 3 * 2,
                          angle=np.rad2deg(np.arccos(v[0, 0])), linewidth=1, facecolor='none', edgecolor='r')
            ax.add_patch(ell)
            plt.legend(loc='upper left')
            colbar = plt.colorbar(parts)
            colbar.set_label('Particle Radius(meters)')
            plt.xlabel('E(meters)')
            plt.ylabel('N(meters)')
            plt.title('Particle Distribution at Time %d' % float(PFtime[ii]))
            plt.savefig('/home/sean/AFRL/pfresults/PF21/party_img_pres_%d' % int(ii), bbox_inches='tight', dpi=400)

            plt.figure()
            ax = plt.gca()
            ax.set_xlim(erange[0], erange[1])
            ax.set_ylim(nrange[0], nrange[1])
            ax.autoscale(False)
            obslocs = ax.scatter(imgobs[:, 2], imgobs[:, 1], s=10, marker='^', c=imgobs[:, 0], cmap='viridis',
                                 label='Features')
            ax.scatter(truthNED[ii, 1], truthNED[ii, 0], s=15, marker='o', facecolor='g', edgecolors='g',
                       label='True Pos.')
            ax.scatter(posmean[ii][1], posmean[ii][0], s=15, marker='x', facecolor='r', edgecolors='r',
                       label='Est. Pos.')
            ell = Ellipse(xy=[posmean[ii][1], posmean[ii][0]], width=lambda_[1] * 3 * 2, height=lambda_[0] * 3 * 2,
                          angle=np.rad2deg(np.arccos(v[0, 0])), linewidth=1, facecolor='none', edgecolor='r')
            ax.add_patch(ell)
            plt.legend(loc='upper left')
            colbar = plt.colorbar(parts)
            colbar.set_label('Feature Match Weights')
            plt.xlabel('E(meters)')
            plt.ylabel('N(meters)')
            plt.title('Matched Database Feature Distribution at Time %d' % float(PFtime[ii]))
            plt.savefig('/home/sean/AFRL/pfresults/PF21/party_feat_img_pres_%d' % int(ii), bbox_inches='tight', dpi=400)

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
    FOV = np.abs(FOV).reshape(5621, 1)
    raderr = np.array(raderr).reshape(5621, 1)
    FOVerr = np.concatenate((FOV, raderr), axis=1)
    # PFtime = np.array(PFtime).reshape(5621, 1)
    print(FOVerr.shape)

    # colorvari = np.sum(posvar,axis = 1)
    colorvari = np.sum(poserr, axis=1)
    posmuvar = np.hstack((posmean, posvar, poserr, radmuvar, FOVerr, experr, PFtime))
    Position = np.hstack((posmean, posvar))
    Errors = np.concatenate((poserr, raderr, experr), axis=1)
    np.savetxt('/home/sean/AFRL/pfresults/PF21/PositionEst.txt', Position)
    np.savetxt('/home/sean/AFRL/pfresults/PF21/FOVEst.txt', radmuvar)
    np.savetxt('/home/sean/AFRL/pfresults/PF21/Errors.txt', Errors)
    np.savetxt('/home/sean/AFRL/pfresults/PF21/Time.txt', PFtime)
    np.savetxt('/home/sean/AFRL/pfresults/PF21/ResampIdx.txt', ResampIdx)
    np.save('/home/sean/AFRL/pfresults/PF21/partestpos.npy', posmuvar)
    poserr2d = (poserr[:, 0] ** 2) + (poserr[:, 1] ** 2)
    poserr2d = np.sqrt(poserr2d)
    Time = PFtime
    # Time = 2 * (np.arange(0, poserr2d.shape[0]))
    plt.figure()
    plt.plot(Time, poserr2d, marker='o')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('2D Position Error (Meters)')
    plt.title('2D Position Error')
    plt.savefig('/home/sean/AFRL/pfresults/PF21/2DPosErr')
    plt.figure()
    plt.plot(Time, raderr, marker='o')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('FOV est. error (Meters)')
    plt.title('Field of View estimation error')
    plt.savefig('/home/sean/AFRL/pfresults/PF21/FOVErr')
    plt.figure()
    plt.plot(Time, FOV, marker='o', label='True FOV')
    plt.plot(Time, radmean, marker='x', label='Est. FOV')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('FOV (Meters)')
    plt.title('Field of View')
    plt.legend(loc='upper right')
    plt.savefig('/home/sean/AFRL/pfresults/PF21/FOV')
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
    plt.savefig('/home/sean/AFRL/pfresults/PF21/partyposinf', bbox_inches='tight', dpi=400)
    plt.figure()
    plt.plot(Time, poserr[:, 1], marker='o')
    plt.plot(Time, experr[:, 1], marker='x')
    plt.legend(loc='upper left')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('East Position Error (Meters)')
    plt.title('East Position Error')
    plt.savefig('/home/sean/AFRL/pfresults/PF21/EPosErr')
    plt.figure()
    plt.plot(Time, poserr[:, 0], marker='o')
    plt.plot(Time, experr[:, 0], marker='x')
    plt.legend(loc='upper left')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('North Position Error (Meters)')
    plt.title('North Position Error')
    plt.savefig('/home/sean/AFRL/pfresults/PF21/NPosErr')
    # plt.show()
