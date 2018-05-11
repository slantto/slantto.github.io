from neogeo import database as db
from neogeo import utils
from neogeo import extent as neoextent
import copy
import neogeodb.pytables_db as pdb
import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
from numpy.random import uniform
from numpy.random import randn
import scipy.stats
from numpy.linalg import norm
import multiprocessing as mp
import math
from functools import partial


class PartycleFilt(object):
    """

    """

    def __init__(self):
        """
        Class constructor for neogeo
        """
        self.extent = None
        # self.p_x = None
        self.db = None
        self.particles = None
        self.weights = None
        self.mp_pool = mp.Pool()

    def create_uniform_particles(self, x_range, y_range, rad_range, n):
        # uniformly distributes particles within search extent, each particle has a radius, within which the feature
        # matches should reside
        self.particles = np.empty((n, 3))
        self.particles[:, 0] = uniform(x_range[0], x_range[1], size=n)  # particle east location
        self.particles[:, 1] = uniform(y_range[0], y_range[1], size=n)  # particle north location
        self.particles[:, 2] = uniform(rad_range[0], rad_range[1], size=n)  # particle radius
        self.weights = np.ones(n)
        return self.particles, self.weights, n

    # def create_gaussian_particles(self, mean, std, radius, N):
    #     #same as the uniform creation of particles, except it normally distributes them about some "known" point
    #     self.particles = np.empty((N, 3))
    #     self.particles[:, 0] = mean[0] + (randn(N) * std[0])
    #     self.particles[:, 1] = mean[1] + (randn(N) * std[1])
    #     self.particles[:, 2] = radius
    #     self.weights = np.ones(N)
    #     return self.particles, self.weights

    def motion_model(self, n_e_delta_pos, delta_pos_sigma, rad_sigma):
        dx = n_e_delta_pos
        n = len(self.particles)
        move_n = dx[0] + (randn(n) * delta_pos_sigma[0])
        move_e = dx[1] + (randn(n) * delta_pos_sigma[1])
        self.particles[:, 0] += move_n
        self.particles[:, 1] += move_e
        self.particles[:, 2] += randn(n) * rad_sigma
        return self.particles

    def update(self, obs):
        # weight each particle by summing the weights of each matching feature inside the particle
        n = len(self.particles)
        my_tuning_param = .08
        meas_likelihood = np.ones(n) * my_tuning_param
        # self.weights.fill(1.)

        # add_weight = np.zeros(n)
        # The faster (hopefully) multi-threaded way
        func = partial(compute_meas, obs)
        add_weight = self.mp_pool.map(func, self.particles, mp.cpu_count() * 2)
        # print('First few meas before is ',meas_likelihood[0:10])
        meas_likelihood += np.array(add_weight)
        # meas_likelihood += np.array(add_weight)
        # The older, ?slower? way
        # for ii in np.arrange(self.particles.shape[0]):
        #    meas_likelihood[ii] += compute_meas(obs, self.particles[ii])
        # end comment if statement
        # print('Debug:  max diff between two meas_likelihoods is:', \
        #    np.max(np.abs(meas_likelihood-meas_likelihood2)))

        self.weights *= meas_likelihood
        # self.weights /= ((self.particles[:,2]**2)*math.pi)
        self.weights /= self.weights.sum()
        # print(self.weights.sum())
        return self.weights

    # def mpupdate(self, processes, obs):
    #     pool = mp.Pool(processes=processes)
    #     self.weights = pool.apply(update, args = (obs))
    #     return self.weights

    @property
    def estimate_pos(self):
        est_pos = self.particles[:, [0, 1]]
        pos_mean = np.average(est_pos, weights=self.weights, axis=0)
        pos_var = np.average((est_pos - pos_mean) ** 2, weights=self.weights, axis=0)
        partrad_mean = np.average(self.particles[:, 2], weights=self.weights)
        partrad_var = np.average((self.particles[:, 2] - partrad_mean) ** 2, weights=self.weights)
        particle_mean = np.hstack((pos_mean, partrad_mean))
        covariance = (1/(1-np.sum(self.weights[:]**2)))*np.dot((self.weights[:]*((self.particles[:] - particle_mean).transpose())), (self.particles[:] - particle_mean))

        return pos_mean, pos_var, partrad_mean, partrad_var, covariance

    def neff(self, n):
        # print(np.square(self.weights))
        neff = 1 / np.sum(np.square(self.weights))
        # print(neff)
        return neff

    def simple_resample(self, n, x_range, y_range, rad_range, res):
        # Resample only the percentage designated by 'res' of the particles only, and redistribute the remaining 10%
        # uniformly over search area
        cum_sum = np.cumsum(self.weights[:int(res * n)])
        cum_sum[-1] = 1
        indexes = np.searchsorted(cum_sum, uniform(size=int(res * n)))

        self.particles[:int(res * n)] = self.particles[indexes]
        if res < 1:
            self.particles[int(res * n):, 0] = uniform(x_range[0], x_range[1],
                                                       size=self.particles[int(res * n):, 0].shape[0])
            self.particles[int(res * n):, 1] = uniform(y_range[0], y_range[1],
                                                       size=self.particles[int(res * n):, 1].shape[0])
            self.particles[int(res * n):, 2] = uniform(rad_range[0], rad_range[1],
                                                       size=self.particles[int(res * n):, 2].shape[0])
        self.weights[:] = np.ones(n)
        return self.particles, self.weights


def compute_meas(obs, particle):
    nedist = particle[0:2] - obs[:, [1, 2]]
    # print(self.particles[ii,[0,1]])
    # print(obs[:,[1,2]])
    nedist = norm(nedist, axis=1)
    # N = len(particle)
    w2sumidx = np.where(nedist <= particle[2])[0]
    w2sum = obs[w2sumidx, 0]

    # w2sum = []
    # for jj in np.arange(obs.shape[0]):
    #     if nedist[jj] <= self.particles[ii,2]:
    #         #print("in")
    #         w2sum.append(obs[jj,0])
    # w2sum = np.array(w2sum)
    if w2sum.shape[0] > 0:
        # print(w2sum)
        # print(w2sumidx)
        return (np.sum(w2sum) * 5000) / (1 * particle[2] * particle[2])
        # meas_likelihood[ii] /= ((self.particles[ii,2]**2)*math.pi)
        # print(self.weights[ii])
    else:
        return 0
