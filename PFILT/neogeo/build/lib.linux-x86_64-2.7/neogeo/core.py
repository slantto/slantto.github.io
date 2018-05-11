from . import database as db
from . import utils
from . import extent as neoextent
import copy
import neogeodb.pytables_db as pdb
import numpy as np
import scipy.ndimage as ndimage


class NeoGeo(object):

    """
    This object is a base class that outlines the structure of the
    class needed to implement neogeo, which is a framework for large-scale
    image database navigation.

    The class contains the following instance variables

    :ivar p_x: Non parametric representation of the PDF over a section of\
        the database
    :ivar extent: Current search extent in which the database is being searched
    :ivar db: Database to search from, currently implemented in Pytables
    """

    def __init__(self):
        """
        Class constructor for neogeo
        """
        self.extent = None
        self.p_x = None
        self.db = None

    def load_database(self, database):
        """
        Loads the database into neogeo for further access. Databases need
        functionality to perform data reduction based on the scale of the
        search being conducted. Databases also need to provide error handling
        when trying to access area out of bounds
        """
        self.db = db.PyTablesDatabase(database)
        self.extent = neoextent.SearchExtent(self.db.zoom, self.db.xb,
                                             self.db.yb, self.tid,
                                             self.tidcount)
        self.pix_size_m = neoextent.get_average_tile_size(self.extent)

    def use_posterior(self, posterior):
        """
        This can either be used as method to initialize the posterior
        distribution of the grid
        """
        raise NotImplementedError

    def motion_model(self, n_e_delta_pos, delta_pos_sigma):
        dx = n_e_delta_pos
        dx[0] = -1*dx[0]
        # Convert velocity to pixels
        dx = dx / self.pix_size_m
        self.p_x = ndimage.interpolation.shift(self.p_x, dx, order=1,
                                               mode='constant',
                                               cval=self.p_empty,
                                               prefilter=True)
        y_blur_sig = delta_pos_sigma[0] / self.pix_size_m
        x_blur_sig = delta_pos_sigma[1] / self.pix_size_m
        self.p_x = ndimage.gaussian_filter(self.p_x,
                                           sigma=(y_blur_sig, x_blur_sig))

    def update(self, obs):
        """
        Blend PX with OBS coming from the weighted observation likelihood
        """
        obs = np.maximum(obs, self.obs_0)
        obs = obs / obs.sum()
        self.p_x = self.p_x * obs
        self.p_x = self.p_x / self.p_x.sum()

    def vocab_update(self, obs):
        """
        Blend PX with OBS coming from the weighted observation likelihood
        """
        obs[obs == 0.0] = obs[obs > 0].min()
        obs = obs / obs.sum()
        self.p_x = self.p_x * obs
        self.p_x = self.p_x / self.p_x.sum()

    def reinit(self):
        self.p_x = np.copy(self.px_0)

    def init_px_from_extent(self, extent):
        """
        Returns a search extent by doing some initial analysis on the database
        """
        self.extent = extent
        self.pix_size_m = neoextent.get_average_tile_size(self.extent)
        self.p_x = np.zeros((self.extent.grid_size, self.extent.grid_size))
        tilexy = np.array([pdb.unpair(tt) for tt in self.extent.tiles])
        tilexy = tilexy.astype(np.int)
        tile_gen = neoextent.get_tiles_in_extent(self.extent)
        all_tiles = np.array([tile for tile in tile_gen])
        empty_tiles = np.setdiff1d(all_tiles, self.extent.tiles)
        emptyxy = np.array([pdb.unpair(tt) for tt in empty_tiles])
        emptyxy = emptyxy.astype(np.int)
        filled_tile_prob = 1.0
        empty_tile_prob = 0.5
        p0 = utils.plot_on_grid(emptyxy[:, 0], emptyxy[:, 1], empty_tile_prob,
                                self.extent.xb, self.extent.yb)
        p1 = utils.plot_on_grid(tilexy[:, 0], tilexy[:, 1], filled_tile_prob,
                                self.extent.xb, self.extent.yb)
        self.p_x = p0 + p1
        self.p_x = self.p_x / self.p_x.sum()
        self.p_empty = np.copy(self.p_x.min())
        self.p_filled = filled_tile_prob
        self.px_0 = np.copy(self.p_x)
        self.obs_0 = p0 + p1
