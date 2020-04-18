# -*- coding: utf-8 -*-
"""reV tech mapping framework.

This module manages the exclusions-to-resource mapping.
The core of this module is a parallel cKDTree.

Created on Fri Jun 21 16:05:47 2019

@author: gbuster
"""
import h5py
from concurrent.futures import as_completed
import numpy as np
import os
from scipy.spatial import cKDTree
import logging
from warnings import warn

from reV.supply_curve.points import SupplyCurveExtent
from reV.utilities.exceptions import FileInputWarning, FileInputError

from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool

logger = logging.getLogger(__name__)


class TechMapping:
    """Framework to create map between tech layer (exclusions), res, and gen"""

    def __init__(self, excl_fpath, res_fpath, dset, distance_upper_bound=0.03,
                 map_chunk=2560, max_workers=None):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 (tech layer). dset will be
            created in excl_fpath.
        res_fpath : str
            Filepath to .h5 resource file that we're mapping to.
        dset : str
            Dataset name in excl_fpath to save mapping results to.
        distance_upper_bound : float | None
            Upper boundary distance for KNN lookup between exclusion points and
            resource points. None will calculate a good distance based on the
            resource meta data coordinates. 0.03 is a good value for a 4km
            resource grid and finer.
        map_chunk : int | None
            Calculation chunk used for the tech mapping calc.
        max_workers : int | None
            Number of cores to run mapping on. None uses all available cpus.
        """

        self._distance_upper_bound = distance_upper_bound
        self._excl_fpath = excl_fpath
        self._res_fpath = res_fpath
        self._dset = dset
        self._check_fout()
        self._map_chunk = map_chunk

        if max_workers is None:
            max_workers = os.cpu_count()
        self._max_workers = max_workers

        with SupplyCurveExtent(self._excl_fpath,
                               resolution=self._map_chunk) as sc:
            self._map_chunk = sc._res
            self._n_sc = len(sc)
            self._excl_shape = sc.exclusions.shape
            self._n_excl = (self._excl_shape[0] * self._excl_shape[1])
            logger.info('Initialized TechMapping object with {} calc chunks '
                        'for {} tech exclusion points'
                        .format(len(sc), self._n_excl))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            raise

    def _check_fout(self):
        """Check the TechMapping output file for cached data."""

        emsg = None
        wmsg = None
        if os.path.exists(self._excl_fpath):
            with h5py.File(self._excl_fpath, 'r') as f:

                if 'latitude' not in f or 'longitude' not in f:
                    emsg = ('Datasets "latitude" and/or "longitude" not in '
                            'pre-existing Exclusions TechMapping file "{}". '
                            'Cannot proceed.'
                            .format(os.path.basename(self._excl_fpath)))

        if wmsg is not None:
            logger.warning(wmsg)
            warn(wmsg, FileInputWarning)

        if emsg is not None:
            logger.exception(emsg)
            raise FileInputError(emsg)

    def _init_out_arrays(self):
        """Initialize full sized output arrays.

        Returns
        -------
        ind : np.ndarray
            1D integer array filled with -1's with length equal to the number
            of tech exclusion points in the supply curve extent.
        coords : np.ndarray
            2D integer array (N, 2) filled with 0's with length equal to the
            number of tech exclusion points in the supply curve extent.
        """

        ind = -1 * np.ones((self._n_excl, ), dtype=np.int32)
        coords = np.zeros((self._n_excl, 2), dtype=np.float32)

        return ind, coords

    @property
    def distance_upper_bound(self):
        """Get the upper bound on NN distance between excl and res points.

        Returns
        -------
        distance_upper_bound : float
            Estimate of the upper bound distance based on the distance between
            resource points. Calculated as half of the diagonal between
            closest resource points, with an extra 5% margin.
        """

        if self._distance_upper_bound is None:

            with Resource(self._res_fpath, str_decode=False) as res:
                lats = res.get_meta_arr('latitude')

            dists = np.abs(lats - np.roll(lats, 1))
            dists = dists[(dists != 0)]
            self._distance_upper_bound = 1.05 * (2 ** 0.5) * (dists.min() / 2)

            logger.info('Distance upper bound was infered to be: {}'
                        .format(self._distance_upper_bound))

        return self._distance_upper_bound

    @staticmethod
    def _unpack_coords(gids, sc, excl_fpath,
                       coord_labels=('latitude', 'longitude')):
        """Unpack the exclusion layer coordinates for TechMapping.

        Parameters
        ----------
        gids : np.ndarray
            Supply curve gids with tech exclusion points to map to the
            resource meta points.
        sc : SupplyCurveExtent
            reV supply curve extent object
        excl_fpath : str
            .h5 filepath to save exclusions tech mapping results to, or to
            read from (if exists).
        coord_labels : tuple
            Labels for the coordinate datasets.

        Returns
        -------
        coords_out : list
            List of arrays of the un-projected latitude, longitude array of
            tech exclusion points. List entries correspond to input gids.
        lat_range : tuple
            Latitude (min, max) values associated with the un-projected
            coordinates of the input gids.
        lon_range : tuple
            Longitude (min, max) values associated with the un-projected
            coordinates of the input gids.
        """

        coords_out = []
        lat_range = None
        lon_range = None

        with h5py.File(excl_fpath, 'r') as f:
            for gid in gids:
                row_slice, col_slice = sc.get_excl_slices(gid)
                try:
                    lats = f[coord_labels[0]][row_slice, col_slice]
                    lons = f[coord_labels[1]][row_slice, col_slice]
                    emeta = np.vstack((lats.flatten(), lons.flatten())).T
                except Exception as e:
                    m = ('Could not unpack coordinates for gid {} with '
                         'row/col slice {}/{}. Received the following '
                         'error:\n{}'.format(gid, row_slice, col_slice, e))
                    logger.error(m)
                    raise e

                coords_out.append(emeta)

                if lat_range is None:
                    lat_range = [np.min(emeta[:, 0]), np.max(emeta[:, 0])]
                    lon_range = [np.min(emeta[:, 1]), np.max(emeta[:, 1])]
                else:
                    lat_range[0] = np.min((lat_range[0], np.min(emeta[:, 0])))
                    lat_range[1] = np.max((lat_range[1], np.max(emeta[:, 0])))
                    lon_range[0] = np.min((lon_range[0], np.min(emeta[:, 1])))
                    lon_range[1] = np.max((lon_range[1], np.max(emeta[:, 1])))

        return coords_out, lat_range, lon_range

    def _parallel_resource_map(self):
        """Map all resource gids to exclusion gids in parallel.

        Returns
        -------
        lats : np.ndarray
            2D un-projected latitude array of tech exclusion points.
            0's if no res point found. Shape is equal to exclusions shape.
        lons : np.ndarray
            2D un-projected longitude array of tech exclusion points.
            0's if no res point found. Shape is equal to exclusions shape.
        ind_all : np.ndarray
            Index values of the NN resource point. -1 if no res point found.
            2D integer array with shape equal to the exclusions extent shape.
        """

        gids = np.array(list(range(self._n_sc)), dtype=np.uint32)
        gid_chunks = np.array_split(gids, int(np.ceil(len(gids) / 2)))

        # init full output arrays
        ind_all, coords_all = self._init_out_arrays()

        n_finished = 0
        futures = {}
        loggers = __name__
        with SpawnProcessPool(max_workers=self._max_workers,
                              loggers=loggers) as exe:

            # iterate through split executions, submitting each to worker
            for i, gid_set in enumerate(gid_chunks):
                # submit executions and append to futures list
                futures[exe.submit(self.map_resource_gids,
                                   gid_set,
                                   self._excl_fpath,
                                   self._res_fpath,
                                   self.distance_upper_bound,
                                   self._map_chunk)] = i

            for future in as_completed(futures):
                n_finished += 1
                logger.info('Parallel TechMapping futures collected: '
                            '{} out of {}'
                            .format(n_finished, len(futures)))

                i = futures[future]
                result = future.result()

                res = self._map_chunk
                with SupplyCurveExtent(self._excl_fpath, resolution=res) as sc:
                    for j, gid in enumerate(gid_chunks[i]):
                        i_out_arr = sc.get_flat_excl_ind(gid)
                        ind_all[i_out_arr] = result[0][j]
                        coords_all[i_out_arr, :] = result[1][j]

        ind_all = ind_all.reshape(self._excl_shape)
        lats = coords_all[:, 0].reshape(self._excl_shape)
        lons = coords_all[:, 1].reshape(self._excl_shape)

        return lats, lons, ind_all

    @staticmethod
    def map_resource_gids(gids, excl_fpath, res_fpath, distance_upper_bound,
                          map_chunk, margin=0.1):
        """Map exclusion gids to the resource meta.

        Parameters
        ----------
        gids : np.ndarray
            Supply curve gids with tech exclusion points to map to the
            resource meta points.
        excl_fpath : str
            Filepath to exclusions h5 (tech layer). dset will be
            created in excl_fpath.
        res_fpath : str
            Filepath to .h5 resource file that we're mapping to.
        distance_upper_bound : float | None
            Upper boundary distance for KNN lookup between exclusion points and
            resource points.
        map_chunk : int
            Calculation chunk used for the tech mapping calc.
        margin : float
            Margin when reducing the resource lat/lon.

        Returns
        -------
        ind : list
            List of arrays of index values from the NN. List entries correspond
            to input gids.
        coords : np.ndarray
            List of arrays of the un-projected latitude, longitude array of
            tech exclusion points. List entries correspond to input gids.
        """

        logger.debug('Getting tech layer coordinates for chunks {} through {}'
                     .format(gids[0], gids[-1]))

        ind_out = []
        coord_labels = ['latitude', 'longitude']

        with SupplyCurveExtent(excl_fpath, resolution=map_chunk) as sc:
            coords_out, lat_range, lon_range = TechMapping._unpack_coords(
                gids, sc, excl_fpath, coord_labels=coord_labels)

        with Resource(res_fpath, str_decode=False) as res:
            res_meta = np.vstack((res.get_meta_arr(coord_labels[0]),
                                  res.get_meta_arr(coord_labels[1]))).T

        mask = ((res_meta[:, 0] > lat_range[0] - margin)
                & (res_meta[:, 0] < lat_range[1] + margin)
                & (res_meta[:, 1] > lon_range[0] - margin)
                & (res_meta[:, 1] < lon_range[1] + margin))

        # pylint: disable-msg=C0121
        mask_ind = np.where(mask == True)[0]  # noqa: E712

        if np.sum(mask) > 0:
            res_tree = cKDTree(res_meta[mask, :])

            logger.debug('Running tech mapping for chunks {} through {}'
                         .format(gids[0], gids[-1]))
            for i, _ in enumerate(gids):
                dist, ind = res_tree.query(coords_out[i])
                ind = mask_ind[ind]
                ind[(dist > distance_upper_bound)] = -1
                ind_out.append(ind)
        else:
            logger.debug('No close res points for chunks {} through {}'
                         .format(gids[0], gids[-1]))
            for _ in gids:
                ind_out.append(-1)

        return ind_out, coords_out

    @staticmethod
    def save_tech_map(lats, lons, ind, fpath_out, res_fpath, dset,
                      distance_upper_bound, chunks=(128, 128)):
        """Save tech mapping indices and coordinates to an h5 output file.

        Parameters
        ----------
        lats : np.ndarray
            2D un-projected latitude array of tech exclusion points.
            0's if no res point found. Shape is equal to exclusions shape.
        lons : np.ndarray
            2D un-projected longitude array of tech exclusion points.
            0's if no res point found. Shape is equal to exclusions shape.
        ind : np.ndarray
            Index values of the NN resource point. -1 if no res point found.
            2D integer array with shape equal to the exclusions extent shape.
        fpath_out : str
            .h5 filepath to save tech mapping results.
        res_fpath : str
            Filepath to .h5 resource file that we're mapping to.
        dset : str
            Dataset name in fpath_out to save mapping results to.
        distance_upper_bound : float
            Distance upper bound to save as attr.
        chunks : tuple
            Chunk shape of the 2D output datasets.
        """

        if not fpath_out.endswith('.h5'):
            fpath_out += '.h5'

        logger.info('Writing tech map "{}" to {}'.format(dset, fpath_out))

        shape = ind.shape
        chunks = (np.min((shape[0], chunks[0])), np.min((shape[1], chunks[1])))

        if not os.path.exists(fpath_out):
            if lats is not None and lons is None:
                with h5py.File(fpath_out, 'w') as f:
                    f.create_dataset('latitude', shape=shape,
                                     dtype=lats.dtype,
                                     data=lats,
                                     chunks=chunks)
                    f.create_dataset('longitude', shape=shape,
                                     dtype=lons.dtype,
                                     data=lons,
                                     chunks=chunks)

        with h5py.File(fpath_out, 'a') as f:
            if dset in list(f):
                wmsg = ('TechMap results dataset "{}" is being replaced '
                        'in pre-existing Exclusions TechMapping file "{}"'
                        .format(dset, fpath_out))
                logger.warning(wmsg)
                warn(wmsg, FileInputWarning)
                f[dset][...] = ind
            else:
                f.create_dataset(dset, shape=shape, dtype=ind.dtype,
                                 data=ind, chunks=chunks)

            f[dset].attrs['fpath'] = res_fpath
            f[dset].attrs['distance_upper_bound'] = distance_upper_bound

        logger.info('Successfully saved tech map "{}" to {}'
                    .format(dset, fpath_out))

    @classmethod
    def run(cls, excl_fpath, res_fpath, dset, save_flag=True,
            distance_upper_bound=0.03, map_chunk=2560, max_workers=None):
        """Run parallel mapping and save to h5 file.

        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 (tech layer). dset will be
            created in excl_fpath.
        res_fpath : str
            Filepath to .h5 resource file that we're mapping to.
        dset : str
            Dataset name in excl_fpath to save mapping results to.
        save_flag : bool
            Flag to write techmap to excl_fpath.
        kwargs : dict
            Keyword args to initialize the TechMapping object.

        Returns
        -------
        lats : np.ndarray
            2D un-projected latitude array of tech exclusion points.
            0's if no res point found. Shape is equal to exclusions shape.
        lons : np.ndarray
            2D un-projected longitude array of tech exclusion points.
            0's if no res point found. Shape is equal to exclusions shape.
        ind_all : np.ndarray
            Index values of the NN resource point. -1 if no res point found.
            2D integer array with shape equal to the exclusions extent shape.
        """
        kwargs = {"distance_upper_bound": distance_upper_bound,
                  "map_chunk": map_chunk, "max_workers": max_workers}
        with cls(excl_fpath, res_fpath, dset, **kwargs) as mapper:
            lats, lons, ind = mapper._parallel_resource_map()
            distance_upper_bound = mapper._distance_upper_bound

        if save_flag:
            mapper.save_tech_map(lats, lons, ind, excl_fpath, res_fpath,
                                 dset, distance_upper_bound)

        return lats, lons, ind
