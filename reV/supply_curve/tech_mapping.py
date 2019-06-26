# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:05:47 2019

@author: gbuster
"""
import h5py
import concurrent.futures as cf
import numpy as np
import os
from scipy.spatial import cKDTree
import logging

from reV.handlers.outputs import Outputs
from reV.supply_curve.points import SupplyCurveExtent


logger = logging.getLogger(__name__)


class TechMapping:
    """Framework to create map between tech layer (exclusions) and gen."""

    def __init__(self, fpath_excl, fpath_gen, distance_upper_bound=0.03,
                 resolution=2560, n_cores=None):
        """
        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff (tech layer).
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        distance_upper_bound : float | None
            Upper boundary distance for KNN lookup between exclusion points and
            generation points. None will calculate a good distance based on the
            generation meta data coordinates. 0.03 is a good value for a 4km
            resource grid and finer.
        resolution : int | None
            Supply curve resolution used for the tech mapping calc. This is not
            the final supply curve point resolution.
        n_cores : int | None
            Number of cores to run mapping on. None uses all available cpus.
        """

        self._distance_upper_bound = distance_upper_bound
        self._fpath_excl = fpath_excl
        self._fpath_gen = fpath_gen
        self._sc = SupplyCurveExtent(fpath_excl, resolution=resolution)
        self._resolution = self._sc._res

        if n_cores is None:
            n_cores = os.cpu_count()
        self._n_cores = n_cores

        logger.info('Initialized TechMapping object with {} calc chunks for '
                    '{} tech exclusion points'
                    .format(len(self._sc), len(self._sc.exclusions)))

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

        N = len(self._sc.exclusions)
        ind = -1 * np.ones((N, ), dtype=np.int32)
        coords = np.zeros((N, 2), dtype=np.float32)
        return ind, coords

    @property
    def distance_upper_bound(self):
        """Get the upper bound on NN distance between excl and gen points.

        Returns
        -------
        distance_upper_bound : float
            Estimate of the upper bound distance based on the distance between
            generation points. Calculated as half of the diagonal between
            closest generation points, with an extra 5% margin.
        """

        if self._distance_upper_bound is None:

            with Outputs(self._fpath_gen, str_decode=False) as o:
                lats = o.get_meta_arr('latitude')

            dists = np.abs(lats - np.roll(lats, 1))
            dists = dists[(dists != 0)]
            self._distance_upper_bound = 1.05 * (2 ** 0.5) * (dists.min() / 2)

            logger.info('Distance upper bound was infered to be: {}'
                        .format(self._distance_upper_bound))

        return self._distance_upper_bound

    def _parallel_map(self, gids=None):
        """Map all gids in parallel.

        Parameters
        ----------
        gids : np.ndarray | None
            Supply curve gids with tech exclusion points to map to the
            generation meta points. None defaults to all SC gids.

        Returns
        -------
        ind_all : np.ndarray
            Index values of the NN generation point. -1 if no gen point found.
            1D integer array with length equal to the number of tech exclusion
            points in the supply curve extent.
        coords_all : np.ndarray
            Un-projected latitude, longitude array of tech exclusion points.
            0's if no gen point found. 2D integer array (N, 2) filled with 0's
            with length equal to the number of tech exclusion points in the
            supply curve extent.
        """

        if gids is None:
            gids = np.array(list(range(len(self._sc))), dtype=np.uint32)
        elif isinstance(gids, (list, tuple)):
            gids = np.array(gids, dtype=np.uint32)

        gid_chunks = np.array_split(gids, int(np.ceil(len(gids) / 2)))

        # init full output arrays
        ind_all, coords_all = self._init_out_arrays()

        n_finished = 0
        futures = {}
        with cf.ProcessPoolExecutor(max_workers=self._n_cores) as executor:

            # iterate through split executions, submitting each to worker
            for i, gid_set in enumerate(gid_chunks):
                # submit executions and append to futures list
                futures[executor.submit(TechMapping.map_gids,
                                        gid_set,
                                        self._fpath_excl,
                                        self._fpath_gen,
                                        self.distance_upper_bound,
                                        self._resolution)] = i

            for future in cf.as_completed(futures):
                n_finished += 1
                logger.info('Parallel TechMapping futures collected: '
                            '{} out of {}'
                            .format(n_finished, len(futures)))

                i = futures[future]
                result = future.result()
                for j, gid in enumerate(gid_chunks[i]):
                    i_out_arr = self._sc.get_flat_excl_ind(gid)
                    ind_all[i_out_arr] = result[0][j]
                    coords_all[i_out_arr, :] = result[1][j]

        return ind_all, coords_all

    @staticmethod
    def map_gids(gids, fpath_excl, fpath_gen, distance_upper_bound,
                 resolution, margin=0.1):
        """Map exclusion gids to the gen meta.

        Parameters
        ----------
        gids : np.ndarray
            Supply curve gids with tech exclusion points to map to the
            generation meta points.
        fpath_excl : str
            Filepath to exclusions geotiff (tech layer).
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        distance_upper_bound : float | None
            Upper boundary distance for KNN lookup between exclusion points and
            generation points.
        resolution : int
            Supply curve resolution used for the tech mapping. Must correspond
            to the resolution used to make the gids input. This is not
            the final supply curve point resolution.
        margin : float
            Margin when reducing the generation lat/lon.

        Returns
        -------
        ind : list
            List of arrays of index values from the NN. List entries correspond
            to input gids.
        coords : np.ndarray
            List of arrays of the un-projected latitude, longitude array of
            tech exclusion points. List entries correspond to input gids.
        """

        sc = SupplyCurveExtent(fpath_excl, resolution=resolution)

        ind_out = []
        coords_out = []
        coord_labels = ['latitude', 'longitude']
        lat_range = None
        lon_range = None
        logger.debug('Getting tech layer coordinates for chunks {} through {}'
                     .format(gids[0], gids[-1]))
        for gid in gids:
            row_slice, col_slice = sc.get_excl_slices(gid)
            emeta = sc.exclusions['meta', row_slice, col_slice]
            emeta = emeta[coord_labels].values
            coords_out.append(emeta)

            if lat_range is None:
                lat_range = [np.min(emeta[:, 0]), np.max(emeta[:, 0])]
                lon_range = [np.min(emeta[:, 1]), np.max(emeta[:, 1])]
            else:
                lat_range[0] = np.min((lat_range[0], np.min(emeta[:, 0])))
                lat_range[1] = np.max((lat_range[1], np.max(emeta[:, 0])))
                lon_range[0] = np.min((lon_range[0], np.min(emeta[:, 1])))
                lon_range[1] = np.max((lon_range[1], np.max(emeta[:, 1])))

        with Outputs(fpath_gen, str_decode=False) as o:
            gen_meta = np.vstack((o.get_meta_arr(coord_labels[0]),
                                  o.get_meta_arr(coord_labels[1]))).T

        mask = ((gen_meta[:, 0] > lat_range[0] - margin) &
                (gen_meta[:, 0] < lat_range[1] + margin) &
                (gen_meta[:, 1] > lon_range[0] - margin) &
                (gen_meta[:, 1] < lon_range[1] + margin))

        # pylint: disable-msg=C0121
        mask_ind = np.where(mask == True)[0]  # noqa: E712

        if np.sum(mask) > 0:
            gen_tree = cKDTree(gen_meta[mask, :])

            logger.debug('Running tech mapping for chunks {} through {}'
                         .format(gids[0], gids[-1]))
            for i, gid in enumerate(gids):
                dist, ind = gen_tree.query(coords_out[i])
                ind = mask_ind[ind]
                ind[(dist > distance_upper_bound)] = -1
                ind_out.append(ind)
        else:
            logger.debug('No close gen points for chunks {} through {}'
                         .format(gids[0], gids[-1]))
            for _ in gids:
                ind_out.append(-1)

        return ind_out, coords_out

    def save_tech_map(self, ind, coords, fpath_out, chunks=(128, 128)):
        """Save tech mapping indices and coordinates to an h5 output file.

        Parameters
        ----------
        ind : np.ndarray
            Index values of the NN generation point.
        coords : np.ndarray
            Un-projected latitude, longitude array of tech exclusion points.
        fpath_out : str
            .h5 filepath to save tech mapping results.
        chunks : tuple
            Chunk shape of the 2D output datasets.
        """

        if not fpath_out.endswith('.h5'):
            fpath_out += '.h5'

        logger.info('Writing tech map to {}'.format(fpath_out))

        shape = self._sc.exclusions.shape
        chunks = (np.min((shape[0], chunks[0])), np.min((shape[1], chunks[1])))

        with h5py.File(fpath_out, 'w') as f:

            f.create_dataset('gen_ind', shape=shape, dtype=ind.dtype,
                             data=ind.reshape(shape), chunks=chunks)

            f.create_dataset('latitude', shape=shape, dtype=coords.dtype,
                             data=coords[:, 0].reshape(shape), chunks=chunks)

            f.create_dataset('longitude', shape=shape, dtype=coords.dtype,
                             data=coords[:, 1].reshape(shape), chunks=chunks)

            f.attrs['resolution'] = self._resolution
            f.attrs['fpath_excl'] = self._fpath_excl
            f.attrs['fpath_gen'] = self._fpath_gen
            f.attrs['distance_upper_bound'] = self.distance_upper_bound

        logger.info('Successfully saved tech map to {}'.format(fpath_out))

    @classmethod
    def run_map(cls, fpath_excl, fpath_gen, fpath_out, **kwargs):
        """Run parallel mapping and save to h5 file.

        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff (tech layer).
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        fpath_out : str
            .h5 filepath to save tech mapping results.
        """

        mapper = cls(fpath_excl, fpath_gen, **kwargs)
        ind, coords = mapper._parallel_map()
        mapper.save_tech_map(ind, coords, fpath_out)
