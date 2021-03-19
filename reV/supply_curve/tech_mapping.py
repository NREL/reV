# -*- coding: utf-8 -*-
"""reV tech mapping framework.

This module manages the exclusions-to-resource mapping.
The core of this module is a parallel cKDTree.

Created on Fri Jun 21 16:05:47 2019

@author: gbuster
"""
from concurrent.futures import as_completed
import h5py
import logging
from math import ceil
import numpy as np
import os
from scipy.spatial import cKDTree
from warnings import warn

from reV.supply_curve.points import SupplyCurveExtent
from reV.utilities.exceptions import FileInputWarning, FileInputError

from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.utilities import res_dist_threshold

logger = logging.getLogger(__name__)


class TechMapping:
    """Framework to create map between tech layer (exclusions), res, and gen"""

    def __init__(self, excl_fpath, res_fpath, sc_resolution=2560,
                 dist_margin=1.05):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 file, must contain latitude and longitude
            arrays to allow for mapping to resource points
        res_fpath : str
            Filepath to .h5 resource file that we're mapping to.
        sc_resolution : int | None, optional
            Supply curve resolution, does not affect the exclusion to resource
            (tech) mapping, but defines how many exclusion pixels are mapped
            at a time, by default 2560
        dist_margin : float, optional
            Extra margin to multiply times the computed distance between
            neighboring resource points, by default 1.05
        """
        self._excl_fpath = excl_fpath
        self._check_fout()

        self._tree, self._dist_thresh = \
            self._build_tree(res_fpath, dist_margin=dist_margin)

        with SupplyCurveExtent(self._excl_fpath,
                               resolution=sc_resolution) as sc:
            self._sc_resolution = sc.resolution
            self._gids = np.array(list(range(len(sc))), dtype=np.uint32)
            self._excl_shape = sc.exclusions.shape
            self._n_excl = np.product(self._excl_shape)
            self._sc_row_indices = sc.row_indices
            self._sc_col_indices = sc.col_indices
            self._excl_row_slices = sc.excl_row_slices
            self._excl_col_slices = sc.excl_col_slices
            logger.info('Initialized TechMapping object with {} calc chunks '
                        'for {} tech exclusion points'
                        .format(len(self._gids), self._n_excl))

    @property
    def distance_threshold(self):
        """Get the upper bound on NN distance between excl and res points.

        Returns
        -------
        float
            Estimate the distance between resource points. Calculated as half
            of the diagonal between closest resource points, with desired
            extra margin
        """
        return self._dist_thresh

    @staticmethod
    def _build_tree(res_fpath, dist_margin=1.05):
        """
        Build cKDTree from resource lat, lon coordinates. Compute minimum
        intra point distance between resource gids with provided extra margin.

        Parameters
        ----------
        res_fpath : str
            Filepath to .h5 resource file that we're mapping to.
        dist_margin : float, optional
            Extra margin to multiply times the computed distance between
            neighboring resource points, by default 1.05

        Returns
        -------
        tree : cKDTree
            cKDTree built from resource lat, lon coordinates
        dist_tresh : float
            Estimate the distance between resource points. Calculated as half
            of the diagonal between closest resource points, with desired
            extra margin
        """
        with Resource(res_fpath) as f:
            lat_lons = f.lat_lon

        # pylint: disable=not-callable
        tree = cKDTree(lat_lons)

        dist_thresh = res_dist_threshold(lat_lons, tree=tree,
                                         margin=dist_margin)

        return tree, dist_thresh

    @staticmethod
    def _make_excl_iarr(shape):
        """
        Create 2D array of 1D index values for the flattened h5 excl extent

        Parameters
        ----------
        shape : tuple
            exclusion extent shape

        Returns
        -------
        iarr : ndarray
            2D array of 1D index values for the flattened h5 excl extent
        """
        iarr = np.arange(np.product(shape), dtype=np.uint32)

        return iarr.reshape(shape)

    @staticmethod
    def _get_excl_slices(gid, sc_row_indices, sc_col_indices, excl_row_slices,
                         excl_col_slices):
        """
        Get the row and column slices of the exclusions grid corresponding
        to the supply curve point gid.

        Parameters
        ----------
        gid : int
            Supply curve point gid.
        sc_row_indices : list
            List of row indices in exclusion array for for every sc_point gid
        sc_col_indices : list
            List of column indices in exclusion array for for every sc_point
            gid
        excl_row_slices : list
            List representing the supply curve points rows. Each list entry
            contains the exclusion row slice that are included in the sc
            point.
        excl_col_slices : list
            List representing the supply curve points columns. Each list entry
            contains the exclusion columns slice that are included in the sc
            point.

        Returns
        -------
        row_slice : int
            Exclusions grid row index slice corresponding to the sc point gid.
        col_slice : int
            Exclusions grid col index slice corresponding to the sc point gid.
        """

        row_slice = excl_row_slices[sc_row_indices[gid]]
        col_slice = excl_col_slices[sc_col_indices[gid]]

        return row_slice, col_slice

    @classmethod
    def _get_excl_coords(cls, excl_fpath, gids, sc_row_indices, sc_col_indices,
                         excl_row_slices, excl_col_slices,
                         coord_labels=('latitude', 'longitude')):
        """
        Extract the exclusion coordinates for teh desired gids for TechMapping.

        Parameters
        ----------
        gids : np.ndarray
            Supply curve gids with tech exclusion points to map to the
            resource meta points.
        excl_fpath : str
            Filepath to exclusions h5 file, must contain latitude and longitude
            arrays to allow for mapping to resource points
        sc_row_indices : list
            List of row indices in exclusion array for for every sc_point gid
        sc_col_indices : list
            List of column indices in exclusion array for for every sc_point
            gid
        excl_row_slices : list
            List representing the supply curve points rows. Each list entry
            contains the exclusion row slice that are included in the sc
            point.
        excl_col_slices : list
            List representing the supply curve points columns. Each list entry
            contains the exclusion columns slice that are included in the sc
            point.
        coord_labels : tuple
            Labels for the coordinate datasets.

        Returns
        -------
        coords_out : list
            List of arrays of the un-projected latitude, longitude array of
            tech exclusion points. List entries correspond to input gids.
        """
        coords_out = []
        with h5py.File(excl_fpath, 'r') as f:
            for gid in gids:
                row_slice, col_slice = cls._get_excl_slices(gid,
                                                            sc_row_indices,
                                                            sc_col_indices,
                                                            excl_row_slices,
                                                            excl_col_slices)
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

        return coords_out

    @classmethod
    def map_resource_gids(cls, gids, excl_fpath, sc_row_indices,
                          sc_col_indices, excl_row_slices, excl_col_slices,
                          tree, dist_thresh):
        """Map exclusion gids to the resource meta.

        Parameters
        ----------
        gids : np.ndarray
            Supply curve gids with tech exclusion points to map to the
            resource meta points.
        excl_fpath : str
            Filepath to exclusions h5 file, must contain latitude and longitude
            arrays to allow for mapping to resource points
        sc_row_indices : list
            List of row indices in exclusion array for for every sc_point gid
        sc_col_indices : list
            List of column indices in exclusion array for for every sc_point
            gid
        excl_row_slices : list
            List representing the supply curve points rows. Each list entry
            contains the exclusion row slice that are included in the sc
            point.
        excl_col_slices : list
            List representing the supply curve points columns. Each list entry
            contains the exclusion columns slice that are included in the sc
            point.
        tree : cKDTree
            cKDTree built from resource lat, lon coordinates
        dist_tresh : float
            Estimate the distance between resource points. Calculated as half
            of the diagonal between closest resource points, with an extra
            5% margin

        Returns
        -------
        ind : list
            List of arrays of index values from the NN. List entries correspond
            to input gids.
        """
        logger.debug('Getting tech map coordinates for chunks {} through {}'
                     .format(gids[0], gids[-1]))
        ind_out = []
        coords_out = cls._get_excl_coords(excl_fpath, gids, sc_row_indices,
                                          sc_col_indices, excl_row_slices,
                                          excl_col_slices)

        logger.debug('Running tech mapping for chunks {} through {}'
                     .format(gids[0], gids[-1]))
        for i, _ in enumerate(gids):
            dist, ind = tree.query(coords_out[i])
            ind[(dist >= dist_thresh)] = -1
            ind_out.append(ind)

        return ind_out

    @staticmethod
    def save_tech_map(excl_fpath, dset, indices, distance_threshold=None,
                      res_fpath=None, chunks=(128, 128)):
        """Save tech mapping indices and coordinates to an h5 output file.

        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 file to add techmap to as 'dset'
        dset : str
            Dataset name in fpath_out to save mapping results to.
        indices : np.ndarray
            Index values of the NN resource point. -1 if no res point found.
            2D integer array with shape equal to the exclusions extent shape.
        distance_threshold : float
            Distance upper bound to save as attr.
        res_fpath : str, optional
            Filepath to .h5 resource file that we're mapping to,
            by default None
        chunks : tuple
            Chunk shape of the 2D output datasets.
        """
        logger.info('Writing tech map "{}" to {}'.format(dset, excl_fpath))

        shape = indices.shape
        chunks = (np.min((shape[0], chunks[0])), np.min((shape[1], chunks[1])))

        with h5py.File(excl_fpath, 'a') as f:
            if dset in list(f):
                wmsg = ('TechMap results dataset "{}" is being replaced '
                        'in pre-existing Exclusions TechMapping file "{}"'
                        .format(dset, excl_fpath))
                logger.warning(wmsg)
                warn(wmsg, FileInputWarning)
                f[dset][...] = indices
            else:
                f.create_dataset(dset, shape=shape, dtype=indices.dtype,
                                 data=indices, chunks=chunks)

            if distance_threshold:
                f[dset].attrs['distance_threshold'] = distance_threshold

            if res_fpath:
                f[dset].attrs['src_res_fpath'] = res_fpath

        logger.info('Successfully saved tech map "{}" to {}'
                    .format(dset, excl_fpath))

    def _check_fout(self):
        """Check the TechMapping output file for cached data."""
        with h5py.File(self._excl_fpath, 'r') as f:
            if 'latitude' not in f or 'longitude' not in f:
                emsg = ('Datasets "latitude" and/or "longitude" not in '
                        'pre-existing Exclusions TechMapping file "{}". '
                        'Cannot proceed.'
                        .format(os.path.basename(self._excl_fpath)))
                logger.exception(emsg)
                raise FileInputError(emsg)

    def map_resource(self, max_workers=None, points_per_worker=10):
        """
        Map all resource gids to exclusion gids

        Parameters
        ----------
        max_workers : int, optional
            Number of cores to run mapping on. None uses all available cpus,
            by default None
        points_per_worker : int, optional
            Number of supply curve points to map to resource gids on each
            worker, by default 10

        Returns
        -------
        indices : np.ndarray
            Index values of the NN resource point. -1 if no res point found.
            2D integer array with shape equal to the exclusions extent shape.
        """
        gid_chunks = ceil(len(self._gids) / points_per_worker)
        gid_chunks = np.array_split(self._gids, gid_chunks)

        # init full output arrays
        indices = -1 * np.ones((self._n_excl, ), dtype=np.int32)
        iarr = self._make_excl_iarr(self._excl_shape)

        futures = {}
        loggers = [__name__, 'reV']
        with SpawnProcessPool(max_workers=max_workers,
                              loggers=loggers) as exe:

            # iterate through split executions, submitting each to worker
            for i, gid_set in enumerate(gid_chunks):
                # submit executions and append to futures list
                futures[exe.submit(self.map_resource_gids,
                                   gid_set,
                                   self._excl_fpath,
                                   self._sc_row_indices,
                                   self._sc_col_indices,
                                   self._excl_row_slices,
                                   self._excl_col_slices,
                                   self._tree,
                                   self.distance_threshold)] = i

            n_finished = 0
            for future in as_completed(futures):
                n_finished += 1
                logger.info('Parallel TechMapping futures collected: '
                            '{} out of {}'
                            .format(n_finished, len(futures)))

                i = futures[future]
                result = future.result()

                for j, gid in enumerate(gid_chunks[i]):
                    row_slice, col_slice = self._get_excl_slices(
                        gid,
                        self._sc_row_indices,
                        self._sc_col_indices,
                        self._excl_row_slices,
                        self._excl_col_slices)
                    ind_slice = iarr[row_slice, col_slice].flatten()
                    indices[ind_slice] = result[j]

        indices = indices.reshape(self._excl_shape)

        return indices

    @classmethod
    def run(cls, excl_fpath, res_fpath, dset=None, sc_resolution=2560,
            dist_margin=1.05, max_workers=None, points_per_worker=10):
        """Run parallel mapping and save to h5 file.

        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 (tech layer). dset will be
            created in excl_fpath.
        res_fpath : str
            Filepath to .h5 resource file that we're mapping to.
        dset : str, optional
            Dataset name in excl_fpath to save mapping results to, if None
            do not save tech_map to excl_fpath, by default None
        sc_resolution : int | None, optional
            Supply curve resolution, does not affect the exclusion to resource
            (tech) mapping, but defines how many exclusion pixels are mapped
            at a time, by default 2560
        dist_margin : float, optional
            Extra margin to multiply times the computed distance between
            neighboring resource points, by default 1.05
        max_workers : int, optional
            Number of cores to run mapping on. None uses all available cpus,
            by default None
        points_per_worker : int, optional
            Number of supply curve points to map to resource gids on each
            worker, by default 10

        Returns
        -------
        indices : np.ndarray
            Index values of the NN resource point. -1 if no res point found.
            2D integer array with shape equal to the exclusions extent shape.
        """
        kwargs = {"dist_margin": dist_margin,
                  "sc_resolution": sc_resolution}
        mapper = cls(excl_fpath, res_fpath, **kwargs)
        indices = mapper.map_resource(max_workers=max_workers,
                                      points_per_worker=points_per_worker)

        if dset:
            mapper.save_tech_map(excl_fpath, dset, indices,
                                 distance_threshold=mapper.distance_threshold,
                                 res_fpath=res_fpath)

        return indices
