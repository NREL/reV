# -*- coding: utf-8 -*-
"""reV tech mapping framework.

This module manages the exclusions-to-resource mapping.
The core of this module is a parallel cKDTree.

Created on Fri Jun 21 16:05:47 2019

@author: gbuster
"""
import logging
import os
from concurrent.futures import as_completed
from math import ceil
from warnings import warn

import h5py
import numpy as np
from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.utilities import res_dist_threshold
from scipy.spatial import cKDTree

from reV.supply_curve.extent import SupplyCurveExtent, LATITUDE, LONGITUDE
from reV.utilities.exceptions import FileInputError, FileInputWarning

logger = logging.getLogger(__name__)


class TechMapping:
    """Framework to create map between tech layer (exclusions), res, and gen"""

    def __init__(
        self, excl_fpath, res_fpath, resolution=2560, dist_margin=1.05
    ):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 file, must contain latitude and longitude
            arrays to allow for mapping to resource points
        res_fpath : str
            Filepath to .h5 resource file that we're mapping to.
        resolution : int | None, optional
            Supply curve resolution, does not affect the exclusion to resource
            (tech) mapping, but defines how many exclusion pixels are mapped
            at a time, by default 2560
        dist_margin : float, optional
            Extra margin to multiply times the computed distance between
            neighboring resource points, by default 1.05
        """
        self._excl_fpath = excl_fpath
        self._res_fpath = res_fpath
        self._check_fout()

        self._tree, self._dist_thresh = self._build_tree(
            self._res_fpath, dist_margin=dist_margin
        )

        with SupplyCurveExtent(
            self._excl_fpath, resolution=resolution
        ) as sc:
            self._resolution = sc.resolution
            self._gids = np.array(list(range(len(sc))), dtype=np.uint32)
            self._excl_shape = sc.exclusions.shape
            self._n_excl = np.product(self._excl_shape)
            self._sc_row_indices = sc.row_indices
            self._sc_col_indices = sc.col_indices
            self._excl_row_slices = sc.excl_row_slices
            self._excl_col_slices = sc.excl_col_slices
            logger.info(
                "Initialized TechMapping object with {} calc chunks "
                "for {} tech exclusion points".format(
                    len(self._gids), self._n_excl
                )
            )

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

        dist_thresh = res_dist_threshold(
            lat_lons, tree=tree, margin=dist_margin
        )

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
    def _get_excl_slices(
        gid, sc_row_indices, sc_col_indices, excl_row_slices, excl_col_slices
    ):
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
                         coord_labels=(LATITUDE, LONGITUDE)):
        """
        Extract the exclusion coordinates for the desired gids for TechMapping.

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
        with h5py.File(excl_fpath, "r") as f:
            for gid in gids:
                row_slice, col_slice = cls._get_excl_slices(
                    gid,
                    sc_row_indices,
                    sc_col_indices,
                    excl_row_slices,
                    excl_col_slices,
                )
                try:
                    lats = f[coord_labels[0]][row_slice, col_slice]
                    lons = f[coord_labels[1]][row_slice, col_slice]
                    emeta = np.vstack((lats.flatten(), lons.flatten())).T
                except Exception as e:
                    m = (
                        "Could not unpack coordinates for gid {} with "
                        "row/col slice {}/{}. Received the following "
                        "error:\n{}".format(gid, row_slice, col_slice, e)
                    )
                    logger.error(m)
                    raise e

                coords_out.append(emeta)

        return coords_out

    @classmethod
    def map_resource_gids(
        cls,
        gids,
        excl_coords,
        tree,
        dist_thresh,
    ):
        """Map exclusion gids to the resource meta.

        Parameters
        ----------
        gids : np.ndarray
            Supply curve gids with tech exclusion points to map to the
            resource meta points.
        excl_coords : list
            List of arrays of the un-projected latitude, longitude array of
            tech exclusion points. List entries correspond should correspond to
            input gids.
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
        logger.debug(
            "Getting tech map coordinates for chunks {} through {}".format(
                gids[0], gids[-1]
            )
        )
        ind_out = []

        logger.debug(
            "Running tech mapping for chunks {} through {}".format(
                gids[0], gids[-1]
            )
        )
        for i, _ in enumerate(gids):
            dist, ind = tree.query(excl_coords[i])
            ind[(dist >= dist_thresh)] = -1
            ind_out.append(ind)

        return ind_out

    def initialize_dataset(self, tm_dset, chunks=(128, 128)):
        """
        Initialize output dataset in exclusions h5 file. If dataset already
        exists, a warning will be issued.

        Parameters
        ----------
        tm_dset : str
            Name of the dataset in the exclusions H5 file to create.
        chunks : tuple, optional
            Chunk size for the dataset, by default (128, 128).
        """

        with h5py.File(self._excl_fpath, "a") as f:
            if tm_dset in list(f):
                wmsg = (
                    'TechMap results dataset "{}" already exists '
                    'in pre-existing Exclusions TechMapping file "{}"'.format(
                        tm_dset, self._excl_fpath
                    )
                )
                logger.warning(wmsg)
                warn(wmsg, FileInputWarning)
            else:
                f.create_dataset(
                    tm_dset,
                    shape=self._excl_shape,
                    dtype=np.int32,
                    chunks=chunks,
                )
                f[tm_dset][:] = -1

            if self._dist_thresh:
                f[tm_dset].attrs["distance_threshold"] = self._dist_thresh

            if self._res_fpath:
                f[tm_dset].attrs["src_res_fpath"] = self._res_fpath

    def _check_fout(self):
        """Check the TechMapping output file for cached data."""
        with h5py.File(self._excl_fpath, 'r') as f:
            if LATITUDE not in f or LONGITUDE not in f:
                emsg = ('Datasets "latitude" and/or "longitude" not in '
                        'pre-existing Exclusions TechMapping file "{}". '
                        'Cannot proceed.'
                        .format(os.path.basename(self._excl_fpath)))
                logger.exception(emsg)
                raise FileInputError(emsg)

    def map_resource(self, tm_dset, max_workers=None, points_per_worker=10):
        """
        Map all resource gids to exclusion gids. Save results to dset in
        exclusions h5 file.

        Parameters
        ----------
        tm_dset : str, optional
            Name of the output dataset in the exclusions H5 file to which the
            tech map will be saved.
        max_workers : int, optional
            Number of cores to run mapping on. None uses all available cpus,
            by default None
        points_per_worker : int, optional
            Number of supply curve points to map to resource gids on each
            worker, by default 10
        """
        gid_chunks = ceil(len(self._gids) / points_per_worker)
        gid_chunks = np.array_split(self._gids, gid_chunks)

        futures = {}
        loggers = [__name__, "reV"]
        with SpawnProcessPool(max_workers=max_workers, loggers=loggers) as exe:
            # iterate through split executions, submitting each to worker
            for i, gid_set in enumerate(gid_chunks):
                # submit executions and append to futures list
                excl_coords = self._get_excl_coords(
                    self._excl_fpath,
                    gid_set,
                    self._sc_row_indices,
                    self._sc_col_indices,
                    self._excl_row_slices,
                    self._excl_col_slices,
                )
                futures[
                    exe.submit(
                        self.map_resource_gids,
                        gid_set,
                        excl_coords,
                        self._tree,
                        self.distance_threshold,
                    )
                ] = i

            with h5py.File(self._excl_fpath, "a") as f:
                indices = f[tm_dset]
                n_finished = 0
                for future in as_completed(futures):
                    n_finished += 1
                    logger.info(
                        "Parallel TechMapping futures collected: "
                        "{} out of {}".format(n_finished, len(futures))
                    )

                    i = futures[future]
                    result = future.result()

                    for j, gid in enumerate(gid_chunks[i]):
                        row_slice, col_slice = self._get_excl_slices(
                            gid,
                            self._sc_row_indices,
                            self._sc_col_indices,
                            self._excl_row_slices,
                            self._excl_col_slices,
                        )
                        n_rows = row_slice.stop - row_slice.start
                        n_cols = col_slice.stop - col_slice.start
                        result_shape = (n_rows, n_cols)
                        indices[row_slice, col_slice] = result[j].reshape(
                            result_shape
                        )

    @classmethod
    def run(
        cls,
        excl_fpath,
        res_fpath,
        tm_dset,
        resolution=64,
        dist_margin=1.05,
        max_workers=None,
        points_per_worker=10,
    ):
        """Run parallel mapping and save to h5 file.

        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions data HDF5 file. This file must must contain
            latitude and longitude datasets.
        res_fpath : str
            Filepath to HDF5 resource file (e.g. WTK or NSRDB) to which
            the exclusions will be mapped. Can refer to a single file (e.g.,
            "/path/to/nsrdb_2024.h5" or a wild-card e.g.,
            "/path/to/nsrdb_{}.h5")
        tm_dset : str
            Dataset name in the `excl_fpath` file to which the the
            techmap (exclusions-to-resource mapping data) will be saved.

            .. Important:: If this dataset already exists in the h5 file,
              it will be overwritten.
        resolution : int | None, optional
            Supply Curve resolution. This value defines how many pixels
            are in a single side of a supply curve cell. For example,
            a value of ``64`` would generate a supply curve where the
            side of each supply curve cell is ``64x64`` exclusion
            pixels. By default, ``64``.
        dist_margin : float, optional
            Extra margin to multiply times the computed distance between
            neighboring resource points, by default 1.05
        max_workers : int, optional
            Number of cores to run mapping on. None uses all available cpus,
            by default None
        points_per_worker : int, optional
            Number of supply curve points to map to resource gids on each
            worker, by default 10
        """
        kwargs = {"dist_margin": dist_margin, "resolution": resolution}
        mapper = cls(excl_fpath, res_fpath, **kwargs)
        mapper.initialize_dataset(tm_dset)
        mapper.map_resource(
            max_workers=max_workers,
            points_per_worker=points_per_worker,
            tm_dset=tm_dset
        )
