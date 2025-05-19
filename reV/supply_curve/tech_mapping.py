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
from scipy.spatial import cKDTree
from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.utilities import check_res_file, res_dist_threshold

from reV.supply_curve.extent import SupplyCurveExtent, LATITUDE, LONGITUDE
from reV.utilities.exceptions import FileInputError, FileInputWarning

logger = logging.getLogger(__name__)


class TechMapping:
    """Framework to create map between tech layer (exclusions), res, and gen"""

    def __init__(self, excl_fpath, sc_resolution=1200):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 file, must contain latitude and longitude
            arrays to allow for mapping to resource points
        sc_resolution : int | None, optional
            Defines how many exclusion pixels are mapped at a time. Units
            indicate the length of one dimension, in pixels, of each square
            chunk to be mapped. By default, this value is 1200, which will
            map the exclusion pixels in 1200x1200 pixel chunks.

            .. Note:: This parameter does not affect the exclusion to resource
            (tech) mapping, which deviates from how the effect of the
            ``sc_resolution`` parameter works in other functionality within
            ``reV``.

        """
        self._excl_fpath = excl_fpath
        self._check_fout()

        with SupplyCurveExtent(
            self._excl_fpath, resolution=sc_resolution
        ) as sc:
            self._gids = np.array(list(range(len(sc))), dtype=np.uint32)
            self._excl_shape = sc.exclusions.shape
            self._n_excl = np.prod(self._excl_shape)
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
        __, hsds = check_res_file(res_fpath)
        with Resource(res_fpath, hsds=hsds) as f:
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
        iarr = np.arange(np.prod(shape), dtype=np.uint32)

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
    def _get_excl_coords(cls, excl_fpath, gid, sc_row_indices, sc_col_indices,
                         excl_row_slices, excl_col_slices,
                         coord_labels=(LATITUDE, LONGITUDE)):
        """
        Extract the exclusion coordinates for the desired supply curve point
        gid for TechMapping.

        Parameters
        ----------
        gid : int
            Supply curve gid with tech exclusion points to map to the
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
        coords_out : ndarray
            2D array (Nx2) of the un-projected latitude, longitude of
            tech exclusion pixels within the specified gid point. Rows
            correspond to exclusion pixels of the specified gid, columns
            correspond to latitude and longitude, respectively.
        """
        with h5py.File(excl_fpath, "r") as f:
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
                coords_out = np.vstack((lats.flatten(), lons.flatten())).T
            except Exception as e:
                m = (
                    "Could not unpack coordinates for gid {} with "
                    "row/col slice {}/{}. Received the following "
                    "error:\n{}".format(gid, row_slice, col_slice, e)
                )
                logger.error(m)
                raise e

        return coords_out

    @classmethod
    def map_resource_gids(
        cls,
        excl_coords,
        tree,
        dist_thresh,
    ):
        """Map exclusion pixels to the resource meta.

        Parameters
        ----------
        excl_coords : ndarray
            2D Array of the un-projected latitude, longitude array of
            tech exclusion pixels.  Rows correspond to exclusion pixels,
            columns correspond to latitude and longitude, respectively.
        tree : cKDTree
            cKDTree built from resource lat, lon coordinates
        dist_tresh : float
            Estimate the distance between resource points. Calculated as half
            of the diagonal between closest resource points, with an extra
            5% margin

        Returns
        -------
        ind : ndarray
            1D arrays of index values from the NN. Entries correspond
            to input exclusion pixels.
        """
        dist, ind = tree.query(excl_coords)
        ind[(dist >= dist_thresh)] = -1

        return ind

    def initialize_dataset(self, dset, chunks=(128, 128)):
        """
        Initialize output dataset in exclusions h5 file. If dataset already
        exists, a warning will be issued.

        Parameters
        ----------
        dset : str
            Name of the dataset in the exclusions H5 file to create.
        chunks : tuple, optional
            Chunk size for the dataset, by default (128, 128).
        """

        with h5py.File(self._excl_fpath, "a") as f:
            if dset in list(f):
                wmsg = (
                    'TechMap results dataset "{}" already exists '
                    'in pre-existing Exclusions TechMapping file "{}"'.format(
                        dset, self._excl_fpath
                    )
                )
                logger.warning(wmsg)
                warn(wmsg, FileInputWarning)
            else:
                logger.info(
                    f"Initializing tech map dataset {dset} in "
                    f"{self._excl_fpath}"
                )
                f.create_dataset(
                    dset,
                    shape=self._excl_shape,
                    dtype=np.int32,
                    chunks=chunks,
                    fillvalue=-1
                )

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

    def map_resource(self, dset, res_fpath, dist_margin=1.05,
                     max_workers=None, batch_size=100):
        """
        Map all resource gids to exclusion gids. Save results to dset in
        exclusions h5 file.

        Parameters
        ----------
        dset : str, optional
            Name of the output dataset in the exclusions H5 file to which the
            tech map will be saved.
        res_fpath : str
            Filepath to .h5 resource file that we're mapping to.
        dist_margin : float, optional
            Extra margin to multiply times the computed distance between
            neighboring resource points, by default 1.05
        max_workers : int, optional
            Number of cores to run mapping on. None uses all available cpus,
            by default None
        batch_size : int, optional
            Number of tasks to be submitted to parallel worker pool at one
            time, by default 1000. As a rule of thumb, this number should be
            set to ~10x the number of max_workers. Higher values are not
            necessarily better, and may slow down processing and/or result in
            out-of-memory errors. Values less than the number of workers can
            also lead to slower processing, due to poor load balancing.
        """
        loggers = [__name__, "reV"]

        logger.info(
            f"Computing cKDtree for tech mapping using {res_fpath=!r} "
            f"and {dist_margin=!r}"
        )
        tree, dist_thresh = self._build_tree(
            res_fpath, dist_margin=dist_margin
        )
        with h5py.File(self._excl_fpath, "a") as f:
            f[dset].attrs["distance_threshold"] = dist_thresh
            f[dset].attrs["src_res_fpath"] = res_fpath

        n_jobs = len(self._gids)
        n_batches = ceil(n_jobs / batch_size)
        gid_batches = np.array_split(self._gids, n_batches)

        logger.info(
            f"Kicking off {n_jobs} resource mapping jobs in {n_batches} "
            "batches."
        )
        n_finished = 0
        with SpawnProcessPool(max_workers=max_workers, loggers=loggers) as exe:
            for gid_batch in gid_batches:
                futures = {}
                # iterate through split executions, submitting each to worker
                for i, gid in enumerate(gid_batch):
                    # submit executions and append to futures list
                    excl_coords = self._get_excl_coords(
                        self._excl_fpath,
                        gid,
                        self._sc_row_indices,
                        self._sc_col_indices,
                        self._excl_row_slices,
                        self._excl_col_slices,
                    )
                    futures[
                        exe.submit(
                            self.map_resource_gids,
                            excl_coords,
                            tree,
                            dist_thresh,
                        )
                    ] = i

                with h5py.File(self._excl_fpath, "a") as f:
                    indices = f[dset]
                    for future in as_completed(futures):
                        i = futures[future]
                        result = future.result()

                        gid = gid_batch[i]
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
                        indices[row_slice, col_slice] = result.reshape(
                            result_shape
                        )

                n_finished += 1
                logger.info(
                    "Parallel TechMapping batches completed: "
                    f"{n_finished} out of {n_batches}"
                )

    @classmethod
    def run(
        cls,
        excl_fpath,
        res_fpath,
        dset,
        sc_resolution=1200,
        dist_margin=1.05,
        max_workers=None,
        batch_size=1000,
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
        dset : str
            Dataset name in the `excl_fpath` file to which the the
            techmap (exclusions-to-resource mapping data) will be saved.

            .. Important:: If this dataset already exists in the h5 file,
              it will be overwritten.

        sc_resolution : int | None, optional
            Defines how many exclusion pixels are mapped at a time. Units
            indicate the length of one dimension, in pixels, of each square
            chunk to be mapped. By default, this value is 1200, which will
            map the exclusion pixels in 1200x1200 pixel chunks.

            .. Note:: This parameter does not affect the exclusion to resource
            (tech) mapping, which deviates from how the effect of the
            ``sc_resolution`` parameter works in other functionality within
            ``reV``.

        dist_margin : float, optional
            Extra margin to multiply times the computed distance between
            neighboring resource points, by default 1.05
        max_workers : int, optional
            Number of cores to run mapping on. None uses all available cpus,
            by default None
        batch_size : int, optional
            Number of tasks to be submitted to parallel worker pool at one
            time, by default 1000. As a rule of thumb, this number should be
            set to ~10x the number of max_workers. Higher values are not
            necessarily better, and may slow down processing and/or result in
            out-of-memory errors. Values less than the number of workers can
            also lead to slower processing, due to poor load balancing.
        """
        mapper = cls(excl_fpath, sc_resolution=sc_resolution)
        mapper.initialize_dataset(dset)
        mapper.map_resource(
            dset=dset,
            res_fpath=res_fpath,
            dist_margin=dist_margin,
            max_workers=max_workers,
            batch_size=batch_size,
        )
