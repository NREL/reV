# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:24:31 2019

@author: gbuster
"""
import concurrent.futures as cf
import os
import h5py
import numpy as np
import pandas as pd
from warnings import warn
import logging

from reV.handlers.outputs import Outputs
from reV.supply_curve.points import (ExclusionPoints, SupplyCurvePoint,
                                     SupplyCurveExtent)
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      OutputWarning, FileInputError)


logger = logging.getLogger(__name__)


class SupplyCurvePointSummary(SupplyCurvePoint):
    """Supply curve summary framework with extra methods for summary calc."""

    def latitude(self):
        """Get the SC point latitude"""
        return self.centroid[0]

    def longitude(self):
        """Get the SC point longitude"""
        return self.centroid[1]

    def res_gids(self):
        """Get the list of resource gids corresponding to this sc point.

        Returns
        -------
        res_gids : list
            List of resource gids.
        """
        res_gids = list(set(self._res_gids[np.isin(self._res_gids,
                                                   self.gen.meta['gid'])]))
        return res_gids

    def gen_gids(self):
        """Get the list of generation gids corresponding to this sc point.

        Returns
        -------
        gen_gids : list
            List of generation gids.
        """
        gen_gids = list(set(self._gen_gids))
        if -1 in gen_gids:
            gen_gids.remove(-1)
        return gen_gids

    @classmethod
    def summary(cls, gid, fpath_excl, fpath_gen, fpath_techmap, dset_tm,
                gen_index, args=None, **kwargs):
        """Get a summary dictionary of a single supply curve point.

        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        fpath_techmap : str
            Filepath to tech mapping between exclusions and generation results
            (created using the reV TechMapping framework).
        dset_tm : str
            Dataset name in the techmap file containing the
            exclusions-to-generation mapping data.
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        args : tuple | list | None
            List of summary arguments to include. None defaults to all
            available args defined in the class attr.
        kwargs : dict
            Keyword args to init the SC point.

        Returns
        -------
        summary : dict
            Dictionary of summary outputs for this sc point.
        """

        with cls(gid, fpath_excl, fpath_gen, fpath_techmap, dset_tm, gen_index,
                 **kwargs) as point:

            ARGS = {'resource_gids': point.res_gids,
                    'gen_gids': point.gen_gids,
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    }

            if args is None:
                args = list(ARGS.keys())

            summary = {}
            for arg in args:
                if arg in ARGS:
                    summary[arg] = ARGS[arg]()
                else:
                    warn('Cannot find "{}" as an available SC point summary '
                         'output', OutputWarning)

        return summary


class Aggregation:
    """Supply points aggregation framework."""

    def __init__(self, fpath_excl, fpath_gen, fpath_techmap, dset_tm,
                 resolution=64, gids=None, n_cores=None):
        """
        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        fpath_techmap : str
            Filepath to tech mapping between exclusions and generation results
            The tech mapping module will be run if this file does not exist.
        dset_tm : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        n_cores : int | None
            Number of cores to run summary on. 1 is serial, None is all
            available cpus.
        """

        self._fpath_excl = fpath_excl
        self._fpath_gen = fpath_gen
        self._fpath_techmap = fpath_techmap
        self._dset_tm = dset_tm
        self._resolution = resolution

        if n_cores is None:
            n_cores = os.cpu_count()
        self._n_cores = n_cores

        if gids is None:
            with SupplyCurveExtent(fpath_excl, resolution=resolution) as sc:
                gids = np.array(range(len(sc)), dtype=np.uint32)
        elif not isinstance(gids, np.ndarray):
            gids = np.array(gids)
        self._gids = gids

        self._check_files()
        self._gen_index = self._parse_gen_index(self._fpath_gen)

    def _check_files(self):
        """Do a preflight check on input files"""

        for fpath in (self._fpath_excl, self._fpath_gen, self._fpath_techmap):
            if not os.path.exists(fpath):
                raise FileNotFoundError('Could not find required input file: '
                                        '{}'.format(fpath))

        with h5py.File(self._fpath_techmap, 'r') as f:
            if self._dset_tm not in f:
                raise FileInputError('Could not find "{}" in techmap file: {}'
                                     .format(self._dset_tm,
                                             self._fpath_techmap))

    @staticmethod
    def _parse_gen_index(fpath_gen):
        """Parse gen outputs for an array of generation gids corresponding to
        the resource gids.

        Parameters
        ----------
        fpath_gen : str
            Filepath to .h5 reV generation output results.

        Returns
        -------
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        """

        with Outputs(fpath_gen, mode='r') as gen:
            gen_index = gen.meta

        gen_index = gen_index.rename(columns={'gid': 'res_gids'})
        gen_index['gen_gids'] = gen_index.index
        gen_index = gen_index[['res_gids', 'gen_gids']]
        gen_index = gen_index.set_index(keys='res_gids')
        gen_index = gen_index.reindex(range(gen_index.index.max() + 1))
        gen_index = gen_index['gen_gids'].values
        gen_index[np.isnan(gen_index)] = -1
        gen_index = gen_index.astype(np.int32)

        return gen_index

    @staticmethod
    def _serial_summary(fpath_excl, fpath_gen, fpath_techmap, dset_tm,
                        gen_index, resolution=64, gids=None):
        """Standalone method to create agg summary - can be parallelized.

        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        fpath_techmap : str
            Filepath to tech mapping between exclusions and generation results
            (created using the reV TechMapping framework).
        dset_tm : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.

        Returns
        -------
        summary : dict
            Summary dictionary of the SC points keyed by SC point gid.
        """

        summary = {}

        with SupplyCurveExtent(fpath_excl, resolution=resolution) as sc:

            exclusion_shape = sc.exclusions.shape

            if gids is None:
                gids = range(len(sc))

            # pre-extract handlers so they are not repeatedly initialized
            with ExclusionPoints(fpath_excl) as excl:
                with Outputs(fpath_gen, mode='r') as gen:
                    with h5py.File(fpath_techmap, 'r') as techmap:

                        # pre-extract meta before iter
                        _ = gen.meta

                        for gid in gids:
                            try:
                                pointsum = SupplyCurvePointSummary.summary(
                                    gid, excl, gen, techmap, dset_tm,
                                    gen_index, resolution=resolution,
                                    exclusion_shape=exclusion_shape,
                                    close=False)

                            except EmptySupplyCurvePointError:
                                pass

                            else:
                                pointsum['sc_gid'] = gid
                                pointsum['sc_row_ind'] = \
                                    sc.points.loc[gid, 'row_ind']
                                pointsum['sc_col_ind'] = \
                                    sc.points.loc[gid, 'col_ind']
                                summary[gid] = pointsum

        return summary

    def _parallel_summary(self):
        """Get the supply curve points aggregation summary using futures.

        Returns
        -------
        summary : dict
            Summary dictionary of the SC points keyed by SC point gid.
        """

        chunks = np.array_split(self._gids,
                                int(np.ceil(len(self._gids) / 1000)))

        logger.info('Running supply curve point aggregation for '
                    'points {} through {} at a resolution of {} '
                    'on {} cores in {} chunks.'
                    .format(self._gids[0], self._gids[-1], self._resolution,
                            self._n_cores, len(chunks)))

        n_finished = 0
        futures = []
        summary = {}

        with cf.ProcessPoolExecutor(max_workers=self._n_cores) as executor:

            # iterate through split executions, submitting each to worker
            for gid_set in chunks:
                # submit executions and append to futures list
                futures.append(executor.submit(self._serial_summary,
                                               self._fpath_excl,
                                               self._fpath_gen,
                                               self._fpath_techmap,
                                               self._dset_tm,
                                               self._gen_index,
                                               resolution=self._resolution,
                                               gids=gid_set))
            # gather results
            for future in cf.as_completed(futures):
                n_finished += 1
                logger.info('Parallel aggregation futures collected: '
                            '{} out of {}'
                            .format(n_finished, len(chunks)))
                summary.update(future.result())

        return summary

    @classmethod
    def summary(cls, fpath_excl, fpath_gen, fpath_techmap, dset_tm,
                resolution=64, gids=None, n_cores=None,
                option='dataframe'):
        """Get the supply curve points aggregation summary.

        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        fpath_techmap : str
            Filepath to tech mapping between exclusions and generation results
            The tech mapping module will be run if this file does not exist.
        dset_tm : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        n_cores : int | None
            Number of cores to run summary on. 1 is serial, None is all
            available cpus.
        option : str
            Output dtype option (dict, dataframe).

        Returns
        -------
        summary : dict | DataFrame
            Summary of the SC points keyed by SC point gid.
        """

        agg = cls(fpath_excl, fpath_gen, fpath_techmap,
                  dset_tm, resolution=resolution, gids=gids,
                  n_cores=n_cores)

        if n_cores == 1:
            summary = agg._serial_summary(agg._fpath_excl, agg._fpath_gen,
                                          agg._fpath_techmap,
                                          agg._dset_tm, agg._gen_index,
                                          resolution=agg._resolution,
                                          gids=gids)
        else:
            summary = agg._parallel_summary()

        if 'dataframe' in option.lower():
            summary = pd.DataFrame(summary).T
            summary = summary.set_index('sc_gid', drop=True).sort_index()

        return summary
