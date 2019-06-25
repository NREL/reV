# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:24:31 2019

@author: gbuster
"""
import concurrent.futures as cf
import os
import numpy as np
import pandas as pd
from warnings import warn
import logging
import time

from reV.supply_curve.tech_mapping import TechMapping
from reV.supply_curve.points import SupplyCurvePoint, SupplyCurveExtent
from reV.utilities.exceptions import EmptySupplyCurvePointError, OutputWarning


logger = logging.getLogger(__name__)


class SupplyCurvePointSummary(SupplyCurvePoint):
    """Supply curve point summary with extra method for summary calc."""

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
        return list(self.meta['res_gid'].unique())

    def gen_gids(self):
        """Get the list of generation gids corresponding to this sc point.

        Returns
        -------
        gen_gids : list
            List of generation gids.
        """
        return list(self.meta['gen_gid'].unique())

    @classmethod
    def summary(cls, gid, fpath_excl, fpath_gen, fpath_techmap, args=None,
                **kwargs):
        """Get a summary dictionary of a supply curve point.

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

        point = cls(gid, fpath_excl, fpath_gen, fpath_techmap, **kwargs)

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

    @staticmethod
    def _serial_summary(fpath_excl, fpath_gen, fpath_techmap, resolution=64,
                        gids=None):
        """Hidden summary method that can be parallelized.

        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        fpath_techmap : str
            Filepath to tech mapping between exclusions and generation results
            (created using the reV TechMapping framework).
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.

        Returns
        -------
        summary : pd.DataFrame
            Summary dataframe of the SC points.
        """

        summary = pd.DataFrame()

        with SupplyCurveExtent(fpath_excl, resolution=resolution) as sc:

            if gids is None:
                gids = range(len(sc))

            logger.info('Running serial supply curve point aggregation for '
                        'sc points {} through {} at a resolution of {}'
                        .format(gids[0], gids[-1], resolution))

            for gid in gids:
                t0 = time.time()
                try:
                    pointsum = SupplyCurvePointSummary.summary(
                        gid, fpath_excl, fpath_gen, fpath_techmap,
                        resolution=resolution)

                except EmptySupplyCurvePointError as _:
                    pass

                else:
                    pointsum['gid'] = gid
                    pointsum['row_ind'] = sc[gid]['row_ind']
                    pointsum['col_ind'] = sc[gid]['col_ind']
                    series = pd.Series(pointsum, name=gid)
                    summary = summary.append(series)

                t1 = time.time() - t0
                logger.debug('Aggregating gid {} took {} seconds'
                             .format(gid, t1))

        return summary

    @classmethod
    def _parallel_summary(cls, fpath_excl, fpath_gen, fpath_techmap,
                          resolution=64, gids=None, n_cores=None):
        """Get the supply curve points aggregation summary.

        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        fpath_techmap : str
            Filepath to tech mapping between exclusions and generation results
            (created using the reV TechMapping framework).
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        n_cores : int | None
            Number of cores to run summary on. None runs on all available cpus.

        Returns
        -------
        summary : pd.DataFrame
            Summary dataframe of the SC points.
        """

        if n_cores is None:
            n_cores = os.cpu_count()

        if gids is None:
            with SupplyCurveExtent(fpath_excl, resolution=resolution) as sc:
                gids = np.array(range(len(sc)), dtype=np.uint32)

        logger.info('Running parallel supply curve point aggregation for '
                    'sc points {} through {} at a resolution of {} on {} cores'
                    .format(gids[0], gids[-1], resolution, n_cores))

        chunks = np.array_split(gids, n_cores)

        futures = []
        summary = pd.DataFrame()

        with cf.ProcessPoolExecutor(max_workers=n_cores) as executor:

            # iterate through split executions, submitting each to worker
            for gid_set in chunks:
                # submit executions and append to futures list
                futures.append(executor.submit(Aggregation._serial_summary,
                                               fpath_excl, fpath_gen,
                                               fpath_techmap,
                                               resolution=resolution,
                                               gids=gid_set))
            # gather results
            for future in futures:
                summary = summary.append(future.result())

        return summary

    @classmethod
    def summary(cls, fpath_excl, fpath_gen, fpath_techmap, resolution=64,
                gids=None, n_cores=1):
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
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        n_cores : int
            Number of cores to run summary on. 1 is serial, None is all
            available cpus.

        Returns
        -------
        summary : pd.DataFrame
            Summary dataframe of the SC points.
        """

        if not os.path.exists(fpath_techmap):
            logger.info('Supply curve point aggregation could not find the '
                        'tech map file, so running the TechMapping module.')
            TechMapping.run_map(fpath_excl, fpath_gen, fpath_techmap)

        if n_cores == 1:
            summary = cls._serial_summary(fpath_excl, fpath_gen, fpath_techmap,
                                          resolution=resolution, gids=gids)
        else:
            summary = cls._parallel_summary(fpath_excl, fpath_gen,
                                            fpath_techmap,
                                            resolution=resolution, gids=gids,
                                            n_cores=n_cores)
        return summary
