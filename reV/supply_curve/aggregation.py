# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:24:31 2019

@author: gbuster
"""
import concurrent.futures as cf
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from warnings import warn

from reV.supply_curve.points import SupplyCurvePoint, SupplyCurveExtent
from reV.handlers.outputs import Outputs
from reV.utilities.exceptions import EmptySupplyCurvePointError, OutputWarning


class SupplyCurvePointSummary(SupplyCurvePoint):
    """Supply curve point summary with extra method for summary calc."""

    def latitude(self):
        """Get the SC point latitude"""
        return self.centroid[0]

    def longitude(self):
        """Get the SC point longitude"""
        return self.centroid[0]

    def resource_gids(self):
        """Get the list of resource gids corresponding to this sc point.

        Returns
        -------
        res_gids : list
            List of resource gids.
        """
        return list(self.exclusion_meta['resource_gid'].unique())

    def gen_gids(self):
        """Get the list of generation gids corresponding to this sc point.

        Returns
        -------
        gen_gids : list
            List of generation gids.
        """
        return list(self.exclusion_meta['gen_gid'].unique())

    @classmethod
    def summary(cls, fpath_excl, fpath_gen, args=None, **kwargs):
        """Get a summary dictionary of a supply curve point.

        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
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

        point = cls(fpath_excl, fpath_gen, **kwargs)

        ARGS = {'resource_gids': point.resource_gids,
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
    def _serial_summary(fpath_excl, fpath_gen, resolution=64, gids=None):
        """Hidden summary method that can be parallelized.

        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
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

        with Outputs(fpath_gen) as o:
            gen_mask = (o.meta['latitude'] > -1000)
            gen_tree = cKDTree(o.meta.loc[gen_mask, ['latitude', 'longitude']])

        with SupplyCurveExtent(fpath_excl, resolution=resolution) as sc:

            if gids is None:
                gids = range(len(sc))

            for gid in gids:
                try:
                    pointsum = SupplyCurvePointSummary.summary(
                        fpath_excl, fpath_gen, gid=gid, resolution=resolution,
                        gen_tree=gen_tree, gen_mask=gen_mask)

                except EmptySupplyCurvePointError as _:
                    pass

                else:
                    pointsum['gid'] = gid
                    pointsum['row_ind'] = sc[gid]['row_ind']
                    pointsum['col_ind'] = sc[gid]['col_ind']
                    series = pd.Series(pointsum, name=gid)
                    summary = summary.append(series)

        return summary

    @classmethod
    def _parallel_summary(cls, fpath_excl, fpath_gen, resolution=64, gids=None,
                          n_cores=None):
        """Get the supply curve points aggregation summary.

        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
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

        chunks = np.array_split(gids, n_cores)

        futures = []
        summary = pd.DataFrame()

        with cf.ProcessPoolExecutor(max_workers=n_cores) as executor:

            # iterate through split executions, submitting each to worker
            for gid_set in chunks:
                # submit executions and append to futures list
                futures.append(executor.submit(Aggregation._serial_summary,
                                               fpath_excl, fpath_gen,
                                               resolution=resolution,
                                               gids=gid_set))
            # gather results
            for future in futures:
                summary = summary.append(future.result())

        return summary

    @classmethod
    def summary(cls, fpath_excl, fpath_gen, resolution=64, gids=None,
                n_cores=1):
        """Get the supply curve points aggregation summary.

        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
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
        if n_cores == 1:
            summary = cls._serial_summary(fpath_excl, fpath_gen,
                                          resolution=resolution, gids=gids)
        else:
            summary = cls._parallel_summary(fpath_excl, fpath_gen,
                                            resolution=resolution, gids=gids,
                                            n_cores=n_cores)
        return summary
