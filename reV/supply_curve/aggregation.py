# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:24:31 2019

@author: gbuster
"""
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


class Aggregation(SupplyCurveExtent):
    """Supply points aggregation framework."""

    @classmethod
    def summary(cls, fpath_excl, fpath_gen, resolution=64, gids=None):
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
        """

        summary = pd.DataFrame()

        with Outputs(fpath_gen) as o:
            gen_mask = (o.meta['latitude'] > -1000)
            gen_tree = cKDTree(o.meta.loc[gen_mask, ['latitude', 'longitude']])

        with cls(fpath_excl, resolution=resolution) as agg:

            if gids is None:
                gids = range(len(agg))

            for gid in gids:
                try:
                    pointsum = SupplyCurvePointSummary.summary(
                        fpath_excl, fpath_gen, gid=gid, resolution=resolution,
                        gen_tree=gen_tree, gen_mask=gen_mask)

                except EmptySupplyCurvePointError as _:
                    pass

                else:
                    pointsum['gid'] = gid
                    pointsum['row_ind'] = agg[gid]['row_ind']
                    pointsum['col_ind'] = agg[gid]['col_ind']
                    series = pd.Series(pointsum, name=gid)
                    summary = summary.append(series)

        return summary
