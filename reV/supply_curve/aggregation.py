# -*- coding: utf-8 -*-
"""reV supply curve aggregation framework.

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

from reV.handlers.geotiff import Geotiff
from reV.handlers.outputs import Outputs
from reV.supply_curve.points import ExclusionPoints, SupplyCurveExtent
from reV.supply_curve.point_summary import SupplyCurvePointSummary
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      OutputWarning, FileInputError)


logger = logging.getLogger(__name__)


class AggFileHandler:
    """Simple framework to handle aggregation file context managers."""

    def __init__(self, fpath_excl, fpath_gen, fpath_techmap, data_layers):
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
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "band", "method",
            and "fpath".
        """

        self._excl = ExclusionPoints(fpath_excl)
        self._gen = Outputs(fpath_gen, mode='r')
        self._techmap = h5py.File(fpath_techmap, 'r')
        self._data_layers = self._open_data_layers(data_layers)

        # pre-initialize any import attributes
        _ = self._gen.meta

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if type is not None:
            raise

    @staticmethod
    def _open_data_layers(data_layers):
        """Open data layer geotiff handlers.

        Parameters
        ----------
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "band", "method",
            and "fpath".

        Returns
        -------
        data_layers : None | dict
            Aggregation data layers. fobj is added to the dictionary of each
            layer.
        """

        if data_layers is not None:
            for name, attrs in data_layers.items():
                data_layers[name]['fobj'] = Geotiff(attrs['fpath'])
        return data_layers

    @staticmethod
    def _close_data_layers(data_layers):
        """Open data layer geotiff handlers.

        Parameters
        ----------
        data_layers : None | dict
            Aggregation data layers. Must have fobj geotiff handlers to close.
        """

        if data_layers is not None:
            for name in data_layers.keys():
                data_layers[name]['fobj'].close()

    def close(self):
        """Close all file handlers."""
        self._excl.close()
        self._gen.close()
        self._techmap.close()
        self._close_data_layers(self._data_layers)

    @property
    def exclusions(self):
        """Get the exclusions file handler object.

        Returns
        -------
        _excl : ExclusionPoints
            Exclusions geotiff handler object.
        """
        return self._excl

    @property
    def gen(self):
        """Get the gen file handler object.

        Returns
        -------
        _gen : Outputs
            reV gen outputs handler object.
        """
        return self._gen

    @property
    def techmap(self):
        """Get the techmap file handler object.

        Returns
        -------
        _techmap : h5py.File
            Techmap h5 file handler.
        """
        return self._techmap

    @property
    def data_layers(self):
        """Get the data layers object.

        Returns
        -------
        _data_layers : dict
            Data layers namespace.
        """
        return self._data_layers


class Aggregation:
    """Supply points aggregation framework."""

    def __init__(self, fpath_excl, fpath_gen, fpath_techmap, dset_tm,
                 res_class_dset=None, res_class_bins=None,
                 dset_cf='cf_mean-means', dset_lcoe='lcoe_fcr-means',
                 data_layers=None, resolution=64, gids=None, n_cores=None):
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
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of two-entry lists dictating the resource class bins.
            None if no resource classes.
        dset_cf : str
            Dataset name from f_gen containing capacity factor mean values.
        dset_lcoe : str
            Dataset name from f_gen containing LCOE mean values.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "band", "method",
            and "fpath".
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
        self._res_class_dset = res_class_dset
        self._res_class_bins = res_class_bins
        self._dset_cf = dset_cf
        self._dset_lcoe = dset_lcoe
        self._resolution = resolution
        self._data_layers = data_layers

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
        self._check_data_layers()
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

    def _check_data_layers(self):
        """Run pre-flight checks on requested aggregation data layers."""

        if self._data_layers is not None:

            with Geotiff(self._fpath_excl) as f:
                shape_base = f.shape

            for k, v in self._data_layers.items():
                if 'band' not in v:
                    raise KeyError('Data band for aggregation data layer "{}" '
                                   'must be specified.'.format(k))
                if 'method' not in v:
                    raise KeyError('Data aggregation method data layer "{}" '
                                   'must be specified.'.format(k))
                if 'fpath' not in v:
                    raise KeyError('File path (fpath) for aggregation data '
                                   'layer "{}" must be specified.'.format(k))
                if not os.path.exists(v['fpath']):
                    raise FileInputError('Cannot find file path (fpath) for '
                                         'aggregation data layer "{}".'
                                         .format(k))
                with Geotiff(v['fpath']) as f:
                    if f.shape != shape_base:
                        raise FileInputError('Geotiff shape of data layer '
                                             '"{}" is {}, which does not '
                                             'match the baseline exclusions '
                                             'shape {}.'
                                             .format(k, f.shape, shape_base))

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
    def _get_input_data(gen, fpath_gen, res_class_dset, res_class_bins,
                        dset_cf, dset_lcoe):
        """Extract SC point agg input data args from higher level inputs.

        Parameters
        ----------
        gen : reV.handlers.outputs.Outputs
            reV outputs handler.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of two-entry lists dictating the resource class bins.
            None if no resource classes.
        dset_cf : str
            Dataset name from f_gen containing capacity factor mean values.
        dset_lcoe : str
            Dataset name from f_gen containing LCOE mean values.

        Returns
        -------
        res_data : np.ndarray | None
            Extracted resource data from res_class_dset
        res_class_bins : list
            List of resouce class bin ranges.
        cf_data : np.ndarray | None
            Capacity factor data extracted from dset_cf in gen
        lcoe_data : np.ndarray | None
            LCOE data extracted from dset_lcoe in gen
        """

        if res_class_dset is None:
            res_data = None
            res_class_bins = [None]
        else:
            res_data = gen[res_class_dset]

        if dset_cf in gen.dsets:
            cf_data = gen[dset_cf]
        else:
            cf_data = None
            warn('Could not find cf dataset "{}" in '
                 'generation file: {}'
                 .format(dset_cf, fpath_gen), OutputWarning)
        if dset_lcoe in gen.dsets:
            lcoe_data = gen[dset_lcoe]
        else:
            lcoe_data = None
            warn('Could not find lcoe dataset "{}" in '
                 'generation file: {}'
                 .format(dset_lcoe, fpath_gen), OutputWarning)
        return res_data, res_class_bins, cf_data, lcoe_data

    @staticmethod
    def _serial_summary(fpath_excl, fpath_gen, fpath_techmap, dset_tm,
                        gen_index, res_class_dset=None, res_class_bins=None,
                        dset_cf='cf_mean-means', dset_lcoe='lcoe_fcr-means',
                        data_layers=None, resolution=64, gids=None, **kwargs):
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
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of two-entry lists dictating the resource class bins.
            None if no resource classes.
        dset_cf : str
            Dataset name from f_gen containing capacity factor mean values.
        dset_lcoe : str
            Dataset name from f_gen containing LCOE mean values.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "band", "method",
            and "fpath".
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        kwargs : dict
            Namespace of additional keyword args to init
            SupplyCurvePointSummary.

        Returns
        -------
        summary : list
            List of dictionaries, each being an SC point summary.
        """

        summary = []

        with SupplyCurveExtent(fpath_excl, resolution=resolution) as sc:
            points = sc.points
            exclusion_shape = sc.exclusions.shape
            if gids is None:
                gids = range(len(sc))

        # pre-extract handlers so they are not repeatedly initialized
        with AggFileHandler(fpath_excl, fpath_gen, fpath_techmap,
                            data_layers) as fhandler:

            inputs = Aggregation._get_input_data(fhandler.gen, fpath_gen,
                                                 res_class_dset,
                                                 res_class_bins,
                                                 dset_cf,
                                                 dset_lcoe)

            for gid in gids:
                for ri, res_bin in enumerate(inputs[1]):
                    try:
                        pointsum = SupplyCurvePointSummary.summary(
                            gid,
                            fhandler.exclusions,
                            fhandler.gen,
                            fhandler.techmap,
                            dset_tm,
                            gen_index,
                            res_class_dset=inputs[0],
                            res_class_bin=res_bin,
                            dset_cf=inputs[2],
                            dset_lcoe=inputs[3],
                            data_layers=fhandler.data_layers,
                            resolution=resolution,
                            exclusion_shape=exclusion_shape,
                            **kwargs)

                    except EmptySupplyCurvePointError as _:
                        pass

                    else:
                        pointsum['sc_gid'] = gid
                        pointsum['sc_row_ind'] = points.loc[gid, 'row_ind']
                        pointsum['sc_col_ind'] = points.loc[gid, 'col_ind']
                        pointsum['res_class'] = ri

                        summary.append(pointsum)

        return summary

    def _parallel_summary(self, **kwargs):
        """Get the supply curve points aggregation summary using futures.

        Parameters
        ----------
        kwargs : dict
            Namespace of additional keyword args to init
            SupplyCurvePointSummary.

        Returns
        -------
        summary : list
            List of dictionaries, each being an SC point summary.
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
        summary = []

        with cf.ProcessPoolExecutor(max_workers=self._n_cores) as executor:

            # iterate through split executions, submitting each to worker
            for gid_set in chunks:
                # submit executions and append to futures list
                futures.append(executor.submit(
                    self._serial_summary,
                    self._fpath_excl, self._fpath_gen, self._fpath_techmap,
                    self._dset_tm, self._gen_index,
                    res_class_dset=self._res_class_dset,
                    res_class_bins=self._res_class_bins,
                    dset_cf=self._dset_cf, dset_lcoe=self._dset_lcoe,
                    data_layers=self._data_layers,
                    resolution=self._resolution,
                    gids=gid_set, **kwargs))

            # gather results
            for future in cf.as_completed(futures):
                n_finished += 1
                logger.info('Parallel aggregation futures collected: '
                            '{} out of {}'
                            .format(n_finished, len(chunks)))
                summary += future.result()

        return summary

    @classmethod
    def summary(cls, fpath_excl, fpath_gen, fpath_techmap, dset_tm,
                res_class_dset=None, res_class_bins=None,
                dset_cf='cf_mean-means', dset_lcoe='lcoe_fcr-means',
                data_layers=None, resolution=64, gids=None, n_cores=None,
                option='dataframe', **kwargs):
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
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of two-entry lists dictating the resource class bins.
            None if no resource classes.
        dset_cf : str
            Dataset name from f_gen containing capacity factor mean values.
        dset_lcoe : str
            Dataset name from f_gen containing LCOE mean values.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "band", "method",
            and "fpath".
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
        kwargs : dict
            Namespace of additional keyword args to init
            SupplyCurvePointSummary.

        Returns
        -------
        summary : list | DataFrame
            Summary of the SC points.
        """

        agg = cls(fpath_excl, fpath_gen, fpath_techmap, dset_tm,
                  res_class_dset=res_class_dset, res_class_bins=res_class_bins,
                  dset_cf=dset_cf, dset_lcoe=dset_lcoe,
                  data_layers=data_layers,
                  resolution=resolution, gids=gids, n_cores=n_cores)

        if n_cores == 1:
            summary = agg._serial_summary(agg._fpath_excl, agg._fpath_gen,
                                          agg._fpath_techmap,
                                          agg._dset_tm, agg._gen_index,
                                          res_class_dset=agg._res_class_dset,
                                          res_class_bins=agg._res_class_bins,
                                          dset_cf=agg._dset_cf,
                                          dset_lcoe=agg._dset_lcoe,
                                          data_layers=agg._data_layers,
                                          resolution=agg._resolution,
                                          gids=gids, **kwargs)
        else:
            summary = agg._parallel_summary(**kwargs)

        if 'dataframe' in option.lower():
            summary = pd.DataFrame(summary)
            summary = summary.sort_values('sc_gid')

        return summary
