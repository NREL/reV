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

from reV.handlers.outputs import Outputs
from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.exclusions import ExclusionMask
from reV.supply_curve.points import SupplyCurveExtent
from reV.supply_curve.point_summary import SupplyCurvePointSummary
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      OutputWarning, FileInputError,
                                      InputWarning)


logger = logging.getLogger(__name__)


class AggFileHandler:
    """Simple framework to handle aggregation file context managers."""

    def __init__(self, excl_fpath, gen_fpath, data_layers, excl_dict):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        excl_dict : dict
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        """

        self._excl_fpath = excl_fpath
        self._excl = ExclusionMask.from_dict(excl_fpath, excl_dict)
        self._gen = Outputs(gen_fpath, mode='r')
        self._data_layers = self._open_data_layers(data_layers)

        # pre-initialize any import attributes
        _ = self._gen.meta

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if type is not None:
            raise

    def _open_data_layers(self, data_layers):
        """Open data layer Exclusion h5 handlers.

        Parameters
        ----------
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".

        Returns
        -------
        data_layers : None | dict
            Aggregation data layers. fobj is added to the dictionary of each
            layer.
        """

        if data_layers is not None:
            for name, attrs in data_layers.items():
                data_layers[name]['fobj'] = self._excl.excl_h5
                if 'fpath' in attrs:
                    if attrs['fpath'] != self._excl_fpath:
                        data_layers[name]['fobj'] = ExclusionLayers(
                            attrs['fpath'])
        return data_layers

    @staticmethod
    def _close_data_layers(data_layers):
        """Close all data layers with exclusion h5 handlers.

        Parameters
        ----------
        data_layers : None | dict
            Aggregation data layers. Must have fobj exclusion handlers to close
        """

        if data_layers is not None:
            for layer in data_layers.values():
                if 'fobj' in layer:
                    layer['fobj'].close()

    def close(self):
        """Close all file handlers."""
        self._excl.close()
        self._gen.close()
        self._close_data_layers(self._data_layers)

    @property
    def exclusions(self):
        """Get the exclusions file handler object.

        Returns
        -------
        _excl : ExclusionMask
            Exclusions h5 handler object.
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

    def __init__(self, excl_fpath, gen_fpath, tm_dset, excl_dict,
                 res_class_dset=None, res_class_bins=None,
                 cf_dset='cf_mean-means', lcoe_dset='lcoe_fcr-means',
                 data_layers=None, resolution=64, power_density=None,
                 gids=None, n_cores=None):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        excl_dict : dict
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of floats or ints (bin edges) to convert to list of two-entry
            bin boundaries or list of two-entry bind boundaries in final format
        cf_dset : str
            Dataset name from f_gen containing capacity factor mean values.
        lcoe_dset : str
            Dataset name from f_gen containing LCOE mean values.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        power_density : float | str | None
            Power density in MW/km2 or filepath to variable power
            density file. None will attempt to infer a constant
            power density from the generation meta data technology
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        n_cores : int | None
            Number of cores to run summary on. 1 is serial, None is all
            available cpus.
        """

        self._excl_fpath = excl_fpath
        self._gen_fpath = gen_fpath
        self._tm_dset = tm_dset
        self._excl_dict = excl_dict
        self._res_class_dset = res_class_dset
        self._res_class_bins = self._convert_bins(res_class_bins)
        self._cf_dset = cf_dset
        self._lcoe_dset = lcoe_dset
        self._resolution = resolution
        self._power_density = power_density
        self._data_layers = data_layers

        logger.debug('Resource class bins: {}'.format(self._res_class_bins))

        if self._power_density is None:
            msg = ('Supply curve aggregation power density not specified. '
                   'Will try to infer based on lookup table: {}'
                   .format(SupplyCurvePointSummary.POWER_DENSITY))
            logger.warning(msg)
            warn(msg, InputWarning)

        if n_cores is None:
            n_cores = os.cpu_count()
        self._n_cores = n_cores

        if gids is None:
            with SupplyCurveExtent(excl_fpath, resolution=resolution) as sc:
                gids = np.array(range(len(sc)), dtype=np.uint32)
        elif not isinstance(gids, np.ndarray):
            gids = np.array(gids)
        self._gids = gids

        self._check_files()
        self._check_data_layers()
        self._gen_index = self._parse_gen_index(self._gen_fpath)

    def _check_files(self):
        """Do a preflight check on input files"""

        for fpath in (self._excl_fpath, self._gen_fpath):
            if not os.path.exists(fpath):
                raise FileNotFoundError('Could not find required input file: '
                                        '{}'.format(fpath))

        with h5py.File(self._excl_fpath, 'r') as f:
            if self._tm_dset not in f:
                raise FileInputError('Could not find techmap dataset "{}" '
                                     'in exclusions file: {}'
                                     .format(self._tm_dset,
                                             self._excl_fpath))

    def _check_data_layers(self):
        """Run pre-flight checks on requested aggregation data layers."""

        if self._data_layers is not None:

            with ExclusionLayers(self._excl_fpath) as f:
                shape_base = f.shape

            for k, v in self._data_layers.items():
                if 'dset' not in v:
                    raise KeyError('Data aggregation "dset" data layer "{}" '
                                   'must be specified.'.format(k))
                if 'method' not in v:
                    raise KeyError('Data aggregation "method" data layer "{}" '
                                   'must be specified.'.format(k))
                elif (v['method'].lower() != 'mean'
                      and v['method'].lower() != 'mode'):
                    raise ValueError('Cannot recognize data layer agg method: '
                                     '"{}". Can only do mean and mode.'
                                     .format(v['method']))
                if 'fpath' in v:
                    with ExclusionLayers(v['fpath']) as f:
                        if any(f.shape != shape_base):
                            msg = ('Data shape of data layer "{}" is {}, '
                                   'which does not match the baseline '
                                   'exclusions shape {}.'
                                   .format(k, f.shape, shape_base))
                            raise FileInputError(msg)

    @staticmethod
    def _parse_gen_index(gen_fpath):
        """Parse gen outputs for an array of generation gids corresponding to
        the resource gids.

        Parameters
        ----------
        gen_fpath : str
            Filepath to .h5 reV generation output results.

        Returns
        -------
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        """

        with Outputs(gen_fpath, mode='r') as gen:
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
    def _get_input_data(gen, gen_fpath, res_class_dset, res_class_bins,
                        cf_dset, lcoe_dset):
        """Extract SC point agg input data args from higher level inputs.

        Parameters
        ----------
        gen : reV.handlers.outputs.Outputs
            reV outputs handler.
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of two-entry lists dictating the resource class bins.
            None if no resource classes.
        cf_dset : str
            Dataset name from f_gen containing capacity factor mean values.
        lcoe_dset : str
            Dataset name from f_gen containing LCOE mean values.

        Returns
        -------
        res_data : np.ndarray | None
            Extracted resource data from res_class_dset
        res_class_bins : list
            List of resouce class bin ranges.
        cf_data : np.ndarray | None
            Capacity factor data extracted from cf_dset in gen
        lcoe_data : np.ndarray | None
            LCOE data extracted from lcoe_dset in gen
        """

        if res_class_dset is None:
            res_data = None
            res_class_bins = [None]
        else:
            res_data = gen[res_class_dset]

        if cf_dset in gen.dsets:
            cf_data = gen[cf_dset]
        else:
            cf_data = None
            warn('Could not find cf dataset "{}" in '
                 'generation file: {}'
                 .format(cf_dset, gen_fpath), OutputWarning)
        if lcoe_dset in gen.dsets:
            lcoe_data = gen[lcoe_dset]
        else:
            lcoe_data = None
            warn('Could not find lcoe dataset "{}" in '
                 'generation file: {}'
                 .format(lcoe_dset, gen_fpath), OutputWarning)
        return res_data, res_class_bins, cf_data, lcoe_data

    @staticmethod
    def _serial_summary(excl_fpath, gen_fpath, tm_dset, excl_dict,
                        gen_index, res_class_dset=None, res_class_bins=None,
                        cf_dset='cf_mean-means', lcoe_dset='lcoe_fcr-means',
                        data_layers=None, resolution=64, power_density=None,
                        gids=None, **kwargs):
        """Standalone method to create agg summary - can be parallelized.

        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        tm_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.
        excl_dict : dict
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
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
        cf_dset : str
            Dataset name from f_gen containing capacity factor mean values.
        lcoe_dset : str
            Dataset name from f_gen containing LCOE mean values.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        power_density : float | str | None
            Power density in MW/km2 or filepath to variable power
            density file. None will attempt to infer a constant
            power density from the generation meta data technology
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

        with SupplyCurveExtent(excl_fpath, resolution=resolution) as sc:
            points = sc.points
            exclusion_shape = sc.exclusions.shape
            if gids is None:
                gids = range(len(sc))

        # pre-extract handlers so they are not repeatedly initialized
        file_args = [excl_fpath, gen_fpath, data_layers, excl_dict]
        with AggFileHandler(*file_args) as fhandler:

            inputs = Aggregation._get_input_data(fhandler.gen, gen_fpath,
                                                 res_class_dset,
                                                 res_class_bins,
                                                 cf_dset,
                                                 lcoe_dset)

            for gid in gids:
                for ri, res_bin in enumerate(inputs[1]):
                    try:
                        pointsum = SupplyCurvePointSummary.summary(
                            gid,
                            fhandler.exclusions,
                            fhandler.gen,
                            tm_dset,
                            gen_index,
                            res_class_dset=inputs[0],
                            res_class_bin=res_bin,
                            cf_dset=inputs[2],
                            lcoe_dset=inputs[3],
                            data_layers=fhandler.data_layers,
                            resolution=resolution,
                            exclusion_shape=exclusion_shape,
                            power_density=power_density,
                            **kwargs)

                    except EmptySupplyCurvePointError:
                        pass

                    else:
                        pointsum['sc_point_gid'] = gid
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
                    self._excl_fpath, self._gen_fpath,
                    self._tm_dset, self._excl_dict, self._gen_index,
                    res_class_dset=self._res_class_dset,
                    res_class_bins=self._res_class_bins,
                    cf_dset=self._cf_dset, lcoe_dset=self._lcoe_dset,
                    data_layers=self._data_layers,
                    resolution=self._resolution,
                    power_density=self._power_density,
                    gids=gid_set, **kwargs))

            # gather results
            for future in cf.as_completed(futures):
                n_finished += 1
                logger.info('Parallel aggregation futures collected: '
                            '{} out of {}'
                            .format(n_finished, len(chunks)))
                summary += future.result()

        return summary

    @staticmethod
    def _convert_bins(bins):
        """Convert a list of floats or ints to a list of two-entry bin bounds.

        Parameters
        ----------
        bins : list | None
            List of floats or ints (bin edges) to convert to list of two-entry
            bin boundaries or list of two-entry bind boundaries in final format

        Returns
        -------
        bins : list
            List of two-entry bin boundaries
        """

        if bins is None:
            return None

        type_check = [isinstance(x, (list, tuple)) for x in bins]

        if all(type_check):
            return bins

        elif any(type_check):
            raise TypeError('Resource class bins has inconsistent '
                            'entry type: {}'.format(bins))

        else:
            bbins = []
            for i, b in enumerate(sorted(bins)):
                if i < len(bins) - 1:
                    bbins.append([b, bins[i + 1]])

            return bbins

    @classmethod
    def summary(cls, excl_fpath, gen_fpath, tm_dset, excl_dict,
                res_class_dset=None, res_class_bins=None,
                cf_dset='cf_mean-means', lcoe_dset='lcoe_fcr-means',
                data_layers=None, resolution=64, power_density=None,
                gids=None, n_cores=None, option='dataframe', **kwargs):
        """Get the supply curve points aggregation summary.

        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        tm_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.
        excl_dict : dict
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of floats or ints (bin edges) to convert to list of two-entry
            bin boundaries or list of two-entry bind boundaries in final format
        cf_dset : str
            Dataset name from f_gen containing capacity factor mean values.
        lcoe_dset : str
            Dataset name from f_gen containing LCOE mean values.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        power_density : float | str | None
            Power density in MW/km2 or filepath to variable power
            density file. None will attempt to infer a constant
            power density from the generation meta data technology
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

        agg = cls(excl_fpath, gen_fpath, tm_dset, excl_dict,
                  res_class_dset=res_class_dset, res_class_bins=res_class_bins,
                  cf_dset=cf_dset, lcoe_dset=lcoe_dset,
                  data_layers=data_layers, resolution=resolution,
                  power_density=power_density, gids=gids, n_cores=n_cores)

        if n_cores == 1:
            summary = agg._serial_summary(agg._excl_fpath, agg._gen_fpath,
                                          agg._tm_dset, agg._excl_dict,
                                          agg._gen_index,
                                          res_class_dset=agg._res_class_dset,
                                          res_class_bins=agg._res_class_bins,
                                          cf_dset=agg._cf_dset,
                                          lcoe_dset=agg._lcoe_dset,
                                          data_layers=agg._data_layers,
                                          resolution=agg._resolution,
                                          power_density=agg._power_density,
                                          gids=gids, **kwargs)
        else:
            summary = agg._parallel_summary(**kwargs)

        if 'dataframe' in option.lower():
            summary = pd.DataFrame(summary)
            summary = summary.sort_values('sc_point_gid')
            summary = summary.reset_index(drop=True)
            summary.index.name = 'sc_gid'

        return summary
