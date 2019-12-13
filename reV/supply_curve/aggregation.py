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
from scipy.spatial import cKDTree
from warnings import warn
import logging

from reV.handlers.outputs import Outputs
from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.exclusions import ExclusionMaskFromDict
from reV.supply_curve.points import SupplyCurveExtent
from reV.supply_curve.point_summary import SupplyCurvePointSummary
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      OutputWarning, FileInputError,
                                      InputWarning)


logger = logging.getLogger(__name__)


class AggFileHandler:
    """Simple framework to handle aggregation file context managers."""

    def __init__(self, excl_fpath, gen_fpath, data_layers, excl_dict,
                 power_density, area_filter_kernel='queen', min_area=None):
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
        power_density : float | str | None
            Power density in MW/km2 or filepath to variable power
            density file. None will attempt to infer a constant
            power density from the generation meta data technology
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | NoneType
            Minimum required contiguous area filter in sq-km
        """

        self._excl_fpath = excl_fpath
        self._excl = ExclusionMaskFromDict(excl_fpath, excl_dict,
                                           min_area=min_area,
                                           kernel=area_filter_kernel)
        self._gen = Outputs(gen_fpath, mode='r')
        self._data_layers = self._open_data_layers(data_layers)
        self._power_density = power_density
        self._parse_power_density()

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

    def _parse_power_density(self):
        """Parse the power density input. If file, open file handler."""

        if isinstance(self._power_density, str):
            self._pdf = self._power_density

            if self._pdf.endswith('.csv'):
                self._power_density = pd.read_csv(self._pdf)
                if ('gid' in self._power_density
                        and 'power_density' in self._power_density):
                    self._power_density = self._power_density.set_index('gid')
                else:
                    msg = ('Variable power density file must include "gid" '
                           'and "power_density" columns, but received: {}'
                           .format(self._power_density.columns.values))
                    logger.error(msg)
                    raise FileInputError(msg)
            else:
                msg = ('Variable power density file must be csv but received: '
                       '{}'.format(self._pdf))
                logger.error(msg)
                raise FileInputError(msg)

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

    @property
    def power_density(self):
        """Get the power density object.

        Returns
        -------
        _power_density : float | None | pd.DataFrame
            Constant power density float, None, or opened dataframe with
            (resource) "gid" and "power_density columns".
        """
        return self._power_density


class Aggregation:
    """Supply points aggregation framework."""

    def __init__(self, excl_fpath, gen_fpath, tm_dset, excl_dict,
                 res_class_dset=None, res_class_bins=None,
                 cf_dset='cf_mean-means', lcoe_dset='lcoe_fcr-means',
                 data_layers=None, resolution=64, power_density=None,
                 gids=None, area_filter_kernel='queen', min_area=None,
                 n_cores=None):
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
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | NoneType
            Minimum required contiguous area filter in sq-km
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
        self._area_filter_kernel = area_filter_kernel
        self._min_area = min_area

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
        offshore_flags : np.ndarray | None
            Array of offshore boolean flags if available from wind generation
            data. None if offshore flag is not available.
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

        if 'offshore' in gen.meta:
            offshore_flag = gen.meta['offshore'].values
        else:
            offshore_flag = None

        return res_data, res_class_bins, cf_data, lcoe_data, offshore_flag

    @staticmethod
    def _serial_summary(excl_fpath, gen_fpath, tm_dset, excl_dict,
                        gen_index, res_class_dset=None, res_class_bins=None,
                        cf_dset='cf_mean-means', lcoe_dset='lcoe_fcr-means',
                        data_layers=None, resolution=64, power_density=None,
                        gids=None, area_filter_kernel='queen', min_area=None,
                        **kwargs):
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
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | NoneType
            Minimum required contiguous area filter in sq-km
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
        file_args = [excl_fpath, gen_fpath, data_layers, excl_dict,
                     power_density]
        file_kwargs = {'area_filter_kernel': area_filter_kernel,
                       'min_area': min_area}
        with AggFileHandler(*file_args, **file_kwargs) as fhandler:

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
                            power_density=fhandler.power_density,
                            offshore_flags=inputs[4],
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
                    gids=gid_set,
                    area_filter_kernel=self._area_filter_kernel,
                    min_area=self._min_area,
                    **kwargs))

            # gather results
            for future in cf.as_completed(futures):
                n_finished += 1
                logger.info('Parallel aggregation futures collected: '
                            '{} out of {}'
                            .format(n_finished, len(chunks)))
                summary += future.result()

        return summary

    def _offshore_summary(self, summary, offshore_gid_adder=1e7,
                          offshore_capacity=600, offshore_gid_counts=494,
                          offshore_pixel_area=4):
        """Get the offshore supply curve point summary. Each offshore resource
        pixel will be summarized in its own supply curve point.

        Parameters
        ----------
        summary : list
            List of dictionaries, each being an onshore SC point summary.
        offshore_gid_adder : int | float
            The offshore Supply Curve gids will be set equal to the respective
            resource gids plus this number.
        offshore_capacity : int | float
            Offshore resource pixel generation capacity in MW.
        offshore_gid_counts : int
            Approximate number of exclusion pixels that would fall into an
            offshore pixel area.
        offshore_pixel_area : int | float
            Approximate area of offshore resource pixels in km2.

        Returns
        -------
        summary : list
            List of dictionaries, each being an SC point summary, includng SC
            points for single offshore resource pixels.
        """

        file_args = [self._excl_fpath, self._gen_fpath, self._data_layers,
                     self._excl_dict, self._power_density]
        with AggFileHandler(*file_args) as fhandler:

            inp = Aggregation._get_input_data(fhandler.gen, self._gen_fpath,
                                              self._res_class_dset,
                                              self._res_class_bins,
                                              self._cf_dset,
                                              self._lcoe_dset)

            res_data, res_class_bins, cf_data, lcoe_data, offshore_flag = inp

            if offshore_flag is not None:
                for i, _ in enumerate(summary):
                    summary[i]['offshore'] = 0

                for gen_gid, offshore in enumerate(offshore_flag):
                    if offshore:

                        # pylint: disable-msg=E1101
                        res_gid = fhandler.gen.meta.loc[gen_gid, 'gid']
                        latitude = fhandler.gen.meta.loc[gen_gid, 'latitude']
                        longitude = fhandler.gen.meta.loc[gen_gid, 'longitude']

                        offshore_sc_gid = int(res_gid + offshore_gid_adder)

                        res_class = -1
                        for ri, res_bin in enumerate(res_class_bins):
                            if (res_data[gen_gid] > np.min(res_bin)
                                    and res_data[gen_gid] < np.max(res_bin)):
                                res_class = ri
                                break

                        pointsum = {'sc_point_gid': offshore_sc_gid,
                                    'sc_row_ind': offshore_sc_gid,
                                    'sc_col_ind': offshore_sc_gid,
                                    'res_gids': [res_gid],
                                    'gen_gids': [gen_gid],
                                    'gid_counts': [int(offshore_gid_counts)],
                                    'mean_cf': cf_data[gen_gid],
                                    'mean_lcoe': lcoe_data[gen_gid],
                                    'mean_res': res_data[gen_gid],
                                    'capacity': offshore_capacity,
                                    'area_sq_km': offshore_pixel_area,
                                    'latitude': latitude,
                                    'longitude': longitude,
                                    'res_class': res_class,
                                    'offshore': 1,
                                    }

                        summary.append(pointsum)

        return summary

    def _offshore_data_layers(self, summary):
        """Agg categorical offshore data layers using NN to onshore points.

        Parameters
        ----------
        summary : DataFrame
            Summary of the SC points.

        Returns
        -------
        summary : DataFrame
            Summary of the SC points.
        """
        if 'offshore' in summary and self._data_layers is not None:
            cat_layers = [k for k, v in self._data_layers.items()
                          if v['method'].lower() == 'mode']

            if any(summary['offshore']) and any(cat_layers):
                logger.info('Aggregating the following columns for offshore '
                            'wind sites based on NN onshore sites: {}'
                            .format(cat_layers))
                offshore_mask = (summary['offshore'] == 1)
                offshore_summary = summary[offshore_mask]
                onshore_summary = summary[~offshore_mask]

                tree = cKDTree(onshore_summary[['latitude', 'longitude']])
                _, nn = tree.query(offshore_summary[['latitude', 'longitude']])

                for i, off_gid in enumerate(offshore_summary.index):
                    on_gid = onshore_summary.index.values[nn[i]]
                    logger.debug('Offshore gid {} is closest to onshore gid {}'
                                 .format(off_gid, on_gid))

                    for c in cat_layers:
                        summary.at[off_gid, c] = onshore_summary.at[on_gid, c]

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

    @staticmethod
    def _summary_to_df(summary):
        """Convert the agg summary list to a DataFrame.

        Parameters
        ----------
        summary : list
            List of dictionaries, each being an SC point summary.

        Returns
        -------
        summary : DataFrame
            Summary of the SC points.
        """
        summary = pd.DataFrame(summary)
        summary = summary.sort_values('sc_point_gid')
        summary = summary.reset_index(drop=True)
        summary.index.name = 'sc_gid'
        return summary

    @classmethod
    def summary(cls, excl_fpath, gen_fpath, tm_dset, excl_dict,
                res_class_dset=None, res_class_bins=None,
                cf_dset='cf_mean-means', lcoe_dset='lcoe_fcr-means',
                data_layers=None, resolution=64, power_density=None,
                offshore_gid_adder=1e7, offshore_capacity=600,
                gids=None, area_filter_kernel='queen', min_area=None,
                n_cores=None, **kwargs):
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
        offshore_gid_adder : int | float
            The offshore Supply Curve gids will be set equal to the respective
            resource gids plus this number.
        offshore_capacity : int | float
            Offshore resource pixel generation capacity in MW.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | NoneType
            Minimum required contiguous area filter in sq-km
        n_cores : int | None
            Number of cores to run summary on. 1 is serial, None is all
            available cpus.
        kwargs : dict
            Namespace of additional keyword args to init
            SupplyCurvePointSummary.

        Returns
        -------
        summary : DataFrame
            Summary of the SC points.
        """

        agg = cls(excl_fpath, gen_fpath, tm_dset, excl_dict,
                  res_class_dset=res_class_dset, res_class_bins=res_class_bins,
                  cf_dset=cf_dset, lcoe_dset=lcoe_dset,
                  data_layers=data_layers, resolution=resolution,
                  power_density=power_density, gids=gids,
                  area_filter_kernel=area_filter_kernel, min_area=min_area,
                  n_cores=n_cores)

        if n_cores == 1:
            summary = agg._serial_summary(
                agg._excl_fpath, agg._gen_fpath, agg._tm_dset,
                agg._excl_dict, agg._gen_index,
                res_class_dset=agg._res_class_dset,
                res_class_bins=agg._res_class_bins,
                cf_dset=agg._cf_dset,
                lcoe_dset=agg._lcoe_dset,
                data_layers=agg._data_layers,
                resolution=agg._resolution,
                power_density=agg._power_density,
                area_filter_kernel=agg._area_filter_kernel,
                min_area=agg._min_area,
                gids=gids,
                **kwargs)
        else:
            summary = agg._parallel_summary(**kwargs)

        summary = agg._offshore_summary(summary,
                                        offshore_gid_adder=offshore_gid_adder,
                                        offshore_capacity=offshore_capacity)
        summary = agg._summary_to_df(summary)
        summary = agg._offshore_data_layers(summary)

        return summary
