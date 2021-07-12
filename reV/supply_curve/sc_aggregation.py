# -*- coding: utf-8 -*-
"""reV supply curve aggregation framework.

Created on Fri Jun 21 13:24:31 2019

@author: gbuster
"""
from concurrent.futures import as_completed
import h5py
import logging
import numpy as np
import psutil
import os
import pandas as pd
from warnings import warn

from reV.generation.base import BaseGen
from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.aggregation import (AbstractAggFileHandler,
                                          AbstractAggregation)
from reV.supply_curve.exclusions import FrictionMask
from reV.supply_curve.points import SupplyCurveExtent
from reV.supply_curve.point_summary import SupplyCurvePointSummary
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      OutputWarning, FileInputError,
                                      InputWarning)
from reV.utilities import log_versions

from rex.resource import Resource
from rex.multi_file_resource import MultiFileResource
from rex.utilities.execution import SpawnProcessPool

logger = logging.getLogger(__name__)


class SupplyCurveAggFileHandler(AbstractAggFileHandler):
    """
    Framework to handle aggregation summary context managers:
    - exclusions .h5 file
    - generation .h5 file
    - econ .h5 file (optional)
    - friction surface .h5 file (optional)
    - variable power density .csv (optional)
    """

    def __init__(self, excl_fpath, gen_fpath, econ_fpath=None,
                 data_layers=None, power_density=None, excl_dict=None,
                 friction_fpath=None, friction_dset=None,
                 area_filter_kernel='queen', min_area=None):
        """
        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        econ_fpath : str | None
            Filepath to .h5 reV econ output results. This is optional and only
            used if the lcoe_dset is not present in the gen_fpath file.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        power_density : float | str | None
            Power density in MW/km2 or filepath to variable power
            density file. None will attempt to infer a constant
            power density from the generation meta data technology.
            Variable power density csvs must have "gid" and "power_density"
            columns where gid is the resource gid (typically wtk or nsrdb gid)
            and the power_density column is in MW/km2.
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        friction_fpath : str | None
            Filepath to friction surface data (cost based exclusions).
            Must be paired with friction_dset. The friction data must be the
            same shape as the exclusions. Friction input creates a new output
            "mean_lcoe_friction" which is the nominal LCOE multiplied by the
            friction data.
        friction_dset : str | None
            Dataset name in friction_fpath for the friction surface data.
            Must be paired with friction_fpath. Must be same shape as
            exclusions.
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | None
            Minimum required contiguous area filter in sq-km
        """
        super().__init__(excl_fpath, excl_dict=excl_dict,
                         area_filter_kernel=area_filter_kernel,
                         min_area=min_area)

        self._gen = self._open_gen_econ_resource(gen_fpath, econ_fpath)
        # pre-initialize any import attributes
        _ = self._gen.meta

        self._data_layers = self._open_data_layers(data_layers)
        self._power_density = power_density
        self._parse_power_density()

        self._friction_layer = None
        if friction_fpath is not None and friction_dset is not None:
            self._friction_layer = FrictionMask(friction_fpath, friction_dset)

            if not np.all(self._friction_layer.shape == self._excl.shape):
                e = ('Friction layer shape {} must match exclusions shape {}!'
                     .format(self._friction_layer.shape, self._excl.shape))
                logger.error(e)
                raise FileInputError(e)

    @staticmethod
    def _open_gen_econ_resource(gen_fpath, econ_fpath):
        """Open a rex resource file handler for the reV generation and
        (optionally) the reV econ output(s).

        Parameters
        ----------
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        econ_fpath : str | None
            Filepath to .h5 reV econ output results. This is optional and only
            used if the lcoe_dset is not present in the gen_fpath file.

        Returns
        -------
        handler : Resource | MultiFileResource
            Open resource handler initialized with gen_fpath and
            (optionally) econ_fpath.
        """

        if econ_fpath is None:
            handler = Resource(gen_fpath)
        else:
            handler = MultiFileResource([gen_fpath, econ_fpath],
                                        check_files=True)

        return handler

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
        if self._friction_layer is not None:
            self._friction_layer.close()

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

    @property
    def friction_layer(self):
        """Get the friction layer (cost based exclusions).

        Returns
        -------
        friction_layer : None | FrictionMask
            Friction layer with scalar friction values if valid friction inputs
            were entered. Otherwise, None to not apply friction layer.
        """
        return self._friction_layer


class SupplyCurveAggregation(AbstractAggregation):
    """
    Supply curve points aggregation framework.

    Examples
    --------
    Standard outputs:

    sc_gid : int
        Unique supply curve gid. This is the enumerated supply curve points,
        which can have overlapping geographic locations due to different
        resource bins at the same geographic SC point.
    res_gids : list
        Stringified list of resource gids (e.g. original WTK or NSRDB resource
        GIDs) corresponding to each SC point.
    gen_gids : list
        Stringified list of generation gids (e.g. GID in the reV generation
        output, which corresponds to the reV project points and not
        necessarily the resource GIDs).
    gid_counts : list
        Stringified list of the sum of inclusion scalar values corresponding
        to each gen_gid and res_gid, where 1 is included, 0 is excluded, and
        0.7 is included with 70 percent of available land. Each entry in this
        list is associated with the corresponding entry in the gen_gids and
        res_gids lists.
    n_gids : int
        Total number of included pixels. This is a boolean sum and considers
        partial inclusions to be included (e.g. 1).
    mean_cf : float
        Mean capacity factor of each supply curve point (the arithmetic mean is
        weighted by the inclusion layer) (unitless).
    mean_lcoe : float
        Mean LCOE of each supply curve point (the arithmetic mean is weighted
        by the inclusion layer). Units match the reV econ output ($/MWh). By
        default, the LCOE is re-calculated using the multi-year mean capacity
        factor and annual energy production. This requires several datasets to
        be aggregated in the h5_dsets input: fixed_charge_rate, capital_cost,
        fixed_operating_cost, annual_energy_production, and
        variable_operating_cost. This recalc behavior can be disabled by
        setting recalc_lcoe=False.
    mean_res : float
        Mean resource, the resource dataset to average is provided by the user
        in 'res_class_dset'. The arithmetic mean is weighted by the inclusion
        layer.
    capacity : float
        Total capacity of each supply curve point (MW). Units are contingent on
        the 'power_density' input units of MW/km2.
    area_sq_km : float
        Total included area for each supply curve point in km2. This is based
        on the nominal area of each exclusion pixel which by default is
        calculated from the exclusion profile attributes. The NREL reV default
        is 0.0081 km2 pixels (90m x 90m). The area sum considers partial
        inclusions.
    latitude : float
        Supply curve point centroid latitude coordinate, in degrees
        (does not consider exclusions).
    longitude : float
        Supply curve point centroid longitude coordinate, in degrees
        (does not consider exclusions).
    country : str
        Country of the supply curve point based on the most common country
        of the associated resource meta data. Does not consider exclusions.
    state : str
        State of the supply curve point based on the most common state
        of the associated resource meta data. Does not consider exclusions.
    county : str
        County of the supply curve point based on the most common county
        of the associated resource meta data. Does not consider exclusions.
    elevation : float
        Mean elevation of the supply curve point based on the mean elevation
        of the associated resource meta data. Does not consider exclusions.
    timezone : int
        UTC offset of local timezone based on the most common timezone of the
        associated resource meta data. Does not consider exclusions.
    sc_point_gid : int
        Spatially deterministic supply curve point gid. Duplicate sc_point_gid
        values can exist due to resource binning.
    sc_row_ind : int
        Row index of the supply curve point in the aggregated exclusion grid.
    sc_col_ind : int
        Column index of the supply curve point in the aggregated exclusion grid
    res_class : int
        Resource class for the supply curve gid. Each geographic supply curve
        point (sc_point_gid) can have multiple resource classes associated with
        it, resulting in multiple supply curve gids (sc_gid) associated with
        the same spatially deterministic supply curve point.


    Optional outputs:

    mean_friction : float
        Mean of the friction data provided in 'friction_fpath' and
        'friction_dset'. The arithmetic mean is weighted by boolean
        inclusions and considers partial inclusions to be included.
    mean_lcoe_friction : float
        Mean of the nominal LCOE multiplied by mean_friction value.
    mean_{dset} : float
        Mean input h5 dataset(s) provided by the user in 'h5_dsets'. These
        mean calculations are weighted by the partial inclusion layer.
    data_layers : float | int | str | dict
        Requested data layer aggregations, each data layer must be the same
        shape as the exclusion layers.
        - mode: int | str
            Most common value of a given data layer after applying the
            boolean inclusion mask.
        - mean : float
            Arithmetic mean value of a given data layer weighted by the
            scalar inclusion mask (considers partial inclusions).
        - min : float | int
            Minimum value of a given data layer after applying the
            boolean inclusion mask.
        - max : float | int
            Maximum value of a given data layer after applying the
            boolean inclusion mask.
        - sum : float
            Sum of a given data layer weighted by the scalar inclusion mask
            (considers partial inclusions).
        - category : dict
            Dictionary mapping the unique values in the data_layer to the
            sum of inclusion scalar values associated with all pixels with that
            unique value.
    """

    def __init__(self, excl_fpath, gen_fpath, tm_dset, econ_fpath=None,
                 excl_dict=None, area_filter_kernel='queen', min_area=None,
                 resolution=64, excl_area=None, gids=None,
                 pre_extract_inclusions=False, res_class_dset=None,
                 res_class_bins=None, cf_dset='cf_mean-means',
                 lcoe_dset='lcoe_fcr-means', h5_dsets=None, data_layers=None,
                 power_density=None, friction_fpath=None, friction_dset=None,
                 cap_cost_scale=None, recalc_lcoe=True):
        """
        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        econ_fpath : str | None
            Filepath to .h5 reV econ output results. This is optional and only
            used if the lcoe_dset is not present in the gen_fpath file.
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | None
            Minimum required contiguous area filter in sq-km
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        excl_area : float | None, optional
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath, by default None
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        pre_extract_inclusions : bool, optional
            Optional flag to pre-extract/compute the inclusion mask from the
            provided excl_dict, by default False. Typically faster to compute
            the inclusion mask on the fly with parallel workers.
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
        h5_dsets : list | None
            Optional list of additional datasets from the source h5 gen/econ
            files to aggregate.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        power_density : float | str | None
            Power density in MW/km2 or filepath to variable power
            density file. None will attempt to infer a constant
            power density from the generation meta data technology.
            Variable power density csvs must have "gid" and "power_density"
            columns where gid is the resource gid (typically wtk or nsrdb gid)
            and the power_density column is in MW/km2.
        friction_fpath : str | None
            Filepath to friction surface data (cost based exclusions).
            Must be paired with friction_dset. The friction data must be the
            same shape as the exclusions. Friction input creates a new output
            "mean_lcoe_friction" which is the nominal LCOE multiplied by the
            friction data.
        friction_dset : str | None
            Dataset name in friction_fpath for the friction surface data.
            Must be paired with friction_fpath. Must be same shape as
            exclusions.
        cap_cost_scale : str | None
            Optional LCOE scaling equation to implement "economies of scale".
            Equations must be in python string format and return a scalar
            value to multiply the capital cost by. Independent variables in
            the equation should match the names of the columns in the reV
            supply curve aggregation table.
        recalc_lcoe : bool
            Flag to re-calculate the LCOE from the multi-year mean capacity
            factor and annual energy production data. This requires several
            datasets to be aggregated in the h5_dsets input: system_capacity,
            fixed_charge_rate, capital_cost, fixed_operating_cost,
            and variable_operating_cost.
        """
        log_versions(logger)
        logger.info('Initializing SupplyCurveAggregation...')
        logger.debug('Exclusion filepath: {}'.format(excl_fpath))
        logger.debug('Exclusion dict: {}'.format(excl_dict))

        super().__init__(excl_fpath, tm_dset, excl_dict=excl_dict,
                         area_filter_kernel=area_filter_kernel,
                         min_area=min_area, resolution=resolution,
                         excl_area=excl_area, gids=gids,
                         pre_extract_inclusions=pre_extract_inclusions)

        self._gen_fpath = gen_fpath
        self._econ_fpath = econ_fpath
        self._res_class_dset = res_class_dset
        self._res_class_bins = self._convert_bins(res_class_bins)
        self._cf_dset = cf_dset
        self._lcoe_dset = lcoe_dset
        self._h5_dsets = h5_dsets
        self._cap_cost_scale = cap_cost_scale
        self._power_density = power_density
        self._friction_fpath = friction_fpath
        self._friction_dset = friction_dset
        self._data_layers = data_layers
        self._recalc_lcoe = recalc_lcoe

        logger.debug('Resource class bins: {}'.format(self._res_class_bins))

        if self._cap_cost_scale is not None:
            if self._h5_dsets is None:
                self._h5_dsets = []

            self._h5_dsets += list(BaseGen.LCOE_ARGS)
            self._h5_dsets = list(set(self._h5_dsets))

        if self._power_density is None:
            msg = ('Supply curve aggregation power density not specified. '
                   'Will try to infer based on lookup table: {}'
                   .format(SupplyCurvePointSummary.POWER_DENSITY))
            logger.warning(msg)
            warn(msg, InputWarning)

        self._check_data_layers()
        self._gen_index = self._parse_gen_index(self._gen_fpath)

    def _check_files(self):
        """Do a preflight check on input files"""

        check_exists = [self._excl_fpath, self._gen_fpath]
        if self._econ_fpath is not None:
            check_exists.append(self._econ_fpath)

        for fpath in check_exists:
            if not os.path.exists(fpath):
                raise FileNotFoundError('Could not find input file: {}'
                                        .format(fpath))

        with h5py.File(self._excl_fpath, 'r') as f:
            if self._tm_dset not in f:
                raise FileInputError('Could not find techmap dataset "{}" '
                                     'in exclusions file: {}'
                                     .format(self._tm_dset,
                                             self._excl_fpath))

    def _check_data_layers(self, methods=('mean', 'max', 'min',
                           'mode', 'sum', 'category')):
        """Run pre-flight checks on requested aggregation data layers.

        Parameters
        ----------
        methods : list | tuple
            Data layer aggregation methods that are available to the user.
        """

        if self._data_layers is not None:
            logger.debug('Checking data layers...')

            with ExclusionLayers(self._excl_fpath) as f:
                shape_base = f.shape

            for k, v in self._data_layers.items():
                if 'dset' not in v:
                    raise KeyError('Data aggregation "dset" data layer "{}" '
                                   'must be specified.'.format(k))
                if 'method' not in v:
                    raise KeyError('Data aggregation "method" data layer "{}" '
                                   'must be specified.'.format(k))
                elif v['method'].lower() not in methods:
                    raise ValueError('Cannot recognize data layer agg method: '
                                     '"{}". Can only do: {}.'
                                     .format(v['method'], methods))
                if 'fpath' in v:
                    with ExclusionLayers(v['fpath']) as f:
                        if any(f.shape != shape_base):
                            msg = ('Data shape of data layer "{}" is {}, '
                                   'which does not match the baseline '
                                   'exclusions shape {}.'
                                   .format(k, f.shape, shape_base))
                            raise FileInputError(msg)

        logger.debug('Finished checking data layers.')

    @staticmethod
    def _get_input_data(gen, gen_fpath, econ_fpath, res_class_dset,
                        res_class_bins, cf_dset, lcoe_dset, h5_dsets):
        """Extract SC point agg input data args from higher level inputs.

        Parameters
        ----------
        gen : Resource | MultiFileResource
            Open rex resource handler initialized from gen_fpath and
            (optionally) econ_fpath.
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        econ_fpath : str
            Filepath to .h5 reV econ output results (optional argument if
            lcoe_dset is not found in gen_fpath).
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
        h5_dsets : list | None
            Optional list of additional datasets from the source h5 gen/econ
            files to aggregate.

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
        h5_dsets_data : dict | None
            If additional h5_dsets are requested, this will be a dictionary
            keyed by the h5 dataset names. The corresponding values will be
            the extracted arrays from the h5 files.
        """

        dset_list = (res_class_dset, cf_dset, lcoe_dset)
        labels = ('res_class_dset', 'cf_dset', 'lcoe_dset')
        temp = [None, None, None]
        for i, dset in enumerate(dset_list):
            if dset in gen.datasets:
                temp[i] = gen[dset]
            else:
                w = ('Could not find "{}" input as "{}" in '
                     'generation file: {}. Available datasets: {}'
                     .format(labels[i], dset, gen_fpath, gen.datasets))
                logger.warning(w)
                warn(w, OutputWarning)

        res_data, cf_data, lcoe_data = temp

        if res_class_dset is None or res_class_bins is None:
            res_class_bins = [None]

        # look for the datasets required by the LCOE re-calculation and make
        # lists of the missing datasets
        lcoe_recalc_req = ('fixed_charge_rate', 'capital_cost',
                           'fixed_operating_cost', 'variable_operating_cost',
                           'system_capacity')
        missing_lcoe_source = [k for k in lcoe_recalc_req
                               if k not in gen.datasets]
        missing_lcoe_request = []

        h5_dsets_data = None
        if h5_dsets is not None:
            missing_lcoe_request = [k for k in lcoe_recalc_req
                                    if k not in h5_dsets]

            if not isinstance(h5_dsets, (list, tuple)):
                e = ('Additional h5_dsets argument must be a list or tuple '
                     'but received: {} {}'.format(type(h5_dsets), h5_dsets))
                logger.error(e)
                raise TypeError(e)

            missing_h5_dsets = [k for k in h5_dsets if k not in gen.datasets]
            if any(missing_h5_dsets):
                msg = ('Could not find requested h5_dsets "{}" in '
                       'generation file: {} or econ file: {}. '
                       'Available datasets: {}'
                       .format(missing_h5_dsets, gen_fpath, econ_fpath,
                               gen.datasets))
                logger.error(msg)
                raise FileInputError(msg)

            h5_dsets_data = {dset: gen[dset] for dset in h5_dsets}

        if any(missing_lcoe_source):
            msg = ('Could not find the datasets in the gen source file that '
                   'are required to re-calculate the multi-year LCOE. If you '
                   'are running a multi-year job, it is strongly suggested '
                   'you pass through these datasets to re-calculate the LCOE '
                   'from the multi-year mean CF: {}'
                   .format(missing_lcoe_source))
            logger.warning(msg)
            warn(msg, InputWarning)
        if any(missing_lcoe_request):
            msg = ('It is strongly advised that you include the following '
                   'datasets in the h5_dsets request in order to re-calculate '
                   'the LCOE from the multi-year mean CF and AEP: {}'
                   .format(missing_lcoe_request))
            logger.warning(msg)
            warn(msg, InputWarning)

        offshore_flag = None
        if 'offshore' in gen.meta:
            offshore_flag = gen.meta['offshore'].values

        return (res_data, res_class_bins, cf_data, lcoe_data, offshore_flag,
                h5_dsets_data)

    @classmethod
    def run_serial(cls, excl_fpath, gen_fpath, tm_dset, gen_index,
                   econ_fpath=None, excl_dict=None, inclusion_mask=None,
                   area_filter_kernel='queen', min_area=None,
                   resolution=64, gids=None, args=None, res_class_dset=None,
                   res_class_bins=None, cf_dset='cf_mean-means',
                   lcoe_dset='lcoe_fcr-means', h5_dsets=None, data_layers=None,
                   power_density=None, friction_fpath=None, friction_dset=None,
                   excl_area=0.0081, cap_cost_scale=None, recalc_lcoe=True):
        """Standalone method to create agg summary - can be parallelized.

        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        tm_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        econ_fpath : str | None
            Filepath to .h5 reV econ output results. This is optional and only
            used if the lcoe_dset is not present in the gen_fpath file.
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        inclusion_mask : np.ndarray | dict | optional
            2D array pre-extracted inclusion mask where 1 is included and 0 is
            excluded. This must be either match the full exclusion shape or
            be a dict lookup of single-sc-point exclusion masks corresponding
            to the gids input and keyed by gids, by default None which will
            calculate exclusions on the fly for each sc point.
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | None
            Minimum required contiguous area filter in sq-km
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        args : list | None
            List of positional args for sc_point_method
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
        h5_dsets : list | None
            Optional list of additional datasets from the source h5 gen/econ
            files to aggregate.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        power_density : float | str | None
            Power density in MW/km2 or filepath to variable power
            density file. None will attempt to infer a constant
            power density from the generation meta data technology.
            Variable power density csvs must have "gid" and "power_density"
            columns where gid is the resource gid (typically wtk or nsrdb gid)
            and the power_density column is in MW/km2.
        friction_fpath : str | None
            Filepath to friction surface data (cost based exclusions).
            Must be paired with friction_dset. The friction data must be the
            same shape as the exclusions. Friction input creates a new output
            "mean_lcoe_friction" which is the nominal LCOE multiplied by the
            friction data.
        friction_dset : str | None
            Dataset name in friction_fpath for the friction surface data.
            Must be paired with friction_fpath. Must be same shape as
            exclusions.
        excl_area : float
            Area of an exclusion cell (square km).
        cap_cost_scale : str | None
            Optional LCOE scaling equation to implement "economies of scale".
            Equations must be in python string format and return a scalar
            value to multiply the capital cost by. Independent variables in
            the equation should match the names of the columns in the reV
            supply curve aggregation table.
        recalc_lcoe : bool
            Flag to re-calculate the LCOE from the multi-year mean capacity
            factor and annual energy production data. This requires several
            datasets to be aggregated in the h5_dsets input: system_capacity,
            fixed_charge_rate, capital_cost, fixed_operating_cost,
            and variable_operating_cost.

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
                gids = sc.valid_sc_points(tm_dset)
            elif np.issubdtype(type(gids), np.number):
                gids = [gids]

            slice_lookup = sc.get_slice_lookup(gids)

        logger.debug('Starting SupplyCurveAggregation serial with '
                     'supply curve {} gids'.format(len(gids)))

        cls._check_inclusion_mask(inclusion_mask, gids, exclusion_shape)

        # pre-extract handlers so they are not repeatedly initialized
        file_kwargs = {'econ_fpath': econ_fpath,
                       'data_layers': data_layers,
                       'power_density': power_density,
                       'excl_dict': excl_dict,
                       'area_filter_kernel': area_filter_kernel,
                       'min_area': min_area,
                       'friction_fpath': friction_fpath,
                       'friction_dset': friction_dset}
        with SupplyCurveAggFileHandler(excl_fpath, gen_fpath,
                                       **file_kwargs) as fh:
            inputs = cls._get_input_data(fh.gen, gen_fpath, econ_fpath,
                                         res_class_dset, res_class_bins,
                                         cf_dset, lcoe_dset, h5_dsets)
            n_finished = 0
            for gid in gids:
                gid_inclusions = cls._get_gid_inclusion_mask(
                    inclusion_mask, gid, slice_lookup,
                    resolution=resolution)

                for ri, res_bin in enumerate(inputs[1]):
                    try:
                        pointsum = SupplyCurvePointSummary.summarize(
                            gid,
                            fh.exclusions,
                            fh.gen,
                            tm_dset,
                            gen_index,
                            res_class_dset=inputs[0],
                            res_class_bin=res_bin,
                            cf_dset=inputs[2],
                            lcoe_dset=inputs[3],
                            h5_dsets=inputs[5],
                            data_layers=fh.data_layers,
                            resolution=resolution,
                            exclusion_shape=exclusion_shape,
                            power_density=fh.power_density,
                            args=args,
                            excl_dict=excl_dict,
                            inclusion_mask=gid_inclusions,
                            excl_area=excl_area,
                            close=False,
                            friction_layer=fh.friction_layer,
                            cap_cost_scale=cap_cost_scale,
                            recalc_lcoe=recalc_lcoe)

                    except EmptySupplyCurvePointError:
                        logger.debug('SC point {} is empty'.format(gid))
                    else:
                        pointsum['sc_point_gid'] = gid
                        pointsum['sc_row_ind'] = points.loc[gid, 'row_ind']
                        pointsum['sc_col_ind'] = points.loc[gid, 'col_ind']
                        pointsum['res_class'] = ri

                        summary.append(pointsum)
                        logger.debug('Serial aggregation completed gid {}: '
                                     '{} out of {} points complete'
                                     .format(gid, n_finished, len(gids)))

                n_finished += 1

        return summary

    def run_parallel(self, args=None, max_workers=None, sites_per_worker=100):
        """Get the supply curve points aggregation summary using futures.

        Parameters
        ----------
        args : tuple | list | None
            List of summary arguments to include. None defaults to all
            available args defined in the class attr.
        max_workers : int | None, optional
            Number of cores to run summary on. None is all
            available cpus, by default None
        sites_per_worker : int
            Number of sc_points to summarize on each worker, by default 100

        Returns
        -------
        summary : list
            List of dictionaries, each being an SC point summary.
        """
        chunks = int(np.ceil(len(self.gids) / sites_per_worker))
        chunks = np.array_split(self.gids, chunks)

        logger.info('Running supply curve point aggregation for '
                    'points {} through {} at a resolution of {} '
                    'on {} cores in {} chunks.'
                    .format(self.gids[0], self.gids[-1], self._resolution,
                            max_workers, len(chunks)))

        if self._inclusion_mask is not None:
            with SupplyCurveExtent(self._excl_fpath,
                                   resolution=self._resolution) as sc:
                assert sc.exclusions.shape == self._inclusion_mask.shape
                slice_lookup = sc.get_slice_lookup(self.gids)

        futures = []
        summary = []
        n_finished = 0
        loggers = [__name__, 'reV.supply_curve.point_summary', 'reV']
        with SpawnProcessPool(max_workers=max_workers, loggers=loggers) as exe:

            # iterate through split executions, submitting each to worker
            for gid_set in chunks:
                # submit executions and append to futures list
                chunk_incl_masks = None
                if self._inclusion_mask is not None:
                    chunk_incl_masks = {}
                    for gid in gid_set:
                        rs, cs = slice_lookup[gid]
                        chunk_incl_masks[gid] = self._inclusion_mask[rs, cs]

                futures.append(exe.submit(
                    self.run_serial,
                    self._excl_fpath, self._gen_fpath,
                    self._tm_dset, self._gen_index,
                    econ_fpath=self._econ_fpath,
                    excl_dict=self._excl_dict,
                    inclusion_mask=chunk_incl_masks,
                    res_class_dset=self._res_class_dset,
                    res_class_bins=self._res_class_bins,
                    cf_dset=self._cf_dset,
                    lcoe_dset=self._lcoe_dset,
                    h5_dsets=self._h5_dsets,
                    data_layers=self._data_layers,
                    resolution=self._resolution,
                    power_density=self._power_density,
                    friction_fpath=self._friction_fpath,
                    friction_dset=self._friction_dset,
                    area_filter_kernel=self._area_filter_kernel,
                    min_area=self._min_area,
                    gids=gid_set,
                    args=args,
                    excl_area=self._excl_area,
                    cap_cost_scale=self._cap_cost_scale,
                    recalc_lcoe=self._recalc_lcoe))

            # gather results
            for future in as_completed(futures):
                n_finished += 1
                summary += future.result()
                if n_finished % 10 == 0:
                    mem = psutil.virtual_memory()
                    logger.info('Parallel aggregation futures collected: '
                                '{} out of {}. Memory usage is {:.3f} GB out '
                                'of {:.3f} GB ({:.2f}% utilized).'
                                .format(n_finished, len(chunks),
                                        mem.used / 1e9, mem.total / 1e9,
                                        100 * mem.used / mem.total))

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
        sort_by = [x for x in ('sc_point_gid', 'res_class') if x in summary]
        summary = summary.sort_values(sort_by)
        summary = summary.reset_index(drop=True)
        summary.index.name = 'sc_gid'

        return summary

    def summarize(self, args=None, max_workers=None, sites_per_worker=100):
        """
        Get the supply curve points aggregation summary

        Parameters
        ----------
        args : tuple | list | None
            List of summary arguments to include. None defaults to all
            available args defined in the class attr.
        max_workers : int | None, optional
            Number of cores to run summary on. None is all
            available cpus, by default None
        sites_per_worker : int
            Number of sc_points to summarize on each worker, by default 100

        Returns
        -------
        summary : list
            List of dictionaries, each being an SC point summary.
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        if max_workers == 1:
            afk = self._area_filter_kernel
            summary = self.run_serial(self._excl_fpath, self._gen_fpath,
                                      self._tm_dset, self._gen_index,
                                      econ_fpath=self._econ_fpath,
                                      excl_dict=self._excl_dict,
                                      inclusion_mask=self._inclusion_mask,
                                      res_class_dset=self._res_class_dset,
                                      res_class_bins=self._res_class_bins,
                                      cf_dset=self._cf_dset,
                                      lcoe_dset=self._lcoe_dset,
                                      h5_dsets=self._h5_dsets,
                                      data_layers=self._data_layers,
                                      resolution=self._resolution,
                                      power_density=self._power_density,
                                      friction_fpath=self._friction_fpath,
                                      friction_dset=self._friction_dset,
                                      area_filter_kernel=afk,
                                      min_area=self._min_area,
                                      gids=self.gids, args=args,
                                      excl_area=self._excl_area,
                                      cap_cost_scale=self._cap_cost_scale,
                                      recalc_lcoe=self._recalc_lcoe)
        else:
            summary = self.run_parallel(args=args,
                                        max_workers=max_workers,
                                        sites_per_worker=sites_per_worker)

        if not any(summary):
            e = ('Supply curve aggregation found no non-excluded SC points. '
                 'Please check your exclusions or subset SC GID selection.')
            logger.error(e)
            raise EmptySupplyCurvePointError(e)

        summary = self._summary_to_df(summary)

        return summary

    @classmethod
    def summary(cls, excl_fpath, gen_fpath, tm_dset, econ_fpath=None,
                excl_dict=None, area_filter_kernel='queen', min_area=None,
                resolution=64, gids=None, pre_extract_inclusions=False,
                sites_per_worker=100, res_class_dset=None, res_class_bins=None,
                cf_dset='cf_mean-means', lcoe_dset='lcoe_fcr-means',
                h5_dsets=None, data_layers=None, power_density=None,
                friction_fpath=None, friction_dset=None,
                args=None, excl_area=None, max_workers=None,
                cap_cost_scale=None, recalc_lcoe=True):
        """Get the supply curve points aggregation summary.

        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        econ_fpath : str | None
            Filepath to .h5 reV econ output results. This is optional and only
            used if the lcoe_dset is not present in the gen_fpath file.
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | None
            Minimum required contiguous area filter in sq-km
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        pre_extract_inclusions : bool, optional
            Optional flag to pre-extract/compute the inclusion mask from the
            provided excl_dict, by default False. Typically faster to compute
            the inclusion mask on the fly with parallel workers.
        sites_per_worker : int
            Number of sc_points to summarize on each worker, by default 100
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
        h5_dsets : list | None
            Optional list of additional datasets from the source h5 gen/econ
            files to aggregate.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        power_density : float | str | None
            Power density in MW/km2 or filepath to variable power
            density file. None will attempt to infer a constant
            power density from the generation meta data technology.
            Variable power density csvs must have "gid" and "power_density"
            columns where gid is the resource gid (typically wtk or nsrdb gid)
            and the power_density column is in MW/km2.
        friction_fpath : str | None
            Filepath to friction surface data (cost based exclusions).
            Must be paired with friction_dset. The friction data must be the
            same shape as the exclusions. Friction input creates a new output
            "mean_lcoe_friction" which is the nominal LCOE multiplied by the
            friction data.
        friction_dset : str | None
            Dataset name in friction_fpath for the friction surface data.
            Must be paired with friction_fpath. Must be same shape as
            exclusions.
        args : tuple | list | None
            List of summary arguments to include. None defaults to all
            available args defined in the class attr.
        excl_area : float | None
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath.
        max_workers : int | None, optional
            Number of cores to run summary on. None is all
            available cpus, by default None
        cap_cost_scale : str | None
            Optional LCOE scaling equation to implement "economies of scale".
            Equations must be in python string format and return a scalar
            value to multiply the capital cost by. Independent variables in
            the equation should match the names of the columns in the reV
            supply curve aggregation table.
        recalc_lcoe : bool
            Flag to re-calculate the LCOE from the multi-year mean capacity
            factor and annual energy production data. This requires several
            datasets to be aggregated in the h5_dsets input: system_capacity,
            fixed_charge_rate, capital_cost, fixed_operating_cost,
            and variable_operating_cost.

        Returns
        -------
        summary : DataFrame
            Summary of the SC points.
        """

        agg = cls(excl_fpath, gen_fpath, tm_dset,
                  econ_fpath=econ_fpath,
                  excl_dict=excl_dict,
                  res_class_dset=res_class_dset,
                  res_class_bins=res_class_bins,
                  cf_dset=cf_dset,
                  lcoe_dset=lcoe_dset,
                  h5_dsets=h5_dsets,
                  data_layers=data_layers,
                  resolution=resolution,
                  power_density=power_density,
                  gids=gids,
                  pre_extract_inclusions=pre_extract_inclusions,
                  friction_fpath=friction_fpath,
                  friction_dset=friction_dset,
                  area_filter_kernel=area_filter_kernel,
                  min_area=min_area,
                  excl_area=excl_area,
                  cap_cost_scale=cap_cost_scale,
                  recalc_lcoe=recalc_lcoe)

        summary = agg.summarize(args=args,
                                max_workers=max_workers,
                                sites_per_worker=sites_per_worker)

        return summary
