# -*- coding: utf-8 -*-
"""
reV supply curve configs

Created on Mon Jan 28 11:43:27 2019

@author: gbuster
"""
import h5py
import os
import logging

from reV.utilities.exceptions import ConfigError, PipelineError
from reV.config.base_analysis_config import AnalysisConfig
from reV.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


class SupplyCurveAggregationConfig(AnalysisConfig):
    """SC Aggregation config."""

    NAME = 'agg'
    REQUIREMENTS = ('excl_fpath', 'gen_fpath', 'tm_dset')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        super().__init__(config)

        self._default_res_fpath = None
        self._default_res_class_dset = None
        self._default_res_class_bins = None
        self._default_cf_dset = 'cf_mean-means'
        self._default_lcoe_dset = 'lcoe_fcr-means'
        self._default_data_layers = None
        self._default_resolution = 64
        self._default_area_filter_kernel = 'queen'
        self._default_min_area = None
        self._default_excl_dict = None

        self._sc_agg_preflight()

    def _sc_agg_preflight(self):
        """Perform pre-flight checks on the SC agg config inputs"""
        with h5py.File(self.excl_fpath, mode='r') as f:
            dsets = list(f)

        if self.tm_dset not in dsets and self.res_fpath is None:
            raise ConfigError('Techmap dataset "{}" not found in exclusions '
                              'file, resource file input "res_fpath" is '
                              'required to create the techmap file.'
                              .format(self.tm_dset))

    @property
    def excl_fpath(self):
        """Get the exclusions filepath"""

        fpath = self['excl_fpath']

        if fpath == 'PIPELINE':
            fpath = Pipeline.parse_previous(
                self.dirout, 'aggregation', target='fpath',
                target_module='exclusions')[0]

        return fpath

    @property
    def gen_fpath(self):
        """Get the generation data filepath"""

        fpath = self['gen_fpath']

        if fpath == 'PIPELINE':
            target_modules = ['multi-year', 'collect', 'generation']
            for target_module in target_modules:
                try:
                    fpath = Pipeline.parse_previous(
                        self.dirout, 'supply-curve-aggregation',
                        target='fpath',
                        target_module=target_module)[0]
                except KeyError:
                    pass
                else:
                    break

            if fpath == 'PIPELINE':
                raise PipelineError('Could not parse gen_fpath from previous '
                                    'pipeline jobs.')
            else:
                logger.info('Supply curve aggregation using the following '
                            'pipeline input for gen_fpath: {}'.format(fpath))

        return fpath

    @property
    def res_fpath(self):
        """Get the resource data filepath"""
        res_fpath = self.get('res_fpath', self._default_res_fpath)
        if isinstance(res_fpath, str):
            if '{}' in res_fpath:
                for year in range(1998, 2018):
                    if os.path.exists(res_fpath.format(year)):
                        break

                res_fpath = res_fpath.format(year)

        return res_fpath

    @property
    def tm_dset(self):
        """Get the techmap dataset"""
        return self['tm_dset']

    @property
    def excl_dict(self):
        """Get the exclusions dictionary"""
        return self.get('excl_dict', self._default_excl_dict)

    @property
    def res_class_dset(self):
        """Get the resource class dataset"""
        return self.get('res_class_dset', self._default_res_class_dset)

    @property
    def res_class_bins(self):
        """Get the resource class bins"""
        return self.get('res_class_bins', self._default_res_class_bins)

    @property
    def cf_dset(self):
        """Get the capacity factor dataset"""
        return self.get('cf_dset', self._default_cf_dset)

    @property
    def lcoe_dset(self):
        """Get the LCOE dataset"""
        return self.get('lcoe_dset', self._default_lcoe_dset)

    @property
    def data_layers(self):
        """Get the data layers dict"""
        return self.get('data_layers', self._default_data_layers)

    @property
    def resolution(self):
        """Get the SC resolution"""
        return self.get('resolution', self._default_resolution)

    @property
    def excl_area(self):
        """Get the exclusion pixel area in km2"""
        return self.get('excl_area', None)

    @property
    def power_density(self):
        """Get the power density (MW/km2) or string to variable power
        density file path."""
        return self.get('power_density', None)

    @property
    def area_filter_kernel(self):
        """Get the minimum area filter kernel name ('queen' or 'rook')."""
        return self.get('area_filter_kernel', self._default_area_filter_kernel)

    @property
    def min_area(self):
        """Get the minimum area filter minimum area in km2."""
        return self.get('min_area', self._default_min_area)

    @property
    def friction_fpath(self):
        """Get the filepath to a friction surface h5 file
        (must be paired with friction_dset)."""
        return self.get('friction_fpath', None)

    @property
    def friction_dset(self):
        """Get the friction dataset name in friction_fpath."""
        return self.get('friction_dset', None)

    @property
    def check_excl_layers(self):
        """Get the check_excl_layers flag."""
        return self.get('check_excl_layers', False)


class SupplyCurveConfig(AnalysisConfig):
    """SC config."""

    NAME = 'sc'
    REQUIREMENTS = ('sc_points', 'trans_table', 'fixed_charge_rate')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        super().__init__(config)

        self._default_sc_features = None
        self._default_transmission_costs = None
        self._default_sort_on = 'total_lcoe'
        self._default_n_dirs = 2

        self._sc_preflight()

    def _sc_preflight(self):
        """Perform pre-flight checks on the SC config inputs"""
        missing = []
        for req in self.REQUIREMENTS:
            if self.get(req, None) is None:
                missing.append(req)
        if any(missing):
            raise ConfigError('Supply Curve config missing the following '
                              'keys: {}'.format(missing))

    @property
    def sc_points(self):
        """Get the supply curve points summary file path"""

        sc_points = self['sc_points']

        if sc_points == 'PIPELINE':
            sc_points = Pipeline.parse_previous(
                self.dirout, 'supply-curve', target='fpath')[0]

            logger.info('Supply curve using the following '
                        'pipeline input for sc_points: {}'.format(sc_points))

        return sc_points

    @property
    def trans_table(self):
        """Get the transmission table file path"""
        return self['trans_table']

    @property
    def fixed_charge_rate(self):
        """Get the fixed charge rate input"""
        return self['fixed_charge_rate']

    @property
    def sc_features(self):
        """Get the supply curve features input."""
        return self.get('sc_features', self._default_sc_features)

    @property
    def transmission_costs(self):
        """Get the transmission costs input."""
        return self.get('transmission_costs', self._default_transmission_costs)

    @property
    def simple(self):
        """Get the simple flag."""
        simple = bool(self.get('simple', False))
        if simple:
            self.check_overwrite_keys('simple', 'line_limited')

        return simple

    @property
    def line_limited(self):
        """Get the line-limited flag."""
        return bool(self.get('line_limited', False))

    @property
    def sort_on(self):
        """Get the SC table column label to sort on.
        This determines the ordering of the buildout algorithm."""
        return self.get('sort_on', self._default_sort_on)

    @property
    def wind_dirs(self):
        """Get the supply curve power-rose wind directions"""
        return self.get('wind_dirs', None)

    @property
    def n_dirs(self):
        """Get the number of prominent wind directions to exclude"""
        return self.get('n_dirs', self._default_n_dirs)

    @property
    def downwind(self):
        """Get the flag to exclude downwind as well as upwind sites"""
        return self.get('downwind', False)

    @property
    def max_workers(self):
        """Get the number of workers to use during computation"""
        return self.get('max_workers', None)
