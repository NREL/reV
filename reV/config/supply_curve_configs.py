# -*- coding: utf-8 -*-
"""
reV supply curve configs

Created on Mon Jan 28 11:43:27 2019

@author: gbuster
"""
import os
import logging

from reV.utilities.exceptions import ConfigError, PipelineError
from reV.config.base_analysis_config import AnalysisConfig
from reV.pipeline.pipeline import Pipeline


logger = logging.getLogger(__name__)


class AggregationConfig(AnalysisConfig):
    """SC Aggregation config."""

    NAME = 'sc_agg'
    REQUIREMENTS = ('fpath_excl', 'fpath_gen', 'fpath_techmap', 'dset_tm')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        super().__init__(config)

        self._default_fpath_res = None
        self._default_res_class_dset = None
        self._default_res_class_bins = None
        self._default_dset_cf = 'cf_mean-means'
        self._default_dset_lcoe = 'lcoe_fcr-means'
        self._default_data_layers = None
        self._default_resolution = 64

        self._preflight()

    def _preflight(self):
        """Perform pre-flight checks on the SC agg config inputs"""
        missing = []
        for req in self.REQUIREMENTS:
            if self.get(req, None) is None:
                missing.append(req)
        if any(req):
            raise ConfigError('SC Aggregation config missing the following '
                              'keys: {}'.format(missing))

        if not os.path.exists(self.fpath_techmap) and self.fpath_res is None:
            raise ConfigError('Techmap file not found, resource file input'
                              '"fpath_res" is required to create the techmap '
                              'file.')

    @property
    def fpath_excl(self):
        """Get the exclusions filepath"""
        return self['fpath_excl']

    @property
    def fpath_gen(self):
        """Get the generation data filepath"""
        fpath = self['fpath_gen']

        if fpath == 'PIPELINE':
            target_modules = ['multi-year', 'collect', 'generation']
            for target_module in target_modules:
                try:
                    fpath = Pipeline.parse_previous(
                        self.dirout, 'aggregation', target='fpath',
                        target_module=target_module)[0]
                except KeyError:
                    pass
                else:
                    break

            if fpath == 'PIPELINE':
                raise PipelineError('Could not parse fpath_gen from previous '
                                    'pipeline jobs.')
            else:
                logger.info('Supply curve aggregation using the following '
                            'pipeline input for fpath_gen: {}'.format(fpath))

        return fpath

    @property
    def fpath_res(self):
        """Get the resource data filepath"""
        return self.get('fpath_res', self._default_fpath_res)

    @property
    def fpath_techmap(self):
        """Get the techmap data filepath"""
        return self['fpath_techmap']

    @property
    def dset_tm(self):
        """Get the techmap dataset"""
        return self['dset_tm']

    @property
    def res_class_dset(self):
        """Get the resource class dataset"""
        return self.get('res_class_dset', self._default_res_class_dset)

    @property
    def res_class_bins(self):
        """Get the resource class bins"""
        return self.get('res_class_bins', self._default_res_class_bins)

    @property
    def dset_cf(self):
        """Get the capacity factor dataset"""
        return self.get('dset_cf', self._default_dset_cf)

    @property
    def dset_lcoe(self):
        """Get the LCOE dataset"""
        return self.get('dset_lcoe', self._default_dset_lcoe)

    @property
    def data_layers(self):
        """Get the data layers dict"""
        return self.get('data_layers', self._default_data_layers)

    @property
    def resolution(self):
        """Get the SC resolution"""
        return self.get('resolution', self._default_resolution)


class SupplyCurveConfig(AnalysisConfig):
    """SC config."""

    NAME = 'sc'

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
        self._default_simple = False

    @property
    def sc_points(self):
        """Get the supply curve points summary file path"""

        sc_points = self['sc_points']

        if sc_points == 'PIPELINE':
            fpath = Pipeline.parse_previous(
                self.dirout, 'supply-curve', target='fpath')[0]

            logger.info('Supply curve using the following '
                        'pipeline input for sc_points: {}'.format(fpath))

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
        return self.get('simple', self._default_simple)
