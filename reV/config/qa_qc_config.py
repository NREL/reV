# -*- coding: utf-8 -*-
"""
reV QA/QC config
"""
import logging
from warnings import warn

from reV.utilities.exceptions import PipelineError
from reV.config.base_analysis_config import AnalysisConfig
from reV.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


class QaQcConfig(AnalysisConfig):
    """QA/QC config."""

    NAME = 'QA-QC'
    REQUIREMENTS = ('modules',)

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        super().__init__(config)
        self._modules = None

    @property
    def modules(self):
        """
        Get the sub-group of modules to run QA/QC on
        """
        if self._modules is None:
            self._modules = self['modules']

        return self._modules

    @property
    def module_names(self):
        """
        Get list of module names to be run through QA/QC
        """
        return list(self.modules.keys())

    @property
    def collect(self):
        """Get the collect QA/QC inputs in the config dict."""
        collect = self.modules.get('collect', None)
        if collect is not None:
            collect = QaQcModule('collect', collect, self.dirout)

        return collect

    @property
    def multi_year(self):
        """Get the multi-year QA/QC inputs in the config dict."""
        multi_year = self.modules.get('multi-year', None)
        if multi_year is not None:
            multi_year = QaQcModule('multi-year', multi_year, self.dirout)

        return multi_year

    @property
    def rep_profiles(self):
        """Get the representative profile QA/QC inputs in the config dict."""
        rep_profiles = self.modules.get('rep-profiles', None)
        if rep_profiles is not None:
            rep_profiles = QaQcModule('rep-profiles', rep_profiles,
                                      self.dirout)

        return rep_profiles

    @property
    def supply_curve_aggregation(self):
        """Get the aggregation QA/QC inputs in the config dict."""
        aggregation = self.modules.get('supply-curve-aggregation', None)
        if aggregation is not None:
            aggregation = QaQcModule(
                'aggregation', aggregation, self.dirout)

        return aggregation

    @property
    def supply_curve(self):
        """Get the supply curve QA/QC inputs in the config dict."""
        supply_curve = self.modules.get('supply-curve', None)
        if supply_curve is not None:
            supply_curve = QaQcModule(
                'supply-curve', supply_curve, self.dirout)

        return supply_curve

    def get_module_inputs(self, module_name):
        """
        Get QA/QC inputs for module

        Parameters
        ----------
        module_name : str
            Module name / config section name to get QA/QC inputs for

        Returns
        -------
        QaQcModule
        """
        return QaQcModule(module_name, self.modules[module_name], self.dirout)


class QaQcModule:
    """Class to handle Module QA/QC"""

    def __init__(self, module_name, config, out_root):
        """
        Parameters
        ----------
        config : dict
            Dictionary with pre-extracted config input group.
        """
        if not isinstance(config, dict):
            raise TypeError('Config input must be a dict but received: {}'
                            .format(type(config)))

        self._name = module_name
        self._config = config
        self._out_root = out_root
        self._default_plot_type = 'plotly'
        self._default_cmap = 'viridis'
        self._default_plot_step = 100
        self._default_lcoe = 'mean_lcoe'
        self._default_area_filter_kernel = 'queen'

    @property
    def fpath(self):
        """Get the reV module output filepath(s)

        Returns
        -------
        fpaths : str | list
            One or more filepaths output by current module being QA'd
        """

        fpath = self._config['fpath']

        if fpath == 'PIPELINE':
            target_modules = [self._name]
            for target_module in target_modules:
                try:
                    fpath = Pipeline.parse_previous(
                        self._out_root, 'qa-qc', target='fpath',
                        target_module=target_module)
                except KeyError:
                    pass
                else:
                    break

            if fpath == 'PIPELINE':
                raise PipelineError('Could not parse fpath from previous '
                                    'pipeline jobs.')
            else:
                logger.info('QA/QC using the following '
                            'pipeline input for fpath: {}'.format(fpath))

        return fpath

    @property
    def sub_dir(self):
        """
        QA/QC sub directory for this module's outputs
        """
        return self._config.get('sub_dir', None)

    @property
    def plot_type(self):
        """Get the QA/QC plot type: either 'plot' or 'plotly'"""
        return self._config.get('plot_type', self._default_plot_type)

    @property
    def dsets(self):
        """Get the reV_h5 dsets to QA/QC"""
        return self._config.get('dsets', None)

    @property
    def group(self):
        """Get the reV_h5 group to QA/QC"""
        return self._config.get('group', None)

    @property
    def process_size(self):
        """Get the reV_h5 process_size for QA/QC"""
        return self._config.get('process_size', None)

    @property
    def max_workers(self):
        """Get the reV_h5 max_workers for QA/QC"""
        return self._config.get('max_workers', None)

    @property
    def cmap(self):
        """Get the QA/QC plot colormap"""
        return self._config.get('cmap', self._default_cmap)

    @property
    def plot_step(self):
        """Get the QA/QC step between exclusion mask points to plot"""
        return self._config.get('cmap', self._default_plot_step)

    @property
    def columns(self):
        """Get the supply_curve columns to QA/QC"""
        return self._config.get('columns', None)

    @property
    def lcoe(self):
        """Get the supply_curve lcoe column to plot"""
        return self._config.get('lcoe', self._default_lcoe)

    @property
    def excl_fpath(self):
        """Get the source exclusions filepath"""
        excl_fpath = self._config.get('excl_fpath', 'PIPELINE')

        if excl_fpath == 'PIPELINE':
            try:
                excl_fpath = Pipeline.parse_previous(
                    self._out_root, 'qa-qc', target='excl_fpath',
                    target_module='supply-curve-aggregation')[0]
            except KeyError:
                excl_fpath = None
                msg = ('Could not parse excl_fpath from previous '
                       'pipeline jobs, defaulting to: {}'.format(excl_fpath))
                logger.warning(msg)
                warn(msg)
            else:
                logger.info('QA/QC using the following '
                            'pipeline input for excl_fpath: {}'
                            .format(excl_fpath))

        return excl_fpath

    @property
    def excl_dict(self):
        """Get the exclusions dictionary"""
        excl_dict = self._config.get('excl_dict', 'PIPELINE')

        if excl_dict == 'PIPELINE':
            try:
                excl_dict = Pipeline.parse_previous(
                    self._out_root, 'qa-qc', target='excl_dict',
                    target_module='supply-curve-aggregation')[0]
            except KeyError:
                excl_dict = None
                msg = ('Could not parse excl_dict from previous '
                       'pipeline jobs, defaulting to: {}'.format(excl_dict))
                logger.warning(msg)
                warn(msg)
            else:
                logger.info('QA/QC using the following '
                            'pipeline input for excl_dict: {}'
                            .format(excl_dict))

        return excl_dict

    @property
    def area_filter_kernel(self):
        """Get the minimum area filter kernel name ('queen' or 'rook')."""
        area_filter_kernel = self._config.get('area_filter_kernel', 'PIPELINE')

        if area_filter_kernel == 'PIPELINE':
            try:
                area_filter_kernel = Pipeline.parse_previous(
                    self._out_root, 'qa-qc', target='area_filter_kernel',
                    target_module='supply-curve-aggregation')[0]
            except KeyError:
                area_filter_kernel = self._default_area_filter_kernel
                msg = ('Could not parse area_filter_kernel from previous '
                       'pipeline jobs, defaulting to: {}'
                       .format(area_filter_kernel))
                logger.warning(msg)
                warn(msg)
            else:
                logger.info('QA/QC using the following '
                            'pipeline input for area_filter_kernel: {}'
                            .format(area_filter_kernel))

        return area_filter_kernel

    @property
    def min_area(self):
        """Get the minimum area filter minimum area in km2."""
        min_area = self._config.get('min_area', 'PIPELINE')

        if min_area == 'PIPELINE':
            try:
                min_area = Pipeline.parse_previous(
                    self._out_root, 'qa-qc', target='min_area',
                    target_module='supply-curve-aggregation')[0]
            except KeyError:
                min_area = None
                msg = ('Could not parse min_area from previous '
                       'pipeline jobs, defaulting to: {}.'.format(min_area))
                logger.warning(msg)
                warn(msg)
            else:
                logger.info('QA/QC using the following '
                            'pipeline input for min_area: {}'
                            .format(min_area))

        return min_area
