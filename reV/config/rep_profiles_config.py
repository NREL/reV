# -*- coding: utf-8 -*-
"""
reV representative profile config

Created on Mon Jan 28 11:43:27 2019

@author: gbuster
"""
import logging

from reV.utilities.exceptions import ConfigError, PipelineError
from reV.config.base_analysis_config import AnalysisConfig
from reV.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


class RepProfilesConfig(AnalysisConfig):
    """Representative Profiles config."""

    NAME = 'rep_profiles'
    REQUIREMENTS = ('gen_fpath', 'rev_summary', 'reg_cols')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        super().__init__(config)

        self._default_cf_dset = 'cf_profile'
        self._default_rep_method = 'meanoid'
        self._default_err_method = 'rmse'
        self._default_weight = 'gid_counts'
        self._default_n_profiles = 1

        self._rep_profiles_preflight()

    def _rep_profiles_preflight(self):
        """Perform pre-flight checks on the rep profiles config inputs"""
        missing = []
        for req in self.REQUIREMENTS:
            if self.get(req, None) is None:
                missing.append(req)
        if any(missing):
            raise ConfigError('Rep profiles config missing the following '
                              'keys: {}'.format(missing))

    @property
    def gen_fpath(self):
        """Get the generation data filepath"""

        fpath = self['gen_fpath']

        if fpath == 'PIPELINE':
            target_modules = ['multi-year', 'collect', 'generation']
            for target_module in target_modules:
                try:
                    fpath = Pipeline.parse_previous(
                        self.dirout, 'rep-profiles', target='fpath',
                        target_module=target_module)[0]
                except KeyError:
                    pass
                else:
                    break

            if fpath == 'PIPELINE':
                raise PipelineError('Could not parse gen_fpath from previous '
                                    'pipeline jobs.')
            else:
                logger.info('Rep profiles using the following '
                            'pipeline input for gen_fpath: {}'.format(fpath))

        return fpath

    @property
    def cf_dset(self):
        """Get the capacity factor dataset to get gen profiles from"""
        return self.get('cf_dset', self._default_cf_dset)

    @property
    def rev_summary(self):
        """Get the rev summary input arg."""

        fpath = self['rev_summary']

        if fpath == 'PIPELINE':
            target_modules = ['aggregation', 'supply-curve']
            for target_module in target_modules:
                try:
                    fpath = Pipeline.parse_previous(
                        self.dirout, 'rep-profiles', target='fpath',
                        target_module=target_module)[0]
                except KeyError:
                    pass
                else:
                    break

            if fpath == 'PIPELINE':
                raise PipelineError('Could not parse rev_summary from '
                                    'previous pipeline jobs.')
            else:
                logger.info('Rep profiles using the following '
                            'pipeline input for rev_summary: {}'.format(fpath))

        return fpath

    @property
    def reg_cols(self):
        """Get the region columns input arg."""
        reg_cols = self['reg_cols']
        if isinstance(reg_cols, str):
            reg_cols = [reg_cols]
        return reg_cols

    @property
    def rep_method(self):
        """Get the representative profile method"""
        return self.get('rep_method', self._default_rep_method)

    @property
    def err_method(self):
        """Get the representative profile error method"""
        return self.get('err_method', self._default_err_method)

    @property
    def n_profiles(self):
        """Get the number of representative profiles to save."""
        return self.get('n_profiles', self._default_n_profiles)

    @property
    def weight(self):
        """Get the reV supply curve column to use for a weighted average in
        the representative profile meanoid algorithm."""
        return self.get('weight', self._default_weight)

    @property
    def aggregate_profiles(self):
        """Flag to calculate the aggregate (weighted meanoid) profile for each
        supply curve point. This behavior is instead of finding the single
        profile per region closest to the meanoid."""
        return bool(self.get('aggregate_profiles', False))
