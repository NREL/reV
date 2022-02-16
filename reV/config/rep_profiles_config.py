# -*- coding: utf-8 -*-
"""
reV representative profile config

Created on Mon Jan 28 11:43:27 2019

@author: gbuster
"""
import logging

from reV.utilities.exceptions import PipelineError
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
                try:
                    fpath = Pipeline.parse_previous(
                        self.dirout, 'rep-profiles', target='gen_fpath',
                        target_module='supply-curve-aggregation')[0]
                except KeyError:
                    pass

            if fpath == 'PIPELINE':
                msg = 'Could not parse gen_fpath from previous pipeline jobs.'
                logger.error(msg)
                raise PipelineError(msg)
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
        reg_cols = self.get('reg_cols', None)
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
        aggregate = bool(self.get('aggregate_profiles', False))
        if aggregate:
            self.check_overwrite_keys('aggregate_profiles', 'rep_method'
                                      'err_method', 'n_profiles')

        return aggregate
