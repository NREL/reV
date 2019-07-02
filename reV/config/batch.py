# -*- coding: utf-8 -*-
"""reV config for batch run config.

Created on Mon Jun 10 13:49:53 2019

@author: gbuster
"""
import os

from reV.utilities.exceptions import ConfigError
from reV.config.base_config import BaseConfig


class BatchConfig(BaseConfig):
    """Config for reV batch jobs."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str
            File path to config json (str).
        """

        if not isinstance(config, str):
            raise ConfigError('Batch config can only take a str filepath, '
                              'but received a "{}".'.format(type(config)))

        super().__init__(config)
        self._pre_flight()

    def _pre_flight(self):
        """Run pre-flight checks on the config."""

        self._check_pipeline()
        self._check_sets()

    def _check_pipeline(self):
        """Check the pipeline config file in the batch config."""

        if 'pipeline_config' not in self:
            raise ConfigError('Batch config needs "pipeline_config" arg!')

        if not os.path.exists(self['pipeline_config']):
            raise ConfigError('Could not find the pipeline config file: {}'
                              .format(self['pipeline_config']))

    def _check_sets(self):
        """Check the batch sets for required inputs and valid files."""

        if 'sets' not in self:
            raise ConfigError('Batch config needs "sets" arg!')

        if not isinstance(self['sets'], list):
            raise ConfigError('Batch config needs "sets" arg to be a list!')

        for s in self['sets']:
            if not isinstance(s, dict):
                raise ConfigError('Batch sets must be dictionaries.')
            if 'args' not in s:
                raise ConfigError('All batch sets must have "args" key.')
            if 'files' not in s:
                raise ConfigError('All batch sets must have "files" key.')

            for fpath in s['files']:
                if not os.path.exists(fpath):
                    raise ConfigError('Could not find file to modify in batch '
                                      'jobs: {}'.format(fpath))

    @property
    def config_pipeline(self):
        """Get the base pipeline config file with full file path."""
        return self['pipeline_config']
