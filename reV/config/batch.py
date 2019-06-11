# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:49:53 2019

@author: gbuster
"""
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

    def _pre_flight(self):
        """Run pre-flight checks on the config."""
        if 'pipeline_config' not in self:
            raise ConfigError('Batch config needs "pipeline_config" arg!')

    @property
    def config_pipeline(self):
        """Get the base pipeline config file with full file path."""
        return self['pipeline_config']
