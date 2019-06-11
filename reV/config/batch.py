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
