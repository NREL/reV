# -*- coding: utf-8 -*-
"""reV config for batch run config.

Created on Mon Jun 10 13:49:53 2019

@author: gbuster
"""
import json
import pandas as pd
import os
import logging

from reV.utilities.exceptions import ConfigError
from reV.config.base_config import BaseConfig


logger = logging.getLogger(__name__)


class BatchConfig(BaseConfig):
    """Class for reV batch project json or csv configurations."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str
            File path to config json or csv (str).
        """

        self._pre_flight_fp(config)
        config_dir = None
        if config.endswith('.csv'):
            config_dir = os.path.dirname(os.path.realpath(config))
            config = BatchCsv(config)

        super().__init__(config, perform_str_rep=False)

        if config_dir is not None:
            self._config_dir = config_dir

        os.chdir(self.config_dir)
        self._pre_flight()

    @staticmethod
    def _pre_flight_fp(config):
        """Check to see that a valid config filepath was input

        Parameters
        ----------
        config : str
            File path to config json or csv (str).
        """
        if not isinstance(config, str):
            msg = ('Batch config can only take a str filepath, '
                   'but received a "{}".'.format(type(config)))
            logger.error(msg)
            raise ConfigError(msg)

        if not config.endswith('.json') and not config.endswith('.csv'):
            msg = ('Batch config needs to be .json or .csv but received: {}'
                   .format(config))
            logger.error(msg)
            raise ConfigError(msg)

        if not os.path.exists(config):
            msg = 'Batch config does not exist: {}'.format(config)
            logger.error(msg)
            raise FileNotFoundError(msg)

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
    def logging(self):
        """Get logging kwargs for the batch job.

        Returns
        -------
        dict
        """
        return self.get('logging', {"log_file": None, "log_level": "INFO"})

    @property
    def sets(self):
        """Get the list of batch job sets"""
        return self['sets']

    @property
    def pipeline_config(self):
        """Get the base pipeline config file with full file path."""
        return self['pipeline_config']


class BatchCsv(dict):
    """Class to parse a batch job CSV into standard dictionary (json) format"""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str
            File path to config json or csv (str).
        """

        self._pre_flight_fp(config)
        table = pd.read_csv(config, index_col=0)
        self._pre_flight_table(table)
        batch_dict = self._parse_batch_table(table)

        super().__init__(batch_dict)

    @staticmethod
    def _pre_flight_fp(config):
        """Check to see that a valid config filepath was input

        Parameters
        ----------
        config : str
            File path to config csv (str).
        """

        if not isinstance(config, str):
            msg = ('Batch config can only take a str filepath, '
                   'but received a "{}".'.format(type(config)))
            logger.error(msg)
            raise ConfigError(msg)

        if not config.endswith('.csv'):
            msg = ('BatchCsv config needs a csv filepath but received: {}'
                   .format(config))
            logger.error(msg)
            raise ConfigError(msg)

    @staticmethod
    def _pre_flight_table(table):
        """Check to ensure that the batch config csv table is valid.

        Parameters
        ----------
        table : pd.dataframe
            Extracted batch config csv. Must have "job" index (1st column)
            and "set_tag" and "files" columns.
        """
        if table.index.name != 'job':
            msg = 'Batch CSV config must have "job" as the first column.'
            logger.error(msg)
            raise ConfigError(msg)

        if 'set_tag' not in table or 'files' not in table:
            msg = 'Batch CSV config must have "set_tag" and "files" columns'
            logger.error(msg)
            raise ConfigError(msg)

        if (len(table.set_tag.unique()) != len(table)
                or len(table.index.unique()) != len(table)):
            msg = ('Batch CSV config must have completely '
                   'unique "set_tag" and "job" columns')
            logger.error(msg)
            raise ConfigError(msg)

    @staticmethod
    def _parse_batch_table(table):
        """Parse a batch config table into the typical batch config dict format

        Parameters
        ----------
        table : pd.dataframe
            Extracted batch config csv. Must have "job" index (1st column)
            and "set_tag" and "files" columns.

        Returns
        -------
        batch_dict : dict
            Batch config dictionary in the typical json-style format.
        """
        sets = []
        for _, job in table.iterrows():
            job_dict = job.to_dict()
            args = {k: [v] for k, v in job_dict.items()
                    if k not in ('set_tag', 'files')}
            files = json.loads(job_dict['files'].replace("'", '"'))
            set_config = {'args': args,
                          'set_tag': job_dict['set_tag'],
                          'files': files}
            sets.append(set_config)

        batch_dict = {'logging': {'log_file': None, 'log_level': 'INFO'},
                      'pipeline_config': './config_pipeline.json',
                      'sets': sets}

        return batch_dict
