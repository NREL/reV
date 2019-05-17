# -*- coding: utf-8 -*-
"""
reV analysis configs (generation, lcoe, etc...)

Created on Mon Jan 28 11:43:27 2019

@author: gbuster
"""
import logging
from warnings import warn

from reV.config.base_config import BaseConfig
from reV.config.execution import (BaseExecutionConfig, PeregrineConfig,
                                  EagleConfig)
from reV.utilities.exceptions import ConfigError


logger = logging.getLogger(__name__)


class CollectionConfig(BaseConfig):
    """Base analysis config (generation, lcoe, etc...)."""

    def __init__(self, config_dict):
        self._dirout = None
        self._logdir = None
        self._ti = False
        self._parallel = False
        self._dsets = None
        self._file_prefixes = None
        self._ec = None
        super().__init__(config_dict)

    @property
    def dirout(self):
        """Get the output directory.

        Returns
        -------
        _dirout : str
            Target path for reV output files.
        """
        if self._dirout is None:
            # set default value
            self._dirout = './out'
            if 'output_directory' in self['directories']:
                self._dirout = self['directories']['output_directory']
        return self._dirout

    @property
    def logdir(self):
        """Get the logging directory.

        Returns
        -------
        _logdir : str
            Target path for reV log files.
        """
        if self._logdir is None:
            # set default value
            self._logdir = './logs'
            if 'logging_directory' in self['directories']:
                self._logdir = self['directories']['logging_directory']
        return self._logdir

    @property
    def coldir(self):
        """Get the directory to collect files from.

        Returns
        -------
        _coldir : str
            Target path to collect h5 files from.
        """
        return self['directories']['collect_directory']

    @property
    def project_points(self):
        """Get the collection project points.

        Returns
        -------
        _project_points : str
            Target path for project points file.
        """
        return self['project_points']

    @property
    def parallel(self):
        """Get the flag to do a parallel collection.

        Returns
        -------
        _parallel : bool
            Flag to collect data in parallel.
        """
        if 'parallel' in self['project_control']:
            self._parallel = self['project_control']['parallel']
        return self._parallel

    @property
    def time_index(self):
        """Get the flag to collect the time index.

        Returns
        -------
        _ti : bool
            Flag to collect time index.
        """
        if 'time_index' in self['project_control']:
            self._ti = self['project_control']['time_index']
        return self._ti

    @property
    def dsets(self):
        """Get dset names to collect.

        Returns
        -------
        _dsets : list
            list of dset names to collect.
        """

        if self._dsets is None:
            self._dsets = self['project_control']['dsets']
            if not isinstance(self._dsets, list):
                self._dsets = list(self._dsets)
        return self._dsets

    @property
    def file_prefixes(self):
        """Get the file prefixes to collect.

        Returns
        -------
        _file_prefixes : list
            list of file prefixes to collect.
        """

        if self._file_prefixes is None:
            self._file_prefixes = self['project_control']['file_prefixes']
            if not isinstance(self._file_prefixes, list):
                self._file_prefixes = list(self._file_prefixes)
        return self._file_prefixes

    @property
    def execution_control(self):
        """Get the execution control object.

        Returns
        -------
        _ec : BaseExecutionConfig | PeregrineConfig | EagleConfig
            reV execution config object specific to the execution_control
            option.
        """
        if self._ec is None:
            ec = self['execution_control']
            # static map of avail execution options with corresponding classes
            ec_config_types = {'local': BaseExecutionConfig,
                               'peregrine': PeregrineConfig,
                               'eagle': EagleConfig}
            if 'option' in ec:
                try:
                    # Try setting the attribute to the appropriate exec option
                    self._ec = ec_config_types[ec['option'].lower()](ec)
                except KeyError:
                    # Option not found
                    raise ConfigError('Execution control option not '
                                      'recognized: "{}". '
                                      'Available options are: {}.'
                                      .format(ec['option'].lower(),
                                              list(ec_config_types.keys())))
            else:
                # option not specified, default to a base execution (local)
                warn('Execution control option not specified. '
                     'Defaulting to a local run.')
                self._ec = BaseExecutionConfig(ec)
        return self._ec
