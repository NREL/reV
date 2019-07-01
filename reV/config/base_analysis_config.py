# -*- coding: utf-8 -*-
"""
reV Base analysis Configuration Frameworks
"""
import os
import logging
from warnings import warn

from reV.config.base_config import BaseConfig
from reV.config.execution import (BaseExecutionConfig, PeregrineConfig,
                                  EagleConfig)
from reV.utilities.exceptions import ConfigError, ConfigWarning


logger = logging.getLogger(__name__)


class AnalysisConfig(BaseConfig):
    """Base analysis config (generation, lcoe, etc...)."""

    NAME = None

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        self._years = None
        self._dirout = None
        self._logdir = None
        self._ec = None
        super().__init__(config)

    @property
    def years(self):
        """Get the analysis years.

        Returns
        -------
        _years : list
            List of years to analyze. If this is a single year run, this return
            value is a single entry list. If no analysis_years are specified,
            the code will look anticipate a year in the input files.
        """

        if self._years is None:
            self._years = [None]
            if 'analysis_years' in self['project_control']:
                self._years = self['project_control']['analysis_years']
                if isinstance(self._years, list) is False:
                    self._years = [self._years]
            else:
                warn('Analysis years may not have been specified, may default '
                     'to available years in inputs files.', ConfigWarning)
        return self._years

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

    @property
    def name(self):
        """Get the job name, defaults to the output directory name.

        Returns
        -------
        _name : str
            reV job name.
        """

        if self._name is None:

            # name defaults to base directory name
            self._name = os.path.basename(os.path.normpath(self.dirout))

            # collect name is simple, will be added to what is being collected
            if self.NAME == 'collect':
                self._name = self.NAME

            # Analysis job name tag (helps ensure unique job name)
            elif self.NAME is not None:
                self._name += '_{}'.format(self.NAME)

            # name specified by user config
            if 'name' in self['project_control']:
                if self['project_control']['name']:
                    self._name = self['project_control']['name']

        return self._name
