# -*- coding: utf-8 -*-
"""
reV Base analysis Configuration Frameworks
"""
import os
import logging
from warnings import warn

from reV.config.base_config import BaseConfig
from reV.config.execution import (BaseExecutionConfig, SlurmConfig)
from reV.utilities.exceptions import (ConfigError, ConfigWarning,
                                      reVDeprecationWarning)
from reV.utilities import ModuleName

logger = logging.getLogger(__name__)


class AnalysisConfig(BaseConfig):
    """Base analysis config (generation, lcoe, etc...)."""

    NAME = None

    def __init__(self, config, run_preflight=True, check_keys=True):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        run_preflight : bool, optional
            Flag to run or disable preflight checks, by default True
        check_keys : bool, optional
            Flag to check config keys against Class properties, by default True
        """
        super().__init__(config, check_keys=check_keys)

        self._analysis_years = None
        self._ec = None
        self.dirout = self.config_dir
        self.__config_fn = config

        self._preflight()

        if run_preflight:
            self._analysis_config_preflight()

    def _analysis_config_preflight(self):
        """Check for required config blocks"""

        if 'execution_control' not in self:
            e = 'reV config must have "execution_control" block!'
            logger.error(e)
            raise ConfigError(e)

    @property
    def analysis_years(self):
        """Get the analysis years.

        Returns
        -------
        analysis_years : list
            List of years to analyze. If this is a single year run, this return
            value is a single entry list. If no analysis_years are specified,
            the code will look anticipate a year in the input files.
        """

        if self._analysis_years is None:
            self._analysis_years = self.get('analysis_years', [None])
            if not isinstance(self._analysis_years, list):
                self._analysis_years = [self._analysis_years]

            if self._analysis_years[0] is None:
                warn('Years may not have been specified, may default '
                     'to available years in inputs files.', ConfigWarning)

        return self._analysis_years

    @property
    def log_directory(self):
        """Get the logging directory, look for key "log_directory" in the
        config.
        Returns
        -------
        log_directory : str
            Target path for reV log files.
        """
        return self.get('log_directory', './logs/')

    @property
    def execution_control(self):
        """Get the execution control object.

        Returns
        -------
        _ec : BaseExecutionConfig | EagleConfig
            reV execution config object specific to the execution_control
            option.
        """
        if self._ec is None:
            ec = self['execution_control']
            # static map of avail execution options with corresponding classes
            ec_config_types = {'local': BaseExecutionConfig,
                               'slurm': SlurmConfig,
                               'eagle': SlurmConfig,
                               'kestrel': SlurmConfig,
                               }
            if 'option' in ec:
                try:
                    # Try setting the attribute to the appropriate exec option
                    self._ec = ec_config_types[ec['option'].lower()](ec)
                except KeyError as exc:
                    # Option not found
                    msg = ('Execution control option not '
                           'recognized: "{}". '
                           'Available options are: {}.'
                           .format(ec['option'].lower(),
                                   list(ec_config_types.keys())))
                    raise ConfigError(msg) from exc
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
            if self.NAME == ModuleName.COLLECT:
                self._name = self.NAME

            # Analysis job name tag (helps ensure unique job name)
            elif self.NAME is not None:
                self._name += '_{}'.format(self.NAME)

            # Throw warning if user still has 'name' key in config
            if self.get('name') is not None:
                msg = ("Specifying a job name using config key 'name' is "
                       "deprecated. Job names are now inferred from the run "
                       "directory name. To silence this warning, remove "
                       "the 'name' key from the following config file: {!r}'"
                       .format(self.__config_fn))
                logger.warning(msg)
                warn(reVDeprecationWarning(msg))

        return self._name
