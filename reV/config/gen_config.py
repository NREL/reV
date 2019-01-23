"""
reV Configuration
"""
import logging
from math import ceil
import os
from warnings import warn

from reV import __dir__ as REVDIR
from reV import __testdatadir__ as TESTDATADIR
from reV.config.sam import SAMGenConfig
from reV.config.base_config import BaseConfig
from reV.config.project_points import ProjectPoints, PointsControl
from reV.config.execution import (BaseExecutionConfig, PeregrineConfig,
                                  EagleConfig)
from reV.utilities.exceptions import ConfigWarning, ConfigError


logger = logging.getLogger(__name__)


class GenConfig(BaseConfig):
    """Class to import and manage user configuration inputs."""

    def __init__(self, fname):
        """Initialize a config object.

        Parameters
        ----------
        fname : str
            Generation config name (with path).
        """

        # get the directory of the config file
        self.dir = os.path.dirname(os.path.realpath(fname)) + '/'

        # str_rep is a mapping of config strings to replace with real values
        self.str_rep = {'REVDIR': REVDIR,
                        'TESTDATADIR': TESTDATADIR,
                        './': self.dir,
                        }

        # Get file, Perform string replacement, save config to self instance
        config = self.str_replace(self.get_file(fname), self.str_rep)
        self.set_self_dict(config)

    @property
    def res_files(self):
        """Get a list of the resource files with years filled in.

        Returns
        -------
        _res_files : list
            List of config-specified resource files. Resource files with {}
            formatting will be filled with the specified year(s). This return
            value is a list with len=1 for a single year run.
        """
        if not hasattr(self, '_res_files'):
            # get base filename, may have {} for year format
            fname = self['resource_file']
            if '{}' in fname:
                # need to make list of res files for each year
                self._res_files = [fname.format(year) for year in self.years]
            else:
                # only one resource file request, still put in list
                self._res_files = [fname]
        self.check_files(self._res_files)
        if len(self._res_files) != len(self.years):
            raise ConfigError('The number of resource files does not match '
                              'the number of analysis years!'
                              '\n\tResource files: \n\t\t{}'
                              '\n\tYears: \n\t\t{}'
                              .format(self._res_files, self.years))
        return self._res_files

    @property
    def sam_gen(self):
        """Get the SAM generation configuration object.

        Returns
        -------
        _sam_gen : reV.config.sam.SAMGenConfig
            SAM generation config object. This object emulates a dictionary.
        """
        if not hasattr(self, '_sam_gen'):
            self._sam_gen = SAMGenConfig(self['sam_generation'])
        return self._sam_gen

    @property
    def tech(self):
        """Get the tech property from the config.

        Returns
        -------
        _tech : str
            reV generation technology string to analyze. This string should
            match the available technologies in the following dictionary:
            reV.generation.generation.Gen.REVTECHS
        """
        if not hasattr(self, '_tech'):
            self._tech = self['project_control']['technology']
            self._tech = self._tech.lower().replace(' ', '')
        return self._tech

    @property
    def years(self):
        """Get the analysis years.

        Returns
        -------
        _years : list
            List of years to analyze. If this is a single year run, this return
            value is a single entry list. If no analysis_years are specified,
            the code will look anticipate a year in the resource file spec.
        """
        if not hasattr(self, '_years'):
            self._years = [None]
            if 'analysis_years' in self['project_control']:
                self._years = self['project_control']['analysis_years']
                if isinstance(self._years, list) is False:
                    self._years = [self._years]
            else:
                warn('Analysis years may not have been '
                     'specified, may default to year '
                     'specification in resource_file input.',
                     ConfigWarning)
        return self._years

    @property
    def dirout(self):
        """Get the output directory.

        Returns
        -------
        _dirout : str
            Target path for reV generation output files.
        """
        default = './gen_out'
        if not hasattr(self, '_dirout'):
            self._dirout = default
            if 'output_directory' in self['directories']:
                self._dirout = self['directories']['output_directory']
        return self._dirout

    @property
    def logdir(self):
        """Get the logging directory.

        Returns
        -------
        _logdir : str
            Target path for reV generation log files.
        """
        default = './logs'
        if not hasattr(self, '_logdir'):
            self._logdir = default
            if 'logging_directory' in self['directories']:
                self._logdir = self['directories']['logging_directory']
        return self._logdir

    @property
    def write_profiles(self):
        """Get the boolean arg whether to write the CF profiles.

        Returns
        -------
        _profiles : bool
            Boolean flag on whether to write capacity factor profiles to disk.
        """
        default = False
        if not hasattr(self, '_profiles'):
            self._profiles = default
            if 'write_profiles' in self['project_control']:
                self._profiles = self['project_control']['write_profiles']
        return self._profiles

    @property
    def lcoe(self):
        """Get the boolean arg whether to calculate LCOE as part of gen run.

        Returns
        -------
        _lcoe : bool
            Boolean flag on whether to calc LCOE as part of the generation run.
        """
        default = False
        if not hasattr(self, '_lcoe'):
            self._lcoe = default
            if 'lcoe' in self['project_control']:
                self._lcoe = self['project_control']['lcoe']
        return self._lcoe

    @property
    def execution_control(self):
        """Get the execution control object.

        Returns
        -------
        _ec : BaseExecutionConfig | PeregrineConfig | EagleConfig
            reV execution config object specific to the execution_control
            option.
        """
        if not hasattr(self, '_ec'):
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
    def points_control(self):
        """Get the generation points control object.

        Returns
        -------
        _pc : reV.config.project_points.PointsControl
            PointsControl object based on specified project points and
            execution control option.
        """
        if not hasattr(self, '_pc'):
            pp = ProjectPoints(self['project_points'], self['sam_generation'],
                               self.tech)
            if (self.execution_control.option == 'peregrine' or
                    self.execution_control.option == 'eagle'):
                sites_per_split = ceil(len(pp) / self.execution_control.nodes)
            elif self.execution_control.option == 'local':
                sites_per_split = ceil(len(pp) / self.execution_control.ppn)
            self._pc = PointsControl(pp, sites_per_split=sites_per_split)
        return self._pc
