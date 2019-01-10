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
from reV.config.execution import BaseExecutionConfig, PeregrineConfig
from reV.utilities.exceptions import ConfigWarning, ConfigError


logger = logging.getLogger(__name__)


class GenConfig(BaseConfig):
    """Class to import and manage user configuration inputs."""

    def __init__(self, fname):
        """Initialize a config object."""

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
        """Get a list of the resource files with years filled in."""
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
        """Get the SAM generation configuration object."""
        if not hasattr(self, '_sam_gen'):
            self._sam_gen = SAMGenConfig(self['sam_generation'])
        return self._sam_gen

    @property
    def tech(self):
        """Get the tech property from the config."""
        if not hasattr(self, '_tech'):
            self._tech = self['project_control']['technology']
            self._tech = self._tech.lower().replace(' ', '')
        return self._tech

    @property
    def years(self):
        """Get the analysis years."""
        if not hasattr(self, '_years'):
            try:
                self._years = self['project_control']['analysis_years']
                if isinstance(self._years, list) is False:
                    self._years = [self._years]
            except KeyError as e:
                warn('Analysis years may not have been '
                     'specified, may default to year '
                     'specification in resource_file input. '
                     '\n\nKey Error: {}'.format(e), ConfigWarning)
        return self._years

    @property
    def dirout(self):
        """Get the output directory."""
        default = './gen_out'
        if not hasattr(self, '_dirout'):
            if 'output_directory' in self['directories']:
                self._dirout = self['directories']['output_directory']
            else:
                self._dirout = default
        return self._dirout

    @property
    def logdir(self):
        """Get the logging directory."""
        default = './logs'
        if not hasattr(self, '_logdir'):
            if 'logging_directory' in self['directories']:
                self._logdir = self['directories']['logging_directory']
            else:
                self._logdir = default
        return self._logdir

    @property
    def write_profiles(self):
        """Get the boolean arg whether to write the CF profiles."""
        default = False
        if not hasattr(self, '_profiles'):
            if 'write_profiles' in self['project_control']:
                self._profiles = self['project_control']['write_profiles']
            else:
                self._profiles = default
        return self._profiles

    @property
    def execution_control(self):
        """Get the execution control object."""
        if not hasattr(self, '_ec'):
            ec = self['execution_control']
            # static map of avail execution options with corresponding classes
            ec_config_types = {'local': BaseExecutionConfig,
                               'peregrine': PeregrineConfig}
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
        """Get the generation points control object."""
        if not hasattr(self, '_pc'):
            pp = ProjectPoints(self['project_points'], self['sam_generation'],
                               self.tech)
            if self.execution_control.option == 'peregrine':
                sites_per_split = ceil(len(pp) / self.execution_control.nodes)
            elif self.execution_control.option == 'local':
                sites_per_split = ceil(len(pp) / self.execution_control.ppn)
            self._pc = PointsControl(pp, sites_per_split=sites_per_split)
        return self._pc
