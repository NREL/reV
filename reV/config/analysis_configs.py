# -*- coding: utf-8 -*-
"""
reV analysis configs (generation, lcoe, etc...)

Created on Mon Jan 28 11:43:27 2019

@author: gbuster
"""
import logging
from math import ceil
import os
from warnings import warn

from reV import REVDIR, TESTDATADIR
from reV.config.base_config import BaseConfig
from reV.config.execution import (BaseExecutionConfig, PeregrineConfig,
                                  EagleConfig)
from reV.config.sam_config import SAMConfig
from reV.config.curtailment import Curtailment
from reV.config.project_points import PointsControl, ProjectPoints
from reV.utilities.exceptions import ConfigError, ConfigWarning


logger = logging.getLogger(__name__)


class AnalysisConfig(BaseConfig):
    """Base analysis config (generation, lcoe, etc...)."""

    def __init__(self, config_dict):
        self._years = None
        self._dirout = None
        self._logdir = None
        self._ec = None
        super().__init__(config_dict)

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


class SAMAnalysisConfig(AnalysisConfig):
    """SAM-based analysis config (generation, lcoe, etc...)."""

    def __init__(self, config_dict):
        self._tech = None
        self._sam_config = None
        self._pc = None
        self._output_request = None
        super().__init__(config_dict)

    @property
    def tech(self):
        """Get the tech property from the config.

        Returns
        -------
        _tech : str
            reV technology string to analyze (e.g. pv, csp, wind, etc...).
        """
        if self._tech is None:
            self._tech = self['project_control']['technology']
            self._tech = self._tech.lower().replace(' ', '')
        return self._tech

    @property
    def sam_config(self):
        """Get the SAM configuration object.

        Returns
        -------
        _sam_gen : reV.config.sam.SAMConfig
            SAM config object. This object emulates a dictionary.
        """
        if self._sam_config is None:
            self._sam_config = SAMConfig(self['sam_files'])
        return self._sam_config

    @property
    def points_control(self):
        """Get the generation points control object.

        Returns
        -------
        _pc : reV.config.project_points.PointsControl
            PointsControl object based on specified project points and
            execution control option.
        """

        if self._pc is None:
            # make an instance of project points
            pp = ProjectPoints(self['project_points'], self['sam_files'],
                               self.tech)

            if (self.execution_control.option == 'peregrine' or
                    self.execution_control.option == 'eagle'):
                # sites per split on peregrine or eagle is the number of sites
                # in project points / number of nodes. This is for the initial
                # division of the project sites between HPC nodes (jobs)
                sites_per_split = ceil(len(pp) / self.execution_control.nodes)

            elif self.execution_control.option == 'local':
                # sites per split on local is number of sites / # of processes
                sites_per_split = ceil(len(pp) / self.execution_control.ppn)

            # make an instance of points control and set to protected attribute
            self._pc = PointsControl(pp, sites_per_split=sites_per_split)

        return self._pc

    @property
    def output_request(self):
        """Get the list of requested output variables.

        Returns
        -------
        _output_request : list
            List of requested reV output variables corresponding to SAM
            variable names.
        """

        # map of commonly expected typos
        corrections = {'cf_means': 'cf_mean',
                       'cf': 'cf_mean',
                       'capacity_factor': 'cf_mean',
                       'capacityfactor': 'cf_mean',
                       'cf_profiles': 'cf_profile',
                       'profiles': 'cf_profile',
                       'profile': 'cf_profile',
                       'generation': 'annual_energy',
                       'yield': 'energy_yield',
                       'generation_profile': 'gen_profile',
                       'generation_profiles': 'gen_profile',
                       'plane_of_array': 'poa',
                       'plane_of_array_irradiance': 'poa',
                       'gen_profiles': 'gen_profile',
                       'lcoe': 'lcoe_fcr',
                       'lcoe_nominal': 'lcoe_nom',
                       'real_lcoe': 'lcoe_real',
                       'net_present_value': 'npv',
                       'ppa': 'ppa_price',
                       'single_owner': 'ppa_price',
                       'singleowner': 'ppa_price',
                       }

        if self._output_request is None:
            self._output_request = []
            # default output request if not specified
            temp = ['cf_mean']
            if 'output_request' in self['project_control']:
                temp = self['project_control']['output_request']

            if isinstance(temp, str):
                temp = [temp]

            for request in temp:
                if request in corrections.values():
                    self._output_request.append(request)
                elif request in corrections.keys():
                    self._output_request.append(corrections[request])
                    warn('Correcting output request "{}" to "{}".'
                         .format(request, corrections[request]), ConfigWarning)
                else:
                    self._output_request.append(request)
                    warn('Did not recognize requested output variable "{}". '
                         'Passing forward, but this may cause a downstream '
                         'error. Available known output variables are: {}'
                         .format(request, list(set(corrections.values()))),
                         ConfigWarning)

        return self._output_request


class GenConfig(SAMAnalysisConfig):
    """Class to import and manage user configuration inputs."""

    def __init__(self, fname):
        """Initialize a config object.

        Parameters
        ----------
        fname : str
            Generation config name (with path).
        """
        self._curtailment = None
        self._res_files = None
        # get the directory of the config file
        self.dir = os.path.dirname(os.path.realpath(fname)) + '/'

        # str_rep is a mapping of config strings to replace with real values
        self.str_rep = {'REVDIR': REVDIR,
                        'TESTDATADIR': TESTDATADIR,
                        './': self.dir,
                        }

        # Get file, Perform string replacement, save config to self instance
        config = self.str_replace(self.get_file(fname), self.str_rep)
        super().__init__(config)

    @property
    def curtailment(self):
        """Get the curtailment config object that the gen config points to.

        Returns
        -------
        _curtailment : NoneType | reV.config.curtailment.Curtailment
            Returns None if no curtailment config is specified. If one is
            specified, this returns the reV curtailment config object.
        """
        if self._curtailment is None:
            if 'curtailment' in self:
                if self['curtailment']:
                    # curtailment was specified and is not None or False
                    self._curtailment = Curtailment(self['curtailment'])

        return self._curtailment

    @property
    def downscale(self):
        """Get the resource downscale request (nsrdb only!).

        Returns
        -------
        _downscale : NoneType | str
            Returns None if no downscaling is requested. Otherwise, expects a
            downscale variable in the project_control section in the Pandas
            frequency format, e.g. '5min'.
        """

        if not hasattr(self, '_downscale'):
            self._downscale = None
            if 'downscale' in self['project_control']:
                if self['project_control']['downscale']:
                    # downscaling was requested and is not None or False
                    self._downscale = str(self['project_control']['downscale'])
        return self._downscale

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
        if self._res_files is None:
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


class EconConfig(SAMAnalysisConfig):
    """Class to import and manage configuration inputs for econ analysis."""

    def __init__(self, fname):
        """Initialize a config object.

        Parameters
        ----------
        fname : str
            Econ config name (with path).
        """
        self._cf_files = None
        self._site_data = None
        # get the directory of the config file
        self.dir = os.path.dirname(os.path.realpath(fname)) + '/'

        # str_rep is a mapping of config strings to replace with real values
        self.str_rep = {'REVDIR': REVDIR,
                        'TESTDATADIR': TESTDATADIR,
                        './': self.dir,
                        }

        # Get file, Perform string replacement, save config to self instance
        config = self.str_replace(self.get_file(fname), self.str_rep)
        super().__init__(config)

    @property
    def cf_files(self):
        """Get the capacity factor files (reV generation output data).

        Returns
        -------
        _cf_files : list
            Target paths for capacity factor files (reV generation output
            data) for input to reV LCOE calculation.
        """

        if self._cf_files is None:
            # get base filename, may have {} for year format
            fname = self['cf_file']
            if '{}' in fname:
                # need to make list of res files for each year
                self._cf_files = [fname.format(year) for year in self.years]
            else:
                # only one resource file request, still put in list
                self._cf_files = [fname]
        self.check_files(self._cf_files)
        if len(self._cf_files) != len(self.years):
            raise ConfigError('The number of cf files does not match '
                              'the number of analysis years!'
                              '\n\tCF files: \n\t\t{}'
                              '\n\tYears: \n\t\t{}'
                              .format(self._cf_files, self.years))

        return self._cf_files

    @property
    def site_data(self):
        """Get the site-specific data file.

        Returns
        -------
        _site_data : str | NoneType
            Target path for site-specific data file.
        """
        if self._site_data is None:
            self._site_data = None
            if 'site_data' in self:
                self._site_data = self['site_data']
        return self._site_data
