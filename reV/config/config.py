"""
reV Configuration
"""
from copy import deepcopy
from configobj import ConfigObj
import json
import logging
from math import ceil
import os
import pandas as pd

from reV import __dir__ as REVDIR
from reV import __testdata__ as TESTDATA


logger = logging.getLogger(__name__)


class BaseConfig(dict):
    """Base class for configuration frameworks."""

    @staticmethod
    def check_files(flist):
        """Make sure all files in the input file list exist."""
        for f in flist:
            if os.path.exists(f) is False:
                raise IOError('File does not exist: {}'.format(f))

    @staticmethod
    def load_ini(fname):
        """Load ini config into config class instance."""
        return ConfigObj(fname, unrepr=True)

    @staticmethod
    def load_json(fname):
        """Load json config into config class instance."""
        with open(fname, 'r') as f:
            # get config file
            config = json.load(f)
        return config

    @staticmethod
    def str_replace(d, strrep):
        """Perform a deep string replacement in d.

        Parameters
        ----------
        d : dict
            Config dictionary potentially containing strings to replace.
        strrep : dict
            Replacement mapping where keys are strings to search for and values
            are the new values.

        Returns
        -------
        d : dict
            Config dictionary with replaced strings.
        """

        if isinstance(d, dict):
            # go through dict keys and values
            for key, val in d.items():
                if isinstance(val, dict):
                    # if the value is also a dict, go one more level deeper
                    d[key] = BaseConfig.str_replace(val, strrep)
                elif isinstance(val, str):
                    # if val is a str, check to see if str replacements apply
                    for old_str, new in strrep.items():
                        # old_str is in the value, replace with new value
                        d[key] = val.replace(old_str, new)
                        val = val.replace(old_str, new)
        # return updated dictionary
        return d

    def set_self_dict(self, dictlike):
        """Save a dict-like variable as object instance dictionary items."""
        for key, val in dictlike.items():
            self.__setitem__(key, val)


class Config(BaseConfig):
    """Class to import and manage user configuration inputs."""

    # STRREP is a mapping of config strings to replace with variable values
    STRREP = {'REVDIR': REVDIR,
              'TESTDATA': TESTDATA}

    def __init__(self, fname):
        """Initialize a config object."""

        # Get file, Perform string replacement, save config to self instance
        config = self.get_file(fname)
        config = self.str_replace(config, self.STRREP)
        self.set_self_dict(config)
        self.check_conflicts()

    @property
    def execution_control(self):
        """Get the execution control property."""
        if not hasattr(self, '_execution_control'):

            pp_dict = deepcopy(self['project_points'][self.tech])
            sam_inputs_dict = deepcopy(self.sam_gen.inputs)

            _project_points = ProjectPoints(pp_dict, sam_inputs_dict,
                                            self.tech)

            self._execution_control = ExecutionControl(
                self['execution_control'], _project_points, self.years)

        return self._execution_control

    @property
    def project_points(self):
        """Get the project points attribute from execution control."""
        return self.execution_control.project_points

    @property
    def logging_level(self):
        """Get the user-specified logging level."""
        if not hasattr(self, '_logging_level'):
            levels = {'DEBUG': logging.DEBUG,
                      'INFO': logging.INFO,
                      'WARNING': logging.WARNING,
                      'ERROR': logging.ERROR,
                      'CRITICAL': logging.CRITICAL,
                      }
            x = self.__getitem__('project_control')['model_run_logging_level']
            self._logging_level = levels[x.upper()]
        return self._logging_level

    @property
    def name(self):
        """Get the project name."""
        default = 'rev2'
        if not hasattr(self, '_name'):
            if 'name' in self.__getitem__('project_control'):
                if self.__getitem__('project_control')['name']:
                    self._name = self.__getitem__('project_control')['name']
                else:
                    self._name = default
            else:
                self._name = default

        return self._name

    @property
    def res_files(self):
        """Get a list of the resource files with years filled in."""
        if not hasattr(self, '_res_files'):
            # get base filename, may have {} for year format
            fname = self['resource'][self.tech]['resource_file']
            if '{}' in fname:
                # need to make list of res files for each year
                self._res_files = [fname.format(year) for year in self.years]
            else:
                # only one resource file request, still put in list
                self._res_files = [fname]
        self.check_files(self._res_files)
        return self._res_files

    @property
    def sam_gen(self):
        """Get the SAM generation configuration object."""
        if not hasattr(self, '_sam_gen'):
            self._sam_gen = SAMGenConfig(self['sam_generation'], self.tech)
        return self._sam_gen

    @property
    def tech(self):
        """Get the tech property from the config."""
        if not hasattr(self, '_tech'):
            self._tech = self['project_control']['technologies']
            if isinstance(self._tech, list) and len(self._tech) == 1:
                self._tech = self._tech[0]
            if isinstance(self._tech, str):
                self._tech = self._tech.lower()
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
                logger.warning('Analysis years may not have been '
                               'specified, may default to year '
                               'specification in resource_file input. '
                               '\n\nKey Error: {}'.format(e))
        return self._years

    def check_conflicts(self):
        """Check to find conflicts in input specification"""
        if ('nodes' in self['execution_control'] and
                'sites_per_node' in self['execution_control'] and
                isinstance(self.project_points.sites, (list, tuple))):
            ec = self['execution_control']
            if ec['nodes'] is None and ec['sites_per_node'] is None:
                n_sites = ec['nodes'] * ec['sites_per_node']
                if n_sites < len(self.project_points.sites):
                    raise ValueError('Conflict in site specification and node '
                                     'allocation. There are {} sites to run, '
                                     'but only {} nodes and {} sites per node'
                                     .format(len(self.project_points.sites),
                                             ec['nodes'],
                                             ec['sites_per_node']))

    def get_file(self, fname):
        """Read the config file.

        Parameters
        ----------
        fname : str
            Full path + filename.

        Returns
        -------
        config : dict
            Config data.
        """
        logger.info('Getting "{}"'.format(fname))
        if os.path.exists(fname) and fname.endswith('.json'):
            config = self.load_json(fname)
        elif os.path.exists(fname) and fname.endswith('.ini'):
            config = self.load_ini(fname)
        elif os.path.exists(fname) is False:
            raise Exception('Configuration file does not exist: {}'
                            .format(fname))
        else:
            raise Exception('Unknown error getting configuration file: {}'
                            .format(fname))
        return config


class ExecutionControl:
    """Class to manage execution control parameters and split ProjectPoints."""
    def __init__(self, config_exec, project_points, years, level='supervisor'):
        """Initialize execution control object.

        Parameters
        ----------
        config_exec : dict
            execution_control section of the configuration input file.
        project_points : config.ProjectPoints
            ProjectPoints instance to be split between execution workers.
        years : list
            List of years to run. (interpreted as duplicates of ProjectPoints)
        level : str
            CURRENT execution level, will control how this instance of
            ExecutionControl is being split and iterated upon.
            Options: 'supervisor' or 'core'
        """

        self._years = years
        self._raw = config_exec
        self.level = level
        self._project_points = project_points

    def __iter__(self):
        """Iterator initialization dunder."""
        self._i = 0
        self._last_site_ind = 0
        logger.info('Starting ExecutionControl at level "{}" '
                    'and split level "{}" with iter limit: {}'
                    .format(self.level, self.split_level, self.N))
        return self

    def __next__(self):
        """Iterate through and return next site resource data.

        Returns
        -------
        new_exec : config.ExecutionControl
            Split instance of this class with a subset of project points based
            on the number of sites per node (or per core) depending on the
            execution control level.
        """

        if self._i < self.N:
            i0 = self._last_site_ind
            i1 = i0 + self.split_increment
            logger.info('ExecutionControl iterating from site index '
                        '{} to {} on worker #{}'
                        .format(i0, i1, self._i))
            self._i += 1
            self._last_site_ind = i1

            new_exec = ExecutionControl.split(self.raw, self.years, i0, i1,
                                              self.project_points.raw,
                                              self.project_points.sam_configs,
                                              self.project_points.df,
                                              self.project_points.tech,
                                              split_level=self.split_level)
            return new_exec
        else:
            # iter attribute equal or greater than iter limit
            raise StopIteration

    @property
    def split_level(self):
        """Get the level of the split of this object (one level down)."""

        if not hasattr(self, '_split_level'):
            split_levels = {'supervisor': 'core'}
            try:
                self._split_level = split_levels[self.level]
            except KeyError:
                raise KeyError('Current execution level cannot be split: {}'
                               .format(self.level))

        return self._split_level

    @property
    def split_increment(self):
        """Get the iterator increment property (number of sites per iter)."""
        return self.sites_per_core

    @property
    def N(self):
        """Get the iterator limit (number of splits)."""
        if not hasattr(self, '_N'):
            self._N = self.p_tot
        return self._N

    @property
    def nodes(self):
        """Get the number of nodes property."""
        if not hasattr(self, '_nodes'):
            if 'nodes' in self.raw:
                self._nodes = self.raw['nodes']
            else:
                self._nodes = 1
        return self._nodes

    @property
    def ppn(self):
        """Get the process per node (ppn) property."""
        if not hasattr(self, '_ppn'):
            if 'ppn' in self.raw:
                self._ppn = self.raw['ppn']
            else:
                self._ppn = 1
        return self._ppn

    @property
    def p_tot(self):
        """Get the total number of available processes."""
        return self.nodes * self.ppn

    @property
    def option(self):
        """Get the HPC vs. local vs. serial option."""
        default = 'serial'
        if not hasattr(self, '_option'):
            if 'option' in self.raw:
                if self.raw['option']:
                    self._option = self.raw['option'].lower()
                else:
                    # default option if not specified is serial
                    self._option = default
            else:
                # default option if not specified is serial
                self._option = default

        return self._option

    @property
    def hpc_queue(self):
        """Get the HPC queue property."""
        default = 'short'
        if not hasattr(self, '_hpc_queue'):
            if 'queue' in self.raw:
                if self.raw['queue']:
                    self._hpc_queue = self.raw['queue'].lower()
                else:
                    # default option if not specified is serial
                    self._hpc_queue = default
            else:
                # default option if not specified is serial
                self._hpc_queue = default

        return self._hpc_queue

    @property
    def hpc_alloc(self):
        """Get the HPC allocation property."""
        default = 'rev'
        if not hasattr(self, '_hpc_alloc'):
            if 'allocation' in self.raw:
                if self.raw['allocation']:
                    self._hpc_alloc = self.raw['allocation'].lower()
                else:
                    # default option if not specified is serial
                    self._hpc_alloc = default
            else:
                # default option if not specified is serial
                self._hpc_alloc = default

        return self._hpc_alloc

    @property
    def hpc_node_mem(self):
        """Get the HPC node memory property."""
        default = '32GB'
        if not hasattr(self, '_hpc_node_mem'):
            if 'memory' in self.raw:
                if self.raw['memory']:
                    self._hpc_node_mem = self.raw['memory']
                else:
                    # default option if not specified is serial
                    self._hpc_node_mem = default
            else:
                # default option if not specified is serial
                self._hpc_node_mem = default

        return self._hpc_node_mem

    @property
    def project_points(self):
        """Get the project points property"""
        return self._project_points

    @property
    def raw(self):
        """Get the raw configuration input dict before any mods/mutations."""
        return self._raw

    @property
    def sites_per_node(self):
        """Get the number of sites to be computed per node."""

        msg = ('User needs to input either project points slice "stop" value '
               'or "sites_per_node", otherwise "sites_per_node" cannot be '
               'computed.')

        if 'sites_per_node' in self.raw:
            if self.raw['sites_per_node']:
                self._sites_per_node = self.raw['sites_per_node']
            elif self.raw['sites_per_node'] is None and self.n_sites == 'inf':
                raise ValueError(msg)
            else:
                self._sites_per_node = ceil(self.n_sites / self.nodes)
        else:
            if self.n_sites == 'inf':
                raise ValueError(msg)
            else:
                self._sites_per_node = ceil(self.n_sites / self.nodes)
        return self._sites_per_node

    @property
    def sites_per_core(self):
        """Get the number of sites to be computed per core."""

        msg = ('User needs to input either project points slice "stop" value '
               'or "sites_per_core", otherwise "sites_per_core" cannot be '
               'computed.')

        if 'sites_per_core' in self.raw:
            if self.raw['sites_per_core']:
                self._sites_per_core = self.raw['sites_per_core']
            elif self.raw['sites_per_core'] is None and self.n_sites == 'inf':
                raise ValueError(msg)
            else:
                self._sites_per_core = ceil(self.n_sites / self.p_tot)
        else:
            if self.n_sites == 'inf':
                raise ValueError(msg)
            else:
                self._sites_per_core = ceil(self.n_sites / self.p_tot)
        return self._sites_per_core

    @property
    def n_sites(self):
        """Get the total number of sites."""
        if not hasattr(self, '_n_sites'):
            if isinstance(self.project_points.sites, slice):
                site_slice = self.project_points.sites
                if site_slice.stop is None:
                    self._n_sites = 'inf'
                else:
                    self._n_sites = (len(
                        list(range(*site_slice.indices(site_slice.stop)))) *
                        len(self.years))
            else:
                self._n_sites = (len(self.project_points.sites) *
                                 len(self.years))
        return self._n_sites

    @property
    def years(self):
        """Get the year list."""
        return self._years

    @classmethod
    def split(cls, config_exec, years, i0, i1, config_project_points,
              sam_configs, config_df, tech, split_level='core'):
        """Split this execution by splitting the project points attribute.

        Parameters
        ----------
        config_exec : dict
            The raw execution control configuration input dict before any
            mods/mutations.
        years : list
            List of years to execute
        i0/i1 : int
            Beginning/end (inclusive/exclusive, respetively) index split
            parameters for ProjectPoints.split.
        config_project_points : dict
            The raw project points configuration input dict before any
            mods/mutations.
        sam_configs : dict
            Multi-level dictionary containing multiple SAM input
            configurations.
        config_df : pd.DataFrame
            Sites to SAM configuration dictionary IDs mapping dataframe.
        tech : str
            Generation technology.
        split_level : str
            Level (core or node) of the split execution control instance.

        Returns
        -------
        sub : ExecutionControl
            New instance of execution control with a subset of the original
            project points.
        """

        new_points = ProjectPoints.split(i0, i1, config_project_points,
                                         sam_configs, config_df, tech)
        sub = cls(config_exec, new_points, years, level=split_level)
        return sub


class ProjectPoints(BaseConfig):
    """Class to manage site and SAM input configuration requests.

    Use Cases
    ---------
    config_id@site0, SAM_config_dict@site0 = ProjectPoints[0]
    site_list_or_slice = ProjectPoints.sites
    site_list_or_slice = ProjectPoints.get_sites_from_config(config_id)
    ProjectPoints_sub = ProjectPoints.split(0, 10)
    h_list_int_float = ProjectPoints.h
    """

    def __init__(self, config_pp, sam_configs, tech):
        """Init project points containing sites and corresponding SAM configs.

        Parameters
        ----------
        config_pp : dict
            Single-level dictionary containing the project points configuration
            inputs for the SAM tech to be used.
        sam_configs : dict
            Multi-level dictionary containing multiple SAM input
            configurations. The top level key is the SAM config ID, top level
            value is the SAM config. Each SAM config is a dictionary with keys
            equal to input names, values equal to the actual inputs.
        tech : str
            reV technology being executed.
        """

        self._raw = config_pp
        self._sam_configs = sam_configs
        self._tech = tech
        self.parse_project_points(config_pp)

    def __getitem__(self, site):
        """Get the SAM config ID and dictionary for the requested site.

        Parameters
        ----------
        site : int | str
            Site number of interest.

        Returns
        -------
        config_id : str
            Configuration ID (variable name) specified in the sam_generation
            config section.
        config : dict
            Actual SAM input values in a single level dictionary with variable
            names (keys) and values.
        """

        if not hasattr(self, 'default_config'):
            # was set w/ JSON w/ one config for each site, return mapped value
            site_bool = (self.df['sites'] == site)
            config_id = self.df.loc[site_bool, 'configs'].values[0]
            return config_id, self.sam_configs[config_id]
        else:
            # was set w/ slice w/ only default config
            return self.default_config_id, self.default_config

    @property
    def df(self):
        """Get the project points dataframe property.

        Returns
        -------
        self._df : pd.DataFrame
            Table of sites and SAM configuration IDs.
            Has columns 'sites' and 'configs'.
        """

        return self._df

    @df.setter
    def df(self, data):
        """Set the project points dataframe property

        Parameters
        ----------
        data : str | dict | pd.DataFrame
            Either a csv filename, dict with sites and configs keys, or full
            dataframe.
        """

        if isinstance(data, str):
            if data.endswith('.csv'):
                self._df = pd.read_csv(data)
            else:
                raise TypeError('Project points file must be csv but received:'
                                ' {}'.format(data))
        elif isinstance(data, dict):
            if 'sites' in data.keys() and 'configs' in data.keys():
                self._df = pd.DataFrame(data)
            else:
                raise KeyError('Project points data must contain sites and '
                               'configs column headers.')
        elif isinstance(data, pd.DataFrame):
            if ('sites' in data.columns.values and
                    'configs' in data.columns.values):
                self._df = data
            else:
                raise KeyError('Project points data must contain sites and '
                               'configs column headers.')
        else:
            raise TypeError('Project points data must be csv filename or '
                            'dictionary but received: {}'.format(type(data)))

    @property
    def h(self, h_var='wind_turbine_hub_ht'):
        """Get the hub heights corresponding to the site list or None for solar
        """
        if not hasattr(self, '_h') and 'wind' in self.tech:
            if not hasattr(self, 'default_config'):
                self._h = [self[site][1][h_var] for site in self.sites]
            else:
                self._h = self.default_config[h_var]
        elif not hasattr(self, '_h') and 'wind' not in self.tech:
            self._h = None

        return self._h

    @property
    def raw(self):
        """Get the raw configuration input dict before any mods/mutations."""
        return self._raw

    @property
    def sam_configs(self):
        """Get the SAM configs dictionary property.

        Returns
        -------
        _sam_configs : dict
            Multi-level dictionary containing multiple SAM input
            configurations. The top level key is the SAM config ID, top level
            value is the SAM config. Each SAM config is a dictionary with keys
            equal to input names, values equal to the actual inputs.
        """
        return self._sam_configs

    @property
    def sites(self):
        """Get sites property, type is list if possible
        (if slice stop is None, this will be a slice)."""
        return self._sites

    @property
    def sites_as_slice(self):
        """Get sites property, type is slice if possible
        (if sites is a non-sequential list, this will return a list)."""

        if not hasattr(self, '_sites_as_slice'):
            # try_slice is what the sites list would be if it is sequential
            try_slice = slice(self.sites[0], self.sites[-1] + 1)

            if self.sites == list(range(*try_slice.indices(try_slice.stop))):
                # try_slice is equivelant to the site list
                self._sites_as_slice = try_slice
            else:
                # cannot be converted to a sequential slice, return the list
                self._sites_as_slice = self.sites

        return self._sites_as_slice

    @property
    def tech(self):
        """Get the tech property from the config."""
        return self._tech

    def get_sites_from_config(self, config):
        """Get a site list that corresponds to a config key."""
        if not hasattr(self, 'default_config'):
            # was set w/ JSON w/ one config for each site, return mapped values
            sites = self.df.loc[(self.df['configs'] == config), 'sites'].values
            return list(sites)
        else:
            if config == self.default_config_id:
                # was set w/ slice w/ only default config, return all sites
                return self.sites
            else:
                return []

    def parse_project_points(self, config_pp):
        """Parse and set the project points using either a file or slice."""
        if 'file' in config_pp:
            if config_pp['file'] is not None:
                self.csv_project_points(config_pp['file'])
                if 'stop' in config_pp:
                    msg = ('More than one project points selection '
                           'method has been requested. Defaulting '
                           'to file input. Site selection '
                           'preference is file then slice')
                    logger.warning(msg)
        elif 'stop' in config_pp:
            self.slice_project_points(config_pp)

    def csv_project_points(self, fname):
        """Set the project points using the target csv."""
        if fname.endswith('.csv'):
            self.df = fname
            self._sites = list(self.df['sites'].values)
        else:
            raise ValueError('Config project points file must be '
                             '.csv, but received: {}'
                             .format(fname))

    def slice_project_points(self, config_pp):
        """Set the project points using slice parameters"""
        if config_pp['stop'] == 'inf':
            config_pp['stop'] = None
            logger.info('Site selection is set to run to the '
                        'last site in the resource file.')

        if config_pp['stop']:
            # try to always store the sites as a list
            site_slice = slice(config_pp['start'],
                               config_pp['stop'],
                               config_pp['step'])
            self._sites = list(range(*site_slice.indices(site_slice.stop)))
        else:
            # if stop is None, a list cannot be made, so store as a slice
            if config_pp['step'] == 1 or config_pp['step'] is None:
                self._sites = slice(config_pp['start'],
                                    config_pp['stop'],
                                    config_pp['step'])
            else:
                raise ValueError('If no Project Points slice stop is '
                                 'specified, step must equal 1 or None.')

        avail_configs = sorted(list(self.sam_configs.keys()))
        n_configs = len(avail_configs)
        self.default_config_id = avail_configs[0]
        if n_configs > 1:
            logger.warning('Multiple SAM input configurations detected '
                           'for a slice-based site project points. '
                           'Defaulting to: "{}"'.format(avail_configs[0]))
        if config_pp['stop'] is None:
            # No stop, set default config that will be returned for any site
            self.default_config = self.sam_configs[self.default_config_id]
        else:
            # if the end of the slice is known, you can make a list of the
            # indices and set the config map with the site-to-sam_config map
            site_config_dict = {'sites': self.sites,
                                'configs': [avail_configs[0]
                                            for s in self.sites]}
            self.df = site_config_dict

    @classmethod
    def split(cls, i0, i1, config_pp, sam_configs, config_df, tech):
        """Return split instance of this ProjectPoints w/ site subset.

        Parameters
        ----------
        i0 : int
            Starting INDEX (not site number) (inclusive) of the site property
            attribute to include in the split instance. This is not necessarily
            the same as the starting site number, for instance if ProjectPoints
            is sites 20:100, i0=0 i1=10 will result in sites 20:30.
        i1 : int
            Ending INDEX (not site number) (exclusive) of the site property
            attribute to include in the split instance. This is not necessarily
            the same as the final site number, for instance if ProjectPoints is
            sites 20:100, i0=0 i1=10 will result in sites 20:30.
        config_pp : dict
            Single-level dictionary containing the project points configuration
            inputs for the SAM tech to be used.
        sam_configs : dict
            Multi-level dictionary containing multiple SAM input
            configurations. The top level key is the SAM config ID, top level
            value is the SAM config. Each SAM config is a dictionary with keys
            equal to input names, values equal to the actual inputs.
        config_df : pd.DataFrame
            Sites to SAM configuration dictionary IDs mapping dataframe.
        tech : str
            reV technology being executed.

        Returns
        -------
        sub : ProjectPoints
            New instance of ProjectPoints with a subset of the following
            attributes: sites, project points df, and the self dictionary data
            struct.
        """

        # make a new instance of ProjectPoints
        sub = cls(config_pp, sam_configs, tech)

        if isinstance(sub.sites, (list, tuple)):
            # Reset the site attribute using a subset of the original
            sub._sites = sub.sites[i0:i1]

            # clear the dictionary attributes
            sub.clear()

            # set the new config map dataframe
            sub.df = config_df[config_df['sites'].isin(sub.sites)]

        elif isinstance(sub.sites, slice):
            # this is only the case if stop=None,
            # in which case default config is used.
            if (sub.sites.stop is None and sub.sites.step != 1 and
                    sub.sites.step is not None):
                raise ValueError('Cannot perform a project points split on a '
                                 'non-sequential slice with no stop. Project '
                                 'point site slice: {}'.format(sub.sites))
            else:
                sub._sites = list(range(sub.sites.start + i0,
                                        sub.sites.start + (i1 - i0)))

        return sub


class SAMGenConfig(BaseConfig):
    """Class to handle the SAM generation section of config input."""
    def __init__(self, SAM_config, tech):
        """Initialize the SAM generation section of config as an object.

        Parameters
        ----------
        SAM_config : dict
            Multi-level dictionary containing inputs from the "sam_generation"
            section of the config input file. Should contain keys equal to the
            specified technology.
        tech : str
            Generation technology specification. Should correspond to one of
            the keys in the SAM_config input.
        """

        self._tech = tech

        # Initialize the SAM generation config section as a dictionary.
        self.set_self_dict(SAM_config)

    @property
    def inputs(self):
        """Get the SAM input file(s) (JSON) and return as a dictionary.

        Parameters
        ----------
        _inputs : dict
            They keys of this dictionary are variable names in the config input
            file. They are basically "configuration ID's". The values are the
            imported json SAM input dictionaries.
        """

        if not hasattr(self, '_inputs'):
            self._inputs = {}
            for key, fname in self.__getitem__(self.tech).items():
                # key is ID (i.e. sam_param_0) that matches project points json
                # fname is the actual SAM config file name (with path)

                if fname.endswith('.json') is True:
                    if os.path.exists(fname):
                        with open(fname, 'r') as f:
                            # get unit test inputs
                            self._inputs[key] = json.load(f)
                    else:
                        raise IOError('SAM inputs file does not exist: {}'
                                      .format(fname))
                else:
                    raise IOError('SAM inputs file must be a JSON: {}'
                                  .format(fname))
        return self._inputs

    @property
    def tech(self):
        """Get the tech property from the config."""
        return self._tech

    @property
    def write_profiles(self):
        """Get the boolean write profiles option."""
        if not hasattr(self, '_write_profiles'):
            self._write_profiles = self.__getitem__('write_profiles')
        return self._write_profiles
