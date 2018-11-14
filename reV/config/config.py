"""
reV Configuration
"""
from copy import deepcopy
from configobj import ConfigObj
import json
import logging
import os
import pandas as pd

from reV import __dir__ as REVDIR
from reV import __testdata__ as TESTDATA


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
        log_name = '{}.{}'.format(self.__module__, self.__class__.__name__)
        self._logger = logging.getLogger(log_name)

        # Get file, Perform string replacement, save config to self instance
        config = self.get_file(fname)
        config = self.str_replace(config, self.STRREP)
        self.set_self_dict(config)

        # initialize protected property attributes as None
        self._tech = None
        self._res_files = None
        self._SAM_gen = None
        self._project_points = None
        self._years = None
        self._logging_level = None
        self._execution_control = None

    @property
    def execution_control(self):
        """Get the execution control property."""
        if self._execution_control is None:
            self._execution_control = ExecutionControl(
                self.__getitem__('execution_control'), self.project_points)
        return self._execution_control

    @property
    def logging_level(self):
        """Get the user-specified logging level."""
        if self._logging_level is None:
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
    def project_points(self):
        """Set project points property with instance of ProjectPoints."""
        if self._project_points is None:
            pp_dict = deepcopy(self['project_points'][self.tech])
            sam_inputs_dict = deepcopy(self.SAM_gen.inputs)

            self._project_points = ProjectPoints(pp_dict,
                                                 sam_inputs_dict,
                                                 self.tech)
        return self._project_points

    @property
    def res_files(self):
        """Get a list of the resource files with years filled in."""
        if self._res_files is None:
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
    def SAM_gen(self):
        """Get the SAM generation configuration object."""
        if self._SAM_gen is None:
            self._SAM_gen = SAM_Gen(self['sam_generation'], self.tech)
        return self._SAM_gen

    @property
    def tech(self):
        """Get the tech property from the config."""
        if self._tech is None:
            self._tech = self['project_control']['technologies']
            if isinstance(self._tech, list) and len(self._tech) == 1:
                self._tech = self._tech[0]
            if isinstance(self._tech, str):
                self._tech = self._tech.lower()
        return self._tech

    @property
    def years(self):
        """Get the analysis years."""
        if self._years is None:
            try:
                self._years = self['project_control']['analysis_years']
                if isinstance(self._years, list) is False:
                    self._years = [self._years]
            except KeyError as e:
                self._logger.warning('Analysis years may not have been '
                                     'specified, may default to year '
                                     'specification in resource_file input. '
                                     '\n\nKey Error: {}'.format(e))
        return self._years

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
        self._logger.info('Getting "{}"'.format(fname))
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


class ExecutionControl(BaseConfig):
    """Class to manage execution control parameters and split ProjectPoints."""
    def __init__(self, config_exec, project_points):
        """Initialize execution control object.

        Parameters
        ----------
        config_exec : dict
            execution_control section of the configuration input file.
        project_points : config.ProjectPoints
            ProjectPoints instance to be split between execution workers.
        """
        self._nodes = None
        self._ppn = None
        self._project_points = project_points
        self.set_self_dict(config_exec)

    @property
    def nodes(self):
        """Get the nodes property."""
        if self._nodes is None:
            if 'nodes' in self:
                self._nodes = self.__getitem__('nodes')
            else:
                self._nodes = 1
        return self._nodes

    @property
    def ppn(self):
        """Get the process per node (ppn) property."""
        if self._ppn is None:
            if 'ppn' in self:
                self._ppn = self.__getitem__('ppn')
            else:
                self._ppn = 1
        return self._ppn

    @property
    def p_tot(self):
        """Get the total number of available processes."""
        return self.nodes * self.ppn

    @property
    def project_points(self):
        """Get the project points property"""
        return self._project_points

    @property
    def sites_per_node(self):
        """Get the number of sites to be computed per node."""
        if 'sites_per_node' in self:
            self._sites_per_node = self.__getitem__('sites_per_node')
        else:
            if isinstance(self.project_points.sites, slice):
                site_slice = self.project_points.sites
                if site_slice.stop is None:
                    raise ValueError('User needs to input either project '
                                     'points slice "stop" value or '
                                     '"sites_per_node", otherwise '
                                     '"sites_per_node" cannot be computed.')
                else:
                    self._sites_per_node = (len(
                        list(range(*site_slice.indices(site_slice.stop)))) /
                        self.nodes)
            else:
                self._sites_per_node = (len(self.project_points.sites) /
                                        self.nodes)
        return self._sites_per_node

    @property
    def sites_per_core(self):
        """Get the number of sites to be computed per core."""
        if 'sites_per_core' in self:
            self._sites_per_core = self.__getitem__('sites_per_core')
        else:
            if isinstance(self.project_points.sites, slice):
                site_slice = self.project_points.sites
                if site_slice.stop is None:
                    raise ValueError('User needs to input either project '
                                     'points slice "stop" value or '
                                     '"sites_per_core", otherwise '
                                     '"sites_per_core" cannot be computed.')
                else:
                    self._sites_per_core = (len(
                        list(range(*site_slice.indices(site_slice.stop)))) /
                        self.p_tot)
            else:
                self._sites_per_core = (len(self.project_points.sites) /
                                        self.p_tot)
        return self._sites_per_core


class ProjectPoints(BaseConfig):
    """Class to manage site and SAM input configuration requests."""
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

        self.add_logger()

        self._raw = config_pp
        self._sam_configs = sam_configs
        self._tech = tech
        self.default_config = None
        self.parse_project_points(config_pp)

    def __getitem__(self, site):
        """Get the SAM config dictionary for the requested site."""
        if self.default_config is None:
            # was set w/ JSON w/ one config for each site, return mapped value
            config_id = self.config_map[site]
            return config_id, self.sam_configs[config_id]
        else:
            # was set w/ slice w/ only default config
            return self.default_config_id, self.default_config

    @property
    def raw(self):
        """Get the raw configuration input dict before any mods/mutations."""
        return self._raw

    @property
    def sam_configs(self):
        """Multi-level dictionary containing multiple SAM input
        configurations. The top level key is the SAM config ID, top level
        value is the SAM config. Each SAM config is a dictionary with keys
        equal to input names, values equal to the actual inputs.
        """
        return self._sam_configs

    @property
    def sites(self):
        """Get sites list property."""
        return self._sites

    @property
    def tech(self):
        """Get the tech property from the config."""
        return self._tech

    def get_sites_from_config(self, config):
        """Get a site list that corresponds to a config key."""
        if self.default_config is None:
            # was set w/ JSON w/ one config for each site, return mapped values
            sites = self.df.loc[(self.df['configs'] == config), 'sites'].values
            return list(sites)
        else:
            # was set w/ slice w/ only default config, return all sites
            return self.sites

    def json_project_points(self, fname):
        """Set the project points using the target json."""
        if fname.endswith('.json'):
            js = self.load_json(fname)
            self._sites = js['sites']
            self.df = pd.DataFrame(js)
            self.set_config_map(js)

        else:
            raise ValueError('Config site selection file must be '
                             '.json, but received: {}'
                             .format(fname))

    def parse_project_points(self, config_pp):
        """Parse and set the project points using either a file or slice."""
        if 'file' in config_pp:
            if config_pp['file'] is not None:
                self.json_project_points(config_pp['file'])
                if 'stop' in config_pp:
                    msg = ('More than one project points selection '
                           'method has been requested. Defaulting '
                           'to json input. Site selection '
                           'preference is json then slice')
                    self._logger.warning(msg)
        elif 'stop' in config_pp:
            self.slice_project_points(config_pp)

    def set_config_map(self, site_config_json):
        """Save a dict-like variable to the self dictionary and config map."""
        self.config_map = {}
        for i, site in enumerate(site_config_json['sites']):
            self.config_map[site] = site_config_json['configs'][i]
            self.__setitem__(site, site_config_json['configs'][i])

    def slice_project_points(self, config_pp):
        """Set the project points using slice parameters"""
        if config_pp['stop'] == 'inf':
            config_pp['stop'] = None
            self._logger.info('Site selection is set to run to the '
                              'last site in the resource file.')

        if config_pp['stop']:
            # try to always store the sites as a list
            self._sites = list(range(config_pp['start'],
                                     config_pp['stop'],
                                     config_pp['step']))
        else:
            # if stop is None, a list cannot be made, so store as a slice
            if config_pp['step'] == 1:
                self._sites = slice(config_pp['start'],
                                    config_pp['stop'],
                                    config_pp['step'])
            else:
                raise ValueError('If no Project Points slice stop is '
                                 'specified, step must equal 1.')

        avail_configs = sorted(list(self.sam_configs.keys()))
        n_configs = len(avail_configs)
        self.default_config_id = avail_configs[0]
        if n_configs > 1:
            self._logger.warning('Multiple SAM input configurations detected '
                                 'for a slice-based site project points. '
                                 'Defaulting to: {}'.format(avail_configs[0]))
        if config_pp['stop'] is None:
            # No stop, set default config that will be returned for any site
            self.default_config = self.sam_configs[self.default_config_id]
        else:
            # if the end of the slice is known, you can make a list of the
            # indices and set the config map with the site-to-sam_config map
            site_config_dict = {'sites': self.sites,
                                'configs': [avail_configs[0]
                                            for s in self.sites]}
            self.set_config_map(site_config_dict)

    def add_logger(self):
        """Add a logger attribute."""
        log_name = '{}.{}'.format(self.__module__, self.__class__.__name__)
        self._logger = logging.getLogger(log_name)

    def del_logger(self):
        """Del the logger attr so a ProjectPoints instance can be deepcopied"""
        del self._logger

    def split(self, i0, i1):
        """Return a deepcopy split subset instance of this ProjectPoints.

        Parameters
        ----------
        i0 : int
            Starting index (inclusive) of the site property attribute to
            include in the split instance.
        i1 : int
            Ending index (exclusive) of the site property attribute to
            include in the split instance.

        Returns
        -------
        sub : config.ProjectPoints
            Deepcopy instance of self (ProjectPoints instance) with a subset
            of the following attributes: sites, config_map, and the self
            dictionary data struct.
        """

        # make a deep-copied instance of the ProjectPoints instance
        self.del_logger()
        sub = deepcopy(self)
        self.add_logger()
        sub.add_logger()

        if isinstance(sub.sites, (list, tuple)):
            # Reset the site attribute using a subset of the original
            sub._sites = sub.sites[i0:i1]

            # clear the dictionary attributes
            sub.config_map = {}
            sub.clear()

            # re-build the dictionary attributes for items in the new site list
            for key, val in self.config_map.items():
                if key in sub.sites:
                    sub[key] = val
                    sub.config_map[key] = val

        elif isinstance(sub.sites, slice):
            if sub.sites.stop is None and sub.sites.step != 1:
                raise ValueError('Cannot perform a project points split on a '
                                 'non-sequential slice with no stop. Project '
                                 'point site slice: {}'.format(sub.sites))
            else:
                sub._sites = list(range(sub.sites.start,
                                        sub.sites.start + (i1 - i0)))

        return sub


class SAM_Gen(BaseConfig):
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

        log_name = '{}.{}'.format(self.__module__, self.__class__.__name__)
        self._logger = logging.getLogger(log_name)

        self._write_profiles = None
        self._inputs = None
        self._tech = tech

        # Initialize the SAM generation config section as a dictionary.
        self.set_self_dict(SAM_config)

    @property
    def inputs(self):
        """Get the SAM input file(s) (JSON) and return as a dictionary.

        They keys of this dictionary are variable names in the config input
        file. They are basically "configuration ID's". The values are the
        imported json SAM input dictionaries.
        """

        if self._inputs is None:
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
        if self._write_profiles is None:
            self._write_profiles = self.__getitem__('write_profiles')
        return self._write_profiles
