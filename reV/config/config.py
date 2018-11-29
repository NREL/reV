"""
reV Configuration
"""
from configobj import ConfigObj
import json
import logging
import os
import pandas as pd
from warnings import warn

from reV import __dir__ as REVDIR
from reV import __testdatadir__ as TESTDATA
from reV.exceptions import ConfigWarning
from reV.handlers.resource import Resource


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
        config = self.str_replace(self.get_file(fname), self.STRREP)
        self.set_self_dict(config)

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
                warn('Analysis years may not have been '
                     'specified, may default to year '
                     'specification in resource_file input. '
                     '\n\nKey Error: {}'.format(e), ConfigWarning)
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

        logger.debug('Getting "{}"'.format(fname))
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


class PointsControl:
    """Class to manage and split ProjectPoints."""
    def __init__(self, project_points, level='supervisor',
                 sites_per_split=100):
        """Initialize points control object.

        Parameters
        ----------
        project_points : reV.config.ProjectPoints
            ProjectPoints instance to be split between execution workers.
        level : str
            CURRENT execution level, will control how this instance of
            PointsControl is being split and iterated upon.
            Options: 'supervisor' or 'core'
        sites_per_split : int
            Sites per project points split instance returned in the __next__
            iterator function.
        """

        self._project_points = project_points
        self._sites_per_split = sites_per_split
        self.level = level

    def __iter__(self):
        """Iterator initialization dunder."""
        self._i = 0
        self._last_site_ind = 0
        return self

    def __next__(self):
        """Iterate through and return next site resource data.

        Returns
        -------
        new_exec : config.PointsControl
            Split instance of this class with a subset of project points based
            on the number of sites per split.
        """

        i0 = self._last_site_ind
        i1 = i0 + self.sites_per_split
        logger.debug('PointsControl iterating from site index '
                     '{} to {} on worker #{}'
                     .format(i0, i1, self._i))
        self._i += 1
        self._last_site_ind = i1

        new_exec = PointsControl.split(i0, i1, self.project_points,
                                       sites_per_split=self.sites_per_split)
        if not new_exec.project_points.sites:
            # no more sites left to analyze, reached end of iter.
            raise StopIteration
        return new_exec

    def __repr__(self):
        msg = "{} for sites: {}".format(self.__class__.__name__, self.sites)
        return msg

    @property
    def sites_per_split(self):
        """Get the iterator increment: number of sites per split."""
        return self._sites_per_split

    @property
    def project_points(self):
        """Get the project points property"""
        return self._project_points

    @property
    def sites(self):
        """Get the project points sites for this instance."""
        return self._project_points.sites

    @classmethod
    def split(cls, i0, i1, project_points, sites_per_split=100):
        """Split this execution by splitting the project points attribute.

        Parameters
        ----------
        i0/i1 : int
            Beginning/end (inclusive/exclusive, respectively) index split
            parameters for ProjectPoints.split.
        project_points : reV.config.ProjectPoints
            Project points instance that will be split.
        sites_per_split : int
            Sites per project points split instance returned in the __next__
            iterator function.

        Returns
        -------
        sub : PointsControl
            New instance of PointsControl with a subset of the original
            project points.
        """

        new_points = ProjectPoints.split(i0, i1, project_points.points,
                                         project_points.sam_files,
                                         project_points.df,
                                         project_points.tech,
                                         project_points.res_file)
        sub = cls(new_points, sites_per_split=sites_per_split)
        return sub


class ProjectPoints:
    """Class to manage site and SAM input configuration requests.

    Use Cases
    ---------
    config_id@site0, SAM_config_dict@site0 = ProjectPoints[0]
    site_list_or_slice = ProjectPoints.sites
    site_list_or_slice = ProjectPoints.get_sites_from_config(config_id)
    ProjectPoints_sub = ProjectPoints.split(0, 10, ...)
    h_list = ProjectPoints.h
    """

    def __init__(self, points, sam_files, tech, res_file=None):
        """Init project points containing sites and corresponding SAM configs.

        Parameters
        ----------
        points : slice | str
            Slice specifying project points or string pointing to a project
            points csv.
        sam_files : dict | str | list
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s), top level value is the SAM path. Can also be a single
            config file str. If it's a list, it is mapped to the sorted list
            of unique configs requested by points csv.
        tech : str
            reV technology being executed.
        res_file : str
            Optional resource file to find maximum length of project points if
            points slice stop is None.
        """

        # set protected attributes
        self._points = points
        self._res_file = res_file
        self._tech = tech

        # if sam files is a dict or string, set first.
        if isinstance(sam_files, (dict, str)):
            self.sam_files = sam_files

        # create the project points from the raw configuration dict
        self.parse_project_points(points)

        # If the sam files is a list, set last
        if isinstance(sam_files, list):
            self.sam_files = sam_files

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

        site_bool = (self.df['sites'] == site)
        try:
            config_id = self.df.loc[site_bool, 'configs'].values[0]
        except KeyError:
            raise KeyError('Site {} not found in this instance of '
                           'ProjectPoints. Available sites include: {}'
                           .format(site, self.sites))
        return config_id, self.sam_configs[config_id]

    def __repr__(self):
        msg = "{} for sites: {}".format(self.__class__.__name__, self.sites)
        return msg

    @property
    def df(self):
        """Get the project points dataframe property.

        Returns
        -------
        self._df : pd.DataFrame
            Table of sites and corresponding SAM configuration IDs.
            Has columns 'sites' and 'configs'.
        """

        return self._df

    @df.setter
    def df(self, data):
        """Set the project points dataframe property.

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
            # wind technology, get a list of h values
            self._h = [self[site][1][h_var] for site in self.sites]

        elif not hasattr(self, '_h') and 'wind' not in self.tech:
            # not a wind tech, return None
            self._h = None

        return self._h

    @property
    def points(self):
        """Get the original points input."""
        return self._points

    @property
    def sam_files(self):
        """Get the SAM files dictionary property.

        Returns
        -------
        _sam_files: dict
            Multi-level dictionary containing multiple SAM input config files.
            The top level key is the SAM config ID, top level value is the SAM
            config file path
        """
        return self._sam_files

    @sam_files.setter
    def sam_files(self, files):
        """Set the SAM files dictionary."""

        if isinstance(files, dict):
            self._sam_files = files
        elif isinstance(files, str):
            self._sam_files = {0: files}
        elif isinstance(files, list):
            files = sorted(files)
            ids = pd.unique(self.df['configs'])
            self._sam_files = {}
            for i, config_id in enumerate(sorted(ids)):
                try:
                    logger.debug('Mapping project points config ID #{} "{}" '
                                 'to {}'
                                 .format(i, config_id, files[i]))
                    self._sam_files[config_id] = files[i]
                except IndexError:
                    raise IndexError('Setting project points SAM configs with '
                                     'a list raised an error. Project points '
                                     'has the following unique configs: {}, '
                                     'while the following list of SAM configs '
                                     'were input: {}'
                                     .format(ids, files))

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

        if not hasattr(self, '_sam_configs'):
            self._sam_configs = SAMGenConfig(self.sam_files).inputs
        return self._sam_configs

    @property
    def sites(self):
        """Get sites property, type is list if possible
        (if slice stop is None, this will be a slice)."""
        return self._sites

    @sites.setter
    def sites(self, sites):
        """Set the sites property.

        Parameters
        ----------
        sites : list | tuple | slice
            Data to be interpreted as the site list. Can be an explicit site
            list (list or tuple) or a slice that will be converted to a list.
            If slice stop is None, the length of the first resource meta
            dataset is taken as the stop value.
        """

        if isinstance(sites, (list, tuple)):
            # explicit site list, set directly
            self._sites = sites
        elif isinstance(sites, slice):
            if sites.stop:
                # there is an end point, can store as list
                self._sites = list(range(*sites.indices(sites.stop)))
            else:
                # no end point, find one from the length of the meta data
                res = Resource(self.res_file)
                stop = res.shape[1]
                self._sites = list(range(*sites.indices(stop)))
        else:
            raise TypeError('Project Points sites needs to be set as a list, '
                            'tuple, or slice, but was set as: {}'
                            .format(type(sites)))

    @property
    def sites_as_slice(self):
        """Get sites property, type is slice if possible
        (if sites is a non-sequential list, this will return a list)."""

        if not hasattr(self, '_sites_as_slice'):
            if isinstance(self.sites, slice):
                self._sites_as_slice = self.sites
            else:
                # try_slice is what the sites list would be if it is sequential
                if len(self.sites) > 1:
                    try_step = self.sites[1] - self.sites[0]
                else:
                    try_step = 1
                try_slice = slice(self.sites[0], self.sites[-1] + 1, try_step)
                try_list = list(range(*try_slice.indices(try_slice.stop)))

                if self.sites == try_list:
                    # try_slice is equivelant to the site list
                    self._sites_as_slice = try_slice
                else:
                    # cannot be converted to a sequential slice, return list
                    self._sites_as_slice = self.sites

        return self._sites_as_slice

    @property
    def res_file(self):
        """Get the resource file (only used for getting number of sites)."""
        return self._res_file

    @property
    def tech(self):
        """Get the tech property from the config."""
        return self._tech

    def get_sites_from_config(self, config):
        """Get a site list that corresponds to a config key.

        Parameters
        ----------
        config : str
            SAM configuration ID associated with sites.

        Returns
        -------
        sites : list
            List of sites associated with the requested configuration ID. If
            the configuration ID is not recognized, an empty list is returned.
        """

        sites = self.df.loc[(self.df['configs'] == config), 'sites'].values
        return list(sites)

    def parse_project_points(self, points):
        """Parse and set the project points using either a file or slice."""
        if isinstance(points, str):
            self.csv_project_points(points)
        elif isinstance(points, slice):
            if points.stop is None and self.res_file is None:
                raise ValueError('If a project points slice stop is not '
                                 'specified, a resource file must be '
                                 'input to find the number of sites to '
                                 'analyze.')
            self.slice_project_points(points)
        else:
            raise TypeError('Unacceptable project points input type: {}'
                            .format(type(points)))

    def csv_project_points(self, fname):
        """Set the project points using the target csv."""
        if fname.endswith('.csv'):
            self.df = fname
            self.sites = list(self.df['sites'].values)
        else:
            raise ValueError('Config project points file must be '
                             '.csv, but received: {}'
                             .format(fname))

    def slice_project_points(self, points):
        """Set the project points using slice parameters.

        Parameters
        ----------
        points : slice
            Project points slice specifying points to run.
        """
        self.sites = points

        # get the sorted list of available SAM configurations and use the first
        avail_configs = sorted(list(self.sam_configs.keys()))
        self.default_config_id = avail_configs[0]

        if len(avail_configs) > 1:
            warn('Multiple SAM input configurations detected '
                 'for a slice-based site project points. '
                 'Defaulting to: "{}"'
                 .format(avail_configs[0]), ConfigWarning)

        # Make a site-to-config dataframe using the default config
        site_config_dict = {'sites': self.sites,
                            'configs': [self.default_config_id
                                        for s in self.sites]}
        self.df = site_config_dict

    @classmethod
    def split(cls, i0, i1, points, sam_files, config_df, tech, res_file):
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
        points : slice | str
            Slice specifying project points or string pointing to a project
            points csv.
        sam_files : dict | str
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s), top level value is the SAM path. Can also be a single
            config file str.
        config_df : pd.DataFrame
            Sites to SAM configuration dictionary IDs mapping dataframe.
        tech : str
            reV technology being executed.
        res_file : str
            Optional resource file to find maximum length of project points if
            points slice stop is None.

        Returns
        -------
        sub : ProjectPoints
            New instance of ProjectPoints with a subset of the following
            attributes: sites, project points df, and the self dictionary data
            struct.
        """

        # make a new instance of ProjectPoints
        sub = cls(points, sam_files, tech, res_file=res_file)

        # Reset the site attribute using a subset of the original
        sub.sites = sub.sites[i0:i1]

        # set the new config map dataframe
        sub.df = config_df[config_df['sites'].isin(sub.sites)]

        return sub


class SAMGenConfig(BaseConfig):
    """Class to handle the SAM generation section of config input."""
    def __init__(self, SAM_configs):
        """Initialize the SAM generation section of config as an object.

        Parameters
        ----------
        SAM_config : dict
            Keys are config ID's, values are filepaths to the SAM configs.
        """

        # Initialize the SAM generation config section as a dictionary.
        self.set_self_dict(SAM_configs)

    @property
    def inputs(self):
        """Get the SAM input file(s) (JSON) and return as a dictionary.

        Parameters
        ----------
        _inputs : dict
            The keys of this dictionary are the "configuration ID's".
            The values are the imported json SAM input dictionaries.
        """

        if not hasattr(self, '_inputs'):
            self._inputs = {}
            for key, fname in self.items():
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
    def write_profiles(self):
        """Get the boolean write profiles option."""
        if not hasattr(self, '_write_profiles'):
            self._write_profiles = self.__getitem__('write_profiles')
        return self._write_profiles


class HPC(BaseConfig):
    """Class to handle HPC configuration inputs."""

    def __init__(self, **kwargs):
        """Initialize an HPC configuration object with keywords."""
        self.set_self_dict(**kwargs)

    @property
    def option(self):
        """Get the HPC vs. local parallel vs. serial option."""
        default = 'serial'
        if not hasattr(self, '_option'):
            if 'option' in self:
                if self['option']:
                    self._option = self['option'].lower()
                else:
                    # default option if not specified is serial
                    self._option = default
            else:
                # default option if not specified is serial
                self._option = default

        return self._option

    @property
    def queue(self):
        """Get the HPC queue property."""
        default = 'short'
        if not hasattr(self, '_hpc_queue'):
            if 'queue' in self:
                if self['queue']:
                    self._hpc_queue = self['queue']
                else:
                    # default option if not specified
                    self._hpc_queue = default
            else:
                # default option if not specified
                self._hpc_queue = default

        return self._hpc_queue

    @property
    def alloc(self):
        """Get the HPC allocation property."""
        default = 'rev'
        if not hasattr(self, '_hpc_alloc'):
            if 'allocation' in self:
                if self['allocation']:
                    self._hpc_alloc = self['allocation']
                else:
                    # default option if not specified
                    self._hpc_alloc = default
            else:
                # default option if not specified
                self._hpc_alloc = default

        return self._hpc_alloc

    @property
    def node_mem(self):
        """Get the HPC node memory property."""
        defaults = {'short': '32GB',
                    'debug': '32GB',
                    'batch': '32GB',
                    'batch-h': '64GB',
                    'long': '32GB',
                    'bigmem': '64GB',
                    'data-transfer': '32GB',
                    }
        if not hasattr(self, '_hpc_node_mem'):
            if 'memory' in self:
                if self['memory']:
                    self._hpc_node_mem = self['memory']
                else:
                    # default option if not specified
                    self._hpc_node_mem = defaults[self.queue]
            else:
                # default option if not specified
                self._hpc_node_mem = defaults[self.queue]

        return self._hpc_node_mem

    @property
    def walltime(self):
        """Get the HPC node walltime property."""
        defaults = {'short': '04:00:00',
                    'debug': '01:00:00',
                    'batch': '48:00:00',
                    'batch-h': '48:00:00',
                    'long': '240:00:00',
                    'bigmem': '240:00:00',
                    'data-transfer': '120:00:00',
                    }
        if not hasattr(self, '_hpc_walltime'):
            if 'walltime' in self:
                if self['walltime']:
                    self._hpc_walltime = self['walltime']
                else:
                    # default option if not specified
                    self._hpc_walltime = defaults[self.queue]
            else:
                # default option if not specified
                self._hpc_walltime = defaults[self.queue]

        return self._hpc_walltime

    @property
    def nodes(self):
        """Get the number of nodes property."""
        if not hasattr(self, '_nodes'):
            if 'nodes' in self:
                self._nodes = self['nodes']
            else:
                self._nodes = 1
        return self._nodes

    @property
    def ppn(self):
        """Get the process per node (ppn) property."""
        if not hasattr(self, '_ppn'):
            if 'ppn' in self:
                self._ppn = self['ppn']
            else:
                self._ppn = 1
        return self._ppn
