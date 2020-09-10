# -*- coding: utf-8 -*-
"""
reV Base Configuration Framework
"""
import json
import logging
import os

from rex.utilities import safe_json_load
from rex.utilities.utilities import get_class_properties

from reV.utilities.exceptions import ConfigError

logger = logging.getLogger(__name__)
REVDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTDATADIR = os.path.join(os.path.dirname(REVDIR), 'tests', 'data')


class BaseConfig(dict):
    """Base class for configuration frameworks."""
    REQUIREMENTS = ()

    def __init__(self, config, check_keys=True, perform_str_rep=True):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        check_keys : bool, optional
            Flag to check config keys against Class properties, by default True
        perform_str_rep : bool
            Flag to perform string replacement for REVDIR, TESTDATADIR, and ./
        """

        # str_rep is a mapping of config strings to replace with real values
        self._perform_str_rep = perform_str_rep
        self.str_rep = {'REVDIR': REVDIR,
                        'TESTDATADIR': TESTDATADIR,
                        }

        self._config_dir = None
        self._log_level = None
        self._name = None
        self._parse_config(config)

        self._preflight()

        self._keys = self._get_properties()
        if check_keys:
            self._check_keys()

    @property
    def config_dir(self):
        """Get the directory that the config file is in.

        Returns
        -------
        config_dir : str
            Directory path that the config file is in.
        """
        return self._config_dir

    @property
    def config_keys(self):
        """
        List of valid config keys

        Returns
        -------
        list
        """
        return self._keys

    @property
    def log_level(self):
        """Get user-specified "log_level" (DEBUG, INFO, WARNING, etc...).

        Returns
        -------
        log_level : int
            Python logging module level (integer format) corresponding to the
            config-specified log level string.
        """

        if self._log_level is None:
            levels = {'DEBUG': logging.DEBUG,
                      'INFO': logging.INFO,
                      'WARNING': logging.WARNING,
                      'ERROR': logging.ERROR,
                      'CRITICAL': logging.CRITICAL,
                      }

            x = str(self.get('log_level', 'INFO'))
            self._log_level = levels[x.upper()]

        return self._log_level

    @property
    def name(self):
        """Get the project name from the "name" key.

        Returns
        -------
        name : str
            Config-specified project control name.
        """

        if self._name is None:
            self._name = self.get('name', 'rev')
        return self._name

    def _preflight(self):
        """Run a preflight check on the config."""
        if 'project_control' in self:
            msg = ('config "project_control" block is no '
                   'longer used. All project control keys should be placed at '
                   'the top config level.')
            logger.error(msg)
            raise ConfigError(msg)

        missing = []
        for req in self.REQUIREMENTS:
            if req not in self:
                missing.append(req)

        if any(missing):
            e = ('{} missing the following keys: {}'
                 .format(self.__class__.__name__, missing))
            logger.error(e)
            raise ConfigError(e)

    @classmethod
    def _get_properties(cls):
        """
        Get all class properties
        Used to check against config keys

        Returns
        -------
        properties : list
            List of class properties, each of which should represent a valid
            config key/entry
        """
        return get_class_properties(cls)

    def _check_keys(self):
        """
        Check on config keys to ensure they match available
        properties
        """
        for key, value in self.items():
            if isinstance(value, str) and key not in self._keys:
                msg = ('{} is not a valid config entry for {}! Must be one of:'
                       '\n{}'.format(key, self.__class__.__name__, self._keys))
                logger.error(msg)
                raise ConfigError(msg)

    def check_overwrite_keys(self, primary_key, *overwrite_keys):
        """
        Check for overwrite keys and raise a ConfigError if present

        Parameters
        ----------
        primary_key : str
            Primary key that overwrites overwrite_keys, used for error message
        overwrite_keys : str
            Key(s) to overwrite
        """
        overwrite = []
        for key in overwrite_keys:
            if key in self:
                overwrite.append(key)

        if overwrite:
            msg = ('A value for "{}" was provided which overwrites the '
                   ' following key: "{}", please remove them from the config'
                   .format(primary_key, ', '.join(overwrite)))
            logger.error(msg)
            raise ConfigError(msg)

    def _parse_config(self, config):
        """Parse a config input and set appropriate instance attributes.

        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """

        # str is either json file path or serialized json object
        if isinstance(config, str):
            if config.endswith('.json'):
                self._config_dir = os.path.dirname(os.path.realpath(config))
                self._config_dir += '/'
                self._config_dir = self._config_dir.replace('\\', '/')
                self.str_rep['./'] = self.config_dir
                config = self.get_file(config)
            else:
                # attempt to deserialize non-json string
                config = json.loads(config)

        # Perform string replacement, save config to self instance
        if self._perform_str_rep:
            config = self.str_replace(config, self.str_rep)

        self.set_self_dict(config)

    @staticmethod
    def check_files(flist):
        """Make sure all files in the input file list exist.

        Parameters
        ----------
        flist : list
            List of files (with paths) to check existance of.
        """
        for f in flist:
            # ignore files that are to be specified using pipeline utils
            if 'PIPELINE' not in os.path.basename(f):
                if os.path.exists(f) is False:
                    raise IOError('File does not exist: {}'.format(f))

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
                d[key] = BaseConfig.str_replace(val, strrep)

        elif isinstance(d, list):
            # if the value is also a list, iterate through
            for i, entry in enumerate(d):
                d[i] = BaseConfig.str_replace(entry, strrep)

        elif isinstance(d, str):
            # if val is a str, check to see if str replacements apply
            for old_str, new in strrep.items():
                # old_str is in the value, replace with new value
                d = d.replace(old_str, new)

        # return updated
        return d

    def set_self_dict(self, dictlike):
        """Save a dict-like variable as object instance dictionary items.

        Parameters
        ----------
        dictlike : dict
            Python namespace object to set to this dictionary-emulating class.
        """
        for key, val in dictlike.items():
            self.__setitem__(key, val)

    @staticmethod
    def get_file(fname):
        """Read the config file.

        Parameters
        ----------
        fname : str
            Full path + filename. Must be a .json file.

        Returns
        -------
        config : dict
            Config data.
        """

        logger.debug('Getting "{}"'.format(fname))
        if os.path.exists(fname) and fname.endswith('.json'):
            config = safe_json_load(fname)
        elif os.path.exists(fname) is False:
            raise FileNotFoundError('Configuration file does not exist: "{}"'
                                    .format(fname))
        else:
            raise ConfigError('Unknown error getting configuration file: "{}"'
                              .format(fname))
        return config
