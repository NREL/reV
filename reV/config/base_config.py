# -*- coding: utf-8 -*-
"""
reV Base Configuration Framework
"""
import json
import logging
import os

from reV.utilities import safe_json_load
from reV.utilities.exceptions import ConfigError


logger = logging.getLogger(__name__)
REVDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTDATADIR = os.path.join(os.path.dirname(REVDIR), 'tests', 'data')


class BaseConfig(dict):
    """Base class for configuration frameworks."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """

        # str_rep is a mapping of config strings to replace with real values
        self.str_rep = {'REVDIR': REVDIR,
                        'TESTDATADIR': TESTDATADIR,
                        }

        self.dir = None
        self._logging_level = None
        self._name = None
        self._parse_config(config)

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
                # get the directory of the config file
                self.dir = os.path.dirname(os.path.realpath(config)) + '/'
                self.dir = self.dir.replace('\\', '/')
                self.str_rep['./'] = self.dir
                config = self.get_file(config)
            else:
                # attempt to deserialize non-json string
                config = json.loads(config)

        # Perform string replacement, save config to self instance
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

    @property
    def logging_level(self):
        """Get user-specified logging level in "project_control" namespace.

        Returns
        -------
        _logging_level : int
            Python logging module level (integer format) corresponding to the
            config-specified logging level string.
        """

        if self._logging_level is None:
            levels = {'DEBUG': logging.DEBUG,
                      'INFO': logging.INFO,
                      'WARNING': logging.WARNING,
                      'ERROR': logging.ERROR,
                      'CRITICAL': logging.CRITICAL,
                      }
            # set default value
            self._logging_level = logging.INFO
            if 'project_control' in self:
                if 'logging_level' in self['project_control']:
                    x = self['project_control']['logging_level']
                    self._logging_level = levels[x.upper()]
        return self._logging_level

    @property
    def name(self):
        """Get the project name in "project_control" namespace.

        Returns
        -------
        _name : str
            Config-specified project control name.
        """

        if self._name is None:
            # set default value
            self._name = 'rev'
            if 'project_control' in self:
                if 'name' in self['project_control']:
                    if self['project_control']['name']:
                        self._name = self['project_control']['name']
        return self._name
