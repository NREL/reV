"""
reV Configuration
"""
import json
import logging
import os

from reV.exceptions import ConfigError


logger = logging.getLogger(__name__)


class BaseConfig(dict):
    """Base class for configuration frameworks."""

    def __init__(self, config_dict):
        """Initialize configuration object with keyword dict."""
        self.set_self_dict(config_dict)

    @staticmethod
    def check_files(flist):
        """Make sure all files in the input file list exist."""
        for f in flist:
            if os.path.exists(f) is False:
                raise IOError('File does not exist: {}'.format(f))

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
        elif os.path.exists(fname) is False:
            raise IOError('Configuration file does not exist: "{}"'
                          .format(fname))
        else:
            raise ConfigError('Unknown error getting configuration file: "{}"'
                              .format(fname))
        return config

    @property
    def logging_level(self):
        """Get user-specified logging level in "project_control" namespace."""
        default = 'WARNING'
        if not hasattr(self, '_logging_level'):
            levels = {'DEBUG': logging.DEBUG,
                      'INFO': logging.INFO,
                      'WARNING': logging.WARNING,
                      'ERROR': logging.ERROR,
                      'CRITICAL': logging.CRITICAL,
                      }
            if 'logging_level' in self['project_control']:
                x = self['project_control']['logging_level']
                self._logging_level = levels[x.upper()]
            else:
                self._logging_level = levels[default]
        return self._logging_level

    @property
    def name(self):
        """Get the project name in "project_control" namespace."""
        default = 'rev'
        if not hasattr(self, '_name'):
            if 'name' in self['project_control']:
                if self['project_control']['name']:
                    self._name = self['project_control']['name']
                else:
                    self._name = default
            else:
                self._name = default

        return self._name
