"""
reV SAM Configuration
"""
import json
import os
from reV.config.base_config import BaseConfig


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
