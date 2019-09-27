# -*- coding: utf-8 -*-
"""
reV configuration framework for SAM config inputs.
"""
import logging
import os
from warnings import warn

from reV.utilities import safe_json_load
from reV.utilities.exceptions import ConfigWarning
from reV.config.base_config import BaseConfig


logger = logging.getLogger(__name__)


class SAMConfig(BaseConfig):
    """Class to handle the SAM section of config input."""

    def __init__(self, SAM_configs):
        """
        Parameters
        ----------
        SAM_configs : dict
            Keys are config ID's, values are filepaths to the SAM configs.
        """
        self._clearsky = None
        self._icing = None
        self._inputs = None
        super().__init__(SAM_configs)

    @property
    def clearsky(self):
        """Get a boolean for whether solar resource requires clearsky irrad.

        Returns
        -------
        _clearsky : bool
            Flag set in the SAM config input with key "clearsky" for solar
            analysis to process generation for clearsky irradiance.
            Defaults to False (normal all-sky irradiance).
        """

        if self._clearsky is None:
            self._clearsky = False
            for v in self.inputs.values():
                self._clearsky = any((self._clearsky,
                                      bool(v.get('clearsky', False))))
            if self._clearsky:
                warn('Solar analysis being performed on clearsky irradiance.',
                     ConfigWarning)
        return self._clearsky

    @property
    def icing(self):
        """Get a boolean for whether wind generation is considering icing.

        Returns
        -------
        _icing : bool
            Flag for whether wind generation is considering icing effects.
            Based on whether SAM input json has "en_icing_cutoff" == 1.
        """

        if self._icing is None:
            self._icing = False
            for v in self.inputs.values():
                self._icing = any((self._icing,
                                   bool(v.get('en_icing_cutoff', False))))
            if self._icing:
                logger.debug('Icing analysis active for wind gen.')
        return self._icing

    @property
    def inputs(self):
        """Get the SAM input file(s) (JSON) and return as a dictionary.

        Parameters
        ----------
        _inputs : dict
            The keys of this dictionary are the "configuration ID's".
            The values are the imported json SAM input dictionaries.
        """

        if self._inputs is None:
            self._inputs = {}
            for key, fname in self.items():
                # key is ID (i.e. sam_param_0) that matches project points json
                # fname is the actual SAM config file name (with path)

                if fname.endswith('.json') is True:
                    if os.path.exists(fname):
                        self._inputs[key] = safe_json_load(fname)

                    else:
                        raise IOError('SAM inputs file does not exist: "{}"'
                                      .format(fname))
                else:
                    raise IOError('SAM inputs file must be a JSON: "{}"'
                                  .format(fname))
        return self._inputs
