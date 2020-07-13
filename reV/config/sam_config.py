# -*- coding: utf-8 -*-
"""
reV configuration framework for SAM config inputs.
"""
import logging
import os
from warnings import warn

from rex.utilities import safe_json_load

from reV.utilities.exceptions import ConfigWarning, SAMInputWarning
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
        super().__init__(SAM_configs, check_keys=False)
        self._clearsky = None
        self._icing = None
        self._inputs = None

    @property
    def clearsky(self):
        """Get a boolean for whether solar resource requires clearsky irrad.

        Returns
        -------
        clearsky : bool
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
                        config = safe_json_load(fname)
                        SAMInputsChecker.check(config)
                        self._inputs[key] = config

                    else:
                        raise IOError('SAM inputs file does not exist: "{}"'
                                      .format(fname))
                else:
                    raise IOError('SAM inputs file must be a JSON: "{}"'
                                  .format(fname))
        return self._inputs


class SAMInputsChecker:
    """Class to check SAM input jsons and warn against bad inputs."""

    # Keys that are used to identify a technology config
    KEYS_PV = ('tilt', 'azimuth', 'module_type', 'array_type')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Extracted SAM technology input config in dict form.
        """
        if isinstance(config, dict):
            self._config = config
        else:
            raise TypeError('Bad SAM tech config type: {}'
                            .format(type(config)))

    def check_pv(self):
        """Run input checks for a pv input config."""
        if self._config['array_type'] >= 2 and self._config['tilt'] != 0:
            w = ('SAM input for PV has array type {} (tracking) and tilt '
                 'of {}. This is uncommon!'
                 .format(self._config['array_type'], self._config['tilt']))
            logger.warning(w)
            warn(w, SAMInputWarning)

    def _run_checks(self):
        """Infer config type and run applicable checks."""
        if all([c in self._config for c in self.KEYS_PV]):
            self.check_pv()

    @classmethod
    def check(cls, config):
        """Run checks on a SAM input json config.

        Parameters
        ----------
        config : dict
            Extracted SAM technology input config in dict form.
        """
        c = cls(config)
        c._run_checks()
