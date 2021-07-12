# -*- coding: utf-8 -*-
"""
reV configuration framework for SAM config inputs.
"""
import logging
import os
from warnings import warn

from rex.utilities import safe_json_load

from reV.utilities.exceptions import SAMInputError, SAMInputWarning
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
        self._bifacial = None
        self._icing = None
        self._inputs = None
        self._downscale = None
        self._time_index_step = None

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
                logger.debug('Solar analysis being performed on clearsky '
                             'irradiance.')

        return self._clearsky

    @property
    def bifacial(self):
        """Get a boolean for whether bifacial solar analysis is being run.

        Returns
        -------
        bifacial : bool
            Flag set in the SAM config input with key "bifaciality" for solar
            analysis to analyze bifacial PV panels. Will require albedo input.
            Defaults to False (no bifacial panels is default).
        """
        if self._bifacial is None:
            self._bifacial = False
            for v in self.inputs.values():
                bi_flags = ('bifaciality', 'spe_is_bifacial',
                            'cec_is_bifacial', '6par_is_bifacial')
                bi_bools = [bool(v.get(flag, 0)) for flag in bi_flags]
                self._bifacial = any(bi_bools + [self._bifacial])

        return self._bifacial

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
    def time_index_step(self):
        """
        Step size for time_index for SAM profile output resolution

        Returns
        -------
        int | None
            Step size for time_index, used to reduce temporal resolution
        """
        if self._time_index_step is None:
            time_index_step = []
            for v in self.inputs.values():
                time_index_step.append(v.get('time_index_step', None))

            self._time_index_step = list(set(time_index_step))

        if len(self._time_index_step) > 1:
            msg = ('Expecting a single unique value for "time_index_step" but '
                   'received: {}'.format(self._time_index_step))
            logger.error(msg)
            raise SAMInputError(msg)

        return self._time_index_step[0]

    @property
    def downscale(self):
        """
        Resolution to downscale NSRDB resource to.

        Returns
        -------
        dict | None
            Option for NSRDB resource downscaling to higher temporal
            resolution. The config expects a str entry in the Pandas
            frequency format, e.g. '5min' or a dict of downscaling kwargs
            such as {'frequency': '5min', 'variability_kwargs':
            {'var_frac': 0.05, 'distribution': 'uniform'}}.
            A str entry will be converted to a kwarg dict for the output
            of this property e.g. '5min' -> {'frequency': '5min'}
        """
        if self._downscale is None:
            ds_list = []
            for v in self.inputs.values():
                ds_list.append(v.get('downscale', None))

            self._downscale = ds_list[0]
            ds_list = list({str(x) for x in ds_list})

            if len(ds_list) > 1:
                msg = ('Expecting a single unique value for "downscale" but '
                       'received: {}'.format(ds_list))
                logger.error(msg)
                raise SAMInputError(msg)

        if isinstance(self._downscale, str):
            self._downscale = {'frequency': self._downscale}

        return self._downscale

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
            for key, config in self.items():
                # key is ID (i.e. sam_param_0) that matches project points json
                # fname is the actual SAM config file name (with path)
                if isinstance(config, str):
                    if not config.endswith('.json'):
                        raise IOError('SAM config file must be a JSON: "{}"'
                                      .format(config))
                    elif not os.path.exists(config):
                        raise IOError('SAM config file does not exist: "{}"'
                                      .format(config))
                    else:
                        config = safe_json_load(config)

                if not isinstance(config, dict):
                    raise RuntimeError('SAM config must be a JSON file or a '
                                       'pre-extracted dictionary, but got: {}'
                                       .format(config))

                SAMInputsChecker.check(config)
                self._inputs[key] = config

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
        if all(c in self._config for c in self.KEYS_PV):
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
