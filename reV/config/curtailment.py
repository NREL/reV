# -*- coding: utf-8 -*-
"""
reV config for curtailment inputs.

Created on Mon Jan 28 11:43:27 2019

@author: gbuster
"""
import logging

from reV.config.base_config import BaseConfig

logger = logging.getLogger(__name__)


class Curtailment(BaseConfig):
    """Config for generation curtailment."""

    def __init__(self, curtailment_parameters):
        """
        Parameters
        ----------
        curtailment_parameters : str | dict
            Configuration json file (with path) containing curtailment
            information. Could also be a pre-extracted curtailment config
            dictionary (the contents of the curtailment json).
        """
        self._wind_speed = None
        self._dawn_dusk = None
        self._months = None
        self._temperature = None
        self._precipitation = None
        self._probability = None
        self._random_seed = None

        if isinstance(curtailment_parameters, str):
            # received json, extract to dictionary
            curtailment_parameters = self.get_file(curtailment_parameters)

        # intialize config object with curtailment parameters
        super().__init__(curtailment_parameters)

    @property
    def wind_speed(self):
        """Get the wind speed threshold below which curtailment is possible.

        Returns
        -------
        _wind_speed : float
            Wind speed threshold below which curtailment is possible. Will
            default to 5.0 m/s (curtailment when wspd < 5.0 m/s).
        """

        if self._wind_speed is None:
            self._wind_speed = float(self.get('wind_speed', 5.0))

        return self._wind_speed

    @property
    def dawn_dusk(self):
        """Get the solar zenith angle that signifies dawn and dusk.

        Returns
        -------
        _dawn_dusk : float
            Solar zenith angle at dawn and dusk. Default is nautical, 12
            degrees below the horizon (sza=102).
        """

        # preset commonly used dawn/dusk values in solar zenith angles.
        presets = {'nautical': 102.0,
                   'astronomical': 108.0,
                   'civil': 96.0}

        if self._dawn_dusk is None:
            # set a default value
            self._dawn_dusk = presets['nautical']

            if 'dawn_dusk' in self:
                if isinstance(self['dawn_dusk'], str):
                    # Use a pre-set dawn/dusk
                    self._dawn_dusk = presets[self['dawn_dusk']]

                if isinstance(self['dawn_dusk'], (int, float)):
                    # Use an explicit solar zenith angle
                    self._dawn_dusk = float(self['dawn_dusk'])

        return self._dawn_dusk

    @property
    def months(self):
        """Get the months during which curtailment is possible (inclusive).

        Returns
        -------
        _months : tuple
            Tuple of month integers. These are the months during which
            curtailment could be in effect. Default is April through July.
        """

        if self._months is None:
            self._months = tuple(self.get('months', (4, 5, 6, 7)))

        return self._months

    @property
    def temperature(self):
        """Get the temperature (C) over which curtailment is possible.

        Returns
        -------
        _temperature : float | NoneType
            Temperature over which curtailment is possible. Defaults to None.
        """

        if self._temperature is None:
            self._temperature = self.get('temperature', None)
        return self._temperature

    @property
    def precipitation(self):
        """Get the precip rate (mm/hour) under which curtailment is possible.

        Returns
        -------
        _precipitation : float | NoneType
            Precipitation rate under which curtailment is possible. This is
            compared to the WTK resource dataset "precipitationrate_0m" in
            mm/hour. Defaults to None.
        """

        if self._precipitation is None:
            self._precipitation = self.get('precipitation', None)
        return self._precipitation

    @property
    def probability(self):
        """Get the probability that curtailment is in-effect if all other
        screening criteria are met.

        Returns
        -------
        _probability : float
            Fractional probability that curtailment is in-effect if all other
            screening criteria are met. Defaults to 1 (curtailment is always
            in effect if all other criteria are met).
        """

        if self._probability is None:
            self._probability = float(self.get('probability', 1.0))

        return self._probability

    @property
    def random_seed(self):
        """
        Random seed to use for curtailment probability

        Returns
        -------
        int
        """
        if self._random_seed is None:
            self._random_seed = int(self.get('random_seed', 0))

        return self._random_seed
