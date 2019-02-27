# -*- coding: utf-8 -*-
"""
reV analysis configs (generation, lcoe, etc...)

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

        if isinstance(curtailment_parameters, str):
            # received json, extract to dictionary
            self.file = curtailment_parameters
            curtailment_parameters = self.get_file(self.file)

        # intialize config object with curtailment parameters
        super().__init__(curtailment_parameters)

    @property
    def wind_speed(self):
        """Get the wind speed threshold below which curtailment is possible.

        Returns
        -------
        _wind_speed : int | float
            Wind speed threshold below which curtailment is possible. Will
            default to 5.0 m/s (curtailment when wspd < 5.0 m/s).
        """

        if not hasattr(self, '_wind_speed'):
            # set a default value
            self._wind_speed = 5.0
            if 'wind_speed' in self:
                if self['wind_speed']:
                    self._wind_speed = self['wind_speed']
        return self._wind_speed

    @property
    def dawn_dusk(self):
        """Get the solar zenith angle that signifies dawn and dusk.

        Returns
        -------
        _dawn_dusk : int | float
            Solar zenith angle at dawn and dusk. Default is nautical, 12
            degrees below the horizon (sza=102).
        """

        # preset commonly used dawn/dusk values in solar zenith angles.
        presets = {'nautical': 102.0,
                   'astronomical': 108.0,
                   'civil': 96.0}

        if not hasattr(self, '_dawn_dusk'):
            # set a default value
            self._dawn_dusk = presets['nautical']
            if 'dawn_dusk' in self:
                if isinstance(self['dawn_dusk'], str):
                    # Use a pre-set dawn/dusk
                    self._dawn_dusk = presets[self['dawn_dusk']]
                if isinstance(self['dawn_dusk'], (int, float)):
                    # Use an explicit solar zenith angle
                    self._dawn_dusk = self['dawn_dusk']
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

        if not hasattr(self, '_months'):
            # set a default value
            self._months = (4, 5, 6, 7)
            if 'months' in self:
                if self['months']:
                    self._months = tuple(self['months'])
        return self._months

    @property
    def temperature(self):
        """Get the temperature (C) over which curtailment is possible.

        Returns
        -------
        _temperature : int | float
            Temperature over which curtailment is possible. Defaults to a low
            value so that this screening metric is not used by default.
        """

        if not hasattr(self, '_temperature'):
            # set a default value
            self._temperature = -1000.0
            if 'temperature' in self:
                if self['temperature']:
                    self._temperature = self['temperature']
        return self._temperature

    @property
    def precipitation(self):
        """Get the precipitation under which curtailment is possible.

        Returns
        -------
        _precipitation : int | float
            Precipitation under which curtailment is possible. Defaults to a
            high value so that this screening metric is not used by default.
        """

        if not hasattr(self, '_precipitation'):
            # set a default value
            self._precipitation = 1000.0
            if 'precipitation' in self:
                if self['precipitation']:
                    self._precipitation = self['precipitation']
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

        if not hasattr(self, '_probability'):
            # set a default value
            self._probability = 1.0
            if 'probability' in self:
                if self['probability']:
                    self._probability = self['probability']
        return self._probability
