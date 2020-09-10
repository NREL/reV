# -*- coding: utf-8 -*-
"""
reV config for curtailment inputs.

Created on Mon Jan 28 11:43:27 2019

@author: gbuster
"""
import logging
from rex.utilities import check_eval_str
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
            curtailment_parameters = self.get_file(curtailment_parameters)

        # intialize config object with curtailment parameters
        super().__init__(curtailment_parameters)

    @property
    def wind_speed(self):
        """Get the wind speed threshold below which curtailment is possible.

        Returns
        -------
        _wind_speed : float | None
            Wind speed threshold below which curtailment is possible.
        """
        return self.get('wind_speed', None)

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

        # set a default value
        dd = presets['nautical']

        if 'dawn_dusk' in self:
            if isinstance(self['dawn_dusk'], str):
                # Use a pre-set dawn/dusk
                dd = presets[self['dawn_dusk']]

            if isinstance(self['dawn_dusk'], (int, float)):
                # Use an explicit solar zenith angle
                dd = float(self['dawn_dusk'])

        return dd

    @property
    def months(self):
        """Get the months during which curtailment is possible (inclusive).
        This can be overridden by the date_range input.

        Returns
        -------
        months : tuple | None
            Tuple of month integers. These are the months during which
            curtailment could be in effect. Default is None.
        """
        m = self.get('months', None)
        if isinstance(m, list):
            m = tuple(m)
        return m

    @property
    def date_range(self):
        """Get the date range tuple (start, end) over which curtailment is
        possible (inclusive, exclusive) ("MMDD", "MMDD"). This overrides the
        months input.

        Returns
        -------
        date_range : tuple
            Two-entry tuple of the starting date (inclusive) and ending date
            (exclusive) over which curtailment is possible. Input format is a
            zero-padded string: "MMDD".
        """
        dr = self.get('date_range', None)
        if dr is not None:
            msg = 'date_range input needs to be a tuple!'
            assert isinstance(dr, (list, tuple)), msg
            msg = 'date_range input needs to have two entries!'
            assert len(dr) == 2, msg
            dr = (str(int(dr[0])).zfill(4), str(int(dr[1])).zfill(4))

        return dr

    @property
    def temperature(self):
        """Get the temperature (C) over which curtailment is possible.

        Returns
        -------
        temperature : float | NoneType
            Temperature over which curtailment is possible. Defaults to None.
        """
        return self.get('temperature', None)

    @property
    def precipitation(self):
        """Get the precip rate (mm/hour) under which curtailment is possible.

        Returns
        -------
        precipitation : float | NoneType
            Precipitation rate under which curtailment is possible. This is
            compared to the WTK resource dataset "precipitationrate_0m" in
            mm/hour. Defaults to None.
        """
        return self.get('precipitation', None)

    @property
    def equation(self):
        """Get an equation-based curtailment scenario.

        Returns
        -------
        equation : str
            A python equation based on other curtailment variables (wind_speed,
            temperature, precipitation_rate, solar_zenith_angle) that returns
            a True or False output to signal curtailment.
        """
        eq = self.get('equation', None)
        if isinstance(eq, str):
            check_eval_str(eq)
        return eq

    @property
    def probability(self):
        """Get the probability that curtailment is in-effect if all other
        screening criteria are met.

        Returns
        -------
        probability : float
            Fractional probability that curtailment is in-effect if all other
            screening criteria are met. Defaults to 1 (curtailment is always
            in effect if all other criteria are met).
        """
        return float(self.get('probability', 1.0))

    @property
    def random_seed(self):
        """
        Random seed to use for curtailment probability

        Returns
        -------
        int
        """
        return int(self.get('random_seed', 0))
