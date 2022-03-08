# -*- coding: utf-8 -*-
"""
reV hybrids profile config

@author: ppinchuk
"""
import os
import glob
import logging

from reV.utilities import ModuleName
from reV.utilities.exceptions import PipelineError
from reV.config.base_analysis_config import AnalysisConfig

from rex.utilities.utilities import parse_year

logger = logging.getLogger(__name__)


class HybridsConfig(AnalysisConfig):
    """Hybrids config."""

    NAME = ModuleName.HYBRIDS
    REQUIREMENTS = ('solar_fpath', 'wind_fpath')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        super().__init__(config)
        self._default_ratio = 'solar_capacity/wind_capacity'

    @property
    def solar_fpath(self):
        """Get the solar data filepath"""

        fpath = self['solar_fpath']
        _raise_err_if_pipeline(fpath)

        return _glob_to_yearly_dict(fpath)

    @property
    def wind_fpath(self):
        """Get the wind data filepath"""

        fpath = self['wind_fpath']
        _raise_err_if_pipeline(fpath)

        return _glob_to_yearly_dict(fpath)

    @property
    def allow_solar_only(self):
        """Get the flag indicator for allowing solar-only capacity, """
        return bool(self.get('allow_solar_only', False))

    @property
    def allow_wind_only(self):
        """Get the flag indicator for allowing wind-only capacity, """
        return bool(self.get('allow_wind_only', False))

    @property
    def fillna(self):
        """Get the mapping for specifying input fill values. """
        return self.get('fillna', None)

    @property
    def limits(self):
        """Get the mapping for limiting hybrid meta values. """
        return self.get('limits', None)

    @property
    def ratio(self):
        """Get the ratio input string. """
        return self.get('ratio', self._default_ratio)

    @property
    def ratio_bounds(self):
        """Get the ratio_bounds input. """
        return self.get('ratio_bounds', None)


def _raise_err_if_pipeline(fpath):
    """Raise error if fpath input is 'PIPELINE'. """

    if fpath == 'PIPELINE':
        msg = ('Hybrids module cannot infer fpath from "PIPELINE" - '
               'input is ambiguous. Please specify both the solar and '
               'wind fpath before running hybrids module.')
        logger.error(msg)
        raise PipelineError(msg)


def _glob_to_yearly_dict(fpath):
    """Glob the filepaths into a dictionary based on years. """
    paths = {}
    for fp in glob.glob(fpath):
        fname = os.path.basename(fp)

        try:
            year = parse_year(fname)
        except RuntimeError:
            year = None

        paths.setdefault(year, []).append(fp)

    return paths
