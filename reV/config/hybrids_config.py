# -*- coding: utf-8 -*-
"""
reV hybrids profile config

@author: ppinchuk
"""
import logging

from reV.utilities.exceptions import PipelineError
from reV.config.base_analysis_config import AnalysisConfig

logger = logging.getLogger(__name__)


class HybridsConfig(AnalysisConfig):
    """Hybrids config."""

    NAME = 'hybrids'
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
        self._default_ratio_cols = ('solar_capacity', 'wind_capacity')

    @property
    def solar_fpath(self):
        """Get the solar data filepath"""

        fpath = self['solar_fpath']

        if fpath == 'PIPELINE':
            msg = 'Cannot run hybrids module as a pipeline job.'
            logger.error(msg)
            raise PipelineError(msg)

        return fpath

    @property
    def solar_fpath(self):
        """Get the wind data filepath"""

        fpath = self['wind_fpath']

        if fpath == 'PIPELINE':
            msg = 'Cannot run hybrids module as a pipeline job.'
            logger.error(msg)
            raise PipelineError(msg)

        return fpath

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
    def allowed_ratio(self):
        """Get the allowed ratio (or ratio bounds) for the input columns. """
        return self.get('allowed_ratio', None)

    @property
    def ratio_cols(self):
        """Get the columns used to calculate the ratio."""
        return self.get('ratio_cols', self._default_ratio_cols)
