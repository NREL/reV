# -*- coding: utf-8 -*-
"""
reV hybrids profile config

@author: ppinchuk
"""
import os
import glob
import logging
import json

from reV.utilities.exceptions import PipelineError, ConfigError
from reV.config.base_analysis_config import AnalysisConfig

from rex.utilities import parse_year

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
    def ratios(self):
        """Get the ratio limit input mapping. """
        ratios = self.get('ratios', None)
        if ratios is not None:
            try:
                ratios = convert_str_keys_to_tuples(ratios)
            except json.decoder.JSONDecodeError:
                msg = ('One of the keys of "ratios" input is not in proper '
                       'JSON format! Please ensure that the tuple key values '
                       'are represented with square brackets and that the '
                       'column names are in quotation marks. Here is '
                       'an example of a valid "ratios" key: '
                       '`["solar_capacity", \'wind_capacity\']`.')
                logger.error(msg)
                raise ConfigError(msg) from None

        return ratios


def _raise_err_if_pipeline(fpath):
    """Raise error if fpath input is 'PIPELINE'. """

    if fpath == 'PIPELINE':
        msg = 'Cannot run hybrids module as a pipeline job.'
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


def convert_str_keys_to_tuples(input_dict):
    """Load a json dict with tuples as keys.

    Parameters
    ----------
    input_dict : dict
        Dictionary with keys that represent tuples in the format
        `"['col_name', 'col_name2', ...]"`.

    Returns
    -------
    dict
        The input dictionary with the keys converted from str to tuple.
    """
    return {tuple(json.loads(k.replace("'", '"'))): v
            for k, v in input_dict.items()}
