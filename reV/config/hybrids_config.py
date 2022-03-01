# -*- coding: utf-8 -*-
"""
reV hybrids profile config

@author: ppinchuk
"""
import os
import glob
import logging

from reV.hybrids.hybrids import RatioColumns
from reV.utilities.exceptions import PipelineError, InputError
from reV.config.base_analysis_config import AnalysisConfig

from rex.utilities.utilities import parse_year, dict_str_load

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


# def parse_ratio_input(ratio):
#     """Parse the input and convert it to the expected ratio input dict.

#     Parameters
#     ----------
#     ratio : str | dict | None
#         Input string or dictionary to be converted to Hybrids ratio input
#         dict.
#     """
#     if isinstance(ratio, str):
#         ratio = dict_str_load(ratio)

#     if ratio is not None:
#         ratio = convert_cli_ratio_dict_to_hybrids_input_dict(ratio)

#     return ratio


# def convert_cli_ratio_dict_to_hybrids_input_dict(input_dict):
#     """Convert cli input to dictionary expected by Hybridization class.

#     Parameters
#     ----------
#     input_dict : dict
#         Input cli dictionary with the following required keys:
#         ['numerator_col', 'denominator_col', 'min_ratio', 'max_ratio'].
#         The values of these keys are not validated. A value for the key
#         'fixed' may also be provided, but it must match the value of
#         either 'numerator_col' or 'denominator_col'.

#     Returns
#     -------
#     dict
#         Input dictionary formatted to be used as input to Hybridization
#         class: keys are tuples of the column names, and the values are
#         the ratio bounds.

#     Raises
#     ------
#     InputError
#         If the input dictionary is missing one of the required keys:
#         ['numerator_col', 'denominator_col', 'min_ratio', 'max_ratio'].
#     InputError
#         If the "fixed" key is provided and does not match the values of one
#         ratio columns: ['numerator_col', 'denominator_col'].
#     """
#     _validate_ratio_keys(input_dict)

#     fixed_col = input_dict.get('fixed')
#     _validate_fixed_col_input(fixed_col, input_dict)

#     new_key = RatioColumns(input_dict['numerator_col'],
#                            input_dict['denominator_col'],
#                            fixed_col)
#     new_value = input_dict['min_ratio'], input_dict['max_ratio']
#     return {new_key: new_value}


# def _validate_ratio_keys(input_dict):
#     """Make sure input_dict contains expected ratio dict keys."""
#     expected_keys = ['numerator_col', 'denominator_col',
#                      'min_ratio', 'max_ratio']
#     for key_name in expected_keys:
#         if key_name not in input_dict:
#             msg = "Key {!r} (required) not found in input ratio dictionary!"
#             e = msg.format(key_name)
#             logger.error(e)
#             raise InputError(e) from None


# def _validate_fixed_col_input(fixed_col, input_dict):
#     """Make sure fixed_col input is a valid column name of input_dict."""
#     if fixed_col is not None:
#         allowed_values = {input_dict['numerator_col'],
#                           input_dict['denominator_col']}
#         if fixed_col not in allowed_values:
#             msg = ('Received input {!r} for "fixed" key that does '
#                    'not match either ratio column input: {!r}! Please '
#                    'ensure this input matches one of the ratio columns '
#                    'or is set to None (null).')
#             e = msg.format(fixed_col, allowed_values)
#             logger.error(e)
#             raise InputError(e) from None


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
