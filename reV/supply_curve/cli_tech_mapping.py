# -*- coding: utf-8 -*-
"""
reV Tech Mapping CLI utility functions.
"""
import logging

from gaps.cli import as_click_command, CLICommandFromClass

from reV.supply_curve.tech_mapping import TechMapping
from reV.utilities import ModuleName
from reV.utilities.exceptions import ConfigError
from reV.supply_curve.cli_sc_aggregation import _validate_res_fpath

logger = logging.getLogger(__name__)


def _preprocessor(config):
    """Preprocess tech mapping config user input.

    Parameters
    ----------
    config : dict
        User configuration file input as (nested) dict.

    Returns
    -------
    dict
        Updated config file.
    """
    _validate_excl_fpath(config)
    config = _validate_res_fpath(config)
    _validate_dset(config)

    return config


def _validate_excl_fpath(config):
    paths = config["excl_fpath"]
    if isinstance(paths, list):
        raise ConfigError(
            "Multiple exclusion file paths passed via excl_fpath. "
            "Cannot run tech mapping with arbitrary multiple exclusion. "
            "Specify a single exclusion file path to write to."
        )


def _validate_dset(config):
    if config.get("dset") is None:
        raise ConfigError(
            "dset must be specified to run tech mapping."
        )


tm_command = CLICommandFromClass(TechMapping, method="run",
                                 name=str(ModuleName.TECH_MAPPING),
                                 add_collect=False, split_keys=None,
                                 config_preprocessor=_preprocessor)
main = as_click_command(tm_command)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Tech Mapping CLI.')
        raise
