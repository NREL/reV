# -*- coding: utf-8 -*-
"""
reV Supply Curve Aggregation CLI utility functions.
"""
import os
import logging

from rex.multi_file_resource import MultiFileResource
from gaps.cli import as_click_command, CLICommandFromClass

from reV.supply_curve.sc_aggregation import SupplyCurveAggregation
from reV.utilities import ModuleName
from reV.utilities.cli_functions import parse_from_pipeline
from reV.utilities.exceptions import ConfigError


logger = logging.getLogger(__name__)


def _preprocessor(config, out_dir):
    """Preprocess aggregation config user input.

    Parameters
    ----------
    config : dict
        User configuration file input as (nested) dict.
    out_dir : str
        Path to output file directory.

    Returns
    -------
    dict
        Updated config file.
    """
    config = _format_res_fpath(config)
    _validate_tm(config)

    key_to_modules = {"gen_fpath": [ModuleName.MULTI_YEAR,
                                    ModuleName.COLLECT,
                                    ModuleName.GENERATION],
                      "econ_fpath": [ModuleName.MULTI_YEAR,
                                     ModuleName.COLLECT,
                                     ModuleName.ECON]}
    for key, modules in key_to_modules.items():
        config = parse_from_pipeline(config, out_dir, key, modules)

    return config


def _format_res_fpath(config):
    """Format res_fpath with year, if need be. """
    res_fpath = config.setdefault("res_fpath", None)
    if isinstance(res_fpath, str):
        if '{}' in res_fpath:
            for year in range(1950, 2100):
                if os.path.exists(res_fpath.format(year)):
                    break
            else:
                msg = ("Could not find any files that match the pattern"
                       "{!r}".format(res_fpath.format("<year>")))
                logger.error(msg)
                raise FileNotFoundError(msg)

            res_fpath = res_fpath.format(year)

        elif not os.path.exists(res_fpath):
            msg = "Could not find resource file: {!r}".format(res_fpath)
            logger.error(msg)
            raise FileNotFoundError(msg)

        config["res_fpath"] = res_fpath

    return config


def _validate_tm(config):
    """Check that tm_dset exists or that res_fpath is given (to generate tm)"""
    paths = config["excl_fpath"]
    if isinstance(paths, str):
        paths = [paths]

    with MultiFileResource(paths, check_files=False) as res:
        dsets = res.datasets

    if config["tm_dset"] not in dsets and config["res_fpath"] is None:
        raise ConfigError('Techmap dataset "{}" not found in exclusions '
                          'file, resource file input "res_fpath" is '
                          'required to create the techmap file.'
                          .format(config["tm_dset"]))


NAME = str(ModuleName.SUPPLY_CURVE_AGGREGATION)
sc_agg_command = CLICommandFromClass(SupplyCurveAggregation, method="run",
                                     name=NAME, add_collect=False,
                                     split_keys=None,
                                     config_preprocessor=_preprocessor)
main = as_click_command(sc_agg_command)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV SC Aggregation CLI.')
        raise
