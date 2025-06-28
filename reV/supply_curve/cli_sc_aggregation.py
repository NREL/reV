# -*- coding: utf-8 -*-
"""
reV Supply Curve Aggregation CLI utility functions.
"""
import logging
from pathlib import Path

import click
import pandas as pd
from rex import init_logger
from rex.multi_file_resource import MultiFileResource
from rex.utilities.utilities import check_res_file
from gaps.cli import as_click_command, CLICommandFromClass

from reV import __version__
from reV.supply_curve.sc_aggregation import SupplyCurveAggregation
from reV.utilities import ModuleName
from reV.utilities.cli_functions import (parse_from_pipeline,
                                         compile_descriptions)
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
    config = _validate_res_fpath(config)
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


def _validate_res_fpath(config):
    """Format res_fpath with year (if needed) and check for file existence"""
    res_fpath = config.setdefault("res_fpath", None)
    if not isinstance(res_fpath, str):
        return config

    if '{}' in res_fpath:
        config["res_fpath"] = _get_filepath_with_year(res_fpath)
    else:
        check_res_file(res_fpath)

    return config


def _get_filepath_with_year(res_fpath):
    """Find first file that exists on disk with year filled in"""

    for year in range(1950, 2100):
        fp = res_fpath.format(year)
        try:
            check_res_file(fp)
        except FileNotFoundError:
            continue
        return fp

    msg = ("Could not find any files that match the pattern"
           "{!r}".format(res_fpath.format("<year>")))
    logger.error(msg)
    raise FileNotFoundError(msg)


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


@click.command()
@click.version_option(version=__version__)
@click.option('--sc_fpath', '-sc', type=click.Path(), default=None,
              help='Supply curve CSV file path used to subset the columns')
@click.option('--out_fp', '-of', default=None, type=click.Path(),
              help='Output CSV file for the column descriptions')
@click.pass_context
def sc_col_descriptions(ctx, sc_fpath, out_fp):
    """Generate reV supply curve column descriptions"""
    if ctx.obj.get('VERBOSE', False):
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    init_logger('reV', log_level=log_level)

    cols = []
    if sc_fpath:
        cols = pd.read_csv(sc_fpath, nrows=0).columns.tolist()

    if not out_fp:
        if sc_fpath:
            sc_name = Path(sc_fpath).stem
            out_fp = Path(sc_fpath).parent / f"{sc_name}_column_lookup.csv"
        else:
            out_fp = "rev_supply_curve_column_lookup.csv"

    columns = compile_descriptions(cols)
    columns.to_csv(out_fp, index=False)
    logger.info("Column descriptions saved to %s", out_fp)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV SC Aggregation CLI.')
        raise
