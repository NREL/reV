# -*- coding: utf-8 -*-
"""
Econ CLI utility functions.
"""
import pprint
import os
import logging

from rex.utilities.utilities import parse_year
from gaps.pipeline import parse_previous_status
from gaps.cli import as_click_command, CLICommandFromClass

from reV.econ.econ import Econ
from reV.utilities import ModuleName
from reV.utilities.cli_functions import format_analysis_years, init_cli_logging
from reV.utilities.exceptions import ConfigError


logger = logging.getLogger(__name__)


def _preprocessor(config, out_dir, job_name, log_directory, verbose,
                  analysis_years=None):
    """Preprocess econ config user input.

    Parameters
    ----------
    config : dict
        User configuration file input as (nested) dict.
    out_dir : str
        Path to output file directory.
    job_name : str
        Name of ``reV`` job being run.
    log_directory : str
        Path to log output directory.
    verbose : bool
        Flag to signal ``DEBUG`` verbosity (``verbose=True``).
    analysis_years : int | list, optional
        A single year or list of years to perform analysis for. These
        years will be used to fill in any brackets ``{}`` in the
        ``resource_file`` input. If ``None``, the ``resource_file``
        input is assumed to be the full path to the single resource
        file to be processed.  By default, ``None``.

    Returns
    -------
    dict
        Updated config file.
    """
    init_cli_logging(job_name, log_directory, verbose)
    analysis_years = format_analysis_years(analysis_years)
    config["cf_file"] = _parse_cf_files(config["cf_file"], analysis_years,
                                        out_dir)

    _log_econ_cli_inputs(config)
    return config


def _parse_cf_files(cf_file, analysis_years, out_dir):
    """Get the capacity factor files (reV generation output data). """

    # get base filename, may have {} for year format
    if '{}' in cf_file:
        # need to make list of res files for each year
        cf_files = [cf_file.format(year) for year in analysis_years]
    elif 'PIPELINE' in cf_file:
        cf_files = parse_previous_status(out_dir, command=str(ModuleName.ECON))
    else:
        # only one resource file request, still put in list
        cf_files = [cf_file]

    for f in cf_files:
        # ignore files that are to be specified using pipeline utils
        if 'PIPELINE' not in os.path.basename(f):
            if not os.path.exists(f):
                raise IOError('File does not exist: {}'.format(f))

    # check year/cf_file matching if not a pipeline input
    if 'PIPELINE' not in cf_file:
        if len(cf_files) != len(analysis_years):
            raise ConfigError('The number of cf files does not match '
                              'the number of analysis years!'
                              '\n\tCF files: \n\t\t{}'
                              '\n\tYears: \n\t\t{}'
                              .format(cf_files, analysis_years))
        for year in analysis_years:
            if str(year) not in str(cf_files):
                raise ConfigError('Could not find year {} in cf '
                                  'files: {}'.format(year, cf_files))

    return [fn for fn in cf_files if parse_year(fn) in analysis_years]


def _log_econ_cli_inputs(config):
    """Log initial econ CLI inputs"""

    logger.info('The following project points were specified: "{}"'
                .format(config.get('project_points', None)))
    logger.info('The following SAM configs are available to this run:\n{}'
                .format(pprint.pformat(config.get('sam_files', None),
                                       indent=4)))
    logger.debug('Submitting jobs for the following cf_files: {}'
                 .format(config.get("cf_file")))


econ_command = CLICommandFromClass(Econ, method="run",
                                   name=str(ModuleName.ECON),
                                   add_collect=False,
                                   split_keys=["project_points", "cf_file"],
                                   config_preprocessor=_preprocessor)
main = as_click_command(econ_command)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Econ CLI.')
        raise
