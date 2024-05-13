# -*- coding: utf-8 -*-
"""
Bespoke wind plant optimization CLI utility functions.
"""
import pprint
import logging

from reV.bespoke.bespoke import BespokeWindPlants
from reV.utilities import ModuleName
from reV.utilities.cli_functions import init_cli_logging
from gaps.cli import as_click_command, CLICommandFromClass


logger = logging.getLogger(__name__)


def _preprocessor(config, job_name, log_directory, verbose):
    """Preprocess bespoke config user input.

    Parameters
    ----------
    config : dict
        User configuration file input as (nested) dict.
    job_name : str
        Name of ``reV`` job being run.
    log_directory : str
        Path to log output directory.
    verbose : bool
        Flag to signal ``DEBUG`` verbosity (``verbose=True``).

    Returns
    -------
    dict
        Updated config file.
    """
    init_cli_logging(job_name, log_directory, verbose)
    if isinstance(config["sam_files"], str):
        config["sam_files"] = {'default': config["sam_files"]}

    _log_bespoke_cli_inputs(config)
    return config


def _log_bespoke_cli_inputs(config):
    """Log initial bespoke CLI inputs"""

    logger.info('Source resource file: "{}"'.format(config.get("res_fpath")))
    logger.info('Source exclusion file: "{}"'.format(config.get("excl_fpath")))
    logger.info('Bespoke optimization objective function: "{}"'
                .format(config.get("objective_function")))
    logger.info('Bespoke capital cost function: "{}"'
                .format(config.get("capital_cost_function")))
    logger.info('Bespoke fixed operating cost function: "{}"'
                .format(config.get("fixed_operating_cost_function")))
    logger.info('Bespoke variable operating cost function: "{}"'
                .format(config.get("variable_operating_cost_function")))
    logger.info('Bespoke balance of system cost function: "{}"'
                .format(config.get("balance_of_system_cost_function")))
    logger.info('The following project points were specified: "{}"'
                .format(config.get('project_points', None)))
    logger.info('The following SAM configs are available to this run:\n{}'
                .format(pprint.pformat(config.get('sam_files'), indent=4)))
    logger.debug('The following exclusion dictionary was specified\n{}'
                 .format(pprint.pformat(config.get('excl_dict'), indent=4)))


bespoke_command = CLICommandFromClass(BespokeWindPlants, method="run",
                                      name=str(ModuleName.BESPOKE),
                                      add_collect=False,
                                      split_keys=["project_points"],
                                      config_preprocessor=_preprocessor)
main = as_click_command(bespoke_command)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Bespoke CLI.')
        raise
