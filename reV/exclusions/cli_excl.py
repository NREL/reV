"""
Exclusions CLI entry points.
"""
import click
import logging
import os
import pprint

from reV.exclusions.exclusions import Exclusions
from reV.config.analysis_configs import ExclConfig
from reV.utilities.cli_dtypes import STR
from reV.utilities.loggers import init_logger, REV_LOGGERS

logger = logging.getLogger(__name__)


def init_excl_loggers(verbose, name, node=False, logdir='./out/log',
                      modules=[__name__, 'reV.exclusions.exclusions',
                               'reV.config', 'reV.utilities']):
    """Init multiple loggers to a single file or stdout for the exclusion compute.

    Parameters
    ----------
    verbose : bool
        Option to turn on debug vs. info logging.
    name : str
        Generation compute job name, interpreted as name of intended log file.
    node : bool
        Flag for whether this is a node-level logger. If this is a node logger,
        and the log level is info, the log_file will be None (sent to stdout).
    logdir : str
        Target directory to save .log files.
    modules : list
        List of reV modules to initialize loggers for.
        Note: From the generation cli, __name__ AND 'reV.generation.generation'
        must both be initialized.

    Returns
    -------
    loggers : list
        List of logging instances that were initialized.
    """

    if verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    loggers = []
    for module in modules:
        log_file = os.path.join(logdir, '{}.log'.format(name))

        # check for redundant loggers in the REV_LOGGERS singleton
        logger = REV_LOGGERS[module]

        if ((not node or (node and log_level == 'DEBUG')) and
                'log_file' not in logger):
            # No log file belongs to this logger, init a logger file
            logger = init_logger(module, log_level=log_level,
                                 log_file=log_file)
        elif node and log_level == 'INFO':
            # Node level info loggers only go to STDOUT/STDERR files
            logger = init_logger(module, log_level=log_level, log_file=None)
        loggers.append(logger)
    return loggers


@click.group()
@click.option('--name', '-n', default='reV_gen', type=STR,
              help='Exclusion job name. Default is "reV_gen".')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """Command line interface (CLI) for the reV 2.0 Exclusion Module."""
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c', required=True, type=STR,
              help='reV exclusion configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV gen from a config file."""
    name = ctx.obj['NAME']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # Instantiate the config object
    config = ExclConfig(config_file)

    # take name from config if not default
    if config.name.lower() != 'rev':
        name = config.name
        ctx.obj['NAME'] = config.name

    # Enforce verbosity if logging level is specified in the config
    if config.logging_level == logging.DEBUG:
        verbose = True

    # initialize loggers. Not SAM (will be logged in the invoked processes).
    init_excl_loggers(verbose, name, logdir=config.logdir)

    # Initial log statements
    logger.info('Running reV 2.0 exclusions from config file: "{}"'
                .format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))
    logger.info('The following exclusion layers are available:\n{}'
                .format(pprint.pformat(config.get('exclusions', None),
                                       indent=4)))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    exclusions = Exclusions(config['exclusions'])
    exclusions.export(fname='exclusions.tif')

    return None
