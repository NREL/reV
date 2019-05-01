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
from reV.utilities.loggers import init_mult


logger = logging.getLogger(__name__)


@click.command()
@click.option('--name', '-n', default='reV_gen', type=STR,
              help='Exclusion job name. Default is "reV_gen".')
@click.option('--config_file', '-c', required=True, type=STR,
              help='reV exclusion configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, config_file, verbose):
    """Command line interface (CLI) for the reV 2.0 Exclusion Module."""
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose

    # Instantiate the config object
    config = ExclConfig(config_file)

    # take name from config if not default
    if config.name.lower() != 'rev':
        name = config.name
        ctx.obj['NAME'] = config.name

    # Enforce verbosity if logging level is specified in the config
    if config.logging_level == logging.DEBUG:
        verbose = True

    # initialize loggers
    init_mult(verbose, name, logdir=config.logdir)

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

    if 'exclusions_output_file' in config:
        fname = os.path.join(config.dirout, config['exclusions_output_file'])
    else:
        fname = os.path.join(config.dirout, 'exclusions.tif')

    exclusions = Exclusions(config['exclusions'])
    exclusions.build_from_config()
    exclusions.export(fname=fname)

    return None
