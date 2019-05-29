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
from reV.utilities.execution import SLURM


logger = logging.getLogger(__name__)


@click.command()
@click.option('--name', '-n', default='reV_excl', type=STR,
              help='Exclusion job name. Default is "reV_excl".')
@click.option('--config_file', '-c', required=True, type=STR,
              help='reV exclusion configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
def main(name, config_file, verbose):
    """Command line interface (CLI) for the reV 2.0 Exclusion Module."""

    # Instantiate the config object
    config = ExclConfig(config_file)

    # take name from config if not default
    if config.name.lower() != 'rev':
        name = config.name

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

    use_blocks = False
    if 'exclusions_use_blocks' in config:
        use_blocks = config['exclusions_output_file']

    contiguous_filter = None
    if 'exclusions_filter' in config:
        contiguous_filter = config['exclusions_filter']

    Exclusions.run(config=config['exclusions'],
                   output_fname=fname,
                   use_blocks=use_blocks,
                   contiguous_filter=contiguous_filter)

    return None


@click.command()
@click.option('--name', '-n', default='reV_gen', type=STR,
              help='Exclusion job name. Default is "reV_gen".')
@click.option('--config_file', '-c', required=True, type=STR,
              help='reV exclusion configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
def submit_eagle(name, config_file, verbose):
    """Run exclusions on Eagle HPC via SLURM job submission."""

    # Instantiate the config object
    config = ExclConfig(config_file)

    # Enforce verbosity if logging level is specified in the config
    if config.logging_level == logging.DEBUG:
        verbose = True

    # initialize loggers
    init_mult(verbose, name, logdir=config.logdir)

    # Python command that will be executed on a node
    if verbose:
        verbose_flag = '-v'
    else:
        verbose_flag = ''
    cmd = ('python -m reV.exclusions.cli_excl main'
           '-n {name} -c {config_file} {verbose_flag}'
           .format(name=name,
                   config_file=config_file,
                   verbose_flag=verbose_flag))
    logger.debug('Creating the following command line call:\n\t{}'.format(cmd))
    logger.info('Running reV exclusions on Eagle with node name "{}"'
                .format(config.name))

    # create and submit the SLURM job
    slurm = SLURM(cmd,
                  alloc=config.execution_control.allocation,
                  walltime=config.execution_control.walltime,
                  name=config.name,
                  stdout_path=os.path.join(config.logdir, 'stdout'))
    if slurm.id:
        msg = ('Kicked off reV exclusions job "{}" (SLURM jobid #{}) on '
               'Eagle.'.format(config.name, slurm.id))
    else:
        msg = ('Was unable to kick off reV exclusions job "{}". '
               'Please see the stdout error messages'
               .format(config.name))
    click.echo(msg)
    logger.info(msg)

    return slurm
