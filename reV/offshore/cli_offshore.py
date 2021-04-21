# -*- coding: utf-8 -*-
# pylint: disable=all
"""
reV offshore wind module command line interface (CLI).

This module uses the NRWAL library to assess offshore losses and LCOE to
complement the simple SAM windpower module.

Everything in this module operates on the native wind resource resolution.
"""
import pprint
import os
import click
import logging
import time

from reV.config.offshore_config import OffshoreConfig
from reV.pipeline.status import Status
from reV.offshore.offshore import Offshore
from reV.utilities.cli_dtypes import SAMFILES, PROJECTPOINTS
from reV import __version__

from rex.utilities.cli_dtypes import STR, INT, STRLIST
from rex.utilities.loggers import init_mult
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default='reV-off', type=STR,
              help='Job name. Default is "reV-off".')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """reV Offshore Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
def valid_config_keys():
    """
    Echo the valid Offshore config keys
    """
    click.echo(', '.join(get_class_properties(OffshoreConfig)))


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV offshore configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV offshore aggregation from a config file."""
    # Instantiate the config object
    config = OffshoreConfig(config_file)
    name = ctx.obj['NAME']

    # take name from config if not default
    if config.name.lower() != 'rev':
        name = config.name
        ctx.obj['NAME'] = name

    # Enforce verbosity if logging level is specified in the config
    if config.log_level == logging.DEBUG:
        verbose = True

    # initialize loggers
    init_mult(name, config.logdir, modules=[__name__, 'reV', 'rex'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV offshore aggregation from config '
                'file: "{}"'.format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    for i, gen_fpath in enumerate(config.parse_gen_fpaths()):
        job_name = '{}_{}'.format(name, str(i).zfill(2))

        if config.execution_control.option == 'local':
            status = Status.retrieve_job_status(config.dirout, 'offshore',
                                                job_name)
            if status != 'successful':
                Status.add_job(
                    config.dirout, 'offshore', job_name, replace=True,
                    job_attrs={'hardware': 'local',
                               'fout': '{}_offshore.h5'.format(job_name),
                               'dirout': config.dirout,
                               'finput': gen_fpath})
                ctx.invoke(direct,
                           gen_fpath=gen_fpath,
                           offshore_fpath=config.offshore_fpath,
                           points=config.project_points,
                           sam_files=config.sam_files,
                           nrwal_configs=config.nrwal_configs,
                           offshore_meta_cols=config.offshore_meta_cols,
                           offshore_nrwal_keys=config.offshore_nrwal_keys,
                           logdir=config.logdir,
                           verbose=verbose)

        elif config.execution_control.option in ('eagle', 'slurm'):
            ctx.obj['NAME'] = job_name
            ctx.obj['GEN_FPATH'] = gen_fpath
            ctx.obj['OFFSHORE_FPATH'] = config.offshore_fpath
            ctx.obj['PROJECT_POINTS'] = config.project_points
            ctx.obj['SAM_FILES'] = config.sam_files
            ctx.obj['NRWAL_CONFIGS'] = config.nrwal_configs
            ctx.obj['OFFSHORE_META_COLS'] = config.offshore_meta_cols
            ctx.obj['OFFSHORE_NRWAL_KEYS'] = config.offshore_nrwal_keys
            ctx.obj['OUT_DIR'] = config.dirout
            ctx.obj['LOG_DIR'] = config.logdir
            ctx.obj['VERBOSE'] = verbose

            ctx.invoke(slurm,
                       alloc=config.execution_control.allocation,
                       memory=config.execution_control.memory,
                       walltime=config.execution_control.walltime,
                       feature=config.execution_control.feature,
                       module=config.execution_control.module,
                       conda_env=config.execution_control.conda_env)


@main.group(invoke_without_command=True)
@click.option('--gen_fpath', '-gf', type=STR, required=True,
              help='reV wind generation/econ output file.')
@click.option('--offshore_fpath', '-of', type=STR, required=True,
              help='Offshore wind geospatial inputs such as depth and '
              'distance to port. Needs "gid" and "config" columns matching '
              'the project points input.')
@click.option('--points', '-pp', required=True, type=PROJECTPOINTS,
              help='reV project points to analyze. Has to be a string file '
              'path to a project points csv with "gid" and "config" columns. '
              'The config column maps to the sam_files and nrwal_configs '
              'inputs.')
@click.option('--sam_files', '-sf', required=True, type=SAMFILES,
              help='SAM config files lookup mapping config keys to config '
              'filepaths. (required) (dict). Should have the same config '
              'keys as the nrwal_configs input.')
@click.option('--nrwal_configs', '-nc', required=True, type=SAMFILES,
              help='NRWAL config files lookup mapping config keys to config '
              'filepaths. (required) (dict). Should have the same config '
              'keys as the sam_files input.')
@click.option('--offshore_meta_cols', '-mc', default=None, type=STRLIST,
              help='Column labels from offshore_fpath to pass through to the '
              'output meta data. None (default) will use class variable '
              'DEFAULT_META_COLS, and any additional cols requested here will '
              'be added to DEFAULT_META_COLS.')
@click.option('--offshore_nrwal_keys', '-nk', default=None, type=STRLIST,
              help='Keys from the offshore nrwal configs to pass through as '
              'new datasets in the reV output h5.')
@click.option('--log_dir', '-ld', type=STR, default='./logs/',
              help='Directory to save offshore logs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def direct(ctx, gen_fpath, offshore_fpath, points, sam_files, nrwal_configs,
           offshore_meta_cols, offshore_nrwal_keys, log_dir, verbose):
    """Main entry point to run offshore wind aggregation"""
    name = ctx.obj['NAME']
    ctx.obj['GEN_FPATH'] = gen_fpath
    ctx.obj['OFFSHORE_FPATH'] = offshore_fpath
    ctx.obj['POINTS'] = points
    ctx.obj['SAM_FILES'] = sam_files
    ctx.obj['NRWAL_CONFIGS'] = nrwal_configs
    ctx.obj['OFFSHORE_META_COLS'] = offshore_meta_cols
    ctx.obj['OFFSHORE_NRWAL_KEYS'] = offshore_nrwal_keys
    ctx.obj['OUT_DIR'] = os.path.dirname(gen_fpath)
    ctx.obj['LOG_DIR'] = log_dir
    ctx.obj['VERBOSE'] = verbose

    if ctx.invoked_subcommand is None:
        t0 = time.time()
        init_mult(name, log_dir, modules=[__name__, 'reV', 'rex'],
                  verbose=verbose, node=True)

        try:
            Offshore.run(gen_fpath, offshore_fpath, sam_files, nrwal_configs,
                         points, offshore_meta_cols=offshore_meta_cols,
                         offshore_nrwal_keys=offshore_nrwal_keys)
        except Exception as e:
            logger.exception('Offshore module failed, received the '
                             'following exception:\n{}'.format(e))
            raise e

        runtime = (time.time() - t0) / 60

        status = {'dirout': os.path.dirname(gen_fpath),
                  'fout': os.path.basename(gen_fpath),
                  'job_status': 'successful',
                  'runtime': runtime, 'finput': gen_fpath}
        Status.make_job_file(os.path.dirname(gen_fpath), 'offshore',
                             name, status)


def get_node_cmd(name, gen_fpath, offshore_fpath, points, sam_files,
                 nrwal_configs, offshore_meta_cols, offshore_nrwal_keys,
                 log_dir, verbose):
    """Get a CLI call command for the offshore aggregation cli."""

    args = ['-gf {}'.format(SLURM.s(gen_fpath)),
            '-of {}'.format(SLURM.s(offshore_fpath)),
            '-pp {}'.format(SLURM.s(points)),
            '-sf {}'.format(SLURM.s(sam_files)),
            '-nc {}'.format(SLURM.s(nrwal_configs)),
            '-mc {}'.format(SLURM.s(offshore_meta_cols)),
            '-nk {}'.format(SLURM.s(offshore_nrwal_keys)),
            '-ld {}'.format(SLURM.s(log_dir)),
            ]

    if verbose:
        args.append('-v')

    cmd = ('python -m reV.offshore.cli_offshore -n {} direct {}'
           .format(SLURM.s(name), ' '.join(args)))
    logger.debug('Creating the following command line call:\n\t{}'.format(cmd))

    return cmd


@direct.command()
@click.option('--alloc', '-a', required=True, type=STR,
              help='SLURM allocation account name.')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--memory', '-mem', default=None, type=INT, help='SLURM node '
              'memory request in GB. Default is None')
@click.option('--walltime', '-wt', default=1.0, type=float,
              help='SLURM walltime request in hours. Default is 1.0')
@click.option('--module', '-mod', default=None, type=STR,
              help='Module to load')
@click.option('--conda_env', '-env', default=None, type=STR,
              help='Conda env to activate')
@click.option('--stdout_path', '-sout', default=None, type=STR,
              help='Subprocess standard output path. Default is in out_dir.')
@click.pass_context
def slurm(ctx, alloc, feature, memory, walltime, module, conda_env,
          stdout_path):
    """slurm (Eagle) submission tool for reV supply curve aggregation."""

    name = ctx.obj['NAME']
    gen_fpath = ctx.obj['GEN_FPATH']
    offshore_fpath = ctx.obj['OFFSHORE_FPATH']
    project_points = ctx.obj['PROJECT_POINTS']
    sam_files = ctx.obj['SAM_FILES']
    nrwal_configs = ctx.obj['NRWAL_CONFIGS']
    offshore_meta_cols = ctx.obj['OFFSHORE_META_COLS']
    offshore_nrwal_keys = ctx.obj['OFFSHORE_NRWAL_KEYS']
    log_dir = ctx.obj['LOG_DIR']
    out_dir = ctx.obj['OUT_DIR']
    verbose = ctx.obj['VERBOSE']

    if stdout_path is None:
        stdout_path = os.path.join(log_dir, 'stdout/')

    cmd = get_node_cmd(name, gen_fpath, offshore_fpath, project_points,
                       sam_files, nrwal_configs, offshore_meta_cols,
                       offshore_nrwal_keys, log_dir, verbose)
    slurm_manager = ctx.obj.get('SLURM_MANAGER', None)
    if slurm_manager is None:
        slurm_manager = SLURM()
        ctx.obj['SLURM_MANAGER'] = slurm_manager

    status = Status.retrieve_job_status(out_dir, 'offshore', name,
                                        hardware='eagle',
                                        subprocess_manager=slurm_manager)

    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'
               .format(name, out_dir))
    elif 'fail' not in str(status).lower() and status is not None:
        msg = ('Job "{}" was found with status "{}", not resubmitting'
               .format(name, status))
    else:
        logger.info('Running reV offshore aggregation on SLURM with '
                    'node name "{}"'.format(name))
        out = slurm_manager.sbatch(cmd,
                                   alloc=alloc,
                                   memory=memory,
                                   walltime=walltime,
                                   feature=feature,
                                   name=name,
                                   stdout_path=stdout_path,
                                   conda_env=conda_env,
                                   module=module)[0]
        if out:
            msg = ('Kicked off reV offshore job "{}" '
                   '(SLURM jobid #{}).'
                   .format(name, out))
            Status.add_job(
                out_dir, 'offshore', name, replace=True,
                job_attrs={'job_id': out, 'hardware': 'eagle',
                           'fout': '{}.csv'.format(name), 'dirout': out_dir})

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV offshore CLI.')
        raise
