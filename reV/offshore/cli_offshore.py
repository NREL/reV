# -*- coding: utf-8 -*-
"""
reV offshore wind farm aggregation module command line interface (CLI).

This module aggregates offshore data from high res wind resource data to
coarse wind farm sites and then calculates the ORCA econ data.

Offshore resource / generation data refers to WTK 2km (fine resolution)
Offshore farms refer to ORCA data on 600MW wind farms (coarse resolution)
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

from rex.utilities.cli_dtypes import STR, INT
from rex.utilities.loggers import init_mult
from rex.utilities.execution import SLURM

logger = logging.getLogger(__name__)


@click.group()
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

    # Enforce verbosity if logging level is specified in the config
    if config.log_level == logging.DEBUG:
        verbose = True

    # initialize loggers
    init_mult(name, config.logdir, modules=[__name__, 'reV.config',
                                            'reV.utilities', 'rex.utilities'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV offshore aggregation from config '
                'file: "{}"'.format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    for i, gen_fpath in enumerate(config.gen_fpaths):
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
                ctx.invoke(main, job_name, gen_fpath, config.offshore_fpath,
                           config.project_points, config.sam_files,
                           config.logdir, verbose)

        elif config.execution_control.option in ('eagle', 'slurm'):

            ctx.obj['NAME'] = job_name
            ctx.obj['GEN_FPATH'] = gen_fpath
            ctx.obj['OFFSHORE_FPATH'] = config.offshore_fpath
            ctx.obj['PROJECT_POINTS'] = config.project_points
            ctx.obj['SAM_FILES'] = config.sam_files
            ctx.obj['OUT_DIR'] = config.dirout
            ctx.obj['LOG_DIR'] = config.logdir
            ctx.obj['VERBOSE'] = verbose

            ctx.invoke(slurm,
                       alloc=config.execution_control.alloc,
                       memory=config.execution_control.node_mem,
                       walltime=config.execution_control.walltime,
                       feature=config.execution_control.feature,
                       module=config.execution_control.module,
                       conda_env=config.execution_control.conda_env)


@main.group(invoke_without_command=True)
@click.option('--gen_fpath', '-gf', type=STR, required=True,
              help='reV wind generation/econ output file.')
@click.option('--offshore_fpath', '-of', type=STR, required=True,
              help='reV wind farm meta and ORCA cost data inputs.')
@click.option('--points', '-pp', default=slice(0, 100), type=PROJECTPOINTS,
              help=('reV project points to analyze '
                    '(slice, list, or file string). '
                    'Default is slice(0, 100)'))
@click.option('--sam_files', '-sf', required=True, type=SAMFILES,
              help='SAM config files (required) (str, dict, or list).')
@click.option('--log_dir', '-ld', type=STR, default='./logs/',
              help='Directory to save offshore logs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def direct(ctx, gen_fpath, offshore_fpath, points, sam_files,
           log_dir, verbose):
    """Main entry point to run offshore wind aggregation"""
    name = ctx.obj['NAME']
    ctx.obj['GEN_FPATH'] = gen_fpath
    ctx.obj['OFFSHORE_FPATH'] = offshore_fpath
    ctx.obj['POINTS'] = points
    ctx.obj['SAM_FILES'] = sam_files
    ctx.obj['OUT_DIR'] = os.path.dirname(gen_fpath)
    ctx.obj['LOG_DIR'] = log_dir
    ctx.obj['VERBOSE'] = verbose

    if ctx.invoked_subcommand is None:
        t0 = time.time()
        init_mult(name, log_dir, modules=[__name__, 'reV.offshore',
                                          'reV.handlers', 'rex'],
                  verbose=verbose, node=True)

        fpath_out = gen_fpath.replace('.h5', '_offshore.h5')

        try:
            Offshore.run(gen_fpath, offshore_fpath, points, sam_files,
                         fpath_out=fpath_out)
        except Exception as e:
            logger.exception('Offshore module failed, received the '
                             'following exception:\n{}'.format(e))
            raise e

        runtime = (time.time() - t0) / 60

        status = {'dirout': os.path.dirname(fpath_out),
                  'fout': os.path.basename(fpath_out),
                  'job_status': 'successful',
                  'runtime': runtime, 'finput': gen_fpath}
        Status.make_job_file(os.path.dirname(fpath_out), 'offshore',
                             name, status)


def get_node_cmd(name, gen_fpath, offshore_fpath, points, sam_files,
                 log_dir, verbose):
    """Get a CLI call command for the offshore aggregation cli."""

    args = ('-n {name} '
            '-gf {gen_fpath} '
            '-of {offshore_fpath} '
            '-pp {points} '
            '-sf {sam_files} '
            '-ld {log_dir} '
            )

    args = args.format(name=SLURM.s(name),
                       gen_fpath=SLURM.s(gen_fpath),
                       offshore_fpath=SLURM.s(offshore_fpath),
                       points=SLURM.s(points),
                       sam_files=SLURM.s(sam_files),
                       log_dir=SLURM.s(log_dir),
                       )

    if verbose:
        args += '-v '

    cmd = 'python -m reV.offshore.cli_offshore {}'.format(args)
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
    log_dir = ctx.obj['LOG_DIR']
    out_dir = ctx.obj['OUT_DIR']
    verbose = ctx.obj['VERBOSE']

    if stdout_path is None:
        stdout_path = os.path.join(log_dir, 'stdout/')

    cmd = get_node_cmd(name, gen_fpath, offshore_fpath, project_points,
                       sam_files, log_dir, verbose)

    status = Status.retrieve_job_status(out_dir, 'offshore', name)
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'
               .format(name, out_dir))
    else:
        logger.info('Running reV offshore aggregation on SLURM with '
                    'node name "{}"'.format(name))
        slurm = SLURM(cmd, alloc=alloc, memory=memory,
                      walltime=walltime, feature=feature,
                      name=name, stdout_path=stdout_path, conda_env=conda_env,
                      module=module)
        if slurm.id:
            msg = ('Kicked off reV offshore job "{}" '
                   '(SLURM jobid #{}).'
                   .format(name, slurm.id))
            Status.add_job(
                out_dir, 'offshore', name, replace=True,
                job_attrs={'job_id': slurm.id, 'hardware': 'eagle',
                           'fout': '{}.csv'.format(name), 'dirout': out_dir})
        else:
            msg = ('Was unable to kick off reV offshore job "{}". Please see '
                   'the stdout error messages'.format(name))
    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV offshore CLI.')
        raise
