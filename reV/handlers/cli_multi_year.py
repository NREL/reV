# -*- coding: utf-8 -*-
"""
Multi-year means CLI entry points.
"""
import click
import json
import logging
import os
import time

from reV.config.multi_year import MultiYearConfig
from reV.handlers.multi_year import MultiYear
from reV.pipeline.status import Status

from rex.utilities.cli_dtypes import STR, STRLIST, PATHLIST, INT
from rex.utilities.loggers import init_mult
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties

logger = logging.getLogger(__name__)


@click.group()
@click.option('--name', '-n', default='reV_multi-year', type=str,
              help='Multi-year job name. Default is "reV_multi-year".')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """reV Multi-Year Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
def valid_config_keys():
    """
    Echo the valid MultiYear config keys
    """
    click.echo(', '.join(get_class_properties(MultiYearConfig)))


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV multi-year configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV gen from a config file."""
    name = ctx.obj['NAME']

    # Instantiate the config object
    config = MultiYearConfig(config_file)

    # take name from config if not default
    if config.name.lower() != 'rev':
        name = config.name
        ctx.obj['NAME'] = name

    # Enforce verbosity if logging level is specified in the config
    if config.log_level == logging.DEBUG:
        verbose = True

    # make output directory if does not exist
    if not os.path.exists(config.dirout):
        os.makedirs(config.dirout)

    # initialize loggers.
    init_mult(name, config.logdir,
              modules=[__name__, 'reV.handlers.multi_year'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV multi-year from config file: "{}"'
                .format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))

    ctx.obj['MY_FILE'] = config.my_file
    if config.execution_control.option == 'local':

        ctx.obj['NAME'] = name
        status = Status.retrieve_job_status(config.dirout, 'multi-year', name)
        if status != 'successful':
            Status.add_job(
                config.dirout, 'multi-year', name, replace=True,
                job_attrs={'hardware': 'local',
                           'fout': ctx.obj['MY_FILE'],
                           'dirout': config.dirout})
            group_params = json.dumps(config.group_params)
            ctx.invoke(multi_year_groups, group_params=group_params)

    elif config.execution_control.option in ('eagle', 'slurm'):
        ctx.obj['NAME'] = name
        ctx.invoke(multi_year_slurm,
                   alloc=config.execution_control.allocation,
                   walltime=config.execution_control.walltime,
                   feature=config.execution_control.feature,
                   memory=config.execution_control.memory,
                   conda_env=config.execution_control.conda_env,
                   module=config.execution_control.module,
                   stdout_path=os.path.join(config.logdir, 'stdout'),
                   group_params=json.dumps(config.group_params),
                   verbose=verbose)


@main.group()
@click.option('--my_file', '-f', required=True, type=click.Path(),
              help='h5 file to use for multi-year collection.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def direct(ctx, my_file, verbose):
    """Main entry point for collection with context passing."""
    ctx.obj['MY_FILE'] = my_file
    ctx.obj['VERBOSE'] = verbose


@direct.command()
@click.option('--group', '-g', type=STR, default=None,
              help=('Group to collect into. Useful for collecting multiple '
                    'scenarios into a single file.'))
@click.option('--source_files', '-sf', required=True, type=PATHLIST,
              help='List of files to collect from.')
@click.option('--dsets', '-ds', required=True, type=STRLIST,
              help=('Dataset names to be collected. If means, multi-year '
                    'means will be computed.'))
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def multi_year(ctx, group, source_files, dsets, verbose):
    """Run multi year collection and means on local worker."""

    name = ctx.obj['NAME']
    my_file = ctx.obj['MY_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # initialize loggers for multiple modules
    log_dir = os.path.dirname(my_file)
    init_mult(name, log_dir, modules=[__name__, 'reV.handlers.multi_year'],
              verbose=verbose, node=True)

    for key, val in ctx.obj.items():
        logger.debug('ctx var passed to collection method: "{}" : "{}" '
                     'with type "{}"'.format(key, val, type(val)))

    logger.info('Multi-year collection is being run for "{}" '
                'with job name "{}" on {}. Target output path is: {}'
                .format(dsets, name, source_files, my_file))
    t0 = time.time()

    for dset in dsets:
        if MultiYear.is_profile(source_files, dset):
            MultiYear.collect_profiles(my_file, source_files, dset,
                                       group=group)
        else:
            MultiYear.collect_means(my_file, source_files, dset,
                                    group=group)

    runtime = (time.time() - t0) / 60
    logger.info('Multi-year collection completed in: {:.2f} min.'
                .format(runtime))

    # add job to reV status file.
    status = {'dirout': os.path.dirname(my_file),
              'fout': os.path.basename(my_file),
              'job_status': 'successful',
              'runtime': runtime,
              'finput': source_files}
    Status.make_job_file(os.path.dirname(my_file), 'multi-year', name,
                         status)


@direct.command()
@click.option('--group_params', '-gp', required=True, type=str,
              help=('List of groups and their parameters'
                    '(group, source_files, dsets) to collect'))
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def multi_year_groups(ctx, group_params, verbose):
    """Run multi year collection and means for multiple groups."""
    name = ctx.obj['NAME']
    my_file = ctx.obj['MY_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # initialize loggers for multiple modules
    log_dir = os.path.dirname(my_file)
    init_mult(name, log_dir, modules=[__name__, 'reV.handlers.multi_year'],
              verbose=verbose, node=True)

    for key, val in ctx.obj.items():
        logger.debug('ctx var passed to collection method: "{}" : "{}" '
                     'with type "{}"'.format(key, val, type(val)))

    logger.info('Multi-year collection is being run with job name "{}". '
                'Target output path is: {}'
                .format(name, my_file))
    ts = time.time()
    for group_name, group in json.loads(group_params).items():
        logger.info('- Collecting datasets "{}" from "{}" into "{}/"'
                    .format(group['dsets'], group['source_files'],
                            group_name))
        t0 = time.time()
        for dset in group['dsets']:
            if MultiYear.is_profile(group['source_files'], dset):
                MultiYear.collect_profiles(my_file, group['source_files'],
                                           dset, group=group['group'])
            else:
                MultiYear.collect_means(my_file, group['source_files'],
                                        dset, group=group['group'])

        runtime = (time.time() - t0) / 60
        logger.info('- {} collection completed in: {:.2f} min.'
                    .format(group_name, runtime))

    runtime = (time.time() - ts) / 60
    logger.info('Multi-year collection completed in : {:.2f} min.'
                .format(runtime))

    # add job to reV status file.
    status = {'dirout': os.path.dirname(my_file),
              'fout': os.path.basename(my_file),
              'job_status': 'successful',
              'runtime': runtime}
    Status.make_job_file(os.path.dirname(my_file), 'multi-year', name,
                         status)


def get_slurm_cmd(name, my_file, group_params, verbose=False):
    """Make a reV multi-year collection local CLI call string.

    Parameters
    ----------
    name : str
        reV collection jobname.
    my_file : str
        Path to .h5 file to use for multi-year collection.
    group_params : list
        List of groups and their parameters to collect
    verbose : bool
        Flag to turn on DEBUG logging

    Returns
    -------
    cmd : str
        Argument to call the neccesary CLI calls on the node to collect
        desired groups
    """
    # make a cli arg string for direct() in this module
    main_args = ['-n {}'.format(SLURM.s(name))]

    if verbose:
        main_args.append('-v')

    direct_args = '-f {}'.format(SLURM.s(my_file))

    collect_args = '-gp {}'.format(SLURM.s(group_params))

    # Python command that will be executed on a node
    # command strings after cli v7.0 use dashes instead of underscores
    cmd = ('python -m reV.handlers.cli_multi_year {} direct {} '
           'multi-year-groups {}'
           .format(' '.join(main_args), direct_args, collect_args))
    logger.debug('Creating the following command line call:\n\t{}'
                 .format(cmd))

    return cmd


@direct.command()
@click.option('--alloc', '-a', default='rev', type=str,
              help='SLURM allocation account name. Default is "rev".')
@click.option('--walltime', '-wt', default=4.0, type=float,
              help='SLURM walltime request in hours. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--memory', '-mem', default=None, type=INT,
              help='SLURM node memory request in GB. Default is None')
@click.option('--conda_env', '-env', default=None, type=STR,
              help='Conda env to activate')
@click.option('--module', '-mod', default=None, type=STR,
              help='Module to load')
@click.option('--stdout_path', '-sout', default='./out/stdout', type=str,
              help='Subprocess standard output path. Default is ./out/stdout')
@click.option('--group_params', '-gp', required=True, type=str,
              help=('List of groups and their parameters'
                    '(group, source_files, dsets) to collect'))
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def multi_year_slurm(ctx, alloc, walltime, feature, memory, conda_env,
                     module, stdout_path, group_params, verbose):
    """
    Run multi year collection and means on HPC via SLURM job submission.
    """

    name = ctx.obj['NAME']
    my_file = ctx.obj['MY_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    slurm_manager = ctx.obj.get('SLURM_MANAGER', None)
    if slurm_manager is None:
        slurm_manager = SLURM()
        ctx.obj['SLURM_MANAGER'] = slurm_manager

    status = Status.retrieve_job_status(os.path.dirname(my_file), 'multi-year',
                                        name, hardware='eagle',
                                        subprocess_manager=slurm_manager)

    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'
               .format(name, os.path.dirname(my_file)))
    elif 'fail' not in str(status).lower() and status is not None:
        msg = ('Job "{}" was found with status "{}", not resubmitting'
               .format(name, status))
    else:
        logger.info('Running reV multi-year collection on SLURM with node '
                    ' name "{}", collecting into "{}".'
                    .format(name, my_file))
        # create and submit the SLURM job
        slurm_cmd = get_slurm_cmd(name, my_file, group_params, verbose=verbose)
        out = slurm_manager.sbatch(slurm_cmd, alloc=alloc, memory=memory,
                                   walltime=walltime, feature=feature,
                                   name=name, stdout_path=stdout_path,
                                   conda_env=conda_env, module=module)[0]
        if out:
            msg = ('Kicked off reV multi-year collection job "{}" '
                   '(SLURM jobid #{}).'.format(name, out))
            # add job to reV status file.
            Status.add_job(
                os.path.dirname(my_file), 'multi-year', name, replace=True,
                job_attrs={'job_id': out, 'hardware': 'eagle',
                           'fout': os.path.basename(my_file),
                           'dirout': os.path.dirname(my_file)})

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Multi-Year CLI')
        raise
