# -*- coding: utf-8 -*-
"""
Multi-year means CLI entry points.
"""
import click
import logging
import os
import time

from reV.config.multi_year import MultiYearConfig
from reV.handlers.multi_year import MultiYear
from reV.pipeline.status import Status
from reV.utilities.cli_dtypes import STR, STRLIST, PATHLIST
from reV.utilities.loggers import init_mult
from reV.utilities.execution import SubprocessManager, SLURM

logger = logging.getLogger(__name__)


@click.command()
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
    if config.logging_level == logging.DEBUG:
        verbose = True

    # make output directory if does not exist
    if not os.path.exists(config.dirout):
        os.makedirs(config.dirout)

    # initialize loggers.
    init_mult(name, config.logdir,
              modules=[__name__, 'reV.handlers.multi_year'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV 2.0 multi-year from config file: "{}"'
                .format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))

    for group_name, group in config.group_params.items():
        # set config objects to be passed through invoke to direct methods
        ctx.obj['NAME'] = "{}-{}".format(config.name, group_name)
        ctx.obj['MY_FILE'] = config.my_file
        ctx.obj['SOURCE_FILES'] = group.source_files
        ctx.obj['DSETS'] = group.dsets
        ctx.obj['GROUP'] = group.name

        if config.execution_control.option == 'local':
            ctx.invoke(collect)
        elif config.execution_control.option == 'eagle':
            ctx.invoke(collect_eagle,
                       alloc=config.execution_control.alloc,
                       walltime=config.execution_control.walltime,
                       feature=config.execution_control.feature,
                       memory=config.execution_control.node_mem,
                       stdout_path=os.path.join(config.logdir, 'stdout'),
                       verbose=verbose)


@click.group()
@click.option('--name', '-n', default='reV_multi-year', type=str,
              help='Multi-year job name. Default is "reV_multi-year".')
@click.option('--my_file', '-f', required=True, type=click.Path(),
              help='h5 file to use for multi-year collection.')
@click.option('--source_files', '-sf', required=True, type=PATHLIST,
              help='List of files to collect from.')
@click.option('--dsets', '-ds', required=True, type=STRLIST,
              help=('Dataset names to be collected. If means, multi-year '
                    'means will be computed.'))
@click.option('--group', '-g', type=STR, default=None,
              help=('Group to collect into. Useful for collecting multiple '
                    'scenarios into a single file.'))
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def main(ctx, name, my_file, source_files, dsets, group, verbose):
    """Main entry point for collection with context passing."""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['MY_FILE'] = my_file
    ctx.obj['SOURCE_FILES'] = source_files
    ctx.obj['DSETS'] = dsets
    ctx.obj['GROUP'] = group
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def collect(ctx, verbose):
    """Run collection on local worker."""

    name = ctx.obj['NAME']
    my_file = ctx.obj['MY_FILE']
    source_files = ctx.obj['SOURCE_FILES']
    dsets = ctx.obj['DSETS']
    group = ctx.obj['GROUP']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # initialize loggers for multiple modules
    log_dir = os.path.basename(my_file)
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
        if 'mean' in dset:
            MultiYear.collect_means(my_file, source_files, dset, group=group)
        else:
            MultiYear.collect_profiles(my_file, source_files, dset,
                                       group=group)
    runtime = (time.time() - t0) / 60
    logger.info('Multi-year collection completed in: {:.2f} min.'
                .format(runtime))

    # add job to reV status file.
    status = {'dirout': os.path.dirname(my_file),
              'fout': os.path.basename(my_file), 'job_status': 'successful',
              'runtime': runtime,
              'finput': source_files}
    Status.make_job_file(os.path.dirname(my_file), 'multi-year', name, status)


def get_node_cmd(name, my_file, source_files, dsets, group=None,
                 verbose=False):
    """Make a reV multi-year collection local CLI call string.

    Parameters
    ----------
    name : str
        reV collection jobname.
    my_file : str
        Path to .h5 file to use for multi-year collection.
    source_files : list
        Root directory containing .h5 files to combine
    dsets : list
        List of datasets (strings) to be collected.
    group : str | NoneType
        Group to collect and compute multi-year means into.
        Usefull when collecting multiple scenarios
    verbose : bool
        Flag to turn on DEBUG logging

    Returns
    -------
    cmd : str
        Single line command line argument to call the following CLI with
        appropriately formatted arguments based on input args:
            python -m reV.handlers.cli_collect [args] collect
    """

    # make a cli arg string for direct() in this module
    arg_main = ('-n {name} '
                '-f {my_file} '
                '-sf {h5_dir} '
                '-ds {dsets} '
                '-g {group} '
                '{v}'
                .format(name=SubprocessManager.s(name),
                        my_file=SubprocessManager.s(my_file),
                        h5_dir=SubprocessManager.s(source_files),
                        dsets=SubprocessManager.s(dsets),
                        group=SubprocessManager.s(group),
                        v='-v ' if verbose else '',
                        ))

    # Python command that will be executed on a node
    # command strings after cli v7.0 use dashes instead of underscores
    cmd = ('python -m reV.handlers.cli_multi_year {arg_main} collect'
           .format(arg_main=arg_main))
    logger.debug('Creating the following command line call:\n\t{}'.format(cmd))
    return cmd


@main.command()
@click.option('--alloc', '-a', default='rev', type=str,
              help='Eagle allocation account name. Default is "rev".')
@click.option('--walltime', '-wt', default=4.0, type=float,
              help='Eagle walltime request in hours. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--memory', '-mem', default=90, type=int,
              help='Eagle node memory request in GB. Default is 90')
@click.option('--stdout_path', '-sout', default='./out/stdout', type=str,
              help='Subprocess standard output path. Default is ./out/stdout')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def collect_eagle(ctx, alloc, walltime, feature, memory, stdout_path, verbose):
    """Run collection on Eagle HPC via SLURM job submission."""

    name = ctx.obj['NAME']
    my_file = ctx.obj['MY_FILE']
    source_files = ctx.obj['SOURCE_FILES']
    dsets = ctx.obj['DSETS']
    group = ctx.obj['GROUP']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    cmd = get_node_cmd(name, my_file, source_files, dsets, group=group,
                       verbose=verbose)

    status = Status.retrieve_job_status(os.path.dirname(my_file), 'multi-year',
                                        name)
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'
               .format(name, os.path.dirname(my_file)))
    else:
        logger.info('Running reV multi-year collection on Eagle with node '
                    ' name "{}", collecting "{}" to "{}" from "{}".'
                    .format(name, dsets, my_file, source_files))
        # create and submit the SLURM job
        slurm = SLURM(cmd, alloc=alloc, memory=memory, walltime=walltime,
                      feature=feature, name=name, stdout_path=stdout_path)
        if slurm.id:
            msg = ('Kicked off reV multi-year collection job "{}" '
                   '(SLURM jobid #{}) on Eagle.'.format(name, slurm.id))
            # add job to reV status file.
            Status.add_job(
                os.path.dirname(my_file), 'multi-year', name, replace=True,
                job_attrs={'job_id': slurm.id, 'hardware': 'eagle',
                           'fout': os.path.basename(my_file),
                           'dirout': os.path.dirname(my_file)})
        else:
            msg = ('Was unable to kick off reV collection job "{}". '
                   'Please see the stdout error messages'
                   .format(name))
    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    main(obj={})
