# -*- coding: utf-8 -*-
"""
File collection CLI entry points.
"""
import click
import logging
import os
import pprint
import time

from reV.config.collection import CollectionConfig
from reV.handlers.collection import Collector
from reV.pipeline.status import Status

from rex.utilities.cli_dtypes import STR, STRLIST, INT
from rex.utilities.execution import SubprocessManager, SLURM
from rex.utilities.loggers import init_mult

logger = logging.getLogger(__name__)


@click.group()
@click.option('--name', '-n', default='reV_collect', type=str,
              help='Collection job name. Default is "reV_collect".')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """reV Collect Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV collection configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV gen from a config file."""
    name = ctx.obj['NAME']

    # Instantiate the config object
    config = CollectionConfig(config_file)

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
              modules=[__name__, 'reV.handlers.collection'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV collection from config file: "{}"'
                .format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))
    logger.info('Target collection directory: "{}"'.format(config.coldir))
    logger.info('The following project points were specified: "{}"'
                .format(config.get('project_points', None)))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    # set config objects to be passed through invoke to direct methods
    ctx.obj['H5_DIR'] = config.coldir
    ctx.obj['LOG_DIR'] = config.logdir
    ctx.obj['DSETS'] = config.dsets
    ctx.obj['PROJECT_POINTS'] = config.project_points
    ctx.obj['PURGE_CHUNKS'] = config.purge_chunks
    ctx.obj['VERBOSE'] = verbose

    for file_prefix in config.file_prefixes:
        ctx.obj['NAME'] = name + '_{}'.format(file_prefix)
        ctx.obj['H5_FILE'] = os.path.join(config.dirout, file_prefix + '.h5')
        ctx.obj['FILE_PREFIX'] = file_prefix

        if config.execution_control.option == 'local':
            status = Status.retrieve_job_status(config.dirout, 'collect',
                                                ctx.obj['NAME'])
            if status != 'successful':
                Status.add_job(
                    config.dirout, 'collect', ctx.obj['NAME'], replace=True,
                    job_attrs={'hardware': 'local',
                               'fout': file_prefix + '.h5',
                               'dirout': config.dirout})
                ctx.invoke(collect)

        elif config.execution_control.option in ('eagle', 'slurm'):
            ctx.invoke(collect_slurm,
                       alloc=config.execution_control.alloc,
                       memory=config.execution_control.node_mem,
                       walltime=config.execution_control.walltime,
                       feature=config.execution_control.feature,
                       conda_env=config.execution_control.conda_env,
                       module=config.execution_control.module,
                       stdout_path=os.path.join(config.logdir, 'stdout'),
                       verbose=verbose)


@main.group()
@click.option('--h5_file', '-f', required=True, type=click.Path(),
              help='H5 file to be collected into.')
@click.option('--h5_dir', '-d', required=True, type=click.Path(exists=True),
              help='Directory containing h5 files to collect.')
@click.option('--project_points', '-pp', type=STR,
              help='Project points file representing the full '
              'collection scope.')
@click.option('--dsets', '-ds', required=True, type=STRLIST,
              help='Dataset names to be collected.')
@click.option('--file_prefix', '-fp', type=STR, default=None,
              help='File prefix found in the h5 file names to be collected.')
@click.option('--log_dir', '-ld', type=STR, default='./logs',
              help='Directory to put log files.')
@click.option('-p', '--purge_chunks', is_flag=True,
              help='Flag to delete chunked files after collection.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def direct(ctx, h5_file, h5_dir, project_points, dsets, file_prefix,
           log_dir, purge_chunks, verbose):
    """Main entry point for collection with context passing."""
    ctx.obj['H5_FILE'] = h5_file
    ctx.obj['H5_DIR'] = h5_dir
    ctx.obj['PROJECT_POINTS'] = project_points
    ctx.obj['DSETS'] = dsets
    ctx.obj['FILE_PREFIX'] = file_prefix
    ctx.obj['LOG_DIR'] = log_dir
    ctx.obj['PURGE_CHUNKS'] = purge_chunks
    ctx.obj['VERBOSE'] = verbose


@direct.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def collect(ctx, verbose):
    """Run collection on local worker."""

    name = ctx.obj['NAME']
    h5_file = ctx.obj['H5_FILE']
    h5_dir = ctx.obj['H5_DIR']
    project_points = ctx.obj['PROJECT_POINTS']
    dsets = ctx.obj['DSETS']
    file_prefix = ctx.obj['FILE_PREFIX']
    log_dir = ctx.obj['LOG_DIR']
    purge_chunks = ctx.obj['PURGE_CHUNKS']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # initialize loggers for multiple modules
    init_mult(name, log_dir, modules=[__name__, 'reV.handlers.collection'],
              verbose=verbose, node=True)

    for key, val in ctx.obj.items():
        logger.debug('ctx var passed to collection method: "{}" : "{}" '
                     'with type "{}"'.format(key, val, type(val)))

    logger.info('Collection is being run for "{}" with job name "{}" '
                'and collection dir: {}. Target output path is: {}'
                .format(dsets, name, h5_dir, h5_file))
    t0 = time.time()

    Collector.collect(h5_file, h5_dir, project_points, dsets[0],
                      file_prefix=file_prefix)

    if len(dsets) > 1:
        for dset_name in dsets[1:]:
            Collector.add_dataset(h5_file, h5_dir, dset_name,
                                  file_prefix=file_prefix)

    if purge_chunks:
        Collector.purge_chunks(h5_file, h5_dir, project_points,
                               file_prefix=file_prefix)
    else:
        Collector.move_chunks(h5_file, h5_dir, project_points,
                              file_prefix=file_prefix)

    runtime = (time.time() - t0) / 60
    logger.info('Collection completed in: {:.2f} min.'.format(runtime))

    # add job to reV status file.
    status = {'dirout': os.path.dirname(h5_file),
              'fout': os.path.basename(h5_file), 'job_status': 'successful',
              'runtime': runtime,
              'finput': os.path.join(h5_dir, '{}*.h5'.format(file_prefix))}
    Status.make_job_file(os.path.dirname(h5_file), 'collect', name, status)


def get_node_cmd(name, h5_file, h5_dir, project_points, dsets,
                 file_prefix=None, log_dir='./logs/',
                 purge_chunks=False, verbose=False):
    """Make a reV collection local CLI call string.

    Parameters
    ----------
    name : str
        reV collection jobname.
    h5_file : str
        Path to .h5 file into which data will be collected
    h5_dir : str
        Root directory containing .h5 files to combine
    project_points : str | slice | list | pandas.DataFrame
        Project points that correspond to the full collection of points
        contained in the .h5 files to be collected
    dsets : list
        List of datasets (strings) to be collected.
    file_prefix : str
        .h5 file prefix, if None collect all files on h5_dir
    log_dir : str
        Log directory.
    purge_chunks : bool
        Flag to delete the chunked files after collection.
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
                '-f {h5_file} '
                '-d {h5_dir} '
                '-pp {project_points} '
                '-ds {dsets} '
                '-fp {file_prefix} '
                '-ld {log_dir} '
                '{purge}'
                '{v}'
                .format(name=SubprocessManager.s(name),
                        h5_file=SubprocessManager.s(h5_file),
                        h5_dir=SubprocessManager.s(h5_dir),
                        project_points=SubprocessManager.s(project_points),
                        dsets=SubprocessManager.s(dsets),
                        file_prefix=SubprocessManager.s(file_prefix),
                        log_dir=SubprocessManager.s(log_dir),
                        purge='-p ' if purge_chunks else '',
                        v='-v ' if verbose else '',
                        ))

    # Python command that will be executed on a node
    # command strings after cli v7.0 use dashes instead of underscores
    cmd = ('python -m reV.handlers.cli_collect {arg_main} collect'
           .format(arg_main=arg_main))
    logger.debug('Creating the following command line call:\n\t{}'.format(cmd))
    return cmd


@direct.command()
@click.option('--alloc', '-a', default='rev', type=str,
              help='SLURM allocation account name. Default is "rev".')
@click.option('--memory', '-mem', default=None, type=INT,
              help='SLURM node memory request in GB. Default is None')
@click.option('--walltime', '-wt', default=1.0, type=float,
              help='SLURM walltime request in hours. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--conda_env', '-env', default=None, type=STR,
              help='Conda env to activate')
@click.option('--module', '-mod', default=None, type=STR,
              help='Module to load')
@click.option('--stdout_path', '-sout', default='./out/stdout', type=str,
              help='Subprocess standard output path. Default is ./out/stdout')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def collect_slurm(ctx, alloc, memory, walltime, feature, conda_env, module,
                  stdout_path, verbose):
    """Run collection on HPC via SLURM job submission."""

    name = ctx.obj['NAME']
    h5_file = ctx.obj['H5_FILE']
    h5_dir = ctx.obj['H5_DIR']
    log_dir = ctx.obj['LOG_DIR']
    project_points = ctx.obj['PROJECT_POINTS']
    dsets = ctx.obj['DSETS']
    file_prefix = ctx.obj['FILE_PREFIX']
    purge_chunks = ctx.obj['PURGE_CHUNKS']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    cmd = get_node_cmd(name, h5_file, h5_dir, project_points, dsets,
                       file_prefix=file_prefix, log_dir=log_dir,
                       purge_chunks=purge_chunks, verbose=verbose)

    status = Status.retrieve_job_status(os.path.dirname(h5_file), 'collect',
                                        name)
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'
               .format(name, os.path.dirname(h5_file)))
    else:
        logger.info('Running reV collection on SLURM with node name "{}", '
                    'collecting data to "{}" from "{}" with file prefix "{}".'
                    .format(name, h5_file, h5_dir, file_prefix))
        # create and submit the SLURM job
        slurm = SLURM(cmd, alloc=alloc, memory=memory, walltime=walltime,
                      feature=feature, name=name, conda_env=conda_env,
                      module=module, stdout_path=stdout_path)
        if slurm.id:
            msg = ('Kicked off reV collection job "{}" (SLURM jobid #{}).'
                   .format(name, slurm.id))
            # add job to reV status file.
            Status.add_job(
                os.path.dirname(h5_file), 'collect', name, replace=True,
                job_attrs={'job_id': slurm.id, 'hardware': 'eagle',
                           'fout': os.path.basename(h5_file),
                           'dirout': os.path.dirname(h5_file)})
        else:
            msg = ('Was unable to kick off reV collection job "{}". '
                   'Please see the stdout error messages'
                   .format(name))
    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    main(obj={})
