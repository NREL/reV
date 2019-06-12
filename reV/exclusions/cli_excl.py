"""
Exclusions CLI entry points.
"""
import click
import logging
import os
import pprint
import time
import json

from reV.exclusions.exclusions import Exclusions
from reV.config.analysis_configs import ExclConfig
from reV.utilities.cli_dtypes import STR
from reV.utilities.loggers import init_mult
from reV.utilities.execution import SLURM
from reV.pipeline.status import Status

logger = logging.getLogger(__name__)


@click.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV exclusions configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV gen from a config file."""
    name = ctx.obj['NAME']

    # Instantiate the config object
    config = ExclConfig(config_file)

    # take name from config if not default
    if config.name.lower() != 'rev':
        name = config.name

    # Enforce verbosity if logging level is specified in the config
    if config.logging_level == logging.DEBUG:
        verbose = True

    # initialize loggers
    init_mult(name, config.logdir,
              modules=[__name__, 'reV.exclusions.exclusions',
                       'reV.config', 'reV.utilities'], verbose=verbose)

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

    # set config objects to be passed through invoke to direct methods
    ctx.obj['EXCLUSIONS'] = config.exclusions
    ctx.obj['DIROUT'] = config.dirout
    ctx.obj['FOUT'] = config.fout
    ctx.obj['USE_BLOCKS'] = config.use_blocks
    ctx.obj['FILTER'] = config.filter
    ctx.obj['LOGDIR'] = config.logdir

    if config.execution_control.option == 'local':
        ctx.invoke(exclusions)

    elif config.execution_control.option == 'eagle':
        ctx.invoke(exclusions_eagle,
                   alloc=config.execution_control.alloc,
                   memory=config.execution_control.node_mem,
                   walltime=config.execution_control.walltime,
                   feature=config.execution_control.feature,
                   stdout_path=os.path.join(config.logdir, 'stdout'),
                   verbose=verbose)


@click.group()
@click.option('-n', '--name', default='reV_exclusions', type=str,
              help='Collection job name. Default is "reV_collect".')
@click.option('-c', '--layers_config_str', default=None, type=str,
              help='Layer configuration json as string for exclusions.')
@click.option('-o', '--fout',
              required=True, type=click.Path(exists=False),
              help='Output file path for exclusion.')
@click.option('--dirout', '-d', required=True, type=click.Path(exists=True),
              help='Directory for exclusions output.')
@click.option('--logdir', '-l', required=True, type=click.Path(exists=True),
              help='Directory for exclusions output.')
@click.option('-f', '--filter_method', default=None, type=str,
              help='Contiguous filter method to apply')
@click.option('-b', '--use_blocks', is_flag=True,
              help='Flag to turn on block windows.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def main(ctx, name, layers_config_str, fout,
         dirout, logdir, filter_method, use_blocks, verbose):
    """Main entry point for exclusions with context passing."""

    ctx.obj['NAME'] = name
    ctx.obj['DIROUT'] = dirout
    ctx.obj['LOGDIR'] = logdir
    ctx.obj['EXCLUSIONS'] = json.loads(layers_config_str)
    ctx.obj['FILTER'] = filter_method
    ctx.obj['FOUT'] = fout
    ctx.obj['USE_BLOCKS'] = use_blocks
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def exclusions(ctx, verbose):
    """Command line interface (CLI) for the reV 2.0 Exclusion Module."""

    name = ctx.obj['NAME']
    dirout = ctx.obj['DIROUT']
    logdir = ctx.obj['LOGDIR']
    config = ctx.obj['EXCLUSIONS']
    contiguous_filter = ctx.obj['FILTER']
    fout = ctx.obj['FOUT']
    use_blocks = ctx.obj['USE_BLOCKS']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    fpath = os.path.join(dirout, fout)

    # initialize loggers
    init_mult(name, logdir,
              modules=[__name__, 'reV.exclusions.exclusions',
                       'reV.config', 'reV.utilities'], verbose=verbose)

    t0 = time.time()
    Exclusions.run(config=config,
                   output_fpath=fpath,
                   use_blocks=use_blocks,
                   contiguous_filter=contiguous_filter)
    runtime = (time.time() - t0) / 60

    # add job to reV status file.
    finput = str([excl.fpath for excl in config])
    status = {'finput': finput, 'fpath': fpath,
              'job_status': 'successful', 'runtime': runtime}
    Status.make_job_file(fpath, 'exclusions', name, status)

    return None


@main.command()
@click.option('--alloc', '-a', default='rev', type=str,
              help='Eagle allocation account name. Default is "rev".')
@click.option('--memory', '-mem', default=90, type=int,
              help='Eagle node memory request in GB. Default is 90')
@click.option('--walltime', '-wt', default=1.0, type=float,
              help='Eagle walltime request in hours. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--stdout_path', '-sout', default='./out/stdout', type=str,
              help='Subprocess standard output path. Default is ./out/stdout')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def exclusions_eagle(ctx, alloc, memory, walltime,
                     feature, stdout_path, verbose):
    """Run exclusions on Eagle HPC via SLURM job submission."""

    name = ctx.obj['NAME']
    dirout = ctx.obj['DIROUT']
    config = ctx.obj['EXCLUSIONS']
    contiguous_filter = ctx.obj['FILTER']
    fout = ctx.obj['FOUT']
    use_blocks = ctx.obj['USE_BLOCKS']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    fpath = os.path.join(dirout, fout)

    # Python command that will be executed on a node
    if verbose:
        verbose_flag = '-v'
    else:
        verbose_flag = ''
    cmd = ('python -m reV.exclusions.cli_excl '
           '-n {name} '
           '-c {layers_config_str} '
           '-o {fout} '
           '-d {dirout} '
           '-f {contiguous_filter} '
           '-b {use_blocks} '
           '{verbose_flag} exclusions'
           .format(name=name,
                   layers_config_str=json.dumps(config),
                   fout=fout,
                   dirout=dirout,
                   contiguous_filter=contiguous_filter,
                   use_blocks=use_blocks,
                   verbose_flag=verbose_flag))

    status = Status.retrieve_job_status(fpath, 'exclusions', name)
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'
               .format(name, dirout))

    else:
        logger.info('Running reV exclusions on Eagle with node name "{}"'
                    .format(name))

        # create and submit the SLURM job
        slurm = SLURM(cmd,
                      alloc=alloc,
                      memory=memory,
                      walltime=walltime,
                      name=name,
                      stdout_path=stdout_path,
                      feature=feature)
        if slurm.id:
            msg = ('Kicked off reV exclusions job "{}" (SLURM jobid #{}) on '
                   'Eagle.'.format(name, slurm.id))
        else:
            msg = ('Was unable to kick off reV exclusions job "{}". '
                   'Please see the stdout error messages'
                   .format(name))
    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    main(obj={})
