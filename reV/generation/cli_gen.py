"""
Generation CLI entry points.
"""
import click
import logging
from math import ceil
import os
import pprint
import re
import time
from warnings import warn

from reV.config.project_points import ProjectPoints, PointsControl
from reV.config.analysis_configs import GenConfig
from reV.generation.generation import Gen
from reV.utilities.cli_dtypes import (INT, STR, SAMFILES, PROJECTPOINTS,
                                      INTLIST, STRLIST)
from reV.utilities.execution import PBS, SLURM, SubprocessManager
from reV.utilities.loggers import init_mult
from reV.utilities.exceptions import ConfigError
from reV.pipeline.status import Status


logger = logging.getLogger(__name__)


@click.group()
@click.option('--name', '-n', default='reV_gen', type=STR,
              help='Generation job name. Default is "reV_gen".')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """Command line interface (CLI) for the reV 2.0 Generation Module."""
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV generation configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV gen from a config file."""
    name = ctx.obj['NAME']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # Instantiate the config object
    config = GenConfig(config_file)

    # take name from config if not default
    if config.name.lower() != 'rev':
        name = config.name
        ctx.obj['NAME'] = config.name

    # Enforce verbosity if logging level is specified in the config
    if config.logging_level == logging.DEBUG:
        verbose = True

    # make output directory if does not exist
    if not os.path.exists(config.dirout):
        os.makedirs(config.dirout)

    # initialize loggers.
    init_mult(name, config.logdir,
              modules=[__name__, 'reV.generation.generation',
                       'reV.config', 'reV.utilities', 'reV.SAM'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV 2.0 generation from config file: "{}"'
                .format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))
    logger.info('The following project points were specified: "{}"'
                .format(config.get('project_points', None)))
    logger.info('The following SAM configs are available to this run:\n{}'
                .format(pprint.pformat(config.get('sam_files', None),
                                       indent=4)))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    # set config objects to be passed through invoke to direct methods
    ctx.obj['TECH'] = config.tech
    ctx.obj['POINTS'] = config['project_points']
    ctx.obj['SAM_FILES'] = config.sam_config
    ctx.obj['DIROUT'] = config.dirout
    ctx.obj['LOGDIR'] = config.logdir
    ctx.obj['OUTPUT_REQUEST'] = config.output_request
    ctx.obj['SITES_PER_CORE'] = config.execution_control['sites_per_core']
    ctx.obj['MEM_UTIL_LIM'] = config.execution_control.mem_util_lim

    # get downscale request and raise exception if not NSRDB
    ctx.obj['DOWNSCALE'] = config.downscale
    if (config.downscale is not None and config.tech != 'pv' and
            config.tech != 'csp'):
        raise ConfigError('User requested downscaling for a non-solar '
                          'technology. reV does not have this capability at '
                          'the current time. Please contact a developer for '
                          'more information on this feature.')

    ctx.obj['CURTAILMENT'] = None
    if config.curtailment is not None:
        # pass through the curtailment file, not the curtailment object
        ctx.obj['CURTAILMENT'] = config['curtailment']

    for i, year in enumerate(config.years):
        submit_from_config(ctx, name, year, config, i, verbose=verbose)


def submit_from_config(ctx, name, year, config, i, verbose=False):
    """Function to submit one year from a config file.

    Parameters
    ----------
    ctx : cli.ctx
        Click context object. Use case: data = ctx.obj['key']
    name : str
        Job name.
    year : int | str | NoneType
        4 digit year or None.
    config : reV.config.GenConfig
        Generation config object.
    i : int
        Index variable associated with the index of the year in analysis years.
    verbose : bool
        Flag to turn on debug logging. Default is not verbose.
    """

    # set the year-specific variables
    ctx.obj['RES_FILE'] = config.res_files[i]

    # check to make sure that the year matches the resource file
    if str(year) not in config.res_files[i]:
        warn('Resource file and year do not appear to match. '
             'Expected the string representation of the year '
             'to be in the resource file name. '
             'Year: {}, Resource file: {}'
             .format(year, config.res_files[i]))

    # if the year isn't in the name, add it before setting the file output
    match = re.match(r'.*([1-3][0-9]{3})', name)
    if match and year:
        if str(year) != match:
            raise ConfigError('Tried to submit job for {}, but found a '
                              'different year in the base job name: "{}". '
                              'Please remove the year from the job name.'
                              .format(year, name))
    if year:
        ctx.obj['FOUT'] = '{}{}.h5'.format(name, '_{}'.format(year) if not
                                           match else '')
    else:
        ctx.obj['FOUT'] = '{}.h5'.format(name)

    # invoke direct methods based on the config execution option
    if config.execution_control.option == 'local':
        sites_per_core = ceil(len(config.points_control) /
                              config.execution_control.ppn)
        ctx.obj['SITES_PER_CORE'] = sites_per_core
        ctx.invoke(gen_local, n_workers=config.execution_control.ppn,
                   points_range=None, verbose=verbose)

    elif config.execution_control.option == 'peregrine':
        if not match and year:
            # Add year to name before submitting
            # 8 chars for pbs job name (lim is 16, -8 for "_year_ID")
            ctx.obj['NAME'] = '{}_{}'.format(name[:8], str(year))
        ctx.invoke(gen_peregrine, nodes=config.execution_control.nodes,
                   alloc=config.execution_control.alloc,
                   queue=config.execution_control.queue,
                   feature=config.execution_control.feature,
                   stdout_path=os.path.join(config.logdir, 'stdout'),
                   verbose=verbose)

    elif config.execution_control.option == 'eagle':
        if not match and year:
            # Add year to name before submitting
            ctx.obj['NAME'] = '{}_{}'.format(name, str(year))
        ctx.invoke(gen_eagle, nodes=config.execution_control.nodes,
                   alloc=config.execution_control.alloc,
                   walltime=config.execution_control.walltime,
                   memory=config.execution_control.node_mem,
                   feature=config.execution_control.feature,
                   stdout_path=os.path.join(config.logdir, 'stdout'),
                   verbose=verbose)


@main.group()
@click.option('--tech', '-t', required=True, type=STR,
              help='reV tech to analyze (required).')
@click.option('--sam_files', '-sf', required=True, type=SAMFILES,
              help='SAM config files (required) (str, dict, or list).')
@click.option('--res_file', '-rf', required=True, type=click.Path(exists=True),
              help='Single resource file (required) (str).')
@click.option('--points', '-p', default=slice(0, 100), type=PROJECTPOINTS,
              help=('reV project points to analyze '
                    '(slice, list, or file string). '
                    'Default is slice(0, 100)'))
@click.option('--sites_per_core', '-spc', default=None, type=INT,
              help=('Number of sites to run in series on a single core. '
                    'Default is the resource column chunk size.'))
@click.option('--fout', '-fo', default='gen_output.h5', type=STR,
              help=('Filename output specification (should be .h5). '
                    'Default is "gen_output.h5"'))
@click.option('--dirout', '-do', default='./out/gen_out', type=STR,
              help='Output directory specification. Default is ./out/gen_out')
@click.option('--logdir', '-lo', default='./out/log_gen', type=STR,
              help='Generation log file directory. Default is ./out/log_gen')
@click.option('-or', '--output_request', type=STRLIST, default=['cf_mean'],
              help=('List of requested output variable names. '
                    'Default is ["cf_mean"].'))
@click.option('-mem', '--mem_util_lim', type=float, default=0.4,
              help='Fractional node memory utilization limit. Default is 0.4 '
              'to account for numpy memory spikes and memory bloat.')
@click.option('-curt', '--curtailment', type=click.Path(exists=True),
              default=None,
              help=('JSON file with curtailment inputs parameters. '
                    'Default is None (no curtailment).'))
@click.option('-ds', '--downscale', type=STR, default=None,
              help=('Option to request temporal downscaling for NSRDB '
                    'resource data. Example request: "5min".'))
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def direct(ctx, tech, sam_files, res_file, points, sites_per_core, fout,
           dirout, logdir, output_request, mem_util_lim, curtailment,
           downscale, verbose):
    """Run reV gen directly w/o a config file."""
    ctx.ensure_object(dict)
    ctx.obj['TECH'] = tech
    ctx.obj['POINTS'] = points
    ctx.obj['SAM_FILES'] = sam_files
    ctx.obj['RES_FILE'] = res_file
    ctx.obj['SITES_PER_CORE'] = sites_per_core
    ctx.obj['FOUT'] = fout
    ctx.obj['DIROUT'] = dirout
    ctx.obj['LOGDIR'] = logdir
    ctx.obj['OUTPUT_REQUEST'] = output_request
    ctx.obj['MEM_UTIL_LIM'] = mem_util_lim
    ctx.obj['CURTAILMENT'] = curtailment
    ctx.obj['DOWNSCALE'] = downscale
    verbose = any([verbose, ctx.obj['VERBOSE']])


@direct.command()
@click.option('--n_workers', '-nw', type=INT,
              help='Number of workers. Use 1 for serial, None for all cores.')
@click.option('--points_range', '-pr', default=None, type=INTLIST,
              help='Optional range list to run a subset of sites.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def gen_local(ctx, n_workers, points_range, verbose):
    """Run generation on local worker(s)."""

    name = ctx.obj['NAME']
    tech = ctx.obj['TECH']
    points = ctx.obj['POINTS']
    sam_files = ctx.obj['SAM_FILES']
    res_file = ctx.obj['RES_FILE']
    sites_per_core = ctx.obj['SITES_PER_CORE']
    fout = ctx.obj['FOUT']
    dirout = ctx.obj['DIROUT']
    logdir = ctx.obj['LOGDIR']
    output_request = ctx.obj['OUTPUT_REQUEST']
    mem_util_lim = ctx.obj['MEM_UTIL_LIM']
    curtailment = ctx.obj['CURTAILMENT']
    downscale = ctx.obj['DOWNSCALE']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # add job to reV status file.
    Status.add_job(dirout, 'generation', name, replace=False)

    # initialize loggers for multiple modules
    init_mult(name, logdir, modules=[__name__, 'reV.generation.generation',
                                     'reV.config', 'reV.utilities', 'reV.SAM'],
              verbose=verbose, node=True)

    for key, val in ctx.obj.items():
        logger.debug('ctx var passed to local method: "{}" : "{}" with type '
                     '"{}"'.format(key, val, type(val)))

    logger.info('Gen local is being run with with job name "{}" and resource '
                'file: {}. Target output path is: {}'
                .format(name, res_file, os.path.join(dirout, fout)))
    t0 = time.time()

    Status.retrieve_job_status(dirout, 'generation', name)

    # Execute the Generation module with smart data flushing.
    Gen.run_smart(tech=tech,
                  points=points,
                  sam_files=sam_files,
                  res_file=res_file,
                  output_request=output_request,
                  curtailment=curtailment,
                  downscale=downscale,
                  n_workers=n_workers,
                  sites_per_split=sites_per_core,
                  points_range=points_range,
                  fout=fout,
                  dirout=dirout,
                  mem_util_lim=mem_util_lim)

    tmp_str = ' with points range {}'.format(points_range)
    logger.info('Gen compute complete for project points "{0}"{1}. '
                'Time elapsed: {2:.2f} min. Target output dir: {3}'
                .format(points, tmp_str if points_range else '',
                        (time.time() - t0) / 60, dirout))

    Status.set_job_status(dirout, 'generation', name, 'successful')


def get_node_pc(points, sam_files, tech, res_file, nodes):
    """Get a PointsControl object to be send to HPC nodes.

    Parameters
    ----------
    points : slice | str | list | tuple
        Slice/list specifying project points, string pointing to a project
        points csv.
    sam_files : dict | str | list
        SAM input configuration ID(s) and file path(s). Keys are the SAM
        config ID(s), top level value is the SAM path. Can also be a single
        config file str. If it's a list, it is mapped to the sorted list
        of unique configs requested by points csv.
    tech : str
        reV technology being executed.
    res_file : str
        Optional resource file to find maximum length of project points if
        points slice stop is None.
    nodes : int
        Number of nodes that the PointsControl object is being split to.

    Returns
    -------
    pc : reV.config.project_points.PointsControl
        PointsControl object to be iterated and send to HPC nodes.
    """

    if isinstance(points, (str, slice, list, tuple)):
        # create points control via points
        pp = ProjectPoints(points, sam_files, tech, res_file=res_file)
        sites_per_node = ceil(len(pp) / nodes)
        pc = PointsControl(pp, sites_per_split=sites_per_node)
    else:
        raise TypeError('Generation Points input type is unrecognized: '
                        '"{}"'.format(type(points)))
    return pc


def get_node_name_fout(name, fout, i, pc, hpc='slurm'):
    """Make a node name and fout unique to the run name, year, and node number.

    Parameters
    ----------
    name : str
        Base node/job name
    fout : str
        Base file output name (no path) (with or without .h5 extension)
    i : int | None
        Node number. If None, only a single node is being used and no
        enumeration is necessary.
    pc : reV.config.PointsControl
        A PointsControl instance that i is enumerated from.
    hpc : str
        HPC job submission tool name (e.g. slurm or pbs). Affects job name.

    Returns
    -------
    node_name : str
        Base node name with _00 tag. 16 char max.
    fout_node : str
        Base file output name with _node00 tag.
    """

    if not fout.endswith('.h5'):
        fout += '.h5'

    if i is None or len(pc) == 1:
        fout_node = fout
        node_name = fout

    else:
        if hpc is 'slurm':
            node_name = '{0}_{1:02d}'.format(name, i)
        elif hpc is 'pbs':
            # 13 chars for pbs, (lim is 16, -3 for "_ID")
            node_name = '{0}_{1:02d}'.format(name[:13], i)

        fout_node = fout.replace('.h5', '_node{0:02d}.h5'.format(i))

    return node_name, fout_node


def get_node_cmd(name, tech, sam_files, res_file, points=slice(0, 100),
                 points_range=None, sites_per_core=None, n_workers=None,
                 fout='reV.h5', dirout='./out/gen_out', logdir='./out/log_gen',
                 output_request=('cf_mean',), mem_util_lim=0.4,
                 curtailment=None, downscale=None, verbose=False):
    """Make a reV geneneration direct-local CLI call string.

    Parameters
    ----------
    name : str
        Name of the job to be submitted.
    tech : str
        Name of the reV technology to be analyzed.
        (e.g. pv, csp, landbasedwind, offshorewind).
    sam_files : dict | str | list
        SAM input configuration ID(s) and file path(s). Keys are the SAM
        config ID(s), top level value is the SAM path. Can also be a single
        config file str. If it's a list, it is mapped to the sorted list
        of unique configs requested by points csv.
    res_file : str
        WTK or NSRDB resource file name + path.
    points : slice | str | list | tuple
        Slice/list specifying project points, string pointing to a project
    points_range : list | None
        Optional range list to run a subset of sites
    sites_per_core : int | None
        Number of sites to be analyzed in serial on a single local core.
    n_workers : int | None
        Number of workers to use on a node. None defaults to all available
        workers.
    fout : str
        Target filename to dump generation outputs.
    dirout : str
        Target directory to dump generation fout.
    logdir : str
        Target directory to save log files.
    output_request : list | tuple
        Output variables requested from SAM.
    mem_util_lim : float
        Memory utilization limit (fractional).
    curtailment : NoneType | str
        Pointer to a file containing curtailment input parameters or None if
        no curtailment.
    downscale : NoneType | str
        Option for NSRDB resource downscaling to higher temporal
        resolution. Expects a string in the Pandas frequency format,
        e.g. '5min'.
    verbose : bool
        Flag to turn on debug logging. Default is False.

    Returns
    -------
    cmd : str
        Single line command line argument to call the following CLI with
        appropriately formatted arguments based on input args:
            python -m reV.generation.cli_gen [args] direct [args] local [args]
    """

    # mark a cli arg string for main() in this module
    arg_main = ('-n {name} '.format(name=SubprocessManager.s(name)))

    # make some strings only if specified
    cstr = '-curt {} '.format(SubprocessManager.s(curtailment))
    dstr = '-ds {} '.format(SubprocessManager.s(downscale))

    # make a cli arg string for direct() in this module
    arg_direct = ('-t {tech} '
                  '-p {points} '
                  '-sf {sam_files} '
                  '-rf {res_file} '
                  '-spc {sites_per_core} '
                  '-fo {fout} '
                  '-do {dirout} '
                  '-lo {logdir} '
                  '-or {out_req} '
                  '-mem {mem} '
                  '{curt}'
                  '{ds}'
                  .format(tech=SubprocessManager.s(tech),
                          points=SubprocessManager.s(points),
                          sam_files=SubprocessManager.s(sam_files),
                          res_file=SubprocessManager.s(res_file),
                          sites_per_core=SubprocessManager.s(sites_per_core),
                          fout=SubprocessManager.s(fout),
                          dirout=SubprocessManager.s(dirout),
                          logdir=SubprocessManager.s(logdir),
                          out_req=SubprocessManager.s(output_request),
                          mem=SubprocessManager.s(mem_util_lim),
                          curt=cstr if curtailment else '',
                          ds=dstr if downscale else '',
                          ))

    # make a cli arg string for local() in this module
    arg_loc = ('-nw {n_workers} '
               '-pr {points_range} '
               '{v}'.format(n_workers=SubprocessManager.s(n_workers),
                            points_range=SubprocessManager.s(points_range),
                            v='-v' if verbose else ''))

    # Python command that will be executed on a node
    # command strings after cli v7.0 use dashes instead of underscores
    cmd = ('python -m reV.generation.cli_gen '
           '{arg_main} direct {arg_direct} gen-local {arg_loc}'
           .format(arg_main=arg_main,
                   arg_direct=arg_direct,
                   arg_loc=arg_loc))
    logger.debug('Creating the following command line call:\n\t{}'.format(cmd))
    return cmd


@direct.command()
@click.option('--nodes', '-no', default=1, type=INT,
              help='Number of Peregrine nodes for gen job. Default is 1.')
@click.option('--alloc', '-a', default='rev', type=STR,
              help='Peregrine allocation account name. Default is "rev".')
@click.option('--queue', '-q', default='batch-h', type=STR,
              help='Peregrine target job queue. Default is "batch-h".')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Feature request. Format is "feature=64GB" or "qos=high". '
                    'Default is None.'))
@click.option('--stdout_path', '-sout', default='./out/stdout', type=STR,
              help='Subprocess standard output path. Default is ./out/stdout')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def gen_peregrine(ctx, nodes, alloc, queue, feature, stdout_path, verbose):
    """Run generation on Peregrine HPC via PBS job submission."""

    name = ctx.obj['NAME']
    tech = ctx.obj['TECH']
    points = ctx.obj['POINTS']
    sam_files = ctx.obj['SAM_FILES']
    res_file = ctx.obj['RES_FILE']
    sites_per_core = ctx.obj['SITES_PER_CORE']
    fout = ctx.obj['FOUT']
    dirout = ctx.obj['DIROUT']
    logdir = ctx.obj['LOGDIR']
    output_request = ctx.obj['OUTPUT_REQUEST']
    mem_util_lim = ctx.obj['MEM_UTIL_LIM']
    curtailment = ctx.obj['CURTAILMENT']
    downscale = ctx.obj['DOWNSCALE']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # initialize an info logger on the year level
    init_mult(name, logdir, modules=[__name__, 'reV.generation.generation',
                                     'reV.config', 'reV.utilities', 'reV.SAM'],
              verbose=False)

    pc = get_node_pc(points, sam_files, tech, res_file, nodes)

    jobs = {}

    for i, split in enumerate(pc):
        node_name, fout_node = get_node_name_fout(name, fout, i, pc,
                                                  hpc='pbs')

        cmd = get_node_cmd(node_name, tech, sam_files, res_file,
                           points=points, points_range=split.split_range,
                           sites_per_core=sites_per_core, n_workers=None,
                           fout=fout_node, dirout=dirout, logdir=logdir,
                           output_request=output_request,
                           mem_util_lim=mem_util_lim, curtailment=curtailment,
                           downscale=downscale, verbose=verbose)

        logger.info('Running reV generation on Peregrine with node name "{}" '
                    'for {} (points range: {}).'
                    .format(node_name, pc, split.split_range))

        # create and submit the PBS job
        pbs = PBS(cmd, alloc=alloc, queue=queue, name=node_name,
                  feature=feature, stdout_path=stdout_path)
        if pbs.id:
            msg = ('Kicked off reV generation job "{}" (PBS jobid #{}) on '
                   'Peregrine.'.format(node_name, pbs.id))

            # add job to reV status file.
            Status.add_job(dirout, 'generation', node_name,
                           job_attrs={'job_id': pbs.id,
                                      'hardware': 'peregrine'})
        else:
            msg = ('Was unable to kick off reV generation job "{}". '
                   'Please see the stdout error messages'
                   .format(node_name))
        click.echo(msg)
        logger.info(msg)
        jobs[i] = pbs

    return jobs


@direct.command()
@click.option('--nodes', '-no', default=1, type=INT,
              help='Number of Eagle nodes for gen job. Default is 1.')
@click.option('--alloc', '-a', default='rev', type=STR,
              help='Eagle allocation account name. Default is "rev".')
@click.option('--memory', '-mem', default=90, type=INT,
              help='Eagle node memory request in GB. Default is 90')
@click.option('--walltime', '-wt', default=1.0, type=float,
              help='Eagle walltime request in hours. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--stdout_path', '-sout', default='./out/stdout', type=STR,
              help='Subprocess standard output path. Default is ./out/stdout')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def gen_eagle(ctx, nodes, alloc, memory, walltime, feature, stdout_path,
              verbose):
    """Run generation on Eagle HPC via SLURM job submission."""

    name = ctx.obj['NAME']
    tech = ctx.obj['TECH']
    points = ctx.obj['POINTS']
    sam_files = ctx.obj['SAM_FILES']
    res_file = ctx.obj['RES_FILE']
    sites_per_core = ctx.obj['SITES_PER_CORE']
    fout = ctx.obj['FOUT']
    dirout = ctx.obj['DIROUT']
    logdir = ctx.obj['LOGDIR']
    output_request = ctx.obj['OUTPUT_REQUEST']
    mem_util_lim = ctx.obj['MEM_UTIL_LIM']
    curtailment = ctx.obj['CURTAILMENT']
    downscale = ctx.obj['DOWNSCALE']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # initialize an info logger on the year level
    init_mult(name, logdir, modules=[__name__, 'reV.generation.generation',
                                     'reV.config', 'reV.utilities', 'reV.SAM'],
              verbose=False)

    pc = get_node_pc(points, sam_files, tech, res_file, nodes)

    jobs = {}

    for i, split in enumerate(pc):
        node_name, fout_node = get_node_name_fout(name, fout, i, pc,
                                                  hpc='slurm')

        cmd = get_node_cmd(node_name, tech, sam_files, res_file,
                           points=points, points_range=split.split_range,
                           sites_per_core=sites_per_core, n_workers=None,
                           fout=fout_node, dirout=dirout, logdir=logdir,
                           output_request=output_request,
                           mem_util_lim=mem_util_lim, curtailment=curtailment,
                           downscale=downscale, verbose=verbose)

        logger.info('Running reV generation on Eagle with node name "{}" for '
                    '{} (points range: {}).'
                    .format(node_name, pc, split.split_range))

        # create and submit the SLURM job
        slurm = SLURM(cmd, alloc=alloc, memory=memory, walltime=walltime,
                      feature=feature, name=node_name, stdout_path=stdout_path)
        if slurm.id:
            msg = ('Kicked off reV generation job "{}" (SLURM jobid #{}) on '
                   'Eagle.'.format(node_name, slurm.id))

            # add job to reV status file.
            Status.add_job(dirout, 'generation', node_name,
                           job_attrs={'job_id': slurm.id, 'hardware': 'eagle'})
        else:
            msg = ('Was unable to kick off reV generation job "{}". '
                   'Please see the stdout error messages'
                   .format(node_name))
        click.echo(msg)
        logger.info(msg)
        jobs[i] = slurm

    return jobs


if __name__ == '__main__':
    main(obj={})
