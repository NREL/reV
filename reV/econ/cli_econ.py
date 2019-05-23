"""
Econ CLI entry points.
"""
import click
import logging
from math import ceil
import os
import pprint
import time
import re
from warnings import warn

from reV.generation.cli_gen import get_node_name_fout, make_fout
from reV.config.project_points import ProjectPoints, PointsControl
from reV.config.analysis_configs import EconConfig
from reV.econ.econ import Econ
from reV.utilities.cli_dtypes import (INT, STR, SAMFILES, PROJECTPOINTS,
                                      INTLIST, STRLIST)
from reV.utilities.execution import PBS, SLURM, SubprocessManager
from reV.utilities.loggers import init_mult
from reV.pipeline.pipeline import Pipeline
from reV.pipeline.status import Status


logger = logging.getLogger(__name__)


@click.group()
@click.option('--name', '-n', default='reV_econ', type=STR,
              help='Econ analysis job name. Default is "reV_econ".')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """Command line interface (CLI) for the reV 2.0 Econ Module."""
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV econ configuration json file.')
@click.option('--status_dir', '-st', default=None, type=STR,
              help='Optional directory containing reV status json.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, status_dir, verbose):
    """Run reV econ from a config file."""
    name = ctx.obj['NAME']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # Instantiate the config object
    config = EconConfig(config_file)

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
    init_mult(name, config.logdir, modules=[__name__, 'reV.econ.econ',
                                            'reV.config', 'reV.utilities',
                                            'reV.SAM'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV 2.0 econ from config file: "{}"'
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
    ctx.obj['POINTS'] = config['project_points']
    ctx.obj['SAM_FILES'] = config.sam_config
    ctx.obj['SITE_DATA'] = config.site_data
    ctx.obj['DIROUT'] = config.dirout
    ctx.obj['LOGDIR'] = config.logdir
    ctx.obj['OUTPUT_REQUEST'] = config.output_request
    ctx.obj['SITES_PER_CORE'] = config.execution_control['sites_per_core']

    # Send status dir to methods to be used for status file
    if status_dir is None:
        status_dir = config.dirout
    ctx.obj['STATUS_DIR'] = status_dir

    for i, year in enumerate(config.years):
        submit_from_config(ctx, name, year, config, verbose, i)


def submit_from_config(ctx, name, year, config, verbose, i):
    """Function to submit one year from a config file.

    Parameters
    ----------
    ctx : cli.ctx
        Click context object. Use case: data = ctx.obj['key']
    name : str
        Job name.
    year : int | str | NoneType
        4 digit year or None.
    config : reV.config.EconConfig
        Econ config object.
    """

    # set the year-specific variables
    ctx.obj['CF_FILE'] = config.cf_files[i]
    ctx.obj['CF_YEAR'] = year

    # parse pipeline for cf_file if specified
    if ctx.obj['CF_FILE'].startswith('PIPELINE'):
        ctx.obj['CF_FILE'] = Pipeline.parse_previous(
            ctx.obj['STATUS_DIR'], 'econ',
            target=ctx.obj['CF_FILE'].split('_')[-1])

    # check to make sure that the year matches the resource file
    if str(year) not in config.cf_files[i]:
        warn('reV gen results file and year do not appear to match. '
             'Expected the string representation of the year '
             'to be in the generation results file name. '
             'Year: {}, generation results file: {}'
             .format(year, config.cf_files[i]))

    # if the year isn't in the name, add it before setting the file output
    ctx.obj['FOUT'] = make_fout(name, year)

    # invoke direct methods based on the config execution option
    if config.execution_control.option == 'local':
        sites_per_core = ceil(len(config.points_control) /
                              config.execution_control.ppn)
        ctx.obj['SITES_PER_CORE'] = sites_per_core
        ctx.invoke(econ_local, n_workers=config.execution_control.ppn,
                   points_range=None, verbose=verbose)

    elif config.execution_control.option == 'peregrine':
        match = re.match(r'.*([1-3][0-9]{3})', name)
        if not match and year:
            # Add year to name before submitting
            # 8 chars for pbs job name (lim is 16, -8 for "_year_ID")
            ctx.obj['NAME'] = '{}_{}'.format(name[:8], str(year))
        ctx.invoke(econ_peregrine, nodes=config.execution_control.nodes,
                   alloc=config.execution_control.alloc,
                   queue=config.execution_control.queue,
                   feature=config.execution_control.feature,
                   stdout_path=os.path.join(config.logdir, 'stdout'),
                   verbose=verbose)

    elif config.execution_control.option == 'eagle':
        match = re.match(r'.*([1-3][0-9]{3})', name)
        if not match and year:
            # Add year to name before submitting
            ctx.obj['NAME'] = '{}_{}'.format(name, str(year))
        ctx.invoke(econ_eagle, nodes=config.execution_control.nodes,
                   alloc=config.execution_control.alloc,
                   walltime=config.execution_control.walltime,
                   memory=config.execution_control.node_mem,
                   stdout_path=os.path.join(config.logdir, 'stdout'),
                   verbose=verbose)


@main.group()
@click.option('--sam_files', '-sf', required=True, type=SAMFILES,
              help='SAM config files (required) (str, dict, or list).')
@click.option('--cf_file', '-cf', default=None, type=click.Path(exists=True),
              help='Single generation results file (str).')
@click.option('--cf_year', '-cfy', default=None, type=INT,
              help='Year of generation results to analyze (if multiple years '
              'in cf_file). Default is None (use the only cf_mean dataset in '
              'cf_file).')
@click.option('--points', '-p', default=slice(0, 100), type=PROJECTPOINTS,
              help=('reV project points to analyze (slice, list, or file '
                    'string). Default is slice(0, 100)'))
@click.option('--site_data', '-sd', default=None, type=click.Path(exists=True),
              help='Site-specific data file for econ calculation.')
@click.option('--sites_per_core', '-spc', default=None, type=INT,
              help=('Number of sites to run in series on a single core. '
                    'Default is the resource column chunk size.'))
@click.option('--fout', '-fo', default='econ_output.h5', type=STR,
              help=('Filename output specification (should be .h5). '
                    'Default is "econ_output.h5"'))
@click.option('--dirout', '-do', default='./out/econ_out', type=STR,
              help='Output directory specification. Default is ./out/econ_out')
@click.option('--status_dir', '-st', default=None, type=STR,
              help='Directory containing the status file. Default is dirout.')
@click.option('--logdir', '-lo', default='./out/log_econ', type=STR,
              help='Econ log file directory. Default is ./out/log_econ')
@click.option('-or', '--output_request', type=STRLIST, default=['lcoe_fcr'],
              help=('Requested output variable name(s). '
                    'Default is ["lcoe_fcr"].'))
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def direct(ctx, sam_files, cf_file, cf_year, points, site_data, sites_per_core,
           fout, dirout, status_dir, logdir, output_request, verbose):
    """Run reV gen directly w/o a config file."""
    ctx.ensure_object(dict)
    ctx.obj['POINTS'] = points
    ctx.obj['SAM_FILES'] = sam_files
    ctx.obj['CF_FILE'] = cf_file
    ctx.obj['CF_YEAR'] = cf_year
    ctx.obj['SITE_DATA'] = site_data
    ctx.obj['SITES_PER_CORE'] = sites_per_core
    ctx.obj['FOUT'] = fout
    ctx.obj['DIROUT'] = dirout
    ctx.obj['STATUS_DIR'] = status_dir
    ctx.obj['LOGDIR'] = logdir
    ctx.obj['OUTPUT_REQUEST'] = output_request
    verbose = any([verbose, ctx.obj['VERBOSE']])


@direct.command()
@click.option('--n_workers', '-nw', type=INT,
              help='Number of workers. Use 1 for serial, None for all cores.')
@click.option('--points_range', '-pr', default=None, type=INTLIST,
              help='Optional range list to run a subset of sites.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def econ_local(ctx, n_workers, points_range, verbose):
    """Run econ on local worker(s)."""

    name = ctx.obj['NAME']
    points = ctx.obj['POINTS']
    sam_files = ctx.obj['SAM_FILES']
    cf_file = ctx.obj['CF_FILE']
    cf_year = ctx.obj['CF_YEAR']
    site_data = ctx.obj['SITE_DATA']
    sites_per_core = ctx.obj['SITES_PER_CORE']
    fout = ctx.obj['FOUT']
    dirout = ctx.obj['DIROUT']
    status_dir = ctx.obj['STATUS_DIR']
    logdir = ctx.obj['LOGDIR']
    output_request = ctx.obj['OUTPUT_REQUEST']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # add job to reV status file.
    if status_dir is None:
        status_dir = dirout
    Status.add_job(status_dir, 'econ', name, replace=False)

    # initialize loggers for multiple modules
    init_mult(name, logdir, modules=[__name__, 'reV.econ.econ', 'reV.config',
                                     'reV.utilities', 'reV.SAM'],
              verbose=verbose, node=True)

    for key, val in ctx.obj.items():
        logger.debug('ctx var passed to local method: "{}" : "{}" with type '
                     '"{}"'.format(key, val, type(val)))

    logger.info('Econ local is being run with with job name "{}" and '
                'generation results file: {}. Target output path is: {}'
                .format(name, cf_file, os.path.join(dirout, fout)))
    t0 = time.time()

    Status.retrieve_job_status(status_dir, 'econ', name)

    # Execute the Generation module with smart data flushing.
    Econ.run_smart(points=points,
                   sam_files=sam_files,
                   cf_file=cf_file,
                   cf_year=cf_year,
                   site_data=site_data,
                   output_request=output_request,
                   n_workers=n_workers,
                   sites_per_split=sites_per_core,
                   points_range=points_range,
                   fout=fout,
                   dirout=dirout)

    tmp_str = ' with points range {}'.format(points_range)
    logger.info('Econ compute complete for project points "{0}"{1}. '
                'Time elapsed: {2:.2f} min. Target output dir: {3}'
                .format(points, tmp_str if points_range else '',
                        (time.time() - t0) / 60, dirout))

    Status.set_job_status(status_dir, 'econ', name, 'successful')


def get_node_pc(points, sam_files, nodes):
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
    nodes : int
        Number of nodes that the PointsControl object is being split to.

    Returns
    -------
    pc : reV.config.project_points.PointsControl
        PointsControl object to be iterated and send to HPC nodes.
    """

    if isinstance(points, (str, slice, list, tuple)):
        # create points control via points
        pp = ProjectPoints(points, sam_files, tech=None)
        sites_per_node = ceil(len(pp) / nodes)
        pc = PointsControl(pp, sites_per_split=sites_per_node)
    else:
        raise TypeError('Econ Points input type is unrecognized: '
                        '"{}"'.format(type(points)))
    return pc


def get_node_cmd(name, sam_files, cf_file, cf_year=None, site_data=None,
                 points=slice(0, 100), points_range=None, sites_per_core=None,
                 n_workers=None, fout='reV.h5', dirout='./out/econ_out',
                 status_dir=None, logdir='./out/log_econ',
                 output_request='lcoe_fcr', verbose=False):
    """Made a reV econ direct-local command line interface call string.

    Parameters
    ----------
    name : str
        Name of the job to be submitted.
    sam_files : dict | str | list
        SAM input configuration ID(s) and file path(s). Keys are the SAM
        config ID(s), top level value is the SAM path. Can also be a single
        config file str. If it's a list, it is mapped to the sorted list
        of unique configs requested by points csv.
    cf_file : str
        reV generation results file name + path.
    cf_year : int | str
        reV generation year to calculate econ for. cf_year='my' will look
        for the multi-year mean generation results.
    site_data : str | None
        Site-specific data for econ calculation.
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
        Target filename to dump econ outputs.
    dirout : str
        Target directory to dump econ fout.
    status_dir : str
        Optional directory to save status file.
    logdir : str
        Target directory to save log files.
    output_request : list | tuple
        Output variable requested from SAM.
    verbose : bool
        Flag to turn on debug logging. Default is False.

    Returns
    -------
    cmd : str
        Single line command line argument to call the following CLI with
        appropriately formatted arguments based on input args:
            python -m reV.econ.cli_econ [args] direct [args] local [args]
    """

    # mark a cli arg string for main() in this module
    arg_main = ('-n {name} '.format(name=SubprocessManager.s(name)))

    s_site_data = '-sd {} '.format(SubprocessManager.s(site_data))
    ststr = '-st {} '.format(SubprocessManager.s(status_dir))

    # make a cli arg string for direct() in this module
    arg_direct = ('-p {points} '
                  '-sf {sam_files} '
                  '-cf {cf_file} '
                  '-cfy {cf_year} '
                  '{site_data}'
                  '-spc {sites_per_core} '
                  '-fo {fout} '
                  '-do {dirout} '
                  '{sdir}'
                  '-lo {logdir} '
                  '-or {out_req} '
                  .format(points=SubprocessManager.s(points),
                          sam_files=SubprocessManager.s(sam_files),
                          cf_file=SubprocessManager.s(cf_file),
                          cf_year=SubprocessManager.s(cf_year),
                          site_data=s_site_data if site_data else '',
                          sites_per_core=SubprocessManager.s(sites_per_core),
                          fout=SubprocessManager.s(fout),
                          dirout=SubprocessManager.s(dirout),
                          sdir=ststr if status_dir else '',
                          logdir=SubprocessManager.s(logdir),
                          out_req=SubprocessManager.s(output_request),
                          ))

    # make a cli arg string for local() in this module
    arg_loc = ('-nw {n_workers} '
               '-pr {points_range} '
               '{v}'.format(n_workers=SubprocessManager.s(n_workers),
                            points_range=SubprocessManager.s(points_range),
                            v='-v' if verbose else ''))

    # Python command that will be executed on a node
    # command strings after cli v7.0 use dashes instead of underscores
    cmd = ('python -m reV.econ.cli_econ '
           '{arg_main} direct {arg_direct} econ-local {arg_loc}'
           .format(arg_main=arg_main,
                   arg_direct=arg_direct,
                   arg_loc=arg_loc))
    logger.debug('Creating the following command line call:\n\t{}'.format(cmd))
    return cmd


@direct.command()
@click.option('--nodes', '-no', default=1, type=INT,
              help='Number of Peregrine nodes for econ job. Default is 1.')
@click.option('--alloc', '-a', default='rev', type=STR,
              help='Peregrine allocation account name. Default is "rev".')
@click.option('--queue', '-q', default='short', type=STR,
              help='Peregrine target job queue. Default is "short".')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Feature request. Format is "feature=64GB" or "qos=high". '
                    'Default is None.'))
@click.option('--stdout_path', '-sout', default='./out/stdout', type=STR,
              help='Subprocess standard output path. Default is ./out/stdout')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def econ_peregrine(ctx, nodes, alloc, queue, feature, stdout_path, verbose):
    """Run econ on Peregrine HPC via PBS job submission."""

    name = ctx.obj['NAME']
    points = ctx.obj['POINTS']
    sam_files = ctx.obj['SAM_FILES']
    cf_file = ctx.obj['CF_FILE']
    cf_year = ctx.obj['CF_YEAR']
    site_data = ctx.obj['SITE_DATA']
    sites_per_core = ctx.obj['SITES_PER_CORE']
    fout = ctx.obj['FOUT']
    dirout = ctx.obj['DIROUT']
    status_dir = ctx.obj['STATUS_DIR']
    logdir = ctx.obj['LOGDIR']
    output_request = ctx.obj['OUTPUT_REQUEST']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # initialize an info logger on the year level
    init_mult(name, logdir, modules=[__name__, 'reV.econ.econ', 'reV.config',
                                     'reV.utilities', 'reV.SAM'],
              verbose=False)

    pc = get_node_pc(points, sam_files, nodes)

    jobs = {}

    for i, split in enumerate(pc):
        node_name, fout_node = get_node_name_fout(name, fout, i, pc,
                                                  hpc='pbs')

        cmd = get_node_cmd(node_name, sam_files, cf_file, cf_year=cf_year,
                           site_data=site_data, points=points,
                           points_range=split.split_range,
                           sites_per_core=sites_per_core, n_workers=None,
                           fout=fout_node, dirout=dirout,
                           status_dir=status_dir, logdir=logdir,
                           output_request=output_request,
                           verbose=verbose)

        logger.info('Running reV econ on Peregrine with node name "{}" '
                    'for {} (points range: {}).'
                    .format(node_name, pc, split.split_range))

        # create and submit the PBS job
        pbs = PBS(cmd, alloc=alloc, queue=queue, name=node_name,
                  feature=feature, stdout_path=stdout_path)
        if pbs.id:
            msg = ('Kicked off reV econ job "{}" (PBS jobid #{}) on '
                   'Peregrine.'.format(node_name, pbs.id))
            # add job to reV status file.
            Status.add_job(status_dir, 'econ', node_name,
                           job_attrs={'job_id': pbs.id,
                                      'hardware': 'peregrine',
                                      'fout': fout_node,
                                      'dirout': dirout})
        else:
            msg = ('Was unable to kick off reV econ job "{}". '
                   'Please see the stdout error messages'
                   .format(node_name))
        click.echo(msg)
        logger.info(msg)
        jobs[i] = pbs

    return jobs


@direct.command()
@click.option('--nodes', '-no', default=1, type=INT,
              help='Number of Eagle nodes for econ job. Default is 1.')
@click.option('--alloc', '-a', default='rev', type=STR,
              help='Eagle allocation account name. Default is "rev".')
@click.option('--memory', '-mem', default=90, type=INT,
              help='Eagle node memory request in GB. Default is 90')
@click.option('--walltime', '-wt', default=0.5, type=float,
              help='Eagle walltime request in hours. Default is 0.5')
@click.option('--stdout_path', '-sout', default='./out/stdout', type=STR,
              help='Subprocess standard output path. Default is ./out/stdout')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def econ_eagle(ctx, nodes, alloc, memory, walltime, stdout_path, verbose):
    """Run econ on Eagle HPC via SLURM job submission."""

    name = ctx.obj['NAME']
    points = ctx.obj['POINTS']
    sam_files = ctx.obj['SAM_FILES']
    cf_file = ctx.obj['CF_FILE']
    cf_year = ctx.obj['CF_YEAR']
    site_data = ctx.obj['SITE_DATA']
    sites_per_core = ctx.obj['SITES_PER_CORE']
    fout = ctx.obj['FOUT']
    dirout = ctx.obj['DIROUT']
    status_dir = ctx.obj['STATUS_DIR']
    logdir = ctx.obj['LOGDIR']
    output_request = ctx.obj['OUTPUT_REQUEST']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # initialize an info logger on the year level
    init_mult(name, logdir, modules=[__name__, 'reV.econ.econ', 'reV.config',
                                     'reV.utilities', 'reV.SAM'],
              verbose=False)

    pc = get_node_pc(points, sam_files, nodes)

    jobs = {}

    for i, split in enumerate(pc):
        node_name, fout_node = get_node_name_fout(name, fout, i, pc,
                                                  hpc='slurm')

        cmd = get_node_cmd(node_name, sam_files, cf_file, cf_year=cf_year,
                           site_data=site_data, points=points,
                           points_range=split.split_range,
                           sites_per_core=sites_per_core, n_workers=None,
                           fout=fout_node, dirout=dirout,
                           status_dir=status_dir, logdir=logdir,
                           output_request=output_request,
                           verbose=verbose)

        logger.info('Running reV econ on Eagle with node name "{}" for '
                    '{} (points range: {}).'
                    .format(node_name, pc, split.split_range))

        # create and submit the SLURM job
        slurm = SLURM(cmd, alloc=alloc, memory=memory, walltime=walltime,
                      name=node_name, stdout_path=stdout_path)
        if slurm.id:
            msg = ('Kicked off reV econ job "{}" (SLURM jobid #{}) on '
                   'Eagle.'.format(node_name, slurm.id))
            # add job to reV status file.
            Status.add_job(status_dir, 'econ', node_name,
                           job_attrs={'job_id': slurm.id, 'hardware': 'eagle',
                                      'fout': fout_node, 'dirout': dirout})
        else:
            msg = ('Was unable to kick off reV econ job "{}". '
                   'Please see the stdout error messages'
                   .format(node_name))
        click.echo(msg)
        logger.info(msg)
        jobs[i] = slurm

    return jobs


if __name__ == '__main__':
    main(obj={})
