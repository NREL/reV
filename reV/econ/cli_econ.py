# -*- coding: utf-8 -*-
"""
Econ CLI entry points.
"""
import click
import logging
from math import ceil
import os
import pprint
import time
from warnings import warn

from reV.config.project_points import ProjectPoints, PointsControl
from reV.config.sam_analysis_configs import EconConfig
from reV.econ.econ import Econ
from reV.generation.cli_gen import get_node_name_fout, make_fout
from reV.pipeline.status import Status
from reV.utilities.cli_dtypes import SAMFILES, PROJECTPOINTS

from rex.utilities.cli_dtypes import INT, STR, INTLIST, STRLIST
from rex.utilities.execution import SLURM, SubprocessManager
from rex.utilities.loggers import init_mult
from rex.utilities.utilities import parse_year

logger = logging.getLogger(__name__)


@click.group()
@click.option('--name', '-n', default='reV-econ', type=STR,
              help='reV Economics job name, by default "reV-econ".')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """reV Economics Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    ctx.obj['NAME'] = name


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV econ configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
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
    if config.log_level == logging.DEBUG:
        verbose = True

    # make output directory if does not exist
    if not os.path.exists(config.dirout):
        os.makedirs(config.dirout)

    # initialize loggers.
    init_mult(name, config.logdir, modules=[__name__, 'reV.econ.econ',
                                            'reV.config', 'reV.utilities',
                                            'reV.SAM', 'rex.utilities'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV Econ from config file: "{}"'
                .format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))
    logger.info('The following project points were specified: "{}"'
                .format(config.get('project_points', None)))
    logger.info('The following SAM configs are available to this run:\n{}'
                .format(pprint.pformat(config.get('sam_files', None),
                                       indent=4)))
    logger.debug('Submitting jobs for the following cf_files: {}'
                 .format(config.cf_files))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    # set config objects to be passed through invoke to direct methods
    ctx.obj['POINTS'] = config['project_points']
    ctx.obj['SAM_FILES'] = config.sam_config
    ctx.obj['SITE_DATA'] = config.site_data
    ctx.obj['DIROUT'] = config.dirout
    ctx.obj['LOGDIR'] = config.logdir
    ctx.obj['APPEND'] = config.append
    ctx.obj['OUTPUT_REQUEST'] = config.output_request
    ctx.obj['SITES_PER_WORKER'] = config.execution_control.sites_per_worker
    ctx.obj['MAX_WORKERS'] = config.execution_control.max_workers
    ctx.obj['TIMEOUT'] = config.timeout

    if len(config.years) == len(config.cf_files):
        for i, year in enumerate(config.years):
            cf_file = config.cf_files[i]
            submit_from_config(ctx, name, cf_file, year, config, verbose)
    else:
        for i, cf_file in enumerate(config.cf_files):
            year = parse_year(cf_file)
            if str(year) in [str(y) for y in config.years]:
                submit_from_config(ctx, name, cf_file, year, config, verbose)


def submit_from_config(ctx, name, cf_file, year, config, verbose):
    """Function to submit one year from a config file.

    Parameters
    ----------
    ctx : cli.ctx
        Click context object. Use case: data = ctx.obj['key']
    cf_file : str
        reV generation file with capacity factors to calculate econ for.
    name : str
        Job name.
    year : int | str | NoneType
        4 digit year or None.
    config : reV.config.EconConfig
        Econ config object.
    """

    # set the year-specific variables
    ctx.obj['CF_FILE'] = cf_file
    ctx.obj['CF_YEAR'] = year

    # check to make sure that the year matches the resource file
    if str(year) not in cf_file:
        warn('reV gen results file and year do not appear to match. '
             'Expected the string representation of the year '
             'to be in the generation results file name. '
             'Year: {}, generation results file: {}'
             .format(year, cf_file))

    # if the year isn't in the name, add it before setting the file output
    ctx.obj['FOUT'] = make_fout(name, year)
    if config.append:
        ctx.obj['FOUT'] = os.path.basename(cf_file)

    # invoke direct methods based on the config execution option
    if config.execution_control.option == 'local':
        name_year = make_fout(name, year).replace('.h5', '')
        name_year = name_year.replace('gen', 'econ')
        ctx.obj['NAME'] = name_year
        status = Status.retrieve_job_status(config.dirout, 'econ', name_year)
        if status != 'successful':
            Status.add_job(
                config.dirout, 'econ', name_year, replace=True,
                job_attrs={'hardware': 'local',
                           'fout': ctx.obj['FOUT'],
                           'dirout': config.dirout})
            ctx.invoke(econ_local,
                       max_workers=config.execution_control.max_workers,
                       timeout=config.timeout, points_range=None,
                       verbose=verbose)

    elif config.execution_control.option in ('eagle', 'slurm'):
        if not parse_year(name, option='bool') and year:
            # Add year to name before submitting
            ctx.obj['NAME'] = '{}_{}'.format(name, str(year))
        ctx.invoke(econ_slurm, nodes=config.execution_control.nodes,
                   alloc=config.execution_control.alloc,
                   walltime=config.execution_control.walltime,
                   memory=config.execution_control.node_mem,
                   feature=config.execution_control.feature,
                   module=config.execution_control.module,
                   conda_env=config.execution_control.conda_env,
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
@click.option('--sites_per_worker', '-spw', default=None, type=INT,
              help=('Number of sites to run in series on a single worker. '
                    'Default is the resource column chunk size.'))
@click.option('--fout', '-fo', default='econ_output.h5', type=STR,
              help=('Filename output specification (should be .h5). '
                    'Default is "econ_output.h5"'))
@click.option('--dirout', '-do', default='./out/econ_out', type=STR,
              help='Output directory specification. Default is ./out/econ_out')
@click.option('--logdir', '-lo', default='./out/log_econ', type=STR,
              help='Econ log file directory. Default is ./out/log_econ')
@click.option('-or', '--output_request', type=STRLIST, default=['lcoe_fcr'],
              help=('Requested output variable name(s). '
                    'Default is ["lcoe_fcr"].'))
@click.option('-ap', '--append', is_flag=True,
              help='Flag to append econ datasets to source cf_file. This has '
              'priority over fout and dirout inputs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def direct(ctx, sam_files, cf_file, cf_year, points, site_data,
           sites_per_worker, fout, dirout, logdir, output_request,
           append, verbose):
    """Run reV gen directly w/o a config file."""
    ctx.ensure_object(dict)
    ctx.obj['POINTS'] = points
    ctx.obj['SAM_FILES'] = sam_files
    ctx.obj['CF_FILE'] = cf_file
    ctx.obj['CF_YEAR'] = cf_year
    ctx.obj['SITE_DATA'] = site_data
    ctx.obj['SITES_PER_WORKER'] = sites_per_worker
    ctx.obj['FOUT'] = fout
    ctx.obj['DIROUT'] = dirout
    ctx.obj['LOGDIR'] = logdir
    ctx.obj['OUTPUT_REQUEST'] = output_request
    ctx.obj['APPEND'] = append
    verbose = any([verbose, ctx.obj['VERBOSE']])


@direct.command()
@click.option('--max_workers', '-mw', type=INT,
              help='Number of workers. Use 1 for serial, None for all cores.')
@click.option('--timeout', '-to', type=INT, default=1800,
              help='Number of seconds to wait for econ parallel run '
              'iterations to complete before returning zeros. '
              'Default is 1800 seconds.')
@click.option('--points_range', '-pr', default=None, type=INTLIST,
              help='Optional range list to run a subset of sites.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def econ_local(ctx, max_workers, timeout, points_range, verbose):
    """Run econ on local worker(s)."""

    name = ctx.obj['NAME']
    points = ctx.obj['POINTS']
    sam_files = ctx.obj['SAM_FILES']
    cf_file = ctx.obj['CF_FILE']
    cf_year = ctx.obj['CF_YEAR']
    site_data = ctx.obj['SITE_DATA']
    sites_per_worker = ctx.obj['SITES_PER_WORKER']
    fout = ctx.obj['FOUT']
    dirout = ctx.obj['DIROUT']
    logdir = ctx.obj['LOGDIR']
    output_request = ctx.obj['OUTPUT_REQUEST']
    append = ctx.obj['APPEND']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    if append:
        fout = os.path.basename(cf_file)
        dirout = os.path.dirname(cf_file)

    # initialize loggers for multiple modules
    log_modules = [__name__, 'reV.econ.econ', 'reV.generation', 'reV.config',
                   'reV.utilities', 'reV.SAM', 'reV.handlers', 'rex.utilities']
    init_mult(name, logdir, modules=log_modules,
              verbose=verbose, node=True)

    for key, val in ctx.obj.items():
        logger.debug('ctx var passed to local method: "{}" : "{}" with type '
                     '"{}"'.format(key, val, type(val)))

    logger.info('Econ local is being run with with job name "{}" and '
                'generation results file: {}. Target output path is: {}'
                .format(name, cf_file, os.path.join(dirout, fout)))
    t0 = time.time()

    # Execute the Generation module with smart data flushing.
    Econ.reV_run(points=points,
                 sam_files=sam_files,
                 cf_file=cf_file,
                 cf_year=cf_year,
                 site_data=site_data,
                 output_request=output_request,
                 max_workers=max_workers,
                 timeout=timeout,
                 sites_per_worker=sites_per_worker,
                 points_range=points_range,
                 fout=fout,
                 dirout=dirout,
                 append=append)

    tmp_str = ' with points range {}'.format(points_range)
    runtime = (time.time() - t0) / 60
    logger.info('Econ compute complete for project points "{0}"{1}. '
                'Time elapsed: {2:.2f} min. Target output dir: {3}'
                .format(points, tmp_str if points_range else '',
                        runtime, dirout))

    # add job to reV status file.
    status = {'dirout': dirout, 'fout': fout, 'job_status': 'successful',
              'runtime': runtime, 'finput': cf_file}
    Status.make_job_file(dirout, 'econ', name, status)


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
                 points=slice(0, 100), points_range=None,
                 sites_per_worker=None, max_workers=None, timeout=1800,
                 fout='reV.h5', dirout='./out/econ_out',
                 logdir='./out/log_econ', output_request='lcoe_fcr',
                 append=False, verbose=False):
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
    sites_per_worker : int | None
        Number of sites to be analyzed in serial on a single local core.
    max_workers : int | None
        Number of workers to use on a node. None defaults to all available
        workers.
    timeout : int | float
        Number of seconds to wait for parallel run iteration to complete
        before returning zeros. Default is 1800 seconds.
    fout : str
        Target filename to dump econ outputs.
    dirout : str
        Target directory to dump econ fout.
    logdir : str
        Target directory to save log files.
    output_request : list | tuple
        Output variable requested from SAM.
    append : bool
        Flag to append econ datasets to source cf_file. This has priority
        over the fout and dirout inputs.
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

    # make a cli arg string for direct() in this module
    arg_direct = ('-p {points} '
                  '-sf {sam_files} '
                  '-cf {cf_file} '
                  '-cfy {cf_year} '
                  '{site_data}'
                  '-spw {sites_per_worker} '
                  '-fo {fout} '
                  '-do {dirout} '
                  '-lo {logdir} '
                  '-or {out_req} '
                  '{append}')
    arg_direct = arg_direct.format(
        points=SubprocessManager.s(points),
        sam_files=SubprocessManager.s(sam_files),
        cf_file=SubprocessManager.s(cf_file),
        cf_year=SubprocessManager.s(cf_year),
        site_data=s_site_data if site_data else '',
        sites_per_worker=SubprocessManager.s(sites_per_worker),
        fout=SubprocessManager.s(fout),
        dirout=SubprocessManager.s(dirout),
        logdir=SubprocessManager.s(logdir),
        out_req=SubprocessManager.s(output_request),
        append='-ap ' if append else '')

    # make a cli arg string for local() in this module
    arg_loc = ('-mw {max_workers} '
               '-to {timeout} '
               '-pr {points_range} '
               '{v}'.format(max_workers=SubprocessManager.s(max_workers),
                            timeout=SubprocessManager.s(timeout),
                            points_range=SubprocessManager.s(points_range),
                            v='-v ' if verbose else ''))

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
              help='Number of SLURM nodes for econ job. Default is 1.')
@click.option('--alloc', '-a', default='rev', type=STR,
              help='SLURM allocation account name. Default is "rev".')
@click.option('--memory', '-mem', default=None, type=INT,
              help='SLURM node memory request in GB. Default is None')
@click.option('--walltime', '-wt', default=0.5, type=float,
              help='SLURM walltime request in hours. Default is 0.5')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--module', '-mod', default=None, type=STR,
              help='Module to load')
@click.option('--conda_env', '-env', default=None, type=STR,
              help='Conda env to activate')
@click.option('--stdout_path', '-sout', default='./out/stdout', type=STR,
              help='Subprocess standard output path. Default is ./out/stdout')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def econ_slurm(ctx, nodes, alloc, memory, walltime, feature, module, conda_env,
               stdout_path, verbose):
    """Run econ on HPC via SLURM job submission."""

    name = ctx.obj['NAME']
    points = ctx.obj['POINTS']
    sam_files = ctx.obj['SAM_FILES']
    cf_file = ctx.obj['CF_FILE']
    cf_year = ctx.obj['CF_YEAR']
    site_data = ctx.obj['SITE_DATA']
    sites_per_worker = ctx.obj['SITES_PER_WORKER']
    max_workers = ctx.obj['MAX_WORKERS']
    timeout = ctx.obj['TIMEOUT']
    fout = ctx.obj['FOUT']
    dirout = ctx.obj['DIROUT']
    logdir = ctx.obj['LOGDIR']
    output_request = ctx.obj['OUTPUT_REQUEST']
    append = ctx.obj['APPEND']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # initialize an info logger on the year level
    init_mult(name, logdir, modules=[__name__, 'reV.econ.econ', 'reV.config',
                                     'reV.utilities', 'reV.SAM',
                                     'rex.utilities'],
              verbose=False)

    if append:
        pc = [None]
    else:
        pc = get_node_pc(points, sam_files, nodes)

    for i, split in enumerate(pc):
        node_name, fout_node = get_node_name_fout(name, fout, i, pc,
                                                  hpc='slurm')
        node_name = node_name.replace('gen', 'econ')

        points_range = split.split_range if split is not None else None
        cmd = get_node_cmd(node_name, sam_files, cf_file, cf_year=cf_year,
                           site_data=site_data, points=points,
                           points_range=points_range,
                           sites_per_worker=sites_per_worker,
                           max_workers=max_workers, timeout=timeout,
                           fout=fout_node,
                           dirout=dirout, logdir=logdir,
                           output_request=output_request, append=append,
                           verbose=verbose)

        status = Status.retrieve_job_status(dirout, 'econ', node_name)

        if status == 'successful':
            msg = ('Job "{}" is successful in status json found in "{}", '
                   'not re-running.'
                   .format(node_name, dirout))
        else:
            logger.info('Running reV econ on SLURM with node name "{}" for '
                        '{} (points range: {}).'
                        .format(node_name, pc, points_range))
            # create and submit the SLURM job
            slurm = SLURM(cmd, alloc=alloc, memory=memory, walltime=walltime,
                          feature=feature, name=node_name,
                          stdout_path=stdout_path, conda_env=conda_env,
                          module=module)
            if slurm.id:
                msg = ('Kicked off reV econ job "{}" (SLURM jobid #{}).'
                       .format(node_name, slurm.id))
                # add job to reV status file.
                Status.add_job(
                    dirout, 'econ', node_name, replace=True,
                    job_attrs={'job_id': slurm.id, 'hardware': 'eagle',
                               'fout': fout_node, 'dirout': dirout})
            else:
                msg = ('Was unable to kick off reV econ job "{}". '
                       'Please see the stdout error messages'
                       .format(node_name))

        click.echo(msg)
        logger.info(msg)


if __name__ == '__main__':
    main(obj={})
