"""
Generation CLI entry points.
"""
import click
from dask.distributed import Client, LocalCluster
import logging
from math import ceil
import os
import pprint
import re
import time

from reV import __testdatadir__
from reV.config.project_points import ProjectPoints, PointsControl
from reV.config.gen_config import GenConfig
from reV.generation.generation import Gen
from reV.utilities.cli_dtypes import INT, STR, SAMFILES, PROJECTPOINTS, INTLIST
from reV.utilities.exceptions import ConfigError
from reV.utilities.execution import PBS
from reV.utilities.rev_logger import init_logger


logger = logging.getLogger(__name__)


def init_gen_loggers(verbose, name, logdir='./out/log',
                     modules=['reV.SAM', 'reV.config', 'reV.generation',
                              'reV.execution']):
    """Initialize multiple loggers to a single file for the gen compute."""
    if verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    for module in modules:
        init_logger(module, log_level=log_level,
                    log_file=os.path.join(logdir, '{}.log').format(name))


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


def submit_from_config(ctx, name, year, config, verbose, i):
    """Function to submit one year from a config file.

    Parameters
    ----------
    ctx : cli.ctx
        Click context object. Use case: data = ctx.obj['key']
    name : str
        Job name.
    year : int | str
        4 digit year
    config : reV.config.GenConfig
        Generation config object.
    """
    # set the year-specific variables
    ctx.obj['RES_FILE'] = config.res_files[i]

    # if the year isn't in the name, add it before setting the file output
    match = re.match(r'.*([1-3][0-9]{3})', name)
    if not match:
        ctx.obj['FOUT'] = '{}_{}.h5'.format(name, year)
        # 8 chars for pbs job name (lim is 16, -8 for "_year_2charID")
        ctx.obj['NAME'] = '{}_{}'.format(name[:8], year)

    msg = ('Running reV generation for year: {} with resource file: {}'
           .format(year, config.res_files[i]))
    logger.debug(msg)
    click.echo(msg)

    # check to make sure that the year matches the resource file
    if str(year) not in config.res_files[i]:
        raise Exception('Resource file and year do not appear to match. '
                        'Expected the string representation of the year '
                        'to be in the resource file name. '
                        'Year: {}, Resource file: {}'
                        .format(year, config.res_files[i]))

    # invoke direct methods based on the config execution option
    if config.execution_control.option == 'local':
        sites_per_core = ceil(len(config.points_control) /
                              config.execution_control.ppn)
        ctx.obj['SITES_PER_CORE'] = sites_per_core
        ctx.invoke(local, n_workers=config.execution_control.ppn,
                   points_range=None, verbose=verbose)

    elif config.execution_control.option == 'peregrine':
        ctx.invoke(peregrine, nodes=config.execution_control.nodes,
                   alloc=config.execution_control.alloc,
                   queue=config.execution_control.queue,
                   stdout_path=os.path.join(config.dirout, 'stdout'),
                   verbose=verbose)
    else:
        raise ConfigError('Execution option not recognized: "{}"'
                          .format(config.execution_control.option))


@main.command()
@click.option('--config_file', '-c', required=True, type=STR,
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

    # Enforce verbosity if logging level is specified in the config
    if config.logging_level == logging.DEBUG:
        verbose = True

    # initialize loggers. Not SAM (will be logged in the invoked processes).
    init_gen_loggers(verbose, name, logdir=config.logdir,
                     modules=['reV.config', 'reV.generation', 'reV.utilities'])

    # log the config
    logger.debug('Running reV 2.0 generation with the following config dict: '
                 '\n{}'.format(pprint.pformat(config, indent=4)))

    # set config objects to be passed through invoke to direct methods
    ctx.obj['TECH'] = config.tech
    ctx.obj['POINTS'] = config['project_points']
    ctx.obj['SAM_FILES'] = config.sam_gen
    ctx.obj['DIROUT'] = config.dirout
    ctx.obj['LOGDIR'] = config.logdir
    ctx.obj['CF_PROFILES'] = config.write_profiles
    ctx.obj['SITES_PER_CORE'] = config.execution_control['sites_per_core']

    # submit w Dask to segregate year logs initialized in Local(), Peregrine()
    with Client(LocalCluster(n_workers=1)) as client:
        # iterate through the years in config (could be just one).
        for i, year in enumerate(config.years):
            client.submit(submit_from_config, ctx, name, year, config,
                          verbose, i)


@main.group()
@click.option('--tech', '-t', default='pv', type=STR,
              help='reV tech to analyze. Default is "pv".')
@click.option('--points', '-p', default=slice(0, 100), type=PROJECTPOINTS,
              help=('reV project points to analyze '
                    '(slice, list, or file string). '
                    'Default is slice(0, 100)'))
@click.option('--sam_files', '-sf',
              default=__testdatadir__ + '/SAM/naris_pv_1axis_inv13.json',
              type=SAMFILES, help=('SAM config files (str, dict, or list). '
                                   'Default is test SAM config.'))
@click.option('--res_file', '-rf',
              default=__testdatadir__ + '/nsrdb/ri_100_nsrdb_2012.h5',
              help='Single resource file (str). Default is test NSRDB file.')
@click.option('--sites_per_core', '-spc', default=100, type=INT,
              help=('Number of sites to run in series on a single core. '
                    'Default is 100.'))
@click.option('--fout', '-fo', default='gen_output.h5', type=STR,
              help=('Filename output specification (should be .h5). '
                    'Default is "gen_output.h5"'))
@click.option('--dirout', '-do', default='./out/gen_out', type=STR,
              help='Output directory specification. Default is ./out/gen_out')
@click.option('--logdir', '-lo', default='./out/log_gen', type=STR,
              help='Generation log file directory. Default is ./out/log_gen')
@click.option('-cfp', '--cf_profiles', is_flag=True,
              help=('Flag to output/save capacity factor profiles. '
                    'Default is not to save profiles.'))
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def direct(ctx, tech, points, sam_files, res_file, sites_per_core,
           fout, dirout, logdir, cf_profiles, verbose):
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
    ctx.obj['CF_PROFILES'] = cf_profiles
    verbose = any([verbose, ctx.obj['VERBOSE']])


@direct.command()
@click.option('--n_workers', '-nw', type=INT,
              help=('Required input: number of workers. '
                    'Use 1 for serial, None for all cores.'))
@click.option('--points_range', '-pr', default=None, type=INTLIST,
              help='Optional range list to run a subset of sites.')
@click.option('--legacy', is_flag=True,
              help='Flag to use legacy run_direct function.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def local(ctx, n_workers, points_range, legacy, verbose):
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
    cf_profiles = ctx.obj['CF_PROFILES']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    init_gen_loggers(verbose, name, logdir=logdir)

    for key, val in ctx.obj.items():
        logger.debug('ctx item: "{}" : "{}" with type "{}"'
                     .format(key, val, type(val)))
    click.echo('Kicking off reV generation local run "{}".'.format(name))
    t0 = time.time()

    if legacy:
        # Execute the Generation module directly.
        Gen.run_direct(tech=tech,
                       points=points,
                       sam_files=sam_files,
                       res_file=res_file,
                       cf_profiles=cf_profiles,
                       n_workers=n_workers,
                       sites_per_split=sites_per_core,
                       points_range=points_range,
                       fout=fout,
                       dirout=dirout,
                       return_obj=False)
    else:
        # Execute the Generation module with smart data flushing.
        Gen.run_smart(tech=tech,
                      points=points,
                      sam_files=sam_files,
                      res_file=res_file,
                      cf_profiles=cf_profiles,
                      n_workers=n_workers,
                      sites_per_split=sites_per_core,
                      points_range=points_range,
                      fout=fout,
                      dirout=dirout)

    click.echo('reV generation local run complete. Total time elapsed: '
               '{0:.2f} minutes.'.format((time.time() - t0) / 60))


@direct.command()
@click.option('--nodes', '-no', default=1, type=INT,
              help='Number of Peregrine nodes. Default is 1.')
@click.option('--alloc', '-a', default='rev', type=STR,
              help='Peregrine allocation account name. Default is "rev".')
@click.option('--queue', '-q', default='short', type=STR,
              help='Peregrine target job queue. Default is "short".')
@click.option('--stdout_path', '-sout', default='./out/stdout', type=STR,
              help='Subprocess standard output path. Default is ./out/stdout')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def peregrine(ctx, nodes, alloc, queue, stdout_path, verbose):
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
    cf_profiles = ctx.obj['CF_PROFILES']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    init_gen_loggers(verbose, name, logdir=logdir)

    if isinstance(points, (str, slice, list, tuple)):
        # create points control via points
        pp = ProjectPoints(points, sam_files, tech, res_file=res_file)
        sites_per_node = ceil(len(pp.sites) / nodes)
        pc = PointsControl(pp, sites_per_split=sites_per_node)
    else:
        raise TypeError('Generation Points input type is unrecognized: '
                        '"{}"'.format(type(points)))

    jobs = {}

    for i, split in enumerate(pc):

        # make a node name unique to the run name, year, and node number
        # 8 chars for pbs job name (lim is 16, -3 for "_2charID")
        node_name = '{}_{}'.format(name[:13], i)
        if fout.endswith('.h5'):
            # remove file extension to add additional node and year strings
            fout = fout.strip('.h5')
        # add node number to file name.
        fout_node = fout + '_node{}.h5'.format(i)

        # mark a cli arg string for main() in this module
        arg_main = ('-n {name} '.format(name=PBS.s(node_name)))

        # make a cli arg string for direct() in this module
        arg_direct = ('-t {tech} '
                      '-p {points} '
                      '-sf {sam_files} '
                      '-rf {res_file} '
                      '-spc {sites_per_core} '
                      '-fo {fout} '
                      '-do {dirout} '
                      '-lo {logdir} '
                      '{cfp} '
                      .format(tech=PBS.s(tech),
                              points=PBS.s(points),
                              sam_files=PBS.s(sam_files),
                              res_file=PBS.s(res_file),
                              sites_per_core=PBS.s(sites_per_core),
                              fout=PBS.s(fout_node),
                              dirout=PBS.s(dirout),
                              logdir=PBS.s(logdir),
                              cfp='-cfp' if cf_profiles else '',
                              ))

        # make a cli arg string for local() in this module
        arg_loc = ('-nw {n_workers} '
                   '-pr {points_range} '
                   '{v}'.format(n_workers=PBS.s(None),
                                points_range=PBS.s(split.split_range),
                                v='-v' if verbose else ''))

        # Python command that will be executed on a node
        cmd = ('python -m reV.generation.cli_gen '
               '{arg_main} direct {arg_direct} local {arg_loc}'
               .format(arg_main=arg_main,
                       arg_direct=arg_direct,
                       arg_loc=arg_loc))

        # create and submit the PBS job
        pbs = PBS(cmd, alloc=alloc, queue=queue, name=node_name,
                  stdout_path=stdout_path)
        if pbs.id:
            click.echo('Kicked off reV generation job "{}" ({}) on Peregrine.'
                       .format(node_name, pbs.id))
        else:
            click.echo('Was unable to kick of reV generation job "{}". '
                       'Please see the stdout error messages'
                       .format(node_name))
        jobs[i] = pbs

    return jobs


if __name__ == '__main__':
    main(obj={})
