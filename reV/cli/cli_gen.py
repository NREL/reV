"""
Generation CLI entry points.
"""
import click
import logging
from math import ceil
import os

from reV.config.config import ProjectPoints, PointsControl
from reV.execution.execution import PBS
from reV.generation.generation import Gen
from reV.rev_logger import init_logger
from reV import __testdatadir__
from reV.cli.dtypes import INT, PROJECTPOINTS, INTLIST, STRLIST, SAMFILES


logger = logging.getLogger(__name__)


@click.group()
@click.option('--name', '-n', default='reV', type=str,
              help='Job name.')
@click.option('--tech', '-t', default='pv', type=str,
              help='reV tech to analyze.')
@click.option('--points', '-p', default=slice(0, 100), type=PROJECTPOINTS,
              help='reV project points to analyze (slice or str).')
@click.option('--sam_files', '-sf',
              default=__testdatadir__ + '/SAM/naris_pv_1axis_inv13.json',
              type=SAMFILES, help='SAM config files (str, dict, or list).')
@click.option('--res_file', '-rf',
              default=__testdatadir__ + '/nsrdb/ri_100_nsrdb_2012.h5',
              help='Single resource file (str).')
@click.option('--output_request', '-or', default=['cf_mean'], type=STRLIST,
              help='Output variable requests from SAM (list or tuple)')
@click.option('--sites_per_core', '-spc', default=100, type=INT,
              help='Number of sites to run in series on a single core.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def main(ctx, name, tech, points, sam_files, res_file, output_request,
         sites_per_core, verbose):
    """reV 2.0 generation command line interface."""
    ctx.ensure_object(dict)
    ctx.obj['name'] = name
    ctx.obj['tech'] = tech
    ctx.obj['points'] = points
    ctx.obj['sam_files'] = sam_files
    ctx.obj['res_file'] = res_file
    ctx.obj['output_request'] = output_request
    ctx.obj['sites_per_core'] = sites_per_core
    ctx.obj['verbose'] = verbose


@main.command()
@click.option('--n_workers', '-nw', default=1, type=INT,
              help='Number of workers to use (default is serial compute).')
@click.option('--points_range', '-pr', default=None, type=INTLIST,
              help='Optional range list to run a subset of sites.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def local(ctx, n_workers, points_range, verbose):
    """Run reV 2.0 generation on local workers."""

    for key, val in ctx.obj.items():
        if isinstance(val, str):
            ctx.obj[key] = val.lstrip('=')

    name = ctx.obj['name']
    tech = ctx.obj['tech']
    points = ctx.obj['points']
    sam_files = ctx.obj['sam_files']
    res_file = ctx.obj['res_file']
    output_request = ctx.obj['output_request']
    sites_per_core = ctx.obj['sites_per_core']
    verbose = any([verbose, ctx.obj['verbose']])

    if verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    log_modules = [__name__, 'reV.SAM', 'reV.config', 'reV.generation',
                   'reV.execution']
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    for module in log_modules:
        init_logger(module, log_level=log_level,
                    log_file=os.path.join(log_dir, '{}.log'.format(name)))

    logger.debug('Executing local cli with '
                 'n_workers={} ({}) points_range={} ({})'
                 .format(n_workers, type(n_workers),
                         points_range, type(points_range)))

    for key, val in ctx.obj.items():
        logger.debug('ctx item: {} = {} with type {}'
                     .format(key, val, type(val)))

    out = Gen.direct(tech=tech,
                     points=points,
                     sam_files=sam_files,
                     res_file=res_file,
                     output_request=output_request,
                     n_workers=n_workers,
                     sites_per_split=sites_per_core,
                     points_range=points_range)
    return out


@main.command()
@click.option('--nodes', '-no', default=3, type=INT,
              help='Number of PBS nodes')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def pbs(ctx, nodes, verbose):
    """Run reV 2.0 generation via PBS job submission."""

    name = ctx.obj['name']
    tech = ctx.obj['tech']
    points = ctx.obj['points']
    sam_files = ctx.obj['sam_files']
    res_file = ctx.obj['res_file']
    output_request = ctx.obj['output_request']
    sites_per_core = ctx.obj['sites_per_core']
    verbose = any([verbose, ctx.obj['verbose']])

    if verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    log_modules = [__name__, 'reV.SAM', 'reV.config', 'reV.generation',
                   'reV.execution']
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    for module in log_modules:
        init_logger(module, log_level=log_level,
                    log_file=os.path.join(log_dir,
                                          '{}_PBS.log'
                                          .format(name)))

    pp = ProjectPoints(points, sam_files, tech, res_file=res_file)
    sites_per_node = ceil(len(pp.sites) / nodes)
    pc = PointsControl(pp, sites_per_split=sites_per_node)

    jobs = {}

    for i, split in enumerate(pc):
        node_name = '{}_{}'.format(name, i)
        arg_main = ('-n={name} '
                    '-t={tech} '
                    '-p={points} '
                    '-sf={sam_files} '
                    '-rf={res_file} '
                    '-or={output_request} '
                    '-spc={sites_per_core} '
                    .format(name=PBS.s(node_name),
                            tech=PBS.s(tech),
                            points=PBS.s(points),
                            sam_files=PBS.s(sam_files),
                            res_file=PBS.s(res_file),
                            output_request=PBS.s(output_request),
                            sites_per_core=PBS.s(sites_per_core)))

        arg_loc = ('-nw={n_workers} '
                   '-pr={points_range} '
                   '{v}'.format(n_workers=PBS.s(None),
                                points_range=PBS.s(split.split_range),
                                v='-v' if verbose else ''))

        cmd = ('python -m reV.generation.generation {arg_main} local {arg_loc}'
               .format(arg_main=arg_main, arg_loc=arg_loc))

        pbs = PBS(cmd, name=node_name)

        jobs[i] = pbs

    return jobs
