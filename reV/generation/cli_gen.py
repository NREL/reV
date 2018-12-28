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
from reV.cli import INT, PROJECTPOINTS, INTLIST, SAMFILES, STR


logger = logging.getLogger(__name__)


@click.group()
@click.option('--name', '-n', default='reV', type=STR,
              help='Job name. Default is "reV".')
@click.option('--tech', '-t', default='pv', type=STR,
              help='reV tech to analyze. Default is "pv".')
@click.option('--points', '-p', default=slice(0, 100), type=PROJECTPOINTS,
              help=('reV project points to analyze (slice or str). '
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
@click.option('--logdir', '-lo', default='./out/log', type=STR,
              help='Log file output directory. Default is ./out/log')
@click.option('-cfp', '--cf_profiles', is_flag=True,
              help=('Flag to output/save capacity factor profiles. '
                    'Default is not to save profiles.'))
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, tech, points, sam_files, res_file, sites_per_core,
         fout, dirout, logdir, cf_profiles, verbose):
    """reV 2.0 generation command line interface."""
    ctx.ensure_object(dict)
    ctx.obj['name'] = name
    ctx.obj['tech'] = tech
    ctx.obj['points'] = points
    ctx.obj['sam_files'] = sam_files
    ctx.obj['res_file'] = res_file
    ctx.obj['sites_per_core'] = sites_per_core
    ctx.obj['fout'] = fout
    ctx.obj['dirout'] = dirout
    ctx.obj['logdir'] = logdir
    ctx.obj['cf_profiles'] = cf_profiles
    ctx.obj['verbose'] = verbose


@main.command()
@click.option('--n_workers', '-nw', type=INT,
              help=('Required input: number of workers. '
                    'Use 1 for serial, None for all cores.'))
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
    sites_per_core = ctx.obj['sites_per_core']
    fout = ctx.obj['fout']
    dirout = ctx.obj['dirout']
    logdir = ctx.obj['logdir']
    cf_profiles = ctx.obj['cf_profiles']
    verbose = any([verbose, ctx.obj['verbose']])

    if verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    log_modules = [__name__, 'reV.SAM', 'reV.config', 'reV.generation',
                   'reV.execution']
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    for module in log_modules:
        init_logger(module, log_level=log_level,
                    log_file=os.path.join(logdir, '{}.log'.format(name)))

    logger.debug('Executing local cli with '
                 'n_workers={} ({}) points_range={} ({})'
                 .format(n_workers, type(n_workers),
                         points_range, type(points_range)))

    for key, val in ctx.obj.items():
        logger.debug('ctx item: {} = {} with type {}'
                     .format(key, val, type(val)))

    Gen.direct(tech=tech,
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


@main.command()
@click.option('--nodes', '-no', default=1, type=INT,
              help='Number of PBS nodes. Default is 1.')
@click.option('--alloc', '-a', default='rev', type=STR,
              help='PBS HPC allocation account name. Default is "rev".')
@click.option('--queue', '-q', default='short', type=STR,
              help='PBS HPC target job queue. Default is "short".')
@click.option('--stdout_path', '-sout', default='./out/stdout', type=STR,
              help='Subprocess standard output path. Default is ./out/stdout')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def pbs(ctx, nodes, alloc, queue, stdout_path, verbose):
    """Run reV 2.0 generation via PBS job submission."""

    name = ctx.obj['name']
    tech = ctx.obj['tech']
    points = ctx.obj['points']
    sam_files = ctx.obj['sam_files']
    res_file = ctx.obj['res_file']
    sites_per_core = ctx.obj['sites_per_core']
    fout = ctx.obj['fout']
    dirout = ctx.obj['dirout']
    logdir = ctx.obj['logdir']
    cf_profiles = ctx.obj['cf_profiles']
    verbose = any([verbose, ctx.obj['verbose']])

    if verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    log_modules = [__name__, 'reV.SAM', 'reV.config', 'reV.generation',
                   'reV.execution']
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    for module in log_modules:
        init_logger(module, log_level=log_level,
                    log_file=os.path.join(logdir,
                                          '{}_PBS.log'
                                          .format(name)))

    pp = ProjectPoints(points, sam_files, tech, res_file=res_file)
    sites_per_node = ceil(len(pp.sites) / nodes)
    pc = PointsControl(pp, sites_per_split=sites_per_node)

    jobs = {}

    for i, split in enumerate(pc):
        node_name = '{}_{}'.format(name, i)
        if fout.endswith('.h5'):
            fout = fout.strip('.h5')
        arg_main = ('-n={name} '
                    '-t={tech} '
                    '-p={points} '
                    '-sf={sam_files} '
                    '-rf={res_file} '
                    '-spc={sites_per_core} '
                    '-fo={fout} '
                    '-do={dirout} '
                    '-lo={logdir} '
                    '{cfp} '
                    .format(name=PBS.s(node_name),
                            tech=PBS.s(tech),
                            points=PBS.s(points),
                            sam_files=PBS.s(sam_files),
                            res_file=PBS.s(res_file),
                            sites_per_core=PBS.s(sites_per_core),
                            fout=PBS.s(fout + '_node_{}.h5'.format(i)),
                            dirout=PBS.s(dirout),
                            logdir=PBS.s(logdir),
                            cfp='-cfp' if cf_profiles else '',
                            ))

        arg_loc = ('-nw={n_workers} '
                   '-pr={points_range} '
                   '{v}'.format(n_workers=PBS.s(None),
                                points_range=PBS.s(split.split_range),
                                v='-v' if verbose else ''))

        cmd = ('python -m reV.generation.cli_gen {arg_main} local {arg_loc}'
               .format(arg_main=arg_main, arg_loc=arg_loc))

        pbs = PBS(cmd, alloc=alloc, queue=queue, name=node_name,
                  stdout_path=stdout_path)

        jobs[i] = pbs

    return jobs


if __name__ == '__main__':
    main(obj={})
