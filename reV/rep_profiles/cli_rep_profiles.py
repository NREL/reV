# -*- coding: utf-8 -*-
"""
reV Representative Profiles command line interface (cli).
"""
import os
import click
import logging
import pprint
import time

from reV.config.rep_profiles_config import RepProfilesConfig
from reV.pipeline.status import Status
from reV.rep_profiles.rep_profiles import RepProfiles, AggregatedRepProfiles

from rex.utilities.execution import SLURM
from rex.utilities.cli_dtypes import STR, INT, STRLIST
from rex.utilities.loggers import init_mult
from rex.utilities.utilities import get_class_properties

logger = logging.getLogger(__name__)


@click.group()
@click.option('--name', '-n', default='reV-rep_profiles', type=STR,
              help='Job name. Default is "reV-rep_profiles".')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """reV Representative Profiles Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
def valid_config_keys():
    """
    Echo the valid RepProfiles config keys
    """
    click.echo(', '.join(get_class_properties(RepProfilesConfig)))


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV representative profiles configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV representative profiles from a config file."""
    name = ctx.obj['NAME']

    # Instantiate the config object
    config = RepProfilesConfig(config_file)

    # take name from config if not default
    if config.name.lower() != 'rev':
        name = config.name

    ctx.obj['NAME'] = name

    # Enforce verbosity if logging level is specified in the config
    if config.log_level == logging.DEBUG:
        verbose = True

    # initialize loggers
    init_mult(name, config.logdir, modules=[__name__, 'reV.config',
                                            'reV.utilities', 'rex.utilities'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV representative profiles from config '
                'file: "{}"'.format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    if config.years[0] is not None and '{}' in config.cf_dset:
        fpaths = [config.gen_fpath for _ in config.years]
        names = [name + '_{}'.format(y) for y in config.years]
        dsets = [config.cf_dset.format(y) for y in config.years]

    elif config.years[0] is not None and '{}' in config.gen_fpath:
        fpaths = [config.gen_fpath.format(y) for y in config.years]
        names = [name + '_{}'.format(y) for y in config.years]
        dsets = [config.cf_dset for _ in config.years]

    else:
        fpaths = [config.gen_fpath]
        names = [name]
        dsets = [config.cf_dset]

    for name, gen_fpath, dset in zip(names, fpaths, dsets):

        if config.execution_control.option == 'local':
            status = Status.retrieve_job_status(config.dirout, 'rep-profiles',
                                                name)
            if status != 'successful':
                Status.add_job(
                    config.dirout, 'rep-profiles', name, replace=True,
                    job_attrs={'hardware': 'local',
                               'fout': '{}.h5'.format(name),
                               'dirout': config.dirout})
                ctx.invoke(direct,
                           gen_fpath=gen_fpath,
                           rev_summary=config.rev_summary,
                           reg_cols=config.reg_cols,
                           cf_dset=dset,
                           rep_method=config.rep_method,
                           err_method=config.err_method,
                           weight=config.weight,
                           out_dir=config.dirout,
                           log_dir=config.logdir,
                           n_profiles=config.n_profiles,
                           max_workers=config.execution_control.max_workers,
                           aggregate_profiles=config.aggregate_profiles,
                           verbose=verbose)

        elif config.execution_control.option in ('eagle', 'slurm'):
            ctx.obj['NAME'] = name
            ctx.obj['GEN_FPATH'] = gen_fpath
            ctx.obj['REV_SUMMARY'] = config.rev_summary
            ctx.obj['REG_COLS'] = config.reg_cols
            ctx.obj['CF_DSET'] = dset
            ctx.obj['REP_METHOD'] = config.rep_method
            ctx.obj['ERR_METHOD'] = config.err_method
            ctx.obj['WEIGHT'] = config.weight
            ctx.obj['N_PROFILES'] = config.n_profiles
            ctx.obj['OUT_DIR'] = config.dirout
            ctx.obj['LOG_DIR'] = config.logdir
            ctx.obj['MAX_WORKERS'] = config.execution_control.max_workers
            ctx.obj['AGGREGATE_PROFILES'] = config.aggregate_profiles
            ctx.obj['VERBOSE'] = verbose

            ctx.invoke(slurm,
                       alloc=config.execution_control.allocation,
                       memory=config.execution_control.memory,
                       walltime=config.execution_control.walltime,
                       feature=config.execution_control.feature,
                       conda_env=config.execution_control.conda_env,
                       module=config.execution_control.module)


@main.group(invoke_without_command=True)
@click.option('--gen_fpath', '-g', type=click.Path(exists=True), required=True,
              help='Filepath to reV gen file.')
@click.option('--rev_summary', '-r', type=click.Path(exists=True),
              required=True, help='Filepath to reV SC summary (agg) file.')
@click.option('--reg_cols', '-rc', type=STRLIST, default=None,
              help='List of column rev summary column labels to define '
              'regions to get rep profiles for.')
@click.option('--cf_dset', '-cf', type=str, default='cf_profile',
              help='Capacity factor dataset in gen_fpath to get profiles from')
@click.option('--rep_method', '-rm', type=STR, default='meanoid',
              help='String identifier for representative method '
              '(e.g. meanoid, medianoid).')
@click.option('--err_method', '-em', type=STR, default='rmse',
              help='String identifier for error method '
              '(e.g. rmse, mae, mbe).')
@click.option('--weight', '-w', type=STR, default='gid_counts',
              help='The supply curve column to use for a weighted average in '
              'the representative profile meanoid algorithm. '
              'Default weighting factor is "gid_counts".')
@click.option('--n_profiles', '-np', type=INT, default=1,
              help='Number of representative profiles to save.')
@click.option('--out_dir', '-od', type=STR, default='./',
              help='Directory to save rep profile output h5.')
@click.option('--log_dir', '-ld', type=STR, default='./logs/',
              help='Directory to save rep profile logs.')
@click.option('--max_workers', '-mw', type=INT, default=None,
              help='Number of parallel workers. 1 will run in serial. '
              'None will use all available.')
@click.option('-agg', '--aggregate_profiles', is_flag=True,
              help='Flag to calculate the aggregate (weighted meanoid) '
              'profile for each supply curve point. This behavior is instead '
              'of finding the single profile per region closest to the '
              'meanoid.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def direct(ctx, gen_fpath, rev_summary, reg_cols, cf_dset, rep_method,
           err_method, weight, n_profiles, out_dir, log_dir, max_workers,
           aggregate_profiles, verbose):
    """reV representative profiles CLI."""
    name = ctx.obj['NAME']
    ctx.obj['GEN_FPATH'] = gen_fpath
    ctx.obj['REV_SUMMARY'] = rev_summary
    ctx.obj['REG_COLS'] = reg_cols
    ctx.obj['CF_DSET'] = cf_dset
    ctx.obj['REP_METHOD'] = rep_method
    ctx.obj['ERR_METHOD'] = err_method
    ctx.obj['WEIGHT'] = weight
    ctx.obj['N_PROFILES'] = n_profiles
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['LOG_DIR'] = log_dir
    ctx.obj['MAX_WORKERS'] = max_workers
    ctx.obj['AGGREGATE_PROFILES'] = aggregate_profiles
    ctx.obj['VERBOSE'] = verbose

    if ctx.invoked_subcommand is None:
        t0 = time.time()
        init_mult(name, log_dir, modules=[__name__, 'reV.rep_profiles'],
                  verbose=verbose)

        fn_out = '{}.h5'.format(name)
        fout = os.path.join(out_dir, fn_out)

        if aggregate_profiles:
            AggregatedRepProfiles.run(gen_fpath, rev_summary, cf_dset=cf_dset,
                                      weight=weight, fout=fout,
                                      max_workers=max_workers)
        else:
            RepProfiles.run(gen_fpath, rev_summary, reg_cols, cf_dset=cf_dset,
                            rep_method=rep_method, err_method=err_method,
                            weight=weight, fout=fout, n_profiles=n_profiles,
                            max_workers=max_workers)

        runtime = (time.time() - t0) / 60
        logger.info('reV representative profiles complete. '
                    'Time elapsed: {:.2f} min. Target output dir: {}'
                    .format(runtime, out_dir))

        status = {'dirout': out_dir, 'fout': fn_out,
                  'job_status': 'successful',
                  'runtime': runtime,
                  'finput': [gen_fpath, rev_summary]}
        Status.make_job_file(out_dir, 'rep-profiles', name, status)


def get_node_cmd(name, gen_fpath, rev_summary, reg_cols, cf_dset, rep_method,
                 err_method, weight, n_profiles, out_dir, log_dir, max_workers,
                 aggregate_profiles, verbose):
    """Get a CLI call command for the rep profiles cli."""

    args = ('-g {gen_fpath} '
            '-r {rev_summary} '
            '-rc {reg_cols} '
            '-cf {cf_dset} '
            '-rm {rep_method} '
            '-em {err_method} '
            '-w {weight} '
            '-np {n_profiles} '
            '-od {out_dir} '
            '-ld {log_dir} '
            '-mw {max_workers} '
            )

    args = args.format(gen_fpath=SLURM.s(gen_fpath),
                       rev_summary=SLURM.s(rev_summary),
                       reg_cols=SLURM.s(reg_cols),
                       cf_dset=SLURM.s(cf_dset),
                       rep_method=SLURM.s(rep_method),
                       err_method=SLURM.s(err_method),
                       weight=SLURM.s(weight),
                       n_profiles=SLURM.s(n_profiles),
                       out_dir=SLURM.s(out_dir),
                       log_dir=SLURM.s(log_dir),
                       max_workers=SLURM.s(max_workers),
                       )

    if aggregate_profiles:
        args += '-agg '

    if verbose:
        args += '-v '

    cmd = ('python -m reV.rep_profiles.cli_rep_profiles -n {} direct {}'
           .format(SLURM.s(name), args))

    return cmd


@direct.command()
@click.option('--alloc', '-a', required=True, type=STR,
              help='SLURM allocation account name.')
@click.option('--memory', '-mem', default=None, type=INT,
              help='SLURM node memory request in GB. Default is None')
@click.option('--walltime', '-wt', default=1.0, type=float,
              help='SLURM walltime request in hours for single year '
              'rep_profiles run. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--conda_env', '-env', default=None, type=STR,
              help='Conda env to activate')
@click.option('--module', '-mod', default=None, type=STR,
              help='Module to load')
@click.option('--stdout_path', '-sout', default=None, type=STR,
              help='Subprocess standard output path. Default is in out_dir.')
@click.pass_context
def slurm(ctx, alloc, memory, walltime, feature, conda_env, module,
          stdout_path):
    """slurm (Eagle) submission tool for reV representative profiles."""

    name = ctx.obj['NAME']
    gen_fpath = ctx.obj['GEN_FPATH']
    rev_summary = ctx.obj['REV_SUMMARY']
    reg_cols = ctx.obj['REG_COLS']
    cf_dset = ctx.obj['CF_DSET']
    rep_method = ctx.obj['REP_METHOD']
    err_method = ctx.obj['ERR_METHOD']
    weight = ctx.obj['WEIGHT']
    n_profiles = ctx.obj['N_PROFILES']
    out_dir = ctx.obj['OUT_DIR']
    log_dir = ctx.obj['LOG_DIR']
    max_workers = ctx.obj['MAX_WORKERS']
    aggregate_profiles = ctx.obj['AGGREGATE_PROFILES']
    verbose = ctx.obj['VERBOSE']

    if stdout_path is None:
        stdout_path = os.path.join(log_dir, 'stdout/')

    cmd = get_node_cmd(name, gen_fpath, rev_summary, reg_cols, cf_dset,
                       rep_method, err_method, weight, n_profiles,
                       out_dir, log_dir, max_workers, aggregate_profiles,
                       verbose)

    status = Status.retrieve_job_status(out_dir, 'rep-profiles', name)
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'
               .format(name, out_dir))
    else:
        logger.info('Running reV SC rep profiles on SLURM with '
                    'node name "{}"'.format(name))
        slurm = SLURM(cmd, alloc=alloc, memory=memory,
                      walltime=walltime, feature=feature,
                      name=name, stdout_path=stdout_path,
                      conda_env=conda_env, module=module)
        if slurm.id:
            msg = ('Kicked off reV rep profiles job "{}" '
                   '(SLURM jobid #{}).'
                   .format(name, slurm.id))
            Status.add_job(
                out_dir, 'rep-profiles', name, replace=True,
                job_attrs={'job_id': slurm.id, 'hardware': 'eagle',
                           'fout': '{}.h5'.format(name), 'dirout': out_dir})
        else:
            msg = ('Was unable to kick off reV rep profiles job "{}". '
                   'Please see the stdout error messages'
                   .format(name))
    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Rep Profiles CLI.')
        raise
