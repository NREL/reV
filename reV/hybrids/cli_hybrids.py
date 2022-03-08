# -*- coding: utf-8 -*-
# pylint: disable=all
"""
reV Representative Profiles command line interface (cli).
"""
import os
import click
import logging
import pprint
import time

from reV.config.hybrids_config import HybridsConfig
from reV.pipeline.status import Status
from reV.hybrids.hybrids import Hybridization, SOLAR_PREFIX, WIND_PREFIX
from reV.utilities import ModuleName
from reV import __version__

from rex.utilities.hpc import SLURM
from rex.utilities.cli_dtypes import STR, INT, FLOATLIST
from rex.utilities.loggers import init_mult
from rex.utilities.utilities import get_class_properties, dict_str_load

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default=os.path.basename(os.getcwd()),
              type=STR, show_default=True, help='reV Hybrids job name.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """reV Hybridization Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
def valid_config_keys():
    """
    Echo the valid Hybrids config keys
    """
    click.echo(', '.join(get_class_properties(HybridsConfig)))


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV hybrid profiles configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV hybridization from a config file."""
    # Instantiate the config object
    config = HybridsConfig(config_file)

    # take name from config
    name = ctx.obj['NAME'] = config.name

    # Enforce verbosity if logging level is specified in the config
    if config.log_level == logging.DEBUG:
        verbose = True

    # initialize loggers
    init_mult(name, config.log_directory, modules=[__name__, 'reV', 'rex'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV hybridization from config '
                'file: "{}"'.format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.log_directory))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    names, solar_fpaths, wind_fpaths = _get_paths_from_config(config, name)

    for name, solar_fpath, wind_fpath in zip(names, solar_fpaths, wind_fpaths):

        ctx.obj['NAME'] = name
        if config.execution_control.option == 'local':
            status = Status.retrieve_job_status(config.dirout,
                                                module=ModuleName.HYBRIDS,
                                                job_name=name)

            if status != 'successful':
                Status.add_job(
                    config.dirout, module=ModuleName.HYBRIDS,
                    job_name=name, replace=True,
                    job_attrs={'hardware': 'local',
                               'fout': '{}.h5'.format(name),
                               'dirout': config.dirout})

                ctx.invoke(direct,
                           solar_fpath=solar_fpath,
                           wind_fpath=wind_fpath,
                           allow_solar_only=config.allow_solar_only,
                           allow_wind_only=config.allow_wind_only,
                           fillna=config.fillna,
                           limits=config.limits,
                           ratio_bounds=config.ratio_bounds,
                           ratio=config.ratio,
                           out_dir=config.dirout,
                           log_dir=config.log_directory,
                           verbose=verbose)

        elif config.execution_control.option in ('eagle', 'slurm'):
            ctx.obj['SOLAR_FPATH'] = solar_fpath
            ctx.obj['WIND_FPATH'] = wind_fpath
            ctx.obj['ALLOW_SOLAR_ONLY'] = config.allow_solar_only
            ctx.obj['ALLOW_WIND_ONLY'] = config.allow_wind_only
            ctx.obj['FILLNA'] = config.fillna
            ctx.obj['LIMITS'] = config.limits
            ctx.obj['RATIO_BOUNDS'] = config.ratio_bounds
            ctx.obj['RATIO'] = config.ratio
            ctx.obj['OUT_DIR'] = config.dirout
            ctx.obj['LOG_DIR'] = config.log_directory
            ctx.obj['VERBOSE'] = verbose

            ctx.invoke(slurm,
                       alloc=config.execution_control.allocation,
                       memory=config.execution_control.memory,
                       walltime=config.execution_control.walltime,
                       feature=config.execution_control.feature,
                       conda_env=config.execution_control.conda_env,
                       module=config.execution_control.module,
                       sh_script=config.execution_control.sh_script)


def _get_paths_from_config(config, name):
    """Pair solar and wind files and corresponding process names. """

    solar_glob_paths, wind_glob_paths = config.solar_fpath, config.wind_fpath
    all_years = set(solar_glob_paths) | set(wind_glob_paths)
    common_years = set(solar_glob_paths) & set(wind_glob_paths)
    if not all_years:
        msg = "No files found that match the input: {!r} and/or {!r}"
        e = msg.format(config['solar_fpath'], config['wind_fpath'])
        logger.error(e)
        raise RuntimeError(e)

    solar_fpaths = []
    wind_fpaths = []
    names = []
    for year in all_years:
        if year not in common_years:
            msg = ("No corresponding {} file found for {} input file "
                   "(year: '{}'): {!r}. No hybridization performed for "
                   "this input!")
            resources = (['solar', 'wind'] if year not in solar_glob_paths else
                         ['wind', 'solar'])
            paths = (solar_glob_paths.get(year, [])
                     + wind_glob_paths.get(year, []))
            w = msg.format(*resources, paths, year)
            logger.warning(w)
            RuntimeWarning(w)
            continue

        for fpaths in (solar_glob_paths, wind_glob_paths):
            if len(fpaths[year]) > 1:
                msg = ("Ambiguous number of files found for year '{}': {!r} "
                       "Please ensure there is only one input file per year. "
                       "No hybridization performed for this input!")
                w = msg.format(year, fpaths[year])
                logger.warning(w)
                RuntimeWarning(w)
                break
        else:
            solar_fpaths += solar_glob_paths[year]
            wind_fpaths += wind_glob_paths[year]
            names += ["{}_{}".format(name, year) if year is not None else name]

    return names, solar_fpaths, wind_fpaths


@main.group(invoke_without_command=True)
@click.option('--solar_fpath', '-s', type=click.Path(exists=True),
              required=True, help='Filepath to solar rep profile file.')
@click.option('--wind_fpath', '-w', type=click.Path(exists=True),
              required=True, help='Filepath to wind rep profile file.')
@click.option('-so', '--allow_solar_only', is_flag=True,
              help='Flag to to allow SC points with only solar capcity '
              '(no wind) in hybrid profiles.')
@click.option('-wo', '--allow_wind_only', is_flag=True,
              help='Flag to to allow SC points with only wind capcity '
              '(no solard) in hybrid profiles.')
@click.option('--fillna', '-fna', type=STR, default=None,
              show_default=True,
              help=('String representation of a dictionary of fill values '
                    'for merged columns "{{\'col_name\': val}}" where '
                    '\'col_name\' is the name of a merged column '
                    '(prefixed by {!r} or {!r}) and "val" is the values '
                    'used to fill any n.a. values resulting from a merge '
                    '(e.g. wind column values for a row consisting purely '
                    'of solar capacity).'
                    .format(SOLAR_PREFIX, WIND_PREFIX)))
@click.option('--limits', '-l', type=STR, default=None,
              show_default=True,
              help='String representation of a dictionary specifying a '
                   'mapping of "{{\'colum_name\': max_value}}" representing '
                   'the upper limit (maximum value) for the values of a '
                   'column in the merged meta. For example, '
                   '`--limits "{{\'solar_capacity\': 100}}"` '
                   'would limit all the values of the solar capacity in the '
                   'merged meta to a maximum value of 100. This limit is '
                   'applied *BEFORE* ratio calculations. The names of '
                   'the columns should match the column names in the merged '
                   'meta, so they are likely prefixed by {!r} or {!r}. '
                   'By default, None (no limits applied).'
                   .format(SOLAR_PREFIX, WIND_PREFIX))
@click.option('--ratio_bounds', '-rb', type=FLOATLIST, default=None,
              show_default=True,
              help='List of two floats representing the lower and upper '
                   'bounds on the ratio of the two columns specified with the '
                   '`--ratio` input, which is set to "solar_capacity/'
                   'wind_capacity" by default. The same value can be used as '
                   'the upper and lower bound to represent a single ratio. '
                   'For example, `--ratio_bounds=[0.5, 1.5]` would adjust the '
                   'values of both of the `--ratio` columns such that their '
                   'ratio is always between half and double (e.g., no value '
                   'would be more than double the other). To specify a single '
                   'ratio value, use the same value as the upper and lower '
                   'bound. For example, `--ratio_bounds=(1, 1)` would adjust '
                   'the values of both of the `ratio` columns such that their '
                   'ratio is always equal.')
@click.option('--ratio', '-r', type=STR,
              default='solar_capacity/wind_capacity',
              show_default=True,
              help='String representation of the ratio calculation used to '
                   'calculate the ratio that is limited by the `ratio_bounds`'
                   'input. This string must be in the form '
                   '"numerator_column_name/denominator_column_name". For '
                   'example, `--ratio \'solar_capacity/wind_capacity\'` would '
                   'limit the ratio of the solar to wind capacities as '
                   'specified by the `ratio_bounds` input. If `ratio_bounds` '
                   'is not specified, this input does nothing. The names of '
                   'the columns should match the column names in the merged '
                   'meta, so they are likely prefixed by {!r} or {!r}.'
                   .format(SOLAR_PREFIX, WIND_PREFIX))
@click.option('--out_dir', '-od', type=STR, default='./',
              show_default=True,
              help='Directory to save rep profile output h5.')
@click.option('--log_dir', '-ld', type=STR, default='./logs/',
              show_default=True,
              help='Directory to save rep profile logs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def direct(ctx, solar_fpath, wind_fpath, allow_solar_only, allow_wind_only,
           fillna, limits, ratio_bounds, ratio, out_dir, log_dir, verbose):
    """reV hybridization CLI."""
    name = ctx.obj['NAME']
    ctx.obj['SOLAR_FPATH'] = solar_fpath
    ctx.obj['WIND_FPATH'] = wind_fpath
    ctx.obj['ALLOW_SOLAR_ONLY'] = allow_solar_only
    ctx.obj['ALLOW_WIND_ONLY'] = allow_wind_only
    ctx.obj['FILLNA'] = fillna
    ctx.obj['LIMITS'] = limits
    ctx.obj['RATIO_BOUNDS'] = ratio_bounds
    ctx.obj['RATIO'] = ratio
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['LOG_DIR'] = log_dir
    ctx.obj['VERBOSE'] = verbose

    if ctx.invoked_subcommand is None:
        t0 = time.time()
        init_mult(name, log_dir, modules=[__name__, 'reV', 'rex'],
                  verbose=verbose)

        fn_out = '{}.h5'.format(name)
        fout = os.path.join(out_dir, fn_out)

        if isinstance(fillna, str):
            fillna = dict_str_load(fillna)

        if isinstance(limits, str):
            limits = dict_str_load(limits)

        try:
            Hybridization(
                solar_fpath, wind_fpath,
                allow_solar_only=allow_solar_only,
                allow_wind_only=allow_wind_only,
                fillna=fillna, limits=limits,
                ratio_bounds=ratio_bounds,
                ratio=ratio
            ).run_all(fout=fout)

        except Exception as e:
            msg = ('Hybridization of rep profiles failed. Received the '
                   'following error:\n{}'.format(e))
            logger.exception(msg)
            raise RuntimeError(msg) from e

        runtime = (time.time() - t0) / 60
        logger.info('reV hybrid profiles complete. '
                    'Time elapsed: {:.2f} min. Target output dir: {}'
                    .format(runtime, out_dir))

        status = {'dirout': out_dir, 'fout': fn_out,
                  'job_status': 'successful',
                  'runtime': runtime,
                  'finput': [solar_fpath, wind_fpath]}
        Status.make_job_file(out_dir, ModuleName.HYBRIDS, name, status)


def get_node_cmd(ctx):
    """Get a CLI call command for the hybrids cli."""

    name = ctx.obj['NAME']
    solar_fpath = ctx.obj['SOLAR_FPATH']
    wind_fpath = ctx.obj['WIND_FPATH']
    allow_solar_only = ctx.obj['ALLOW_SOLAR_ONLY']
    allow_wind_only = ctx.obj['ALLOW_WIND_ONLY']
    fillna = ctx.obj['FILLNA']
    limits = ctx.obj['LIMITS']
    ratio_bounds = ctx.obj['RATIO_BOUNDS']
    ratio = ctx.obj['RATIO']
    out_dir = ctx.obj['OUT_DIR']
    log_dir = ctx.obj['LOG_DIR']
    verbose = ctx.obj['VERBOSE']

    args = ['-s {}'.format(SLURM.s(solar_fpath)),
            '-w {}'.format(SLURM.s(wind_fpath)),
            '-fna {}'.format(SLURM.s(fillna)),
            '-l {}'.format(SLURM.s(limits)),
            '-od {}'.format(SLURM.s(out_dir)),
            '-ld {}'.format(SLURM.s(log_dir)),
            ]

    if ratio_bounds is not None:
        args.append('-rb {}'.format(SLURM.s(ratio_bounds)))

    if ratio is not None:
        args.append('-r {}'.format(SLURM.s(ratio)))

    if allow_solar_only:
        args.append('-so')

    if allow_wind_only:
        args.append('-wo')

    if verbose:
        args.append('-v')

    cmd = ('python -m reV.hybrids.cli_hybrids -n {} direct {}'
           .format(SLURM.s(name), ' '.join(args)))
    logger.debug('Creating the following command line call:\n\t{}'.format(cmd))

    return cmd


@direct.command()
@click.option('--alloc', '-a', required=True, type=STR,
              help='SLURM allocation account name.')
@click.option('--memory', '-mem', default=None, type=INT,
              show_default=True,
              help='SLURM node memory request in GB. Default is None')
@click.option('--walltime', '-wt', default=1.0, type=float,
              show_default=True,
              help='SLURM walltime request in hours for single year '
              'rep_profiles run. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR,
              show_default=True,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--conda_env', '-env', default=None, type=STR,
              show_default=True,
              help='Conda env to activate')
@click.option('--module', '-mod', default=None, type=STR,
              show_default=True,
              help='Module to load')
@click.option('--stdout_path', '-sout', default=None, type=STR,
              show_default=True,
              help='Subprocess standard output path. Default is in out_dir.')
@click.option('--sh_script', '-sh', default=None, type=STR,
              show_default=True,
              help='Extra shell script commands to run before the reV call.')
@click.pass_context
def slurm(ctx, alloc, memory, walltime, feature, conda_env, module,
          stdout_path, sh_script):
    """slurm (Eagle) submission tool for reV representative profiles."""

    name = ctx.obj['NAME']
    out_dir = ctx.obj['OUT_DIR']
    log_dir = ctx.obj['LOG_DIR']

    if stdout_path is None:
        stdout_path = os.path.join(log_dir, 'stdout/')

    cmd = get_node_cmd(ctx)

    if sh_script:
        cmd = sh_script + '\n' + cmd

    slurm_manager = ctx.obj.get('SLURM_MANAGER', None)
    if slurm_manager is None:
        slurm_manager = SLURM()
        ctx.obj['SLURM_MANAGER'] = slurm_manager

    status = Status.retrieve_job_status(out_dir, module=ModuleName.HYBRIDS,
                                        job_name=name, hardware='eagle',
                                        subprocess_manager=slurm_manager)

    msg = 'Hybrids CLI failed to submit jobs!'
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'
               .format(name, out_dir))
    elif 'fail' not in str(status).lower() and status is not None:
        msg = ('Job "{}" was found with status "{}", not resubmitting'
               .format(name, status))
    else:
        logger.info('Running reV SC hybridization on SLURM with '
                    'node name "{}"'.format(name))
        out = slurm_manager.sbatch(cmd, alloc=alloc, memory=memory,
                                   walltime=walltime, feature=feature,
                                   name=name, stdout_path=stdout_path,
                                   conda_env=conda_env, module=module)[0]
        if out:
            msg = ('Kicked off reV hybridization job "{}" '
                   '(SLURM jobid #{}).'
                   .format(name, out))
        Status.add_job(
            out_dir, module=ModuleName.HYBRIDS, job_name=name, replace=True,
            job_attrs={'job_id': out, 'hardware': 'eagle',
                       'fout': '{}.h5'.format(name), 'dirout': out_dir})

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Hybridization CLI.')
        raise
