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
import json

from reV.config.hybrids_config import HybridsConfig, convert_str_keys_to_tuples
from reV.pipeline.status import Status
from reV.hybrids.hybrids import Hybridization, SOLAR_PREFIX, WIND_PREFIX
from reV import __version__

from rex.utilities.hpc import SLURM
from rex.utilities.cli_dtypes import STR, INT
from rex.utilities.loggers import init_mult
from rex.utilities.utilities import get_class_properties, dict_str_load
from reV.utilities.exceptions import InputError

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default='reV-hybrids', type=STR,
              show_default=True,
              help='Job name. Default is "reV-hybrids".')
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
    name = ctx.obj['NAME']

    # Instantiate the config object
    config = HybridsConfig(config_file)

    # take name from config if not default
    if config.name.lower() != 'rev':
        name = config.name
        ctx.obj['NAME'] = name

    # Enforce verbosity if logging level is specified in the config
    if config.log_level == logging.DEBUG:
        verbose = True

    # initialize loggers
    init_mult(name, config.logdir, modules=[__name__, 'reV', 'rex'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV hybridization from config '
                'file: "{}"'.format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    names, solar_fpaths, wind_fpaths = _get_paths_from_config(config, name)

    for name, solar_fpath, wind_fpath in zip(names, solar_fpaths, wind_fpaths):

        ctx.obj['NAME'] = name
        if config.execution_control.option == 'local':
            status = Status.retrieve_job_status(config.dirout, 'hybrids', name)

            if status != 'successful':
                Status.add_job(
                    config.dirout, 'hybrids', name, replace=True,
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
                           ratios=config.ratios,
                           out_dir=config.dirout,
                           log_dir=config.logdir,
                           verbose=verbose)

        elif config.execution_control.option in ('eagle', 'slurm'):
            ctx.obj['SOLAR_FPATH'] = solar_fpath
            ctx.obj['WIND_FPATH'] = wind_fpath
            ctx.obj['ALLOW_SOLAR_ONLY'] = config.allow_solar_only
            ctx.obj['ALLOW_WIND_ONLY'] = config.allow_wind_only
            ctx.obj['FILLNA'] = config.fillna
            ctx.obj['LIMITS'] = config.limits
            ctx.obj['RATIOS'] = config.ratios
            ctx.obj['OUT_DIR'] = config.dirout
            ctx.obj['LOG_DIR'] = config.logdir
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
@click.option('--ratios', '-r', type=STR, default=None,
              show_default=True,
              help='String representation of a (JSON) dictionary specifying a '
                   'mapping of "{{\\"[\'numerator_column\', '
                   '\'denominator_column\']\\": ratio}}" representing a limit '
                   'on the ratio of the values of the specified columns. '
                   'The input format is particularly important because it is '
                   'parsed as a JSON dict (in particular, the dict keys must '
                   'use escaped double quotes and multiple ratio values must '
                   'be input as a square-bracketed list). '
                   'The names of the columns should match the column names in '
                   'the merged meta, so they are likely prefixed by {!r} or '
                   '{!r}. The order of the column names specifies the way the '
                   'ratio is calculated: the first column is always '
                   'treated as the ratio numerator and the second column is '
                   'the ratio denominator. The ratio value can be a single '
                   'value or an iterable of length two representing the lower '
                   'and upper ratio bounds merged meta. A single value is '
                   'treated as both an upper and a lower bound. For example, '
                   '`--ratios "{{\\"[\'solar_capacity\', \'wind_capacity\']'
                   '\\": 1}}"` would limit the solar and wind capacities to '
                   'always be equal. On the other hand, '
                   '`--ratios "{{\\"[\'solar_capacity\', \'wind_capacity\']'
                   '\\": [0.5, 1.5]}}\'` would limit the solar and wind '
                   'capacity ratio to be between half and double (e.g., '
                   'no capacity value would be more than double the other).'
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
           fillna, limits, ratios, out_dir, log_dir, verbose):
    """reV hybridization CLI."""
    name = ctx.obj['NAME']
    ctx.obj['SOLAR_FPATH'] = solar_fpath
    ctx.obj['WIND_FPATH'] = wind_fpath
    ctx.obj['ALLOW_SOLAR_ONLY'] = allow_solar_only
    ctx.obj['ALLOW_WIND_ONLY'] = allow_wind_only
    ctx.obj['FILLNA'] = fillna
    ctx.obj['LIMITS'] = limits
    ctx.obj['RATIOS'] = ratios
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

        if isinstance(ratios, str):
            ratios = json.loads(ratios)
            try:
                ratios = convert_str_keys_to_tuples(ratios)
            except json.decoder.JSONDecodeError:
                msg = ('One of the keys of "ratios" input is not in proper '
                       'JSON format! Please ensure that the tuple key values '
                       'are represented with square brackets and that the '
                       'column names are in double quotation marks. Here is '
                       'an example of a valid "ratios" key: '
                       '`\\"[\'solar_capacity\', \'wind_capacity\']\\"`.')
                logger.error(msg)
                raise InputError(msg) from None

        try:
            Hybridization(
                solar_fpath, wind_fpath,
                allow_solar_only=allow_solar_only,
                allow_wind_only=allow_wind_only,
                fillna=fillna, limits=limits,
                ratios=ratios
            ).run(fout=fout)

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
        Status.make_job_file(out_dir, 'hybrids', name, status)


def get_node_cmd(ctx):
    """Get a CLI call command for the hybrids cli."""

    name = ctx.obj['NAME']
    solar_fpath = ctx.obj['SOLAR_FPATH']
    wind_fpath = ctx.obj['WIND_FPATH']
    allow_solar_only = ctx.obj['ALLOW_SOLAR_ONLY']
    allow_wind_only = ctx.obj['ALLOW_WIND_ONLY']
    fillna = ctx.obj['FILLNA']
    limits = ctx.obj['LIMITS']
    ratios = ctx.obj['RATIOS']
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

    if ratios:
        ratios_as_str = convert_ratio_dict_to_cli_input(ratios)
        args.append('-r {}'.format(ratios_as_str))

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


def convert_ratio_dict_to_cli_input(input_dict):
    """Convert input ratio dict to cli-formatted string.

    Parameters
    ----------
    input_dict : dict
        A dictionary with tuples of len 2 as keys and
        values as floats or lists. This dictionary will
        be converted to a string representation that is
        ready to be used with the cli.

    Returns
    -------
    str
        String that can be appended to cli call with the input
        dictionary formatted as necessary. Empty string if dict is
        empty.
    """
    kv_strings = []
    for k, v in input_dict.items():
        try:
            v = list(v)
        except TypeError:
            v = [v]
        kv_strings.append(
            "\\\"['{}', '{}']\\\": {!r}".format(*k, v)
        )

    if kv_strings:
        ratios_as_str = ", ".join(kv_strings)
        ratios_as_str = "".join(["{", ratios_as_str, "}"])
        return '"{}"'.format(ratios_as_str)
    else:
        return ''


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

    status = Status.retrieve_job_status(out_dir, 'hybrids', name,
                                        hardware='eagle',
                                        subprocess_manager=slurm_manager)

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
                out_dir, 'hybrids', name, replace=True,
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
