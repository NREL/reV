# -*- coding: utf-8 -*-
"""
reV Supply Curve command line interface (cli).
"""
import os
import click
import logging
import pprint
import time

from reV.config.supply_curve_configs import SupplyCurveConfig
from reV.pipeline.status import Status
from reV.supply_curve.supply_curve import SupplyCurve

from rex.utilities.execution import SLURM
from rex.utilities.cli_dtypes import STR, INT
from rex.utilities.loggers import init_mult
from rex.utilities.utilities import dict_str_load, get_class_properties

logger = logging.getLogger(__name__)


@click.group()
@click.option('--name', '-n', default='reV-sc', type=STR,
              help='Job name. Default is "reV-sc".')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """reV Supply Curve Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
def valid_config_keys():
    """
    Echo the valid SupplyCurve config keys
    """
    click.echo(', '.join(get_class_properties(SupplyCurveConfig)))


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV supply curve configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV supply curve compute from a config file."""
    name = ctx.obj['NAME']

    # Instantiate the config object
    config = SupplyCurveConfig(config_file)

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
    logger.info('Running reV supply curve from config '
                'file: "{}"'.format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    if config.execution_control.option == 'local':
        status = Status.retrieve_job_status(config.dirout, 'supply-curve',
                                            name)
        if status != 'successful':
            Status.add_job(
                config.dirout, 'supply-curve', name, replace=True,
                job_attrs={'hardware': 'local',
                           'fout': '{}.csv'.format(name),
                           'dirout': config.dirout})
            ctx.invoke(direct,
                       sc_points=config.sc_points,
                       trans_table=config.trans_table,
                       fixed_charge_rate=config.fixed_charge_rate,
                       sc_features=config.sc_features,
                       transmission_costs=config.transmission_costs,
                       sort_on=config.sort_on,
                       wind_dirs=config.wind_dirs,
                       n_dirs=config.n_dirs,
                       downwind=config.downwind,
                       max_workers=config.max_workers,
                       out_dir=config.dirout,
                       log_dir=config.logdir,
                       simple=config.simple,
                       line_limited=config.line_limited,
                       verbose=verbose)

    elif config.execution_control.option in ('eagle', 'slurm'):

        ctx.obj['NAME'] = name
        ctx.obj['SC_POINTS'] = config.sc_points
        ctx.obj['TRANS_TABLE'] = config.trans_table
        ctx.obj['FIXED_CHARGE_RATE'] = config.fixed_charge_rate
        ctx.obj['SC_FEATURES'] = config.sc_features
        ctx.obj['TRANSMISSION_COSTS'] = config.transmission_costs
        ctx.obj['SORT_ON'] = config.sort_on
        ctx.obj['WIND_DIRS'] = config.wind_dirs
        ctx.obj['N_DIRS'] = config.n_dirs
        ctx.obj['DOWNWIND'] = config.downwind
        ctx.obj['MAX_WORKERS'] = config.max_workers
        ctx.obj['OUT_DIR'] = config.dirout
        ctx.obj['LOG_DIR'] = config.logdir
        ctx.obj['SIMPLE'] = config.simple
        ctx.obj['LINE_LIMITED'] = config.line_limited
        ctx.obj['VERBOSE'] = verbose

        ctx.invoke(slurm,
                   alloc=config.execution_control.allocation,
                   memory=config.execution_control.memory,
                   walltime=config.execution_control.walltime,
                   feature=config.execution_control.feature,
                   conda_env=config.execution_control.conda_env,
                   module=config.execution_control.module)


@main.group(invoke_without_command=True)
@click.option('--sc_points', '-sc', type=STR, required=True,
              help='Supply curve point summary table (.csv or .json).')
@click.option('--trans_table', '-tt', type=STR, required=True,
              help='Supply curve transmission mapping table (.csv or .json).')
@click.option('--fixed_charge_rate', '-fcr', type=float, required=True,
              help='Fixed charge rate used to compute LCOT')
@click.option('--sc_features', '-scf', type=STR, default=None,
              help='Table containing additional supply curve features '
              '(.csv or .json)')
@click.option('--transmission_costs', '-tc', type=STR, default=None,
              help='Table or serialized dict of transmission cost inputs.')
@click.option('--sort_on', '-so', type=str, default='total_lcoe',
              help='The supply curve table column label to sort on. '
              'This determines the ordering of the SC buildout algorithm.')
@click.option('--wind_dirs', '-wd', type=click.Path(exists=True), default=None,
              help=('Path to .csv containing reVX.wind_dirs.wind_dirs.WindDirs'
                    ' output with the neighboring supply curve point gids and '
                    'power-rose value at each cardinal direction'))
@click.option('--n_dirs', '-dirs', type=int, default=2,
              help='Number of prominent directions to use')
@click.option('--downwind', '-dw', is_flag=True,
              help=('Flag to remove downwind neighbors as well as upwind '
                    'neighbors'))
@click.option('--max_workers', '-mw', type=INT, default=None,
              help=('Number of workers to use to compute lcot, if > 1 run in '
                    'parallel. None uses all available cpus.'))
@click.option('--out_dir', '-o', type=STR, default='./',
              help='Directory to save aggregation summary output.')
@click.option('--log_dir', '-ld', type=STR, default='./logs/',
              help='Directory to save aggregation logs.')
@click.option('-s', '--simple', is_flag=True,
              help='Flag to turn on simple supply curve calculation.')
@click.option('-ll', '--line_limited', is_flag=True,
              help='Flag to turn on line-limited substation capacity '
              'calculation (legacy methodology). Alternative is multi-line '
              'spread capacity.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def direct(ctx, sc_points, trans_table, fixed_charge_rate, sc_features,
           transmission_costs, sort_on, wind_dirs, n_dirs, downwind,
           max_workers, out_dir, log_dir, simple, line_limited, verbose):
    """reV Supply Curve CLI."""
    name = ctx.obj['NAME']
    ctx.obj['SC_POINTS'] = sc_points
    ctx.obj['TRANS_TABLE'] = trans_table
    ctx.obj['FIXED_CHARGE_RATE'] = fixed_charge_rate
    ctx.obj['SC_FEATURES'] = sc_features
    ctx.obj['TRANSMISSION_COSTS'] = transmission_costs
    ctx.obj['SORT_ON'] = sort_on
    ctx.obj['WIND_DIRS'] = wind_dirs
    ctx.obj['N_DIRS'] = n_dirs
    ctx.obj['DOWNWIND'] = downwind
    ctx.obj['MAX_WORKERS'] = max_workers
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['LOG_DIR'] = log_dir
    ctx.obj['SIMPLE'] = simple
    ctx.obj['LINE_LIMITED'] = line_limited
    ctx.obj['VERBOSE'] = verbose

    if ctx.invoked_subcommand is None:
        t0 = time.time()
        init_mult(name, log_dir, modules=[__name__, 'reV.supply_curve',
                                          'reV.handlers', 'rex'],
                  verbose=verbose)

        if isinstance(transmission_costs, str):
            transmission_costs = dict_str_load(transmission_costs)

        try:
            if simple:
                out = SupplyCurve.simple(sc_points, trans_table,
                                         fixed_charge_rate,
                                         sc_features=sc_features,
                                         transmission_costs=transmission_costs,
                                         sort_on=sort_on, wind_dirs=wind_dirs,
                                         n_dirs=n_dirs, downwind=downwind,
                                         max_workers=max_workers)
            else:
                out = SupplyCurve.full(sc_points, trans_table,
                                       fixed_charge_rate,
                                       sc_features=sc_features,
                                       transmission_costs=transmission_costs,
                                       line_limited=line_limited,
                                       sort_on=sort_on, wind_dirs=wind_dirs,
                                       n_dirs=n_dirs, downwind=downwind,
                                       max_workers=max_workers)
        except Exception as e:
            logger.exception('Supply curve compute failed. Received the '
                             'following error:\n{}'.format(e))
            raise e

        fn_out = '{}.csv'.format(name)
        fpath_out = os.path.join(out_dir, fn_out)
        out.to_csv(fpath_out, index=False)

        runtime = (time.time() - t0) / 60
        logger.info('Supply curve complete. Time elapsed: {:.2f} min. '
                    'Target output dir: {}'.format(runtime, out_dir))

        finput = [sc_points, trans_table]
        if sc_features is not None:
            finput.append(sc_features)

        if transmission_costs is not None:
            finput.append(transmission_costs)

        # add job to reV status file.
        status = {'dirout': out_dir, 'fout': fn_out,
                  'job_status': 'successful',
                  'runtime': runtime,
                  'finput': finput}
        Status.make_job_file(out_dir, 'supply-curve', name, status)


def get_node_cmd(name, sc_points, trans_table, fixed_charge_rate, sc_features,
                 transmission_costs, sort_on, wind_dirs, n_dirs, downwind,
                 max_workers, out_dir, log_dir, simple, line_limited, verbose):
    """Get a CLI call command for the Supply Curve cli."""

    args = ('-sc {sc_points} '
            '-tt {trans_table} '
            '-fcr {fixed_charge_rate} '
            '-scf {sc_features} '
            '-tc {transmission_costs} '
            '-so {sort_on} '
            '-dirs {n_dirs} '
            '-mw {max_workers} '
            '-o {out_dir} '
            '-ld {log_dir} '
            )

    args = args.format(sc_points=SLURM.s(sc_points),
                       trans_table=SLURM.s(trans_table),
                       fixed_charge_rate=SLURM.s(fixed_charge_rate),
                       sc_features=SLURM.s(sc_features),
                       transmission_costs=SLURM.s(transmission_costs),
                       sort_on=SLURM.s(sort_on),
                       n_dirs=SLURM.s(n_dirs),
                       max_workers=SLURM.s(max_workers),
                       out_dir=SLURM.s(out_dir),
                       log_dir=SLURM.s(log_dir),
                       )

    if wind_dirs is not None:
        args += '-wd {wind_dirs} '.format(wind_dirs=SLURM.s(wind_dirs))

    if downwind:
        args += '-dw '

    if simple:
        args += '-s '
    elif line_limited:
        args += '-ll '

    if verbose:
        args += '-v '

    cmd = ('python -m reV.supply_curve.cli_supply_curve -n {} direct {}'
           .format(SLURM.s(name), args))

    return cmd


@direct.command()
@click.option('--alloc', '-a', required=True, type=STR,
              help='SLURM allocation account name.')
@click.option('--memory', '-mem', default=None, type=INT, help='SLURM node '
              'memory request in GB. Default is None')
@click.option('--walltime', '-wt', default=1.0, type=float,
              help='SLURM walltime request in hours. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--module', '-mod', default=None, type=STR,
              help='Module to load')
@click.option('--conda_env', '-env', default=None, type=STR,
              help='Conda env to activate')
@click.option('--stdout_path', '-sout', default=None, type=STR,
              help='Subprocess standard output path. Default is in out_dir.')
@click.pass_context
def slurm(ctx, alloc, memory, walltime, feature, module, conda_env,
          stdout_path):
    """slurm (eagle) submission tool for reV supply curve."""
    name = ctx.obj['NAME']
    sc_points = ctx.obj['SC_POINTS']
    trans_table = ctx.obj['TRANS_TABLE']
    fixed_charge_rate = ctx.obj['FIXED_CHARGE_RATE']
    sc_features = ctx.obj['SC_FEATURES']
    transmission_costs = ctx.obj['TRANSMISSION_COSTS']
    simple = ctx.obj['SIMPLE']
    line_limited = ctx.obj['LINE_LIMITED']
    sort_on = ctx.obj['SORT_ON']
    wind_dirs = ctx.obj['WIND_DIRS']
    n_dirs = ctx.obj['N_DIRS']
    downwind = ctx.obj['DOWNWIND']
    max_workers = ctx.obj['MAX_WORKERS']
    out_dir = ctx.obj['OUT_DIR']
    log_dir = ctx.obj['LOG_DIR']
    verbose = ctx.obj['VERBOSE']

    if stdout_path is None:
        stdout_path = os.path.join(log_dir, 'stdout/')

    cmd = get_node_cmd(name, sc_points, trans_table, fixed_charge_rate,
                       sc_features, transmission_costs, sort_on, wind_dirs,
                       n_dirs, downwind, max_workers, out_dir, log_dir, simple,
                       line_limited, verbose)

    status = Status.retrieve_job_status(out_dir, 'supply-curve', name)
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'
               .format(name, out_dir))
    else:
        logger.info('Running reV Supply Curve on SLURM with '
                    'node name "{}"'.format(name))
        logger.debug('\t{}'.format(cmd))
        slurm = SLURM(cmd, alloc=alloc, memory=memory,
                      walltime=walltime, feature=feature,
                      name=name, stdout_path=stdout_path,
                      conda_env=conda_env, module=module)
        if slurm.id:
            msg = ('Kicked off reV SC job "{}" (SLURM jobid #{}).'
                   .format(name, slurm.id))
            Status.add_job(
                out_dir, 'supply-curve', name, replace=True,
                job_attrs={'job_id': slurm.id, 'hardware': 'eagle',
                           'fout': '{}.csv'.format(name), 'dirout': out_dir})
        else:
            msg = ('Was unable to kick off reV SC job "{}". Please see the '
                   'stdout error messages'.format(name))
    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Supply Curve CLI.')
        raise
