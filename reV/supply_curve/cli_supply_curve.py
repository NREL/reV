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

from rex.utilities.hpc import SLURM
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
        ctx.obj['OFFSHORE_TRANS_TABLE'] = config.offshore_trans_table
        ctx.obj['WIND_DIRS'] = config.wind_dirs
        ctx.obj['N_DIRS'] = config.n_dirs
        ctx.obj['DOWNWIND'] = config.downwind
        ctx.obj['OFFSHORE_COMPETE'] = config.offshore_compete
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
@click.option('--offshore_trans_table', '-ott', type=STR, default=None,
              help=('Path to offshore transmission table, if None offshore sc '
                    'points will not be included, by default None'))
@click.option('--wind_dirs', '-wd', type=click.Path(exists=True), default=None,
              help=('Path to .csv containing reVX.wind_dirs.wind_dirs.WindDirs'
                    ' output with the neighboring supply curve point gids and '
                    'power-rose value at each cardinal direction'))
@click.option('--n_dirs', '-dirs', type=int, default=2,
              help='Number of prominent directions to use')
@click.option('--downwind', '-dw', is_flag=True,
              help=('Flag to remove downwind neighbors as well as upwind '
                    'neighbors'))
@click.option('--offshore_compete', '-oc', is_flag=True,
              help=('Flag as to whether offshore farms should be included '
                    'during CompetitiveWindFarms, by default False'))
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
           transmission_costs, sort_on, offshore_trans_table, wind_dirs,
           n_dirs, downwind, offshore_compete, max_workers, out_dir, log_dir,
           simple, line_limited, verbose):
    """reV Supply Curve CLI."""
    name = ctx.obj['NAME']
    ctx.obj['SC_POINTS'] = sc_points
    ctx.obj['TRANS_TABLE'] = trans_table
    ctx.obj['FIXED_CHARGE_RATE'] = fixed_charge_rate
    ctx.obj['SC_FEATURES'] = sc_features
    ctx.obj['TRANSMISSION_COSTS'] = transmission_costs
    ctx.obj['SORT_ON'] = sort_on
    ctx.obj['OFFSHORE_TRANS_TABLE'] = offshore_trans_table
    ctx.obj['WIND_DIRS'] = wind_dirs
    ctx.obj['N_DIRS'] = n_dirs
    ctx.obj['DOWNWIND'] = downwind
    ctx.obj['MAX_WORKERS'] = max_workers
    ctx.obj['OFFSHORE_COMPETE'] = offshore_compete
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

        offshore_table = offshore_trans_table
        try:
            if simple:
                out = SupplyCurve.simple(sc_points, trans_table,
                                         fixed_charge_rate,
                                         sc_features=sc_features,
                                         transmission_costs=transmission_costs,
                                         sort_on=sort_on, wind_dirs=wind_dirs,
                                         n_dirs=n_dirs, downwind=downwind,
                                         max_workers=max_workers,
                                         offshore_trans_table=offshore_table,
                                         offshore_compete=offshore_compete)
            else:
                out = SupplyCurve.full(sc_points, trans_table,
                                       fixed_charge_rate,
                                       sc_features=sc_features,
                                       transmission_costs=transmission_costs,
                                       line_limited=line_limited,
                                       sort_on=sort_on, wind_dirs=wind_dirs,
                                       n_dirs=n_dirs, downwind=downwind,
                                       max_workers=max_workers,
                                       offshore_trans_table=offshore_table,
                                       offshore_compete=offshore_compete)
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
                 transmission_costs, sort_on, offshore_trans_table, wind_dirs,
                 n_dirs, downwind, offshore_compete, max_workers, out_dir,
                 log_dir, simple, line_limited, verbose):
    """Get a CLI call command for the Supply Curve cli."""

    args = ['-sc {}'.format(SLURM.s(sc_points)),
            '-tt {}'.format(SLURM.s(trans_table)),
            '-fcr {}'.format(SLURM.s(fixed_charge_rate)),
            '-scf {}'.format(SLURM.s(sc_features)),
            '-tc {}'.format(SLURM.s(transmission_costs)),
            '-so {}'.format(SLURM.s(sort_on)),
            '-ott {}'.format(SLURM.s(offshore_trans_table)),
            '-dirs {}'.format(SLURM.s(n_dirs)),
            '-mw {}'.format(SLURM.s(max_workers)),
            '-o {}'.format(SLURM.s(out_dir)),
            '-ld {}'.format(SLURM.s(log_dir)),
            ]

    if wind_dirs is not None:
        args.append('-wd {}'.format(SLURM.s(wind_dirs)))

    if downwind:
        args.append('-dw')

    if offshore_compete:
        args.append('-oc')

    if simple:
        args.append('-s')
    elif line_limited:
        args.append('-ll')

    if verbose:
        args.append('-v')

    cmd = ('python -m reV.supply_curve.cli_supply_curve -n {} direct {}'
           .format(SLURM.s(name), ' '.join(args)))
    logger.debug('Creating the following command line call:\n\t{}'.format(cmd))

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
    offshore_trans_table = ctx.obj['OFFSHORE_TRANS_TABLE']
    wind_dirs = ctx.obj['WIND_DIRS']
    n_dirs = ctx.obj['N_DIRS']
    downwind = ctx.obj['DOWNWIND']
    offshore_compete = ctx.obj['OFFSHORE_COMPETE']
    max_workers = ctx.obj['MAX_WORKERS']
    out_dir = ctx.obj['OUT_DIR']
    log_dir = ctx.obj['LOG_DIR']
    verbose = ctx.obj['VERBOSE']

    if stdout_path is None:
        stdout_path = os.path.join(log_dir, 'stdout/')

    cmd = get_node_cmd(name, sc_points, trans_table, fixed_charge_rate,
                       sc_features, transmission_costs, sort_on,
                       offshore_trans_table, wind_dirs, n_dirs, downwind,
                       offshore_compete, max_workers, out_dir, log_dir,
                       simple, line_limited, verbose)

    slurm_manager = ctx.obj.get('SLURM_MANAGER', None)
    if slurm_manager is None:
        slurm_manager = SLURM()
        ctx.obj['SLURM_MANAGER'] = slurm_manager

    status = Status.retrieve_job_status(out_dir, 'supply-curve', name,
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
        logger.info('Running reV Supply Curve on SLURM with '
                    'node name "{}"'.format(name))
        logger.debug('\t{}'.format(cmd))
        out = slurm_manager.sbatch(cmd, alloc=alloc, memory=memory,
                                   walltime=walltime, feature=feature,
                                   name=name, stdout_path=stdout_path,
                                   conda_env=conda_env, module=module)[0]
        if out:
            msg = ('Kicked off reV SC job "{}" (SLURM jobid #{}).'
                   .format(name, out))
            Status.add_job(
                out_dir, 'supply-curve', name, replace=True,
                job_attrs={'job_id': out, 'hardware': 'eagle',
                           'fout': '{}.csv'.format(name), 'dirout': out_dir})

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Supply Curve CLI.')
        raise
