# -*- coding: utf-8 -*-
"""
reV Supply Curve Aggregation command line interface (cli).
"""
import os
import click
import logging
import pprint

from reV.config.supply_curve_configs import SupplyCurveConfig
from reV.utilities.execution import SLURM
from reV.utilities.cli_dtypes import STR, INT
from reV.utilities.loggers import init_mult
from reV.supply_curve.supply_curve import SupplyCurve

logger = logging.getLogger(__name__)


@click.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV exclusions configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV SC aggregation from a config file."""
    name = ctx.obj['NAME']

    # Instantiate the config object
    config = SupplyCurveConfig(config_file)

    # take name from config if not default
    if config.name.lower() != 'rev':
        name = config.name

    # Enforce verbosity if logging level is specified in the config
    if config.logging_level == logging.DEBUG:
        verbose = True

    # initialize loggers
    init_mult(name, config.logdir, modules=[__name__, 'reV.supply_curve',
                                            'reV.config', 'reV.utilities'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV 2.0 supply curve from config '
                'file: "{}"'.format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    if config.execution_control.option == 'local':
        ctx.invoke(main, name, config.sc_points, config.trans_table,
                   config.fixed_charge_rate, config.sc_features,
                   config.transmission_costs, config.dirout, config.logdir,
                   config.simple, verbose)

    elif config.execution_control.option == 'eagle':

        ctx.obj['NAME'] = name
        ctx.obj['SC_POINTS'] = config.sc_points
        ctx.obj['TRANS_TABLE'] = config.trans_table
        ctx.obj['FIXED_CHARGE_RATE'] = config.fixed_charge_rate
        ctx.obj['SC_FEATURES'] = config.sc_features
        ctx.obj['TRANSMISSION_COSTS'] = config.transmission_costs
        ctx.obj['OUT_DIR'] = config.dirout
        ctx.obj['LOG_DIR'] = config.logdir
        ctx.obj['SIMPLE'] = config.simple
        ctx.obj['VERBOSE'] = verbose

        ctx.invoke(eagle,
                   alloc=config.execution_control.alloc,
                   memory=config.execution_control.node_mem,
                   walltime=config.execution_control.walltime,
                   feature=config.execution_control.feature)


@click.group(invoke_without_command=True)
@click.option('--name', '-n', default='agg', type=STR,
              help='Job name. Default is "agg".')
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
@click.option('--out_dir', '-o', type=STR, default='./',
              help='Directory to save aggregation summary output.')
@click.option('--log_dir', '-ld', type=STR, default='./logs/',
              help='Directory to save aggregation logs.')
@click.option('-s', '--simple', is_flag=True,
              help='Flag to turn on simple supply curve calculation.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, sc_points, trans_table, fixed_charge_rate, sc_features,
         transmission_costs, out_dir, log_dir, simple, verbose):
    """reV Supply Curve CLI."""

    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['SC_POINTS'] = sc_points
    ctx.obj['TRANS_TABLE'] = trans_table
    ctx.obj['FIXED_CHARGE_RATE'] = fixed_charge_rate
    ctx.obj['SC_FEATURES'] = sc_features
    ctx.obj['TRANSMISSION_COSTS'] = transmission_costs
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['LOG_DIR'] = log_dir
    ctx.obj['SIMPLE'] = simple
    ctx.obj['VERBOSE'] = verbose

    if ctx.invoked_subcommand is None:
        init_mult(name, log_dir, modules=[__name__, 'reV.supply_curve'],
                  verbose=verbose)

        if simple:
            out = SupplyCurve.simple(sc_points, trans_table, fixed_charge_rate,
                                     sc_features=sc_features,
                                     transmission_costs=transmission_costs)
        else:
            out = SupplyCurve.full(sc_points, trans_table, fixed_charge_rate,
                                   sc_features=sc_features,
                                   transmission_costs=transmission_costs)

        fpath_out = os.path.join(out_dir, '{}.csv'.format(name))
        out.to_csv(fpath_out)


def get_node_cmd(name, sc_points, trans_table, fixed_charge_rate, sc_features,
                 transmission_costs, out_dir, log_dir, simple, verbose):
    """Get a CLI call command for the Supply Curve cli."""

    args = ('-n {name} '
            '-sc {sc_points} '
            '-tt {trans_table} '
            '-fcr {fixed_charge_rate} '
            '-scf {sc_features} '
            '-tc {transmission_costs} '
            '-o {out_dir} '
            '-ld {log_dir} '
            )

    args = args.format(name=SLURM.s(name),
                       sc_points=SLURM.s(sc_points),
                       trans_table=SLURM.s(trans_table),
                       fixed_charge_rate=SLURM.s(fixed_charge_rate),
                       sc_features=SLURM.s(sc_features),
                       transmission_costs=SLURM.s(transmission_costs),
                       out_dir=SLURM.s(out_dir),
                       log_dir=SLURM.s(log_dir),
                       )

    if simple:
        args += '-s '
    if verbose:
        args += '-v '

    cmd = 'python -m reV.supply_curve.supply_curve_cli {}'.format(args)
    return cmd


@main.command()
@click.option('--alloc', '-a', required=True, type=STR,
              help='Eagle allocation account name.')
@click.option('--memory', '-mem', default=90, type=INT, help='Eagle node '
              'memory request in GB. Default is 90')
@click.option('--walltime', '-wt', default=1.0, type=float,
              help='Eagle walltime request in hours. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--stdout_path', '-sout', default=None, type=STR,
              help='Subprocess standard output path. Default is in out_dir.')
@click.pass_context
def eagle(ctx, alloc, memory, walltime, feature, stdout_path):
    """Eagle submission tool for reV supply curve aggregation."""

    name = ctx.obj['NAME']
    sc_points = ctx.obj['SC_POINTS']
    trans_table = ctx.obj['TRANS_TABLE']
    fixed_charge_rate = ctx.obj['FIXED_CHARGE_RATE']
    sc_features = ctx.obj['SC_FEATURES']
    transmission_costs = ctx.obj['TRANSMISSION_COSTS']
    simple = ctx.obj['SIMPLE']
    out_dir = ctx.obj['OUT_DIR']
    log_dir = ctx.obj['LOG_DIR']
    verbose = ctx.obj['VERBOSE']

    if stdout_path is None:
        stdout_path = os.path.join(log_dir, 'stdout/')

    node_name = 'sc_{}'.format(name)
    cmd = get_node_cmd(name, sc_points, trans_table, fixed_charge_rate,
                       sc_features, transmission_costs, out_dir, log_dir,
                       simple, verbose)

    logger.info('Running reV Supply Curve on Eagle with '
                'node name "{}"'.format(node_name))

    slurm = SLURM(cmd, alloc=alloc, memory=memory,
                  walltime=walltime, feature=feature,
                  name=node_name, stdout_path=stdout_path)
    if slurm.id:
        msg = ('Kicked off reV SC job "{}" (SLURM jobid #{}) on Eagle.'
               .format(node_name, slurm.id))
    else:
        msg = ('Was unable to kick off reV SC job "{}". Please see the '
               'stdout error messages'.format(node_name))
    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Supply Curve CLI.')
        raise
