# -*- coding: utf-8 -*-
"""
reV Supply Curve Aggregation command line interface (cli).
"""
import os
import click
import logging
import json
import pprint
import time

from reV.config.supply_curve_configs import AggregationConfig
from reV.utilities.execution import SLURM
from reV.utilities.cli_dtypes import STR, INT, FLOATLIST
from reV.utilities.loggers import init_mult
from reV.supply_curve.tech_mapping import TechMapping
from reV.supply_curve.aggregation import Aggregation
from reV.pipeline.status import Status

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
    config = AggregationConfig(config_file)

    # take name from config if not default
    if config.name.lower() != 'rev':
        name = config.name

    # Enforce verbosity if logging level is specified in the config
    if config.logging_level == logging.DEBUG:
        verbose = True

    # initialize loggers
    init_mult(name, config.logdir, modules=[__name__, 'reV.config',
                                            'reV.utilities'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV 2.0 supply curve aggregation from config '
                'file: "{}"'.format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    if config.execution_control.option == 'local':
        status = Status.retrieve_job_status(config.dirout, 'aggregation',
                                            name)
        if status != 'successful':
            Status.add_job(
                config.dirout, 'aggregation', name, replace=True,
                job_attrs={'hardware': 'local',
                           'fout': '{}.csv'.format(name),
                           'dirout': config.dirout})
            ctx.invoke(main, name, config.fpath_excl, config.fpath_gen,
                       config.fpath_res, config.fpath_techmap, config.dset_tm,
                       config.res_class_dset, config.res_class_bins,
                       config.dset_cf, config.dset_lcoe, config.data_layers,
                       config.resolution, config.dirout, config.logdir,
                       verbose)

    elif config.execution_control.option == 'eagle':

        ctx.obj['NAME'] = name
        ctx.obj['FPATH_EXCL'] = config.fpath_excl
        ctx.obj['FPATH_GEN'] = config.fpath_gen
        ctx.obj['FPATH_RES'] = config.fpath_res
        ctx.obj['FPATH_TECHMAP'] = config.fpath_techmap
        ctx.obj['DSET_TM'] = config.dset_tm
        ctx.obj['RES_CLASS_DSET'] = config.res_class_dset
        ctx.obj['RES_CLASS_BINS'] = config.res_class_bins
        ctx.obj['DSET_CF'] = config.dset_cf
        ctx.obj['DSET_LCOE'] = config.dset_lcoe
        ctx.obj['DATA_LAYERS'] = config.data_layers
        ctx.obj['RESOLUTION'] = config.resolution
        ctx.obj['OUT_DIR'] = config.dirout
        ctx.obj['LOG_DIR'] = config.logdir
        ctx.obj['VERBOSE'] = verbose

        ctx.invoke(eagle,
                   alloc=config.execution_control.alloc,
                   memory=config.execution_control.node_mem,
                   walltime=config.execution_control.walltime,
                   feature=config.execution_control.feature)


@click.group(invoke_without_command=True)
@click.option('--name', '-n', default='agg', type=STR,
              help='Job name. Default is "agg".')
@click.option('--fpath_excl', '-fex', type=STR, required=True,
              help='Exclusions file (.tiff).')
@click.option('--fpath_gen', '-fg', type=STR, required=True,
              help='reV generation/econ output file.')
@click.option('--fpath_res', '-fr', type=STR, default=None,
              help='Resource file (required if techmap file is to be '
              'created).')
@click.option('--fpath_techmap', '-ftm', type=STR, required=True,
              help='reV techmap file.')
@click.option('--dset_tm', '-dtm', type=STR, required=True,
              help='Dataset in the techmap corresponding to the resource '
              'being analyzed.')
@click.option('--res_class_dset', '-cd', type=STR, default=None,
              help='Dataset to determine the resource class '
              '(must be in fpath_gen).')
@click.option('--res_class_bins', '-cb', type=FLOATLIST, default=None,
              help='List of resource class bin edges.')
@click.option('--dset_cf', '-dcf', type=STR, default='cf_mean-means',
              help='Dataset containing capacity factor values to aggregate.')
@click.option('--dset_lcoe', '-dlc', type=STR, default='lcoe_fcr-means',
              help='Dataset containing lcoe values to aggregate.')
@click.option('--data_layers', '-d', type=STR, default=None,
              help='A serialized dictionary of additional data layers to '
              'include in the aggregation.')
@click.option('--resolution', '-r', type=INT, default=64,
              help='Number of exclusion points along a squares edge to '
              'include in an aggregated supply curve point.')
@click.option('--out_dir', '-o', type=STR, default='./',
              help='Directory to save aggregation summary output.')
@click.option('--log_dir', '-ld', type=STR, default='./logs/',
              help='Directory to save aggregation logs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, fpath_excl, fpath_gen, fpath_res, fpath_techmap, dset_tm,
         res_class_dset, res_class_bins, dset_cf, dset_lcoe, data_layers,
         resolution, out_dir, log_dir, verbose):
    """reV Supply Curve Aggregation Summary CLI."""

    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['FPATH_EXCL'] = fpath_excl
    ctx.obj['FPATH_GEN'] = fpath_gen
    ctx.obj['FPATH_RES'] = fpath_res
    ctx.obj['FPATH_TECHMAP'] = fpath_techmap
    ctx.obj['DSET_TM'] = dset_tm
    ctx.obj['RES_CLASS_DSET'] = res_class_dset
    ctx.obj['RES_CLASS_BINS'] = res_class_bins
    ctx.obj['DSET_CF'] = dset_cf
    ctx.obj['DSET_LCOE'] = dset_lcoe
    ctx.obj['DATA_LAYERS'] = data_layers
    ctx.obj['RESOLUTION'] = resolution
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['LOG_DIR'] = log_dir
    ctx.obj['VERBOSE'] = verbose

    if ctx.invoked_subcommand is None:
        t0 = time.time()
        init_mult(name, log_dir, modules=[__name__, 'reV.supply_curve'],
                  verbose=verbose)

        if not os.path.exists(fpath_techmap):
            TechMapping.run(fpath_excl, fpath_res, fpath_techmap, dset_tm)

        if data_layers is not None:
            json.loads(data_layers)

        summary = Aggregation.summary(fpath_excl, fpath_gen, fpath_techmap,
                                      dset_tm,
                                      res_class_dset=res_class_dset,
                                      res_class_bins=res_class_bins,
                                      dset_cf=dset_cf,
                                      dset_lcoe=dset_lcoe,
                                      data_layers=data_layers,
                                      resolution=resolution)

        fn_out = '{}.csv'.format(name)
        fpath_out = os.path.join(out_dir, fn_out)
        summary.to_csv(fpath_out)

        runtime = (time.time() - t0) / 60
        logger.info('Supply curve aggregation complete. '
                    'Time elapsed: {:.2f} min. Target output dir: {}'
                    .format(runtime, out_dir))

        finput = [fpath_excl, fpath_gen, fpath_techmap]
        if fpath_res is not None:
            finput.append(fpath_res)

        # add job to reV status file.
        status = {'dirout': out_dir, 'fout': fn_out,
                  'job_status': 'successful',
                  'runtime': runtime,
                  'finput': finput}
        Status.make_job_file(out_dir, 'aggregation', name, status)


def get_node_cmd(name, fpath_excl, fpath_gen, fpath_res, fpath_techmap,
                 dset_tm, res_class_dset, res_class_bins,
                 dset_cf, dset_lcoe, data_layers,
                 resolution, out_dir, log_dir, verbose):
    """Get a CLI call command for the SC aggregation cli."""

    args = ('-n {name} '
            '-fex {fpath_excl} '
            '-fg {fpath_gen} '
            '-fr {fpath_res} '
            '-ftm {fpath_techmap} '
            '-dtm {dset_tm} '
            '-cd {res_class_dset} '
            '-cb {res_class_bins} '
            '-dcf {dset_cf} '
            '-dlc {dset_lcoe} '
            '-d {data_layers} '
            '-r {resolution} '
            '-o {out_dir} '
            '-ld {log_dir} '
            )

    args = args.format(name=SLURM.s(name),
                       fpath_excl=SLURM.s(fpath_excl),
                       fpath_gen=SLURM.s(fpath_gen),
                       fpath_res=SLURM.s(fpath_res),
                       fpath_techmap=SLURM.s(fpath_techmap),
                       dset_tm=SLURM.s(dset_tm),
                       res_class_dset=SLURM.s(res_class_dset),
                       res_class_bins=SLURM.s(res_class_bins),
                       dset_cf=SLURM.s(dset_cf),
                       dset_lcoe=SLURM.s(dset_lcoe),
                       data_layers=SLURM.s(data_layers),
                       resolution=SLURM.s(resolution),
                       out_dir=SLURM.s(out_dir),
                       log_dir=SLURM.s(log_dir),
                       )

    if verbose:
        args += '-v '

    cmd = 'python -m reV.supply_curve.cli_aggregation {}'.format(args)
    return cmd


@main.command()
@click.option('--alloc', '-a', required=True, type=STR,
              help='Eagle allocation account name.')
@click.option('--memory', '-mem', default=None, type=INT,
              help='Eagle node memory request in GB. Default is None')
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
    fpath_excl = ctx.obj['FPATH_EXCL']
    fpath_gen = ctx.obj['FPATH_GEN']
    fpath_res = ctx.obj['FPATH_RES']
    fpath_techmap = ctx.obj['FPATH_TECHMAP']
    dset_tm = ctx.obj['DSET_TM']
    res_class_dset = ctx.obj['RES_CLASS_DSET']
    res_class_bins = ctx.obj['RES_CLASS_BINS']
    dset_cf = ctx.obj['DSET_CF']
    dset_lcoe = ctx.obj['DSET_LCOE']
    data_layers = ctx.obj['DATA_LAYERS']
    resolution = ctx.obj['RESOLUTION']
    out_dir = ctx.obj['OUT_DIR']
    log_dir = ctx.obj['LOG_DIR']
    verbose = ctx.obj['VERBOSE']

    if stdout_path is None:
        stdout_path = os.path.join(log_dir, 'stdout/')

    cmd = get_node_cmd(name, fpath_excl, fpath_gen, fpath_res, fpath_techmap,
                       dset_tm, res_class_dset, res_class_bins,
                       dset_cf, dset_lcoe, data_layers,
                       resolution, out_dir, log_dir, verbose)

    status = Status.retrieve_job_status(out_dir, 'aggregation', name)
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'
               .format(name, out_dir))
    else:
        logger.info('Running reV SC aggregation on Eagle with '
                    'node name "{}"'.format(name))
        slurm = SLURM(cmd, alloc=alloc, memory=memory,
                      walltime=walltime, feature=feature,
                      name=name, stdout_path=stdout_path)
        if slurm.id:
            msg = ('Kicked off reV SC aggregation job "{}" '
                   '(SLURM jobid #{}) on Eagle.'
                   .format(name, slurm.id))
            Status.add_job(
                out_dir, 'aggregation', name, replace=True,
                job_attrs={'job_id': slurm.id, 'hardware': 'eagle',
                           'fout': '{}.csv'.format(name), 'dirout': out_dir})
        else:
            msg = ('Was unable to kick off reV SC job "{}". '
                   'Please see the stdout error messages'
                   .format(name))
    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV SC aggregation CLI.')
        raise
