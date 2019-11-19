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
import h5py

from reV.config.supply_curve_configs import AggregationConfig
from reV.utilities.execution import SLURM
from reV.utilities.cli_dtypes import STR, INT, FLOATLIST, STRFLOAT
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
            ctx.invoke(main, name, config.excl_fpath, config.gen_fpath,
                       config.res_fpath, config.tm_dset, config.excl_dict,
                       config.res_class_dset, config.res_class_bins,
                       config.cf_dset, config.lcoe_dset, config.data_layers,
                       config.resolution, config.power_density, config.dirout,
                       config.logdir, verbose)

    elif config.execution_control.option == 'eagle':

        ctx.obj['NAME'] = name
        ctx.obj['EXCL_FPATH'] = config.excl_fpath
        ctx.obj['GEN_FPATH'] = config.gen_fpath
        ctx.obj['RES_FPATH'] = config.res_fpath
        ctx.obj['TM_DSET'] = config.tm_dset
        ctx.obj['EXCL_DICT'] = config.excl_dict
        ctx.obj['RES_CLASS_DSET'] = config.res_class_dset
        ctx.obj['RES_CLASS_BINS'] = config.res_class_bins
        ctx.obj['CF_DSET'] = config.cf_dset
        ctx.obj['LCOE_DSET'] = config.lcoe_dset
        ctx.obj['DATA_LAYERS'] = config.data_layers
        ctx.obj['RESOLUTION'] = config.resolution
        ctx.obj['POWER_DENSITY'] = config.power_density
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
@click.option('--excl_fpath', '-ef', type=STR, required=True,
              help='Exclusions file (.h5).')
@click.option('--gen_fpath', '-gf', type=STR, required=True,
              help='reV generation/econ output file.')
@click.option('--res_fpath', '-rf', type=STR, default=None,
              help='Resource file, required if techmap dset is to be created.')
@click.option('--tm_dset', '-tm', type=STR, required=True,
              help='Dataset in the exclusions file that maps the exclusions '
              'to the resource being analyzed.')
@click.option('--excl_dict', '-exd', type=STR, required=True,
              help='String representation of a dictionary of exclusion '
              'LayerMask arguments {layer: {kwarg: value}} where layer is a '
              'dataset in excl_fpath and kwarg can be "inclusion_range", '
              '"exclude_values", "include_values", "use_as_weights", '
              'or "weight".')
@click.option('--res_class_dset', '-cd', type=STR, default=None,
              help='Dataset to determine the resource class '
              '(must be in gen_fpath).')
@click.option('--res_class_bins', '-cb', type=FLOATLIST, default=None,
              help='List of resource class bin edges.')
@click.option('--cf_dset', '-cf', type=STR, default='cf_mean-means',
              help='Dataset containing capacity factor values to aggregate.')
@click.option('--lcoe_dset', '-lc', type=STR, default='lcoe_fcr-means',
              help='Dataset containing lcoe values to aggregate.')
@click.option('--data_layers', '-d', type=STR, default=None,
              help='String representation of a dictionary of additional data '
              'layers to include in the aggregation e.g. '
              '{"slope": {"dset": "srtm_slope", "method": "mean"}}')
@click.option('--resolution', '-r', type=INT, default=64,
              help='Number of exclusion points along a squares edge to '
              'include in an aggregated supply curve point.')
@click.option('--power_density', '-pd', type=STRFLOAT, default=None,
              help='Power density in MW/km2 or filepath to variable power '
              'density file. None will attempt to infer a constant '
              'power density from the generation meta data technology.')
@click.option('--out_dir', '-o', type=STR, default='./',
              help='Directory to save aggregation summary output.')
@click.option('--log_dir', '-ld', type=STR, default='./logs/',
              help='Directory to save aggregation logs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, excl_fpath, gen_fpath, res_fpath, tm_dset, excl_dict,
         res_class_dset, res_class_bins, cf_dset, lcoe_dset, data_layers,
         resolution, power_density, out_dir, log_dir, verbose):
    """reV Supply Curve Aggregation Summary CLI."""

    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['EXCL_FPATH'] = excl_fpath
    ctx.obj['GEN_FPATH'] = gen_fpath
    ctx.obj['RES_FPATH'] = res_fpath
    ctx.obj['TM_DSET'] = tm_dset
    ctx.obj['EXCL_DICT'] = excl_dict
    ctx.obj['RES_CLASS_DSET'] = res_class_dset
    ctx.obj['RES_CLASS_BINS'] = res_class_bins
    ctx.obj['CF_DSET'] = cf_dset
    ctx.obj['LCOE_DSET'] = lcoe_dset
    ctx.obj['DATA_LAYERS'] = data_layers
    ctx.obj['RESOLUTION'] = resolution
    ctx.obj['POWER_DENSITY'] = power_density
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['LOG_DIR'] = log_dir
    ctx.obj['VERBOSE'] = verbose

    if ctx.invoked_subcommand is None:
        t0 = time.time()
        init_mult(name, log_dir, modules=[__name__, 'reV.supply_curve'],
                  verbose=verbose)

        with h5py.File(excl_fpath, mode='r') as f:
            dsets = list(f)
        if tm_dset not in dsets:
            try:
                TechMapping.run(excl_fpath, res_fpath, tm_dset)
            except Exception as e:
                logger.exception('TechMapping process failed. Received the '
                                 'following error:\n{}'.format(e))
                raise e

        if isinstance(excl_dict, str):
            excl_dict = excl_dict.replace('\'', '\"')
            excl_dict = excl_dict.replace('None', 'null')
            excl_dict = json.loads(excl_dict)
        if isinstance(data_layers, str):
            data_layers = data_layers.replace('\'', '\"')
            data_layers = data_layers.replace('None', 'null')
            data_layers = json.loads(data_layers)

        try:
            summary = Aggregation.summary(excl_fpath, gen_fpath,
                                          tm_dset, excl_dict,
                                          res_class_dset=res_class_dset,
                                          res_class_bins=res_class_bins,
                                          cf_dset=cf_dset,
                                          lcoe_dset=lcoe_dset,
                                          data_layers=data_layers,
                                          resolution=resolution,
                                          power_density=power_density)
        except Exception as e:
            logger.exception('Supply curve Aggregation failed. Received the '
                             'following error:\n{}'.format(e))
            raise e

        fn_out = '{}.csv'.format(name)
        fpath_out = os.path.join(out_dir, fn_out)
        summary.to_csv(fpath_out)

        runtime = (time.time() - t0) / 60
        logger.info('Supply curve aggregation complete. '
                    'Time elapsed: {:.2f} min. Target output dir: {}'
                    .format(runtime, out_dir))

        finput = [excl_fpath, gen_fpath]
        if res_fpath is not None:
            finput.append(res_fpath)

        # add job to reV status file.
        status = {'dirout': out_dir, 'fout': fn_out,
                  'job_status': 'successful',
                  'runtime': runtime,
                  'finput': finput}
        Status.make_job_file(out_dir, 'aggregation', name, status)


def get_node_cmd(name, excl_fpath, gen_fpath, res_fpath, tm_dset, excl_dict,
                 res_class_dset, res_class_bins, cf_dset, lcoe_dset,
                 data_layers, resolution, power_density,
                 out_dir, log_dir, verbose):
    """Get a CLI call command for the SC aggregation cli."""

    args = ('-n {name} '
            '-ef {excl_fpath} '
            '-gf {gen_fpath} '
            '-rf {res_fpath} '
            '-tm {tm_dset} '
            '-exd {excl_dict} '
            '-cd {res_class_dset} '
            '-cb {res_class_bins} '
            '-cf {cf_dset} '
            '-lc {lcoe_dset} '
            '-d {data_layers} '
            '-r {resolution} '
            '-pd {power_density} '
            '-o {out_dir} '
            '-ld {log_dir} '
            )

    args = args.format(name=SLURM.s(name),
                       excl_fpath=SLURM.s(excl_fpath),
                       gen_fpath=SLURM.s(gen_fpath),
                       res_fpath=SLURM.s(res_fpath),
                       tm_dset=SLURM.s(tm_dset),
                       excl_dict=SLURM.s(excl_dict),
                       res_class_dset=SLURM.s(res_class_dset),
                       res_class_bins=SLURM.s(res_class_bins),
                       cf_dset=SLURM.s(cf_dset),
                       lcoe_dset=SLURM.s(lcoe_dset),
                       data_layers=SLURM.s(data_layers),
                       resolution=SLURM.s(resolution),
                       power_density=SLURM.s(power_density),
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
    excl_fpath = ctx.obj['EXCL_FPATH']
    gen_fpath = ctx.obj['GEN_FPATH']
    res_fpath = ctx.obj['RES_FPATH']
    tm_dset = ctx.obj['TM_DSET']
    excl_dict = ctx.obj['EXCL_DICT']
    res_class_dset = ctx.obj['RES_CLASS_DSET']
    res_class_bins = ctx.obj['RES_CLASS_BINS']
    cf_dset = ctx.obj['CF_DSET']
    lcoe_dset = ctx.obj['LCOE_DSET']
    data_layers = ctx.obj['DATA_LAYERS']
    resolution = ctx.obj['RESOLUTION']
    power_density = ctx.obj['POWER_DENSITY']
    out_dir = ctx.obj['OUT_DIR']
    log_dir = ctx.obj['LOG_DIR']
    verbose = ctx.obj['VERBOSE']

    if stdout_path is None:
        stdout_path = os.path.join(log_dir, 'stdout/')

    cmd = get_node_cmd(name, excl_fpath, gen_fpath, res_fpath,
                       tm_dset, excl_dict, res_class_dset, res_class_bins,
                       cf_dset, lcoe_dset, data_layers, resolution,
                       power_density, out_dir, log_dir, verbose)

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
