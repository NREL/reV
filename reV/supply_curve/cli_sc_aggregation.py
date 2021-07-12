# -*- coding: utf-8 -*-
# pylint: disable=all
# flake8: noqa
"""
reV Supply Curve Aggregation command line interface (cli).
"""
import os
import click
import logging
import pprint
import time
import h5py

from reV.config.supply_curve_configs import SupplyCurveAggregationConfig
from reV.pipeline.status import Status
from reV.supply_curve.tech_mapping import TechMapping
from reV.supply_curve.sc_aggregation import SupplyCurveAggregation
from reV import __version__

from rex.utilities.hpc import SLURM
from rex.utilities.cli_dtypes import (STR, INT, FLOAT, STRLIST, FLOATLIST,
                                      STRFLOAT, STR_OR_LIST)
from rex.utilities.loggers import init_mult
from rex.utilities.utilities import dict_str_load, get_class_properties

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default='reV-agg', type=STR,
              show_default=True,
              help='Job name. Default is "reV-agg".')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """reV Supply Curve Aggregation Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
def valid_config_keys():
    """
    Echo the valid SupplyCurveAggregation config keys
    """
    click.echo(', '.join(get_class_properties(SupplyCurveAggregationConfig)))


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV aggregation configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV SC aggregation from a config file."""
    name = ctx.obj['NAME']

    # Instantiate the config object
    config = SupplyCurveAggregationConfig(config_file)

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
    logger.info('Running reV supply curve aggregation from config '
                'file: "{}"'.format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    if config.execution_control.option == 'local':
        status = Status.retrieve_job_status(config.dirout,
                                            'supply-curve-aggregation',
                                            name)
        if status != 'successful':
            Status.add_job(
                config.dirout, 'supply-curve-aggregation', name, replace=True,
                job_attrs={'hardware': 'local',
                           'fout': '{}.csv'.format(name),
                           'dirout': config.dirout})
            ctx.invoke(
                direct,
                excl_fpath=config.excl_fpath,
                gen_fpath=config.gen_fpath,
                econ_fpath=config.econ_fpath,
                res_fpath=config.res_fpath,
                tm_dset=config.tm_dset,
                excl_dict=config.excl_dict,
                res_class_dset=config.res_class_dset,
                res_class_bins=config.res_class_bins,
                cf_dset=config.cf_dset,
                lcoe_dset=config.lcoe_dset,
                h5_dsets=config.h5_dsets,
                data_layers=config.data_layers,
                resolution=config.resolution,
                excl_area=config.excl_area,
                power_density=config.power_density,
                area_filter_kernel=config.area_filter_kernel,
                min_area=config.min_area,
                friction_fpath=config.friction_fpath,
                friction_dset=config.friction_dset,
                cap_cost_scale=config.cap_cost_scale,
                out_dir=config.dirout,
                max_workers=config.execution_control.max_workers,
                sites_per_worker=config.execution_control.sites_per_worker,
                log_dir=config.logdir,
                pre_extract_inclusions=config.pre_extract_inclusions,
                verbose=verbose)

    elif config.execution_control.option in ('eagle', 'slurm'):
        spw = config.execution_control.sites_per_worker
        ctx.obj['NAME'] = name
        ctx.obj['EXCL_FPATH'] = config.excl_fpath
        ctx.obj['GEN_FPATH'] = config.gen_fpath
        ctx.obj['ECON_FPATH'] = config.econ_fpath
        ctx.obj['RES_FPATH'] = config.res_fpath
        ctx.obj['TM_DSET'] = config.tm_dset
        ctx.obj['EXCL_DICT'] = config.excl_dict
        ctx.obj['RES_CLASS_DSET'] = config.res_class_dset
        ctx.obj['RES_CLASS_BINS'] = config.res_class_bins
        ctx.obj['CF_DSET'] = config.cf_dset
        ctx.obj['LCOE_DSET'] = config.lcoe_dset
        ctx.obj['H5_DSETS'] = config.h5_dsets
        ctx.obj['DATA_LAYERS'] = config.data_layers
        ctx.obj['RESOLUTION'] = config.resolution
        ctx.obj['EXCL_AREA'] = config.excl_area
        ctx.obj['POWER_DENSITY'] = config.power_density
        ctx.obj['AREA_FILTER_KERNEL'] = config.area_filter_kernel
        ctx.obj['MIN_AREA'] = config.min_area
        ctx.obj['FRICTION_FPATH'] = config.friction_fpath
        ctx.obj['FRICTION_DSET'] = config.friction_dset
        ctx.obj['CAP_COST_SCALE'] = config.cap_cost_scale
        ctx.obj['OUT_DIR'] = config.dirout
        ctx.obj['MAX_WORKERS'] = config.execution_control.max_workers
        ctx.obj['SITES_PER_WORKER'] = spw
        ctx.obj['LOG_DIR'] = config.logdir
        ctx.obj['PRE_EXTRACT_INCLUSIONS'] = config.pre_extract_inclusions
        ctx.obj['VERBOSE'] = verbose

        ctx.invoke(slurm,
                   alloc=config.execution_control.allocation,
                   memory=config.execution_control.memory,
                   feature=config.execution_control.feature,
                   walltime=config.execution_control.walltime,
                   conda_env=config.execution_control.conda_env,
                   module=config.execution_control.module)


@main.group(invoke_without_command=True)
@click.option('--excl_fpath', '-exf', type=STR_OR_LIST, required=True,
              help='Single exclusions file (.h5) or a '
              'list of exclusion files (.h5, .h5).')
@click.option('--gen_fpath', '-gf', type=STR, required=True,
              help='reV generation/econ output file.')
@click.option('--tm_dset', '-tm', type=STR, required=True,
              help='Dataset in the exclusions file that maps the exclusions '
              'to the resource being analyzed.')
@click.option('--econ_fpath', '-ef', type=STR, default=None,
              show_default=True,
              help='reV econ output file (optional argument that can be '
              'included if reV gen and econ data are being used from '
              'different files.')
@click.option('--res_fpath', '-rf', type=STR, default=None,
              show_default=True,
              help='Resource file, required if techmap dset is to be created.')
@click.option('--excl_dict', '-exd', type=STR, default=None,
              show_default=True,
              help=('String representation of a dictionary of exclusion '
                    'LayerMask arguments {layer: {kwarg: value}} where layer '
                    'is a dataset in excl_fpath and kwarg can be '
                    '"inclusion_range", "exclude_values", "include_values", '
                    '"inclusion_weights", "force_inclusion_values", '
                    '"use_as_weights", "exclude_nodata", and/or "weight".'))
@click.option('--res_class_dset', '-cd', type=STR, default=None,
              show_default=True,
              help='Dataset to determine the resource class '
              '(must be in gen_fpath).')
@click.option('--res_class_bins', '-cb', type=FLOATLIST, default=None,
              show_default=True,
              help='List of resource class bin edges.')
@click.option('--cf_dset', '-cf', type=STR, default='cf_mean-means',
              show_default=True,
              help='Dataset containing capacity factor values to aggregate.')
@click.option('--lcoe_dset', '-lc', type=STR, default='lcoe_fcr-means',
              show_default=True,
              help='Dataset containing lcoe values to aggregate.')
@click.option('--h5_dsets', '-hd', type=STRLIST, default=None,
              show_default=True,
              help='Additional datasets from the source gen/econ h5 files to '
              'aggregate.')
@click.option('--data_layers', '-d', type=STR, default=None,
              show_default=True,
              help='String representation of a dictionary of additional data '
              'layers to include in the aggregation e.g. '
              '{"slope": {"dset": "srtm_slope", "method": "mean"}}')
@click.option('--resolution', '-r', type=INT, default=64,
              show_default=True,
              help='Number of exclusion points along a squares edge to '
              'include in an aggregated supply curve point.')
@click.option('--excl_area', '-ea', type=FLOAT, default=0.0081,
              show_default=True,
              help='Area of an exclusion pixel in km2. None will try to '
              'infer the area from the profile transform attribute in '
              'excl_fpath.')
@click.option('--power_density', '-pd', type=STRFLOAT, default=None,
              show_default=True,
              help='Power density in MW/km2 or filepath to variable power '
              'density csv file. None will attempt to infer a constant '
              'power density from the generation meta data technology. '
              'Variable power density csvs must have "gid" and '
              '"power_density" columns where gid is the resource gid '
              '(typically wtk or nsrdb gid). and the power_density column '
              'is in MW/km2.')
@click.option('--area_filter_kernel', '-afk', type=STR, default='queen',
              show_default=True,
              help='Contiguous area filter kernel name ("queen", "rook").')
@click.option('--min_area', '-ma', type=FLOAT, default=None,
              show_default=True,
              help='Contiguous area filter minimum area, default is None '
              '(No minimum area filter).')
@click.option('--friction_fpath', '-ff', type=STR, default=None,
              show_default=True,
              help='Optional h5 filepath to friction surface data. '
              'Must match the exclusion shape/resolution and be '
              'paired with the --friction_dset input arg. The friction data '
              'creates a "mean_lcoe_friction" output which is the nominal '
              'LCOE multiplied by the friction data.')
@click.option('--friction_dset', '-fd', type=STR, default=None,
              show_default=True,
              help='Optional friction surface dataset in friction_fpath.')
@click.option('--cap_cost_scale', '-cs', type=STR, default=None,
              show_default=True,
              help='Optional LCOE scaling equation to implement "economies '
              'of scale". Equations must be in python string format and '
              'return a scalar value to multiply the capital cost by. '
              'Independent variables in the equation should match the names '
              'of the columns in the reV supply curve aggregation table. '
              'This equation may use numpy functions with the package prefix '
              '"np". This will not affect offshore wind LCOE.')
@click.option('--out_dir', '-o', type=STR, default='./',
              show_default=True,
              help='Directory to save aggregation summary output.')
@click.option('--max_workers', '-mw', type=INT, default=None,
              show_default=True,
              help=("Number of cores to run summary on. None is all "
                    "available cpus"))
@click.option('--sites_per_worker', '-spw', type=INT, default=100,
              show_default=True,
              help="Number of sc_points to summarize on each worker")
@click.option('--log_dir', '-ld', type=STR, default='./logs/',
              show_default=True,
              help='Directory to save aggregation logs.')
@click.option('-pre', '--pre_extract_inclusions', is_flag=True,
              help='Optional flag to pre-extract/compute the inclusion mask '
              'from the provided excl_dict, by default False. Typically '
              'faster to compute the inclusion mask on the fly with parallel '
              'workers.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def direct(ctx, excl_fpath, gen_fpath, tm_dset, econ_fpath, res_fpath,
           excl_dict, res_class_dset, res_class_bins,
           cf_dset, lcoe_dset, h5_dsets, data_layers, resolution, excl_area,
           power_density, area_filter_kernel, min_area, friction_fpath,
           friction_dset, cap_cost_scale, out_dir, max_workers,
           sites_per_worker, log_dir, pre_extract_inclusions, verbose):
    """reV Supply Curve Aggregation Summary CLI."""
    sites_per_worker = sites_per_worker if sites_per_worker else 100

    name = ctx.obj['NAME']
    ctx.obj['EXCL_FPATH'] = excl_fpath
    ctx.obj['GEN_FPATH'] = gen_fpath
    ctx.obj['ECON_FPATH'] = econ_fpath
    ctx.obj['RES_FPATH'] = res_fpath
    ctx.obj['TM_DSET'] = tm_dset
    ctx.obj['EXCL_DICT'] = excl_dict
    ctx.obj['RES_CLASS_DSET'] = res_class_dset
    ctx.obj['RES_CLASS_BINS'] = res_class_bins
    ctx.obj['CF_DSET'] = cf_dset
    ctx.obj['LCOE_DSET'] = lcoe_dset
    ctx.obj['H5_DSETS'] = h5_dsets
    ctx.obj['DATA_LAYERS'] = data_layers
    ctx.obj['RESOLUTION'] = resolution
    ctx.obj['EXCL_AREA'] = excl_area
    ctx.obj['POWER_DENSITY'] = power_density
    ctx.obj['AREA_FILTER_KERNEL'] = area_filter_kernel
    ctx.obj['MIN_AREA'] = min_area
    ctx.obj['FRICTION_FPATH'] = friction_fpath
    ctx.obj['FRICTION_DSET'] = friction_dset
    ctx.obj['CAP_COST_SCALE'] = cap_cost_scale
    ctx.obj['OUT_DIR'] = out_dir
    ctx.obj['MAX_WORKERS'] = max_workers
    ctx.obj['SITES_PER_WORKER'] = sites_per_worker
    ctx.obj['LOG_DIR'] = log_dir
    ctx.obj['PRE_EXTRACT_INCLUSIONS'] = pre_extract_inclusions
    ctx.obj['VERBOSE'] = verbose

    if ctx.invoked_subcommand is None:
        t0 = time.time()
        init_mult(name, log_dir, modules=[__name__, 'reV', 'rex'],
                  verbose=verbose)

        dsets = []
        paths = excl_fpath
        if isinstance(excl_fpath, str):
            paths = [excl_fpath]
        for fp in paths:
            with h5py.File(fp, mode='r') as f:
                dsets += list(f)

        if tm_dset in dsets:
            logger.info('Found techmap "{}".'.format(tm_dset))
        elif tm_dset not in dsets and not isinstance(excl_fpath, str):
            msg = ('Could not find techmap dataset "{}" and cannot run '
                   'techmap with arbitrary multiple exclusion filepaths '
                   'to write to: {}'.format(tm_dset, excl_fpath))
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            logger.info('Could not find techmap "{}". Running techmap module.'
                        .format(tm_dset))
            try:
                TechMapping.run(excl_fpath, res_fpath, dset=tm_dset)
            except Exception as e:
                logger.exception('TechMapping process failed. Received the '
                                 'following error:\n{}'.format(e))
                raise e

        if isinstance(excl_dict, str):
            excl_dict = dict_str_load(excl_dict)

        if isinstance(data_layers, str):
            data_layers = dict_str_load(data_layers)

        try:
            summary = SupplyCurveAggregation.summary(
                excl_fpath, gen_fpath, tm_dset,
                econ_fpath=econ_fpath,
                excl_dict=excl_dict,
                res_class_dset=res_class_dset,
                res_class_bins=res_class_bins,
                cf_dset=cf_dset,
                lcoe_dset=lcoe_dset,
                h5_dsets=h5_dsets,
                data_layers=data_layers,
                resolution=resolution,
                excl_area=excl_area,
                power_density=power_density,
                area_filter_kernel=area_filter_kernel,
                min_area=min_area,
                friction_fpath=friction_fpath,
                friction_dset=friction_dset,
                cap_cost_scale=cap_cost_scale,
                pre_extract_inclusions=pre_extract_inclusions,
                max_workers=max_workers,
                sites_per_worker=sites_per_worker)

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
        status = {'dirout': out_dir,
                  'fout': fn_out,
                  'job_status': 'successful',
                  'runtime': runtime,
                  'finput': finput,
                  'gen_fpath': gen_fpath,
                  'econ_fpath': econ_fpath,
                  'excl_fpath': excl_fpath,
                  'excl_dict': excl_dict,
                  'friction_fpath': friction_fpath,
                  'friction_dset': friction_dset,
                  'cf_dset': cf_dset,
                  'lcoe_dset': lcoe_dset,
                  'tm_dset': tm_dset,
                  'res_class_dset': res_class_dset,
                  'res_class_bins': res_class_bins,
                  'cap_cost_scale': cap_cost_scale,
                  'power_density': power_density,
                  'resolution': resolution,
                  'area_filter_kernel': area_filter_kernel,
                  'min_area': min_area}

        Status.make_job_file(out_dir, 'supply-curve-aggregation', name, status)


def get_node_cmd(name, excl_fpath, gen_fpath, econ_fpath, res_fpath, tm_dset,
                 excl_dict, res_class_dset, res_class_bins,
                 cf_dset, lcoe_dset, h5_dsets, data_layers, resolution,
                 excl_area, power_density, area_filter_kernel, min_area,
                 friction_fpath, friction_dset, cap_cost_scale,
                 out_dir, max_workers, sites_per_worker, log_dir,
                 pre_extract_inclusions, verbose):
    """Get a CLI call command for the SC aggregation cli."""

    args = ['-exf {}'.format(SLURM.s(excl_fpath)),
            '-gf {}'.format(SLURM.s(gen_fpath)),
            '-ef {}'.format(SLURM.s(econ_fpath)),
            '-rf {}'.format(SLURM.s(res_fpath)),
            '-tm {}'.format(SLURM.s(tm_dset)),
            '-exd {}'.format(SLURM.s(excl_dict)),
            '-cd {}'.format(SLURM.s(res_class_dset)),
            '-cb {}'.format(SLURM.s(res_class_bins)),
            '-cf {}'.format(SLURM.s(cf_dset)),
            '-lc {}'.format(SLURM.s(lcoe_dset)),
            '-hd {}'.format(SLURM.s(h5_dsets)),
            '-d {}'.format(SLURM.s(data_layers)),
            '-r {}'.format(SLURM.s(resolution)),
            '-ea {}'.format(SLURM.s(excl_area)),
            '-pd {}'.format(SLURM.s(power_density)),
            '-afk {}'.format(SLURM.s(area_filter_kernel)),
            '-ma {}'.format(SLURM.s(min_area)),
            '-ff {}'.format(SLURM.s(friction_fpath)),
            '-fd {}'.format(SLURM.s(friction_dset)),
            '-cs {}'.format(SLURM.s(cap_cost_scale)),
            '-o {}'.format(SLURM.s(out_dir)),
            '-mw {}'.format(SLURM.s(max_workers)),
            '-spw {}'.format(SLURM.s(sites_per_worker)),
            '-ld {}'.format(SLURM.s(log_dir)),
            ]

    if pre_extract_inclusions:
        args.append('-pre')

    if verbose:
        args.append('-v')

    cmd = ('python -m reV.supply_curve.cli_sc_aggregation -n {} direct {}'
           .format(SLURM.s(name), ' '.join(args)))
    logger.debug('Creating the following command line call:\n\t{}'.format(cmd))

    return cmd


@direct.command()
@click.option('--alloc', '-a', required=True, type=STR,
              help='SLURM allocation account name.')
@click.option('--walltime', '-wt', default=1.0, type=float,
              show_default=True,
              help='SLURM walltime request in hours. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR,
              show_default=True,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--memory', '-mem', default=None, type=INT,
              show_default=True,
              help='SLURM node memory request in GB. Default is None')
@click.option('--module', '-mod', default=None, type=STR,
              show_default=True,
              help='Module to load')
@click.option('--conda_env', '-env', default=None, type=STR,
              show_default=True,
              help='Conda env to activate')
@click.option('--stdout_path', '-sout', default=None, type=STR,
              show_default=True,
              help='Subprocess standard output path. Default is in out_dir.')
@click.pass_context
def slurm(ctx, alloc, walltime, feature, memory, module, conda_env,
          stdout_path):
    """slurm (Eagle) submission tool for reV supply curve aggregation."""
    name = ctx.obj['NAME']
    excl_fpath = ctx.obj['EXCL_FPATH']
    gen_fpath = ctx.obj['GEN_FPATH']
    econ_fpath = ctx.obj['ECON_FPATH']
    res_fpath = ctx.obj['RES_FPATH']
    tm_dset = ctx.obj['TM_DSET']
    excl_dict = ctx.obj['EXCL_DICT']
    res_class_dset = ctx.obj['RES_CLASS_DSET']
    res_class_bins = ctx.obj['RES_CLASS_BINS']
    cf_dset = ctx.obj['CF_DSET']
    lcoe_dset = ctx.obj['LCOE_DSET']
    h5_dsets = ctx.obj['H5_DSETS']
    data_layers = ctx.obj['DATA_LAYERS']
    resolution = ctx.obj['RESOLUTION']
    excl_area = ctx.obj['EXCL_AREA']
    power_density = ctx.obj['POWER_DENSITY']
    area_filter_kernel = ctx.obj['AREA_FILTER_KERNEL']
    min_area = ctx.obj['MIN_AREA']
    friction_fpath = ctx.obj['FRICTION_FPATH']
    friction_dset = ctx.obj['FRICTION_DSET']
    cap_cost_scale = ctx.obj['CAP_COST_SCALE']
    out_dir = ctx.obj['OUT_DIR']
    max_workers = ctx.obj['MAX_WORKERS']
    sites_per_worker = ctx.obj['SITES_PER_WORKER']
    log_dir = ctx.obj['LOG_DIR']
    pre_extract_inclusions = ctx.obj['PRE_EXTRACT_INCLUSIONS']
    verbose = ctx.obj['VERBOSE']

    if stdout_path is None:
        stdout_path = os.path.join(log_dir, 'stdout/')

    cmd = get_node_cmd(name, excl_fpath, gen_fpath, econ_fpath, res_fpath,
                       tm_dset, excl_dict,
                       res_class_dset, res_class_bins,
                       cf_dset, lcoe_dset, h5_dsets, data_layers,
                       resolution, excl_area,
                       power_density, area_filter_kernel, min_area,
                       friction_fpath, friction_dset, cap_cost_scale,
                       out_dir, max_workers, sites_per_worker,
                       log_dir, pre_extract_inclusions, verbose)

    slurm_manager = ctx.obj.get('SLURM_MANAGER', None)
    if slurm_manager is None:
        slurm_manager = SLURM()
        ctx.obj['SLURM_MANAGER'] = slurm_manager

    status = Status.retrieve_job_status(out_dir, 'supply-curve-aggregation',
                                        name, hardware='eagle',
                                        subprocess_manager=slurm_manager)

    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'
               .format(name, out_dir))
    elif 'fail' not in str(status).lower() and status is not None:
        msg = ('Job "{}" was found with status "{}", not resubmitting'
               .format(name, status))
    else:
        logger.info('Running reV SC aggregation on SLURM with '
                    'node name "{}"'.format(name))
        out = slurm_manager.sbatch(cmd, alloc=alloc, memory=memory,
                                   walltime=walltime, feature=feature,
                                   name=name, stdout_path=stdout_path,
                                   conda_env=conda_env, module=module)[0]
        if out:
            msg = ('Kicked off reV SC aggregation job "{}" '
                   '(SLURM jobid #{}).'
                   .format(name, out))
            Status.add_job(
                out_dir, 'supply-curve-aggregation', name, replace=True,
                job_attrs={'job_id': out, 'hardware': 'eagle',
                           'fout': '{}.csv'.format(name), 'dirout': out_dir})

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV SC Aggregation CLI.')
        raise
