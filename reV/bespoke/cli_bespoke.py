# -*- coding: utf-8 -*-
"""
Bespoke wind plant optimization CLI entry points.
"""

import click
import logging
import os
import pprint
import time
import re

from reV.bespoke.bespoke import BespokeWindPlants
from reV.config.bespoke import BespokeConfig
from reV.pipeline.status import Status
from reV.utilities.cli_dtypes import SAMFILES, SCPOINTS
from reV.utilities import ModuleName
from reV import __version__

from rex.utilities.cli_dtypes import (FLOAT, INT, STR, INTLIST, FLOATLIST,
                                      STRLIST, STR_OR_LIST, STRFLOAT)
from rex.utilities.hpc import SLURM
from rex.utilities.loggers import init_mult
from rex.utilities.utilities import get_class_properties, dict_str_load

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default='reV-bespoke', type=STR,
              show_default=True,
              help='reV bespoke job name, by default "reV-bespoke".')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """reV Bespoke Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c',
              required=True, type=click.Path(exists=True),
              help='reV bespoke configuration json file.')
@click.option('--points_range', '-pr',
              required=False, default=None, type=INTLIST,
              help='An optional input that specifies the (start, end) index '
              '(inclusive, exclusive) of the project points to analyze. If '
              'this is specified, the requested points are analyzed on a '
              'single worker. This input will override the config value.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, points_range, verbose):
    """Run reV gen from a config file."""

    config = (config_file if isinstance(config_file, BespokeConfig)
              else BespokeConfig(config_file))

    verbose = config.log_level == logging.DEBUG
    ctx.obj['VERBOSE'] = verbose = any([verbose, ctx.obj['VERBOSE']])

    if points_range is not None:
        config['points_range'] = points_range

    # take name from config
    name = ctx.obj['NAME'] = config.name

    # make output directory if does not exist
    if not os.path.exists(config.dirout):
        os.makedirs(config.dirout)

    # initialize loggers.
    node_tag = re.search('_node[0-9]*', name)
    if node_tag is None:
        init_mult(name, config.log_directory, modules=[__name__, 'reV', 'rex'],
                  verbose=verbose)

    # Initial log statements
    logger.info('Running reV Bespoke job with name "{}" from config file: "{}"'
                .format(name, config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.log_directory))
    logger.info('Source resource file: "{}"'.format(config.res_fpath))
    logger.info('Source exclusion file: "{}"'.format(config.excl_fpath))
    logger.info('Bespoke optimization objective function: "{}"'
                .format(config.objective_function))
    logger.info('Bespoke capital cost function: "{}"'
                .format(config.capital_cost_function))
    logger.info('Bespoke fixed operating cost function: "{}"'
                .format(config.fixed_operating_cost_function))
    logger.info('Bespoke variable operating cost function: "{}"'
                .format(config.variable_operating_cost_function))
    logger.info('Bespoke wake loss multiplier: "{}"'
                .format(config.wake_loss_multiplier))
    logger.info('The following project points were specified with '
                'points_range "{}": "{}"'
                .format(config.points_range, config.project_points))
    logger.info('The following SAM config files are available to this run:\n{}'
                .format(config.sam_files))
    logger.debug('The following exclusion dictionary was specified\n{}'
                 .format(pprint.pformat(config.excl_dict, indent=4)))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    # set config objects to be passed through invoke to direct methods
    ctx.obj['EXCL_FPATH'] = config.excl_fpath
    ctx.obj['RES_FPATH'] = config.res_fpath
    ctx.obj['TM_DSET'] = config.tm_dset
    ctx.obj['OBJECTIVE_FUNCTION'] = config.objective_function
    ctx.obj['CAPITAL_COST_FUNCTION'] = config.capital_cost_function
    ctx.obj['FIXED_OPERATING_COST_FUNCTION'] = \
        config.fixed_operating_cost_function
    ctx.obj['VARIABLE_OPERATING_COST_FUNCTION'] = \
        config.variable_operating_cost_function
    ctx.obj['POINTS'] = config.project_points
    ctx.obj['POINTS_RANGE'] = config.points_range
    ctx.obj['SAM_FILES'] = config.sam_files
    ctx.obj['MIN_SPACING'] = config.min_spacing
    ctx.obj['WAKE_LOSS_MULTIPLIER'] = config.wake_loss_multiplier
    ctx.obj['GA_KWARGS'] = config.ga_kwargs
    ctx.obj['OUTPUT_REQUEST'] = config.output_request
    ctx.obj['WS_BINS'] = config.ws_bins
    ctx.obj['WD_BINS'] = config.wd_bins
    ctx.obj['EXCL_DICT'] = config.excl_dict
    ctx.obj['AREA_FILTER_KERNEL'] = config.area_filter_kernel
    ctx.obj['MIN_AREA'] = config.min_area
    ctx.obj['RESOLUTION'] = config.resolution
    ctx.obj['EXCL_AREA'] = config.excl_area
    ctx.obj['DATA_LAYERS'] = config.data_layers
    ctx.obj['PRE_EXTRACT_INCLUSIONS'] = config.pre_extract_inclusions

    ctx.obj['LOG_DIR'] = config.log_directory
    ctx.obj['OUT_DIR'] = config.dirout
    ctx.obj['SITES_PER_WORKER'] = config.execution_control.sites_per_worker
    ctx.obj['MAX_WORKERS'] = config.execution_control.max_workers

    is_multi_node = (config.execution_control.option in ('eagle', 'slurm')
                     and config.points_range is None)
    nodes = config.execution_control.nodes if is_multi_node else 1

    pc = BespokeWindPlants._parse_points(config.excl_fpath,
                                         config.res_fpath,
                                         config.tm_dset,
                                         config.resolution,
                                         config.project_points,
                                         config.points_range,
                                         config.sam_files,
                                         sites_per_worker=1,
                                         workers=nodes)

    if len(pc) > 1:
        logger.info('Distributing the {} Bespoke project points to {} jobs.'
                    .format(len(pc.project_points), len(pc)))
        for i, pc_sub in enumerate(pc):
            logger.info('Creating distributed job submission for: {}'
                        .format(pc_sub.project_points))
            ctx.obj['NAME'] = name + '_node{}'.format(str(i).zfill(2))
            config._name = name + '_node{}'.format(str(i).zfill(2))
            ctx.invoke(from_config, config_file=config,
                       points_range=pc_sub.split_range, verbose=verbose)

    else:
        fout = name + '.h5'
        out_fpath = os.path.join(config.dirout, fout)
        ctx.obj['OUT_FPATH'] = out_fpath
        if config.execution_control.option == 'local':
            logger.info('Running Bespoke project with {} points '
                        'corresponding to points_range {}.'
                        .format(len(pc.project_points), config.points_range))
            status = Status.retrieve_job_status(config.dirout,
                                                ModuleName.BESPOKE,
                                                name)
            if status == 'successful':
                logger.info('Bespoke job with name "{}" was already '
                            'successfully run in directory: {}'
                            .format(name, config.dirout))
            else:
                job_attrs = {'hardware': 'local', 'fout': fout,
                             'dirout': config.dirout}
                Status.add_job(config.dirout, ModuleName.BESPOKE,
                               name, replace=True,
                               job_attrs=job_attrs)
                kws = {'excl_fpath': config.excl_fpath,
                       'res_fpath': config.res_fpath,
                       'out_fpath': out_fpath,
                       'tm_dset': config.tm_dset,
                       'objective_function':
                           config.objective_function,
                       'capital_cost_function':
                           config.capital_cost_function,
                       'fixed_operating_cost_function':
                           config.fixed_operating_cost_function,
                       'variable_operating_cost_function':
                           config.variable_operating_cost_function,
                       'points': config.project_points,
                       'sam_files': config.sam_files,
                       'points_range': config.points_range,
                       'min_spacing': config.min_spacing,
                       'wake_loss_multiplier': config.wake_loss_multiplier,
                       'ga_kwargs': config.ga_kwargs,
                       'output_request': config.output_request,
                       'ws_bins': config.ws_bins,
                       'wd_bins': config.wd_bins,
                       'excl_dict': config.excl_dict,
                       'area_filter_kernel': config.area_filter_kernel,
                       'min_area': config.min_area,
                       'resolution': config.resolution,
                       'excl_area': config.excl_area,
                       'log_dir': config.log_directory,
                       'max_workers': config.execution_control.max_workers,
                       'data_layers': config.data_layers,
                       'pre_extract_inclusions': config.pre_extract_inclusions,
                       'verbose': verbose}
                ctx.invoke(direct, **kws)

        elif config.execution_control.option in ('eagle', 'slurm'):
            stdout_path = os.path.join(config.log_directory, 'stdout')
            ctx.invoke(slurm,
                       alloc=config.execution_control.allocation,
                       walltime=config.execution_control.walltime,
                       feature=config.execution_control.feature,
                       memory=config.execution_control.memory,
                       module=config.execution_control.module,
                       conda_env=config.execution_control.conda_env,
                       stdout_path=stdout_path,
                       )


@main.command()
def valid_config_keys():
    """
    Echo the valid Bespoke config keys
    """
    click.echo(', '.join(get_class_properties(BespokeConfig)))


@main.group(invoke_without_command=True)
@click.option('--excl_fpath', '-exf', type=STR_OR_LIST, required=True,
              help='Single exclusions file (.h5) or a '
              'list of exclusion files (.h5, .h5).')
@click.option('--res_fpath', '-rf', type=STR, required=True,
              help='reV resource data file (e.g. WTK .h5 file). '
              'Can include a unix-style wildcard to source multiple years of '
              'data, e.g. res_fpath="/datasets/WIND/conus/v1.0.0/wtk_*.h5"')
@click.option('--out_fpath', '-of', type=STR, required=True,
              help='Filepath to save output data. Must be a .h5 path.')
@click.option('--tm_dset', '-tm', type=STR, required=True,
              help='Dataset in the exclusions file that maps the exclusions '
              'to the resource data file being analyzed.')
@click.option('--objective_function', '-obj', required=True, type=STR,
              help='The optimization objective function as a string, should'
              'return the objective to be minimized during optimization.'
              'Variables available are:'
              '- n_turbines: the number of turbines'
              '- system_capacity: wind plant capacity'
              '- aep: annual energy production'
              '- fixed_charge_rate: user input if included in sam sys config'
              '- capital_cost: the cap cost output by the cap cost function'
              '- fixed_operating_cost: the foc output by the foc function'
              '- variable_operating_cost: the voc output by the voc function'
              '- self.wind_plant: the SAM wind plant object, through which'
              'all SAM variables can be accessed')
@click.option('--capital_cost_function', '-cc',
              required=True, type=STR,
              help='The plant capital cost function as a string, must return '
              'the total capital cost in $. Has access to the same variables '
              'as the objective_function.')
@click.option('--fixed_operating_cost_function', '-foc',
              required=True, type=STR,
              help='The plant annual fixed operating cost function as a '
              'string, must return the fixed operating cost in $/year. Has '
              'access to the same variables as the objective_function.')
@click.option('--variable_operating_cost_function', '-voc',
              required=True, type=STR,
              help='The plant annual variable operating cost function as a '
              'string, must return the variable operating cost in $/kWh. Has '
              'access to the same variables as the objective_function.')
@click.option('--points', '-p',
              default=None, type=SCPOINTS, show_default=True,
              help='Project points to analyze. This can either be a string '
              'pointing to a csv with "gid" and "config" columns, or None '
              '(default) which will analyze all supply curve points in the '
              'given exclusion techmap.')
@click.option('--sam_files', '-sf', required=True, type=SAMFILES,
              help='SAM config files (required). Should be a single filepath '
              'to a sam config .json if points is None or a dictionary of '
              'key-value pairs where the keys map to the points "config" '
              'table column.')
@click.option('--points_range', '-pr',
              default=None, type=INTLIST, show_default=True,
              help='Optional list of (start, end) (inclusive, exclusive) '
              'index values that will slice the points input.')
@click.option('--min_spacing', '-ms',
              default='5x', type=STRFLOAT, show_default=True,
              help='Minimum spacing between turbines in meters. Can also be '
              'a string like "5x" (default) which is interpreted as 5 times '
              'the turbine rotor diameter.')
@click.option('--wake_loss_multiplier', '-wlm', type=FLOAT, default=1,
              show_default=True,
              help='A multiplier used to scale the annual energy lost due to '
              'wake losses. **IMPORTANT**: This multiplier will ONLY be '
              'applied during the optimization process and will NOT be '
              'come through in output values such as hourly generation '
              'profiles, aep, any of the cost functions, or even the output '
              'objective.')
@click.option('--ga_kwargs', '-gakw', type=STR, default=None,
              show_default=True,
              help='Dictionary of keyword arguments to pass to GA '
              'initialization.If `None`, default initialization values are '
              'used. See the reV.bespoke.gradient_free.GeneticAlgorithm class '
              'for a description of the allowed keyword arguments.')
@click.option('--output_request', '-or', type=STRLIST,
              default=['cf_mean', 'system_capacity'], show_default=True,
              help=('List of requested output variable names from SAM.'))
@click.option('--ws_bins', '-ws',
              default=[0, 20, 5], type=FLOATLIST, show_default=True,
              help='Get the windspeed binning arguments This should be a '
              '3-entry list with [start, stop, step] for the windspeed '
              'binning of the wind joint probability distribution. The stop '
              'value is inclusive, so ws_bins=[0, 20, 5] would result in '
              'four bins with bin edges [0, 5, 10, 15, 20]')
@click.option('--wd_bins', '-wd',
              default=[0, 360, 45], type=FLOATLIST, show_default=True,
              help='Get the winddirection binning arguments This should be a '
              '3-entry list with [start, stop, step] for the winddirection '
              'binning of the wind joint probability distribution. The stop '
              'value is inclusive, so ws_bins=[0, 360, 90] would result in '
              'four bins with bin edges [0, 90, 180, 270, 360]')
@click.option('--excl_dict', '-exd', type=STR, default=None,
              show_default=True,
              help='String representation of a dictionary of exclusion '
              'LayerMask arguments {layer: {kwarg: value}} where layer '
              'is a dataset in excl_fpath and kwarg can be '
              '"exclude_values", "exclude_range", "include_values", '
              '"include_range", "include_weights", "force_include_values", '
              '"force_include_range", "use_as_weights", "weight", '
              '"exclude_nodata", and/or "nodata_value".')
@click.option('--area_filter_kernel', '-afk', type=STR, default='queen',
              show_default=True,
              help='Contiguous area filter kernel name ("queen", "rook").')
@click.option('--min_area', '-ma', type=FLOAT, default=None,
              show_default=True,
              help='Contiguous area filter minimum area, default is None '
              '(No minimum area filter).')
@click.option('--resolution', '-r', type=INT, default=64,
              show_default=True,
              help='Number of exclusion points along a squares edge to '
              'include in an aggregated supply curve point.')
@click.option('--excl_area', '-ea',
              type=FLOAT, default=None, show_default=True,
              help='Area of an exclusion pixel in km2. None will try to '
              'infer the area from the profile transform attribute in '
              'excl_fpath.')
@click.option('--log_dir', '-lo', default='./out/log_bespoke', type=STR,
              help='Bespoke log file directory. Default is ./out/log_bespoke')
@click.option('--max_workers', '-mw', type=INT, default=None,
              show_default=True,
              help='Number of workers. Use 1 for serial, None for all cores.')
@click.option('--data_layers', '-d', type=STR, default=None,
              show_default=True,
              help='String representation of a dictionary of additional data '
              'layers to include in the aggregation e.g. '
              '{"slope": {"dset": "srtm_slope", "method": "mean", '
              '"fpath": "./excl.h5"}}')
@click.option('-pre', '--pre_extract_inclusions', is_flag=True,
              help='Optional flag to pre-extract/compute the inclusion mask '
              'from the provided excl_dict, by default False. Typically '
              'faster to compute the inclusion mask on the fly with parallel '
              'workers.')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def direct(ctx, excl_fpath, res_fpath, out_fpath, tm_dset, objective_function,
           capital_cost_function, fixed_operating_cost_function,
           variable_operating_cost_function, points, sam_files, points_range,
           min_spacing, wake_loss_multiplier, ga_kwargs, output_request,
           ws_bins, wd_bins, excl_dict, area_filter_kernel, min_area,
           resolution, excl_area, log_dir, max_workers, data_layers,
           pre_extract_inclusions, verbose):
    """Run reV Bespoke directly w/o a config file."""
    ctx.obj['EXCL_FPATH'] = excl_fpath
    ctx.obj['RES_FPATH'] = res_fpath
    ctx.obj['OUT_FPATH'] = out_fpath
    ctx.obj['TM_DSET'] = tm_dset
    ctx.obj['OBJECTIVE_FUNCTION'] = objective_function
    ctx.obj['CAPITAL_COST_FUNCTION'] = capital_cost_function
    ctx.obj['FIXED_OPERATING_COST_FUNCTION'] = fixed_operating_cost_function
    ctx.obj['VARIABLE_OPERATING_COST_FUNCTION'] = \
        variable_operating_cost_function
    ctx.obj['POINTS'] = points
    ctx.obj['POINTS_RANGE'] = points_range
    ctx.obj['SAM_FILES'] = sam_files
    ctx.obj['MIN_SPACING'] = min_spacing
    ctx.obj['WAKE_LOSS_MULTIPLIER'] = wake_loss_multiplier
    ctx.obj['GA_KWARGS'] = ga_kwargs
    ctx.obj['OUTPUT_REQUEST'] = output_request
    ctx.obj['WS_BINS'] = ws_bins
    ctx.obj['WD_BINS'] = wd_bins
    ctx.obj['EXCL_DICT'] = excl_dict
    ctx.obj['AREA_FILTER_KERNEL'] = area_filter_kernel
    ctx.obj['MIN_AREA'] = min_area
    ctx.obj['RESOLUTION'] = resolution
    ctx.obj['EXCL_AREA'] = excl_area
    ctx.obj['DATA_LAYERS'] = data_layers
    ctx.obj['PRE_EXTRACT_INCLUSIONS'] = pre_extract_inclusions
    ctx.obj['LOG_DIR'] = log_dir
    verbose = any([verbose, ctx.obj['VERBOSE']])

    name = ctx.obj['NAME']

    if ctx.invoked_subcommand is None:
        init_mult(name, log_dir, modules=[__name__, 'reV', 'rex'],
                  verbose=verbose, node=True)

        for key, val in ctx.obj.items():
            logger.debug('ctx var passed to local method: "{}" : "{}" with '
                         'type "{}"'.format(key, val, type(val)))

        logger.info('Bespoke local is being run with with job name "{}" and '
                    'resource file: {}. Target output path is: {}'
                    .format(name, res_fpath, out_fpath))

        if isinstance(data_layers, str):
            data_layers = dict_str_load(data_layers)

        if isinstance(excl_dict, str):
            excl_dict = dict_str_load(excl_dict)

        if isinstance(ga_kwargs, str):
            ga_kwargs = dict_str_load(ga_kwargs)

        t0 = time.time()

        try:
            # Execute the Bespoke module with smart data flushing.
            BespokeWindPlants.run(
                excl_fpath, res_fpath, tm_dset,
                objective_function,
                capital_cost_function,
                fixed_operating_cost_function,
                variable_operating_cost_function,
                points, sam_files,
                points_range=points_range,
                min_spacing=min_spacing,
                wake_loss_multiplier=wake_loss_multiplier,
                ga_kwargs=ga_kwargs,
                output_request=output_request,
                ws_bins=ws_bins,
                wd_bins=wd_bins,
                excl_dict=excl_dict,
                area_filter_kernel=area_filter_kernel,
                min_area=min_area,
                resolution=resolution,
                excl_area=excl_area,
                data_layers=data_layers,
                pre_extract_inclusions=pre_extract_inclusions,
                max_workers=max_workers,
                out_fpath=out_fpath)
        except Exception as e:
            msg = ('reV Bespoke optimization failed. Received the '
                   'following error:\n{}'.format(e))
            logger.exception(msg)
            raise RuntimeError(msg) from e

        tmp_str = ' with points range {}'.format(points_range)
        out_dir, fout = os.path.split(out_fpath)
        runtime = (time.time() - t0) / 60
        logger.info('Bespoke compute complete for project points "{0}"{1}. '
                    'Time elapsed: {2:.2f} min. Target output dir: {3}'
                    .format(points, tmp_str if points_range else '',
                            runtime, out_dir))

        # add job to reV status file.
        status = {'dirout': out_dir, 'fout': fout, 'job_status': 'successful',
                  'runtime': runtime, 'res_fpath': res_fpath,
                  'excl_fpath': excl_fpath, 'tm_dset': tm_dset,
                  'objective_function': objective_function,
                  'capital_cost_function': capital_cost_function,
                  'fixed_operating_cost_function':
                      fixed_operating_cost_function,
                  'variable_operating_cost_function':
                      variable_operating_cost_function,
                  'excl_dict': excl_dict}
        Status.make_job_file(out_dir, ModuleName.BESPOKE, name, status)


def get_node_cmd(name, kwargs):
    """Make a CLI command string."""
    arg_main = '-n {}'.format(SLURM.s(name))

    arg_direct = [
        '-exf {}'.format(SLURM.s(kwargs['EXCL_FPATH'])),
        '-rf {}'.format(SLURM.s(kwargs['RES_FPATH'])),
        '-of {}'.format(SLURM.s(kwargs['OUT_FPATH'])),
        '-tm {}'.format(SLURM.s(kwargs['TM_DSET'])),
        '-obj {}'.format(SLURM.s(kwargs['OBJECTIVE_FUNCTION'])),
        '-cc {}'.format(SLURM.s(kwargs['CAPITAL_COST_FUNCTION'])),
        '-foc {}'.format(SLURM.s(kwargs['FIXED_OPERATING_COST_FUNCTION'])),
        '-voc {}'.format(SLURM.s(kwargs['VARIABLE_OPERATING_COST_FUNCTION'])),
        '-p {}'.format(SLURM.s(kwargs['POINTS'])),
        '-sf {}'.format(SLURM.s(kwargs['SAM_FILES'])),
        '-pr {}'.format(SLURM.s(kwargs['POINTS_RANGE'])),
        '-ms {}'.format(SLURM.s(kwargs['MIN_SPACING'])),
        '-wlm {}'.format(SLURM.s(kwargs['WAKE_LOSS_MULTIPLIER'])),
        '-gakw {}'.format(SLURM.s(kwargs['GA_KWARGS'])),
        '-or {}'.format(SLURM.s(kwargs['OUTPUT_REQUEST'])),
        '-ws {}'.format(SLURM.s(kwargs['WS_BINS'])),
        '-wd {}'.format(SLURM.s(kwargs['WD_BINS'])),
        '-exd {}'.format(SLURM.s(kwargs['EXCL_DICT'])),
        '-afk {}'.format(SLURM.s(kwargs['AREA_FILTER_KERNEL'])),
        '-ma {}'.format(SLURM.s(kwargs['MIN_AREA'])),
        '-r {}'.format(SLURM.s(kwargs['RESOLUTION'])),
        '-ea {}'.format(SLURM.s(kwargs['EXCL_AREA'])),
        '-d {}'.format(SLURM.s(kwargs['DATA_LAYERS'])),
        '-lo {}'.format(SLURM.s(kwargs['LOG_DIR'])),
        '-mw {}'.format(SLURM.s(kwargs['MAX_WORKERS']))]

    if kwargs['PRE_EXTRACT_INCLUSIONS']:
        arg_direct.append('-pre')

    if kwargs['VERBOSE']:
        arg_direct.append('-v')

    cmd = ('python -m reV.bespoke.cli_bespoke '
           '{arg_main} direct {arg_direct}'
           .format(arg_main=arg_main,
                   arg_direct=' '.join(arg_direct)))
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
    """slurm (Eagle) submission tool for reV Bespoke wind plant optimization"""
    name = ctx.obj['NAME']
    log_dir = ctx.obj['LOG_DIR']
    out_fpath = ctx.obj['OUT_FPATH']
    out_dir = os.path.dirname(out_fpath)

    if stdout_path is None:
        stdout_path = os.path.join(log_dir, 'stdout/')

    cmd = get_node_cmd(name, ctx.obj)

    slurm_manager = ctx.obj.get('SLURM_MANAGER', None)
    if slurm_manager is None:
        slurm_manager = SLURM()
        ctx.obj['SLURM_MANAGER'] = slurm_manager

    status = Status.retrieve_job_status(out_dir, ModuleName.BESPOKE,
                                        name, hardware='slurm',
                                        subprocess_manager=slurm_manager)

    msg = ''
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'
               .format(name, out_dir))
    elif 'fail' not in str(status).lower() and status is not None:
        msg = ('Job "{}" was found with status "{}", not resubmitting'
               .format(name, status))
    else:
        logger.info('Running reV Bespoke on SLURM with '
                    'node name "{}"'.format(name))
        out = slurm_manager.sbatch(cmd, alloc=alloc, memory=memory,
                                   walltime=walltime, feature=feature,
                                   name=name, stdout_path=stdout_path,
                                   conda_env=conda_env, module=module)[0]
        if out:
            msg = ('Kicked off reV Bespoke job "{}" (SLURM jobid #{}).'
                   .format(name, out))
            Status.add_job(
                out_dir, ModuleName.BESPOKE, name, replace=True,
                job_attrs={'job_id': out, 'hardware': 'slurm',
                           'fout': '{}.csv'.format(name), 'dirout': out_dir})

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Bespoke CLI')
        raise
