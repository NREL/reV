# -*- coding: utf-8 -*-
"""
QA/QC CLI entry points.
"""
import click
import logging
import numpy as np
import os
import pprint

from rex.utilities.cli_dtypes import STR, STRLIST, INT, FLOAT
from rex.utilities.execution import SLURM
from rex.utilities.loggers import init_logger, init_mult
from rex.utilities.utilities import dict_str_load, get_class_properties

from reV.config.qa_qc_config import QaQcConfig
from reV.pipeline.status import Status
from reV.qa_qc.qa_qc import QaQc
from reV.qa_qc.summary import (SummarizeH5, SummarizeSupplyCurve,
                               SupplyCurvePlot, ExclusionsMask)

logger = logging.getLogger(__name__)


@click.group()
@click.option('--name', '-n', default='reV-QA_QC', type=STR,
              help='reV QA/QC name, by default "reV-QA/QC".')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """reV QA/QC Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
def valid_config_keys():
    """
    Echo the valid QaQc config keys
    """
    click.echo(', '.join(get_class_properties(QaQcConfig)))


@main.group(chain=True)
@click.option('--out_dir', '-o', type=click.Path(), required=True,
              help="Directory path to save summary tables and plots too")
@click.option('--log_file', '-log', type=click.Path(), default=None,
              help='File to log to, by default None')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def summarize(ctx, out_dir, log_file, verbose):
    """
    Summarize reV data
    """
    ctx.obj['OUT_DIR'] = out_dir
    if any([verbose, ctx.obj['VERBOSE']]):
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    ctx.obj['LOGGER'] = init_logger('reV.qa_qc', log_file=log_file,
                                    log_level=log_level)


@summarize.command()
@click.option('--h5_file', '-h5', type=click.Path(exists=True), required=True,
              help='Path to .h5 file to summarize')
@click.option('--dsets', '-ds', type=STRLIST, default=None,
              help='Datasets to summarize, by default None')
@click.option('--group', '-grp', type=STR, default=None,
              help=('Group within h5_file to summarize datasets for, by '
                    'default None'))
@click.option('--process_size', '-ps', type=INT, default=None,
              help='Number of sites to process at a time, by default None')
@click.option('--max_workers', '-w', type=INT, default=None,
              help=('Number of workers to use when summarizing 2D datasets,'
                    ' by default None'))
@click.pass_context
def h5(ctx, h5_file, dsets, group, process_size, max_workers):
    """
    Summarize datasets in .h5 file
    """
    SummarizeH5.run(h5_file, ctx.obj['OUT_DIR'], group=group, dsets=dsets,
                    process_size=process_size, max_workers=max_workers)


@summarize.command()
@click.option('--plot_type', '-plt', default='plotly',
              type=click.Choice(['plot', 'plotly'], case_sensitive=False),
              help=(" plot_type of plot to create 'plot' or 'plotly', by "
                    "default 'plot'"))
@click.option('--cmap', '-cmap', type=str, default='viridis',
              help="Colormap name, by default 'viridis'")
@click.pass_context
def scatter_plots(ctx, plot_type, cmap):
    """
    create scatter plots from h5 summary tables
    """
    QaQc.create_scatter_plots(ctx.obj['OUT_DIR'], plot_type, cmap)


@summarize.command()
@click.option('--sc_table', '-sct', type=click.Path(exists=True),
              required=True, help='Path to .csv containing Supply Curve table')
@click.option('--columns', '-cols', type=STRLIST, default=None,
              help=('Column(s) to summarize, if None summarize all numeric '
                    'columns, by default None'))
@click.pass_context
def supply_curve_table(ctx, sc_table, columns):
    """
    Summarize Supply Curve Table
    """
    ctx.obj['SC_TABLE'] = sc_table
    SummarizeSupplyCurve.run(sc_table, ctx.obj['OUT_DIR'], columns=columns)


@summarize.command()
@click.option('--sc_table', '-sct', type=click.Path(exists=True), default=None,
              help=("Path to .csv containing Supply Curve table, can be "
                    "supplied in 'supply-curve-table'"))
@click.option('--plot_type', '-plt', default='plotly',
              type=click.Choice(['plot', 'plotly'], case_sensitive=False),
              help=(" plot_type of plot to create 'plot' or 'plotly', by "
                    "default 'plot'"))
@click.option('--lcoe', '-lcoe', type=STR, default='mean_lcoe',
              help="LCOE value to plot, by default 'mean_lcoe'")
@click.pass_context
def supply_curve_plot(ctx, sc_table, plot_type, lcoe):
    """
    Plot Supply Curve (cumulative capacity vs LCOE)
    """
    if sc_table is None:
        sc_table = ctx.obj['SC_TABLE']

    SupplyCurvePlot.plot(sc_table, ctx.obj['OUT_DIR'],
                         plot_type=plot_type, lcoe=lcoe)


@summarize.command()
@click.option('--excl_mask', '-mask', type=click.Path(exists=True),
              required=True,
              help='Path to .npy file containing final exclusions mask')
@click.option('--plot_type', '-plt', default='plotly',
              type=click.Choice(['plot', 'plotly'], case_sensitive=False),
              help=(" plot_type of plot to create 'plot' or 'plotly', by "
                    "default 'plot'"))
@click.option('--cmap', '-cmap', type=str, default='viridis',
              help="Colormap name, by default 'viridis'")
@click.option('--plot_step', '-step', type=int, default=100,
              help="Step between points to plot")
@click.pass_context
def exclusions_mask(ctx, excl_mask, plot_type, cmap, plot_step):
    """
    create heat map of exclusions mask
    """
    excl_mask = np.load(excl_mask)
    ExclusionsMask.plot(excl_mask, ctx.obj['OUT_DIR'],
                        plot_type=plot_type, cmap=cmap,
                        plot_step=plot_step)


@main.command()
@click.option('--h5_file', '-h5', type=click.Path(exists=True), required=True,
              help='Path to .h5 file to summarize')
@click.option('--out_dir', '-o', type=click.Path(), required=True,
              help="Project output directory path.")
@click.option('--sub_dir', '-sd', type=STR, required=True,
              help="Sub directory to save summary tables and plots too")
@click.option('--dsets', '-ds', type=STRLIST, default=None,
              help='Datasets to summarize, by default None')
@click.option('--group', '-grp', type=STR, default=None,
              help=('Group within h5_file to summarize datasets for, by '
                    'default None'))
@click.option('--process_size', '-ps', type=INT, default=None,
              help='Number of sites to process at a time, by default None')
@click.option('--max_workers', '-w', type=INT, default=None,
              help=('Number of workers to use when summarizing 2D datasets,'
                    ' by default None'))
@click.option('--plot_type', '-plt', default='plotly',
              type=click.Choice(['plot', 'plotly'], case_sensitive=False),
              help=(" plot_type of plot to create 'plot' or 'plotly', by "
                    "default 'plot'"))
@click.option('--cmap', '-cmap', type=str, default='viridis',
              help="Colormap name, by default 'viridis'")
@click.option('--log_file', '-log', type=click.Path(), default=None,
              help='File to log to, by default None')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.option('-t', '--terminal', is_flag=True,
              help=('Flag for terminal QA pipeline call. '
                    'Prints successful status file.'))
@click.pass_context
def reV_h5(ctx, h5_file, out_dir, sub_dir, dsets, group, process_size,
           max_workers, plot_type, cmap, log_file, verbose, terminal):
    """
    Summarize and plot data for reV h5_file
    """
    name = ctx.obj['NAME']
    if any([verbose, ctx.obj['VERBOSE']]):
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    init_logger('reV.qa_qc', log_file=log_file, log_level=log_level)

    qa_dir = out_dir
    if sub_dir is not None:
        qa_dir = os.path.join(out_dir, sub_dir)

    QaQc.h5(h5_file, qa_dir, dsets=dsets, group=group,
            process_size=process_size, max_workers=max_workers,
            plot_type=plot_type, cmap=cmap)

    if terminal:
        status = {'dirout': out_dir, 'job_status': 'successful',
                  'finput': h5_file}
        Status.make_job_file(out_dir, 'qa-qc', name, status)


@main.command()
@click.option('--sc_table', '-sct', type=click.Path(exists=True),
              required=True, help='Path to .csv containing Supply Curve table')
@click.option('--out_dir', '-o', type=click.Path(), required=True,
              help="Project output directory path.")
@click.option('--sub_dir', '-sd', type=STR, required=True,
              help="Sub directory to save summary tables and plots too")
@click.option('--columns', '-cols', type=STRLIST, default=None,
              help=('Column(s) to summarize, if None summarize all numeric '
                    'columns, by default None'))
@click.option('--plot_type', '-plt', default='plotly',
              type=click.Choice(['plot', 'plotly'], case_sensitive=False),
              help=(" plot_type of plot to create 'plot' or 'plotly', by "
                    "default 'plot'"))
@click.option('--cmap', '-cmap', type=str, default='viridis',
              help="Colormap name, by default 'viridis'")
@click.option('--lcoe', '-lcoe', type=STR, default='mean_lcoe',
              help="LCOE column label to plot, by default 'mean_lcoe'")
@click.option('--log_file', '-log', type=click.Path(), default=None,
              help='File to log to, by default None')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.option('-t', '--terminal', is_flag=True,
              help=('Flag for terminal QA pipeline call. '
                    'Prints successful status file.'))
@click.pass_context
def supply_curve(ctx, sc_table, out_dir, sub_dir, columns, plot_type, cmap,
                 lcoe, log_file, verbose, terminal):
    """
    Summarize and plot reV Supply Curve data
    """
    name = ctx.obj['NAME']
    if any([verbose, ctx.obj['VERBOSE']]):
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    init_logger('reV.qa_qc', log_file=log_file, log_level=log_level)

    qa_dir = out_dir
    if sub_dir is not None:
        qa_dir = os.path.join(out_dir, sub_dir)

    QaQc.supply_curve(sc_table, qa_dir, columns=columns, lcoe=lcoe,
                      plot_type=plot_type, cmap=cmap)

    if terminal:
        status = {'dirout': out_dir, 'job_status': 'successful',
                  'finput': sc_table}
        Status.make_job_file(out_dir, 'qa-qc', name, status)


@main.command()
@click.option('--excl_fpath', '-excl', type=click.Path(exists=True),
              required=True,
              help='Exclusions file (.h5).')
@click.option('--out_dir', '-o', type=click.Path(), required=True,
              help="Project output directory path.")
@click.option('--sub_dir', '-sd', type=STR, required=True,
              help="Sub directory to save summary tables and plots too")
@click.option('--excl_dict', '-exd', type=STR, default=None,
              help='String representation of a dictionary of exclusions '
              'LayerMask arguments {layer: {kwarg: value}} where layer is a '
              'dataset in excl_fpath and kwarg can be "inclusion_range", '
              '"exclude_values", "include_values", "use_as_weights", '
              '"exclude_nodata", and/or "weight".')
@click.option('--area_filter_kernel', '-afk', type=STR, default='queen',
              help='Contiguous area filter kernel name ("queen", "rook").')
@click.option('--min_area', '-ma', type=FLOAT, default=None,
              help='Contiguous area filter minimum area, default is None '
              '(No minimum area filter).')
@click.option('--plot_type', '-plt', default='plotly',
              type=click.Choice(['plot', 'plotly'], case_sensitive=False),
              help=(" plot_type of plot to create 'plot' or 'plotly', by "
                    "default 'plot'"))
@click.option('--cmap', '-cmap', type=str, default='viridis',
              help="Colormap name, by default 'viridis'")
@click.option('--plot_step', '-step', type=int, default=100,
              help="Step between points to plot")
@click.option('--log_file', '-log', type=click.Path(), default=None,
              help='File to log to, by default None')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.option('-t', '--terminal', is_flag=True,
              help=('Flag for terminal QA pipeline call. '
                    'Prints successful status file.'))
@click.pass_context
def exclusions(ctx, excl_fpath, out_dir, sub_dir, excl_dict,
               area_filter_kernel, min_area, plot_type, cmap, plot_step,
               log_file, verbose, terminal):
    """
    Extract and plot reV exclusions mask
    """
    name = ctx.obj['NAME']
    if any([verbose, ctx.obj['VERBOSE']]):
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    init_logger('reV.qa_qc', log_file=log_file, log_level=log_level)

    qa_dir = out_dir
    if sub_dir is not None:
        qa_dir = os.path.join(out_dir, sub_dir)

    if isinstance(excl_dict, str):
        excl_dict = dict_str_load(excl_dict)

    QaQc.exclusions_mask(excl_fpath, qa_dir, layers_dict=excl_dict,
                         min_area=min_area, kernel=area_filter_kernel,
                         plot_type=plot_type, cmap=cmap, plot_step=plot_step)

    if terminal:
        status = {'dirout': out_dir, 'job_status': 'successful',
                  'finput': excl_fpath}
        Status.make_job_file(out_dir, 'qa-qc', name, status)


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV QA/QC configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV QA/QC from a config file."""
    name = ctx.obj['NAME']

    # Instantiate the config object
    config = QaQcConfig(config_file)

    # take name from config if not default
    if config.name.lower() != 'rev':
        name = config.name

    ctx.obj['NAME'] = name

    # Enforce verbosity if logging level is specified in the config
    verbose = config.log_level == logging.DEBUG

    # initialize loggers
    init_mult(name, config.logdir, modules=[__name__, 'reV.config',
                                            'reV.utilities', 'reV.qa_qc',
                                            'rex.utilities'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV supply curve from config '
                'file: "{}"'.format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    if config.execution_control.option == 'local':
        status = Status.retrieve_job_status(config.dirout, 'qa-qc',
                                            name)
        if status != 'successful':
            Status.add_job(
                config.dirout, 'qa-qc', name, replace=True,
                job_attrs={'hardware': 'local',
                           'dirout': config.dirout})

            terminal = False
            for i, module in enumerate(config.module_names):
                if i == len(config.module_names) - 1:
                    terminal = True

                module_config = config.get_module_inputs(module)
                fpath = module_config.fpath
                if module.lower() == 'exclusions':
                    log_file = os.path.join(
                        config.logdir,
                        os.path.basename(fpath).replace('.h5', '.log'))
                    afk = module_config.area_filter_kernel
                    ctx.invoke(exclusions,
                               excl_fpath=fpath,
                               out_dir=config.dirout,
                               sub_dir=module_config.sub_dir,
                               excl_dict=module_config.excl_dict,
                               area_filter_kernel=afk,
                               min_area=module_config.min_area,
                               plot_type=module_config.plot_type,
                               cmap=module_config.cmap,
                               plot_step=module_config.plot_step,
                               log_file=log_file,
                               verbose=verbose,
                               terminal=terminal)

                elif fpath.endswith('.h5'):
                    log_file = os.path.join(
                        config.logdir,
                        os.path.basename(fpath).replace('.h5', '.log'))
                    ctx.invoke(reV_h5,
                               h5_file=fpath,
                               out_dir=config.dirout,
                               sub_dir=module_config.sub_dir,
                               dsets=module_config.dsets,
                               group=module_config.group,
                               process_size=module_config.process_size,
                               max_workers=module_config.max_workers,
                               plot_type=module_config.plot_type,
                               cmap=module_config.cmap,
                               log_file=log_file,
                               verbose=verbose,
                               terminal=terminal)

                elif fpath.endswith('.csv'):
                    log_file = os.path.join(
                        config.logdir,
                        os.path.basename(fpath).replace('.csv', '.log'))
                    ctx.invoke(supply_curve,
                               sc_table=fpath,
                               out_dir=config.dirout,
                               sub_dir=module_config.sub_dir,
                               columns=module_config.columns,
                               plot_type=module_config.plot_type,
                               cmap=module_config.cmap,
                               lcoe=module_config.lcoe,
                               log_file=log_file,
                               verbose=verbose,
                               terminal=terminal)
                else:
                    msg = ("Cannot run QA/QC for {}: 'fpath' must be a '*.h5' "
                           "or '*.csv' reV output file, but {} was given!"
                           .format(module, fpath))
                    logger.error(msg)
                    raise ValueError(msg)

    elif config.execution_control.option in ('eagle', 'slurm'):
        launch_slurm(config, verbose)


def get_h5_cmd(name, h5_file, out_dir, sub_dir, dsets, group, process_size,
               max_workers, plot_type, cmap, log_file, verbose, terminal):
    """Build CLI call for reV_h5."""

    args = ('-h5 {h5_file} '
            '-o {out_dir} '
            '-sd {sub_dir} '
            '-ds {dsets} '
            '-grp {group} '
            '-ps {process_size} '
            '-w {max_workers} '
            '-plt {plot_type} '
            '-cmap {cmap} '
            '-log {log_file} '
            )

    args = args.format(h5_file=SLURM.s(h5_file),
                       out_dir=SLURM.s(out_dir),
                       sub_dir=SLURM.s(sub_dir),
                       dsets=SLURM.s(dsets),
                       group=SLURM.s(group),
                       process_size=SLURM.s(process_size),
                       max_workers=SLURM.s(max_workers),
                       plot_type=SLURM.s(plot_type),
                       cmap=SLURM.s(cmap),
                       log_file=SLURM.s(log_file),
                       )

    if verbose:
        args += '-v '

    if terminal:
        args += '-t '

    cmd = ('python -m reV.qa_qc.cli_qa_qc -n {} rev-h5 {}'
           .format(SLURM.s(name), args))

    return cmd


def get_sc_cmd(name, sc_table, out_dir, sub_dir, columns, plot_type, cmap,
               lcoe, log_file, verbose, terminal):
    """Build CLI call for supply_curve."""

    args = ('-sct {sc_table} '
            '-o {out_dir} '
            '-sd {sub_dir} '
            '-cols {columns} '
            '-plt {plot_type} '
            '-cmap {cmap} '
            '-lcoe {lcoe} '
            '-log {log_file} '
            )

    args = args.format(sc_table=SLURM.s(sc_table),
                       out_dir=SLURM.s(out_dir),
                       sub_dir=SLURM.s(sub_dir),
                       columns=SLURM.s(columns),
                       plot_type=SLURM.s(plot_type),
                       cmap=SLURM.s(cmap),
                       lcoe=SLURM.s(lcoe),
                       log_file=SLURM.s(log_file),
                       )

    if verbose:
        args += '-v '

    if terminal:
        args += '-t '

    cmd = ('python -m reV.qa_qc.cli_qa_qc -n {} supply-curve {}'
           .format(SLURM.s(name), args))

    return cmd


def get_excl_cmd(name, excl_fpath, out_dir, sub_dir, excl_dict,
                 area_filter_kernel, min_area, plot_type, cmap, plot_step,
                 log_file, verbose, terminal):
    """Build CLI call for exclusions."""

    args = ('-excl {excl_fpath} '
            '-o {out_dir} '
            '-sd {sub_dir} '
            '-exd {excl_dict} '
            '-afk {area_filter_kernel} '
            '-ma {min_area} '
            '-plt {plot_type} '
            '-cmap {cmap} '
            '-step {plot_step} '
            '-log {log_file} '
            )

    args = args.format(excl_fpath=SLURM.s(excl_fpath),
                       out_dir=SLURM.s(out_dir),
                       sub_dir=SLURM.s(sub_dir),
                       excl_dict=SLURM.s(excl_dict),
                       area_filter_kernel=SLURM.s(area_filter_kernel),
                       min_area=SLURM.s(min_area),
                       plot_type=SLURM.s(plot_type),
                       cmap=SLURM.s(cmap),
                       plot_step=SLURM.s(plot_step),
                       log_file=SLURM.s(log_file),
                       )

    if verbose:
        args += '-v '

    if terminal:
        args += '-t '

    cmd = ('python -m reV.qa_qc.cli_qa_qc -n {} exclusions {}'
           .format(SLURM.s(name), args))

    return cmd


def launch_slurm(config, verbose):
    """
    Launch slurm QA/QC job

    Parameters
    ----------
    config : dict
        'reV QA/QC configuration dictionary'
    """

    out_dir = config.dirout
    log_file = os.path.join(config.logdir, config.name + '.log')
    stdout_path = os.path.join(config.logdir, 'stdout/')

    node_cmd = []
    terminal = False
    for i, module in enumerate(config.module_names):
        module_config = config.get_module_inputs(module)
        fpaths = module_config.fpath

        if isinstance(fpaths, (str, type(None))):
            fpaths = [fpaths]

        for j, fpath in enumerate(fpaths):
            if (i == len(config.module_names) - 1) and (j == len(fpaths) - 1):
                terminal = True
            if module.lower() == 'exclusions':
                node_cmd.append(get_excl_cmd(config.name,
                                             module_config.excl_fpath,
                                             out_dir,
                                             module_config.sub_dir,
                                             module_config.excl_dict,
                                             module_config.area_filter_kernel,
                                             module_config.min_area,
                                             module_config.plot_type,
                                             module_config.cmap,
                                             module_config.plot_step,
                                             log_file,
                                             verbose,
                                             terminal))
            elif fpath.endswith('.h5'):
                node_cmd.append(get_h5_cmd(config.name, fpath, out_dir,
                                           module_config.sub_dir,
                                           module_config.dsets,
                                           module_config.group,
                                           module_config.process_size,
                                           module_config.max_workers,
                                           module_config.plot_type,
                                           module_config.cmap,
                                           log_file,
                                           verbose,
                                           terminal))
            elif fpath.endswith('.csv'):
                node_cmd.append(get_sc_cmd(config.name, fpath, out_dir,
                                           module_config.sub_dir,
                                           module_config.columns,
                                           module_config.plot_type,
                                           module_config.cmap,
                                           module_config.lcoe,
                                           log_file,
                                           verbose,
                                           terminal))
            else:
                msg = ("Cannot run QA/QC for {}: 'fpath' must be a '*.h5' "
                       "or '*.csv' reV output file, but {} was given!"
                       .format(module, fpath))
                logger.error(msg)
                raise ValueError(msg)

    status = Status.retrieve_job_status(out_dir, 'qa-qc', config.name)
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'
               .format(config.name, out_dir))
    else:
        node_cmd = '\n'.join(node_cmd)
        logger.info('Running reV QA-QC on SLURM with '
                    'node name "{}"'.format(config.name))
        slurm = SLURM(node_cmd, alloc=config.execution_control.allocation,
                      memory=config.execution_control.memory,
                      feature=config.execution_control.feature,
                      walltime=config.execution_control.walltime,
                      conda_env=config.execution_control.conda_env,
                      module=config.execution_control.module,
                      stdout_path=stdout_path)
        if slurm.id:
            msg = ('Kicked off reV QA-QC job "{}" '
                   '(SLURM jobid #{}).'
                   .format(config.name, slurm.id))
            Status.add_job(
                out_dir, 'qa-qc', config.name, replace=True,
                job_attrs={'job_id': slurm.id, 'hardware': 'eagle',
                           'dirout': out_dir})
        else:
            msg = ('Was unable to kick off reV QA-QC job "{}". '
                   'Please see the stdout error messages'
                   .format(config.name))

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV QA/QC CLI')
        raise
