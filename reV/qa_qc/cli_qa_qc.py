# -*- coding: utf-8 -*-
"""
QA/QC CLI utility functions.
"""
import logging
import os

import click
import numpy as np
from gaps.cli import CLICommandFromFunction, as_click_command
from rex.utilities.cli_dtypes import INT, STR, STRLIST
from rex.utilities.loggers import init_logger

from reV import __version__
from reV.qa_qc.qa_qc import QaQc, QaQcModule
from reV.qa_qc.summary import (
    ExclusionsMask,
    SummarizeH5,
    SummarizeSupplyCurve,
    SupplyCurvePlot,
)
from reV.utilities import ModuleName, SupplyCurveField

logger = logging.getLogger(__name__)


def cli_qa_qc(modules, out_dir, max_workers=None):
    """Run QA/QC on reV outputs

    ``reV`` QA/QC performs quality assurance checks on ``reV`` output
    data. Users can specify the type of QA/QC that should be applied
    to each ``reV`` module.

    Parameters
    ----------
    modules : dict
        Dictionary of modules to QA/QC. Keys should be the names of the
        modules to QA/QC. The values are dictionaries that represent the
        config for the respective QA/QC step. Allowed config keys for
        QA/QC are the "property" attributes of
        :class:`~reV.qa_qc.qa_qc.QaQcModule`.
    out_dir : str
        Path to output directory.
    max_workers : int, optional
        Max number of workers to run for QA/QA. If ``None``, uses all
        CPU cores. By default, ``None``.

    Raises
    ------
    ValueError
        If fpath is not an H5 or CSV file.
    """
    for module, mcf in modules.items():
        module_config = QaQcModule(module, mcf, out_dir)

        qa_dir = out_dir
        if module_config.sub_dir is not None:
            qa_dir = os.path.join(out_dir, module_config.sub_dir)

        if module.lower() == 'exclusions':
            QaQc.exclusions_mask(module_config.fpath, qa_dir,
                                 layers_dict=module_config.excl_dict,
                                 min_area=module_config.min_area,
                                 kernel=module_config.area_filter_kernel,
                                 plot_type=module_config.plot_type,
                                 cmap=module_config.cmap,
                                 plot_step=module_config.plot_step)

        elif module_config.fpath.endswith('.h5'):
            QaQc.h5(module_config.fpath, qa_dir, dsets=module_config.dsets,
                    group=module_config.group,
                    process_size=module_config.process_size,
                    max_workers=max_workers,
                    plot_type=module_config.plot_type, cmap=module_config.cmap)

        elif module_config.fpath.endswith('.csv'):
            QaQc.supply_curve(module_config.fpath, qa_dir,
                              columns=module_config.columns,
                              lcoe=module_config.lcoe,
                              plot_type=module_config.plot_type,
                              cmap=module_config.cmap)
        else:
            msg = ("Cannot run QA/QC for {}: 'fpath' must be a '*.h5' "
                   "or '*.csv' reV output file, but {} was given!"
                   .format(module, module_config.fpath))
            logger.error(msg)
            raise ValueError(msg)


qa_qc_command = CLICommandFromFunction(cli_qa_qc, name=str(ModuleName.QA_QC),
                                       split_keys=None)
main = as_click_command(qa_qc_command)


@click.group()
@click.version_option(version=__version__)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def qa_qc_extra(ctx, verbose):
    """Execute extra QA/QC utility"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose


@qa_qc_extra.group(chain=True)
@click.option('--out_dir', '-o', type=click.Path(), required=True,
              help="Directory path to save summary tables and plots too")
@click.option('--log_file', '-log', type=click.Path(), default=None,
              show_default=True,
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

    init_logger('reV', log_file=log_file, log_level=log_level)


@summarize.command()
@click.option('--h5_file', '-h5', type=click.Path(exists=True), required=True,
              help='Path to .h5 file to summarize')
@click.option('--dsets', '-ds', type=STRLIST, default=None,
              show_default=True,
              help='Datasets to summarize, by default None')
@click.option('--group', '-grp', type=STR, default=None,
              show_default=True,
              help=('Group within h5_file to summarize datasets for, by '
                    'default None'))
@click.option('--process_size', '-ps', type=INT, default=None,
              show_default=True,
              help='Number of sites to process at a time, by default None')
@click.option('--max_workers', '-w', type=INT, default=None,
              show_default=True,
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
              show_default=True,
              help=(" plot_type of plot to create 'plot' or 'plotly', by "
                    "default 'plot'"))
@click.option('--cmap', '-cmap', type=str, default='viridis',
              show_default=True,
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
              show_default=True,
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
              show_default=True,
              help=("Path to .csv containing Supply Curve table, can be "
                    "supplied in 'supply-curve-table'"))
@click.option('--plot_type', '-plt', default='plotly',
              type=click.Choice(['plot', 'plotly'], case_sensitive=False),
              show_default=True,
              help=(" plot_type of plot to create 'plot' or 'plotly', by "
                    "default 'plot'"))
@click.option('--lcoe', '-lcoe', type=STR, default=SupplyCurveField.MEAN_LCOE,
              help="LCOE value to plot, by default %(default)s")
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
              show_default=True,
              help=(" plot_type of plot to create 'plot' or 'plotly', by "
                    "default 'plot'"))
@click.option('--cmap', '-cmap', type=str, default='viridis',
              show_default=True,
              help="Colormap name, by default 'viridis'")
@click.option('--plot_step', '-step', type=int, default=100,
              show_default=True,
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


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV QA/QC CLI.')
        raise
