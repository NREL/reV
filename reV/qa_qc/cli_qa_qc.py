# -*- coding: utf-8 -*-
"""
QA/QC CLI entry points.
"""
import click
from rex.utilities.cli_dtypes import STR, STRLIST, INT
from rex.utilities.loggers import init_logger

from reV.qa_qc.qa_qc import QaQc
from reV.qa_qc.summary import Summarize, SummaryPlots


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


@main.group(chain=True)
@click.option('--out_dir', '-o', type=click.Path(), requried=True,
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
    Summarize.run(h5_file, ctx.obj['OUT_DIR'], group=group, dsets=dsets,
                  process_size=process_size, max_workers=max_workers)


@summarize.command()
@click.option('--plot_type', '-plt', default='plot',
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
    QaQc._scatter_plots(ctx.obj['OUT_DIR'], plot_type, cmap)


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
    Summarize.supply_curve(sc_table, ctx.obj['OUT_DIR'], columns=columns)


@summarize.command()
@click.option('--sc_table', '-sct', type=click.Path(exists=True), default=None,
              help=("Path to .csv containing Supply Curve table, can be "
                    "supplied in 'supply-curve-table'"))
@click.option('--plot_type', '-plt', default='plot',
              type=click.Choice(['plot', 'plotly'], case_sensitive=False),
              help=(" plot_type of plot to create 'plot' or 'plotly', by "
                    "default 'plot'"))
@click.option('--lcoe', '-lcoe', type=STR, default='total_lcoe',
              help="LCOE value to plot, by default 'total_lcoe'")
@click.pass_context
def supply_curve_plot(ctx, sc_table, plot_type, lcoe):
    """
    Plot Supply Curve (cumulative capacity vs LCOE)
    """
    if sc_table is None:
        sc_table = ctx.obj['SC_TABLE']

    SummaryPlots.supply_curve(sc_table, ctx.obj['OUT_DIR'],
                              plot_type=plot_type, lcoe=lcoe)


@main.command()
@click.option('--h5_file', '-h5', type=click.Path(exists=True), required=True,
              help='Path to .h5 file to summarize')
@click.option('--out_dir', '-o', type=click.Path(), requried=True,
              help="Directory path to save summary tables and plots too")
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
@click.option('--plot_type', '-plt', default='plot',
              type=click.Choice(['plot', 'plotly'], case_sensitive=False),
              help=(" plot_type of plot to create 'plot' or 'plotly', by "
                    "default 'plot'"))
@click.option('--cmap', '-cmap', type=str, default='viridis',
              help="Colormap name, by default 'viridis'")
@click.option('--log_file', '-log', type=click.Path(), default=None,
              help='File to log to, by default None')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def reV_h5(ctx, h5_file, out_dir, dsets, group, process_size, max_workers,
           plot_type, cmap, log_file, verbose):
    """
    Summarize and plot data for reV h5_file
    """
    if any([verbose, ctx.obj['VERBOSE']]):
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    init_logger('reV.qa_qc', log_file=log_file, log_level=log_level)

    QaQc.run(h5_file, out_dir, dsets=dsets, group=group,
             process_size=process_size, max_workers=max_workers,
             plot_type=plot_type, cmap=cmap)


@main.command()
@click.option('--sc_table', '-sct', type=click.Path(exists=True),
              required=True, help='Path to .csv containing Supply Curve table')
@click.option('--out_dir', '-o', type=click.Path(), requried=True,
              help="Directory path to save summary tables and plots too")
@click.option('--columns', '-cols', type=STRLIST, default=None,
              help=('Column(s) to summarize, if None summarize all numeric '
                    'columns, by default None'))
@click.option('--plot_type', '-plt', default='plot',
              type=click.Choice(['plot', 'plotly'], case_sensitive=False),
              help=(" plot_type of plot to create 'plot' or 'plotly', by "
                    "default 'plot'"))
@click.option('--lcoe', '-lcoe', type=STR, default='total_lcoe',
              help="LCOE value to plot, by default 'total_lcoe'")
@click.option('--log_file', '-log', type=click.Path(), default=None,
              help='File to log to, by default None')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def supply_curve(ctx, sc_table, out_dir, columns, plot_type, lcoe, log_file,
                 verbose):
    """
    Summarize and plot reV Supply Curve dataß
    """
    if any([verbose, ctx.obj['VERBOSE']]):
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    init_logger('reV.qa_qc', log_file=log_file, log_level=log_level)

    QaQc.supply_curve(sc_table, out_dir, columns=columns, lcoe=lcoe,
                      plot_type=plot_type)


if __name__ == '__main__':
    main(obj={})
