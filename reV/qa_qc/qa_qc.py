# -*- coding: utf-8 -*-
"""
reV quality assurance and control classes
"""
import logging
import os
import pandas as pd

from reV.qa_qc.summary import Summarize, SummaryPlots

logger = logging.getLogger(__name__)


class QaQc:
    """
    reV QA/QC
    """
    def __init__(self, h5_file, out_dir, dsets=None):
        """
        Parameters
        ----------
        h5_file : str
            Path to .h5 file to run QA/QC on
        out_dir : str
            Directory path to save summary tables and plots too
        dsets : str | list, optional
            Datasets to summarize, by default None
        """
        logger.info('QAQC initializing on: {}'.format(h5_file))
        self._h5_file = h5_file
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self._out_dir = out_dir
        self._dsets = dsets

    @staticmethod
    def _scatter_plot(summary_csv, out_root, plot_type='plotly',
                      cmap='viridis', **kwargs):
        """
        Create scatter plot for all summary stats in summary table and save to
        out_dir

        Parameters
        ----------
        summary_csv : str
            Path to .csv file containing summary table
        out_root : str
            Output directory to save plots to
        plot_type : str, optional
            plot_type of plot to create 'plot' or 'plotly', by default 'plotly'
        cmap : str, optional
            Colormap name, by default 'viridis'
        kwargs : dict
            Additional plotting kwargs
        """
        out_dir = os.path.join(out_root,
                               os.path.basename(summary_csv).rstrip('.csv'))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        SummaryPlots.scatter_all(summary_csv, out_dir, plot_type=plot_type,
                                 cmap=cmap, **kwargs)

    @staticmethod
    def _scatter_plots(out_dir, plot_type='plotly', cmap='viridis', **kwargs):
        """
        Create scatter plot for all compatible summary .csv files

        Parameters
        ----------
        out_dir : str
            Directory path to save summary tables and plots too
        plot_type : str, optional
            plot_type of plot to create 'plot' or 'plotly', by default 'plotly'
        cmap : str, optional
            Colormap name, by default 'viridis'
        kwargs : dict
            Additional plotting kwargs
        """
        for file in os.listdir(out_dir):
            if file.endswith('.csv'):
                summary_csv = os.path.join(out_dir, file)
                summary = pd.read_csv(summary_csv)
                if ('gid' in summary and 'latitude' in summary
                        and 'longitude' in summary):
                    QaQc._scatter_plot(summary_csv, out_dir,
                                       plot_type=plot_type, cmap=cmap,
                                       **kwargs)

    def summarize(self, group=None, process_size=None, max_workers=None):
        """
        Summarize all datasets in h5_file and dump to out_dir

        Parameters
        ----------
        group : str, optional
            Group within h5_file to summarize datasets for, by default None
        process_size : int, optional
            Number of sites to process at a time, by default None
        max_workers : int, optional
            Number of workers to use when summarizing 2D datasets,
            by default None
        """
        Summarize.run(self._h5_file, self._out_dir, group=group,
                      dsets=self._dsets, process_size=process_size,
                      max_workers=max_workers)

    def scatter_plots(self, plot_type='plotly', cmap='viridis', **kwargs):
        """
        Create scatter plot for all compatible summary .csv files

        Parameters
        ----------
        plot_type : str, optional
            plot_type of plot to create 'plot' or 'plotly', by default 'plotly'
        cmap : str, optional
            Colormap name, by default 'viridis'
        kwargs : dict
            Additional plotting kwargs
        """
        self._scatter_plots(self._out_dir, plot_type=plot_type, cmap=cmap,
                            **kwargs)

    @classmethod
    def run(cls, h5_file, out_dir, dsets=None, group=None, process_size=None,
            max_workers=None, plot_type='plotly', cmap='viridis', **kwargs):
        """
        Run QA/QC by computing summary stats from dsets in h5_file and
        plotting scatters plots of compatible summary stats

        Parameters
        ----------
        h5_file : str
            Path to .h5 file to run QA/QC on
        out_dir : str
            Directory path to save summary tables and plots too
        dsets : str | list, optional
            Datasets to summarize, by default None
        group : str, optional
            Group within h5_file to summarize datasets for, by default None
        process_size : int, optional
            Number of sites to process at a time, by default None
        max_workers : int, optional
            Number of workers to use when summarizing 2D datasets,
            by default None
        plot_type : str, optional
            plot_type of plot to create 'plot' or 'plotly', by default 'plotly'
        cmap : str, optional
            Colormap name, by default 'viridis'
        kwargs : dict
            Additional plotting kwargs
        """
        try:
            qa_qc = cls(h5_file, out_dir, dsets=dsets)
            qa_qc.summarize(group=group, process_size=process_size,
                            max_workers=max_workers)
            qa_qc.scatter_plots(plot_type=plot_type, cmap=cmap, **kwargs)
        except Exception as e:
            logger.exception('QAQC failed on file: {}. Received exception:\n{}'
                             .format(os.path.basename(h5_file), e))
            raise e
        else:
            logger.info('Finished QAQC on file: {} output directory: {}'
                        .format(os.path.basename(h5_file), out_dir))

    @classmethod
    def supply_curve(cls, sc_table, out_dir, columns=None, lcoe='mean_lcoe',
                     plot_type='plotly', cmap='viridis', sc_plot_kwargs=None,
                     scatter_plot_kwargs=None):
        """
        Plot supply curve

        Parameters
        ----------
        sc_table : str
            Path to .csv file containing supply curve table
        out_dir : str
            Directory path to save summary tables and plots too
        columns : str | list, optional
            Column(s) to summarize, if None summarize all numeric columns,
            by default None
        lcoe : str, optional
            LCOE value to plot, by default 'mean_lcoe'
        plot_type : str, optional
            plot_type of plot to create 'plot' or 'plotly', by default 'plotly'
        cmap : str, optional
            Colormap name, by default 'viridis'
        sc_plot_kwargs : dict, optional
            Kwargs for supply curve plot, by default None
        scatter_plot_kwargs : dict
            Kwargs for scatter plot, by default None
        """
        if sc_plot_kwargs is None:
            sc_plot_kwargs = {}

        if scatter_plot_kwargs is None:
            scatter_plot_kwargs = {}

        try:
            Summarize.supply_curve(sc_table, out_dir, columns=columns)
            SummaryPlots.supply_curve(sc_table, out_dir, plot_type=plot_type,
                                      lcoe=lcoe, **sc_plot_kwargs)
            QaQc._scatter_plot(sc_table, out_dir, plot_type=plot_type,
                               cmap=cmap, **scatter_plot_kwargs)
        except Exception as e:
            logger.exception('QAQC failed on file: {}. Received exception:\n{}'
                             .format(os.path.basename(sc_table), e))
            raise e
        else:
            logger.info('Finished QAQC on file: {} output directory: {}'
                        .format(os.path.basename(sc_table), out_dir))
