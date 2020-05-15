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
        self._h5_file = h5_file
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self._out_dir = out_dir
        self._dsets = dsets

    @staticmethod
    def _plot_summary(summary_csv, out_root, plot_type='plot', cmap='viridis',
                      **kwargs):
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
            plot_type of plot to create 'plot' or 'plotly', by default 'plot'
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

    def scatter_plots(self, plot_type='plot', cmap='viridis', **kwargs):
        """
        Create scatter plot for all compatible summary .csv files

        Parameters
        ----------
        plot_type : str, optional
            plot_type of plot to create 'plot' or 'plotly', by default 'plot'
        cmap : str, optional
            Colormap name, by default 'viridis'
        kwargs : dict
            Additional plotting kwargs
        """
        for file in os.listdir(self._out_dir):
            if file.endswith('.csv'):
                summary_csv = os.path.join(self._out_dir, file)
                summary = pd.read_csv(summary_csv)
                if 'gid' in summary:
                    self._plot_summary(summary_csv, self._out_dir,
                                       plot_type=plot_type, cmap=cmap,
                                       **kwargs)

    @classmethod
    def run(cls, h5_file, out_dir, dsets=None, group=None, process_size=None,
            max_workers=None, plot_type='plot', cmap='viridis', **kwargs):
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
            plot_type of plot to create 'plot' or 'plotly', by default 'plot'
        cmap : str, optional
            Colormap name, by default 'viridis'
        kwargs : dict
            Additional plotting kwargs
        """
        qa_qc = cls(h5_file, out_dir, dsets=dsets)
        qa_qc.summarize(group=group, process_size=process_size,
                        max_workers=max_workers)
        qa_qc.scatter_plots(plot_type=plot_type, cmap=cmap, **kwargs)

    @classmethod
    def supply_curve(cls, sc_table, out_dir, lcoe='total_lcoe',
                     plot_type='plot', **kwargs):
        """
        Plot supply curve

        Parameters
        ----------
        sc_table : str
            Path to .csv file containing supply curve table
        out_dir : str
            Directory path to save summary tables and plots too
        lcoe : str, optional
            LCOE value to plot, by default 'total_lcoe'
        plot_type : str, optional
            plot_type of plot to create 'plot' or 'plotly', by default 'plot'
        kwargs : dict
            Additional plotting kwargs
        """
        SummaryPlots.supply_curve(sc_table, out_dir, plot_type=plot_type,
                                  lcoe=lcoe, **kwargs)
