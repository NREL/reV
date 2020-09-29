# -*- coding: utf-8 -*-
"""
reV quality assurance and control classes
"""
import logging
import numpy as np
import os
import pandas as pd

from reV.qa_qc.summary import (SummarizeH5, SummarizeSupplyCurve, SummaryPlots,
                               SupplyCurvePlot, ExclusionsMask)
from reV.supply_curve.exclusions import ExclusionMaskFromDict

logger = logging.getLogger(__name__)


class QaQc:
    """
    reV QA/QC
    """
    def __init__(self, out_dir):
        """
        Parameters
        ----------
        out_dir : str
            Directory path to save summary data and plots too
        """
        logger.info('QA/QC results to be saved to: {}'.format(out_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self._out_dir = out_dir

    @property
    def out_dir(self):
        """
        Output directory

        Returns
        -------
        str
        """
        return self._out_dir

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

    def create_scatter_plots(self, plot_type='plotly', cmap='viridis',
                             **kwargs):
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
        for file in os.listdir(self.out_dir):
            if file.endswith('.csv'):
                summary_csv = os.path.join(self.out_dir, file)
                summary = pd.read_csv(summary_csv)
                if ('gid' in summary and 'latitude' in summary
                        and 'longitude' in summary):
                    QaQc._scatter_plot(summary_csv, self.out_dir,
                                       plot_type=plot_type, cmap=cmap,
                                       **kwargs)

    @classmethod
    def h5(cls, h5_file, out_dir, dsets=None, group=None, process_size=None,
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
            qa_qc = cls(out_dir)
            SummarizeH5.run(h5_file, out_dir, group=group,
                            dsets=dsets, process_size=process_size,
                            max_workers=max_workers)
            qa_qc.create_scatter_plots(plot_type=plot_type, cmap=cmap,
                                       **kwargs)
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
            qa_qc = cls(out_dir)
            SummarizeSupplyCurve.run(sc_table, out_dir, columns=columns)
            SupplyCurvePlot.plot(sc_table, out_dir, plot_type=plot_type,
                                 lcoe=lcoe, **sc_plot_kwargs)
            qa_qc._scatter_plot(sc_table, out_dir, plot_type=plot_type,
                                cmap=cmap, **scatter_plot_kwargs)
        except Exception as e:
            logger.exception('QAQC failed on file: {}. Received exception:\n{}'
                             .format(os.path.basename(sc_table), e))
            raise e
        else:
            logger.info('Finished QAQC on file: {} output directory: {}'
                        .format(os.path.basename(sc_table), out_dir))

    @classmethod
    def exclusions_mask(cls, excl_h5, out_dir, layers_dict=None, min_area=None,
                        kernel='queen', hsds=False, plot_type='plotly',
                        cmap='viridis', plot_step=100, **kwargs):
        """
        Create inclusion mask from given layers dictionary, dump to disk and
        plot

        Parameters
        ----------
        excl_h5 : str
            Path to exclusions .h5 file
        layers_dict : dict | NoneType
            Dictionary of LayerMask arugments {layer: {kwarg: value}}
        min_area : float | NoneType
            Minimum required contiguous area in sq-km
        kernel : str
            Contiguous filter method to use on final exclusions
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        plot_type : str, optional
            plot_type of plot to create 'plot' or 'plotly', by default 'plotly'
        cmap : str, optional
            Colormap name, by default 'viridis'
        plot_step : int
            Step between points to plot
        kwargs : dict
            Additional plotting kwargs
        """
        try:
            cls(out_dir)
            excl_mask = ExclusionMaskFromDict.run(excl_h5,
                                                  layers_dict=layers_dict,
                                                  min_area=min_area,
                                                  kernel=kernel,
                                                  hsds=hsds)
            excl_mask = np.round(excl_mask * 100).astype('uint8')

            out_file = os.path.basename(excl_h5).replace('.h5', '_mask.npy')
            out_file = os.path.join(out_dir, out_file)
            np.save(out_file, excl_mask)

            ExclusionsMask.plot(excl_mask, out_dir, plot_type=plot_type,
                                cmap=cmap, plot_step=plot_step, **kwargs)
        except Exception as e:
            logger.exception('QAQC failed on file: {}. Received exception:\n{}'
                             .format(os.path.basename(excl_h5), e))
            raise e
        else:
            logger.info('Finished QAQC on file: {} output directory: {}'
                        .format(os.path.basename(excl_h5), out_dir))
