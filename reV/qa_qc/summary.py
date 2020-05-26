# -*- coding: utf-8 -*-
"""
Compute and plot summary data
"""
import logging
import numpy as np
import os
import pandas as pd
import plotting as mplt
import plotly.express as px

from rex import Resource
from rex.utilities.execution import SpawnProcessPool

logger = logging.getLogger(__name__)


class SummarizeH5:
    """
    reV Summary data for QA/QC
    """
    def __init__(self, h5_file, group=None):
        """
        Parameters
        ----------
        h5_file : str
            Path to .h5 file to summarize data from
        group : str, optional
            Group within h5_file to summarize datasets for, by default None
        """
        logger.info('QAQC Summarize initializing on: {}'.format(h5_file))
        self._h5_file = h5_file
        self._group = group

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self.h5_file)

        return msg

    @property
    def h5_file(self):
        """
        .h5 file path

        Returns
        -------
        str
        """
        return self._h5_file

    @staticmethod
    def _compute_sites_summary(h5_file, ds_name, sites=None, group=None):
        """
        Compute summary stats for given sites of given dataset

        Parameters
        ----------
        h5_file : str
            Path to .h5 file to summarize data from
        ds_name : str
            Dataset name of interest
        sites : list | slice, optional
            sites of interest, by default None
        group : str, optional
            Group within h5_file to summarize datasets for, by default None

        Returns
        -------
        sites_summary : pandas.DataFrame
            Summary stats for given sites / dataset
        """
        if sites is None:
            sites = slice(None)

        with Resource(h5_file, group=group) as f:
            sites_meta = f['meta', sites]
            sites_data = f[ds_name, :, sites]

        sites_summary = pd.DataFrame(sites_data, columns=sites_meta.index)
        sites_summary = sites_summary.describe().T.drop(columns=['count'])
        sites_summary['sum'] = sites_data.sum(axis=0)

        return sites_summary

    @staticmethod
    def _compute_ds_summary(h5_file, ds_name, group=None):
        """
        Compute summary statistics for given dataset (assumed to be a vector)

        Parameters
        ----------
        h5_file : str
            Path to .h5 file to summarize data from
        ds_name : str
            Dataset name of interest
        group : str, optional
            Group within h5_file to summarize datasets for, by default None

        Returns
        -------
        ds_summary : pandas.DataFrame
            Summary statistics for dataset
        """
        with Resource(h5_file, group=group) as f:
            ds_data = f[ds_name, :]

        ds_summary = pd.DataFrame(ds_data, columns=[ds_name])
        ds_summary = ds_summary.describe().drop(['count'])
        ds_summary.at['sum'] = ds_data.sum()

        return ds_summary

    def summarize_dset(self, ds_name, process_size=None, max_workers=None,
                       out_path=None):
        """
        Compute dataset summary. If dataset is 2D compute temporal statistics
        for each site

        Parameters
        ----------
        ds_name : str
            Dataset name of interest
        process_size : int, optional
            Number of sites to process at a time, by default None
        max_workers : int, optional
            Number of workers to use in parallel, if 1 run in serial,
            if None use all available cores, by default None
        out_path : str
            File path to save summary to

        Returns
        -------
        summary : pandas.DataFrame
            Summary summary for dataset
        """
        with Resource(self.h5_file, group=self._group) as f:
            ds_shape, _, ds_chunks = f.get_dset_properties(ds_name)

        if len(ds_shape) > 1:
            sites = np.arange(ds_shape[1])
            if max_workers != 1:
                if process_size is None and ds_chunks is not None:
                    process_size = ds_chunks[1]
                if process_size is None:
                    process_size = ds_shape[-1]

                sites = \
                    np.array_split(sites,
                                   int(np.ceil(len(sites) / process_size)))
                loggers = [__name__]
                with SpawnProcessPool(max_workers=max_workers,
                                      loggers=loggers) as ex:
                    futures = []
                    for site_slice in sites:
                        futures.append(ex.submit(
                            self._compute_sites_summary,
                            self.h5_file, ds_name, sites=site_slice,
                            group=self._group))

                    summary = [future.result() for future in futures]

                summary = pd.concat(summary)
            else:
                if process_size is None:
                    summary = self._compute_sites_summary(self.h5_file,
                                                          ds_name,
                                                          sites=sites,
                                                          group=self._group)
                else:
                    sites = np.array_split(
                        sites, int(np.ceil(len(sites) / process_size)))

                    summary = []
                    for site_slice in sites:
                        summary.append(self._compute_sites_summary(
                            self.h5_file, ds_name,
                            sites=site_slice,
                            group=self._group))

                    summary = pd.concat(summary)

            summary.index.name = 'gid'

        else:
            summary = self._compute_ds_summary(self.h5_file, ds_name,
                                               group=self._group)

        if out_path is not None:
            summary.to_csv(out_path)

        return summary

    def summarize_means(self, out_path=None):
        """
        Add means datasets to meta data

        Parameters
        ----------
        out_path : str, optional
            Path to .csv file to save update meta data to, by default None

        Returns
        -------
        meta : pandas.DataFrame
            Meta data with means datasets added
        """
        with Resource(self.h5_file, group=self._group) as f:
            meta = f.meta
            if 'gid' not in meta:
                meta.index.name = 'gid'
                meta = meta.reset_index()

            for ds_name in f.datasets:
                shape, dtype, _ = f.get_dset_properties(ds_name)
                if len(shape) == 1 and np.issubdtype(dtype, np.number):
                    meta[ds_name] = f[ds_name]

        if out_path is not None:
            meta.to_csv(out_path, index=False)

        return meta

    @classmethod
    def run(cls, h5_file, out_dir, group=None, dsets=None,
            process_size=None, max_workers=None):
        """
        Summarize all datasets in h5_file and dump to out_dir

        Parameters
        ----------
        h5_file : str
            Path to .h5 file to summarize data from
        out_dir : str
            Directory to dump summary .csv files to
        group : str, optional
            Group within h5_file to summarize datasets for, by default None
        dsets : str | list, optional
            Datasets to summarize, by default None
        process_size : int, optional
            Number of sites to process at a time, by default None
        max_workers : int, optional
            Number of workers to use when summarizing 2D datasets,
            by default None
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if dsets is None:
            with Resource(h5_file, group=group) as f:
                dsets = [dset for dset in f.datasets
                         if dset not in ['meta', 'time_index']]
        elif isinstance(dsets, str):
            dsets = [dsets]

        summary = cls(h5_file)
        for ds_name in dsets:
            out_path = os.path.join(out_dir,
                                    "{}_summary.csv".format(ds_name))
            summary.summarize_dset(ds_name, process_size=process_size,
                                   max_workers=max_workers, out_path=out_path)

        out_path = os.path.basename(h5_file).replace('.h5', '_summary.csv')
        out_path = os.path.join(out_dir, out_path)
        summary.summarize_means(out_path=out_path)


class SummarizeSupplyCurve:
    """
    Summarize Supply Curve table
    """
    def __init__(self, sc_table):
        self._sc_table = self._parse_summary(sc_table)

    def __repr__(self):
        msg = "{}".format(self.__class__.__name__)

        return msg

    @property
    def sc_table(self):
        """
        Supply Curve table

        Returns
        -------
        pd.DataFrame
        """
        return self._sc_table

    @staticmethod
    def _parse_summary(summary):
        """
        Extract summary statistics

        Parameters
        ----------
        summary : str | pd.DataFrame
            Path to .csv or .json or DataFrame to parse

        Returns
        -------
        summary : pandas.DataFrame
            DataFrame of summary statistics
        """
        if isinstance(summary, str):
            if summary.endswith('.csv'):
                summary = pd.read_csv(summary)
            elif summary.endswith('.json'):
                summary = pd.read_json(summary)
            else:
                raise ValueError('Cannot parse {}'.format(summary))

        elif not isinstance(summary, pd.DataFrame):
            raise ValueError("summary must be a .csv, .json, or "
                             "a pandas DataFrame")

        return summary

    def supply_curve_summary(self, columns=None, out_path=None):
        """
        Summarize Supply Curve Table

        Parameters
        ----------
        sc_table : str | pandas.DataFrame
            Supply curve table or .csv containing table
        columns : str | list, optional
            Column(s) to summarize, if None summarize all numeric columns,
            by default None
        out_path : str, optional
            Path to .csv to save summary to, by default None

        Returns
        -------
        sc_summary : pandas.DataFrame
            Summary statistics (mean, stdev, median, min, max, sum) for
            Supply Curve table columns
        """
        sc_table = self.sc_table
        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]

            sc_table = sc_table[columns]

        sc_table = sc_table.select_dtypes(include=np.number)

        sc_summary = []
        sc_stat = sc_table.mean(axis=0)
        sc_stat.name = 'mean'
        sc_summary.append(sc_stat)

        sc_stat = sc_table.std(axis=0)
        sc_stat.name = 'stdev'
        sc_summary.append(sc_stat)

        sc_stat = sc_table.median(axis=0)
        sc_stat.name = 'median'
        sc_summary.append(sc_stat)

        sc_stat = sc_table.min(axis=0)
        sc_stat.name = 'min'
        sc_summary.append(sc_stat)

        sc_stat = sc_table.max(axis=0)
        sc_stat.name = 'max'
        sc_summary.append(sc_stat)

        sc_stat = sc_table.sum(axis=0)
        sc_stat.name = 'sum'
        sc_summary.append(sc_stat)

        sc_summary = pd.concat(sc_summary, axis=1).T

        if out_path is not None:
            sc_summary.to_csv(out_path)

        return sc_summary

    @classmethod
    def run(cls, sc_table, out_dir, columns=None):
        """
        Summarize Supply Curve Table and save to disk

        Parameters
        ----------
        sc_table : str | pandas.DataFrame
            Path to .csv containing Supply Curve table
        out_dir : str
            Directory to dump summary .csv files to
        columns : str | list, optional
            Column(s) to summarize, if None summarize all numeric columns,
            by default None
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        summary = cls(sc_table)
        out_path = os.path.basename(sc_table).replace('.csv', '_summary.csv')
        out_path = os.path.join(out_dir, out_path)
        summary.supply_curve_summary(columns=columns, out_path=out_path)


class PlotBase:
    """
    QA/QC Plotting base class
    """
    def __init__(self, data):
        """
        Parameters
        ----------
        data : str | pandas.DataFrame | ndarray
            data to plot or file containing data to plot
        """
        self._data = data

    def __repr__(self):
        msg = "{}".format(self.__class__.__name__)

        return msg

    @property
    def data(self):
        """
        Data to plot

        Returns
        -------
        pandas.DataFrame | ndarray
        """
        return self._data

    @staticmethod
    def _save_plotly(fig, out_path):
        """
        Save plotly figure to disk

        Parameters
        ----------
        fig : plotly.Figure
            Plotly Figure object
        out_path : str
            File path to save plot to, can be a .html or static image
        """
        if out_path.endswith('.html'):
            fig.write_html(out_path)
        else:
            fig.write_image(out_path)

    @staticmethod
    def _check_value(df, values, scatter=True):
        """
        Check DataFrame for needed columns

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to check
        values : str | list
            Column(s) to plot
        scatter : bool, optional
            Flag to check for latitude and longitude columns, by default True
        """
        if isinstance(values, str):
            values = [values]

        if scatter:
            values += ['latitude', 'longitude']

        for value in values:
            if value not in df:
                msg = ("{} is not a valid column in summary table:\n{}"
                       .format(value, df))
                logger.error(msg)
                raise ValueError(msg)


class SummaryPlots(PlotBase):
    """
    Plot summary data for QA/QC
    """
    def __init__(self, summary):
        """
        Parameters
        ----------
        summary : str | pandas.DataFrame
            Summary DataFrame or path to summary .csv
        """
        self._data = SummarizeSupplyCurve._parse_summary(summary)

    @property
    def summary(self):
        """
        Summary table

        Returns
        -------
        pandas.DataFrame
        """
        return self._data

    @property
    def columns(self):
        """
        Available columns in summary table

        Returns
        -------
        list
        """
        return list(self.summary.columns)

    def scatter_plot(self, value, cmap='viridis', out_path=None, **kwargs):
        """
        Plot scatter plot of value versus longitude and latitude using
        pandas.plot.scatter

        Parameters
        ----------
        value : str
            Column name to plot as color
        cmap : str, optional
            Matplotlib colormap name, by default 'viridis'
        out_path : str, optional
            File path to save plot to, by default None
        kwargs : dict
            Additional kwargs for plotting.dataframes.df_scatter
        """
        self._check_value(self.summary, value)
        mplt.df_scatter(self.summary, x='longitude', y='latitude', c=value,
                        colormap=cmap, filename=out_path, **kwargs)

    def scatter_plotly(self, value, cmap='Viridis', out_path=None, **kwargs):
        """
        Plot scatter plot of value versus longitude and latitude using
        plotly

        Parameters
        ----------
        value : str
            Column name to plot as color
        cmap : str | px.color, optional
            Continuous color scale to use, by default 'Viridis'
        out_path : str, optional
            File path to save plot to, can be a .html or static image,
            by default None
        kwargs : dict
            Additional kwargs for plotly.express.scatter
        """
        self._check_value(self.summary, value)
        fig = px.scatter(self.summary, x='longitude', y='latitude',
                         color=value, color_continuous_scale=cmap, **kwargs)
        fig.update_layout(font=dict(family="Arial", size=18, color="black"))

        if out_path is not None:
            self._save_plotly(fig, out_path)

        fig.show()

    def _extract_sc_data(self, lcoe='mean_lcoe'):
        """
        Extract supply curve data

        Parameters
        ----------
        lcoe : str, optional
            LCOE value to use for supply curve, by default 'mean_lcoe'

        Returns
        -------
        sc_df : pandas.DataFrame
            Supply curve data
        """
        values = ['capacity', lcoe]
        self._check_value(self.summary, values, scatter=False)
        sc_df = self.summary[values].sort_values(lcoe)
        sc_df['cumulative_capacity'] = sc_df['capacity'].cumsum()

        return sc_df

    def dist_plot(self, value, out_path=None, **kwargs):
        """
        Plot distribution plot of value using seaborn.distplot

        Parameters
        ----------
        value : str
            Column name to plot
        out_path : str, optional
            File path to save plot to, by default None
        kwargs : dict
            Additional kwargs for plotting.dataframes.dist_plot
        """
        self._check_value(self.summary, value, scatter=False)
        series = self.summary(value)
        mplt.dist_plot(series, filename=out_path, **kwargs)

    def dist_plotly(self, value, out_path=None, **kwargs):
        """
        Plot histogram of value using plotly

        Parameters
        ----------
        value : str
            Column name to plot
        out_path : str, optional
            File path to save plot to, by default None
        kwargs : dict
            Additional kwargs for plotly.express.histogram
        """
        self._check_value(self.summary, value, scatter=False)

        fig = px.histogram(self.summary, x=value)

        if out_path is not None:
            self._save_plotly(fig, out_path, **kwargs)

        fig.show()

    @classmethod
    def scatter(cls, summary_csv, out_dir, value, plot_type='plotly',
                cmap='viridis', **kwargs):
        """
        Create scatter plot for given value in summary table and save to
        out_dir

        Parameters
        ----------
        summary_csv : str
            Path to .csv file containing summary table
        out_dir : str
            Output directory to save plots to
        value : str
            Column name to plot as color
        plot_type : str, optional
            plot_type of plot to create 'plot' or 'plotly', by default 'plotly'
        cmap : str, optional
            Colormap name, by default 'viridis'
        kwargs : dict
            Additional plotting kwargs
        """
        splt = cls(summary_csv)
        if plot_type == 'plot':
            out_path = os.path.basename(summary_csv).replace('.csv', '.png')
            out_path = os.path.join(out_dir, out_path)
            splt.scatter_plot(value, cmap=cmap.lower(), out_path=out_path,
                              **kwargs)
        elif plot_type == 'plotly':
            out_path = os.path.basename(summary_csv).replace('.csv', '.html')
            out_path = os.path.join(out_dir, out_path)
            splt.scatter_plotly(value, cmap=cmap.capitalize(),
                                out_path=out_path, **kwargs)
        else:
            msg = ("plot_type must be 'plot' or 'plotly' but {} was given"
                   .format(plot_type))
            logger.error(msg)
            raise ValueError(msg)

    @classmethod
    def scatter_all(cls, summary_csv, out_dir, plot_type='plotly',
                    cmap='viridis', **kwargs):
        """
        Create scatter plot for all summary stats in summary table and save to
        out_dir

        Parameters
        ----------
        summary_csv : str
            Path to .csv file containing summary table
        out_dir : str
            Output directory to save plots to
        plot_type : str, optional
            plot_type of plot to create 'plot' or 'plotly', by default 'plotly'
        cmap : str, optional
            Colormap name, by default 'viridis'
        kwargs : dict
            Additional plotting kwargs
        """
        splt = cls(summary_csv)
        splt._data = splt.summary.select_dtypes(include=np.number)
        datasets = [c for c in splt.summary.columns
                    if not c.startswith(('lat', 'lon'))]

        for value in datasets:
            if plot_type == 'plot':
                out_path = '_{}.png'.format(value)
                out_path = \
                    os.path.basename(summary_csv).replace('.csv', out_path)
                out_path = os.path.join(out_dir, out_path)
                splt.scatter_plot(value, cmap=cmap.lower(), out_path=out_path,
                                  **kwargs)
            elif plot_type == 'plotly':
                out_path = '_{}.html'.format(value)
                out_path = \
                    os.path.basename(summary_csv).replace('.csv', out_path)
                out_path = os.path.join(out_dir, out_path)
                splt.scatter_plotly(value, cmap=cmap.capitalize(),
                                    out_path=out_path, **kwargs)
            else:
                msg = ("plot_type must be 'plot' or 'plotly' but {} was given"
                       .format(plot_type))
                logger.error(msg)
                raise ValueError(msg)


class SupplyCurvePlot(PlotBase):
    """
    Plot supply curve data for QA/QC
    """

    def __init__(self, sc_table):
        """
        Parameters
        ----------
        sc_table : str | pandas.DataFrame
            Supply curve table or path to supply curve .csv
        """
        self._data = SummarizeSupplyCurve._parse_summary(sc_table)

    @property
    def sc_table(self):
        """
        Supply curve table

        Returns
        -------
        pandas.DataFrame
        """
        return self._data

    @property
    def columns(self):
        """
        Available columns in supply curve table

        Returns
        -------
        list
        """
        return list(self.sc_table.columns)

    def _extract_sc_data(self, lcoe='mean_lcoe'):
        """
        Extract supply curve data

        Parameters
        ----------
        lcoe : str, optional
            LCOE value to use for supply curve, by default 'mean_lcoe'

        Returns
        -------
        sc_df : pandas.DataFrame
            Supply curve data
        """
        values = ['capacity', lcoe]
        self._check_value(self.sc_table, values, scatter=False)
        sc_df = self.sc_table[values].sort_values(lcoe)
        sc_df['cumulative_capacity'] = sc_df['capacity'].cumsum()

        return sc_df

    def supply_curve_plot(self, lcoe='mean_lcoe', out_path=None, **kwargs):
        """
        Plot supply curve (cumulative capacity vs lcoe) using seaborn.scatter

        Parameters
        ----------
        lcoe : str, optional
            LCOE value to plot, by default 'mean_lcoe'
        out_path : str, optional
            File path to save plot to, by default None
        kwargs : dict
            Additional kwargs for plotting.dataframes.df_scatter
        """
        sc_df = self._extract_sc_data(lcoe=lcoe)
        mplt.df_scatter(sc_df, x='cumulative_capacity', y=lcoe,
                        filename=out_path, **kwargs)

    def supply_curve_plotly(self, lcoe='mean_lcoe', out_path=None, **kwargs):
        """
        Plot supply curve (cumulative capacity vs lcoe) using plotly

        Parameters
        ----------
        lcoe : str, optional
            LCOE value to plot, by default 'mean_lcoe'
        out_path : str, optional
            File path to save plot to, can be a .html or static image,
            by default None
        kwargs : dict
            Additional kwargs for plotly.express.scatter
        """
        sc_df = self._extract_sc_data(lcoe=lcoe)
        fig = px.scatter(sc_df, x='cumulative_capacity', y=lcoe, **kwargs)
        fig.update_layout(font=dict(family="Arial", size=18, color="black"))

        if out_path is not None:
            self._save_plotly(fig, out_path)

        fig.show()

    @classmethod
    def plot(cls, sc_table, out_dir, plot_type='plotly', lcoe='mean_lcoe',
             **kwargs):
        """
        Create supply curve plot from supply curve table using lcoe value
        and save to out_dir

        Parameters
        ----------
        sc_table : str
            Path to .csv file containing Supply Curve table
        out_dir : str
            Output directory to save plots to
        plot_type : str, optional
            plot_type of plot to create 'plot' or 'plotly', by default 'plotly'
        lcoe : str, optional
            LCOE value to plot, by default 'mean_lcoe'
        kwargs : dict
            Additional plotting kwargs
        """
        splt = cls(sc_table)
        if plot_type == 'plot':
            out_path = os.path.basename(sc_table).replace('.csv', '.png')
            out_path = os.path.join(out_dir, out_path)
            splt.supply_curve_plot(lcoe=lcoe, out_path=out_path, **kwargs)
        elif plot_type == 'plotly':
            out_path = os.path.basename(sc_table).replace('.csv', '.html')
            out_path = os.path.join(out_dir, out_path)
            splt.supply_curve_plotly(lcoe=lcoe, out_path=out_path, **kwargs)
        else:
            msg = ("plot_type must be 'plot' or 'plotly' but {} was given"
                   .format(plot_type))
            logger.error(msg)
            raise ValueError(msg)


class ExclusionsMask(PlotBase):
    """
    Plot Exclusions mask as a heat map data for QA/QC
    """

    def __init__(self, excl_mask):
        """
        Parameters
        ----------
        excl_mask : str | ndarray
            Exclusions mask or path to .npy file containing final mask
        """
        self._data = self._parse_mask(excl_mask)

    @property
    def mask(self):
        """
        Final Exclusions mask

        Returns
        -------
        ndarray
        """
        return self._data

    @staticmethod
    def _parse_mask(excl_mask):
        """
        Load exclusions mask if needed

        Parameters
        ----------
        excl_mask : str | ndarray
            Exclusions mask or path to .npy file containing final mask

        Returns
        -------
        excl_mask : ndarray
            [n, m] array of final exclusion values
        """
        if isinstance(excl_mask, str):
            excl_mask = np.load(excl_mask)
        elif not isinstance(excl_mask, np.ndarray):
            raise ValueError("excl_mask must be a .npy file or an ndarray")

        return excl_mask

    def exclusions_plot(self, cmap='Viridis', plot_step=100, out_path=None,
                        **kwargs):
        """
        Plot exclusions mask as a seaborn heatmap

        Parameters
        ----------
        cmap : str | px.color, optional
            Continuous color scale to use, by default 'Viridis'
        plot_step : int
            Step between points to plot
        out_path : str, optional
            File path to save plot to, can be a .html or static image,
            by default None
        kwargs : dict
            Additional kwargs for plotting.colormaps.heatmap_plot
        """
        mplt.heatmap_plot(self.mask[::plot_step, ::plot_step], cmap=cmap,
                          filename=out_path, **kwargs)

    def exclusions_plotly(self, cmap='Viridis', plot_step=100, out_path=None,
                          **kwargs):
        """
        Plot exclusions mask as a plotly heatmap

        Parameters
        ----------
        cmap : str | px.color, optional
            Continuous color scale to use, by default 'Viridis'
        plot_step : int
            Step between points to plot
        out_path : str, optional
            File path to save plot to, can be a .html or static image,
            by default None
        kwargs : dict
            Additional kwargs for plotly.express.imshow
        """
        fig = px.imshow(self.mask[::plot_step, ::plot_step],
                        color_continuous_scale=cmap, **kwargs)
        fig.update_layout(font=dict(family="Arial", size=18, color="black"))

        if out_path is not None:
            SummaryPlots._save_plotly(fig, out_path)

        fig.show()

    @classmethod
    def plot(cls, mask, out_dir, plot_type='plotly', cmap='Viridis',
             plot_step=100, **kwargs):
        """
        Plot exclusions mask and save to out_dir

        Parameters
        ----------
        mask : ndarray
            ndarray of final exclusions mask
        out_dir : str
            Output directory to save plots to
        plot_type : str, optional
            plot_type of plot to create 'plot' or 'plotly', by default 'plotly'
        cmap : str, optional
            Colormap name, by default 'viridis'
        plot_step : int
            Step between points to plot
        kwargs : dict
            Additional plotting kwargs
        """
        excl_mask = cls(mask)
        if plot_type == 'plot':
            out_path = 'exclusions_mask.png'
            out_path = os.path.join(out_dir, out_path)
            excl_mask.exclusions_plot(cmap=cmap.lower(),
                                      plot_step=plot_step,
                                      out_path=out_path,
                                      **kwargs)
        elif plot_type == 'plotly':
            out_path = 'exclusions_mask.html'
            out_path = os.path.join(out_dir, out_path)
            excl_mask.exclusions_plotly(cmap=cmap.capitalize(),
                                        plot_step=plot_step,
                                        out_path=out_path,
                                        **kwargs)
        else:
            msg = ("plot_type must be 'plot' or 'plotly' but {} was given"
                   .format(plot_type))
            logger.error(msg)
            raise ValueError(msg)
