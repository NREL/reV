# -*- coding: utf-8 -*-
"""
reV quality assurance and control classes
"""

import logging
import os
from warnings import warn

import numpy as np
import pandas as pd
from gaps.status import Status

from reV.qa_qc.summary import (
    ExclusionsMask,
    SummarizeH5,
    SummarizeSupplyCurve,
    SummaryPlots,
    SupplyCurvePlot,
)
from reV.supply_curve.exclusions import ExclusionMaskFromDict
from reV.utilities import ModuleName, SupplyCurveField, log_versions
from reV.utilities.exceptions import PipelineError

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
        log_versions(logger)
        logger.info("QA/QC results to be saved to: {}".format(out_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

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
    def _scatter_plot(
        summary_csv, out_root, plot_type="plotly", cmap="viridis", **kwargs
    ):
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
        out_dir = os.path.join(
            out_root, os.path.basename(summary_csv).rstrip(".csv")
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        SummaryPlots.scatter_all(
            summary_csv, out_dir, plot_type=plot_type, cmap=cmap, **kwargs
        )

    def create_scatter_plots(
        self, plot_type="plotly", cmap="viridis", **kwargs
    ):
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
            if file.endswith(".csv"):
                summary_csv = os.path.join(self.out_dir, file)
                summary = pd.read_csv(summary_csv)
                has_right_cols = ("gid" in summary
                                  and SupplyCurveField.LATITUDE in summary
                                  and SupplyCurveField.LONGITUDE in summary)
                if has_right_cols:
                    self._scatter_plot(summary_csv, self.out_dir,
                                       plot_type=plot_type, cmap=cmap,
                                       **kwargs)

    @classmethod
    def h5(
        cls,
        h5_file,
        out_dir,
        dsets=None,
        group=None,
        process_size=None,
        max_workers=None,
        plot_type="plotly",
        cmap="viridis",
        **kwargs,
    ):
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
            SummarizeH5.run(
                h5_file,
                out_dir,
                group=group,
                dsets=dsets,
                process_size=process_size,
                max_workers=max_workers,
            )
            qa_qc.create_scatter_plots(
                plot_type=plot_type, cmap=cmap, **kwargs
            )
        except Exception as e:
            logger.exception(
                "QAQC failed on file: {}. Received exception:\n{}".format(
                    os.path.basename(h5_file), e
                )
            )
            raise e
        else:
            logger.info(
                "Finished QAQC on file: {} output directory: {}".format(
                    os.path.basename(h5_file), out_dir
                )
            )

    @classmethod
    def supply_curve(cls, sc_table, out_dir, columns=None,
                     lcoe=SupplyCurveField.MEAN_LCOE, plot_type='plotly',
                     cmap='viridis', sc_plot_kwargs=None,
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
            LCOE value to plot, by default :obj:`SupplyCurveField.MEAN_LCOE`
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
            SupplyCurvePlot.plot(
                sc_table,
                out_dir,
                plot_type=plot_type,
                lcoe=lcoe,
                **sc_plot_kwargs,
            )
            qa_qc._scatter_plot(
                sc_table,
                out_dir,
                plot_type=plot_type,
                cmap=cmap,
                **scatter_plot_kwargs,
            )
        except Exception as e:
            logger.exception(
                "QAQC failed on file: {}. Received exception:\n{}".format(
                    os.path.basename(sc_table), e
                )
            )
            raise e
        else:
            logger.info(
                "Finished QAQC on file: {} output directory: {}".format(
                    os.path.basename(sc_table), out_dir
                )
            )

    @classmethod
    def exclusions_mask(
        cls,
        excl_h5,
        out_dir,
        layers_dict=None,
        min_area=None,
        kernel="queen",
        hsds=False,
        plot_type="plotly",
        cmap="viridis",
        plot_step=100,
        **kwargs,
    ):
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
            excl_mask = ExclusionMaskFromDict.run(
                excl_h5,
                layers_dict=layers_dict,
                min_area=min_area,
                kernel=kernel,
                hsds=hsds,
            )
            excl_mask = np.round(excl_mask * 100).astype("uint8")

            out_file = os.path.basename(excl_h5).replace(".h5", "_mask.npy")
            out_file = os.path.join(out_dir, out_file)
            np.save(out_file, excl_mask)

            ExclusionsMask.plot(
                excl_mask,
                out_dir,
                plot_type=plot_type,
                cmap=cmap,
                plot_step=plot_step,
                **kwargs,
            )
        except Exception as e:
            logger.exception(
                "QAQC failed on file: {}. Received exception:\n{}".format(
                    os.path.basename(excl_h5), e
                )
            )
            raise e
        else:
            logger.info(
                "Finished QAQC on file: {} output directory: {}".format(
                    os.path.basename(excl_h5), out_dir
                )
            )


class QaQcModule:
    """Class to handle Module QA/QC"""

    def __init__(self, module_name, config, out_root):
        """
        Parameters
        ----------
        config : dict
            Dictionary with pre-extracted config input group.
        """
        if not isinstance(config, dict):
            raise TypeError(
                "Config input must be a dict but received: {}".format(
                    type(config)
                )
            )

        self._name = module_name
        self._config = config
        self._out_root = out_root
        self._default_plot_type = "plotly"
        self._default_cmap = "viridis"
        self._default_plot_step = 100
        self._default_lcoe = SupplyCurveField.MEAN_LCOE
        self._default_area_filter_kernel = 'queen'

    @property
    def fpath(self):
        """Get the reV module output filepath(s)

        Returns
        -------
        fpaths : str | list
            One or more filepaths output by current module being QA'd
        """

        fpath = self._config["fpath"]

        if fpath == "PIPELINE":
            target_modules = [self._name]
            for target_module in target_modules:
                fpath = Status.parse_step_status(self._out_root, target_module)
                if fpath:
                    break
            else:
                raise PipelineError(
                    "Could not parse fpath from previous pipeline jobs."
                )
            fpath = fpath[0]
            logger.info(
                "QA/QC using the following "
                "pipeline input for fpath: {}".format(fpath)
            )

        return fpath

    @property
    def sub_dir(self):
        """
        QA/QC sub directory for this module's outputs
        """
        return self._config.get("sub_dir", None)

    @property
    def plot_type(self):
        """Get the QA/QC plot type: either 'plot' or 'plotly'"""
        return self._config.get("plot_type", self._default_plot_type)

    @property
    def dsets(self):
        """Get the reV_h5 dsets to QA/QC"""
        return self._config.get("dsets", None)

    @property
    def group(self):
        """Get the reV_h5 group to QA/QC"""
        return self._config.get("group", None)

    @property
    def process_size(self):
        """Get the reV_h5 process_size for QA/QC"""
        return self._config.get("process_size", None)

    @property
    def cmap(self):
        """Get the QA/QC plot colormap"""
        return self._config.get("cmap", self._default_cmap)

    @property
    def plot_step(self):
        """Get the QA/QC step between exclusion mask points to plot"""
        return self._config.get("cmap", self._default_plot_step)

    @property
    def columns(self):
        """Get the supply_curve columns to QA/QC"""
        return self._config.get("columns", None)

    @property
    def lcoe(self):
        """Get the supply_curve lcoe column to plot"""
        return self._config.get("lcoe", self._default_lcoe)

    @property
    def excl_fpath(self):
        """Get the source exclusions filepath"""
        excl_fpath = self._config.get("excl_fpath", "PIPELINE")

        if excl_fpath == "PIPELINE":
            target_module = ModuleName.SUPPLY_CURVE_AGGREGATION
            excl_fpath = Status.parse_step_status(
                self._out_root, target_module, key="excl_fpath"
            )
            if not excl_fpath:
                excl_fpath = None
                msg = (
                    "Could not parse excl_fpath from previous "
                    "pipeline jobs, defaulting to: {}".format(excl_fpath)
                )
                logger.warning(msg)
                warn(msg)
            else:
                excl_fpath = excl_fpath[0]
                logger.info(
                    "QA/QC using the following "
                    "pipeline input for excl_fpath: {}".format(excl_fpath)
                )

        return excl_fpath

    @property
    def excl_dict(self):
        """Get the exclusions dictionary"""
        excl_dict = self._config.get("excl_dict", "PIPELINE")

        if excl_dict == "PIPELINE":
            target_module = ModuleName.SUPPLY_CURVE_AGGREGATION
            excl_dict = Status.parse_step_status(
                self._out_root, target_module, key="excl_dict"
            )
            if not excl_dict:
                excl_dict = None
                msg = (
                    "Could not parse excl_dict from previous "
                    "pipeline jobs, defaulting to: {}".format(excl_dict)
                )
                logger.warning(msg)
                warn(msg)
            else:
                excl_dict = excl_dict[0]
                logger.info(
                    "QA/QC using the following "
                    "pipeline input for excl_dict: {}".format(excl_dict)
                )

        return excl_dict

    @property
    def area_filter_kernel(self):
        """Get the minimum area filter kernel name ('queen' or 'rook')."""
        area_filter_kernel = self._config.get("area_filter_kernel", "PIPELINE")

        if area_filter_kernel == "PIPELINE":
            target_module = ModuleName.SUPPLY_CURVE_AGGREGATION
            key = "area_filter_kernel"
            area_filter_kernel = Status.parse_step_status(
                self._out_root, target_module, key=key
            )
            if not area_filter_kernel:
                area_filter_kernel = self._default_area_filter_kernel
                msg = (
                    "Could not parse area_filter_kernel from previous "
                    "pipeline jobs, defaulting to: {}".format(
                        area_filter_kernel
                    )
                )
                logger.warning(msg)
                warn(msg)
            else:
                area_filter_kernel = area_filter_kernel[0]
                logger.info(
                    "QA/QC using the following "
                    "pipeline input for area_filter_kernel: {}".format(
                        area_filter_kernel
                    )
                )

        return area_filter_kernel

    @property
    def min_area(self):
        """Get the minimum area filter minimum area in km2."""
        min_area = self._config.get("min_area", "PIPELINE")

        if min_area == "PIPELINE":
            target_module = ModuleName.SUPPLY_CURVE_AGGREGATION
            min_area = Status.parse_step_status(
                self._out_root, target_module, key="min_area"
            )
            if not min_area:
                min_area = None
                msg = (
                    "Could not parse min_area from previous "
                    "pipeline jobs, defaulting to: {}".format(min_area)
                )
                logger.warning(msg)
                warn(msg)
            else:
                min_area = min_area[0]
                logger.info(
                    "QA/QC using the following "
                    "pipeline input for min_area: {}".format(min_area)
                )

        return min_area
