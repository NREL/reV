# -*- coding: utf-8 -*-
"""reV econ module (lcoe-fcr, single owner, etc...)"""
import logging
import os
import pprint
from warnings import warn

import numpy as np
import pandas as pd
from rex.multi_file_resource import MultiFileResource
from rex.resource import Resource
from rex.utilities.utilities import check_res_file

from reV.config.project_points import PointsControl
from reV.generation.base import BaseGen
from reV.handlers.outputs import Outputs
from reV.SAM.econ import LCOE as SAM_LCOE
from reV.SAM.econ import SingleOwner
from reV.SAM.windbos import WindBos
from reV.utilities import ModuleName, ResourceMetaField
from reV.utilities.exceptions import ExecutionError, OffshoreWindInputWarning

logger = logging.getLogger(__name__)


class Econ(BaseGen):
    """Econ"""

    # Mapping of reV econ output strings to SAM econ modules
    OPTIONS = {
        "lcoe_fcr": SAM_LCOE,
        "ppa_price": SingleOwner,
        "project_return_aftertax_npv": SingleOwner,
        "lcoe_real": SingleOwner,
        "lcoe_nom": SingleOwner,
        "flip_actual_irr": SingleOwner,
        "gross_revenue": SingleOwner,
        "total_installed_cost": WindBos,
        "turbine_cost": WindBos,
        "sales_tax_cost": WindBos,
        "bos_cost": WindBos,
        "fixed_charge_rate": SAM_LCOE,
        "capital_cost": SAM_LCOE,
        "fixed_operating_cost": SAM_LCOE,
        "variable_operating_cost": SAM_LCOE,
    }
    """Available ``reV`` econ `output_request` options"""

    # Mapping of reV econ outputs to scale factors and units.
    # Type is scalar or array and corresponds to the SAM single-site output
    OUT_ATTRS = BaseGen.ECON_ATTRS

    def __init__(self, project_points, sam_files, cf_file, site_data=None,
                 output_request=('lcoe_fcr',), sites_per_worker=100,
                 memory_utilization_limit=0.4, append=False):
        """ReV econ analysis class.

        ``reV`` econ analysis runs SAM econ calculations, typically to
        compute LCOE (using :py:class:`PySAM.Lcoefcr.Lcoefcr`), though
        :py:class:`PySAM.Singleowner.Singleowner` or
        :py:class:`PySAM.Windbos.Windbos` calculations can also be
        performed simply by requesting outputs from those computation
        modules. See the keys of
        :attr:`Econ.OPTIONS <reV.econ.econ.Econ.OPTIONS>` for all
        available econ outputs. Econ computations rely on an input a
        generation (i.e. capacity factor) profile. You can request
        ``reV`` to run the analysis for one or more "sites", which
        correspond to the meta indices in the generation data.

        Parameters
        ----------
        project_points : int | list | tuple | str | dict | pd.DataFrame | slice
            Input specifying which sites to process. A single integer
            representing the GID of a site may be specified to evaluate
            reV at a single location. A list or tuple of integers
            (or slice) representing the GIDs of multiple sites can be
            specified to evaluate reV at multiple specific locations.
            A string pointing to a project points CSV file may also be
            specified. Typically, the CSV contains the following
            columns:

                - ``gid``: Integer specifying the generation GID of each
                  site.
                - ``config``: Key in the `sam_files` input dictionary
                  (see below) corresponding to the SAM configuration to
                  use for each particular site. This value can also be
                  ``None`` (or left out completely) if you specify only
                  a single SAM configuration file as the `sam_files`
                  input.
                - ``capital_cost_multiplier``: This is an *optional*
                  multiplier input that, if included, will be used to
                  regionally scale the ``capital_cost`` input in the SAM
                  config. If you include this column in your CSV, you
                  *do not* need to specify ``capital_cost``, unless you
                  would like that value to vary regionally and
                  independently of the multiplier (i.e. the multiplier
                  will still be applied on top of the ``capital_cost``
                  input).

            The CSV file may also contain other site-specific inputs by
            including a column named after a config keyword (e.g. a
            column called ``wind_turbine_rotor_diameter`` may be
            included to specify a site-specific turbine diameter for
            each location). Columns that do not correspond to a config
            key may also be included, but they will be ignored. A
            DataFrame following the same guidelines as the CSV input
            (or a dictionary that can be used to initialize such a
            DataFrame) may be used for this input as well.
        sam_files : dict | str
            A dictionary mapping SAM input configuration ID(s) to SAM
            configuration(s). Keys are the SAM config ID(s) which
            correspond to the ``config`` column in the project points
            CSV. Values for each key are either a path to a
            corresponding SAM config file or a full dictionary
            of SAM config inputs. For example::

                sam_files = {
                    "default": "/path/to/default/sam.json",
                    "onshore": "/path/to/onshore/sam_config.yaml",
                    "offshore": {
                        "sam_key_1": "sam_value_1",
                        "sam_key_2": "sam_value_2",
                        ...
                    },
                    ...
                }

            This input can also be a string pointing to a single SAM
            config file. In this case, the ``config`` column of the
            CSV points input should be set to ``None`` or left out
            completely. See the documentation for the ``reV`` SAM class
            (e.g. :class:`reV.SAM.generation.WindPower`,
            :class:`reV.SAM.generation.PvWattsv8`,
            :class:`reV.SAM.generation.Geothermal`, etc.) for
            documentation on the allowed and/or required SAM config file
            inputs.
        cf_file : str
            Path to reV output generation file containing a capacity
            factor output.

            .. Note:: If executing ``reV`` from the command line, this
              path can contain brackets ``{}`` that will be filled in
              by the `analysis_years` input. Alternatively, this input
              can be set to ``"PIPELINE"`` to parse the output of the
              previous step (``reV`` generation) and use it as input to
              this call. However, note that duplicate executions of
              ``reV`` generation within the pipeline may invalidate this
              parsing, meaning the `cf_file` input will have to be
              specified manually.

        site_data : str | pd.DataFrame, optional
            Site-specific input data for SAM calculation. If this input
            is a string, it should be a path that points to a CSV file.
            Otherwise, this input should be a DataFrame with
            pre-extracted site data. Rows in this table should match
            the input sites via a ``gid`` column. The rest of the
            columns should match configuration input keys that will take
            site-specific values. Note that some or all site-specific
            inputs can be specified via the `project_points` input
            table instead. If ``None``, no site-specific data is
            considered. By default, ``None``.
        output_request : list | tuple, optional
            List of output variables requested from SAM. Can be any
            of the parameters in the "Outputs" group of the PySAM module
            (e.g. :py:class:`PySAM.Windpower.Windpower.Outputs`,
            :py:class:`PySAM.Pvwattsv8.Pvwattsv8.Outputs`,
            :py:class:`PySAM.Geothermal.Geothermal.Outputs`, etc.) being
            executed. This list can also include a select number of SAM
            config/resource parameters to include in the output:
            any key in any of the
            `output attribute JSON files <https://tinyurl.com/4bmrpe3j/>`_
            may be requested. Time-series profiles requested via this
            input are output in UTC. By default, ``('lcoe_fcr',)``.
        sites_per_worker : int, optional
            Number of sites to run in series on a worker. ``None``
            defaults to the resource file chunk size.
            By default, ``None``.
        memory_utilization_limit : float, optional
            Memory utilization limit (fractional). Must be a value
            between 0 and 1. This input sets how many site results will
            be stored in-memory at any given time before flushing to
            disk. By default, ``0.4``.
        append : bool
            Option to append econ datasets to source `cf_file`.
            By default, ``False``.
        """

        # get a points control instance
        pc = self.get_pc(
            points=project_points,
            points_range=None,
            sam_configs=sam_files,
            cf_file=cf_file,
            sites_per_worker=sites_per_worker,
            append=append,
        )

        super().__init__(
            pc,
            output_request,
            site_data=site_data,
            memory_utilization_limit=memory_utilization_limit,
        )

        self._cf_file = cf_file
        self._append = append
        self._run_attrs["cf_file"] = cf_file
        self._run_attrs["sam_module"] = self._sam_module.MODULE

    @property
    def cf_file(self):
        """Get the capacity factor output filename and path.

        Returns
        -------
        cf_file : str
            reV generation capacity factor output file with path.
        """
        return self._cf_file

    @property
    def meta(self):
        """Get meta data from the source capacity factors file.

        Returns
        -------
        _meta : pd.DataFrame
            Meta data from capacity factor outputs file.
        """
        if self._meta is None and self.cf_file is not None:
            with Outputs(self.cf_file) as cfh:
                # only take meta that belongs to this project's site list
                self._meta = cfh.meta[
                    cfh.meta[ResourceMetaField.GID].isin(
                        self.points_control.sites)]

            if ("offshore" in self._meta and self._meta["offshore"].sum() > 1):
                w = ('Found offshore sites in econ meta data. '
                     'This functionality has been deprecated. '
                     'Please run the reV offshore module to '
                     'calculate offshore wind lcoe.')
                warn(w, OffshoreWindInputWarning)
                logger.warning(w)

        elif self._meta is None and self.cf_file is None:
            self._meta = pd.DataFrame(
                {ResourceMetaField.GID: self.points_control.sites})

        return self._meta

    @property
    def time_index(self):
        """Get the generation resource time index data."""
        if self._time_index is None and self.cf_file is not None:
            with Outputs(self.cf_file) as cfh:
                if "time_index" in cfh.datasets:
                    self._time_index = cfh.time_index

        return self._time_index

    @staticmethod
    def _econ_append_pc(pp, cf_file, sites_per_worker=None):
        """
        Generate ProjectControls for econ append

        Parameters
        ----------
        pp : reV.config.project_points.ProjectPoints
            ProjectPoints to adjust gids for
        cf_file : str
            reV generation capacity factor output file with path.
        sites_per_worker : int
            Number of sites to run in series on a worker. None defaults to the
            resource file chunk size.

        Returns
        -------
        pc : reV.config.project_points.PointsControl
            PointsControl object instance.
        """
        multi_h5_res, hsds = check_res_file(cf_file)
        if multi_h5_res:
            res_cls = MultiFileResource
            res_kwargs = {}
        else:
            res_cls = Resource
            res_kwargs = {"hsds": hsds}

        with res_cls(cf_file, **res_kwargs) as f:
            gid0 = f.meta[ResourceMetaField.GID].values[0]
            gid1 = f.meta[ResourceMetaField.GID].values[-1]

        i0 = pp.index(gid0)
        i1 = pp.index(gid1) + 1
        pc = PointsControl.split(i0, i1, pp, sites_per_split=sites_per_worker)

        return pc

    @classmethod
    def get_pc(
        cls,
        points,
        points_range,
        sam_configs,
        cf_file,
        sites_per_worker=None,
        append=False,
    ):
        """
        Get a PointsControl instance.

        Parameters
        ----------
        points : slice | list | str | reV.config.project_points.PointsControl
            Slice specifying project points, or string pointing to a project
            points csv, or a fully instantiated PointsControl object.
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. To be taken from the reV.config.PointsControl.split_range
            property.
        sam_configs : dict | str | SAMConfig
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s) which map to the config column in the project points
            CSV. Values are either a JSON SAM config file or dictionary of SAM
            config inputs. Can also be a single config file path or a
            pre loaded SAMConfig object.
        cf_file : str
            reV generation capacity factor output file with path.
        sites_per_worker : int
            Number of sites to run in series on a worker. None defaults to the
            resource file chunk size.
        append : bool
            Flag to append econ datasets to source cf_file. This has priority
            over the out_fpath input.

        Returns
        -------
        pc : reV.config.project_points.PointsControl
            PointsControl object instance.
        """
        pc = super().get_pc(
            points,
            points_range,
            sam_configs,
            ModuleName.ECON,
            sites_per_worker=sites_per_worker,
            res_file=cf_file,
        )

        if append:
            pc = cls._econ_append_pc(
                pc.project_points, cf_file, sites_per_worker=sites_per_worker
            )

        return pc

    @staticmethod
    def _run_single_worker(pc, econ_fun, output_request, **kwargs):
        """Run the SAM econ calculation.

        Parameters
        ----------
        pc : reV.config.project_points.PointsControl
            Iterable points control object from reV config module.
            Must have project_points with df property with all relevant
            site-specific inputs and a `SiteDataField.GID` column.
            By passing site-specific inputs in this dataframe, which
            was split using points_control, only the data relevant to
            the current sites is passed.
        econ_fun : method
            reV_run() method from one of the econ modules (SingleOwner,
            SAM_LCOE, WindBos).
        output_request : str | list | tuple
            Economic output variable(s) requested from SAM.
        kwargs : dict
            Additional input parameters for the SAM run module.

        Returns
        -------
        out : dict
            Output dictionary from the SAM reV_run function. Data is scaled
            within this function to the datatype specified in Econ.OUT_ATTRS.
        """

        # make sure output request is a list
        if isinstance(output_request, str):
            output_request = [output_request]

        # Extract the site df from the project points df.
        site_df = pc.project_points.df
        site_df = site_df.set_index(ResourceMetaField.GID, drop=True)

        # SAM execute econ analysis based on output request
        try:
            out = econ_fun(
                pc, site_df, output_request=output_request, **kwargs
            )
        except Exception as e:
            out = {}
            logger.exception("Worker failed for PC: {}".format(pc))
            raise e

        return out

    def _parse_output_request(self, req):
        """Set the output variables requested from generation.

        Parameters
        ----------
        req : str| list | tuple
            Output variables requested from SAM.

        Returns
        -------
        output_request : list
            Output variables requested from SAM.
        """

        output_request = super()._parse_output_request(req)

        for request in output_request:
            if request not in self.OUT_ATTRS:
                msg = (
                    'User output request "{}" not recognized. '
                    "Will attempt to extract from PySAM.".format(request)
                )
                logger.debug(msg)

        modules = [self.OPTIONS[request] for request in output_request
                   if request in self.OPTIONS]

        if not any(modules):
            msg = (
                "None of the user output requests were recognized. "
                "Cannot run reV econ. "
                "At least one of the following must be requested: {}".format(
                    list(self.OPTIONS.keys())
                )
            )
            logger.exception(msg)
            raise ExecutionError(msg)

        b1 = [m == modules[0] for m in modules]
        b2 = np.array([m == WindBos for m in modules])
        b3 = np.array([m == SingleOwner for m in modules])

        if all(b1):
            self._sam_module = modules[0]
            self._fun = modules[0].reV_run
        elif all(b2 | b3):
            self._sam_module = SingleOwner
            self._fun = SingleOwner.reV_run
        else:
            msg = (
                "Econ outputs requested from different SAM modules not "
                "currently supported. Output request variables require "
                "SAM methods: {}".format(modules)
            )
            raise ValueError(msg)

        return list(set(output_request))

    def _get_data_shape(self, dset, n_sites):
        """Get the output array shape based on OUT_ATTRS or PySAM.Outputs.

        This Econ get data shape method will also first check for the dset in
        the site_data table. If not found in site_data, the dataset will be
        looked for in OUT_ATTRS and PySAM.Outputs as it would for Generation.

        Parameters
        ----------
        dset : str
            Variable name to get shape for.
        n_sites : int
            Number of sites for this data shape.

        Returns
        -------
        shape : tuple
            1D or 2D shape tuple for dset.
        """

        if dset in self.site_data:
            data_shape = (n_sites,)
            data = self.site_data[dset].values[0]

            if isinstance(data, (list, tuple, np.ndarray, str)):
                msg = (
                    "Cannot pass through non-scalar site_data "
                    'input key "{}" as an output_request!'.format(dset)
                )
                logger.error(msg)
                raise ExecutionError(msg)

        else:
            data_shape = super()._get_data_shape(dset, n_sites)

        return data_shape

    def run(self, out_fpath=None, max_workers=1, timeout=1800, pool_size=None):
        """Execute a parallel reV econ run with smart data flushing.

        Parameters
        ----------
        out_fpath : str, optional
            Path to output file. If this class was initialized with
            ``append=True``, this input has no effect. If ``None``, no
            output file will be written. If the filepath is specified
            but the module name (econ) and/or resource data year is not
            included, the module name and/or resource data year will get
            added to the output file name. By default, ``None``.
        max_workers : int, optional
            Number of local workers to run on. By default, ``1``.
        timeout : int, optional
            Number of seconds to wait for parallel run iteration to
            complete before returning zeros. By default, ``1800``
            seconds.
        pool_size : int, optional
            Number of futures to submit to a single process pool for
            parallel futures. If ``None``, the pool size is set to
            ``os.cpu_count() * 2``. By default, ``None``.

        Returns
        -------
        str | None
            Path to output HDF5 file, or ``None`` if results were not
            written to disk.
        """
        if pool_size is None:
            pool_size = os.cpu_count() * 2

        # initialize output file or append econ data to gen file
        if self._append:
            self._out_fpath = self._cf_file
        else:
            self._init_fpath(out_fpath, ModuleName.ECON)

        self._init_h5(mode="a" if self._append else "w")
        self._init_out_arrays()

        diff = list(set(self.points_control.sites)
                    - set(self.meta[ResourceMetaField.GID].values))
        if diff:
            raise Exception(
                "The following analysis sites were requested "
                "through project points for econ but are not "
                'found in the CF file ("{}"): {}'.format(self.cf_file, diff)
            )

        # make a kwarg dict
        kwargs = {
            "output_request": self.output_request,
            "cf_file": self.cf_file,
            "year": self.year,
        }

        logger.info(
            "Running econ with smart data flushing " "for: {}".format(
                self.points_control
            )
        )
        logger.debug(
            'The following project points were specified: "{}"'.format(
                self.project_points
            )
        )
        logger.debug(
            "The following SAM configs are available to this run:\n{}".format(
                pprint.pformat(self.sam_configs, indent=4)
            )
        )
        logger.debug(
            "The SAM output variables have been requested:\n{}".format(
                self.output_request
            )
        )

        try:
            kwargs["econ_fun"] = self._fun
            if max_workers == 1:
                logger.debug(
                    "Running serial econ for: {}".format(self.points_control)
                )
                for i, pc_sub in enumerate(self.points_control):
                    self.out = self._run_single_worker(pc_sub, **kwargs)
                    logger.info(
                        "Finished reV econ serial compute for: {} "
                        "(iteration {} out of {})".format(
                            pc_sub, i + 1, len(self.points_control)
                        )
                    )
                self.flush()
            else:
                logger.debug(
                    "Running parallel econ for: {}".format(self.points_control)
                )
                self._parallel_run(
                    max_workers=max_workers,
                    pool_size=pool_size,
                    timeout=timeout,
                    **kwargs,
                )

        except Exception as e:
            logger.exception("SmartParallelJob.execute() failed for econ.")
            raise e

        return self._out_fpath
