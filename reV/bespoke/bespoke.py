# -*- coding: utf-8 -*-
"""
reV bespoke wind plant analysis tools
"""

# pylint: disable=anomalous-backslash-in-string
import copy
import json
import logging
import os
import time
from concurrent.futures import as_completed
from importlib import import_module
from inspect import signature
from numbers import Number
from warnings import warn

import numpy as np
import pandas as pd
import psutil
from rex.joint_pd.joint_pd import JointPD
from rex.multi_year_resource import MultiYearWindResource
from rex.utilities.bc_parse_table import parse_bc_table
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import create_dirs, log_mem
from rex.utilities.utilities import parse_year

from reV.config.output_request import SAMOutputRequest
from reV.econ.utilities import lcoe_fcr
from reV.generation.generation import Gen
from reV.handlers.exclusions import ExclusionLayers
from reV.handlers.outputs import Outputs
from reV.SAM.generation import WindPower, WindPowerPD
from reV.supply_curve.aggregation import AggFileHandler, BaseAggregation
from reV.supply_curve.extent import SupplyCurveExtent
from reV.supply_curve.points import AggregationSupplyCurvePoint as AggSCPoint
from reV.supply_curve.points import SupplyCurvePoint
from reV.utilities import (
    ModuleName,
    ResourceMetaField,
    SupplyCurveField,
    log_versions,
)
from reV.utilities.exceptions import EmptySupplyCurvePointError, FileInputError

logger = logging.getLogger(__name__)


class BespokeMultiPlantData:
    """Multi-plant preloaded data.

    This object is intended to facilitate the use of pre-loaded data for
    running :class:`BespokeWindPlants` on systems with slow parallel
    reads to a single HDF5 file.
    """

    def __init__(self, res_fpath, sc_gid_to_hh, sc_gid_to_res_gid,
                 pre_load_humidity=False):
        """Initialize BespokeMultiPlantData

        Parameters
        ----------
        res_fpath : str | list
            Unix shell style path (potentially containing wildcard (*)
            patterns) to a single or multi-file resource file set(s).
            Can also be an explicit list of resource file paths, which
            themselves can contain wildcards. This input must be
            readable by
            :py:class:`rex.multi_year_resource.MultiYearWindResource`.
        sc_gid_to_hh : dict
            Dictionary mapping SC GID values to hub-heights. Data for
            each SC GID will be pulled for the corresponding hub-height
            given in this dictionary.
        sc_gid_to_res_gid : dict
            Dictionary mapping SC GID values to an iterable oif resource
            GID values. Resource GID values should correspond to GID
            values in the HDF5 file, so any GID map must be applied
            before initializing :class`BespokeMultiPlantData`.
        pre_load_humidity : optional, default=False
            Option to pre-load relative humidity data (useful for icing
            runs). If ``False``, relative humidities are not loaded.
        """
        self.res_fpath = res_fpath
        self.sc_gid_to_hh = sc_gid_to_hh
        self.sc_gid_to_res_gid = sc_gid_to_res_gid
        self.hh_to_res_gids = {}
        self._wind_dirs = None
        self._wind_speeds = None
        self._temps = None
        self._pressures = None
        self._relative_humidities = None
        self._pre_load_humidity = pre_load_humidity
        self._time_index = None
        self._pre_load_data()

    def _pre_load_data(self):
        """Pre-load the resource data."""

        for sc_gid, gids in self.sc_gid_to_res_gid.items():
            hh = self.sc_gid_to_hh[sc_gid]
            self.hh_to_res_gids.setdefault(hh, set()).update(gids)

        self.hh_to_res_gids = {
            hh: sorted(gids) for hh, gids in self.hh_to_res_gids.items()
        }

        start_time = time.time()
        with MultiYearWindResource(self.res_fpath) as res:
            self._wind_dirs = {
                hh: res[f"winddirection_{hh}m", :, gids]
                for hh, gids in self.hh_to_res_gids.items()
            }
            self._wind_speeds = {
                hh: res[f"windspeed_{hh}m", :, gids]
                for hh, gids in self.hh_to_res_gids.items()
            }
            self._temps = {
                hh: res[f"temperature_{hh}m", :, gids]
                for hh, gids in self.hh_to_res_gids.items()
            }
            self._pressures = {
                hh: res[f"pressure_{hh}m", :, gids]
                for hh, gids in self.hh_to_res_gids.items()
            }
            if self._pre_load_humidity:
                self._relative_humidities = {
                    hh: res["relativehumidity_2m", :, gids]
                    for hh, gids in self.hh_to_res_gids.items()
                }
            self._time_index = res.time_index
            if self._pre_load_humidity:
                self._relative_humidities = {
                    hh: res["relativehumidity_2m", :, gids]
                    for hh, gids in self.hh_to_res_gids.items()
                }

        logger.debug(
            f"Data took {(time.time() - start_time) / 60:.2f} " f"min to load"
        )

    def get_preloaded_data_for_gid(self, sc_gid):
        """Get the pre-loaded data for a single SC GID.

        Parameters
        ----------
        sc_gid : int
            SC GID to load resource data for.

        Returns
        -------
        BespokeSinglePlantData
            A loaded ``BespokeSinglePlantData`` object that can act as
            an HDF5 handler stand-in *for this SC GID only*.
        """
        hh = self.sc_gid_to_hh[sc_gid]
        sc_point_res_gids = sorted(self.sc_gid_to_res_gid[sc_gid])
        data_inds = np.searchsorted(self.hh_to_res_gids[hh], sc_point_res_gids)

        rh = (None if not self._pre_load_humidity
              else self._relative_humidities[hh][:, data_inds])
        return BespokeSinglePlantData(sc_point_res_gids,
                                      self._wind_dirs[hh][:, data_inds],
                                      self._wind_speeds[hh][:, data_inds],
                                      self._temps[hh][:, data_inds],
                                      self._pressures[hh][:, data_inds],
                                      self._time_index, rh)


class BespokeSinglePlantData:
    """Single-plant preloaded data.

    This object is intended to facilitate the use of pre-loaded data for
    running :class:`BespokeSinglePlant` on systems with slow parallel
    reads to a single HDF5 file.
    """

    def __init__(self, data_inds, wind_dirs, wind_speeds, temps, pressures,
                 time_index, relative_humidities=None):
        """Initialize BespokeSinglePlantData

        Parameters
        ----------
        data_inds : 1D np.array
            Array of res GIDs. This array should be the same length as
            the second dimension of `wind_dirs`, `wind_speeds`, `temps`,
            and `pressures`. The GID value of data_inds[0] should
            correspond to the `wind_dirs[:, 0]` data, etc.
        wind_dirs : 2D np.array
            Array of wind directions. Dimensions should be correspond to
            [time, location]. See documentation for `data_inds` for
            required spatial mapping of GID values.
        wind_speeds : 2D np.array
            Array of wind speeds. Dimensions should be correspond to
            [time, location]. See documentation for `data_inds` for
            required spatial mapping of GID values.
        temps : 2D np.array
            Array oftemperatures. Dimensions should be correspond to
            [time, location]. See documentation for `data_inds` for
            required spatial mapping of GID values.
        pressures : 2D np.array
            Array of pressures. Dimensions should be correspond to
            pressures, respectively. Dimensions should be correspond to
            [time, location]. See documentation for `data_inds` for
            required spatial mapping of GID values.
        time_index : 1D np.array
            Time index array corresponding to the temporal dimension of
            the 2D data. Will be exposed directly to user.
        relative_humidities : 2D np.array, optional
            Array of relative humidities. Dimensions should be
            correspond to [time, location]. See documentation for
            `data_inds` for required spatial mapping of GID values.
            If ``None``, relative_humidities cannot be queried.
        """

        self.data_inds = data_inds
        self.wind_dirs = wind_dirs
        self.wind_speeds = wind_speeds
        self.temps = temps
        self.pressures = pressures
        self.time_index = time_index
        self.relative_humidities = relative_humidities
        self._humidities_exist = relative_humidities is not None

    def __getitem__(self, key):
        dset_name, t_idx, gids = key
        data_inds = np.searchsorted(self.data_inds, gids)
        if "winddirection" in dset_name:
            return self.wind_dirs[t_idx, data_inds]
        if "windspeed" in dset_name:
            return self.wind_speeds[t_idx, data_inds]
        if "temperature" in dset_name:
            return self.temps[t_idx, data_inds]
        if "pressure" in dset_name:
            return self.pressures[t_idx, data_inds]
        if self._humidities_exist and "relativehumidity" in dset_name:
            return self.relative_humidities[t_idx, data_inds]
        msg = f"Unknown dataset name: {dset_name!r}"
        logger.error(msg)
        raise ValueError(msg)


class BespokeSinglePlant:
    """Framework for analyzing and optimizing a wind plant layout specific to
    the local wind resource and exclusions for a single reV supply curve point.
    """

    DEPENDENCIES = ("shapely",)
    OUT_ATTRS = copy.deepcopy(Gen.OUT_ATTRS)

    def __init__(self, gid, excl, res, tm_dset, sam_sys_inputs,
                 objective_function, capital_cost_function,
                 fixed_operating_cost_function,
                 variable_operating_cost_function,
                 balance_of_system_cost_function, min_spacing='5x',
                 ga_kwargs=None, output_request=('system_capacity', 'cf_mean'),
                 ws_bins=(0.0, 20.0, 5.0), wd_bins=(0.0, 360.0, 45.0),
                 excl_dict=None, inclusion_mask=None, data_layers=None,
                 resolution=64, excl_area=None, exclusion_shape=None,
                 eos_mult_baseline_cap_mw=200, prior_meta=None, gid_map=None,
                 bias_correct=None, pre_loaded_data=None, close=True):
        """
        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        excl : str | ExclusionMask
            Filepath to exclusions h5 or ExclusionMask file handler.
        res : str | Resource
            Filepath to .h5 wind resource file or pre-initialized Resource
            handler
        tm_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.
        sam_sys_inputs : dict
            SAM windpower compute module system inputs not including the
            wind resource data.
        objective_function : str
            The objective function of the optimization as a string, should
            return the objective to be minimized during layout optimization.
            Variables available are:

                - ``n_turbines``: the number of turbines
                - ``system_capacity``: wind plant capacity
                - ``aep``: annual energy production
                - ``avg_sl_dist_to_center_m``: Average straight-line
                  distance to the supply curve point center from all
                  turbine locations (in m). Useful for computing plant
                  BOS costs.
                - ``avg_sl_dist_to_medoid_m``: Average straight-line
                  distance to the medoid of all turbine locations
                  (in m). Useful for computing plant BOS costs.
                - ``nn_conn_dist_m``: Total BOS connection distance
                  using nearest-neighbor connections. This variable is
                  only available for the
                  ``balance_of_system_cost_function`` equation.
                - ``fixed_charge_rate``: user input fixed_charge_rate if
                  included as part of the sam system config.
                - ``capital_cost``: plant capital cost as evaluated
                  by `capital_cost_function`
                - ``fixed_operating_cost``: plant fixed annual operating
                  cost as evaluated by `fixed_operating_cost_function`
                - ``variable_operating_cost``: plant variable annual
                  operating cost as evaluated by
                  `variable_operating_cost_function`
                - ``balance_of_system_cost``: plant balance of system
                  cost as evaluated by `balance_of_system_cost_function`
                - ``self.wind_plant``: the SAM wind plant object,
                  through which all SAM variables can be accessed

        capital_cost_function : str
            The plant capital cost function as a string, must return the total
            capital cost in $. Has access to the same variables as the
            objective_function.
        fixed_operating_cost_function : str
            The plant annual fixed operating cost function as a string, must
            return the fixed operating cost in $/year. Has access to the same
            variables as the objective_function.
        variable_operating_cost_function : str
            The plant annual variable operating cost function as a string, must
            return the variable operating cost in $/kWh. Has access to the same
            variables as the objective_function. You can set this to "0"
            to effectively ignore variable operating costs.
        balance_of_system_cost_function : str
            The plant balance-of-system cost function as a string, must
            return the variable operating cost in $. Has access to the
            same variables as the objective_function. You can set this
            to "0" to effectively ignore balance-of-system costs.
        balance_of_system_cost_function : str
            The plant balance-of-system cost function as a string, must
            return the variable operating cost in $. Has access to the same
            variables as the objective_function.
        min_spacing : float | int | str
            Minimum spacing between turbines in meters. Can also be a string
            like "5x" (default) which is interpreted as 5 times the turbine
            rotor diameter.
        ga_kwargs : dict | None
            Dictionary of keyword arguments to pass to GA initialization.
            If `None`, default initialization values are used.
            See :class:`~reV.bespoke.gradient_free.GeneticAlgorithm` for
            a description of the allowed keyword arguments.
        output_request : list | tuple
            Outputs requested from the SAM windpower simulation after the
            bespoke plant layout optimization. Can also request resource means
            like ws_mean, windspeed_mean, temperature_mean, pressure_mean.
        ws_bins : tuple
            3-entry tuple with (start, stop, step) for the windspeed binning of
            the wind joint probability distribution. The stop value is
            inclusive, so ws_bins=(0, 20, 5) would result in four bins with bin
            edges (0, 5, 10, 15, 20).
        wd_bins : tuple
            3-entry tuple with (start, stop, step) for the winddirection
            binning of the wind joint probability distribution. The stop value
            is inclusive, so ws_bins=(0, 360, 90) would result in four bins
            with bin edges (0, 90, 180, 270, 360).
        excl_dict : dict | None
            Dictionary of exclusion keyword arugments of the format
            {layer_dset_name: {kwarg: value}} where layer_dset_name is a
            dataset in the exclusion h5 file and kwarg is a keyword argument to
            the reV.supply_curve.exclusions.LayerMask class.
            None if excl input is pre-initialized.
        inclusion_mask : np.ndarray
            2D array pre-extracted inclusion mask where 1 is included and 0 is
            excluded. The shape of this will be checked against the input
            resolution.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        resolution : int
            Number of exclusion points per SC point along an axis.
            This number**2 is the total number of exclusion points per
            SC point.
        excl_area : float | None, optional
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath, by default None
        exclusion_shape : tuple
            Shape of the full exclusions extent (rows, cols). Inputing this
            will speed things up considerably.
        eos_mult_baseline_cap_mw : int | float, optional
            Baseline plant capacity (MW) used to calculate economies of
            scale (EOS) multiplier from the `capital_cost_function`. EOS
            multiplier is calculated as the $-per-kW of the wind plant
            divided by the $-per-kW of a plant with this baseline
            capacity. By default, `200` (MW), which aligns the baseline
            with ATB assumptions. See here: https://tinyurl.com/y85hnu6h.
        prior_meta : pd.DataFrame | None
            Optional meta dataframe belonging to a prior run. This will only
            run the timeseries power generation step and assume that all of the
            wind plant layouts are fixed given the prior run. The meta data
            needs columns "capacity", "turbine_x_coords", and
            "turbine_y_coords".
        gid_map : None | str | dict
            Mapping of unique integer generation gids (keys) to single integer
            resource gids (values). This can be None, a pre-extracted dict, or
            a filepath to json or csv. If this is a csv, it must have the
            columns "gid" (which matches the techmap) and "gid_map" (gids to
            extract from the resource input). This is useful if you're running
            forecasted resource data (e.g., ECMWF) to complement historical
            meteorology (e.g., WTK).
        bias_correct : str | pd.DataFrame, optional
            Optional DataFrame or CSV filepath to a wind or solar
            resource bias correction table. This has columns:

                - ``gid``: GID of site (can be index name of dataframe)
                - ``method``: function name from ``rex.bias_correction`` module

            The ``gid`` field should match the true resource ``gid`` regardless
            of the optional ``gid_map`` input. Only ``windspeed`` **or**
            ``GHI`` + ``DNI`` + ``DHI`` are corrected, depending on the
            technology (wind for the former, PV or CSP for the latter). See the
            functions in the ``rex.bias_correction`` module for available
            inputs for ``method``. Any additional kwargs required for the
            requested ``method`` can be input as additional columns in the
            ``bias_correct`` table e.g., for linear bias correction functions
            you can include ``scalar`` and ``adder`` inputs as columns in the
            ``bias_correct`` table on a site-by-site basis. If ``None``, no
            corrections are applied. By default, ``None``.
        pre_loaded_data : BespokeSinglePlantData, optional
            A pre-loaded :class:`BespokeSinglePlantData` object, or
            ``None``. Can be useful to speed up execution on file
            systems with slow parallel reads.
        close : bool
            Flag to close object file handlers on exit.
        """

        logger.debug(
            "Initializing BespokeSinglePlant for gid {}...".format(gid)
        )
        logger.debug("Resource filepath: {}".format(res))
        logger.debug("Exclusion filepath: {}".format(excl))
        logger.debug("Exclusion dict: {}".format(excl_dict))
        logger.debug(
            "Bespoke objective function: {}".format(objective_function)
        )
        logger.debug("Bespoke cost function: {}".format(objective_function))
        logger.debug("Bespoke GA initialization kwargs: {}".format(ga_kwargs))
        logger.debug(
            "Bespoke EOS multiplier baseline capacity: {:,} MW".format(
                eos_mult_baseline_cap_mw
            )
        )

        if isinstance(min_spacing, str) and min_spacing.endswith("x"):
            rotor_diameter = sam_sys_inputs["wind_turbine_rotor_diameter"]
            min_spacing = float(min_spacing.strip("x")) * rotor_diameter

        if not isinstance(min_spacing, (int, float)):
            try:
                min_spacing = float(min_spacing)
            except Exception as e:
                msg = (
                    "min_spacing must be numeric but received: {}, {}".format(
                        min_spacing, type(min_spacing)
                    )
                )
                logger.error(msg)
                raise TypeError(msg) from e

        self.objective_function = objective_function
        self.capital_cost_function = capital_cost_function
        self.fixed_operating_cost_function = fixed_operating_cost_function
        self.variable_operating_cost_function = (
            variable_operating_cost_function
        )
        self.balance_of_system_cost_function = balance_of_system_cost_function
        self.min_spacing = min_spacing
        self.ga_kwargs = ga_kwargs or {}

        self._sam_sys_inputs = sam_sys_inputs
        self._out_req = list(output_request)
        self._ws_bins = ws_bins
        self._wd_bins = wd_bins
        self._baseline_cap_mw = eos_mult_baseline_cap_mw

        self._res_df = None
        self._prior_meta = prior_meta is not None
        self._meta = prior_meta
        self._wind_dist = None
        self._ws_edges = None
        self._wd_edges = None
        self._wind_plant_pd = None
        self._wind_plant_ts = None
        self._plant_optm = None
        self._gid_map = self._parse_gid_map(gid_map)
        self._bias_correct = Gen._parse_bc(bias_correct)
        self._pre_loaded_data = pre_loaded_data
        self._outputs = {}

        res = res if not isinstance(res, str) else MultiYearWindResource(res)

        self._sc_point = AggSCPoint(
            gid,
            excl,
            res,
            tm_dset,
            excl_dict=excl_dict,
            inclusion_mask=inclusion_mask,
            resolution=resolution,
            excl_area=excl_area,
            exclusion_shape=exclusion_shape,
            close=close,
        )

        self._parse_output_req()
        self._data_layers = data_layers
        self._parse_prior_run()

    def __str__(self):
        s = "BespokeSinglePlant for reV SC gid {} with resolution {}".format(
            self.sc_point.gid, self.sc_point.resolution
        )
        return s

    def __repr__(self):
        s = "BespokeSinglePlant for reV SC gid {} with resolution {}".format(
            self.sc_point.gid, self.sc_point.resolution
        )
        return s

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if type is not None:
            raise

    def _parse_output_req(self):
        """Make sure that the output request has basic important parameters
        (cf_mean, annual_energy) and process mean wind resource datasets
        (ws_mean, *_mean) if requested.
        """

        required = ("cf_mean", "annual_energy")
        for req in required:
            if req not in self._out_req:
                self._out_req.append(req)

        if "ws_mean" in self._out_req:
            self._out_req.remove("ws_mean")
            self._outputs["ws_mean"] = self.res_df["windspeed"].mean()

        for req in copy.deepcopy(self._out_req):
            if req in self.res_df:
                self._out_req.remove(req)
                for annual_ti in self.annual_time_indexes:
                    year = annual_ti.year[0]
                    mask = self.res_df.index.isin(annual_ti)
                    arr = self.res_df.loc[mask, req].values.flatten()
                    self._outputs[req + f"-{year}"] = arr

            elif req.replace("_mean", "") in self.res_df:
                self._out_req.remove(req)
                dset = req.replace("_mean", "")
                self._outputs[req] = self.res_df[dset].mean()

        if "lcoe_fcr" in self._out_req and (
            "fixed_charge_rate" not in self.original_sam_sys_inputs
        ):
            msg = (
                'User requested "lcoe_fcr" but did not input '
                '"fixed_charge_rate" in the SAM system config.'
            )
            logger.error(msg)
            raise KeyError(msg)

    def _parse_prior_run(self):
        """Parse prior bespoke wind plant optimization run meta data and make
        sure the SAM system inputs are set accordingly."""

        # {meta_column: sam_sys_input_key}
        required = {
            SupplyCurveField.CAPACITY_AC_MW: "system_capacity",
            SupplyCurveField.TURBINE_X_COORDS: "wind_farm_xCoordinates",
            SupplyCurveField.TURBINE_Y_COORDS: "wind_farm_yCoordinates",
        }

        if self._prior_meta:
            missing = [k for k in required if k not in self.meta]
            msg = (
                "Prior bespoke run meta data is missing the following "
                "required columns: {}".format(missing)
            )
            assert not any(missing), msg

            for meta_col, sam_sys_key in required.items():
                prior_value = self.meta[meta_col].values[0]
                self._sam_sys_inputs[sam_sys_key] = prior_value

            # convert reV supply curve cap in MW to SAM capacity in kW
            self._sam_sys_inputs["system_capacity"] *= 1e3

    @staticmethod
    def _parse_gid_map(gid_map):
        """Parse the gid map and return the extracted dictionary or None if not
        provided

        Parameters
        ----------
        gid_map : None | str | dict
            Mapping of unique integer generation gids (keys) to single integer
            resource gids (values). This can be None, a pre-extracted dict, or
            a filepath to json or csv. If this is a csv, it must have the
            columns "gid" (which matches the techmap) and "gid_map" (gids to
            extract from the resource input). This is useful if you're running
            forecasted resource data (e.g., ECMWF) to complement historical
            meteorology (e.g., WTK).

        Returns
        -------
        gid_map : dict | None
            Pre-extracted gid_map dictionary if provided or None if not.
        """

        if isinstance(gid_map, str):
            if gid_map.endswith(".csv"):
                gid_map = (
                    pd.read_csv(gid_map)
                    .rename(SupplyCurveField.map_to(ResourceMetaField), axis=1)
                    .to_dict()
                )
                err_msg = f"Need {ResourceMetaField.GID} in gid_map column"
                assert ResourceMetaField.GID in gid_map, err_msg
                assert "gid_map" in gid_map, 'Need "gid_map" in gid_map column'
                gid_map = {
                    gid_map[ResourceMetaField.GID][i]: gid_map["gid_map"][i]
                    for i in gid_map[ResourceMetaField.GID]
                }

            elif gid_map.endswith(".json"):
                with open(gid_map) as f:
                    gid_map = json.load(f)

        return gid_map

    def close(self):
        """Close any open file handlers via the sc point attribute. If this
        class was initialized with close=False, this will not close any
        handlers."""
        self.sc_point.close()

    def bias_correct_ws(self, ws, dset, h5_gids):
        """Bias correct windspeed data if the ``bias_correct`` input was
        provided.

        Parameters
        ----------
        ws : np.ndarray
            Windspeed data in shape (time, space)
        dset : str
            Resource dataset name e.g., "windspeed_100m", "temperature_100m",
            "pressure_100m", or something similar
        h5_gids : list | np.ndarray
            Array of integer gids (spatial indices) from the source h5 file.
            This is used to get the correct bias correction parameters from
            ``bias_correct`` table based on its ``gid`` column

        Returns
        -------
        ws : np.ndarray
            Bias corrected windspeed data in same shape as input
        """

        if self._bias_correct is not None and dset.startswith("windspeed_"):
            out = parse_bc_table(self._bias_correct, h5_gids)
            bc_fun, bc_fun_kwargs, bool_bc = out

            if bool_bc.any():
                logger.debug(
                    "Bias correcting windspeed with function {} "
                    "for h5 gids: {}".format(bc_fun, h5_gids)
                )

                bc_fun_kwargs["ws"] = ws[:, bool_bc]
                sig = signature(bc_fun)
                bc_fun_kwargs = {
                    k: v
                    for k, v in bc_fun_kwargs.items()
                    if k in sig.parameters
                }

                ws[:, bool_bc] = bc_fun(**bc_fun_kwargs)

        return ws

    def get_weighted_res_ts(self, dset):
        """Special method for calculating the exclusion-weighted mean resource
        timeseries data for the BespokeSinglePlant.

        Parameters
        ----------
        dset : str
            Resource dataset name e.g., "windspeed_100m", "temperature_100m",
            "pressure_100m", or something similar

        Returns
        -------
        data : np.ndarray
            Timeseries data of shape (n_time,) for the wind plant weighted by
            the plant inclusions mask.
        """
        gids = self.sc_point.h5_gid_set
        h5_gids = copy.deepcopy(gids)
        if self._gid_map is not None:
            h5_gids = [self._gid_map[g] for g in gids]

        if self._pre_loaded_data is None:
            data = self.sc_point.h5[dset, :, h5_gids]
        else:
            data = self._pre_loaded_data[dset, :, h5_gids]

        data = self.bias_correct_ws(data, dset, h5_gids)

        weights = np.zeros(len(gids))
        for i, gid in enumerate(gids):
            mask = self.sc_point._h5_gids == gid
            weights[i] = self.sc_point.include_mask_flat[mask].sum()

        if "float" not in str(data.dtype):
            data = data.astype("float32")

        weights /= weights.sum()
        data = data.astype(np.float32)
        data *= weights
        data = np.sum(data, axis=1)

        return data

    def get_weighted_res_dir(self):
        """Special method for calculating the exclusion-weighted mean wind
        direction for the BespokeSinglePlant

        Returns
        -------
        mean_wind_dirs : np.ndarray
            Timeseries array of winddirection data in shape (n_time,) in units
            of degrees from north.
        """

        dset = f"winddirection_{self.hub_height}m"
        gids = self.sc_point.h5_gid_set
        h5_gids = copy.deepcopy(gids)
        if self._gid_map is not None:
            h5_gids = [self._gid_map[g] for g in gids]

        if self._pre_loaded_data is None:
            dirs = self.sc_point.h5[dset, :, h5_gids]
        else:
            dirs = self._pre_loaded_data[dset, :, h5_gids]
        angles = np.radians(dirs, dtype=np.float32)

        weights = np.zeros(len(gids))
        for i, gid in enumerate(gids):
            mask = self.sc_point._h5_gids == gid
            weights[i] = self.sc_point.include_mask_flat[mask].sum()

        weights /= weights.sum()
        sin = np.sum(np.sin(angles) * weights, axis=1)
        cos = np.sum(np.cos(angles) * weights, axis=1)

        mean_wind_dirs = np.degrees(np.arctan2(sin, cos))
        mean_wind_dirs[(mean_wind_dirs < 0)] += 360

        return mean_wind_dirs

    @property
    def gid(self):
        """SC point gid for this bespoke plant.

        Returns
        -------
        int
        """
        return self.sc_point.gid

    @property
    def include_mask(self):
        """Get the supply curve point 2D inclusion mask (included is 1,
        excluded is 0)

        Returns
        -------
        np.ndarray
        """
        return self.sc_point.include_mask

    @property
    def pixel_side_length(self):
        """Get the length of a single exclusion pixel side (meters)

        Returns
        -------
        float
        """
        return np.sqrt(self.sc_point.pixel_area) * 1000.0

    @property
    def original_sam_sys_inputs(self):
        """Get the original (pre-optimized) SAM windpower system inputs.

        Returns
        -------
        dict
        """
        return self._sam_sys_inputs

    @property
    def sam_sys_inputs(self):
        """Get the SAM windpower system inputs. If the wind plant has not yet
        been optimized, this returns the initial SAM config. If the wind plant
        has been optimized using the wind_plant_pd object, this returns the
        final optimized SAM plant config.

        Returns
        -------
        dict
        """
        config = copy.deepcopy(self._sam_sys_inputs)
        if self._wind_plant_pd is None:
            return config

        layout_config = copy.deepcopy(self._wind_plant_pd.sam_sys_inputs)
        # `wind_plant_pd` PC may have PC losses applied, so keep the
        # original PC as to not double count losses here
        layout_config.pop("wind_turbine_powercurve_powerout", None)
        config.update(layout_config)

        return config

    @property
    def sc_point(self):
        """Get the reV supply curve point object.

        Returns
        -------
        AggSCPoint
        """
        return self._sc_point

    @property
    def meta(self):
        """Get the basic supply curve point meta data

        Returns
        -------
        pd.DataFrame
        """
        if self._meta is None:
            res_gids = json.dumps([int(g) for g in self.sc_point.h5_gid_set])
            gid_counts = json.dumps(
                [float(np.round(n, 1)) for n in self.sc_point.gid_counts]
            )

            self._meta = pd.DataFrame(
                {
                    "gid": self.sc_point.gid,  # needed for collection
                    SupplyCurveField.LATITUDE: self.sc_point.latitude,
                    SupplyCurveField.LONGITUDE: self.sc_point.longitude,
                    SupplyCurveField.COUNTRY: self.sc_point.country,
                    SupplyCurveField.STATE: self.sc_point.state,
                    SupplyCurveField.COUNTY: self.sc_point.county,
                    SupplyCurveField.ELEVATION: self.sc_point.elevation,
                    SupplyCurveField.TIMEZONE: self.sc_point.timezone,
                    SupplyCurveField.SC_POINT_GID: self.sc_point.sc_point_gid,
                    SupplyCurveField.SC_ROW_IND: self.sc_point.sc_row_ind,
                    SupplyCurveField.SC_COL_IND: self.sc_point.sc_col_ind,
                    SupplyCurveField.RES_GIDS: res_gids,
                    SupplyCurveField.GID_COUNTS: gid_counts,
                    SupplyCurveField.N_GIDS: self.sc_point.n_gids,
                    SupplyCurveField.OFFSHORE: self.sc_point.offshore,
                    SupplyCurveField.AREA_SQ_KM: self.sc_point.area,
                },
                index=[self.sc_point.gid],
            )

        return self._meta

    @property
    def hub_height(self):
        """Get the integer SAM system config turbine hub height (meters)

        Returns
        -------
        int
        """
        return int(self.sam_sys_inputs["wind_turbine_hub_ht"])

    @property
    def res_df(self):
        """Get the reV compliant wind resource dataframe representing the
        aggregated and included wind resource in the current reV supply curve
        point at the turbine hub height. Includes a DatetimeIndex and columns
        for temperature, pressure, windspeed, and winddirection.

        Returns
        -------
        pd.DataFrame
        """
        if self._res_df is None:
            if self._pre_loaded_data is None:
                ti = self.sc_point.h5.time_index
            else:
                ti = self._pre_loaded_data.time_index

            wd = self.get_weighted_res_dir()
            ws = self.get_weighted_res_ts(f"windspeed_{self.hub_height}m")
            temp = self.get_weighted_res_ts(f"temperature_{self.hub_height}m")
            pres = self.get_weighted_res_ts(f"pressure_{self.hub_height}m")

            # convert mbar to atm
            if np.nanmax(pres) > 1000:
                pres *= 9.86923e-6

            data = {
                "temperature": temp,
                "pressure": pres,
                "windspeed": ws,
                "winddirection": wd,
            }

            if self.sam_sys_inputs.get("en_icing_cutoff"):
                rh = self.get_weighted_res_ts("relativehumidity_2m")
                data["relativehumidity"] = rh

            self._res_df = pd.DataFrame(data, index=ti)

            if "time_index_step" in self.original_sam_sys_inputs:
                ti_step = self.original_sam_sys_inputs["time_index_step"]
                self._res_df = self._res_df.iloc[::ti_step]

        return self._res_df

    @property
    def years(self):
        """Get the sorted list of analysis years.

        Returns
        -------
        list
        """
        return sorted(list(self.res_df.index.year.unique()))

    @property
    def annual_time_indexes(self):
        """Get an ordered list of single-year time index objects that matches
        the profile outputs from the wind_plant_ts object.

        Returns
        -------
        list
        """
        tis = []
        for year in self.years:
            ti = self.res_df.index[(self.res_df.index.year == year)]
            tis.append(WindPower.ensure_res_len(ti, ti))
        return tis

    @property
    def wind_dist(self):
        """Get the wind joint probability distribution and corresonding bin
        edges

        Returns
        -------
        wind_dist : np.ndarray
            2D array probability distribution of (windspeed, winddirection)
            normalized so the sum of all values = 1.
        ws_edges : np.ndarray
            1D array of windspeed (m/s) values that set the bin edges for the
            wind probability distribution. Same len as wind_dist.shape[0] + 1
        wd_edges : np.ndarray
            1D array of winddirections (deg) values that set the bin edges
            for the wind probability dist. Same len as wind_dist.shape[1] + 1
        """
        if self._wind_dist is None:
            ws_bins = JointPD._make_bins(*self._ws_bins)
            wd_bins = JointPD._make_bins(*self._wd_bins)

            hist_out = np.histogram2d(
                self.res_df["windspeed"],
                self.res_df["winddirection"],
                bins=(ws_bins, wd_bins),
            )
            self._wind_dist, self._ws_edges, self._wd_edges = hist_out
            self._wind_dist /= self._wind_dist.sum()

        return self._wind_dist, self._ws_edges, self._wd_edges

    def initialize_wind_plant_ts(self):
        """Initialize the annual wind plant timeseries analysis object(s) using
        the annual resource data and the sam system inputs from the optimized
        plant.

        Returns
        -------
        wind_plant_ts : dict
            Annual reV.SAM.generation.WindPower object(s) keyed by year.
        """
        wind_plant_ts = {}
        for year in self.years:
            res_df = self.res_df[(self.res_df.index.year == year)]
            sam_inputs = copy.deepcopy(self.sam_sys_inputs)

            if "lcoe_fcr" in self._out_req:
                lcoe_kwargs = self.get_lcoe_kwargs()
                sam_inputs.update(lcoe_kwargs)

            i_wp = WindPower(
                res_df, self.meta, sam_inputs, output_request=self._out_req
            )
            wind_plant_ts[year] = i_wp

        return wind_plant_ts

    @property
    def wind_plant_pd(self):
        """ReV WindPowerPD compute object for plant layout optimization based
        on wind joint probability distribution

        Returns
        -------
        reV.SAM.generation.WindPowerPD
        """

        if self._wind_plant_pd is None:
            wind_dist, ws_edges, wd_edges = self.wind_dist
            self._wind_plant_pd = WindPowerPD(
                ws_edges,
                wd_edges,
                wind_dist,
                self.meta,
                self.sam_sys_inputs,
                output_request=self._out_req,
            )
        return self._wind_plant_pd

    @property
    def wind_plant_ts(self):
        """ReV WindPower compute object(s) based on wind resource timeseries
        data keyed by year

        Returns
        -------
        dict
        """
        return self._wind_plant_ts

    @property
    def plant_optimizer(self):
        """Bespoke plant turbine placement optimizer object.

        Returns
        -------
        PlaceTurbines
        """
        if self._plant_optm is None:
            # put import here to delay breaking due to special dependencies
            from reV.bespoke.place_turbines import PlaceTurbines

            self._plant_optm = PlaceTurbines(
                self.wind_plant_pd,
                self.objective_function,
                self.capital_cost_function,
                self.fixed_operating_cost_function,
                self.variable_operating_cost_function,
                self.balance_of_system_cost_function,
                self.include_mask,
                self.pixel_side_length,
                self.min_spacing)

        return self._plant_optm

    def recalc_lcoe(self):
        """Recalculate the multi-year mean LCOE based on the multi-year mean
        annual energy production (AEP)"""

        if "lcoe_fcr-means" in self.outputs:
            lcoe_kwargs = self.get_lcoe_kwargs()

            logger.debug(
                "Recalulating multi-year mean LCOE using "
                "multi-year mean AEP."
            )

            fcr = lcoe_kwargs['fixed_charge_rate']
            cc = lcoe_kwargs['capital_cost']
            foc = lcoe_kwargs['fixed_operating_cost']
            voc = lcoe_kwargs['variable_operating_cost']
            aep = self.outputs['annual_energy-means']

            my_mean_lcoe = lcoe_fcr(fcr, cc, foc, aep, voc)

            self._outputs["lcoe_fcr-means"] = my_mean_lcoe
            self._meta[SupplyCurveField.MEAN_LCOE] = my_mean_lcoe

    def get_lcoe_kwargs(self):
        """Get a namespace of arguments for calculating LCOE based on the
        bespoke optimized wind plant capacity

        Returns
        -------
        lcoe_kwargs : dict
            kwargs for the SAM lcoe model. These are based on the original
            sam_sys_inputs, normalized to the original system_capacity, and
            updated based on the bespoke optimized system_capacity, includes
            fixed_charge_rate, system_capacity (kW), capital_cost ($),
            fixed_operating_cos ($), variable_operating_cost ($/kWh),
            balance_of_system_cost ($). Data source priority: outputs,
            plant_optimizer, original_sam_sys_inputs, meta
        """

        kwargs_map = {
            "fixed_charge_rate": SupplyCurveField.FIXED_CHARGE_RATE,
            "system_capacity": SupplyCurveField.CAPACITY_AC_MW,
            "capital_cost": SupplyCurveField.BESPOKE_CAPITAL_COST,
            "fixed_operating_cost": (
                SupplyCurveField.BESPOKE_FIXED_OPERATING_COST
            ),
            "variable_operating_cost": (
                SupplyCurveField.BESPOKE_VARIABLE_OPERATING_COST
            ),
            "balance_of_system_cost": (
                SupplyCurveField.BESPOKE_BALANCE_OF_SYSTEM_COST
            ),
        }
        lcoe_kwargs = {}

        for kwarg, meta_field in kwargs_map.items():
            if kwarg in self.outputs:
                lcoe_kwargs[kwarg] = self.outputs[kwarg]
            elif getattr(self.plant_optimizer, kwarg, None) is not None:
                lcoe_kwargs[kwarg] = getattr(self.plant_optimizer, kwarg)
            elif kwarg in self.original_sam_sys_inputs:
                lcoe_kwargs[kwarg] = self.original_sam_sys_inputs[kwarg]
            elif kwarg in self.meta:
                value = float(self.meta[kwarg].values[0])
                lcoe_kwargs[kwarg] = value
            elif meta_field in self.meta:
                value = float(self.meta[meta_field].values[0])
                if meta_field == SupplyCurveField.CAPACITY_AC_MW:
                    value *= 1000  # MW to kW
                lcoe_kwargs[kwarg] = value

        missing = [k for k in kwargs_map if k not in lcoe_kwargs]
        if any(missing):
            msg = (
                "Could not find these LCOE kwargs in outputs, "
                "plant_optimizer, original_sam_sys_inputs, or meta: {}".format(
                    missing
                )
            )
            logger.error(msg)
            raise KeyError(msg)

        bos = lcoe_kwargs.pop("balance_of_system_cost")
        lcoe_kwargs["capital_cost"] = lcoe_kwargs["capital_cost"] + bos
        return lcoe_kwargs

    @classmethod
    def check_dependencies(cls):
        """Check special dependencies for bespoke"""

        missing = []
        for name in cls.DEPENDENCIES:
            try:
                import_module(name)
            except ModuleNotFoundError:
                missing.append(name)

        if any(missing):
            msg = (
                "The reV bespoke module depends on the following special "
                "dependencies that were not found in the active "
                "environment: {}".format(missing)
            )
            logger.error(msg)
            raise ModuleNotFoundError(msg)

    @staticmethod
    def _check_sys_inputs(plant1, plant2,
                          ignore=('wind_resource_model_choice',
                                  'wind_resource_data',
                                  'wind_turbine_powercurve_powerout',
                                  'adjust_hourly',
                                  'capital_cost',
                                  'fixed_operating_cost',
                                  'variable_operating_cost',
                                  'balance_of_system_cost',
                                  'base_capital_cost',
                                  'base_fixed_operating_cost',
                                  'base_variable_operating_cost')):
        """Check two reV-SAM models for matching system inputs.

        Parameters
        ----------
        plant1/plant2 : reV.SAM.generation.WindPower
            Two WindPower analysis objects to check.
        """
        bad = []
        for k, v in plant1.sam_sys_inputs.items():
            if k not in plant2.sam_sys_inputs or str(v) != str(
                plant2.sam_sys_inputs[k]
            ):
                bad.append(k)
        bad = [b for b in bad if b not in ignore]
        if any(bad):
            msg = "Inputs no longer match: {}".format(bad)
            logger.error(msg)
            raise RuntimeError(msg)

    def run_wind_plant_ts(self):
        """Run the wind plant multi-year timeseries analysis and export output
        requests to outputs property.

        Returns
        -------
        outputs : dict
            Output dictionary for the full BespokeSinglePlant object. The
            multi-year timeseries data is also exported to the
            BespokeSinglePlant.outputs property.
        """

        logger.debug(
            "Running {} years of SAM timeseries analysis for {}".format(
                len(self.years), self
            )
        )
        self._wind_plant_ts = self.initialize_wind_plant_ts()
        for year, plant in self.wind_plant_ts.items():
            self._check_sys_inputs(plant, self.wind_plant_pd)
            try:
                plant.run_gen_and_econ()
            except Exception as e:
                msg = (
                    "{} failed while trying to run SAM WindPower "
                    "timeseries analysis for {}".format(self, year)
                )
                logger.exception(msg)
                raise RuntimeError(msg) from e

            for k, v in plant.outputs.items():
                self._outputs[k + "-{}".format(year)] = v

        means = {}
        for k1, v1 in self._outputs.items():
            if isinstance(v1, Number) and parse_year(k1, option="boolean"):
                year = parse_year(k1)
                base_str = k1.replace(str(year), "")
                all_values = [
                    v2 for k2, v2 in self._outputs.items() if base_str in k2
                ]
                means[base_str + "means"] = np.mean(all_values)

        self._outputs.update(means)

        self._meta[SupplyCurveField.MEAN_RES] = self.res_df["windspeed"].mean()
        self._meta[SupplyCurveField.MEAN_CF_DC] = np.nan
        self._meta[SupplyCurveField.MEAN_CF_AC] = np.nan
        self._meta[SupplyCurveField.MEAN_LCOE] = np.nan
        self._meta[SupplyCurveField.SC_POINT_ANNUAL_ENERGY_MWH] = np.nan
        # copy dataset outputs to meta data for supply curve table summary
        if "cf_mean-means" in self.outputs:
            self._meta.loc[:, SupplyCurveField.MEAN_CF_AC] = self.outputs[
                "cf_mean-means"
            ]
        if "lcoe_fcr-means" in self.outputs:
            self._meta.loc[:, SupplyCurveField.MEAN_LCOE] = self.outputs[
                "lcoe_fcr-means"
            ]
            self.recalc_lcoe()
        if "annual_energy-means" in self.outputs:
            self._meta[SupplyCurveField.SC_POINT_ANNUAL_ENERGY_MWH] = (
                self.outputs["annual_energy-means"] / 1000
            )

        logger.debug("Timeseries analysis complete!")

        return self.outputs

    def run_plant_optimization(self):
        """Run the wind plant layout optimization and export outputs
        to outputs property.

        Returns
        -------
        outputs : dict
            Output dictionary for the full BespokeSinglePlant object. The
            layout optimization output data is also exported to the
            BespokeSinglePlant.outputs property.
        """

        logger.debug("Running plant layout optimization for {}".format(self))
        try:
            self.plant_optimizer.place_turbines(**self.ga_kwargs)
        except Exception as e:
            msg = (
                "{} failed while trying to run the "
                "turbine placement optimizer".format(self)
            )
            logger.exception(msg)
            raise RuntimeError(msg) from e

        self._outputs["full_polygons"] = self.plant_optimizer.full_polygons
        self._outputs["packing_polygons"] = (
            self.plant_optimizer.packing_polygons
        )
        system_capacity_kw = self.plant_optimizer.capacity
        self._outputs["system_capacity"] = system_capacity_kw

        txc = [int(np.round(c)) for c in self.plant_optimizer.turbine_x]
        tyc = [int(np.round(c)) for c in self.plant_optimizer.turbine_y]
        pxc = [int(np.round(c)) for c in self.plant_optimizer.x_locations]
        pyc = [int(np.round(c)) for c in self.plant_optimizer.y_locations]

        txc = json.dumps(txc)
        tyc = json.dumps(tyc)
        pxc = json.dumps(pxc)
        pyc = json.dumps(pyc)

        self._meta[SupplyCurveField.TURBINE_X_COORDS] = txc
        self._meta[SupplyCurveField.TURBINE_Y_COORDS] = tyc
        self._meta[SupplyCurveField.POSSIBLE_X_COORDS] = pxc
        self._meta[SupplyCurveField.POSSIBLE_Y_COORDS] = pyc

        self._meta[SupplyCurveField.N_TURBINES] = self.plant_optimizer.nturbs
        self._meta["avg_sl_dist_to_center_m"] = (
            self.plant_optimizer.avg_sl_dist_to_center_m
        )
        self._meta["avg_sl_dist_to_medoid_m"] = (
            self.plant_optimizer.avg_sl_dist_to_medoid_m
        )
        self._meta["nn_conn_dist_m"] = self.plant_optimizer.nn_conn_dist_m
        self._meta[SupplyCurveField.BESPOKE_AEP] = self.plant_optimizer.aep
        self._meta[SupplyCurveField.BESPOKE_OBJECTIVE] = (
            self.plant_optimizer.objective
        )
        self._meta[SupplyCurveField.BESPOKE_CAPITAL_COST] = (
            self.plant_optimizer.capital_cost
        )
        self._meta[SupplyCurveField.BESPOKE_FIXED_OPERATING_COST] = (
            self.plant_optimizer.fixed_operating_cost
        )
        self._meta[SupplyCurveField.BESPOKE_VARIABLE_OPERATING_COST] = (
            self.plant_optimizer.variable_operating_cost
        )
        self._meta[SupplyCurveField.BESPOKE_BALANCE_OF_SYSTEM_COST] = (
            self.plant_optimizer.balance_of_system_cost
        )
        self._meta[SupplyCurveField.INCLUDED_AREA] = self.plant_optimizer.area
        self._meta[SupplyCurveField.INCLUDED_AREA_CAPACITY_DENSITY] = (
            self.plant_optimizer.capacity_density
        )
        self._meta[SupplyCurveField.CONVEX_HULL_AREA] = (
            self.plant_optimizer.convex_hull_area
        )
        self._meta[SupplyCurveField.CONVEX_HULL_CAPACITY_DENSITY] = (
            self.plant_optimizer.convex_hull_capacity_density
        )
        self._meta[SupplyCurveField.FULL_CELL_CAPACITY_DENSITY] = (
            self.plant_optimizer.full_cell_capacity_density
        )

        # copy dataset outputs to meta data for supply curve table summary
        # convert SAM system capacity in kW to reV supply curve cap in MW
        capacity_ac_mw = system_capacity_kw / 1e3
        self._meta[SupplyCurveField.CAPACITY_AC_MW] = capacity_ac_mw
        self._meta[SupplyCurveField.CAPACITY_DC_MW] = np.nan

        # add required ReEDS multipliers to meta
        baseline_cost = self.plant_optimizer.capital_cost_per_kw(
            capacity_mw=self._baseline_cap_mw
        )
        eos_mult = (self.plant_optimizer.capital_cost
                    / self.plant_optimizer.capacity
                    / baseline_cost)
        reg_mult_cc = self.sam_sys_inputs.get(
            "capital_cost_multiplier", 1)
        reg_mult_foc = self.sam_sys_inputs.get(
            "fixed_operating_cost_multiplier", 1)
        reg_mult_voc = self.sam_sys_inputs.get(
            "variable_operating_cost_multiplier", 1)
        reg_mult_bos = self.sam_sys_inputs.get(
            "balance_of_system_cost_multiplier", 1)

        self._meta[SupplyCurveField.EOS_MULT] = eos_mult
        self._meta[SupplyCurveField.REG_MULT] = reg_mult_cc

        self._meta[SupplyCurveField.COST_SITE_OCC_USD_PER_AC_MW] = (
            (self.plant_optimizer.capital_cost
             + self.plant_optimizer.balance_of_system_cost)
            / capacity_ac_mw
        )
        self._meta[SupplyCurveField.COST_BASE_OCC_USD_PER_AC_MW] = (
            (self.plant_optimizer.capital_cost / eos_mult / reg_mult_cc
             + self.plant_optimizer.balance_of_system_cost / reg_mult_bos)
            / capacity_ac_mw
        )
        self._meta[SupplyCurveField.COST_SITE_FOC_USD_PER_AC_MW] = (
            self.plant_optimizer.fixed_operating_cost
            / capacity_ac_mw
        )
        self._meta[SupplyCurveField.COST_BASE_FOC_USD_PER_AC_MW] = (
            self.plant_optimizer.fixed_operating_cost
            / reg_mult_foc
            / capacity_ac_mw
        )
        self._meta[SupplyCurveField.COST_SITE_VOC_USD_PER_AC_MW] = (
            self.plant_optimizer.variable_operating_cost
            / capacity_ac_mw
        )
        self._meta[SupplyCurveField.COST_BASE_VOC_USD_PER_AC_MW] = (
            self.plant_optimizer.variable_operating_cost
            / reg_mult_voc
            / capacity_ac_mw
        )
        self._meta[SupplyCurveField.FIXED_CHARGE_RATE] = (
            self.plant_optimizer.fixed_charge_rate
        )

        logger.debug("Plant layout optimization complete!")
        return self.outputs

    def agg_data_layers(self):
        """Aggregate optional data layers if requested and save to self.meta"""
        if self._data_layers is not None:
            logger.debug(
                "Aggregating {} extra data layers.".format(
                    len(self._data_layers)
                )
            )
            point_summary = self.meta.to_dict()
            point_summary = self.sc_point.agg_data_layers(
                point_summary, self._data_layers
            )
            self._meta = pd.DataFrame(point_summary)
            logger.debug("Finished aggregating extra data layers.")

    @property
    def outputs(self):
        """Saved outputs for the single wind plant bespoke optimization.

        Returns
        -------
        dict
        """
        return self._outputs

    @classmethod
    def run(cls, *args, **kwargs):
        """Run the bespoke optimization for a single wind plant.

        Parameters
        ----------
        See the class initialization parameters.

        Returns
        -------
        bsp : dict
            Bespoke single plant outputs namespace keyed by dataset name
            including a dataset "meta" for the BespokeSinglePlant meta data.
        """

        with cls(*args, **kwargs) as bsp:
            if bsp._prior_meta:
                logger.debug(
                    "Skipping bespoke plant optimization for gid {}. "
                    "Received prior meta data for this point.".format(bsp.gid)
                )
            else:
                _ = bsp.run_plant_optimization()

            _ = bsp.run_wind_plant_ts()
            bsp.agg_data_layers()

            meta = bsp.meta
            out = bsp.outputs
            out["meta"] = meta
            for year, ti in zip(bsp.years, bsp.annual_time_indexes):
                out["time_index-{}".format(year)] = ti

        return out


class BespokeWindPlants(BaseAggregation):
    """BespokeWindPlants"""

    def __init__(self, excl_fpath, res_fpath, tm_dset, objective_function,
                 capital_cost_function, fixed_operating_cost_function,
                 variable_operating_cost_function,
                 balance_of_system_cost_function, project_points,
                 sam_files, min_spacing='5x', ga_kwargs=None,
                 output_request=('system_capacity', 'cf_mean'),
                 ws_bins=(0.0, 20.0, 5.0), wd_bins=(0.0, 360.0, 45.0),
                 excl_dict=None, area_filter_kernel='queen', min_area=None,
                 resolution=64, excl_area=None, data_layers=None,
                 pre_extract_inclusions=False, eos_mult_baseline_cap_mw=200,
                 prior_run=None, gid_map=None, bias_correct=None,
                 pre_load_data=False):
        """reV bespoke analysis class.

        Much like generation, ``reV`` bespoke analysis runs SAM
        simulations by piping in renewable energy resource data (usually
        from the WTK), loading the SAM config, and then executing the
        :py:class:`PySAM.Windpower.Windpower` compute module.
        However, unlike ``reV`` generation, bespoke analysis is
        performed on the supply-curve grid resolution, and the plant
        layout is optimized for every supply-curve point based on an
        optimization objective specified by the user. See the NREL
        publication on the bespoke methodology for more information.

        See the documentation for the ``reV`` SAM class (e.g.
        :class:`reV.SAM.generation.WindPower`,
        :class:`reV.SAM.generation.PvWattsv8`,
        :class:`reV.SAM.generation.Geothermal`, etc.) for info on the
        allowed and/or required SAM config file inputs.

        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions data HDF5 file. The exclusions HDF5
            file should contain the layers specified in `excl_dict`
            and `data_layers`. These layers may also be spread out
            across multiple HDF5 files, in which case this input should
            be a list or tuple of filepaths pointing to the files
            containing the layers. Note that each data layer must be
            uniquely defined (i.e.only appear once and in a single
            input file).
        res_fpath : str
            Unix shell style path to wind resource HDF5 file in NREL WTK
            format. Can also be a path including a wildcard input like
            ``/h5_dir/prefix*suffix`` to run bespoke on multiple years
            of resource data. Can also be an explicit list of resource
            HDF5 file paths, which themselves can contain wildcards. If
            multiple files are specified in this way, they must have the
            same coordinates but can have different time indices (i.e.
            different years). This input must be readable by
            :py:class:`rex.multi_year_resource.MultiYearWindResource`
            (i.e. the resource data conform to the
            `rex data format <https://tinyurl.com/3fy7v5kx>`_). This
            means the data file(s) must contain a 1D ``time_index``
            dataset indicating the UTC time of observation, a 1D
            ``meta`` dataset represented by a DataFrame with
            site-specific columns, and 2D resource datasets that match
            the dimensions of (time_index, meta). The time index must
            start at 00:00 of January 1st of the year under
            consideration, and its shape must be a multiple of 8760.
        tm_dset : str
            Dataset name in the `excl_fpath` file containing the
            techmap (exclusions-to-resource mapping data). This data
            layer links the supply curve GID's to the generation GID's
            that are used to evaluate the performance metrics of each
            wind plant. By default, the generation GID's are assumed to
            match the resource GID's, but this mapping can be customized
            via the `gid_map` input (see the documentation for `gid_map`
            for more details).

            .. Important:: This dataset uniquely couples the (typically
              high-resolution) exclusion layers to the (typically
              lower-resolution) resource data. Therefore, a separate
              techmap must be used for every unique combination of
              resource and exclusion coordinates.

        objective_function : str
            The objective function of the optimization written out as a
            string. This expression should compute the objective to be
            minimized during layout optimization. Variables available
            for computation are:

                - ``n_turbines``: the number of turbines
                - ``system_capacity``: wind plant capacity
                - ``aep``: annual energy production
                - ``avg_sl_dist_to_center_m``: Average straight-line
                  distance to the supply curve point center from all
                  turbine locations (in m). Useful for computing plant
                  BOS costs.
                - ``avg_sl_dist_to_medoid_m``: Average straight-line
                  distance to the medoid of all turbine locations
                  (in m). Useful for computing plant BOS costs.
                - ``nn_conn_dist_m``: Total BOS connection distance
                  using nearest-neighbor connections. This variable is
                  only available for the
                  ``balance_of_system_cost_function`` equation.
                - ``fixed_charge_rate``: user input fixed_charge_rate if
                  included as part of the sam system config.
                - ``capital_cost``: plant capital cost as evaluated
                  by `capital_cost_function`
                - ``fixed_operating_cost``: plant fixed annual operating
                  cost as evaluated by `fixed_operating_cost_function`
                - ``variable_operating_cost``: plant variable annual
                  operating cost as evaluated by
                  `variable_operating_cost_function`
                - ``balance_of_system_cost``: plant balance of system
                  cost as evaluated by `balance_of_system_cost_function`
                - ``self.wind_plant``: the SAM wind plant object,
                  through which all SAM variables can be accessed

        capital_cost_function : str
            The plant capital cost function written out as a string.
            This expression must return the total plant capital cost in
            $. This expression has access to the same variables as the
            `objective_function` argument above.
        fixed_operating_cost_function : str
            The plant annual fixed operating cost function written out
            as a string. This expression must return the fixed operating
            cost in $/year. This expression has access to the same
            variables as the `objective_function` argument above.
        variable_operating_cost_function : str
            The plant annual variable operating cost function written
            out as a string. This expression must return the variable
            operating cost in $/kWh. This expression has access to the
            same variables as the `objective_function` argument above.
            You can set this to "0" to effectively ignore variable
            operating costs.
        balance_of_system_cost_function : str
            The plant balance-of-system cost function as a string, must
            return the variable operating cost in $. Has access to the
            same variables as the objective_function. You can set this
            to "0" to effectively ignore balance-of-system costs.
        project_points : int | list | tuple | str | dict | pd.DataFrame | slice
            Input specifying which sites to process. A single integer
            representing the supply curve GID of a site may be specified
            to evaluate ``reV`` at a supply curve point. A list or tuple
            of integers (or slice) representing the supply curve GIDs of
            multiple sites can be specified to evaluate ``reV`` at
            multiple specific locations. A string pointing to a project
            points CSV file may also be specified. Typically, the CSV
            contains the following columns:

                - ``gid``: Integer specifying the supply curve GID of
                  each site.
                - ``config``: Key in the `sam_files` input dictionary
                  (see below) corresponding to the SAM configuration to
                  use for each particular site. This value can also be
                  ``None`` (or left out completely) if you specify only
                  a single SAM configuration file as the `sam_files`
                  input.

            The CSV file may also contain site-specific inputs by
            including a column named after a config keyword (e.g. a
            column called ``capital_cost`` may be included to specify a
            site-specific capital cost value for each location). Columns
            that do not correspond to a config key may also be included,
            but they will be ignored. The CSV file input can also have
            these extra, optional columns:

                - ``capital_cost_multiplier``
                - ``fixed_operating_cost_multiplier``
                - ``variable_operating_cost_multiplier``
                - ``balance_of_system_cost_multiplier``

            These particular inputs are treated as multipliers to be
            applied to the respective cost curves
            (`capital_cost_function`, `fixed_operating_cost_function`,
            `variable_operating_cost_function`, and
            `balance_of_system_cost_function`) both during and
            after the optimization. A DataFrame following the same
            guidelines as the CSV input (or a dictionary that can be
            used to initialize such a DataFrame) may be used for this
            input as well. If you would like to obtain all available
            ``reV`` supply curve points to run, you can use the
            :class:`reV.supply_curve.extent.SupplyCurveExtent` class
            like so::

                import pandas as pd
                from reV.supply_curve.extent import SupplyCurveExtent

                excl_fpath = "..."
                resolution = ...
                with SupplyCurveExtent(excl_fpath, resolution) as sc:
                    points = sc.valid_sc_points(tm_dset).tolist()
                    points = pd.DataFrame({"gid": points})
                    points["config"] = "default"  # or a list of config choices

                # Use the points directly or save them to csv for CLI usage
                points.to_csv("project_points.csv", index=False)

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
            info on the allowed and/or required SAM config file inputs.
        min_spacing : float | int | str, optional
            Minimum spacing between turbines (in meters). This input can
            also be a string like "5x", which is interpreted as 5 times
            the turbine rotor diameter. By default, ``"5x"``.
        ga_kwargs : dict, optional
            Dictionary of keyword arguments to pass to GA
            initialization. If ``None``, default initialization values
            are used. See
            :class:`~reV.bespoke.gradient_free.GeneticAlgorithm` for
            a description of the allowed keyword arguments.
            By default, ``None``.
        output_request : list | tuple, optional
            Outputs requested from the SAM windpower simulation after
            the bespoke plant layout optimization. Can be any of the
            parameters in the "Outputs" group of the PySAM module
            :py:class:`PySAM.Windpower.Windpower.Outputs`, PySAM module.
            This list can also include a select number of SAM
            config/resource parameters to include in the output:
            any key in any of the
            `output attribute JSON files <https://tinyurl.com/4bmrpe3j/>`_
            may be requested. Time-series profiles requested via this
            input are output in UTC. This input can also be used to
            request resource means like ``"ws_mean"``,
            ``"windspeed_mean"``, ``"temperature_mean"``, and
            ``"pressure_mean"``. By default,
            ``('system_capacity', 'cf_mean')``.
        ws_bins : tuple, optional
            A 3-entry tuple with ``(start, stop, step)`` for the
            windspeed binning of the wind joint probability
            distribution. The stop value is inclusive, so
            ``ws_bins=(0, 20, 5)`` would result in four bins with bin
            edges (0, 5, 10, 15, 20). By default, ``(0.0, 20.0, 5.0)``.
        wd_bins : tuple, optional
            A 3-entry tuple with ``(start, stop, step)`` for the wind
            direction binning of the wind joint probability
            distribution. The stop value is inclusive, so
            ``wd_bins=(0, 360, 90)`` would result in four bins with bin
            edges (0, 90, 180, 270, 360).
            By default, ``(0.0, 360.0, 45.0)``.
        excl_dict : dict, optional
            Dictionary of exclusion keyword arguments of the format
            ``{layer_dset_name: {kwarg: value}}``, where
            ``layer_dset_name`` is a dataset in the exclusion h5 file
            and the ``kwarg: value`` pair is a keyword argument to
            the :class:`reV.supply_curve.exclusions.LayerMask` class.
            For example::

                excl_dict = {
                    "typical_exclusion": {
                        "exclude_values": 255,
                    },
                    "another_exclusion": {
                        "exclude_values": [2, 3],
                        "weight": 0.5
                    },
                    "exclusion_with_nodata": {
                        "exclude_range": [10, 100],
                        "exclude_nodata": True,
                        "nodata_value": -1
                    },
                    "partial_setback": {
                        "use_as_weights": True
                    },
                    "height_limit": {
                        "exclude_range": [0, 200]
                    },
                    "slope": {
                        "include_range": [0, 20]
                    },
                    "developable_land": {
                        "force_include_values": 42
                    },
                    "more_developable_land": {
                        "force_include_range": [5, 10]
                    },
                    ...
                }

            Note that all the keys given in this dictionary should be
            datasets of the `excl_fpath` file. If ``None`` or empty
            dictionary, no exclusions are applied. By default, ``None``.
        area_filter_kernel : {"queen", "rook"}, optional
            Contiguous area filter method to use on final exclusions
            mask. The filters are defined as::

                # Queen:     # Rook:
                [[1,1,1],    [[0,1,0],
                 [1,1,1],     [1,1,1],
                 [1,1,1]]     [0,1,0]]

            These filters define how neighboring pixels are "connected".
            Once pixels in the final exclusion layer are connected, the
            area of each resulting cluster is computed and compared
            against the `min_area` input. Any cluster with an area
            less than `min_area` is excluded from the final mask.
            This argument has no effect if `min_area` is ``None``.
            By default, ``"queen"``.
        min_area : float, optional
            Minimum area (in km\ :sup:`2`) required to keep an isolated
            cluster of (included) land within the resulting exclusions
            mask. Any clusters of land with areas less than this value
            will be marked as exclusions. See the documentation for
            `area_filter_kernel` for an explanation of how the area of
            each land cluster is computed. If ``None``, no area
            filtering is performed. By default, ``None``.
        resolution : int, optional
            Supply Curve resolution. This value defines how many pixels
            are in a single side of a supply curve cell. For example,
            a value of ``64`` would generate a supply curve where the
            side of each supply curve cell is ``64x64`` exclusion
            pixels. By default, ``64``.
        excl_area : float, optional
            Area of a single exclusion mask pixel (in km\ :sup:`2`).
            If ``None``, this value will be inferred from the profile
            transform attribute in `excl_fpath`. By default, ``None``.
        data_layers : dict, optional
            Dictionary of aggregation data layers of the format::

                data_layers = {
                    "output_layer_name": {
                        "dset": "layer_name",
                        "method": "mean",
                        "fpath": "/path/to/data.h5"
                    },
                    "another_output_layer_name": {
                        "dset": "input_layer_name",
                        "method": "mode",
                        # optional "fpath" key omitted
                    },
                    ...
                }

            The ``"output_layer_name"`` is the column name under which
            the aggregated data will appear in the meta DataFrame of the
            output file. The ``"output_layer_name"`` does not have to
            match the ``dset`` input value. The latter should match
            the layer name in the HDF5 from which the data to aggregate
            should be pulled. The ``method`` should be one of
            ``{"mode", "mean", "min", "max", "sum", "category"}``,
            describing how the high-resolution data should be aggregated
            for each supply curve point. ``fpath`` is an optional key
            that can point to an HDF5 file containing the layer data. If
            left out, the data is assumed to exist in the file(s)
            specified by the `excl_fpath` input. If ``None``, no data
            layer aggregation is performed. By default, ``None``.
        pre_extract_inclusions : bool, optional
            Optional flag to pre-extract/compute the inclusion mask from
            the `excl_dict` input. It is typically faster to compute
            the inclusion mask on the fly with parallel workers.
            By default, ``False``.
        eos_mult_baseline_cap_mw : int | float, optional
            Baseline plant capacity (MW) used to calculate economies of
            scale (EOS) multiplier from the `capital_cost_function`. EOS
            multiplier is calculated as the $-per-kW of the wind plant
            divided by the $-per-kW of a plant with this baseline
            capacity. By default, `200` (MW), which aligns the baseline
            with ATB assumptions. See here: https://tinyurl.com/y85hnu6h.
        prior_run : str, optional
            Optional filepath to a bespoke output HDF5 file belonging to
            a prior run. If specified, this module will only run the
            timeseries power generation step and assume that all of the
            wind plant layouts are fixed from the prior run. The meta
            data of this file must contain the following columns
            (automatically satisfied if the HDF5 file was generated by
            ``reV`` bespoke):

                - ``capacity`` : Capacity of the plant, in MW.
                - ``turbine_x_coords``: A string representation of a
                  python list containing the X coordinates (in m; origin
                  of cell at bottom left) of the turbines within the
                  plant (supply curve cell).
                - ``turbine_y_coords`` : A string representation of a
                  python list containing the Y coordinates (in m; origin
                  of cell at bottom left) of the turbines within the
                  plant (supply curve cell).

            If ``None``, no previous run data is considered.
            By default, ``None``
        gid_map : str | dict, optional
            Mapping of unique integer generation gids (keys) to single
            integer resource gids (values). This enables unique
            generation gids in the project points to map to non-unique
            resource gids, which can be useful when evaluating multiple
            resource datasets in ``reV`` (e.g., forecasted ECMWF
            resource data to complement historical WTK meteorology).
            This input can be a pre-extracted dictionary or a path to a
            JSON or CSV file. If this input points to a CSV file, the
            file must have the columns ``gid`` (which matches the
            project points) and ``gid_map`` (gids to extract from the
            resource input). If ``None``, the GID values in the project
            points are assumed to match the resource GID values.
            By default, ``None``.
        bias_correct : str | pd.DataFrame, optional
            Optional DataFrame or CSV filepath to a wind or solar
            resource bias correction table. This has columns:

                - ``gid``: GID of site (can be index name of dataframe)
                - ``method``: function name from ``rex.bias_correction`` module

            The ``gid`` field should match the true resource ``gid`` regardless
            of the optional ``gid_map`` input. Only ``windspeed`` **or**
            ``GHI`` + ``DNI`` + ``DHI`` are corrected, depending on the
            technology (wind for the former, PV or CSP for the latter). See the
            functions in the ``rex.bias_correction`` module for available
            inputs for ``method``. Any additional kwargs required for the
            requested ``method`` can be input as additional columns in the
            ``bias_correct`` table e.g., for linear bias correction functions
            you can include ``scalar`` and ``adder`` inputs as columns in the
            ``bias_correct`` table on a site-by-site basis. If ``None``, no
            corrections are applied. By default, ``None``.
        pre_load_data : bool, optional
            Option to pre-load resource data. This step can be
            time-consuming up front, but it drastically reduces the
            number of parallel reads to the `res_fpath` HDF5 file(s),
            and can have a significant overall speedup on systems with
            slow parallel I/O capabilities. Pre-loaded data can use a
            significant amount of RAM, so be sure to split execution
            across many nodes (e.g. 100 nodes, 36 workers each for
            CONUS) or request large amounts of memory for a smaller
            number of nodes. By default, ``False``.
        """

        log_versions(logger)
        logger.info('Initializing BespokeWindPlants...')
        logger.info('Resource filepath: {}'.format(res_fpath))
        logger.info('Exclusion filepath: {}'.format(excl_fpath))
        logger.debug('Exclusion dict: {}'.format(excl_dict))
        logger.info('Bespoke objective function: {}'
                    .format(objective_function))
        logger.info('Bespoke capital cost function: {}'
                    .format(capital_cost_function))
        logger.info('Bespoke fixed operating cost function: {}'
                    .format(fixed_operating_cost_function))
        logger.info('Bespoke variable operating cost function: {}'
                    .format(variable_operating_cost_function))
        logger.info('Bespoke balance of system cost function: {}'
                    .format(balance_of_system_cost_function))
        logger.info('Bespoke GA initialization kwargs: {}'.format(ga_kwargs))

        logger.info(
            "Bespoke pre-extracting exclusions: {}".format(
                pre_extract_inclusions
            )
        )
        logger.info(
            "Bespoke pre-extracting resource data: {}".format(pre_load_data)
        )
        logger.info("Bespoke prior run: {}".format(prior_run))
        logger.info("Bespoke GID map: {}".format(gid_map))
        logger.info("Bespoke bias correction table: {}".format(bias_correct))

        BespokeSinglePlant.check_dependencies()

        self._project_points = self._parse_points(project_points, sam_files)

        super().__init__(
            excl_fpath,
            tm_dset,
            excl_dict=excl_dict,
            area_filter_kernel=area_filter_kernel,
            min_area=min_area,
            resolution=resolution,
            excl_area=excl_area,
            gids=self._project_points.gids,
            pre_extract_inclusions=pre_extract_inclusions,
        )

        self._res_fpath = res_fpath
        self._obj_fun = objective_function
        self._cap_cost_fun = capital_cost_function
        self._foc_fun = fixed_operating_cost_function
        self._voc_fun = variable_operating_cost_function
        self._bos_fun = balance_of_system_cost_function
        self._min_spacing = min_spacing
        self._ga_kwargs = ga_kwargs or {}
        self._output_request = SAMOutputRequest(output_request)
        self._ws_bins = ws_bins
        self._wd_bins = wd_bins
        self._data_layers = data_layers
        self._eos_mult_baseline_cap_mw = eos_mult_baseline_cap_mw
        self._prior_meta = self._parse_prior_run(prior_run)
        self._gid_map = BespokeSinglePlant._parse_gid_map(gid_map)
        self._bias_correct = Gen._parse_bc(bias_correct)
        self._outputs = {}
        self._check_files()

        self._pre_loaded_data = None
        self._pre_load_data(pre_load_data)

        self._slice_lookup = None

        logger.info(
            "Initialized BespokeWindPlants with project points: {}".format(
                self._project_points
            )
        )

    @staticmethod
    def _parse_points(points, sam_configs):
        """Parse a project points object using a project points file

        Parameters
        ----------
        points : int | slice | list | str | PointsControl | None
            Slice or list specifying project points, string pointing to a
            project points csv, or a fully instantiated PointsControl object.
            Can also be a single site integer value. Points csv should have
            `SiteDataField.GID` and 'config' column, the config maps to the
            sam_configs dict keys.
        sam_configs : dict | str | SAMConfig
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s) which map to the config column in the project points
            CSV. Values are either a JSON SAM config file or dictionary of SAM
            config inputs. Can also be a single config file path or a
            pre loaded SAMConfig object.

        Returns
        -------
        ProjectPoints : ~reV.config.project_points.ProjectPoints
            Project points object laying out the supply curve gids to
            analyze.
        """
        pc = Gen.get_pc(
            points,
            points_range=None,
            sam_configs=sam_configs,
            tech="windpower",
            sites_per_worker=1,
        )

        return pc.project_points

    @staticmethod
    def _parse_prior_run(prior_run):
        """Extract bespoke meta data from prior run and verify that the run is
        compatible with the new job specs.

        Parameters
        ----------
        prior_run : str | None
            Optional filepath to a bespoke output .h5 file belonging to a prior
            run. This will only run the timeseries power generation step and
            assume that all of the wind plant layouts are fixed given the prior
            run. The meta data of this file needs columns "capacity",
            "turbine_x_coords", and "turbine_y_coords".

        Returns
        -------
        meta : pd.DataFrame | None
            Meta data from the previous bespoke run. This includes the
            previously optimized wind farm layouts. All of the nested list
            columns will be json loaded.
        """

        meta = None

        if prior_run is not None:
            assert os.path.isfile(prior_run)
            assert prior_run.endswith(".h5")

            with Outputs(prior_run, mode="r") as f:
                meta = f.meta
                meta = meta.rename(columns=SupplyCurveField.map_from_legacy())

            # pylint: disable=no-member
            for col in meta.columns:
                val = meta[col].values[0]
                if isinstance(val, str) and val[0] == "[" and val[-1] == "]":
                    meta[col] = meta[col].apply(json.loads)

        return meta

    def _get_prior_meta(self, gid):
        """Get the meta data for a given gid from the prior run (if available)

        Parameters
        ----------
        gid : int
            SC point gid for site to pull prior meta for.

        Returns
        -------
        meta : pd.DataFrame
            Prior meta data for just the requested gid.
        """
        meta = None

        if self._prior_meta is not None:
            mask = self._prior_meta[SupplyCurveField.SC_POINT_GID] == gid
            if any(mask):
                meta = self._prior_meta[mask]

        return meta

    def _check_files(self):
        """Do a preflight check on input files"""

        paths = self._excl_fpath
        if isinstance(self._excl_fpath, str):
            paths = [self._excl_fpath]

        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "Could not find required exclusions file: " "{}".format(
                        path
                    )
                )

        with ExclusionLayers(paths) as excl:
            if self._tm_dset not in excl:
                raise FileInputError(
                    'Could not find techmap dataset "{}" '
                    "in the exclusions file(s): {}".format(
                        self._tm_dset, paths
                    )
                )

        # just check that this file exists, cannot check res_fpath if *glob
        with MultiYearWindResource(self._res_fpath) as f:
            assert any(f.dsets)

    def _pre_load_data(self, pre_load_data):
        """Pre-load resource data, if requested."""
        if not pre_load_data:
            return

        sc_gid_to_hh = {
            gid: self._hh_for_sc_gid(gid)
            for gid in self._project_points.df[ResourceMetaField.GID]
        }

        with ExclusionLayers(self._excl_fpath) as excl:
            tm = excl[self._tm_dset]

        scp_kwargs = {"shape": self.shape, "resolution": self._resolution}
        slices = {
            gid: SupplyCurvePoint.get_agg_slices(gid=gid, **scp_kwargs)
            for gid in self._project_points.df[ResourceMetaField.GID]
        }

        sc_gid_to_res_gid = {
            gid: sorted(set(tm[slx, sly].flatten()))
            for gid, (slx, sly) in slices.items()
        }

        for sc_gid, res_gids in sc_gid_to_res_gid.items():
            if res_gids[0] < 0:
                sc_gid_to_res_gid[sc_gid] = res_gids[1:]

        if self._gid_map is not None:
            for sc_gid, res_gids in sc_gid_to_res_gid.items():
                sc_gid_to_res_gid[sc_gid] = sorted(
                    self._gid_map[g] for g in res_gids
                )

        logger.info("Pre-loading resource data for Bespoke run... ")
        self._pre_loaded_data = BespokeMultiPlantData(
            self._res_fpath,
            sc_gid_to_hh,
            sc_gid_to_res_gid,
            pre_load_humidity=self._project_points.sam_config_obj.icing,
        )

    def _hh_for_sc_gid(self, sc_gid):
        """Fetch the hh for a given sc_gid"""
        config = self.sam_sys_inputs_with_site_data(sc_gid)
        return int(config["wind_turbine_hub_ht"])

    def _pre_loaded_data_for_sc_gid(self, sc_gid):
        """Pre-load data for a given SC GID, if requested."""
        if self._pre_loaded_data is None:
            return None

        return self._pre_loaded_data.get_preloaded_data_for_gid(sc_gid)

    def _get_bc_for_gid(self, gid):
        """Get the bias correction table trimmed down just for the resource
        pixels corresponding to a single supply curve GID. This can help
        prevent excess memory usage when doing complex bias correction
        distributed to parallel workers.

        Parameters
        ----------
        gid : int
            SC point gid for site to pull bias correction data for

        Returns
        -------
        out : pd.DataFrame | None
            If bias_correct was input, this is just the rows from the larger
            bias correction table that correspond to the SC point gid
        """
        out = self._bias_correct

        if self._bias_correct is not None:
            h5_gids = []
            try:
                scp_kwargs = dict(
                    gid=gid,
                    excl=self._excl_fpath,
                    tm_dset=self._tm_dset,
                    resolution=self._resolution,
                )
                with SupplyCurvePoint(**scp_kwargs) as scp:
                    h5_gids = scp.h5_gid_set
            except EmptySupplyCurvePointError:
                pass

            if self._gid_map is not None:
                h5_gids = [self._gid_map[g] for g in h5_gids]

            mask = self._bias_correct.index.isin(h5_gids)
            out = self._bias_correct[mask]

        return out

    @property
    def outputs(self):
        """Saved outputs for the multi wind plant bespoke optimization. Keys
        are reV supply curve gids and values are BespokeSinglePlant.outputs
        dictionaries.

        Returns
        -------
        dict
        """
        return self._outputs

    @property
    def completed_gids(self):
        """Get a sorted list of completed BespokeSinglePlant gids

        Returns
        -------
        list
        """
        return sorted(list(self.outputs.keys()))

    @property
    def meta(self):
        """Meta data for all completed BespokeSinglePlant objects.

        Returns
        -------
        pd.DataFrame
        """
        meta = [self.outputs[g]["meta"] for g in self.completed_gids]
        if len(self.completed_gids) > 1:
            meta = pd.concat(meta, axis=0)
        else:
            meta = meta[0]
        return meta

    @property
    def slice_lookup(self):
        """Dict | None: Lookup mapping sc_point_gid to exclusion slice."""
        if self._slice_lookup is None and self._inclusion_mask is not None:
            with SupplyCurveExtent(
                self._excl_fpath, resolution=self._resolution
            ) as sc:
                assert self.shape == self._inclusion_mask.shape
                self._slice_lookup = sc.get_slice_lookup(self.gids)

        return self._slice_lookup

    def sam_sys_inputs_with_site_data(self, gid):
        """Update the sam_sys_inputs with site data for the given GID.

        Site data is extracted from the project points DataFrame. Every
        column in the project DataFrame becomes a key in the site_data
        output dictionary.

        Parameters
        ----------
        gid : int
            SC point gid for site to pull site data for.

        Returns
        -------
        dictionary : dict
            SAM system config with extra keys from the project points
            DataFrame.
        """

        gid_idx = self._project_points.index(gid)
        site_data = self._project_points.df.iloc[gid_idx]

        site_sys_inputs = self._project_points[gid][1]
        site_sys_inputs.update(
            {
                k: v
                for k, v in site_data.to_dict().items()
                if not (isinstance(v, float) and np.isnan(v))
            }
        )
        return site_sys_inputs

    def _init_fout(self, out_fpath, sample):
        """Initialize the bespoke output h5 file with meta and time index dsets

        Parameters
        ----------
        out_fpath : str
            Full filepath to an output .h5 file to save Bespoke data to. The
            parent directories will be created if they do not already exist.
        sample : dict
            A single sample BespokeSinglePlant output dict that has been run
            and has output data.
        """
        out_dir = os.path.dirname(out_fpath)
        if not os.path.exists(out_dir):
            create_dirs(out_dir)

        with Outputs(out_fpath, mode="w") as f:
            f._set_meta("meta", self.meta, attrs={})
            ti_dsets = [
                d for d in sample.keys() if d.startswith("time_index-")
            ]
            for dset in ti_dsets:
                f._set_time_index(dset, sample[dset], attrs={})
                f._set_time_index("time_index", sample[dset], attrs={})

    def _collect_out_arr(self, dset, sample):
        """Collect single-plant data arrays into complete arrays with data from
        all BespokeSinglePlant objects.

        Parameters
        ----------
        dset : str
            Dataset to collect, this should be an output dataset present in
            BespokeSinglePlant.outputs
        sample : dict
            A single sample BespokeSinglePlant output dict that has been run
            and has output data.

        Returns
        -------
        full_arr : np.ndarray
            Full data array either 1D for scalar data or 2D for timeseries
            data (n_time, n_plant) for all BespokeSinglePlant objects
        """

        single_arr = sample[dset]

        if isinstance(single_arr, Number):
            shape = (len(self.completed_gids),)
            sample_num = single_arr
        elif isinstance(single_arr, (list, tuple, np.ndarray)):
            shape = (len(single_arr), len(self.completed_gids))
            sample_num = single_arr[0]
        else:
            msg = 'Not writing dataset "{}" of type "{}" to disk.'.format(
                dset, type(single_arr)
            )
            logger.info(msg)
            return None

        if isinstance(sample_num, float):
            dtype = np.float32
        else:
            dtype = type(sample_num)
        full_arr = np.zeros(shape, dtype=dtype)

        # collect data from all wind plants
        logger.info(
            'Collecting dataset "{}" with final shape {}'.format(dset, shape)
        )
        for i, gid in enumerate(self.completed_gids):
            if len(full_arr.shape) == 1:
                full_arr[i] = self.outputs[gid][dset]
            else:
                full_arr[:, i] = self.outputs[gid][dset]

        return full_arr

    def save_outputs(self, out_fpath):
        """Save Bespoke Wind Plant optimization outputs to disk.

        Parameters
        ----------
        out_fpath : str
            Full filepath to an output .h5 file to save Bespoke data to. The
            parent directories will be created if they do not already exist.

        Returns
        -------
        out_fpath : str
            Full filepath to desired .h5 output file, the .h5 extension has
            been added if it was not already present.
        """
        if not out_fpath.endswith(".h5"):
            out_fpath += ".h5"

        if ModuleName.BESPOKE not in out_fpath:
            extension_with_module = "_{}.h5".format(ModuleName.BESPOKE)
            out_fpath = out_fpath.replace(".h5", extension_with_module)

        if not self.completed_gids:
            msg = (
                "No output data found! It is likely that all requested "
                "points are excluded."
            )
            logger.warning(msg)
            warn(msg)
            return out_fpath

        sample = self.outputs[self.completed_gids[0]]
        self._init_fout(out_fpath, sample)

        dsets = [
            d
            for d in sample.keys()
            if not d.startswith("time_index-") and d != "meta"
        ]
        with Outputs(out_fpath, mode="a") as f:
            for dset in dsets:
                full_arr = self._collect_out_arr(dset, sample)
                if full_arr is not None:
                    dset_no_year = dset
                    if parse_year(dset, option="boolean"):
                        year = parse_year(dset)
                        dset_no_year = dset.replace("-{}".format(year), "")

                    attrs = BespokeSinglePlant.OUT_ATTRS.get(dset_no_year, {})
                    attrs = copy.deepcopy(attrs)
                    dtype = attrs.pop("dtype", np.float32)
                    chunks = attrs.pop("chunks", None)
                    try:
                        f.write_dataset(
                            dset, full_arr, dtype, chunks=chunks, attrs=attrs
                        )
                    except Exception as e:
                        msg = 'Failed to write "{}" to disk.'.format(dset)
                        logger.exception(msg)
                        raise OSError(msg) from e

        logger.info("Saved output data to: {}".format(out_fpath))
        return out_fpath

    # pylint: disable=arguments-renamed
    @classmethod
    def run_serial(cls, excl_fpath, res_fpath, tm_dset,
                   sam_sys_inputs, objective_function,
                   capital_cost_function,
                   fixed_operating_cost_function,
                   variable_operating_cost_function,
                   balance_of_system_cost_function, min_spacing='5x',
                   ga_kwargs=None,
                   output_request=('system_capacity', 'cf_mean'),
                   ws_bins=(0.0, 20.0, 5.0), wd_bins=(0.0, 360.0, 45.0),
                   excl_dict=None, inclusion_mask=None,
                   area_filter_kernel='queen', min_area=None,
                   resolution=64, excl_area=0.0081, data_layers=None,
                   gids=None, exclusion_shape=None, slice_lookup=None,
                   eos_mult_baseline_cap_mw=200, prior_meta=None,
                   gid_map=None, bias_correct=None, pre_loaded_data=None):
        """
        Standalone serial method to run bespoke optimization.
        See BespokeWindPlants docstring for parameter description.

        This method can only take a single sam_sys_inputs... For a spatially
        variant gid-to-config mapping, see the BespokeWindPlants class methods.

        Returns
        -------
        out : dict
            Bespoke outputs keyed by sc point gid
        """

        out = {}
        with SupplyCurveExtent(excl_fpath, resolution=resolution) as sc:
            if gids is None:
                gids = sc.valid_sc_points(tm_dset)
            elif np.issubdtype(type(gids), np.number):
                gids = [gids]
            if slice_lookup is None:
                slice_lookup = sc.get_slice_lookup(gids)
            if exclusion_shape is None:
                exclusion_shape = sc.exclusions.shape

        cls._check_inclusion_mask(inclusion_mask, gids, exclusion_shape)

        # pre-extract handlers so they are not repeatedly initialized
        file_kwargs = {
            "excl_dict": excl_dict,
            "area_filter_kernel": area_filter_kernel,
            "min_area": min_area,
            "h5_handler": MultiYearWindResource,
        }

        with AggFileHandler(excl_fpath, res_fpath, **file_kwargs) as fh:
            n_finished = 0
            for gid in gids:
                gid_inclusions = cls._get_gid_inclusion_mask(
                    inclusion_mask, gid, slice_lookup, resolution=resolution
                )
                try:
                    bsp_plant_out = BespokeSinglePlant.run(
                        gid,
                        fh.exclusions,
                        fh.h5,
                        tm_dset,
                        sam_sys_inputs,
                        objective_function,
                        capital_cost_function,
                        fixed_operating_cost_function,
                        variable_operating_cost_function,
                        balance_of_system_cost_function,
                        min_spacing=min_spacing,
                        ga_kwargs=ga_kwargs,
                        output_request=output_request,
                        ws_bins=ws_bins,
                        wd_bins=wd_bins,
                        excl_dict=excl_dict,
                        inclusion_mask=gid_inclusions,
                        resolution=resolution,
                        excl_area=excl_area,
                        data_layers=data_layers,
                        exclusion_shape=exclusion_shape,
                        eos_mult_baseline_cap_mw=eos_mult_baseline_cap_mw,
                        prior_meta=prior_meta,
                        gid_map=gid_map,
                        bias_correct=bias_correct,
                        pre_loaded_data=pre_loaded_data,
                        close=False,
                    )

                except EmptySupplyCurvePointError:
                    logger.debug(
                        "SC gid {} is fully excluded or does not "
                        "have any valid source data!".format(gid)
                    )
                except Exception as e:
                    msg = "SC gid {} failed!".format(gid)
                    logger.exception(msg)
                    raise RuntimeError(msg) from e
                else:
                    n_finished += 1
                    logger.debug(
                        "Serial bespoke: "
                        "{} out of {} points complete".format(
                            n_finished, len(gids)
                        )
                    )
                    log_mem(logger)
                    out[gid] = bsp_plant_out

        return out

    def run_parallel(self, max_workers=None):
        """Run the bespoke optimization for many supply curve points in
        parallel.

        Parameters
        ----------
        max_workers : int | None, optional
            Number of cores to run summary on. None is all
            available cpus, by default None

        Returns
        -------
        out : dict
            Bespoke outputs keyed by sc point gid
        """

        logger.info(
            "Running bespoke optimization for points {} through {} "
            "at a resolution of {} on {} cores.".format(
                self.gids[0], self.gids[-1], self._resolution, max_workers
            )
        )

        futures = []
        out = {}
        n_finished = 0
        loggers = [__name__, "reV.supply_curve.point_summary", "reV"]
        with SpawnProcessPool(max_workers=max_workers, loggers=loggers) as exe:
            # iterate through split executions, submitting each to worker
            for gid in self.gids:
                # submit executions and append to futures list
                gid_incl_mask = None
                if self._inclusion_mask is not None:
                    rs, cs = self.slice_lookup[gid]
                    gid_incl_mask = self._inclusion_mask[rs, cs]

                futures.append(exe.submit(
                    self.run_serial,
                    self._excl_fpath,
                    self._res_fpath,
                    self._tm_dset,
                    self.sam_sys_inputs_with_site_data(gid),
                    self._obj_fun,
                    self._cap_cost_fun,
                    self._foc_fun,
                    self._voc_fun,
                    self._bos_fun,
                    self._min_spacing,
                    ga_kwargs=self._ga_kwargs,
                    output_request=self._output_request,
                    ws_bins=self._ws_bins,
                    wd_bins=self._wd_bins,
                    excl_dict=self._excl_dict,
                    inclusion_mask=gid_incl_mask,
                    area_filter_kernel=self._area_filter_kernel,
                    min_area=self._min_area,
                    resolution=self._resolution,
                    excl_area=self._excl_area,
                    data_layers=self._data_layers,
                    gids=gid,
                    exclusion_shape=self.shape,
                    slice_lookup=copy.deepcopy(self.slice_lookup),
                    eos_mult_baseline_cap_mw=self._eos_mult_baseline_cap_mw,
                    prior_meta=self._get_prior_meta(gid),
                    gid_map=self._gid_map,
                    bias_correct=self._get_bc_for_gid(gid),
                    pre_loaded_data=self._pre_loaded_data_for_sc_gid(gid)))

            # gather results
            for future in as_completed(futures):
                n_finished += 1
                out.update(future.result())
                if n_finished % 10 == 0:
                    mem = psutil.virtual_memory()
                    logger.info(
                        "Parallel bespoke futures collected: "
                        "{} out of {}. Memory usage is {:.3f} GB out "
                        "of {:.3f} GB ({:.2f}% utilized).".format(
                            n_finished,
                            len(futures),
                            mem.used / 1e9,
                            mem.total / 1e9,
                            100 * mem.used / mem.total,
                        )
                    )

        return out

    def run(self, out_fpath=None, max_workers=None):
        """Run the bespoke wind plant optimization in serial or parallel.

        Parameters
        ----------
        out_fpath : str, optional
            Path to output file. If ``None``, no output file will
            be written. If the filepath is specified but the module name
            (bespoke) is not included, the module name will get added to
            the output file name. By default, ``None``.
        max_workers : int, optional
            Number of local workers to run on. If ``None``, uses all
            available cores (typically 36). By default, ``None``.

        Returns
        -------
        str | None
            Path to output HDF5 file, or ``None`` if results were not
            written to disk.
        """

        # parallel job distribution test.
        if self._obj_fun == "test":
            return True

        if max_workers == 1:
            slice_lookup = copy.deepcopy(self.slice_lookup)
            for gid in self.gids:
                gid_incl_mask = None
                if self._inclusion_mask is not None:
                    rs, cs = slice_lookup[gid]
                    gid_incl_mask = self._inclusion_mask[rs, cs]

                sam_inputs = self.sam_sys_inputs_with_site_data(gid)
                prior_meta = self._get_prior_meta(gid)
                pre_loaded_data = self._pre_loaded_data_for_sc_gid(gid)
                afk = self._area_filter_kernel
                i_bc = self._get_bc_for_gid(gid)
                ebc = self._eos_mult_baseline_cap_mw

                si = self.run_serial(self._excl_fpath,
                                     self._res_fpath,
                                     self._tm_dset,
                                     sam_inputs,
                                     self._obj_fun,
                                     self._cap_cost_fun,
                                     self._foc_fun,
                                     self._voc_fun,
                                     self._bos_fun,
                                     min_spacing=self._min_spacing,
                                     ga_kwargs=self._ga_kwargs,
                                     output_request=self._output_request,
                                     ws_bins=self._ws_bins,
                                     wd_bins=self._wd_bins,
                                     excl_dict=self._excl_dict,
                                     inclusion_mask=gid_incl_mask,
                                     area_filter_kernel=afk,
                                     min_area=self._min_area,
                                     resolution=self._resolution,
                                     excl_area=self._excl_area,
                                     data_layers=self._data_layers,
                                     slice_lookup=slice_lookup,
                                     eos_mult_baseline_cap_mw=ebc,
                                     prior_meta=prior_meta,
                                     gid_map=self._gid_map,
                                     bias_correct=i_bc,
                                     gids=gid,
                                     pre_loaded_data=pre_loaded_data)
                self._outputs.update(si)
        else:
            self._outputs = self.run_parallel(max_workers=max_workers)

        if out_fpath is not None:
            out_fpath = self.save_outputs(out_fpath)

        return out_fpath
