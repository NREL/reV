# -*- coding: utf-8 -*-
"""
reV bespoke wind plant analysis tools
"""
# TODO update docstring
# TODO check on outputs
import time
import logging
import copy
import pandas as pd
import numpy as np
import os
import json
import psutil
from importlib import import_module
from numbers import Number
from concurrent.futures import as_completed
from warnings import warn

from reV.config.output_request import SAMOutputRequest
from reV.generation.generation import Gen
from reV.SAM.generation import WindPower, WindPowerPD
from reV.econ.utilities import lcoe_fcr
from reV.handlers.outputs import Outputs
from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.extent import SupplyCurveExtent
from reV.supply_curve.points import AggregationSupplyCurvePoint as AggSCPoint
from reV.supply_curve.points import SupplyCurvePoint
from reV.supply_curve.aggregation import BaseAggregation, AggFileHandler
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      FileInputError)
from reV.utilities import log_versions, ModuleName

from rex.joint_pd.joint_pd import JointPD
from rex.renewable_resource import WindResource
from rex.multi_year_resource import MultiYearWindResource
from rex.utilities.loggers import log_mem, create_dirs
from rex.utilities.utilities import parse_year
from rex.utilities.execution import SpawnProcessPool

logger = logging.getLogger(__name__)


class BespokeMultiPlantData:
    """Multi-plant preloaded data.

    This object is intended to facilitate the use of pre-loaded data for
    running :class:`BespokeWindPlants` on systems with slow parallel
    reads to a single HDF5 file.
    """

    def __init__(self, res_fpath, sc_gid_to_hh, sc_gid_to_res_gid):
        """Initialize BespokeMultiPlantData

        Parameters
        ----------
        res_fpath : str
            Path to resource h5 file.
        sc_gid_to_hh : dict
            Dictionary mapping SC GID values to hub-heights. Data for
            each SC GID will be pulled for the corresponding hub-height
            given in this dictionary.
        sc_gid_to_res_gid : dict
            Dictionary mapping SC GID values to an iterable oif resource
            GID values. Resource GID values should correspond to GID
            values in teh HDF5 file, so any GID map must be applied
            before initializing :class`BespokeMultiPlantData`.
        """
        self.res_fpath = res_fpath
        self.sc_gid_to_hh = sc_gid_to_hh
        self.sc_gid_to_res_gid = sc_gid_to_res_gid
        self.hh_to_res_gids = {}
        self._wind_dirs = None
        self._wind_speeds = None
        self._temps = None
        self._pressures = None
        self._time_index = None
        self._pre_load_data()

    def _pre_load_data(self):
        """Pre-load the resource data. """

        for sc_gid, gids in self.sc_gid_to_res_gid.items():
            hh = self.sc_gid_to_hh[sc_gid]
            self.hh_to_res_gids.setdefault(hh, set()).update(gids)

        self.hh_to_res_gids = {hh: sorted(gids)
                               for hh, gids in self.hh_to_res_gids.items()}

        start_time = time.time()
        if '*' in self.res_fpath:
            handler = MultiYearWindResource
        else:
            handler = WindResource

        with handler(self.res_fpath) as res:
            self._wind_dirs = {hh: res[f"winddirection_{hh}m", :, gids]
                               for hh, gids in self.hh_to_res_gids.items()}
            self._wind_speeds = {hh: res[f"windspeed_{hh}m", :, gids]
                                 for hh, gids in self.hh_to_res_gids.items()}
            self._temps = {hh: res[f"temperature_{hh}m", :, gids]
                           for hh, gids in self.hh_to_res_gids.items()}
            self._pressures = {hh: res[f"pressure_{hh}m", :, gids]
                               for hh, gids in self.hh_to_res_gids.items()}
            self._time_index = res.time_index

        logger.debug(f"Data took {(time.time() - start_time) / 60:.2f} "
                     f"min to load")

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
        return BespokeSinglePlantData(sc_point_res_gids,
                                      self._wind_dirs[hh][:, data_inds],
                                      self._wind_speeds[hh][:, data_inds],
                                      self._temps[hh][:, data_inds],
                                      self._pressures[hh][:, data_inds],
                                      self._time_index)


class BespokeSinglePlantData:
    """Single-plant preloaded data.

    This object is intended to facilitate the use of pre-loaded data for
    running :class:`BespokeSinglePlant` on systems with slow parallel
    reads to a single HDF5 file.
    """

    def __init__(self, data_inds, wind_dirs, wind_speeds, temps, pressures,
                 time_index):
        """Initialize BespokeSinglePlantData

        Parameters
        ----------
        data_inds : 1D np.array
            Array of res GIDs. This array should be the same length as
            the second dimension of `wind_dirs`, `wind_speeds`, `temps`,
            and `pressures`. The GID value of data_inds[0] should
            correspond to the `wind_dirs[:, 0]` data, etc.
        wind_dirs, wind_speeds, temps, pressures : 2D np.array
            Array of wind directions, wind speeds, temperatures, and
            pressures, respectively. Dimensions should be correspond to
            [time, location]. See documentation for `data_inds` for
            required spatial mapping of GID values.
        time_index : 1D np.array
            Time index array corresponding to the temporal dimension of
            the 2D data. Will be exposed directly to user.

        """
        self.data_inds = data_inds
        self.wind_dirs = wind_dirs
        self.wind_speeds = wind_speeds
        self.temps = temps
        self.pressures = pressures
        self.time_index = time_index

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
        msg = f"Unknown dataset name: {dset_name!r}"
        logger.error(msg)
        raise ValueError(msg)


class BespokeSinglePlant:
    """Framework for analyzing and optimized a wind plant layout specific to
    the local wind resource and exclusions for a single reV supply curve point.
    """

    DEPENDENCIES = ('shapely',)
    OUT_ATTRS = copy.deepcopy(Gen.OUT_ATTRS)

    def __init__(self, gid, excl, res, tm_dset, sam_sys_inputs,
                 objective_function, capital_cost_function,
                 fixed_operating_cost_function,
                 variable_operating_cost_function,
                 min_spacing='5x', wake_loss_multiplier=1, ga_kwargs=None,
                 output_request=('system_capacity', 'cf_mean'),
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

                - n_turbines: the number of turbines
                - system_capacity: wind plant capacity
                - aep: annual energy production
                - fixed_charge_rate: user input fixed_charge_rate if included
                  as part of the sam system config.
                - self.wind_plant: the SAM wind plant object, through which
                  all SAM variables can be accessed
                - capital_cost: plant capital cost as evaluated
                  by `capital_cost_function`
                - fixed_operating_cost: plant fixed annual operating cost as
                  evaluated by `fixed_operating_cost_function`
                - variable_operating_cost: plant variable annual operating cost
                  as evaluated by `variable_operating_cost_function`

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
            variables as the objective_function.
        min_spacing : float | int | str
            Minimum spacing between turbines in meters. Can also be a string
            like "5x" (default) which is interpreted as 5 times the turbine
            rotor diameter.
        wake_loss_multiplier : float, optional
            A multiplier used to scale the annual energy lost due to
            wake losses.
            .. WARNING:: This multiplier will ONLY be applied during the
            optimization process and will NOT be come through in output
            values such as the hourly profiles,
            aep, any of the cost functions, or even the output objective.
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
        bias_correct : str | pd.DataFrame | None
            Optional DataFrame or csv filepath to a wind bias correction table.
            This has columns: gid (can be index name), adder, scalar. If both
            adder and scalar are present, the wind is corrected by
            (res*scalar)+adder. If either is not present, scalar defaults to 1
            and adder to 0. Only windspeed is corrected. Note that if gid_map
            is provided, the bias_correct gid corresponds to the actual
            resource data gid and not the techmap gid.
        pre_loaded_data : BespokeSinglePlantData, optional
            A pre-loaded :class:`BespokeSinglePlantData` object, or
            ``None``. Can be useful to speed up execution on file
            systems with slow parallel reads.
        close : bool
            Flag to close object file handlers on exit.
        """

        logger.debug('Initializing BespokeSinglePlant for gid {}...'
                     .format(gid))
        logger.debug('Resource filepath: {}'.format(res))
        logger.debug('Exclusion filepath: {}'.format(excl))
        logger.debug('Exclusion dict: {}'.format(excl_dict))
        logger.debug('Bespoke objective function: {}'
                     .format(objective_function))
        logger.debug('Bespoke cost function: {}'.format(objective_function))
        logger.debug('Bespoke wake loss multiplier: {}'
                     .format(wake_loss_multiplier))
        logger.debug('Bespoke GA initialization kwargs: {}'.format(ga_kwargs))
        logger.debug('Bespoke EOS multiplier baseline capacity: {:,} MW'
                     .format(eos_mult_baseline_cap_mw))

        if isinstance(min_spacing, str) and min_spacing.endswith('x'):
            rotor_diameter = sam_sys_inputs["wind_turbine_rotor_diameter"]
            min_spacing = float(min_spacing.strip('x')) * rotor_diameter

        if not isinstance(min_spacing, (int, float)):
            try:
                min_spacing = float(min_spacing)
            except Exception as e:
                msg = ('min_spacing must be numeric but received: {}, {}'
                       .format(min_spacing, type(min_spacing)))
                logger.error(msg)
                raise TypeError(msg) from e

        self.objective_function = objective_function
        self.capital_cost_function = capital_cost_function
        self.fixed_operating_cost_function = fixed_operating_cost_function
        self.variable_operating_cost_function = \
            variable_operating_cost_function
        self.min_spacing = min_spacing
        self.wake_loss_multiplier = wake_loss_multiplier
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

        Handler = self.get_wind_handler(res)
        res = res if not isinstance(res, str) else Handler(res)

        self._sc_point = AggSCPoint(gid, excl, res, tm_dset,
                                    excl_dict=excl_dict,
                                    inclusion_mask=inclusion_mask,
                                    resolution=resolution,
                                    excl_area=excl_area,
                                    exclusion_shape=exclusion_shape,
                                    close=close)

        self._parse_output_req()
        self._data_layers = data_layers
        self._parse_prior_run()

    def __str__(self):
        s = ('BespokeSinglePlant for reV SC gid {} with resolution {}'
             .format(self.sc_point.gid, self.sc_point.resolution))
        return s

    def __repr__(self):
        s = ('BespokeSinglePlant for reV SC gid {} with resolution {}'
             .format(self.sc_point.gid, self.sc_point.resolution))
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

        required = ('cf_mean', 'annual_energy')
        for req in required:
            if req not in self._out_req:
                self._out_req.append(req)

        if 'ws_mean' in self._out_req:
            self._out_req.remove('ws_mean')
            self._outputs['ws_mean'] = self.res_df['windspeed'].mean()

        for req in copy.deepcopy(self._out_req):
            if req in self.res_df:
                self._out_req.remove(req)
                for annual_ti in self.annual_time_indexes:
                    year = annual_ti.year[0]
                    mask = self.res_df.index.isin(annual_ti)
                    arr = self.res_df.loc[mask, req].values.flatten()
                    self._outputs[req + f'-{year}'] = arr

            elif req.replace('_mean', '') in self.res_df:
                self._out_req.remove(req)
                dset = req.replace('_mean', '')
                self._outputs[req] = self.res_df[dset].mean()

        if ('lcoe_fcr' in self._out_req
                and 'fixed_charge_rate' not in self.original_sam_sys_inputs):
            msg = ('User requested "lcoe_fcr" but did not input '
                   '"fixed_charge_rate" in the SAM system config.')
            logger.error(msg)
            raise KeyError(msg)

    def _parse_prior_run(self):
        """Parse prior bespoke wind plant optimization run meta data and make
        sure the SAM system inputs are set accordingly."""

        # {meta_column: sam_sys_input_key}
        required = {'capacity': 'system_capacity',
                    'turbine_x_coords': 'wind_farm_xCoordinates',
                    'turbine_y_coords': 'wind_farm_yCoordinates'}

        if self._prior_meta:
            missing = [k for k in required if k not in self.meta]
            msg = ('Prior bespoke run meta data is missing the following '
                   'required columns: {}'.format(missing))
            assert not any(missing), msg

            for meta_col, sam_sys_key in required.items():
                prior_value = self.meta[meta_col].values[0]
                self._sam_sys_inputs[sam_sys_key] = prior_value

            # convert reV supply curve cap in MW to SAM capacity in kW
            self._sam_sys_inputs['system_capacity'] *= 1e3

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
            if gid_map.endswith('.csv'):
                gid_map = pd.read_csv(gid_map).to_dict()
                assert 'gid' in gid_map, 'Need "gid" in gid_map column'
                assert 'gid_map' in gid_map, 'Need "gid_map" in gid_map column'
                gid_map = {gid_map['gid'][i]: gid_map['gid_map'][i]
                           for i in gid_map['gid'].keys()}

            elif gid_map.endswith('.json'):
                with open(gid_map, 'r') as f:
                    gid_map = json.load(f)

        return gid_map

    def close(self):
        """Close any open file handlers via the sc point attribute. If this
        class was initialized with close=False, this will not close any
        handlers."""
        self.sc_point.close()

    def get_weighted_res_ts(self, dset):
        """Special method for calculating the exclusion-weighted mean resource
        timeseries data for the BespokeSinglePlant.

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

        if self._bias_correct is not None and dset.startswith('windspeed_'):
            missing = [g for g in h5_gids if g not in self._bias_correct.index]
            for missing_gid in missing:
                self._bias_correct.loc[missing_gid, 'scalar'] = 1
                self._bias_correct.loc[missing_gid, 'adder'] = 0

            scalar = self._bias_correct.loc[h5_gids, 'scalar'].values
            adder = self._bias_correct.loc[h5_gids, 'adder'].values
            data = data * scalar + adder
            data = np.maximum(data, 0)

        weights = np.zeros(len(gids))
        for i, gid in enumerate(gids):
            mask = self.sc_point._h5_gids == gid
            weights[i] = self.sc_point.include_mask_flat[mask].sum()

        weights /= weights.sum()
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

        dset = f'winddirection_{self.hub_height}m'
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

        config.update(self._wind_plant_pd.sam_sys_inputs)
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
            gid_counts = json.dumps([float(np.round(n, 1))
                                     for n in self.sc_point.gid_counts])

            with SupplyCurveExtent(self.sc_point._excl_fpath,
                                   resolution=self.sc_point.resolution) as sc:
                row_ind, col_ind = sc.get_sc_row_col_ind(self.sc_point.gid)

            self._meta = pd.DataFrame(
                {'sc_point_gid': self.sc_point.gid,
                 'sc_row_ind': row_ind,
                 'sc_col_ind': col_ind,
                 'gid': self.sc_point.gid,
                 'latitude': self.sc_point.latitude,
                 'longitude': self.sc_point.longitude,
                 'timezone': self.sc_point.timezone,
                 'country': self.sc_point.country,
                 'state': self.sc_point.state,
                 'county': self.sc_point.county,
                 'elevation': self.sc_point.elevation,
                 'offshore': self.sc_point.offshore,
                 'res_gids': res_gids,
                 'gid_counts': gid_counts,
                 'n_gids': self.sc_point.n_gids,
                 'area_sq_km': self.sc_point.area,
                 }, index=[self.sc_point.gid])

        return self._meta

    @property
    def hub_height(self):
        """Get the integer SAM system config turbine hub height (meters)

        Returns
        -------
        int
        """
        return int(self.sam_sys_inputs['wind_turbine_hub_ht'])

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
            ws = self.get_weighted_res_ts(f'windspeed_{self.hub_height}m')
            temp = self.get_weighted_res_ts(f'temperature_{self.hub_height}m')
            pres = self.get_weighted_res_ts(f'pressure_{self.hub_height}m')

            # convert mbar to atm
            if np.nanmax(pres) > 1000:
                pres *= 9.86923e-6

            self._res_df = pd.DataFrame({'temperature': temp,
                                         'pressure': pres,
                                         'windspeed': ws,
                                         'winddirection': wd}, index=ti)
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

            hist_out = np.histogram2d(self.res_df['windspeed'],
                                      self.res_df['winddirection'],
                                      bins=(ws_bins, wd_bins))
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

            if 'lcoe_fcr' in self._out_req:
                lcoe_kwargs = self.get_lcoe_kwargs()
                sam_inputs.update(lcoe_kwargs)

            i_wp = WindPower(res_df, self.meta, sam_inputs,
                             output_request=self._out_req)
            wind_plant_ts[year] = i_wp

        return wind_plant_ts

    @property
    def wind_plant_pd(self):
        """reV WindPowerPD compute object for plant layout optimization based
        on wind joint probability distribution

        Returns
        -------
        reV.SAM.generation.WindPowerPD
        """

        if self._wind_plant_pd is None:
            wind_dist, ws_edges, wd_edges = self.wind_dist
            self._wind_plant_pd = WindPowerPD(ws_edges, wd_edges, wind_dist,
                                              self.meta, self.sam_sys_inputs,
                                              output_request=self._out_req)
        return self._wind_plant_pd

    @property
    def wind_plant_ts(self):
        """reV WindPower compute object(s) based on wind resource timeseries
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
                self.include_mask,
                self.pixel_side_length,
                self.min_spacing,
                self.wake_loss_multiplier)

        return self._plant_optm

    def recalc_lcoe(self):
        """Recalculate the multi-year mean LCOE based on the multi-year mean
        annual energy production (AEP)"""

        if 'lcoe_fcr-means' in self.outputs:
            lcoe_kwargs = self.get_lcoe_kwargs()

            logger.debug('Recalulating multi-year mean LCOE using '
                         'multi-year mean AEP.')

            fcr = lcoe_kwargs['fixed_charge_rate']
            cap_cost = lcoe_kwargs['capital_cost']
            foc = lcoe_kwargs['fixed_operating_cost']
            voc = lcoe_kwargs['variable_operating_cost']
            aep = self.outputs['annual_energy-means']

            my_mean_lcoe = lcoe_fcr(fcr, cap_cost, foc, aep, voc)

            self._outputs['lcoe_fcr-means'] = my_mean_lcoe
            self._meta['mean_lcoe'] = my_mean_lcoe

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
            fixed_operating_cos ($), variable_operating_cost ($/kWh)
        """

        if 'system_capacity' not in self.outputs:
            msg = ('Could not find system_capacity in the outputs, need to '
                   'run_plant_optimization() to get the optimized '
                   'system_capacity before calculating LCOE!')
            logger.error(msg)
            raise RuntimeError(msg)

        lcoe_kwargs = {
            'fixed_charge_rate':
                self.original_sam_sys_inputs['fixed_charge_rate'],
            'system_capacity': self.plant_optimizer.capacity,
            'capital_cost': self.plant_optimizer.capital_cost,
            'fixed_operating_cost': self.plant_optimizer.fixed_operating_cost,
            'variable_operating_cost':
                self.plant_optimizer.variable_operating_cost}

        for k, v in lcoe_kwargs.items():
            self._meta[k] = v

        return lcoe_kwargs

    @staticmethod
    def get_wind_handler(res):
        """Get a wind resource handler for a resource filepath.

        Parameters
        ----------
        res : str
            Resource filepath to wtk .h5 file. Can include * wildcards
            for multi year resource.

        Returns
        -------
        handler : WindResource | MultiYearWindResource
            Wind resource handler or multi year handler
        """
        handler = res
        if isinstance(res, str):
            if '*' in res:
                handler = MultiYearWindResource
            else:
                handler = WindResource
        return handler

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
            msg = ('The reV bespoke module depends on the following special '
                   'dependencies that were not found in the active '
                   'environment: {}'.format(missing))
            logger.error(msg)
            raise ModuleNotFoundError(msg)

    @staticmethod
    def _check_sys_inputs(plant1, plant2,
                          ignore=('wind_resource_model_choice',
                                  'wind_resource_data',
                                  'wind_turbine_powercurve_powerout',
                                  'hourly',
                                  'capital_cost',
                                  'fixed_operating_cost',
                                  'variable_operating_cost')):
        """Check two reV-SAM models for matching system inputs.

        Parameters
        ----------
        plant1/plant2 : reV.SAM.generation.WindPower
            Two WindPower analysis objects to check.
        """
        bad = []
        for k, v in plant1.sam_sys_inputs.items():
            if k not in plant2.sam_sys_inputs:
                bad.append(k)
            elif str(v) != str(plant2.sam_sys_inputs[k]):
                bad.append(k)
        bad = [b for b in bad if b not in ignore]
        if any(bad):
            msg = 'Inputs no longer match: {}'.format(bad)
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

        logger.debug('Running {} years of SAM timeseries analysis for {}'
                     .format(len(self.years), self))
        self._wind_plant_ts = self.initialize_wind_plant_ts()
        for year, plant in self.wind_plant_ts.items():
            self._check_sys_inputs(plant, self.wind_plant_pd)
            try:
                plant.run_gen_and_econ()
            except Exception as e:
                msg = ('{} failed while trying to run SAM WindPower '
                       'timeseries analysis for {}'.format(self, year))
                logger.exception(msg)
                raise RuntimeError(msg) from e

            for k, v in plant.outputs.items():
                self._outputs[k + '-{}'.format(year)] = v

        means = {}
        for k1, v1 in self._outputs.items():
            if isinstance(v1, Number) and parse_year(k1, option='boolean'):
                year = parse_year(k1)
                base_str = k1.replace(str(year), '')
                all_values = [v2 for k2, v2 in self._outputs.items()
                              if base_str in k2]
                means[base_str + 'means'] = np.mean(all_values)

        self._outputs.update(means)

        # copy dataset outputs to meta data for supply curve table summary
        if 'cf_mean-means' in self.outputs:
            self._meta['mean_cf'] = self.outputs['cf_mean-means']
        if 'lcoe_fcr-means' in self.outputs:
            self._meta['mean_lcoe'] = self.outputs['lcoe_fcr-means']
            self.recalc_lcoe()

        logger.debug('Timeseries analysis complete!')

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

        logger.debug('Running plant layout optimization for {}'.format(self))
        try:
            self.plant_optimizer.place_turbines(**self.ga_kwargs)
        except Exception as e:
            msg = ('{} failed while trying to run the '
                   'turbine placement optimizer'
                   .format(self))
            logger.exception(msg)
            raise RuntimeError(msg) from e

        # TODO need to add:
        # total cell area
        # cell capacity density

        txc = [int(np.round(c)) for c in self.plant_optimizer.turbine_x]
        tyc = [int(np.round(c)) for c in self.plant_optimizer.turbine_y]
        pxc = [int(np.round(c)) for c in self.plant_optimizer.x_locations]
        pyc = [int(np.round(c)) for c in self.plant_optimizer.y_locations]

        txc = json.dumps(txc)
        tyc = json.dumps(tyc)
        pxc = json.dumps(pxc)
        pyc = json.dumps(pyc)

        self._meta["turbine_x_coords"] = txc
        self._meta["turbine_y_coords"] = tyc
        self._meta["possible_x_coords"] = pxc
        self._meta["possible_y_coords"] = pyc

        self._outputs["full_polygons"] = self.plant_optimizer.full_polygons
        self._outputs["packing_polygons"] = \
            self.plant_optimizer.packing_polygons
        self._outputs["system_capacity"] = self.plant_optimizer.capacity

        self._meta["n_turbines"] = self.plant_optimizer.nturbs
        self._meta["bespoke_aep"] = self.plant_optimizer.aep
        self._meta["bespoke_objective"] = self.plant_optimizer.objective
        self._meta["bespoke_capital_cost"] = \
            self.plant_optimizer.capital_cost
        self._meta["bespoke_fixed_operating_cost"] = \
            self.plant_optimizer.fixed_operating_cost
        self._meta["bespoke_variable_operating_cost"] = \
            self.plant_optimizer.variable_operating_cost
        self._meta["included_area"] = self.plant_optimizer.area
        self._meta["included_area_capacity_density"] = \
            self.plant_optimizer.capacity_density
        self._meta["convex_hull_area"] = \
            self.plant_optimizer.convex_hull_area
        self._meta["convex_hull_capacity_density"] = \
            self.plant_optimizer.convex_hull_capacity_density
        self._meta["full_cell_capacity_density"] = \
            self.plant_optimizer.full_cell_capacity_density

        logger.debug('Plant layout optimization complete!')

        # copy dataset outputs to meta data for supply curve table summary
        # convert SAM system capacity in kW to reV supply curve cap in MW
        self._meta['capacity'] = self.outputs['system_capacity'] / 1e3

        # add required ReEDS multipliers to meta
        baseline_cost = self.plant_optimizer.capital_cost_per_kw(
            capacity_mw=self._baseline_cap_mw)
        self._meta['eos_mult'] = (self.plant_optimizer.capital_cost
                                  / self.plant_optimizer.capacity
                                  / baseline_cost)
        self._meta['reg_mult'] = (self.sam_sys_inputs
                                  .get("capital_cost_multiplier", 1))

        return self.outputs

    def agg_data_layers(self):
        """Aggregate optional data layers if requested and save to self.meta"""
        if self._data_layers is not None:
            logger.debug('Aggregating {} extra data layers.'
                         .format(len(self._data_layers)))
            point_summary = self.meta.to_dict()
            point_summary = self.sc_point.agg_data_layers(point_summary,
                                                          self._data_layers)
            self._meta = pd.DataFrame(point_summary)
            logger.debug('Finished aggregating extra data layers.')

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
                logger.debug('Skipping bespoke plant optimization for gid {}. '
                             'Received prior meta data for this point.'
                             .format(bsp.gid))
            else:
                _ = bsp.run_plant_optimization()

            _ = bsp.run_wind_plant_ts()
            bsp.agg_data_layers()

            meta = bsp.meta
            out = bsp.outputs
            out['meta'] = meta
            for year, ti in zip(bsp.years, bsp.annual_time_indexes):
                out['time_index-{}'.format(year)] = ti

        return out


class BespokeWindPlants(BaseAggregation):
    """BespokeWindPlants"""

    def __init__(self, excl_fpath, res_fpath, tm_dset, objective_function,
                 capital_cost_function, fixed_operating_cost_function,
                 variable_operating_cost_function, project_points,
                 sam_files, min_spacing='5x', wake_loss_multiplier=1,
                 ga_kwargs=None, output_request=('system_capacity', 'cf_mean'),
                 ws_bins=(0.0, 20.0, 5.0), wd_bins=(0.0, 360.0, 45.0),
                 excl_dict=None, area_filter_kernel='queen', min_area=None,
                 resolution=64, excl_area=None, data_layers=None,
                 pre_extract_inclusions=False, prior_run=None, gid_map=None,
                 bias_correct=None, pre_load_data=False):
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

        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions data HDF5 file. The exclusions HDF5
            file should contain the layers specified in `excl_dict`
            and `data_layers` (though data for the latter may be
            stored in a separate file - see the `data_layers` input
            documentation for more details). These data layers may
            be spread out across multiple files, in which case this
            input should be a list or tuple of filepaths to multiple
            exclusion HDF5 files containing the layers. Note that each
            data layer must be uniquely defined (i.e.only appear once
            and in a single input file).
        res_fpath : str
            Filepath to wind resource data in NREL WTK format. This
            input can be path to a single resource HDF5 file or a path
            including a wildcard input like ``/h5_dir/prefix*suffix`` to
            run bespoke on multiple years of resource data. The former
            must be readable by
            :py:class:`rex.renewable_resource.WindResource` while the
            latter must be readable by
            or :py:class:`rex.multi_year_resource.MultiYearWindResource`
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
            techmap (exclusions-to-resource mapping data). This dataset
            uniquely couples the (typically high-resolution) exclusion
            layers to the (typically lower-resolution) resource data,
            and therefore should be unique for every new resource data
            set that is paired with the exclusion data.
        objective_function : str
            The objective function of the optimization written out as a
            string. This expression should compute the objective to be
            minimized during layout optimization. Variables available
            for computation are:

                - ``n_turbines``: the number of turbines
                - ``system_capacity``: wind plant capacity
                - ``aep``: annual energy production
                - ``fixed_charge_rate``: user input fixed_charge_rate if
                  included as part of the sam system config.
                - ``self.wind_plant``: the SAM wind plant object,
                  through which all SAM variables can be accessed
                - ``capital_cost``: plant capital cost as evaluated
                  by `capital_cost_function`
                - ``fixed_operating_cost``: plant fixed annual operating
                  cost as evaluated by `fixed_operating_cost_function`
                - ``variable_operating_cost``: plant variable annual
                  operating cost, as evaluated by
                  `variable_operating_cost_function`

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
        project_points : int | list | tuple | str | dict | pd.DataFrame | slice
            Input specifying which sites to process. A single integer
            representing the supply curve GID of a site may be specified
            to evaluate ``reV`` at a supply curve point. A list or tuple
            of integers (or slice) representing the supply curve GIDs of
            multiple sites can be specified to evaluate ``reV`` at
            multiple specific locations. A string pointing to a project
            points CSV file may also be specified. Typically, the CSV
            contains two columns:

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
            these extra columns:

                - ``capital_cost_multiplier``
                - ``fixed_operating_cost_multiplier``
                - ``variable_operating_cost_multiplier``

            These particular inputs are treated as multipliers to be
            applied to the respective cost curves
            (`capital_cost_function`, `fixed_operating_cost_function`,
            and `variable_operating_cost_function`) both during and
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
            documentation on the allowed and/or required SAM config file
            inputs.
        min_spacing : float | int | str, optional
            Minimum spacing between turbines (in meters). This input can
            also be a string like "5x", which is interpreted as 5 times
            the turbine rotor diameter. By default, ``"5x"``.
        wake_loss_multiplier : float, optional
            A multiplier used to scale the annual energy lost due to
            wake losses.

            .. WARNING:: This multiplier will ONLY be applied during the
               optimization process and will NOT come through in output
               values such as the hourly profiles, aep, any of the cost
               functions, or even the output objective.

            By default, ``1``.
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
            If ``None`` or empty dictionary, no exclusions are applied.
            By default, ``None``.
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
            left out, the data is assumed to exist in `excl_fpath`. If
            ``None``, no data layer aggregation is performed.
            By default, ``None``.
        pre_extract_inclusions : bool, optional
            Optional flag to pre-extract/compute the inclusion mask from
            the `excl_dict` input. It is typically faster to compute
            the inclusion mask on the fly with parallel workers.
            By default, ``False``.
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

                - ``gid``: GID of site (can be index name)
                - ``adder``: Value to add to resource at each site
                - ``scalar``: Value to scale resource at each site by

            The ``gid`` field should match the true resource ``gid``
            regardless of the optional ``gid_map`` input. If both
            ``adder`` and ``scalar`` are present, the wind or solar
            resource is corrected by :math:`(res*scalar)+adder`. If
            *either* is missing, ``scalar`` defaults to 1 and ``adder``
            to 0. Only `windspeed` **or** `GHI` + `DNI` are corrected,
            depending on the technology (wind for the former, solar
            for the latter). `GHI` and `DNI` are corrected with the
            same correction factors. If ``None``, no corrections are
            applied. By default, ``None``.
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
        logger.info('Bespoke wake loss multiplier: {}'
                    .format(wake_loss_multiplier))
        logger.info('Bespoke GA initialization kwargs: {}'.format(ga_kwargs))

        logger.info('Bespoke pre-extracting exclusions: {}'
                    .format(pre_extract_inclusions))
        logger.info('Bespoke pre-extracting resource data: {}'
                    .format(pre_load_data))
        logger.info('Bespoke prior run: {}'.format(prior_run))
        logger.info('Bespoke GID map: {}'.format(gid_map))
        logger.info('Bespoke bias correction table: {}'.format(bias_correct))

        BespokeSinglePlant.check_dependencies()

        self._project_points = self._parse_points(project_points, sam_files)

        super().__init__(excl_fpath, tm_dset, excl_dict=excl_dict,
                         area_filter_kernel=area_filter_kernel,
                         min_area=min_area, resolution=resolution,
                         excl_area=excl_area, gids=self._project_points.gids,
                         pre_extract_inclusions=pre_extract_inclusions)

        self._res_fpath = res_fpath
        self._obj_fun = objective_function
        self._cap_cost_fun = capital_cost_function
        self._foc_fun = fixed_operating_cost_function
        self._voc_fun = variable_operating_cost_function
        self._min_spacing = min_spacing
        self._wake_loss_multiplier = wake_loss_multiplier
        self._ga_kwargs = ga_kwargs or {}
        self._output_request = SAMOutputRequest(output_request)
        self._ws_bins = ws_bins
        self._wd_bins = wd_bins
        self._data_layers = data_layers
        self._prior_meta = self._parse_prior_run(prior_run)
        self._gid_map = BespokeSinglePlant._parse_gid_map(gid_map)
        self._bias_correct = Gen._parse_bc(bias_correct)
        self._outputs = {}
        self._check_files()

        self._pre_loaded_data = None
        self._pre_load_data(pre_load_data)

        self._slice_lookup = None

        logger.info('Initialized BespokeWindPlants with project points: {}'
                    .format(self._project_points))

    @staticmethod
    def _parse_points(points, sam_configs):
        """Parse a project points object using a project points file

        Parameters
        ----------
        points : int | slice | list | str | PointsControl | None
            Slice or list specifying project points, string pointing to a
            project points csv, or a fully instantiated PointsControl object.
            Can also be a single site integer value. Points csv should have
            'gid' and 'config' column, the config maps to the sam_configs dict
            keys.
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
        pc = Gen.get_pc(points, points_range=None, sam_configs=sam_configs,
                        tech='windpower', sites_per_worker=1)

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
            assert prior_run.endswith('.h5')

            with Outputs(prior_run, mode='r') as f:
                meta = f.meta

            # pylint: disable=no-member
            for col in meta.columns:
                val = meta[col].values[0]
                if isinstance(val, str) and val[0] == '[' and val[-1] == ']':
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
            mask = self._prior_meta['gid'] == gid
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
                    'Could not find required exclusions file: '
                    '{}'.format(path))

        with ExclusionLayers(paths) as excl:
            if self._tm_dset not in excl:
                raise FileInputError('Could not find techmap dataset "{}" '
                                     'in the exclusions file(s): {}'
                                     .format(self._tm_dset, paths))

        # just check that this file exists, cannot check res_fpath if *glob
        Handler = BespokeSinglePlant.get_wind_handler(self._res_fpath)
        with Handler(self._res_fpath) as f:
            assert any(f.dsets)

    def _pre_load_data(self, pre_load_data):
        """Pre-load resource data, if requested. """
        if not pre_load_data:
            return

        sc_gid_to_hh = {gid: self._hh_for_sc_gid(gid)
                        for gid in self._project_points.df["gid"]}

        with ExclusionLayers(self._excl_fpath) as excl:
            tm = excl[self._tm_dset]

        scp_kwargs = {"shape": self.shape, "resolution": self._resolution}
        slices = {gid: SupplyCurvePoint.get_agg_slices(gid=gid, **scp_kwargs)
                  for gid in self._project_points.df["gid"]}

        sc_gid_to_res_gid = {gid: sorted(set(tm[slx, sly].flatten()))
                             for gid, (slx, sly) in slices.items()}

        for sc_gid, res_gids in sc_gid_to_res_gid.items():
            if res_gids[0] < 0:
                sc_gid_to_res_gid[sc_gid] = res_gids[1:]

        if self._gid_map is not None:
            for sc_gid, res_gids in sc_gid_to_res_gid.items():
                sc_gid_to_res_gid[sc_gid] = sorted(self._gid_map[g]
                                                   for g in res_gids)

        logger.info("Pre-loading resource data for Bespoke run... ")
        self._pre_loaded_data = BespokeMultiPlantData(self._res_fpath,
                                                      sc_gid_to_hh,
                                                      sc_gid_to_res_gid)

    def _hh_for_sc_gid(self, sc_gid):
        """Fetch the hh for a given sc_gid"""
        config = self.sam_sys_inputs_with_site_data(sc_gid)
        return int(config["wind_turbine_hub_ht"])

    def _pre_loaded_data_for_sc_gid(self, sc_gid):
        """Pre-load data for a given SC GID, if requested. """
        if self._pre_loaded_data is None:
            return None

        return self._pre_loaded_data.get_preloaded_data_for_gid(sc_gid)

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
        meta = [self.outputs[g]['meta'] for g in self.completed_gids]
        if len(self.completed_gids) > 1:
            meta = pd.concat(meta, axis=0)
        else:
            meta = meta[0]
        return meta

    @property
    def slice_lookup(self):
        """dict | None: Lookup mapping sc_point_gid to exclusion slice. """
        if self._slice_lookup is None and self._inclusion_mask is not None:
            with SupplyCurveExtent(self._excl_fpath,
                                   resolution=self._resolution) as sc:
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
        site_sys_inputs.update({k: v for k, v in site_data.to_dict().items()
                                if not (isinstance(v, float) and np.isnan(v))})
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

        Returns
        -------
        out_fpath : str
            Full filepath to desired .h5 output file, the .h5 extension has
            been added if it was not already present.
        """

        if not out_fpath.endswith('.h5'):
            out_fpath += '.h5'

        if ModuleName.BESPOKE not in out_fpath:
            extension_with_module = "_{}.h5".format(ModuleName.BESPOKE)
            out_fpath = out_fpath.replace(".h5", extension_with_module)

        out_dir = os.path.dirname(out_fpath)
        if not os.path.exists(out_dir):
            create_dirs(out_dir)

        with Outputs(out_fpath, mode='w') as f:
            f._set_meta('meta', self.meta, attrs={})
            ti_dsets = [d for d in sample.keys()
                        if d.startswith('time_index-')]
            for dset in ti_dsets:
                f._set_time_index(dset, sample[dset], attrs={})
                f._set_time_index('time_index', sample[dset], attrs={})

        return out_fpath

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
            msg = ('Not writing dataset "{}" of type "{}" to disk.'
                   .format(dset, type(single_arr)))
            logger.info(msg)
            return None

        if isinstance(sample_num, float):
            dtype = np.float32
        else:
            dtype = type(sample_num)
        full_arr = np.zeros(shape, dtype=dtype)

        # collect data from all wind plants
        logger.info('Collecting dataset "{}" with final shape {}'
                    .format(dset, shape))
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
        """
        if not self.completed_gids:
            msg = ("No output data found! It is likely that all requested "
                   "points are excluded.")
            logger.warning(msg)
            warn(msg)
            return

        sample = self.outputs[self.completed_gids[0]]
        out_fpath = self._init_fout(out_fpath, sample)

        dsets = [d for d in sample.keys()
                 if not d.startswith('time_index-')
                 and d != 'meta']
        with Outputs(out_fpath, mode='a') as f:
            for dset in dsets:
                full_arr = self._collect_out_arr(dset, sample)
                if full_arr is not None:
                    dset_no_year = dset
                    if parse_year(dset, option='boolean'):
                        year = parse_year(dset)
                        dset_no_year = dset.replace('-{}'.format(year), '')

                    attrs = BespokeSinglePlant.OUT_ATTRS.get(dset_no_year, {})
                    attrs = copy.deepcopy(attrs)
                    dtype = attrs.pop('dtype', np.float32)
                    chunks = attrs.pop('chunks', None)
                    try:
                        f.write_dataset(dset, full_arr, dtype, chunks=chunks,
                                        attrs=attrs)
                    except Exception as e:
                        msg = 'Failed to write "{}" to disk.'.format(dset)
                        logger.exception(msg)
                        raise IOError(msg) from e

        logger.info('Saved output data to: {}'.format(out_fpath))

    # pylint: disable=arguments-renamed
    @classmethod
    def run_serial(cls, excl_fpath, res_fpath, tm_dset,
                   sam_sys_inputs, objective_function,
                   capital_cost_function,
                   fixed_operating_cost_function,
                   variable_operating_cost_function,
                   min_spacing='5x', wake_loss_multiplier=1, ga_kwargs=None,
                   output_request=('system_capacity', 'cf_mean'),
                   ws_bins=(0.0, 20.0, 5.0), wd_bins=(0.0, 360.0, 45.0),
                   excl_dict=None, inclusion_mask=None,
                   area_filter_kernel='queen', min_area=None,
                   resolution=64, excl_area=0.0081, data_layers=None,
                   gids=None, exclusion_shape=None, slice_lookup=None,
                   prior_meta=None, gid_map=None, bias_correct=None,
                   pre_loaded_data=None):
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
        Handler = BespokeSinglePlant.get_wind_handler(res_fpath)

        # pre-extract handlers so they are not repeatedly initialized
        file_kwargs = {'excl_dict': excl_dict,
                       'area_filter_kernel': area_filter_kernel,
                       'min_area': min_area,
                       'h5_handler': Handler,
                       }

        with AggFileHandler(excl_fpath, res_fpath, **file_kwargs) as fh:
            n_finished = 0
            for gid in gids:
                gid_inclusions = cls._get_gid_inclusion_mask(
                    inclusion_mask, gid, slice_lookup,
                    resolution=resolution)
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
                        min_spacing=min_spacing,
                        wake_loss_multiplier=wake_loss_multiplier,
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
                        prior_meta=prior_meta,
                        gid_map=gid_map,
                        bias_correct=bias_correct,
                        pre_loaded_data=pre_loaded_data,
                        close=False)

                except EmptySupplyCurvePointError:
                    logger.debug('SC gid {} is fully excluded or does not '
                                 'have any valid source data!'.format(gid))
                except Exception as e:
                    msg = 'SC gid {} failed!'.format(gid)
                    logger.exception(msg)
                    raise RuntimeError(msg) from e
                else:
                    n_finished += 1
                    logger.debug('Serial bespoke: '
                                 '{} out of {} points complete'
                                 .format(n_finished, len(gids)))
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

        logger.info('Running bespoke optimization for points {} through {} '
                    'at a resolution of {} on {} cores.'
                    .format(self.gids[0], self.gids[-1], self._resolution,
                            max_workers))

        futures = []
        out = {}
        n_finished = 0
        loggers = [__name__, 'reV.supply_curve.point_summary', 'reV']
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
                    self._min_spacing,
                    wake_loss_multiplier=self._wake_loss_multiplier,
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
                    prior_meta=self._get_prior_meta(gid),
                    gid_map=self._gid_map,
                    bias_correct=self._bias_correct,
                    pre_loaded_data=self._pre_loaded_data_for_sc_gid(gid)))

            # gather results
            for future in as_completed(futures):
                n_finished += 1
                out.update(future.result())
                if n_finished % 10 == 0:
                    mem = psutil.virtual_memory()
                    logger.info('Parallel bespoke futures collected: '
                                '{} out of {}. Memory usage is {:.3f} GB out '
                                'of {:.3f} GB ({:.2f}% utilized).'
                                .format(n_finished, len(futures),
                                        mem.used / 1e9, mem.total / 1e9,
                                        100 * mem.used / mem.total))

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
        if self._obj_fun == 'test':
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
                wlm = self._wake_loss_multiplier
                si = self.run_serial(self._excl_fpath,
                                     self._res_fpath,
                                     self._tm_dset,
                                     sam_inputs,
                                     self._obj_fun,
                                     self._cap_cost_fun,
                                     self._foc_fun,
                                     self._voc_fun,
                                     min_spacing=self._min_spacing,
                                     wake_loss_multiplier=wlm,
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
                                     prior_meta=prior_meta,
                                     gid_map=self._gid_map,
                                     bias_correct=self._bias_correct,
                                     gids=gid,
                                     pre_loaded_data=pre_loaded_data)
                self._outputs.update(si)
        else:
            self._outputs = self.run_parallel(max_workers=max_workers)

        if out_fpath is not None:
            self.save_outputs(out_fpath)

        return out_fpath
