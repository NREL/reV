# -*- coding: utf-8 -*-
"""
reV bespoke wind plant analysis tools
"""
# TODO update docstring
# TODO check on outputs
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

from reV.config.project_points import ProjectPoints
from reV.generation.generation import Gen
from reV.SAM.generation import WindPower, WindPowerPD
from reV.handlers.outputs import Outputs
from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.extent import SupplyCurveExtent
from reV.supply_curve.points import AggregationSupplyCurvePoint as AggSCPoint
from reV.supply_curve.aggregation import AbstractAggregation, AggFileHandler
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      FileInputError)
from reV.utilities import log_versions

from rex.joint_pd.joint_pd import JointPD
from rex.renewable_resource import WindResource
from rex.multi_year_resource import MultiYearWindResource
from rex.utilities.loggers import log_mem, create_dirs
from rex.utilities.utilities import parse_year
from rex.utilities.execution import SpawnProcessPool

logger = logging.getLogger(__name__)


class BespokeSinglePlant:
    """Framework for analyzing and optimized a wind plant layout specific to
    the local wind resource and exclusions for a single reV supply curve point.
    """

    DEPENDENCIES = ('shapely', 'rasterio')
    OUT_ATTRS = copy.deepcopy(Gen.OUT_ATTRS)

    def __init__(self, gid, excl, res, tm_dset, sam_sys_inputs,
                 objective_function, cost_function,
                 min_spacing='5x', ga_time=20,
                 output_request=('system_capacity', 'cf_mean'),
                 ws_bins=(0.0, 20.0, 5.0), wd_bins=(0.0, 360.0, 45.0),
                 excl_dict=None, inclusion_mask=None, data_layers=None,
                 resolution=64, excl_area=None, exclusion_shape=None,
                 close=True):
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
                - self.wind_plant: the SAM wind plant object, through which
                all SAM variables can be accessed
                - cost: the annual cost of the wind plant (from cost_function)
        cost_function : str
            The cost function as a string, should return the annual cost
            of the wind farm. Variables available are:
                - n_turbines: the number of turbines
                - system_capacity: wind plant capacity
                - aep: annual energy production
                - self.wind_plant: the SAM wind plant object, through which
                all SAM variables can be accessed
        min_spacing : float | int | str
            Minimum spacing between turbines in meters. Can also be a string
            like "5x" (default) which is interpreted as 5 times the turbine
            rotor diameter.
        ga_time : int
            Cutoff time for single-plant genetic algorithm optimization in
            seconds. Default is 20 seconds.
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
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
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
        self.cost_function = cost_function
        self.min_spacing = min_spacing
        self.ga_time = ga_time

        self._sam_sys_inputs = sam_sys_inputs
        self._out_req = list(output_request)
        self._ws_bins = ws_bins
        self._wd_bins = wd_bins

        self._res_df = None
        self._meta = None
        self._wind_dist = None
        self._ws_edges = None
        self._wd_edges = None
        self._wind_plant_pd = None
        self._wind_plant_ts = None
        self._plant_optm = None
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
        self.sc_point.close()
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

        for req in self._out_req:
            dset = req.replace('_mean', '')
            available_dsets = ('windspeed', 'winddirection',
                               'pressure', 'temperature')
            if dset in available_dsets:
                self._out_req.remove(req)
                self._outputs[req] = self.res_df[dset].mean()

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
        data = self.sc_point.h5[dset, :, gids]

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
        dirs = self.sc_point.h5[dset, :, gids]
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
    def sam_sys_inputs(self):
        """Get the SAM windpower system inputs. If the wind plant has not yet
        been optimized, this returns the initial SAM config. If the wind plant
        has been optimized using the wind_plant_pd object, this returns the
        final optimized SAM plant config.

        Returns
        -------
        dict
        """
        if self._wind_plant_pd is None:
            return self._sam_sys_inputs
        else:
            return self._wind_plant_pd.sam_sys_inputs

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

            self._meta = pd.DataFrame(
                {'sc_point_gid': self.sc_point.gid,
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
            ti = self.sc_point.h5.time_index

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
            i_wp = WindPower(res_df, self.meta,
                             copy.deepcopy(self.sam_sys_inputs),
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
            self._plant_optm = PlaceTurbines(self.wind_plant_pd,
                                             self.objective_function,
                                             self.cost_function,
                                             self.include_mask,
                                             self.pixel_side_length,
                                             self.min_spacing,
                                             self.ga_time)
        return self._plant_optm

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
                                  'wind_resource_data')):
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
            self.plant_optimizer.place_turbines()
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

        self._outputs["n_turbines"] = self.plant_optimizer.nturbs
        self._outputs["system_capacity"] = self.plant_optimizer.capacity
        self._outputs["bespoke_aep"] = self.plant_optimizer.aep
        self._outputs["bespoke_objective"] = self.plant_optimizer.objective
        self._outputs["bespoke_annual_cost"] = \
            self.plant_optimizer.annual_cost
        self._outputs["included_area_capacity_density"] = \
            self.plant_optimizer.capacity_density

        logger.debug('Plant layout optimization complete!')

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
            _ = bsp.run_plant_optimization()
            _ = bsp.run_wind_plant_ts()
            bsp.agg_data_layers()

            meta = bsp.meta
            out = bsp.outputs
            out['meta'] = meta
            for year, ti in zip(bsp.years, bsp.annual_time_indexes):
                out['time_index-{}'.format(year)] = ti

        return out


class BespokeWindPlants(AbstractAggregation):
    """Framework for analyzing optimized wind plant layouts specific to the
    local wind resource and exclusions for the full reV supply curve grid.
    """

    def __init__(self, excl_fpath, res_fpath, tm_dset,
                 objective_function, cost_function,
                 points, sam_configs, points_range=None,
                 min_spacing='5x', ga_time=20,
                 output_request=('system_capacity', 'cf_mean'),
                 ws_bins=(0.0, 20.0, 5.0), wd_bins=(0.0, 360.0, 45.0),
                 excl_dict=None,
                 area_filter_kernel='queen', min_area=None,
                 resolution=64, excl_area=None, data_layers=None,
                 pre_extract_inclusions=False):
        """
        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        res_fpath : str
            Wind resource h5 filepath in NREL WTK format. Can also include
            unix-style wildcards like /dir/wind_*.h5 for multiple years of
            resource data.
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        points : int | slice | list | str | PointsControl | None
            Slice or list specifying project points, string pointing to a
            project points csv, or a fully instantiated PointsControl object.
            Can also be a single site integer value. Points csv should have
            'gid' and 'config' column, the config maps to the sam_configs dict
            keys. If this is None, all available reV supply curve points are
            included (or sliced by points_range).
        sam_configs : dict | str | SAMConfig
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s) which map to the config column in the project points
            CSV. Values are either a JSON SAM config file or dictionary of SAM
            config inputs. Can also be a single config file path or a
            pre loaded SAMConfig object.
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. The list is the (Beginning, end) (inclusive/exclusive,
            respectively) index split parameters for ProjectPoints.split()
            method.
        objective_function : str
            The objective function of the optimization as a string, should
            return the objective to be minimized during layout optimization.
            Variables available are:
                - n_turbines: the number of turbines
                - system_capacity: wind plant capacity
                - aep: annual energy production
                - self.wind_plant: the SAM wind plant object, through which
                all SAM variables can be accessed
                - cost: the annual cost of the wind plant (from cost_function)
        cost_function : str
            The cost function as a string, should return the annual cost
            of the wind farm. Variables available are:
                - n_turbines: the number of turbines
                - system_capacity: wind plant capacity
                - aep: annual energy production
                - self.wind_plant: the SAM wind plant object, through which
                all SAM variables can be accessed
        min_spacing : float | int | str
            Minimum spacing between turbines in meters. Can also be a string
            like "5x" (default) which is interpreted as 5 times the turbine
            rotor diameter.
        ga_time : int
            Cutoff time for single-plant genetic algorithm optimization in
            seconds. Default is 20 seconds.
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
        excl_dict : dict, optional
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
            by default None
        area_filter_kernel : str, optional
            Contiguous area filter method to use on final exclusions mask,
            by default "queen"
        min_area : float, optional
            Minimum required contiguous area filter in sq-km,
            by default None
        resolution : int, optional
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead,
            by default None
        excl_area : float, optional
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath,
            by default None
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        gids : list, optional
            List of supply curve point gids to get summary for (can use to
            subset if running in parallel), or None for all gids in the SC
            extent, by default None
        pre_extract_inclusions : bool, optional
            Optional flag to pre-extract/compute the inclusion mask from the
            provided excl_dict, by default False. Typically faster to compute
            the inclusion mask on the fly with parallel workers.
        """

        log_versions(logger)
        logger.info('Initializing BespokeWindPlants...')
        logger.info('Resource filepath: {}'.format(res_fpath))
        logger.info('Exclusion filepath: {}'.format(excl_fpath))
        logger.debug('Exclusion dict: {}'.format(excl_dict))
        logger.info('Bespoke objective function: {}'
                    .format(objective_function))
        logger.info('Bespoke cost function: {}'.format(cost_function))

        BespokeSinglePlant.check_dependencies()

        self._points_control = self._parse_points(excl_fpath, res_fpath,
                                                  tm_dset, resolution, points,
                                                  points_range, sam_configs)
        self._project_points = self._points_control.project_points

        super().__init__(excl_fpath, tm_dset, excl_dict=excl_dict,
                         area_filter_kernel=area_filter_kernel,
                         min_area=min_area, resolution=resolution,
                         excl_area=excl_area, gids=self._project_points.gids,
                         pre_extract_inclusions=pre_extract_inclusions)

        self._res_fpath = res_fpath
        self._obj_fun = objective_function
        self._cost_fun = cost_function
        self._min_spacing = min_spacing
        self._ga_time = ga_time
        self._output_request = output_request
        self._ws_bins = ws_bins
        self._wd_bins = wd_bins
        self._data_layers = data_layers
        self._outputs = {}
        self._check_files()

        logger.info('Initialized BespokeWindPlants with project points: {}'
                    .format(self._project_points))

    @staticmethod
    def _parse_points(excl_fpath, res_fpath, tm_dset, resolution,
                      points, points_range, sam_configs, sites_per_worker=1,
                      workers=None):
        """Parse a project points object using either an explicit project
        points file or if points=None get all available supply curve points
        based on the exclusion file + resolution + techmap

        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        res_fpath : str
            Wind resource h5 filepath in NREL WTK format. Can also include
            unix-style wildcards like /dir/wind_*.h5 for multiple years of
            resource data.
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        resolution : int, optional
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead,
            by default None
        points : int | slice | list | str | PointsControl | None
            Slice or list specifying project points, string pointing to a
            project points csv, or a fully instantiated PointsControl object.
            Can also be a single site integer value. Points csv should have
            'gid' and 'config' column, the config maps to the sam_configs dict
            keys. If this is None, all available reV supply curve points are
            included (or sliced by points_range).
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. The list is the (Beginning, end) (inclusive/exclusive,
            respectively) index split parameters for ProjectPoints.split()
            method.
        sam_configs : dict | str | SAMConfig
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s) which map to the config column in the project points
            CSV. Values are either a JSON SAM config file or dictionary of SAM
            config inputs. Can also be a single config file path or a
            pre loaded SAMConfig object.
        sites_per_worker : int | None
            Number of sites to run per project points split.
        workers : int | None
            Optional input that will calculate the sites_per_worker based on
            the number of points

        Returns
        -------
        PointsControl : reV.config.project_points.PointsControl
            Project points control object laying out the supply curve gids to
            analyze.
        """

        if points is None:
            logger.info('Points input is None, parsing available points '
                        'from exclusion file and techmap dataset.')
            with SupplyCurveExtent(excl_fpath, resolution=resolution) as sc:
                points = sc.valid_sc_points(tm_dset).tolist()

        points = ProjectPoints._parse_points(points, res_file=res_fpath)

        if workers is not None:
            sites_per_worker = int(np.ceil(len(points) / workers))

        pc = Gen.get_pc(points, points_range, sam_configs,
                        tech='windpower', sites_per_worker=sites_per_worker)

        return pc

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
        # initialize output data array for all wind plants
        full_arr = None
        shape = None
        if isinstance(single_arr, Number):
            shape = (len(self.completed_gids),)
            full_arr = np.zeros(shape, type(single_arr))
        elif isinstance(single_arr, (list, tuple, np.ndarray)):
            shape = (len(single_arr), len(self.completed_gids))
            full_arr = np.zeros(shape, dtype=type(single_arr[0]))
        else:
            msg = ('Not writing dataset "{}" of type "{}" to disk.'
                   .format(dset, type(single_arr)))
            logger.info(msg)

        # collect data from all wind plants
        if full_arr is not None:
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

    @classmethod
    def run_serial(cls, excl_fpath, res_fpath, tm_dset,
                   sam_sys_inputs, objective_function, cost_function,
                   min_spacing='5x', ga_time=20,
                   output_request=('system_capacity', 'cf_mean'),
                   ws_bins=(0.0, 20.0, 5.0), wd_bins=(0.0, 360.0, 45.0),
                   excl_dict=None, inclusion_mask=None,
                   area_filter_kernel='queen', min_area=None,
                   resolution=64, excl_area=0.0081, data_layers=None,
                   gids=None, exclusion_shape=None, slice_lookup=None,
                   ):
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
                        cost_function,
                        min_spacing=min_spacing,
                        ga_time=ga_time,
                        output_request=output_request,
                        ws_bins=ws_bins,
                        wd_bins=wd_bins,
                        excl_dict=excl_dict,
                        inclusion_mask=gid_inclusions,
                        resolution=resolution,
                        excl_area=excl_area,
                        data_layers=data_layers,
                        exclusion_shape=exclusion_shape,
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

        slice_lookup = None
        if self._inclusion_mask is not None:
            with SupplyCurveExtent(self._excl_fpath,
                                   resolution=self._resolution) as sc:
                assert self.shape == self._inclusion_mask.shape
                slice_lookup = sc.get_slice_lookup(self.gids)

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
                    rs, cs = slice_lookup[gid]
                    gid_incl_mask = self._inclusion_mask[rs, cs]

                futures.append(exe.submit(
                    self.run_serial,
                    self._excl_fpath,
                    self._res_fpath,
                    self._tm_dset,
                    self._project_points[gid][1],
                    self._obj_fun,
                    self._cost_fun,
                    self._min_spacing,
                    self._ga_time,
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
                    slice_lookup=slice_lookup))

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

    @classmethod
    def run(cls, excl_fpath, res_fpath, tm_dset,
            objective_function, cost_function,
            points, sam_configs, points_range=None,
            min_spacing='5x', ga_time=20,
            output_request=('system_capacity', 'cf_mean'),
            ws_bins=(0.0, 20.0, 5.0), wd_bins=(0.0, 360.0, 45.0),
            excl_dict=None,
            area_filter_kernel='queen', min_area=None,
            resolution=64, excl_area=None, data_layers=None,
            pre_extract_inclusions=False, max_workers=None,
            out_fpath=None):
        """Run the bespoke wind plant optimization in serial or parallel.
        See BespokeWindPlants docstring for parameter description.
        """

        bsp = cls(excl_fpath, res_fpath, tm_dset,
                  objective_function, cost_function,
                  points, sam_configs,
                  points_range=points_range,
                  min_spacing=min_spacing,
                  ga_time=ga_time,
                  output_request=output_request,
                  ws_bins=ws_bins,
                  wd_bins=wd_bins,
                  excl_dict=excl_dict,
                  area_filter_kernel=area_filter_kernel,
                  min_area=min_area,
                  resolution=resolution,
                  excl_area=excl_area,
                  data_layers=data_layers,
                  pre_extract_inclusions=pre_extract_inclusions)

        # parallel job distribution test.
        if objective_function == 'test':
            return True

        if max_workers == 1:
            for gid in bsp.gids:
                si = bsp.run_serial(excl_fpath, res_fpath, tm_dset,
                                    bsp._project_points[gid][1],
                                    objective_function,
                                    cost_function,
                                    min_spacing=bsp._min_spacing,
                                    ga_time=bsp._ga_time,
                                    output_request=bsp._output_request,
                                    ws_bins=bsp._ws_bins,
                                    wd_bins=bsp._wd_bins,
                                    excl_dict=bsp._excl_dict,
                                    area_filter_kernel=bsp._area_filter_kernel,
                                    min_area=bsp._min_area,
                                    resolution=bsp._resolution,
                                    excl_area=bsp._excl_area,
                                    data_layers=bsp._data_layers,
                                    gids=gid)
                bsp._outputs.update(si)
        else:
            bsp._outputs = bsp.run_parallel(max_workers=max_workers)

        if out_fpath is not None:
            bsp.save_outputs(out_fpath)

        return bsp
