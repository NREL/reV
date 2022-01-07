# -*- coding: utf-8 -*-
"""
reV bespoke wind plant analysis tools
"""
# TODO update docstring
# TODO check on outputs
import h5py
import logging
import pandas as pd
import numpy as np
import os
import psutil
from numbers import Number
from concurrent.futures import as_completed

from reV.bespoke.place_turbines import PlaceTurbines
from reV.SAM.generation import WindPower, WindPowerPD
from reV.supply_curve.extent import SupplyCurveExtent
from reV.supply_curve.points import AggregationSupplyCurvePoint as AggSCPoint
from reV.supply_curve.aggregation import AbstractAggregation, AggFileHandler
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      FileInputError)
from reV.utilities import log_versions

from rex.joint_pd.joint_pd import JointPD
from rex.multi_year_resource import MultiYearWindResource
from rex.utilities.loggers import log_mem
from rex.utilities.utilities import parse_year
from rex.utilities.execution import SpawnProcessPool

logger = logging.getLogger(__name__)


class BespokeSingleFarm:
    """Framework for analyzing and optimized a wind farm layout specific to the
    local wind resource and exclusions for a single reV supply curve point.
    """

    def __init__(self, gid, excl, res, tm_dset, sam_sys_inputs,
                 objective_function, cost_function, min_spacing, ga_time,
                 output_request=('system_capacity', 'cf_mean'),
                 ws_bins=(0.0, 20.0, 5.0), wd_bins=(0.0, 360.0, 45.0),
                 excl_dict=None, inclusion_mask=None,
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

        # TODO
        objective_function :
        cost_function :
        min_spacing :
        ga_time :

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

        self.objective_function = objective_function
        self.cost_function = cost_function
        self.min_spacing = min_spacing
        self.ga_time = ga_time

        self._sam_sys_inputs = sam_sys_inputs
        self._out_req = output_request
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

        self._sc_point = AggSCPoint(gid, excl, res, tm_dset,
                                    excl_dict=excl_dict,
                                    inclusion_mask=inclusion_mask,
                                    resolution=resolution,
                                    excl_area=excl_area,
                                    exclusion_shape=exclusion_shape,
                                    close=close)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.sc_point.close()
        if type is not None:
            raise

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
        return np.sqrt(self.sc_point._excl_area) * 1000.0

    @property
    def sam_sys_inputs(self):
        """Get the SAM windpower system inputs. If the wind plant has not yet
        been optimized, this returns the initial SAM config. If the wind plant
        has been optimized, this returns the final SAM plant config.

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
        pd.Series
        """
        self._meta = pd.Series({'latitude': self.sc_point.latitude,
                                'longitude': self.sc_point.longitude,
                                'timezone': self.sc_point.timezone,
                                'country': self.sc_point.country,
                                'state': self.sc_point.state,
                                'county': self.sc_point.county,
                                'elevation': self.sc_point.elevation},
                               name=self.sc_point.gid)
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
            wd = self.sc_point.h5['winddirection_{}m'.format(self.hub_height)]
            ws = self.sc_point.h5['windspeed_{}m'.format(self.hub_height)]
            temp = self.sc_point.h5['temperature_{}m'.format(self.hub_height)]
            pres = self.sc_point.h5['pressure_{}m'.format(self.hub_height)]

            wd = self.sc_point.mean_wind_dirs(wd)
            ws = self.sc_point.exclusion_weighted_mean(ws)
            temp = self.sc_point.exclusion_weighted_mean(temp)
            pres = self.sc_point.exclusion_weighted_mean(pres)

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

    @property
    def wind_plant_pd(self):
        """reV WindPowerPD compute object based on wind joint probability
        distribution

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
        """reV WindPower compute object based on wind resource timeseries data

        Returns
        -------
        reV.SAM.generation.WindPower
        """
        if self._wind_plant_ts is None:
            self._wind_plant_ts = {}
            for year in self.years:
                i_wp = WindPower(self.res_df[(self.res_df.index.year == year)],
                                 self.meta, self.sam_sys_inputs,
                                 output_request=self._out_req)
                self._wind_plant_ts[year] = i_wp
        return self._wind_plant_ts

    @property
    def plant_optimizer(self):
        """Bespoke plant turbine placement optimizer object.

        Returns
        -------
        PlaceTurbines
        """
        if self._plant_optm is None:
            self._plant_optm = PlaceTurbines(self.wind_plant_pd,
                                             self.objective_function,
                                             self.cost_function,
                                             self.include_mask,
                                             self.pixel_side_length,
                                             self.min_spacing,
                                             self.ga_time)
        return self._plant_optm

    def run_wind_plant_ts(self):
        """Run the wind plant multi-year timeseries analysis and export output
        requests to outputs property.

        Returns
        -------
        outputs : dict
            Output dictionary for the full BespokeSingleFarm object. The
            multi-year timeseries data is also exported to the
            BespokeSingleFarm.outputs property.
        """

        for year, plant in self.wind_plant_ts.items():
            plant.run_gen_and_econ()
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

        return self.outputs

    def run_plant_optimization(self):
        """Run the wind plant layout optimization and export outputs
        to outputs property.

        Returns
        -------
        outputs : dict
            Output dictionary for the full BespokeSingleFarm object. The
            layout optimization output data is also exported to the
            BespokeSingleFarm.outputs property.
        """

        self.plant_optimizer.place_turbines()
        # TODO need to add:
        # total cell area
        # cell capacity density
        self._outputs["turbine_x_coords"] = self.plant_optimizer.turbine_x
        self._outputs["turbine_y_coords"] = self.plant_optimizer.turbine_y
        self._outputs["packed_x"] = self.plant_optimizer.x_locations
        self._outputs["packed_y"] = self.plant_optimizer.y_locations
        self._outputs["n_turbines"] = self.plant_optimizer.nturbs
        self._outputs["system_capacity"] = self.plant_optimizer.capacity
        self._outputs["included_area"] = self.plant_optimizer.area
        self._outputs["bespoke_aep"] = self.plant_optimizer.aep
        self._outputs["objective"] = self.plant_optimizer.objective
        self._outputs["full_polygons"] = self.plant_optimizer.full_polygons
        self._outputs["sam_sys_inputs"] = self.sam_sys_inputs
        self._outputs["packing_polygons"] = \
            self.plant_optimizer.packing_polygons
        self._outputs["annual_cost"] = \
            self.cost_function(self.plant_optimizer.capacity)
        self._outputs["included_area_capacity_density"] =\
            self.plant_optimizer.capacity_density
        return self.outputs

    @property
    def outputs(self):
        """Saved outputs for the single wind farm bespoke optimization.

        Returns
        -------
        dict
        """
        return self._outputs

    @classmethod
    def run(cls, *args, **kwargs):
        """Run the bespoke optimization for a single wind farm.

        Parameters
        ----------
        See the class initialization parameters.

        Returns
        -------
        out : dict
            Output dictionary containing turbine locations and other
            information
        """

        with cls(*args, **kwargs) as bsp_plant:
            out = bsp_plant.run_plant_optimization()
            out = bsp_plant.run_wind_plant_ts()

        return out


class BespokeWindFarms(AbstractAggregation):
    """Framework for analyzing optimized wind farm layouts specific to the
    local wind resource and exclusions for the full reV supply curve grid.
    """

    def __init__(self, excl_fpath, res_fpath, tm_dset,
                 sam_sys_inputs, objective_function, cost_function,
                 min_spacing, ga_time,
                 output_request=('system_capacity', 'cf_mean'),
                 excl_dict=None,
                 area_filter_kernel='queen', min_area=None,
                 resolution=64, excl_area=None, gids=None,
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
        logger.info('Initializing BespokeWindFarms...')
        logger.debug('Exclusion filepath: {}'.format(excl_fpath))
        logger.debug('Exclusion dict: {}'.format(excl_dict))

        super().__init__(excl_fpath, tm_dset, excl_dict=excl_dict,
                         area_filter_kernel=area_filter_kernel,
                         min_area=min_area, resolution=resolution,
                         excl_area=excl_area, gids=gids,
                         pre_extract_inclusions=pre_extract_inclusions)

        self._res_fpath = res_fpath
        self._sam_sys_inputs = sam_sys_inputs
        self._obj_fun = objective_function
        self._cost_fun = cost_function
        self._min_spacing = min_spacing
        self._ga_time = ga_time
        self._output_request = output_request
        self._check_files()

    def _check_files(self):
        """Do a preflight check on input files"""

        if not os.path.exists(self._excl_fpath):
            raise FileNotFoundError('Could not find required exclusions file: '
                                    '{}'.format(self._excl_fpath))

        with h5py.File(self._excl_fpath, 'r') as f:
            if self._tm_dset not in f:
                raise FileInputError('Could not find techmap dataset "{}" '
                                     'in exclusions file: {}'
                                     .format(self._tm_dset,
                                             self._excl_fpath))

        # just check that this file exists, cannot check res_fpath if *glob
        with MultiYearWindResource(self._res_fpath) as f:
            assert any(f.dsets)

    @classmethod
    def run_serial(cls, excl_fpath, res_fpath, tm_dset,
                   sam_sys_inputs, objective_function, cost_function,
                   min_spacing, ga_time,
                   output_request=('system_capacity', 'cf_mean'),
                   excl_dict=None, inclusion_mask=None,
                   area_filter_kernel='queen', min_area=None,
                   resolution=64, excl_area=0.0081, gids=None,
                   ):
        """
        Standalone method to aggregate - can be parallelized.

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
        excl_dict : dict, optional
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
            by default None
        inclusion_mask : np.ndarray, optional
            2D array pre-extracted inclusion mask where 1 is included and 0 is
            excluded. This must be either match the full exclusion shape or
            be a list of single-sc-point exclusion masks corresponding to the
            gids input, by default None
        area_filter_kernel : str, optional
            Contiguous area filter method to use on final exclusions mask,
            by default "queen"
        min_area : float, optional
            Minimum required contiguous area filter in sq-km,
            by default None
        resolution : int, optional
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead,
            by default 0.0081
        excl_area : float, optional
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath,
            by default None
        gids : list, optional
            List of supply curve point gids to get summary for (can use to
            subset if running in parallel), or None for all gids in the SC
            extent, by default None

        Returns
        -------
        out : dict
            Bespoke outputs keyed by sc point gid
        """

        out = {}
        with SupplyCurveExtent(excl_fpath, resolution=resolution) as sc:
            exclusion_shape = sc.exclusions.shape
            if gids is None:
                gids = sc.valid_sc_points(tm_dset)
            elif np.issubdtype(type(gids), np.number):
                gids = [gids]

            slice_lookup = sc.get_slice_lookup(gids)

        cls._check_inclusion_mask(inclusion_mask, gids, exclusion_shape)

        # pre-extract handlers so they are not repeatedly initialized
        file_kwargs = {'excl_dict': excl_dict,
                       'area_filter_kernel': area_filter_kernel,
                       'min_area': min_area,
                       'h5_handler': MultiYearWindResource
                       }

        with AggFileHandler(excl_fpath, res_fpath, **file_kwargs) as fh:
            n_finished = 0
            for gid in gids:
                gid_inclusions = cls._get_gid_inclusion_mask(
                    inclusion_mask, gid, slice_lookup,
                    resolution=resolution)
                try:
                    gid_out = BespokeSingleFarm.run(
                        gid,
                        fh.exclusions,
                        fh.h5,
                        tm_dset,
                        sam_sys_inputs,
                        objective_function,
                        cost_function,
                        min_spacing,
                        ga_time,
                        output_request=output_request,
                        excl_dict=excl_dict,
                        inclusion_mask=gid_inclusions,
                        resolution=resolution,
                        excl_area=excl_area,
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
                    out[gid] = gid_out

        return out

    def run_parallel(self, max_workers=None, sites_per_worker=100):
        """Run the bespoke optimization for many supply curve points in
        parallel.

        Parameters
        ----------
        max_workers : int | None, optional
            Number of cores to run summary on. None is all
            available cpus, by default None
        sites_per_worker : int
            Number of sc_points to summarize on each worker, by default 100

        Returns
        -------
        out : dict
            Bespoke outputs keyed by sc point gid
        """

        chunks = int(np.ceil(len(self.gids) / sites_per_worker))
        chunks = np.array_split(self.gids, chunks)

        logger.info('Running bespoke optimization for '
                    'points {} through {} at a resolution of {} '
                    'on {} cores in {} chunks.'
                    .format(self.gids[0], self.gids[-1], self._resolution,
                            max_workers, len(chunks)))

        slice_lookup = None
        if self._inclusion_mask is not None:
            with SupplyCurveExtent(self._excl_fpath,
                                   resolution=self._resolution) as sc:
                assert sc.exclusions.shape == self._inclusion_mask.shape
                slice_lookup = sc.get_slice_lookup(self.gids)

        futures = []
        out = {}
        n_finished = 0
        loggers = [__name__, 'reV.supply_curve.point_summary', 'reV']
        with SpawnProcessPool(max_workers=max_workers, loggers=loggers) as exe:

            # iterate through split executions, submitting each to worker
            for gid_set in chunks:
                # submit executions and append to futures list
                chunk_incl_masks = None
                if self._inclusion_mask is not None:
                    chunk_incl_masks = {}
                    for gid in gid_set:
                        rs, cs = slice_lookup[gid]
                        chunk_incl_masks[gid] = self._inclusion_mask[rs, cs]

                futures.append(exe.submit(
                    self.run_serial,
                    self._excl_fpath,
                    self._res_fpath,
                    self._tm_dset,
                    self._sam_sys_inputs,
                    self._obj_fun,
                    self._cost_fun,
                    self._min_spacing,
                    self._ga_time,
                    output_request=self._output_request,
                    excl_dict=self._excl_dict,
                    inclusion_mask=chunk_incl_masks,
                    area_filter_kernel=self._area_filter_kernel,
                    min_area=self._min_area,
                    resolution=self._resolution,
                    excl_area=self._excl_area,
                    gids=gid_set))

            # gather results
            for future in as_completed(futures):
                n_finished += 1
                out.update(future.result())
                if n_finished % 10 == 0:
                    mem = psutil.virtual_memory()
                    logger.info('Parallel bespoke futures collected: '
                                '{} out of {}. Memory usage is {:.3f} GB out '
                                'of {:.3f} GB ({:.2f}% utilized).'
                                .format(n_finished, len(chunks),
                                        mem.used / 1e9, mem.total / 1e9,
                                        100 * mem.used / mem.total))

        return out

    @classmethod
    def run(cls, excl_fpath, res_fpath, tm_dset,
            sam_sys_inputs, objective_function, cost_function,
            min_spacing, ga_time,
            output_request=('system_capacity', 'cf_mean'),
            excl_dict=None,
            area_filter_kernel='queen', min_area=None,
            resolution=64, excl_area=None, gids=None,
            pre_extract_inclusions=False, max_workers=None,
            sites_per_worker=100):
        """Run the bespoke wind farm optimization in serial or parallel."""

        bsp = cls(excl_fpath, res_fpath, tm_dset,
                  sam_sys_inputs, objective_function, cost_function,
                  min_spacing, ga_time,
                  output_request=output_request,
                  excl_dict=excl_dict,
                  area_filter_kernel=area_filter_kernel, min_area=min_area,
                  resolution=resolution, excl_area=excl_area, gids=gids,
                  pre_extract_inclusions=pre_extract_inclusions)

        if max_workers == 1:
            out = bsp.run_serial(excl_fpath, res_fpath, tm_dset,
                                 sam_sys_inputs,
                                 objective_function,
                                 cost_function,
                                 min_spacing,
                                 ga_time,
                                 output_request=bsp._output_request,
                                 excl_dict=bsp._excl_dict,
                                 area_filter_kernel=bsp._area_filter_kernel,
                                 min_area=bsp._min_area,
                                 resolution=bsp._resolution,
                                 excl_area=bsp._excl_area,
                                 gids=bsp.gids)
        else:
            out = bsp.run_parallel(max_workers=max_workers,
                                   sites_per_worker=sites_per_worker)

        return out
