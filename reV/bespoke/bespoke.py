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
from concurrent.futures import as_completed

from reV.bespoke.place_turbines import PlaceTurbines
from reV.SAM.generation import WindPowerPD
from reV.supply_curve.extent import SupplyCurveExtent
from reV.supply_curve.points import AggregationSupplyCurvePoint as AggSCPoint
from reV.supply_curve.aggregation import AbstractAggregation, AggFileHandler
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      FileInputError)
from reV.utilities import log_versions

from rex.multi_year_resource import MultiYearWindResource
from rex.utilities.loggers import log_mem
from rex.joint_pd.joint_pd import JointPD
from rex.utilities.execution import SpawnProcessPool

logger = logging.getLogger(__name__)


class BespokeSingleFarm:
    """Framework for analyzing an optimized wind farm layout specific to the
    local wind resource and exclusions for a single reV supply curve point.
    """

    @classmethod
    def run(cls, gid, excl, res, tm_dset, ws_dset, wd_dset, sam_sys_inputs,
            objective_function, cost_function, min_spacing, ga_time,
            output_request=('system_capacity', 'cf_mean'),
            ws_bins=(0.0, 20.0, 5.0),
            wd_bins=(0.0, 360.0, 45.0),
            excl_dict=None, inclusion_mask=None,
            resolution=64, excl_area=None, exclusion_shape=None, close=True,
            ):
        """Run the bespoke optimization for a single wind farm.

        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        excl : str | ExclusionMask
            Filepath to exclusions h5 or ExclusionMask file handler.
        agg_h5 : str | Resource
            Filepath to .h5 file to aggregate or Resource handler
        tm_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.
        ws_dset : str
            Windspeed dataset at a target hub height e.g. windspeed_88m. The
            software will interpolate available data to the desired hub height.
        wd_dset : str
            Winddirection dataset at a target hub height e.g.
            winddirection_88m. The software will interpolate available data to
            the desired hub height.
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

        Returns
        -------
        out : dict
            Output dictionary containing turbine locations and other
            information
        """
        kws = {'excl_dict': excl_dict,
               'inclusion_mask': inclusion_mask,
               'resolution': resolution,
               'excl_area': excl_area,
               'exclusion_shape': exclusion_shape,
               'close': close,
               }
        with AggSCPoint(gid, excl, res, tm_dset, **kws) as point:

            wd = point.h5[wd_dset]
            ws = point.h5[ws_dset]

            wd = point.mean_wind_dirs(wd)
            ws = point.exclusion_weighted_mean(ws)

            meta = pd.Series({'timezone': point.timezone,
                              'country': point.country,
                              'state': point.state,
                              'county': point.county,
                              'elevation': point.elevation},
                             name=point.gid)

            ws_bins = JointPD._make_bins(*ws_bins)
            wd_bins = JointPD._make_bins(*wd_bins)

            out = np.histogram2d(ws, wd, bins=(ws_bins, wd_bins))
            wind_dist, ws_edges, wd_edges = out
            wind_dist /= wind_dist.sum()

            wind_plant = WindPowerPD(ws_edges, wd_edges, wind_dist,
                                     meta, sam_sys_inputs,
                                     output_request=output_request)

            pixel_side_length = np.sqrt(point._excl_area) * 1000.0  # in m
            place_turbines = PlaceTurbines(wind_plant, objective_function,
                                           cost_function, point.include_mask,
                                           pixel_side_length, min_spacing,
                                           ga_time)
            place_turbines.place_turbines()

            # TODO need to add:
            # total cell area
            # cell capacity density
            out = {}
            out["turbine_x_coords"] = place_turbines.turbine_x
            out["turbine_y_coords"] = place_turbines.turbine_y
            out["packed_x"] = place_turbines.x_locations
            out["packed_y"] = place_turbines.y_locations
            out["nturbs"] = place_turbines.nturbs
            out["plant_capacity"] = place_turbines.capacity
            out["non_excluded_area"] = place_turbines.area
            out["non_excluded_capacity_density"] =\
                place_turbines.capacity_density
            out["aep"] = place_turbines.aep
            out["objective"] = place_turbines.objective
            out["annual_cost"] = cost_function(place_turbines.capacity)
#            out["ws_sample_points"] = ws_sample_points
#            out["wd_sample_points"] = wd_sample_points
#            out["wind_dist"] = sam_sys_inputs["wind_dist"]
            out["full_polygons"] = place_turbines.full_polygons
            out["packing_polygons"] = place_turbines.packing_polygons
            out["sam_sys_inputs"] = wind_plant.sam_sys_inputs

            return out


class BespokeWindFarms(AbstractAggregation):
    """Framework for analyzing optimized wind farm layouts specific to the
    local wind resource and exclusions for the full reV supply curve grid.
    """

    def __init__(self, excl_fpath, res_fpath, tm_dset, hub_height,
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
        hub_height : int
            Wind turbine hub height to analyze. Windspeed and direction
            will be taken at this height. The software will interpolate to a
            desired hub height using the available data.
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
        self._hh = int(hub_height)
        self._ws_dset = 'windspeed_{}m'.format(self._hh)
        self._wd_dset = 'winddirection_{}m'.format(self._hh)
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
    def run_serial(cls, excl_fpath, res_fpath, tm_dset, ws_dset, wd_dset,
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
        ws_dset : str
            Windspeed dataset at a target hub height e.g. windspeed_88m. The
            software will interpolate available data to the desired hub height.
        wd_dset : str
            Winddirection dataset at a target hub height e.g.
            winddirection_88m. The software will interpolate available data to
            the desired hub height.
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
                        ws_dset,
                        wd_dset,
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
                    self._ws_dset,
                    self._wd_dset,
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
    def run(cls, excl_fpath, res_fpath, tm_dset, hub_height,
            sam_sys_inputs, objective_function, cost_function,
            min_spacing, ga_time,
            output_request=('system_capacity', 'cf_mean'),
            excl_dict=None,
            area_filter_kernel='queen', min_area=None,
            resolution=64, excl_area=None, gids=None,
            pre_extract_inclusions=False, max_workers=None,
            sites_per_worker=100):
        """Run the bespoke wind farm optimization in serial or parallel."""

        bsp = cls(excl_fpath, res_fpath, tm_dset, hub_height,
                  sam_sys_inputs, objective_function, cost_function,
                  min_spacing, ga_time,
                  output_request=output_request,
                  excl_dict=excl_dict,
                  area_filter_kernel=area_filter_kernel, min_area=min_area,
                  resolution=resolution, excl_area=excl_area, gids=gids,
                  pre_extract_inclusions=pre_extract_inclusions)

        if max_workers == 1:
            out = bsp.run_serial(excl_fpath, res_fpath, tm_dset, bsp._ws_dset,
                                 bsp._wd_dset,
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
