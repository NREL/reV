# -*- coding: utf-8 -*-
"""
reV bespoke wind plant analysis tools
"""
# TODO update docstring
# TODO check on outputs
# TODO move sample bounds code
# TODO passing in one cell rather than cropping one big cell
import h5py
import logging
import pandas as pd
import numpy as np
import os

from reV.bespoke.place_turbines import PlaceTurbines
from reV.SAM.generation import WindPowerPD
from reV.supply_curve.extent import SupplyCurveExtent
from reV.supply_curve.points import AggregationSupplyCurvePoint as AggSCPoint
from reV.supply_curve.aggregation import AbstractAggregation, AggFileHandler
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      FileInputError)
from reV.utilities import log_versions

from rex.joint_pd.joint_pd import JointPD
from rex.multi_year_resource import MultiYearWindResource
from rex.utilities.loggers import log_mem

logger = logging.getLogger(__name__)


class BespokeSingleFarm:
    """Framework for analyzing an optimized wind farm layout specific to the
    local wind resource and exclusions for a single reV supply curve point.
    """

    @classmethod
    def run(cls, gid, excl, res, tm_dset, ws_dset, wd_dset, sam_sys_inputs,
            objective_function, cost_function, min_spacing, ga_time,
            output_request=('capacity', 'annual_energy', 'capacity_factor'),
            ws_sample_points=(5.0, 25.0, 5.0),
            wd_sample_points=(0.0, 315.0, 45.0),
            excl_dict=None, inclusion_mask=None,
            resolution=64, excl_area=None, exclusion_shape=None, close=True,
            wind_farm_wake_model=2):
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
        excl_area : float
            Area of an exclusion cell (square km).
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

            ws_sample_points = JointPD._make_bins(*ws_sample_points)
            wd_sample_points = JointPD._make_bins(*wd_sample_points)

            # TODO go over this with Grant
            ws_step = ws_sample_points[1] - ws_sample_points[0]
            ws_edges = ws_sample_points - ws_step / 2.0
            ws_edges = np.append(ws_edges, np.array(ws_sample_points[-1]
                                 + ws_step / 2.0))

            wd_step = wd_sample_points[1] - wd_sample_points[0]
            wd_edges = wd_sample_points - wd_step / 2.0
            wd_edges = np.append(wd_edges, np.array(wd_sample_points[-1]
                                 + wd_step / 2.0))
            # Get the overhangs
            negative_overhang = wd_edges[0]
            positive_overhang = wd_edges[-1] - 360.0
            # Need potentially to wrap high angle direction to negative
            # for correct binning
            if negative_overhang < 0:
                # print("Correcting negative Overhang:%.1f" %
                # negative_overhang)
                wd = np.where(
                    wd >= 360.0 + negative_overhang,
                    wd - 360.0,
                    wd,
                )
            # Check on other side
            if positive_overhang > 0:
                # print("Correcting positive Overhang:%.1f" %
                # positive_overhang)
                wd = np.where(
                    wd <= positive_overhang, wd + 360.0, wd
                )

            out = np.histogram2d(ws, wd, bins=(ws_edges, wd_edges))
            wind_dist, ws_edges, wd_edges = out
            wind_dist /= wind_dist.sum()

            meta = pd.Series({'timezone': point.timezone,
                              'country': point.country,
                              'state': point.state,
                              'county': point.county,
                              'elevation': point.elevation},
                             name=point.gid)

            wind_plant = WindPowerPD(ws_sample_points, wd_sample_points,
                                     wind_dist, meta, sam_sys_inputs,
                                     output_request=output_request)

            wind_plant.sam_sys_inputs["wind_farm_wake_model"] =\
                wind_farm_wake_model

            place_turbines = PlaceTurbines(wind_plant, objective_function,
                                           cost_function, point.exclusions,
                                           min_spacing, ga_time)
            place_turbines.place_turbines()

            # TODO need to add:
            # total cell area
            # cell capacity density
            out = {}
            out["turbine_x_coords"] = place_turbines.turbine_x
            out["turbine_y_coords"] = place_turbines.turbine_y
            out["nturbs"] = place_turbines.nturbs
            out["plant_capacity"] = place_turbines.capacity
            out["non_excluded_area"] = place_turbines.area
            out["non_excluded_capacity_density"] =\
                place_turbines.capacity_density
            out["aep"] = place_turbines.aep
            out["objective"] = place_turbines.objective
            out["annual_cost"] = cost_function(place_turbines.capacity)
            out["ws_sample_points"] = ws_sample_points
            out["wd_sample_points"] = wd_sample_points
            out["wind_dist"] = wind_dist
            out["boundary_polys"] = place_turbines.safe_polygons
            out["sam_sys_inputs"] = wind_plant.sam_sys_inputs
            out["meta"] = meta

            return out


class BespokeWindFarms(AbstractAggregation):
    """Framework for analyzing optimized wind farm layouts specific to the
    local wind resource and exclusions for the full reV supply curve grid.
    """

    def __init__(self, excl_fpath, res_fpath, tm_dset, hub_height,
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
        self._check_files()

    def _check_files(self):
        """Do a preflight check on input files"""

        if not os.path.exists(self._excl_fpath):
            raise FileNotFoundError('Could not find required exclusions file: '
                                    '{}'.format(self._excl_fpath))

        if not os.path.exists(self._res_fpath):
            raise FileNotFoundError('Could not find required h5 file: '
                                    '{}'.format(self._res_fpath))

        with h5py.File(self._excl_fpath, 'r') as f:
            if self._tm_dset not in f:
                raise FileInputError('Could not find techmap dataset "{}" '
                                     'in exclusions file: {}'
                                     .format(self._tm_dset,
                                             self._excl_fpath))

        with MultiYearWindResource(self._res_fpath) as f:
            for dset in (self._ws_dset, self._wd_dset):
                if dset not in f:
                    raise FileInputError('Could not find provided dataset "{}"'
                                         ' in h5 file: {}'
                                         .format(dset, self._res_fpath))

    @classmethod
    def run_serial(cls, excl_fpath, res_fpath, tm_dset, ws_dset, wd_dset,
                   sam_sys_inputs, objective_function, cost_function,
                   min_spacing, ga_time,
                   output_request=('capacity', 'annual_energy',
                                   'capacity_factor'),
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
                    logger.debug('Serial aggregation: '
                                 '{} out of {} points complete'
                                 .format(n_finished, len(gids)))
                    log_mem(logger)
                    out[gid] = gid_out

        return out
