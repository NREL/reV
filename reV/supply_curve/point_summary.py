# -*- coding: utf-8 -*-
"""reV supply curve single point data summary framework.

Created on Fri Jun 21 13:24:31 2019

@author: gbuster
"""
import pandas as pd
import numpy as np
from warnings import warn
import logging
from scipy import stats

from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.points import SupplyCurvePoint
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      OutputWarning, FileInputError)


logger = logging.getLogger(__name__)


class SupplyCurvePointSummary(SupplyCurvePoint):
    """Supply curve summary framework with extra methods for summary calc."""

    # technology-dependent power density estimates in MW/km2
    POWER_DENSITY = {'pv': 36, 'wind': 3}

    def __init__(self, gid, excl, gen, tm_dset, gen_index, excl_dict=None,
                 res_class_dset=None, res_class_bin=None, ex_area=0.0081,
                 power_density=None, cf_dset='cf_mean-means',
                 lcoe_dset='lcoe_fcr-means', resolution=64,
                 exclusion_shape=None, close=False, offshore_flags=None,
                 friction_layer=None):
        """
        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        excl : str | ExclusionMask
            Filepath to exclusions h5 or ExclusionMask file handler.
        gen : str | reV.handlers.Outputs
            Filepath to .h5 reV generation output results or reV Outputs file
            handler.
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
            None if excl input is pre-initialized.
        res_class_dset : str | np.ndarray | None
            Dataset in the generation file dictating resource classes.
            Can be pre-extracted resource data in np.ndarray.
            None if no resource classes.
        res_class_bin : list | None
            Two-entry lists dictating the single resource class bin.
            None if no resource classes.
        ex_area : float
            Area of an exclusion cell (square km).
        power_density : float | None | pd.DataFrame
            Constant power density float, None, or opened dataframe with
            (resource) "gid" and "power_density columns".
        cf_dset : str | np.ndarray
            Dataset name from gen containing capacity factor mean values.
            Can be pre-extracted generation output data in np.ndarray.
        lcoe_dset : str | np.ndarray
            Dataset name from gen containing LCOE mean values.
            Can be pre-extracted generation output data in np.ndarray.
        resolution : int | None
            SC resolution, must be input in combination with gid.
        exclusion_shape : tuple
            Shape of the exclusions extent (rows, cols). Inputing this will
            speed things up considerably.
        close : bool
            Flag to close object file handlers on exit.
        offshore_flags : np.ndarray | None
            Array of offshore boolean flags if available from wind generation
            data. None if offshore flag is not available.
        friction_layer : None | FrictionMask
            Friction layer with scalar friction values if valid friction inputs
            were entered. Otherwise, None to not apply friction layer.
        """

        self._res_class_dset = res_class_dset
        self._res_class_bin = res_class_bin
        self._cf_dset = cf_dset
        self._lcoe_dset = lcoe_dset
        self._res_gid_set = None
        self._gen_gid_set = None
        self._mean_res = None
        self._res_data = None
        self._gen_data = None
        self._lcoe_data = None
        self._ex_area = ex_area
        self._pd_obj = None
        self._power_density = power_density
        self._friction_layer = friction_layer

        super().__init__(gid, excl, gen, tm_dset, gen_index,
                         excl_dict=excl_dict, resolution=resolution,
                         exclusion_shape=exclusion_shape,
                         offshore_flags=offshore_flags, close=close)

        self._apply_exclusions()

    def _apply_exclusions(self):
        """Apply exclusions by masking the generation and resource gid arrays.
        This removes all res/gen entries that are masked by the exclusions or
        resource bin."""

        # exclusions mask is False where excluded
        exclude = (self.excl_data == 0).flatten()
        exclude = self._resource_exclusion(exclude)

        self._gen_gids[exclude] = -1
        self._res_gids[exclude] = -1

        # ensure that excluded pixels (including resource exclusions!)
        # has an exclusions multiplier of 0
        self._excl_data[exclude.reshape(self._excl_data.shape)] = 0.0
        self._excl_data_flat = self._excl_data.flatten()

        if (self._gen_gids != -1).sum() == 0:
            msg = ('Supply curve point gid {} is completely excluded for res '
                   'bin: {}'.format(self._gid, self._res_class_bin))
            raise EmptySupplyCurvePointError(msg)

    def _resource_exclusion(self, boolean_exclude):
        """Include the resource exclusion into a pre-existing bool exclusion.

        Parameters
        ----------
        boolean_exclude : np.ndarray
            Boolean exclusion array (True is exclude).

        Returns
        -------
        boolean_exclude : np.ndarray
            Same as input but includes additional exclusions for resource
            outside of current resource class bin.
        """

        if (self._res_class_dset is not None
                and self._res_class_bin is not None):

            rex = ((self.res_data[self._gen_gids]
                    < np.min(self._res_class_bin))
                   | (self.res_data[self._gen_gids]
                      >= np.max(self._res_class_bin)))

            boolean_exclude = (boolean_exclude | rex)

        return boolean_exclude

    @staticmethod
    def ordered_unique(seq):
        """Get a list of unique values in the same order as the input sequence.

        Parameters
        ----------
        seq : list | tuple
            Sequence of values.

        Returns
        -------
        seq : list
            List of unique values in seq input with original order.
        """

        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]

    @property
    def area(self):
        """Get the non-excluded resource area of the supply curve point in the
        current resource class.

        Returns
        -------
        area : float
            Non-excluded resource/generation area in square km.
        """
        return self.excl_data.sum() * self._ex_area

    @property
    def res_data(self):
        """Get the resource data array.

        Returns
        -------
        _res_data : np.ndarray
            Multi-year-mean resource data array for all sites in the
            generation data output file.
        """

        if isinstance(self._res_class_dset, np.ndarray):
            return self._res_class_dset

        else:
            if self._res_data is None:
                if self._res_class_dset in self.gen.dsets:
                    self._res_data = self.gen[self._res_class_dset]

        return self._res_data

    @property
    def gen_data(self):
        """Get the generation capacity factor data array.

        Returns
        -------
        _gen_data : np.ndarray
            Multi-year-mean capacity factor data array for all sites in the
            generation data output file.
        """

        if isinstance(self._cf_dset, np.ndarray):
            return self._cf_dset

        else:
            if self._gen_data is None:
                if self._cf_dset in self.gen.dsets:
                    self._gen_data = self.gen[self._cf_dset]

        return self._gen_data

    @property
    def lcoe_data(self):
        """Get the LCOE data array.

        Returns
        -------
        _lcoe_data : np.ndarray
            Multi-year-mean LCOE data array for all sites in the
            generation data output file.
        """

        if isinstance(self._lcoe_dset, np.ndarray):
            return self._lcoe_dset

        else:
            if self._lcoe_data is None:
                if self._lcoe_dset in self.gen.dsets:
                    self._lcoe_data = self.gen[self._lcoe_dset]

        return self._lcoe_data

    @property
    def latitude(self):
        """Get the SC point latitude"""
        return self.centroid[0]

    @property
    def longitude(self):
        """Get the SC point longitude"""
        return self.centroid[1]

    @property
    def country(self):
        """Get the SC point country based on the resource meta data."""
        country = None
        if 'country' in self.gen.meta:
            country = self.gen.meta.loc[self.gen_gid_set, 'country'].values
            country = stats.mode(country).mode[0]
        return country

    @property
    def state(self):
        """Get the SC point state based on the resource meta data."""
        state = None
        if 'state' in self.gen.meta:
            state = self.gen.meta.loc[self.gen_gid_set, 'state'].values
            state = stats.mode(state).mode[0]
        return state

    @property
    def county(self):
        """Get the SC point county based on the resource meta data."""
        county = None
        if 'county' in self.gen.meta:
            county = self.gen.meta.loc[self.gen_gid_set, 'county'].values
            county = stats.mode(county).mode[0]
        return county

    @property
    def elevation(self):
        """Get the SC point elevation based on the resource meta data."""
        elevation = None
        if 'elevation' in self.gen.meta:
            elevation = self.gen.meta.loc[self.gen_gid_set, 'elevation'].mean()
        return elevation

    @property
    def timezone(self):
        """Get the SC point timezone based on the resource meta data."""
        timezone = None
        if 'timezone' in self.gen.meta:
            timezone = self.gen.meta.loc[self.gen_gid_set, 'timezone'].values
            timezone = stats.mode(timezone).mode[0]
        return timezone

    @property
    def res_gid_set(self):
        """Get list of unique resource gids corresponding to this sc point.

        Returns
        -------
        res_gids : list
            List of resource gids.
        """
        if self._res_gid_set is None:
            self._res_gid_set = self.ordered_unique(self._res_gids)
            if -1 in self._res_gid_set:
                self._res_gid_set.remove(-1)
        return self._res_gid_set

    @property
    def gen_gid_set(self):
        """Get list of unique generation gids corresponding to this sc point.

        Returns
        -------
        gen_gids : list
            List of generation gids.
        """
        if self._gen_gid_set is None:
            self._gen_gid_set = self.ordered_unique(self._gen_gids)
            if -1 in self._gen_gid_set:
                self._gen_gid_set.remove(-1)
        return self._gen_gid_set

    @property
    def gid_counts(self):
        """Get the number of exclusion pixels in each resource/generation gid
        corresponding to this sc point.

        Returns
        -------
        gid_counts : list
            List of exclusion pixels in each resource/generation gid.
        """
        gid_counts = [self.excl_data_flat[(self._res_gids == gid)].sum()
                      for gid in self.res_gid_set]
        return gid_counts

    @property
    def mean_cf(self):
        """Get the mean capacity factor for the non-excluded data. Capacity
        factor is weighted by the exclusions (usually 0 or 1, but 0.5
        exclusions will weight appropriately).

        Returns
        -------
        mean_cf : float | None
            Mean capacity factor value for the non-excluded data.
        """
        mean_cf = None
        if self.gen_data is not None:
            mean_cf = self.exclusion_weighted_mean(self.gen_data)
        return mean_cf

    @property
    def mean_lcoe(self):
        """Get the mean LCOE for the non-excluded data.

        Returns
        -------
        mean_lcoe : float | None
            Mean LCOE value for the non-excluded data.
        """
        mean_lcoe = None
        if self.lcoe_data is not None:
            mean_lcoe = self.exclusion_weighted_mean(self.lcoe_data)
        return mean_lcoe

    @property
    def mean_res(self):
        """Get the mean resource for the non-excluded data.

        Returns
        -------
        mean_res : float | None
            Mean resource for the non-excluded data.
        """
        mean_res = None
        if (self._res_class_dset is not None
                and self._res_class_bin is not None):
            mean_res = self.exclusion_weighted_mean(self.res_data)
        return mean_res

    @property
    def mean_lcoe_friction(self):
        """Get the mean LCOE for the non-excluded data, multiplied by the
        mean_friction scalar value.

        Returns
        -------
        mean_lcoe_friction : float | None
            Mean LCOE value for the non-excluded data multiplied by the
            mean friction scalar value.
        """
        mean_lcoe_friction = None
        if self.mean_lcoe is not None and self.mean_friction is not None:
            mean_lcoe_friction = self.mean_lcoe * self.mean_friction
        return mean_lcoe_friction

    @property
    def mean_friction(self):
        """Get the mean friction scalar for the non-excluded data.

        Returns
        -------
        friction : None | float
            Mean value of the friction data layer for the non-excluded data.
            If friction layer is not input to this class, None is returned.
        """
        friction = None
        if self._friction_layer is not None:
            friction = self.friction_data * self.excl_data
            friction = friction.flatten()[self.bool_mask]
            friction = friction.mean()
        return friction

    @property
    def friction_data(self):
        """Get the friction data for the full SC point (no exclusions)

        Returns
        -------
        friction_data : None | np.ndarray
            2D friction data layer corresponding to the exclusions grid in
            the SC domain. If friction layer is not input to this class,
            None is returned.
        """
        friction_data = None
        if self._friction_layer is not None:
            friction_data = self._friction_layer[self.rows, self.cols]
        return friction_data

    @property
    def power_density(self):
        """Get the estimated power density either from input or infered from
        generation output meta.

        Returns
        -------
        _power_density : float
            Estimated power density in MW/km2
        """

        if self._power_density is None:
            tech = self.gen.meta['reV_tech'][0]
            if tech in self.POWER_DENSITY:
                self._power_density = self.POWER_DENSITY[tech]
            else:
                warn('Could not recognize reV technology in generation meta '
                     'data: "{}". Cannot lookup an appropriate power density '
                     'to calculate SC point capacity.'.format(tech))

        elif isinstance(self._power_density, pd.DataFrame):
            self._pd_obj = self._power_density

            missing = set(self.res_gid_set) - set(self._pd_obj.index.values)
            if any(missing):
                msg = ('Variable power density input is missing the '
                       'following resource GIDs: {}'.format(missing))
                logger.error(msg)
                raise FileInputError(msg)

            pds = self._pd_obj.loc[self._res_gids, 'power_density'].values
            pds = pds.astype(np.float32)
            pds[(self._res_gids == -1)] = 0.0
            pds *= self.excl_data_flat
            denom = self.excl_data_flat[self.bool_mask].sum()
            self._power_density = pds.sum() / denom

        return self._power_density

    @property
    def capacity(self):
        """Get the estimated capacity in MW of the supply curve point in the
        current resource class with the applied exclusions.

        Returns
        -------
        capacity : float
            Estimated capacity in MW of the supply curve point in the
            current resource class with the applied exclusions.
        """

        capacity = None
        if self.power_density is not None:
            capacity = self.area * self.power_density
        return capacity

    def agg_data_layers(self, summary, data_layers):
        """Perform additional data layer aggregation. If there is no valid data
        in the included area, the data layer will be taken from the full SC
        point extent (ignoring exclusions). If there is still no valid data,
        a warning will be raised and the data layer will have a NaN/None value.

        Parameters
        ----------
        summary : dict
            Dictionary of summary outputs for this sc point.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".

        Returns
        -------
        summary : dict
            Dictionary of summary outputs for this sc point. A new entry for
            each data layer is added.
        """

        if data_layers is not None:
            for name, attrs in data_layers.items():

                if 'fobj' not in attrs:
                    with ExclusionLayers(attrs['fpath']) as f:
                        raw = f[attrs['dset'], self.rows, self.cols]
                        nodata = f.get_nodata_value(attrs['dset'])
                else:
                    raw = attrs['fobj'][attrs['dset'], self.rows, self.cols]
                    nodata = attrs['fobj'].get_nodata_value(attrs['dset'])

                data = raw.flatten()[self.bool_mask]
                if nodata is not None:
                    nodata_mask = (data == nodata)

                    # All included extent is nodata.
                    # Reset data from raw without exclusions.
                    if all(nodata_mask):
                        data = raw.flatten()
                        nodata_mask = (data == nodata)

                    data = data[~nodata_mask]

                    if not data.size:
                        data = None
                        w = ('Data layer "{}" has no valid data for '
                             'SC point gid {} at ({}, {})!'
                             .format(name, self._gid, self.latitude,
                                     self.longitude))
                        logger.warning(w)
                        warn(w, OutputWarning)

                if data is not None:
                    if attrs['method'].lower() == 'mode':
                        data = stats.mode(data).mode[0]
                    elif attrs['method'].lower() == 'mean':
                        data = data.mean()
                    else:
                        e = ('Cannot recognize data layer agg method: '
                             '"{}". Can only do mean and mode.'
                             .format(attrs['method']))
                        logger.error(e)
                        raise ValueError(e)

                summary[name] = data

        return summary

    @classmethod
    def summary(cls, gid, excl_fpath, gen_fpath, tm_dset,
                gen_index, args=None, data_layers=None,
                excl_dict=None, res_class_dset=None, res_class_bin=None,
                ex_area=0.0081, power_density=None,
                cf_dset='cf_mean-means', lcoe_dset='lcoe_fcr-means',
                resolution=64, exclusion_shape=None, close=False,
                offshore_flags=None, friction_layer=None):
        """Get a summary dictionary of a single supply curve point.

        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        excl_fpath : str
            Filepath to exclusions geotiff.
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-generation mapping data.
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        args : tuple | list | None
            List of summary arguments to include. None defaults to all
            available args defined in the class attr.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
            None if excl input is pre-initialized.
        res_class_dset : str | np.ndarray | None
            Dataset in the generation file dictating resource classes.
            Can be pre-extracted resource data in np.ndarray.
            None if no resource classes.
        res_class_bin : list | None
            Two-entry lists dictating the single resource class bin.
            None if no resource classes.
        ex_area : float
            Area of an exclusion cell (square km).
        power_density : float | None | pd.DataFrame
            Constant power density float, None, or opened dataframe with
            (resource) "gid" and "power_density columns".
        cf_dset : str | np.ndarray
            Dataset name from gen containing capacity factor mean values.
            Can be pre-extracted generation output data in np.ndarray.
        lcoe_dset : str | np.ndarray
            Dataset name from gen containing LCOE mean values.
            Can be pre-extracted generation output data in np.ndarray.
        resolution : int | None
            SC resolution, must be input in combination with gid.
        exclusion_shape : tuple
            Shape of the exclusions extent (rows, cols). Inputing this will
            speed things up considerably.
        close : bool
            Flag to close object file handlers on exit.
        offshore_flags : np.ndarray | None
            Array of offshore boolean flags if available from wind generation
            data. None if offshore flag is not available.
        friction_layer : None | ExclusionMask
            Friction layer with scalar friction values if valid friction inputs
            were entered. Otherwise, None to not apply friction layer.

        Returns
        -------
        summary : dict
            Dictionary of summary outputs for this sc point.
        """
        kwargs = {"excl_dict": excl_dict, "res_class_dset": res_class_dset,
                  "res_class_bin": res_class_bin, "ex_area": ex_area,
                  "power_density": power_density, "cf_dset": cf_dset,
                  "lcoe_dset": lcoe_dset, "resolution": resolution,
                  "exclusion_shape": exclusion_shape, "close": close,
                  "offshore_flags": offshore_flags,
                  'friction_layer': friction_layer}
        with cls(gid, excl_fpath, gen_fpath, tm_dset, gen_index,
                 **kwargs) as point:

            ARGS = {'res_gids': point.res_gid_set,
                    'gen_gids': point.gen_gid_set,
                    'gid_counts': point.gid_counts,
                    'mean_cf': point.mean_cf,
                    'mean_lcoe': point.mean_lcoe,
                    'mean_res': point.mean_res,
                    'capacity': point.capacity,
                    'area_sq_km': point.area,
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'country': point.country,
                    'state': point.state,
                    'county': point.county,
                    'elevation': point.elevation,
                    'timezone': point.timezone,
                    }

            if friction_layer is not None:
                ARGS['mean_friction'] = point.mean_friction
                ARGS['mean_lcoe_friction'] = point.mean_lcoe_friction

            if args is None:
                args = list(ARGS.keys())

            summary = {}
            for arg in args:
                if arg in ARGS:
                    summary[arg] = ARGS[arg]
                else:
                    warn('Cannot find "{}" as an available SC point summary '
                         'output', OutputWarning)

            summary = point.agg_data_layers(summary, data_layers)

        return summary
