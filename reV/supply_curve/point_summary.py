# -*- coding: utf-8 -*-
"""reV supply curve single point data summary framework.

Created on Fri Jun 21 13:24:31 2019

@author: gbuster
"""
import logging
import numpy as np
import pandas as pd
from scipy import stats
from warnings import warn

from reV.econ.economies_of_scale import EconomiesOfScale
from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.points import GenerationSupplyCurvePoint
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      OutputWarning, FileInputError,
                                      DataShapeError)

from rex.utilities.utilities import jsonify_dict

logger = logging.getLogger(__name__)


class SupplyCurvePointSummary(GenerationSupplyCurvePoint):
    """Supply curve summary framework with extra methods for summary calc."""

    # technology-dependent power density estimates in MW/km2
    POWER_DENSITY = {'pv': 36, 'wind': 3}

    def __init__(self, gid, excl, gen, tm_dset, gen_index,
                 excl_dict=None, inclusion_mask=None,
                 res_class_dset=None, res_class_bin=None, excl_area=0.0081,
                 power_density=None, cf_dset='cf_mean-means',
                 lcoe_dset='lcoe_fcr-means', h5_dsets=None, resolution=64,
                 exclusion_shape=None, close=False, friction_layer=None):
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
        inclusion_mask : np.ndarray
            2D array pre-extracted inclusion mask where 1 is included and 0 is
            excluded. The shape of this will be checked against the input
            resolution.
        res_class_dset : str | np.ndarray | None
            Dataset in the generation file dictating resource classes.
            Can be pre-extracted resource data in np.ndarray.
            None if no resource classes.
        res_class_bin : list | None
            Two-entry lists dictating the single resource class bin.
            None if no resource classes.
        excl_area : float
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
        h5_dsets : None | list | dict
            Optional list of dataset names to summarize from the gen/econ h5
            files. Can also be pre-extracted data dictionary where keys are
            the dataset names and values are the arrays of data from the
            h5 files.
        resolution : int | None
            SC resolution, must be input in combination with gid.
        exclusion_shape : tuple
            Shape of the exclusions extent (rows, cols). Inputing this will
            speed things up considerably.
        close : bool
            Flag to close object file handlers on exit.
        friction_layer : None | FrictionMask
            Friction layer with scalar friction values if valid friction inputs
            were entered. Otherwise, None to not apply friction layer.
        """

        self._res_class_dset = res_class_dset
        self._res_class_bin = res_class_bin
        self._cf_dset = cf_dset
        self._lcoe_dset = lcoe_dset
        self._h5_dsets = h5_dsets
        self._mean_res = None
        self._res_data = None
        self._gen_data = None
        self._lcoe_data = None
        self._pd_obj = None
        self._power_density = power_density
        self._friction_layer = friction_layer

        super().__init__(gid, excl, gen, tm_dset, gen_index,
                         excl_dict=excl_dict,
                         inclusion_mask=inclusion_mask,
                         resolution=resolution,
                         excl_area=excl_area,
                         exclusion_shape=exclusion_shape,
                         close=close)

        self._apply_exclusions()

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
                if self._res_class_dset in self.gen.datasets:
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
                if self._cf_dset in self.gen.datasets:
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
                if self._lcoe_dset in self.gen.datasets:
                    self._lcoe_data = self.gen[self._lcoe_dset]

        return self._lcoe_data

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
        if self._res_class_dset is not None:
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
            friction = self.friction_data.flatten()[self.bool_mask].mean()

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

            pds = self._pd_obj.loc[self._res_gids[self.bool_mask],
                                   'power_density'].values
            pds = pds.astype(np.float32)
            pds *= self.include_mask_flat[self.bool_mask]
            denom = self.include_mask_flat[self.bool_mask].sum()
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

    @property
    def h5_dsets_data(self):
        """Get any additional/supplemental h5 dataset data to summarize.

        Returns
        -------
        h5_dsets_data : dict | None

        """

        _h5_dsets_data = None

        if isinstance(self._h5_dsets, (list, tuple)):
            _h5_dsets_data = {}
            for dset in self._h5_dsets:
                if dset in self.gen.datasets:
                    _h5_dsets_data[dset] = self.gen[dset]

        elif isinstance(self._h5_dsets, dict):
            _h5_dsets_data = self._h5_dsets

        elif self._h5_dsets is not None:
            e = ('Cannot recognize h5_dsets input type, should be None, '
                 'a list of dataset names, or a dictionary or '
                 'pre-extracted data. Received: {} {}'
                 .format(type(self._h5_dsets), self._h5_dsets))
            logger.error(e)
            raise TypeError(e)

        return _h5_dsets_data

    @property
    def mean_h5_dsets_data(self):
        """Get the mean supplemental h5 datasets data (optional)

        Returns
        -------
        mean_h5_dsets_data : dict | None
            Mean dataset values for the non-excluded data for the optional
            h5_dsets input.
        """
        _mean_h5_dsets_data = None
        if self.h5_dsets_data is not None:
            _mean_h5_dsets_data = {}
            for dset, arr in self.h5_dsets_data.items():
                _mean_h5_dsets_data[dset] = self.exclusion_weighted_mean(arr)

        return _mean_h5_dsets_data

    @staticmethod
    def _mode(data):
        """
        Compute the mode of the data vector and return a single value

        Parameters
        ----------
        data : ndarray
            data layer vector to compute mode for

        Returns
        -------
        float | int
            Mode of data
        """
        if not data.size:
            return None
        else:
            return stats.mode(data).mode[0]

    @staticmethod
    def _categorize(data, incl_mult):
        """
        Extract the sum of inclusion scalar values (where 1 is
        included, 0 is excluded, and 0.7 is included with 70 percent of
        available land) for each unique (categorical value) in data

        Parameters
        ----------
        data : ndarray
            Vector of categorical values
        incl_mult : ndarray
            Vector of inclusion values

        Returns
        -------
        str
            Jsonified string of the dictionary mapping categorical values to
            total inclusions
        """

        data = {category: float(incl_mult[(data == category)].sum())
                for category in np.unique(data)}
        data = jsonify_dict(data)

        return data

    @classmethod
    def _agg_data_layer_method(cls, data, incl_mult, method):
        """Aggregate the data array using specified method.

        Parameters
        ----------
        data : np.ndarray | None
            Data array that will be flattened and operated on using method.
            This must be the included data. Exclusions should be applied
            before this method.
        incl_mult : np.ndarray | None
            Scalar exclusion data for methods with exclusion-weighted
            aggregation methods. Shape must match input data.
        method : str
            Aggregation method (mode, mean, max, min, sum, category)

        Returns
        -------
        data : float | int | str | None
            Result of applying method to data.
        """
        method_func = {'mode': cls._mode,
                       'mean': np.mean,
                       'max': np.max,
                       'min': np.min,
                       'sum': np.sum,
                       'category': cls._categorize}

        if data is not None:
            method = method.lower()
            if method not in method_func:
                e = ('Cannot recognize data layer agg method: '
                     '"{}". Can only {}'.format(method, list(method_func)))
                logger.error(e)
                raise ValueError(e)

            if len(data.shape) > 1:
                data = data.flatten()

            if data.shape != incl_mult.shape:
                e = ('Cannot aggregate data with shape that doesnt '
                     'match excl mult!')
                logger.error(e)
                raise DataShapeError(e)

            if method == 'category':
                data = method_func['category'](data, incl_mult)
            elif method in ['mean', 'sum']:
                data = data * incl_mult
                data = method_func[method](data)
            else:
                data = method_func[method](data)

        return data

    def _apply_exclusions(self):
        """Apply exclusions by masking the generation and resource gid arrays.
        This removes all res/gen entries that are masked by the exclusions or
        resource bin."""

        # exclusions mask is False where excluded
        exclude = self.include_mask_flat == 0
        exclude = self._resource_exclusion(exclude)

        self._gen_gids[exclude] = -1
        self._res_gids[exclude] = -1

        # ensure that excluded pixels (including resource exclusions!)
        # has an exclusions multiplier of 0
        exclude = exclude.reshape(self.include_mask.shape)
        self._incl_mask[exclude] = 0.0
        self._incl_mask = self._incl_mask.flatten()

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

            rex = self.res_data[self._gen_gids]
            rex = ((rex < np.min(self._res_class_bin))
                   | (rex >= np.max(self._res_class_bin)))

            boolean_exclude = (boolean_exclude | rex)

        return boolean_exclude

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
                incl_mult = self.include_mask_flat[self.bool_mask].copy()

                if nodata is not None:
                    valid_data_mask = (data != nodata)
                    data = data[valid_data_mask]
                    incl_mult = incl_mult[valid_data_mask]

                    if not data.size:
                        m = ('Data layer "{}" has no valid data for '
                             'SC point gid {} because of exclusions '
                             'and/or nodata values in the data layer.'
                             .format(name, self._gid))
                        logger.debug(m)

                data = self._agg_data_layer_method(data, incl_mult,
                                                   attrs['method'])
                summary[name] = data

        return summary

    def point_summary(self, args=None):
        """
        Get a summary dictionary of a single supply curve point.

        Parameters
        ----------
        args : tuple | list | None
            List of summary arguments to include. None defaults to all
            available args defined in the class attr.

        Returns
        -------
        summary : dict
            Dictionary of summary outputs for this sc point.
        """

        ARGS = {'res_gids': self.res_gid_set,
                'gen_gids': self.gen_gid_set,
                'gid_counts': self.gid_counts,
                'n_gids': self.n_gids,
                'mean_cf': self.mean_cf,
                'mean_lcoe': self.mean_lcoe,
                'mean_res': self.mean_res,
                'capacity': self.capacity,
                'area_sq_km': self.area,
                'latitude': self.latitude,
                'longitude': self.longitude,
                'country': self.country,
                'state': self.state,
                'county': self.county,
                'elevation': self.elevation,
                'timezone': self.timezone,
                }

        if self.offshore is not None:
            ARGS['offshore'] = self.offshore

        if self._friction_layer is not None:
            ARGS['mean_friction'] = self.mean_friction
            ARGS['mean_lcoe_friction'] = self.mean_lcoe_friction

        if self._h5_dsets is not None:
            for dset, data in self.mean_h5_dsets_data.items():
                ARGS['mean_{}'.format(dset)] = data

        if args is None:
            args = list(ARGS.keys())

        summary = {}
        for arg in args:
            if arg in ARGS:
                summary[arg] = ARGS[arg]
            else:
                warn('Cannot find "{}" as an available SC self summary '
                     'output', OutputWarning)

        return summary

    @staticmethod
    def economies_of_scale(cap_cost_scale, summary):
        """Apply economies of scale to this point summary

        Parameters
        ----------
        cap_cost_scale : str
            LCOE scaling equation to implement "economies of scale".
            Equation must be in python string format and return a scalar
            value to multiply the capital cost by. Independent variables in
            the equation should match the names of the columns in the reV
            supply curve aggregation table.
        summary : dict
            Dictionary of summary outputs for this sc point.

        Returns
        -------
        summary : dict
            Dictionary of summary outputs for this sc point.
        """

        eos = EconomiesOfScale(cap_cost_scale, summary)
        summary['raw_lcoe'] = eos.raw_lcoe
        summary['mean_lcoe'] = eos.scaled_lcoe

        return summary

    @classmethod
    def summarize(cls, gid, excl_fpath, gen_fpath, tm_dset, gen_index,
                  excl_dict=None, inclusion_mask=None,
                  res_class_dset=None, res_class_bin=None,
                  excl_area=0.0081, power_density=None,
                  cf_dset='cf_mean-means', lcoe_dset='lcoe_fcr-means',
                  h5_dsets=None, resolution=64, exclusion_shape=None,
                  close=False, friction_layer=None, args=None,
                  data_layers=None, cap_cost_scale=None):
        """Get a summary dictionary of a single supply curve point.

        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        excl_fpath : str
            Filepath to exclusions h5.
        gen_fpath : str
            Filepath to .h5 reV generation output results.
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
        inclusion_mask : np.ndarray
            2D array pre-extracted inclusion mask where 1 is included and 0 is
            excluded. The shape of this will be checked against the input
            resolution.
        res_class_dset : str | np.ndarray | None
            Dataset in the generation file dictating resource classes.
            Can be pre-extracted resource data in np.ndarray.
            None if no resource classes.
        res_class_bin : list | None
            Two-entry lists dictating the single resource class bin.
            None if no resource classes.
        excl_area : float
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
        h5_dsets : None | list | dict
            Optional list of dataset names to summarize from the gen/econ h5
            files. Can also be pre-extracted data dictionary where keys are
            the dataset names and values are the arrays of data from the
            h5 files.
        resolution : int | None
            SC resolution, must be input in combination with gid.
        exclusion_shape : tuple
            Shape of the exclusions extent (rows, cols). Inputing this will
            speed things up considerably.
        close : bool
            Flag to close object file handlers on exit.
        friction_layer : None | FrictionMask
            Friction layer with scalar friction values if valid friction inputs
            were entered. Otherwise, None to not apply friction layer.
        args : tuple | list, optional
            List of summary arguments to include. None defaults to all
            available args defined in the class attr, by default None
        data_layers : dict, optional
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath", by default None
        cap_cost_scale : str | None
            Optional LCOE scaling equation to implement "economies of scale".
            Equations must be in python string format and return a scalar
            value to multiply the capital cost by. Independent variables in
            the equation should match the names of the columns in the reV
            supply curve aggregation table.

        Returns
        -------
        summary : dict
            Dictionary of summary outputs for this sc point.
        """
        kwargs = {"excl_dict": excl_dict,
                  "inclusion_mask": inclusion_mask,
                  "res_class_dset": res_class_dset,
                  "res_class_bin": res_class_bin,
                  "excl_area": excl_area,
                  "power_density": power_density,
                  "cf_dset": cf_dset,
                  "lcoe_dset": lcoe_dset,
                  "h5_dsets": h5_dsets,
                  "resolution": resolution,
                  "exclusion_shape": exclusion_shape,
                  "close": close,
                  'friction_layer': friction_layer}

        with cls(gid, excl_fpath, gen_fpath, tm_dset, gen_index,
                 **kwargs) as point:
            summary = point.point_summary(args=args)

            if data_layers is not None:
                summary = point.agg_data_layers(summary, data_layers)

            if cap_cost_scale is not None:
                summary = point.economies_of_scale(cap_cost_scale, summary)

        return summary
