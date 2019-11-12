# -*- coding: utf-8 -*-
"""reV supply curve single point data summary framework.

Created on Fri Jun 21 13:24:31 2019

@author: gbuster
"""
import numpy as np
from warnings import warn
import logging
from scipy import stats

from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.points import SupplyCurvePoint
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      OutputWarning)


logger = logging.getLogger(__name__)


class SupplyCurvePointSummary(SupplyCurvePoint):
    """Supply curve summary framework with extra methods for summary calc."""

    # technology-dependent power density estimates in MW/km2
    POWER_DENSITY = {'pv': 36, 'wind': 3}

    def __init__(self, gid, excl, gen, tm_dset, gen_index, excl_dict=None,
                 res_class_dset=None, res_class_bin=None, ex_area=0.0081,
                 power_density=None, cf_dset='cf_mean-means',
                 lcoe_dset='lcoe_fcr-means', resolution=64,
                 exclusion_shape=None, close=False):
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
        power_density : float | None
            Power density in MW/km2. None will attempt to infer power density
            from the generation meta data technology.
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
        self._power_density = power_density

        super().__init__(gid, excl, gen, tm_dset, gen_index,
                         excl_dict=excl_dict, resolution=resolution,
                         exclusion_shape=exclusion_shape, close=close)

        self._apply_exclusions()

    def _apply_exclusions(self):
        """Apply exclusions by masking the generation and resource gid arrays.
        This removes all res/gen entries that are masked by the exclusions or
        resource bin."""

        # exclusions mask is False where excluded
        # pylint: disable-msg=C0121
        exclude = (self.excl_data == False).flatten()  # noqa: E712
        exclude = self._resource_exclusion(exclude)

        self._gen_gids[exclude] = -1
        self._res_gids[exclude] = -1

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
        return self.mask.sum() * self._ex_area

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
    def res_gid_set(self):
        """Get the list of resource gids corresponding to this sc point.

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
        """Get the list of generation gids corresponding to this sc point.

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
        return [(self._res_gids == gid).sum() for gid in self.res_gid_set]

    @property
    def mean_cf(self):
        """Get the mean capacity factor for the non-excluded data.

        Returns
        -------
        mean_cf : float | None
            Mean capacity factor value for the non-excluded data.
        """
        mean_cf = None
        if self.gen_data is not None:
            mean_cf = self.gen_data[self._gen_gids[self.mask]].mean()
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
            mean_lcoe = self.lcoe_data[self._gen_gids[self.mask]].mean()
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
            mean_res = self.res_data[self._gen_gids[self.mask]].mean()
        return mean_res

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
        """Perform additional data layer aggregation.

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
                        data = f[attrs['dset'], self.rows, self.cols]
                else:
                    data = attrs['fobj'][attrs['dset'], self.rows, self.cols]

                data = data.flatten()[self.mask]
                if attrs['method'].lower() == 'mode':
                    data = stats.mode(data)[0][0]
                elif attrs['method'].lower() == 'mean':
                    data = data.mean()
                else:
                    raise ValueError('Cannot recognize data layer agg method: '
                                     '"{}". Can only do mean and mode.'
                                     .format(attrs['method']))

                summary[name] = data

        return summary

    @classmethod
    def summary(cls, gid, excl_fpath, gen_fpath, tm_dset,
                gen_index, args=None, data_layers=None, **kwargs):
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
        kwargs : dict
            Keyword args to init the SC point.

        Returns
        -------
        summary : dict
            Dictionary of summary outputs for this sc point.
        """

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
                    }

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
