# -*- coding: utf-8 -*-
"""
reV supply curve points frameworks.
"""

import logging
from abc import ABC
from warnings import warn

import numpy as np
import pandas as pd
from rex.multi_time_resource import MultiTimeResource
from rex.resource import BaseResource, Resource
from rex.utilities.utilities import jsonify_dict

from reV.econ.economies_of_scale import EconomiesOfScale
from reV.econ.utilities import lcoe_fcr
from reV.handlers.exclusions import LATITUDE, LONGITUDE, ExclusionLayers
from reV.supply_curve.exclusions import ExclusionMask, ExclusionMaskFromDict
from reV.utilities import ResourceMetaField, SupplyCurveField
from reV.utilities.exceptions import (
    DataShapeError,
    EmptySupplyCurvePointError,
    FileInputError,
    InputWarning,
    OutputWarning,
    SupplyCurveInputError,
)

logger = logging.getLogger(__name__)


class AbstractSupplyCurvePoint(ABC):
    """
    Abstract SC point based on only the point gid, SC shape, and resolution.
    """

    def __init__(self, gid, exclusion_shape, resolution=64):
        """
        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        exclusion_shape : tuple
            Shape of the full exclusions extent (rows, cols).
        resolution : int
            Number of exclusion points per SC point along an axis.
            This number**2 is the total number of exclusion points per
            SC point.
        """

        self._gid = gid
        self._resolution = resolution
        self._rows = self._cols = self._sc_row_ind = self._sc_col_ind = None
        self._parse_sc_row_col_ind(resolution, exclusion_shape)
        self._parse_slices(resolution, exclusion_shape)

    @staticmethod
    def _ordered_unique(seq):
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

    def _parse_sc_row_col_ind(self, resolution, exclusion_shape):
        """Parse SC row and column index.

        Parameters
        ----------
        resolution : int | None
            SC resolution, must be input in combination with gid.
        exclusion_shape : tuple
            Shape of the exclusions extent (rows, cols).
        """
        n_sc_cols = int(np.ceil(exclusion_shape[1] / resolution))

        self._sc_row_ind = self._gid // n_sc_cols
        self._sc_col_ind = self._gid % n_sc_cols

    def _parse_slices(self, resolution, exclusion_shape):
        """Parse row and column resource/generation grid slices.

        Parameters
        ----------
        resolution : int | None
            SC resolution, must be input in combination with gid.
        exclusion_shape : tuple
            Shape of the exclusions extent (rows, cols).
        """
        inds = self.get_agg_slices(self._gid, exclusion_shape, resolution)
        self._rows, self._cols = inds

    @property
    def gid(self):
        """Supply curve point gid"""
        return self._gid

    @property
    def sc_point_gid(self):
        """
        Supply curve point gid

        Returns
        -------
        int
        """
        return self._gid

    @property
    def sc_row_ind(self):
        """int: Supply curve row index"""
        return self._sc_row_ind

    @property
    def sc_col_ind(self):
        """int: Supply curve column index"""
        return self._sc_col_ind

    @property
    def resolution(self):
        """Get the supply curve grid aggregation resolution"""
        return self._resolution

    @property
    def rows(self):
        """Get the rows of the exclusions layer associated with this SC point.

        Returns
        -------
        rows : slice
            Row slice to index the high-res layer (exclusions layer) for the
            gid in the agg layer (supply curve layer).
        """
        return self._rows

    @property
    def cols(self):
        """Get the cols of the exclusions layer associated with this SC point.

        Returns
        -------
        cols : slice
            Column slice to index the high-res layer (exclusions layer) for the
            gid in the agg layer (supply curve layer).
        """
        return self._cols

    @staticmethod
    def get_agg_slices(gid, shape, resolution):
        """Get the row, col slices of an aggregation gid.

        Parameters
        ----------
        gid : int
            Gid of interest in the aggregated layer.
        shape : tuple
            (row, col) shape tuple of the underlying high-res layer.
        resolution : int
            Resolution of the aggregation: number of pixels in 1D being
            aggregated.

        Returns
        -------
        row_slice : slice
            Row slice to index the high-res layer for the gid in the agg layer.
        col_slice : slice
            Col slice to index the high-res layer for the gid in the agg layer.
        """

        nrows = int(np.ceil(shape[0] / resolution))
        ncols = int(np.ceil(shape[1] / resolution))
        super_shape = (nrows, ncols)
        arr = np.arange(nrows * ncols).reshape(super_shape)
        try:
            loc = np.where(arr == gid)
            row = loc[0][0]
            col = loc[1][0]
        except IndexError as exc:
            msg = (
                "Gid {} out of bounds for extent shape {} and "
                "resolution {}.".format(gid, shape, resolution)
            )
            raise IndexError(msg) from exc

        if row + 1 != nrows:
            row_slice = slice(row * resolution, (row + 1) * resolution)
        else:
            row_slice = slice(row * resolution, shape[0])

        if col + 1 != ncols:
            col_slice = slice(col * resolution, (col + 1) * resolution)
        else:
            col_slice = slice(col * resolution, shape[1])

        return row_slice, col_slice


class SupplyCurvePoint(AbstractSupplyCurvePoint):
    """Generic single SC point based on exclusions, resolution, and techmap"""

    def __init__(
        self,
        gid,
        excl,
        tm_dset,
        excl_dict=None,
        inclusion_mask=None,
        resolution=64,
        excl_area=None,
        exclusion_shape=None,
        close=True,
        zone_mask=None,
    ):
        """
        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        excl : str | list | tuple | ExclusionMask
            Filepath(s) to exclusions h5 or ExclusionMask file handler.
        tm_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.
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
        zone_mask : np.ndarray | None, optional
            2D array defining zone within the supply curve to be evaluated,
            where values of 0 or False will be excluded. The shape of this
            array will be checked against the input resolution. If not
            specified, no zone mask will be applied.
        """

        self._excl_dict = excl_dict
        self._close = close
        self._excl_fpath, self._excls = self._parse_excl_file(excl)

        if exclusion_shape is None:
            exclusion_shape = self.exclusions.shape

        super().__init__(gid, exclusion_shape, resolution=resolution)

        self._gids = self._parse_techmap(tm_dset)
        self._h5_gids = self._gids
        self._h5_gid_set = None

        self._incl_mask = inclusion_mask
        self._incl_mask_flat = None
        if inclusion_mask is not None:
            msg = (
                "Bad inclusion mask input shape of {} with stated "
                "resolution of {}".format(inclusion_mask.shape, resolution)
            )
            assert len(inclusion_mask.shape) == 2, msg
            assert inclusion_mask.shape[0] <= resolution, msg
            assert inclusion_mask.shape[1] <= resolution, msg
            assert inclusion_mask.size == len(self._gids), msg
            self._incl_mask = inclusion_mask.copy()

        self._zone_mask = zone_mask
        if zone_mask is not None:
            msg = (
                "Bad zone mask input shape of {} with stated "
                "resolution of {}".format(zone_mask.shape, resolution)
            )
            assert len(zone_mask.shape) == 2, msg
            assert zone_mask.shape[0] <= resolution, msg
            assert zone_mask.shape[1] <= resolution, msg
            assert zone_mask.size == len(self._gids), msg
            self._zone_mask = zone_mask.copy()

        self._centroid = None
        self._excl_area = excl_area
        self._check_excl()

    @staticmethod
    def _parse_excl_file(excl):
        """Parse excl filepath input or handler object and set to attrs.

        Parameters
        ----------
        excl : str | ExclusionMask
            Filepath to exclusions geotiff or ExclusionMask handler

        Returns
        -------
        excl_fpath : str | list | tuple
            Filepath(s) for exclusions file
        exclusions : ExclusionMask | None
            Exclusions mask if input is already an open handler or None if it
            is to be lazy instantiated.
        """

        if isinstance(excl, (str, list, tuple)):
            excl_fpath = excl
            exclusions = None
        elif isinstance(excl, ExclusionMask):
            excl_fpath = excl.excl_h5.h5_file
            exclusions = excl
        else:
            raise SupplyCurveInputError(
                "SupplyCurvePoints needs an "
                "exclusions file path, or "
                "ExclusionMask handler but "
                "received: {}".format(type(excl))
            )

        return excl_fpath, exclusions

    def _parse_techmap(self, tm_dset):
        """Parse data from the tech map file (exclusions to resource mapping).
        Raise EmptySupplyCurvePointError if there are no valid resource points
        in this SC point.

        Parameters
        ----------
        tm_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.

        Returns
        -------
        res_gids : np.ndarray
            1D array with length == number of exclusion points. reV resource
            gids (native resource index) from the original resource data
            corresponding to the tech exclusions.
        """
        res_gids = self.exclusions.excl_h5[tm_dset, self.rows, self.cols]
        res_gids = res_gids.astype(np.int32).flatten()

        if (res_gids != -1).sum() == 0:
            emsg = (
                "Supply curve point gid {} has no viable exclusion points "
                'based on exclusions file: "{}"'.format(
                    self._gid, self._excl_fpath
                )
            )
            raise EmptySupplyCurvePointError(emsg)

        return res_gids

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if type is not None:
            raise

    def close(self):
        """Close all file handlers."""
        if self._close:
            if self._excls is not None:
                self._excls.close()

    @property
    def exclusions(self):
        """Get the exclusions object.

        Returns
        -------
        _excls : ExclusionMask
            ExclusionMask h5 handler object.
        """
        if self._excls is None:
            self._excls = ExclusionMaskFromDict(
                self._excl_fpath, layers_dict=self._excl_dict
            )

        return self._excls

    @property
    def centroid(self):
        """Get the supply curve point centroid coordinate.

        Returns
        -------
        centroid : tuple
            SC point centroid (lat, lon).
        """

        if self._centroid is None:
            lats = self.exclusions.excl_h5[LATITUDE, self.rows, self.cols]
            lons = self.exclusions.excl_h5[LONGITUDE, self.rows, self.cols]
            self._centroid = (lats.mean(), lons.mean())

        return self._centroid

    @property
    def pixel_area(self):
        """The area in km2 of a single exclusion pixel. If this value was not
        provided on initialization, it is determined from the profile of the
        exclusion file.

        Returns
        -------
        float
        """
        if self._excl_area is None:
            with ExclusionLayers(self._excl_fpath) as f:
                self._excl_area = f.pixel_area
        return self._excl_area

    @property
    def area(self):
        """Get the non-excluded resource area of the supply curve point in the
        current resource class.

        Returns
        -------
        area : float
            Non-excluded resource/generation area in square km.
        """
        mask = self._gids != -1
        area = np.sum(self.include_mask_flat[mask]) * self.pixel_area

        return area

    @property
    def latitude(self):
        """Get the SC point latitude"""
        return self.centroid[0]

    @property
    def longitude(self):
        """Get the SC point longitude"""
        return self.centroid[1]

    @property
    def n_gids(self):
        """
        Get the total number of not fully excluded pixels associated with the
        available resource/generation gids at the given sc gid.

        Returns
        -------
        n_gids : list
        """
        mask = self._gids != -1
        n_gids = np.sum(self.include_mask_flat[mask] > 0)

        return n_gids

    @property
    def zone_mask(self):
        """
        Get the 2D zone mask, where 1 is included and 0 is excluded.

        Returns
        -------
        np.ndarray
        """
        return self._zone_mask

    @property
    def include_mask(self):
        """Get the 2D inclusion mask (normalized with expected range: [0, 1]
        where 1 is included and 0 is excluded).

        Returns
        -------
        np.ndarray
        """

        if self._incl_mask is None:
            self._incl_mask = self.exclusions[self.rows, self.cols]

            # make sure exclusion pixels outside resource extent are excluded
            out_of_extent = self._gids.reshape(self._incl_mask.shape) == -1
            self._incl_mask[out_of_extent] = 0.0

            if self._incl_mask.max() > 1:
                w = (
                    "Exclusions data max value is > 1: {}".format(
                        self._incl_mask.max()
                    ),
                    InputWarning,
                )
                logger.warning(w)
                warn(w)

        if self.zone_mask is not None:
            out_of_zone = self.zone_mask == 0
            self._incl_mask[out_of_zone] = 0.0

        return self._incl_mask

    @property
    def include_mask_flat(self):
        """Get the flattened inclusion mask (normalized with expected
        range: [0, 1] where 1 is included and 0 is excluded).

        Returns
        -------
        np.ndarray
        """

        if self._incl_mask_flat is None:
            self._incl_mask_flat = self.include_mask.flatten()

        return self._incl_mask_flat

    @property
    def bool_mask(self):
        """Get a boolean inclusion mask (True if excl point is not excluded).

        Returns
        -------
        mask : np.ndarray
            Mask with length equal to the flattened exclusion shape
        """
        return self._gids != -1

    @property
    def h5(self):
        """
        placeholder for h5 Resource handler object
        """

    @property
    def h5_gid_set(self):
        """Get list of unique h5 gids corresponding to this sc point.

        Returns
        -------
        h5_gids : list
            List of h5 gids.
        """
        if self._h5_gid_set is None:
            self._h5_gid_set = self._ordered_unique(self._h5_gids)
            if -1 in self._h5_gid_set:
                self._h5_gid_set.remove(-1)

        return self._h5_gid_set

    @property
    def summary(self):
        """
        Placeholder for Supply curve point's meta data summary
        """

    def _check_excl(self):
        """
        Check to see if supply curve point is fully excluded
        """

        if all(self.include_mask_flat[self.bool_mask] == 0):
            msg = "Supply curve point gid {} is completely excluded!".format(
                self._gid
            )
            raise EmptySupplyCurvePointError(msg)

    def exclusion_weighted_mean(self, arr, drop_nan=True):
        """
        Calc the exclusions-weighted mean value of an array of resource data.

        Parameters
        ----------
        arr : np.ndarray
            Array of resource data.
        drop_nan : bool
            Flag to drop nan values from the mean calculation (only works for
            1D arr input, profiles should not have NaN's)

        Returns
        -------
        mean : float | np.ndarray
            Mean of arr masked by the binary exclusions then weighted by
            the non-zero exclusions. This will be a 1D numpy array if the
            input data is a 2D numpy array (averaged along axis=1)
        """

        if len(arr.shape) == 2:
            x = arr[:, self._gids[self.bool_mask]].astype("float32")
            incl = self.include_mask_flat[self.bool_mask]
            x *= incl
            mean = x.sum(axis=1) / incl.sum()

        else:
            x = arr[self._gids[self.bool_mask]].astype("float32")
            incl = self.include_mask_flat[self.bool_mask]

            if np.isnan(x).all():
                return np.nan
            if drop_nan and np.isnan(x).any():
                nan_mask = np.isnan(x)
                x = x[~nan_mask]
                incl = incl[~nan_mask]

            x *= incl
            mean = x.sum() / incl.sum()

        return mean

    def mean_wind_dirs(self, arr):
        """
        Calc the mean wind directions at every time-step

        Parameters
        ----------
        arr : np.ndarray
            Array of wind direction data.

        Returns
        -------
        mean_wind_dirs : np.ndarray | float
            Mean wind direction of arr masked by the binary exclusions
        """
        incl = self.include_mask_flat[self.bool_mask]
        gids = self._gids[self.bool_mask]
        if len(arr.shape) == 2:
            arr_slice = (slice(None), gids)
            ax = 1

        else:
            arr_slice = gids
            ax = 0

        angle = np.radians(arr[arr_slice], dtype=np.float32)
        sin = np.mean(np.sin(angle) * incl, axis=ax)
        cos = np.mean(np.cos(angle) * incl, axis=ax)

        mean_wind_dirs = np.degrees(np.arctan2(sin, cos))
        mask = mean_wind_dirs < 0
        mean_wind_dirs[mask] += 360

        return mean_wind_dirs

    def aggregate(self, arr):
        """
        Calc sum (aggregation) of the resource data.

        Parameters
        ----------
        arr : np.ndarray
            Array of resource data.

        Returns
        -------
        agg : float
            Sum of arr masked by the binary exclusions
        """
        if len(arr.shape) == 2:
            x = arr[:, self._gids[self.bool_mask]].astype("float32")
            ax = 1
        else:
            x = arr[self._gids[self.bool_mask]].astype("float32")
            ax = 0

        x *= self.include_mask_flat[self.bool_mask]
        agg = x.sum(axis=ax)

        return agg

    @classmethod
    def sc_mean(
        cls,
        gid,
        excl,
        tm_dset,
        data,
        excl_dict=None,
        resolution=64,
        exclusion_shape=None,
        close=True,
    ):
        """
        Compute exclusions weight mean for the sc point from data

        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        excl : str | ExclusionMask
            Filepath to exclusions h5 or ExclusionMask file handler.
        tm_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.
        data : ndarray | ResourceDataset
            Array of data or open dataset handler to apply exclusions too
        excl_dict : dict | None
            Dictionary of exclusion keyword arugments of the format
            {layer_dset_name: {kwarg: value}} where layer_dset_name is a
            dataset in the exclusion h5 file and kwarg is a keyword argument to
            the reV.supply_curve.exclusions.LayerMask class.
            None if excl input is pre-initialized.
        resolution : int
            Number of exclusion points per SC point along an axis.
            This number**2 is the total number of exclusion points per
            SC point.
        exclusion_shape : tuple
            Shape of the full exclusions extent (rows, cols). Inputing this
            will speed things up considerably.
        close : bool
            Flag to close object file handlers on exit

        Returns
        -------
        ndarray
            Exclusions weighted means of data for supply curve point
        """
        kwargs = {
            "excl_dict": excl_dict,
            "resolution": resolution,
            "exclusion_shape": exclusion_shape,
            "close": close,
        }
        with cls(gid, excl, tm_dset, **kwargs) as point:
            means = point.exclusion_weighted_mean(data)

        return means

    @classmethod
    def sc_sum(
        cls,
        gid,
        excl,
        tm_dset,
        data,
        excl_dict=None,
        resolution=64,
        exclusion_shape=None,
        close=True,
    ):
        """
        Compute the aggregate (sum) of data for the sc point

        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        excl : str | ExclusionMask
            Filepath to exclusions h5 or ExclusionMask file handler.
        tm_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.
        data : ndarray | ResourceDataset
            Array of data or open dataset handler to apply exclusions too
        excl_dict : dict | None
            Dictionary of exclusion keyword arugments of the format
            {layer_dset_name: {kwarg: value}} where layer_dset_name is a
            dataset in the exclusion h5 file and kwarg is a keyword argument to
            the reV.supply_curve.exclusions.LayerMask class.
            None if excl input is pre-initialized.
        resolution : int
            Number of exclusion points per SC point along an axis.
            This number**2 is the total number of exclusion points per
            SC point.
        exclusion_shape : tuple
            Shape of the full exclusions extent (rows, cols). Inputing this
            will speed things up considerably.
        close : bool
            Flag to close object file handlers on exit.

        Returns
        -------
        ndarray
            Sum / aggregation of data for supply curve point
        """
        kwargs = {
            "excl_dict": excl_dict,
            "resolution": resolution,
            "exclusion_shape": exclusion_shape,
            "close": close,
        }
        with cls(gid, excl, tm_dset, **kwargs) as point:
            agg = point.aggregate(data)

        return agg

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
        # pd series is more flexible with non-numeric than stats mode
        return pd.Series(data).mode().values[0]

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

        data = {
            category: float(incl_mult[(data == category)].sum())
            for category in np.unique(data)
        }
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
        method_func = {
            "mode": cls._mode,
            "mean": np.mean,
            "max": np.max,
            "min": np.min,
            "sum": np.sum,
            "category": cls._categorize,
        }

        if data is not None:
            method = method.lower()
            if method not in method_func:
                e = (
                    "Cannot recognize data layer agg method: "
                    '"{}". Can only {}'.format(method, list(method_func))
                )
                logger.error(e)
                raise ValueError(e)

            if len(data.shape) > 1:
                data = data.flatten()

            if data.shape != incl_mult.shape:
                e = (
                    "Cannot aggregate data with shape that doesnt "
                    "match excl mult!"
                )
                logger.error(e)
                raise DataShapeError(e)

            if method == "category":
                data = method_func["category"](data, incl_mult)
            elif method in ["mean", "sum"]:
                data = data * incl_mult
                data = method_func[method](data)
            else:
                data = method_func[method](data)

        return data

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
                excl_fp = attrs.get("fpath", self._excl_fpath)
                if excl_fp != self._excl_fpath:
                    fh = ExclusionLayers(attrs["fpath"])
                else:
                    fh = self.exclusions.excl_h5

                raw = fh[attrs["dset"], self.rows, self.cols]
                nodata = fh.get_nodata_value(attrs["dset"])

                data = raw.flatten()[self.bool_mask]
                incl_mult = self.include_mask_flat[self.bool_mask].copy()

                if nodata is not None:
                    valid_data_mask = data != nodata
                    data = data[valid_data_mask]
                    incl_mult = incl_mult[valid_data_mask]

                    if not data.size:
                        m = (
                            'Data layer "{}" has no valid data for '
                            "SC point gid {} because of exclusions "
                            "and/or nodata values in the data layer.".format(
                                name, self._gid
                            )
                        )
                        logger.debug(m)

                data = self._agg_data_layer_method(
                    data, incl_mult, attrs["method"]
                )
                summary[name] = data

                if excl_fp != self._excl_fpath:
                    fh.close()

        return summary


class AggregationSupplyCurvePoint(SupplyCurvePoint):
    """Generic single SC point to aggregate data from an h5 file."""

    def __init__(
        self,
        gid,
        excl,
        agg_h5,
        tm_dset,
        excl_dict=None,
        inclusion_mask=None,
        resolution=64,
        excl_area=None,
        exclusion_shape=None,
        close=True,
        gen_index=None,
        apply_exclusions=True,
        zone_mask=None,
    ):
        """
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
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        apply_exclusions : bool
            Flag to apply exclusions to the resource / generation gid's on
            initialization.
        zone_mask : np.ndarray | None, optional
            2D array defining zone within the supply curve to be evaluated,
            where values of 0 or False will be excluded. The shape of this
            array will be checked against the input resolution. If not
            specified, no zone mask will be applied.
        """
        super().__init__(
            gid,
            excl,
            tm_dset,
            excl_dict=excl_dict,
            inclusion_mask=inclusion_mask,
            resolution=resolution,
            excl_area=excl_area,
            exclusion_shape=exclusion_shape,
            close=close,
            zone_mask=zone_mask,
        )

        self._h5_fpath, self._h5 = self._parse_h5_file(agg_h5)

        if gen_index is not None:
            self._gids, _ = self._map_gen_gids(self._gids, gen_index)

        self._h5_gids = self._gids

        if (self._h5_gids != -1).sum() == 0:
            emsg = (
                "Supply curve point gid {} has no viable exclusion "
                'points based on exclusions file: "{}"'.format(
                    self._gid, self._excl_fpath
                )
            )
            raise EmptySupplyCurvePointError(emsg)

        if apply_exclusions:
            self._apply_exclusions()

    @staticmethod
    def _parse_h5_file(h5):
        """
        Parse .h5 filepath input or handler object and set to attrs.

        Parameters
        ----------
        h5 : str | Resource
            Filepath to .h5 file to aggregate or Resource handler

        Returns
        -------
        h5_fpath : str
            Filepath for .h5 file to aggregate
        h5 : Resource | None
            Resource if input is already an open handler or None if it
            is to be lazy instantiated.
        """

        if isinstance(h5, str):
            h5_fpath = h5
            h5 = None
        elif issubclass(h5.__class__, BaseResource):
            h5_fpath = h5.h5_file
        elif issubclass(h5.__class__, MultiTimeResource):
            h5_fpath = h5.h5_files
        else:
            raise SupplyCurveInputError(
                "SupplyCurvePoints needs a "
                ".h5 file path, or "
                "Resource handler but "
                "received: {}".format(type(h5))
            )

        return h5_fpath, h5

    def _apply_exclusions(self):
        """Apply exclusions by masking the generation and resource gid arrays.
        This removes all res/gen entries that are masked by the exclusions or
        resource bin."""

        # exclusions mask is False where excluded
        exclude = self.include_mask_flat == 0

        self._gids[exclude] = -1
        self._h5_gids[exclude] = -1

        if (self._gids != -1).sum() == 0:
            msg = "Supply curve point gid {} is completely excluded!".format(
                self._gid
            )
            raise EmptySupplyCurvePointError(msg)

    def close(self):
        """Close all file handlers."""
        if self._close:
            if self._excls is not None:
                self._excls.close()

            if self._h5 is not None:
                self._h5.close()

    @staticmethod
    def _map_gen_gids(res_gids, gen_index):
        """
        Map resource gids from techmap to gen gids in .h5 source file

        Parameters
        ----------
        res_gids : ndarray
            resource gids from techmap
        gen_index : ndarray
            Equivalent gen gids to resource gids

        Returns
        -------
        gen_gids : ndarray
            gen gid to excl mapping
        res_gids : ndarray
            updated resource gid to excl mapping
        """
        mask = (res_gids >= len(gen_index)) | (res_gids == -1)
        res_gids[mask] = -1
        gen_gids = gen_index[res_gids]
        gen_gids[mask] = -1
        res_gids[(gen_gids == -1)] = -1

        return gen_gids, res_gids

    @property
    def h5(self):
        """
        h5 Resource handler object

        Returns
        -------
        _h5 : Resource
            Resource h5 handler object.
        """
        if self._h5 is None and "*" in self._h5_fpath:
            self._h5 = MultiTimeResource(self._h5_fpath)
        elif self._h5 is None:
            self._h5 = Resource(self._h5_fpath)

        return self._h5

    @property
    def country(self):
        """Get the SC point country based on the resource meta data."""
        country = None
        county_not_none = self.county is not None
        if ResourceMetaField.COUNTRY in self.h5.meta and county_not_none:
            # make sure country and county are coincident
            counties = self.h5.meta.loc[
                self.h5_gid_set, ResourceMetaField.COUNTY
            ].values
            iloc = np.where(counties == self.county)[0][0]
            country = self.h5.meta.loc[
                self.h5_gid_set, ResourceMetaField.COUNTRY
            ].values
            country = country[iloc]

        elif ResourceMetaField.COUNTRY in self.h5.meta:
            country = self.h5.meta.loc[
                self.h5_gid_set, ResourceMetaField.COUNTRY
            ].mode()
            country = country.values[0]

        return country

    @property
    def state(self):
        """Get the SC point state based on the resource meta data."""
        state = None
        if ResourceMetaField.STATE in self.h5.meta and self.county is not None:
            # make sure state and county are coincident
            counties = self.h5.meta.loc[
                self.h5_gid_set, ResourceMetaField.COUNTY
            ].values
            iloc = np.where(counties == self.county)[0][0]
            state = self.h5.meta.loc[
                self.h5_gid_set, ResourceMetaField.STATE
            ].values
            state = state[iloc]

        elif ResourceMetaField.STATE in self.h5.meta:
            state = self.h5.meta.loc[
                self.h5_gid_set, ResourceMetaField.STATE
            ].mode()
            state = state.values[0]

        return state

    @property
    def county(self):
        """Get the SC point county based on the resource meta data."""
        county = None
        if ResourceMetaField.COUNTY in self.h5.meta:
            county = self.h5.meta.loc[
                self.h5_gid_set, ResourceMetaField.COUNTY
            ].mode()
            county = county.values[0]

        return county

    @property
    def elevation(self):
        """Get the SC point elevation based on the resource meta data."""
        elevation = None
        if ResourceMetaField.ELEVATION in self.h5.meta:
            elevation = self.h5.meta.loc[
                self.h5_gid_set, ResourceMetaField.ELEVATION
            ].mean()

        return elevation

    @property
    def timezone(self):
        """Get the SC point timezone based on the resource meta data."""
        timezone = None
        county_not_none = self.county is not None
        if ResourceMetaField.TIMEZONE in self.h5.meta and county_not_none:
            # make sure timezone flag and county are coincident
            counties = self.h5.meta.loc[
                self.h5_gid_set, ResourceMetaField.COUNTY
            ].values
            iloc = np.where(counties == self.county)[0][0]
            timezone = self.h5.meta.loc[
                self.h5_gid_set, ResourceMetaField.TIMEZONE
            ].values
            timezone = timezone[iloc]

        elif ResourceMetaField.TIMEZONE in self.h5.meta:
            timezone = self.h5.meta.loc[
                self.h5_gid_set, ResourceMetaField.TIMEZONE
            ].mode()
            timezone = timezone.values[0]

        return timezone

    @property
    def offshore(self):
        """Get the SC point offshore flag based on the resource meta data
        (if offshore column is present)."""
        offshore = None
        county_not_none = self.county is not None
        if ResourceMetaField.OFFSHORE in self.h5.meta and county_not_none:
            # make sure offshore flag and county are coincident
            counties = self.h5.meta.loc[
                self.h5_gid_set, ResourceMetaField.COUNTY
            ].values
            iloc = np.where(counties == self.county)[0][0]
            offshore = self.h5.meta.loc[
                self.h5_gid_set, ResourceMetaField.OFFSHORE
            ].values
            offshore = offshore[iloc]

        elif ResourceMetaField.OFFSHORE in self.h5.meta:
            offshore = self.h5.meta.loc[
                self.h5_gid_set, ResourceMetaField.OFFSHORE
            ].mode()
            offshore = offshore.values[0]

        return offshore

    @property
    def gid_counts(self):
        """Get the sum of the inclusion values in each resource/generation gid
        corresponding to this sc point. The sum of the gid counts can be less
        than the value provided by n_gids if fractional exclusion/inclusions
        are provided.

        Returns
        -------
        gid_counts : list
        """
        gid_counts = [
            self.include_mask_flat[(self._h5_gids == gid)].sum()
            for gid in self.h5_gid_set
        ]

        return gid_counts

    @property
    def summary(self):
        """
        Supply curve point's meta data summary

        Returns
        -------
        pandas.Series
            List of supply curve point's meta data
        """
        meta = {
            SupplyCurveField.SC_POINT_GID: self.sc_point_gid,
            SupplyCurveField.SOURCE_GIDS: self.h5_gid_set,
            SupplyCurveField.GID_COUNTS: self.gid_counts,
            SupplyCurveField.N_GIDS: self.n_gids,
            SupplyCurveField.AREA_SQ_KM: self.area,
            SupplyCurveField.LATITUDE: self.latitude,
            SupplyCurveField.LONGITUDE: self.longitude,
            SupplyCurveField.COUNTRY: self.country,
            SupplyCurveField.STATE: self.state,
            SupplyCurveField.COUNTY: self.county,
            SupplyCurveField.ELEVATION: self.elevation,
            SupplyCurveField.TIMEZONE: self.timezone,
        }
        meta = pd.Series(meta)

        return meta

    @classmethod
    def run(
        cls,
        gid,
        excl,
        agg_h5,
        tm_dset,
        *agg_dset,
        agg_method="mean",
        excl_dict=None,
        inclusion_mask=None,
        resolution=64,
        excl_area=None,
        exclusion_shape=None,
        close=True,
        gen_index=None,
    ):
        """
        Compute exclusions weight mean for the sc point from data

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
        agg_dset : str
            Dataset to aggreate, can supply multiple datasets or no datasets.
            The datasets should be scalar values for each site. This method
            cannot aggregate timeseries data.
        agg_method : str
            Aggregation method, either mean or sum/aggregate
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
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.

        Returns
        -------
        out : dict
            Given datasets and meta data aggregated to supply curve points
        """
        if isinstance(agg_dset, str):
            agg_dset = (agg_dset,)

        kwargs = {
            "excl_dict": excl_dict,
            "inclusion_mask": inclusion_mask,
            "resolution": resolution,
            "excl_area": excl_area,
            "exclusion_shape": exclusion_shape,
            "close": close,
            "gen_index": gen_index,
        }

        with cls(gid, excl, agg_h5, tm_dset, **kwargs) as point:
            if agg_method.lower().startswith("mean"):
                agg_method = point.exclusion_weighted_mean
            elif agg_method.lower().startswith(("sum", "agg")):
                agg_method = point.aggregate
            elif "wind_dir" in agg_method.lower():
                agg_method = point.mean_wind_dirs
            else:
                msg = (
                    "Aggregation method must be either mean, "
                    "sum/aggregate, or wind_dir"
                )
                logger.error(msg)
                raise ValueError(msg)

            out = {"meta": point.summary}

            for dset in agg_dset:
                ds = point.h5.open_dataset(dset)
                out[dset] = agg_method(ds)

        return out


class GenerationSupplyCurvePoint(AggregationSupplyCurvePoint):
    """Supply curve point summary framework that ties a reV SC point to its
    respective generation and resource data."""

    # technology-dependent power density estimates in MW/km2
    POWER_DENSITY = {"pv": 36, "wind": 3}

    def __init__(
        self,
        gid,
        excl,
        gen,
        tm_dset,
        gen_index,
        excl_dict=None,
        inclusion_mask=None,
        res_class_dset=None,
        res_class_bin=None,
        excl_area=None,
        power_density=None,
        cf_dset="cf_mean-means",
        lcoe_dset="lcoe_fcr-means",
        h5_dsets=None,
        resolution=64,
        exclusion_shape=None,
        close=False,
        friction_layer=None,
        recalc_lcoe=True,
        apply_exclusions=True,
        zone_mask=None,
    ):
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
            Dictionary of exclusion keyword arugments of the format
            {layer_dset_name: {kwarg: value}} where layer_dset_name is a
            dataset in the exclusion h5 file and kwarg is a keyword argument to
            the reV.supply_curve.exclusions.LayerMask class.
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
        excl_area : float | None, optional
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath, by default None
        power_density : float | None | pd.DataFrame
            Constant power density float, None, or opened dataframe with
            (resource) "gid" and "power_density columns".
        cf_dset : str | np.ndarray
            Dataset name from gen containing capacity factor mean values.
            This name is used to infer AC capacity factor dataset for
            solar runs (i.e. the AC vsersion of "cf_mean-means" would
            be inferred to be "cf_mean_ac-means"). This input can also
            be pre-extracted generation output data in np.ndarray, in
            which case all DC solar outputs are set to `None`.
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
        recalc_lcoe : bool
            Flag to re-calculate the LCOE from the multi-year mean capacity
            factor and annual energy production data. This requires several
            datasets to be aggregated in the gen input: system_capacity,
            fixed_charge_rate, capital_cost, fixed_operating_cost,
            and variable_operating_cost.
        apply_exclusions : bool
            Flag to apply exclusions to the resource / generation gid's on
            initialization.
        zone_mask : np.ndarray | None, optional
            2D array defining zone within the supply curve to be evaluated,
            where values of 0 or False will be excluded. The shape of this
            array will be checked against the input resolution. If not
            specified, no zone mask will be applied.
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
        self._power_density = self._power_density_ac = power_density
        self._friction_layer = friction_layer
        self._recalc_lcoe = recalc_lcoe
        self._ssc = None
        self._slk = {}

        super().__init__(
            gid,
            excl,
            gen,
            tm_dset,
            excl_dict=excl_dict,
            inclusion_mask=inclusion_mask,
            resolution=resolution,
            excl_area=excl_area,
            exclusion_shape=exclusion_shape,
            close=close,
            apply_exclusions=False,
            zone_mask=zone_mask,
        )

        self._res_gid_set = None
        self._gen_gid_set = None

        self._gen_fpath, self._gen = self._h5_fpath, self._h5

        self._gen_gids, self._res_gids = self._map_gen_gids(
            self._gids, gen_index
        )
        self._gids = self._gen_gids
        if (self._gen_gids != -1).sum() == 0:
            emsg = (
                "Supply curve point gid {} has no viable exclusion "
                'points based on exclusions file: "{}"'.format(
                    self._gid, self._excl_fpath
                )
            )
            raise EmptySupplyCurvePointError(emsg)

        if apply_exclusions:
            self._apply_exclusions()

    def exclusion_weighted_mean(self, flat_arr):
        """Calc the exclusions-weighted mean value of a flat array of gen data.

        Parameters
        ----------
        flat_arr : np.ndarray
            Flattened array of resource/generation/econ data. Must be
            index-able with the self._gen_gids array (must be a 1D array with
            an entry for every site in the generation extent).

        Returns
        -------
        mean : float
            Mean of flat_arr masked by the binary exclusions then weighted by
            the non-zero exclusions.
        """
        x = flat_arr[self._gen_gids[self.bool_mask]].astype("float32")
        incl = self.include_mask_flat[self.bool_mask]
        x *= incl
        mean = x.sum() / incl.sum()

        return mean

    @property
    def gen(self):
        """Get the generation output object.

        Returns
        -------
        _gen : Resource
            reV generation Resource object
        """
        if self._gen is None:
            self._gen = Resource(self._gen_fpath, str_decode=False)

        return self._gen

    @property
    def res_gid_set(self):
        """Get list of unique resource gids corresponding to this sc point.

        Returns
        -------
        res_gids : list
            List of resource gids.
        """
        if self._res_gid_set is None:
            self._res_gid_set = self._ordered_unique(self._res_gids)
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
            self._gen_gid_set = self._ordered_unique(self._gen_gids)
            if -1 in self._gen_gid_set:
                self._gen_gid_set.remove(-1)

        return self._gen_gid_set

    @property
    def h5_gid_set(self):
        """Get list of unique h5 gids corresponding to this sc point.
        Same as gen_gid_set

        Returns
        -------
        h5_gids : list
            List of h5 gids.
        """
        return self.gen_gid_set

    @property
    def gid_counts(self):
        """Get the number of exclusion pixels in each resource/generation gid
        corresponding to this sc point.

        Returns
        -------
        gid_counts : list
            List of exclusion pixels in each resource/generation gid.
        """
        gid_counts = [
            self.include_mask_flat[(self._res_gids == gid)].sum()
            for gid in self.res_gid_set
        ]

        return gid_counts

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

        if self._gen_data is None:
            if self._cf_dset in self.gen.datasets:
                self._gen_data = self.gen[self._cf_dset]

        return self._gen_data

    @property
    def gen_ac_data(self):
        """Get the generation ac capacity factor data array.

        This output is only not `None` for solar runs where `cf_dset`
        was specified as a string.

        Returns
        -------
        gen_ac_data : np.ndarray | None
            Multi-year-mean ac capacity factor data array for all sites
            in the generation data output file or `None` if none
            detected.
        """

        if isinstance(self._cf_dset, np.ndarray):
            return None

        ac_cf_dset = _infer_cf_dset_ac(self._cf_dset)
        if ac_cf_dset in self.gen.datasets:
            return self.gen[ac_cf_dset]

        return None

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

        if self._lcoe_data is None:
            if self._lcoe_dset in self.gen.datasets:
                self._lcoe_data = self.gen[self._lcoe_dset]

        return self._lcoe_data

    @property
    def mean_cf(self):
        """Get the mean capacity factor for the non-excluded data. Capacity
        factor is weighted by the exclusions (usually 0 or 1, but 0.5
        exclusions will weight appropriately).

        This value represents DC capacity factor for solar and AC
        capacity factor for all other technologies. This is the capacity
        factor that should be used for all cost calculations for ALL
        technologies (to align with SAM).

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
    def mean_cf_ac(self):
        """Get the mean AC capacity factor for the non-excluded data.

        This output is only not `None` for solar runs.

        Capacity factor is weighted by the exclusions (usually 0 or 1,
        but 0.5 exclusions will weight appropriately).

        Returns
        -------
        mean_cf_ac : float | None
            Mean capacity factor value for the non-excluded data.
        """
        mean_cf_ac = None
        if self.gen_ac_data is not None:
            mean_cf_ac = self.exclusion_weighted_mean(self.gen_ac_data)

        return mean_cf_ac

    @property
    def mean_cf_dc(self):
        """Get the mean DC capacity factor for the non-excluded data.

        This output is only not `None` for solar runs.

        Capacity factor is weighted by the exclusions (usually 0 or 1,
        but 0.5 exclusions will weight appropriately).

        Returns
        -------
        mean_cf_dc : float | None
            Mean capacity factor value for the non-excluded data.
        """
        if self.mean_cf_ac is not None:
            return self.mean_cf

        return None

    @property
    def mean_lcoe(self):
        """Get the mean LCOE for the non-excluded data.

        Returns
        -------
        mean_lcoe : float | None
            Mean LCOE value for the non-excluded data.
        """

        mean_lcoe = None

        # prioritize the calculation of lcoe explicitly from the multi year
        # mean CF (the lcoe re-calc will still happen if mean_cf is a single
        # year CF, but the output should be identical to the original LCOE and
        # so is not consequential).
        if self._recalc_lcoe:
            required = ("fixed_charge_rate", "capital_cost",
                        "fixed_operating_cost", "variable_operating_cost",
                        "system_capacity")
            if all(self._sam_lcoe_kwargs.get(k) is not None for k in required):
                aep = (
                    self._sam_lcoe_kwargs["system_capacity"]
                    * self.mean_cf
                    * 8760
                )
                # Note the AEP computation uses the SAM config
                # `system_capacity`, so no need to scale `capital_cost`
                # or `fixed_operating_cost` by anything
                mean_lcoe = lcoe_fcr(
                    self._sam_lcoe_kwargs["fixed_charge_rate"],
                    self._sam_lcoe_kwargs["capital_cost"],
                    self._sam_lcoe_kwargs["fixed_operating_cost"],
                    aep,
                    self._sam_lcoe_kwargs["variable_operating_cost"],
                )

        # alternative if lcoe was not able to be re-calculated from
        # multi year mean CF
        if mean_lcoe is None and self.lcoe_data is not None:
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
            tech = self.gen.meta["reV_tech"][0]
            if tech in self.POWER_DENSITY:
                self._power_density = self.POWER_DENSITY[tech]
            else:
                warn(
                    "Could not recognize reV technology in generation meta "
                    'data: "{}". Cannot lookup an appropriate power density '
                    "to calculate SC point capacity.".format(tech)
                )

        elif isinstance(self._power_density, pd.DataFrame):
            self._pd_obj = self._power_density

            missing = set(self.res_gid_set) - set(self._pd_obj.index.values)
            if any(missing):
                msg = (
                    "Variable power density input is missing the "
                    "following resource GIDs: {}".format(missing)
                )
                logger.error(msg)
                raise FileInputError(msg)

            pds = self._pd_obj.loc[
                self._res_gids[self.bool_mask], "power_density"
            ].values
            pds = pds.astype(np.float32)
            pds *= self.include_mask_flat[self.bool_mask]
            denom = self.include_mask_flat[self.bool_mask].sum()
            self._power_density = pds.sum() / denom

        return self._power_density

    @property
    def power_density_ac(self):
        """Get the estimated AC power density either from input or
        inferred from generation output meta.

        This value is only available for solar runs with a "dc_ac_ratio"
        dataset in the generation file. If these conditions are not met,
        this value is `None`.

        Returns
        -------
        _power_density_ac : float | None
            Estimated AC power density in MW/km2
        """
        if "dc_ac_ratio" not in self.gen.datasets:
            return None

        ilr = self.gen["dc_ac_ratio", self._gen_gids[self.bool_mask]]
        ilr = ilr.astype("float32")
        weights = self.include_mask_flat[self.bool_mask]
        if self._power_density_ac is None:
            tech = self.gen.meta["reV_tech"][0]
            if tech in self.POWER_DENSITY:
                power_density_ac = self.POWER_DENSITY[tech] / ilr
                power_density_ac *= weights
                power_density_ac = power_density_ac.sum() / weights.sum()
            else:
                warn(
                    "Could not recognize reV technology in generation meta "
                    'data: "{}". Cannot lookup an appropriate power density '
                    "to calculate SC point capacity.".format(tech)
                )
                power_density_ac = None

        elif isinstance(self._power_density_ac, pd.DataFrame):
            self._pd_obj = self._power_density_ac

            missing = set(self.res_gid_set) - set(self._pd_obj.index.values)
            if any(missing):
                msg = (
                    "Variable power density input is missing the "
                    "following resource GIDs: {}".format(missing)
                )
                logger.error(msg)
                raise FileInputError(msg)

            pds = self._pd_obj.loc[
                self._res_gids[self.bool_mask], "power_density"
            ].values
            power_density_ac = pds.astype(np.float32) / ilr
            power_density_ac *= weights
            power_density_ac = power_density_ac.sum() / weights.sum()
        else:
            power_density_ac = self._power_density_ac * weights / ilr
            power_density_ac = power_density_ac.sum() / weights.sum()

        return power_density_ac

    @property
    def capacity(self):
        """Get the estimated capacity in MW of the supply curve point in the
        current resource class with the applied exclusions.

        This value represents DC capacity for solar and AC capacity for
        all other technologies. This is the capacity that should be used
        for all cost calculations for ALL technologies (to align with
        SAM).

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
    def capacity_ac(self):
        """Get the AC estimated capacity in MW of the supply curve point in the
        current resource class with the applied exclusions.

        This value is provided only for solar inputs that have
        the "dc_ac_ratio" dataset in the generation file. If these
        conditions are not met, this value is `None`.

        Returns
        -------
        capacity : float | None
            Estimated AC capacity in MW of the supply curve point in the
            current resource class with the applied exclusions. Only not
            `None` for solar runs with "dc_ac_ratio" dataset in the
            generation file
        """
        if self.power_density_ac is None:
            return None

        return self.area * self.power_density_ac

    @property
    def capacity_dc(self):
        """Get the DC estimated capacity in MW of the supply curve point
        in the current resource class with the applied exclusions.

        This value is provided only for solar inputs that have
        the "dc_ac_ratio" dataset in the generation file. If these
        conditions are not met, this value is `None`.

        Returns
        -------
        capacity : float | None
            Estimated AC capacity in MW of the supply curve point in the
            current resource class with the applied exclusions. Only not
            `None` for solar runs with "dc_ac_ratio" dataset in the
            generation file
        """
        if self.power_density_ac is None:
            return None

        return self.area * self.power_density

    @property
    def sc_point_annual_energy(self):
        """Get the total annual energy (MWh) for the entire SC point.

        This value is computed using the capacity of the supply curve
        point as well as the mean capacity factor. If the mean capacity
        factor is `None`, this value will also be `None`.

        Returns
        -------
        sc_point_annual_energy : float | None
            Total annual energy (MWh) for the entire SC point.
        """
        if self.mean_cf is None:
            return None

        return self.mean_cf * self.capacity * 8760

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
            e = (
                "Cannot recognize h5_dsets input type, should be None, "
                "a list of dataset names, or a dictionary or "
                "pre-extracted data. Received: {} {}".format(
                    type(self._h5_dsets), self._h5_dsets
                )
            )
            logger.error(e)
            raise TypeError(e)

        return _h5_dsets_data

    @property
    def regional_multiplier(self):
        """float: Mean regional capital cost multiplier, defaults to 1."""
        if "capital_cost_multiplier" not in self.gen.datasets:
            return 1

        multipliers = self.gen["capital_cost_multiplier"]
        return self.exclusion_weighted_mean(multipliers)

    @property
    def fixed_charge_rate(self):
        """float: Mean fixed_charge_rate, defaults to 0."""
        if "fixed_charge_rate" not in self.gen.datasets:
            return 0

        return self.exclusion_weighted_mean(self.gen["fixed_charge_rate"])

    @property
    def _sam_system_capacity_kw(self):
        """float: Mean SAM generation system capacity input, defaults to 0. """
        if self._ssc is not None:
            return self._ssc

        self._ssc = 0
        if "system_capacity" in self.gen.datasets:
            self._ssc = self.exclusion_weighted_mean(
                self.gen["system_capacity"]
            )

        return self._ssc

    @property
    def _sam_lcoe_kwargs(self):
        """dict: Mean LCOE inputs, as passed to SAM during generation."""
        if self._slk:
            return self._slk

        self._slk = {"capital_cost": None, "fixed_operating_cost": None,
                     "variable_operating_cost": None,
                     "fixed_charge_rate": None, "system_capacity": None}

        for dset in self._slk:
            if dset in self.gen.datasets:
                self._slk[dset] = self.exclusion_weighted_mean(
                    self.gen[dset]
                )

        return self._slk

    def _compute_cost_per_ac_mw(self, dset):
        """Compute a cost per AC MW for a given input. """
        if self._sam_system_capacity_kw <= 0:
            return None

        if dset not in self.gen.datasets:
            return None

        sam_cost = self.exclusion_weighted_mean(self.gen[dset])
        sam_cost_per_mw = sam_cost / (self._sam_system_capacity_kw / 1000)
        sc_point_cost = sam_cost_per_mw * self.capacity

        ac_cap = (self.capacity
                  if self.capacity_ac is None
                  else self.capacity_ac)
        return sc_point_cost / ac_cap

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
            msg = (
                "Supply curve point gid {} is completely excluded for res "
                "bin: {}".format(self._gid, self._res_class_bin)
            )
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

        if (
            self._res_class_dset is not None
            and self._res_class_bin is not None
        ):
            rex = self.res_data[self._gen_gids]
            rex = (rex < np.min(self._res_class_bin)) | (
                rex >= np.max(self._res_class_bin)
            )

            boolean_exclude = boolean_exclude | rex

        return boolean_exclude

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

        ARGS = {
            SupplyCurveField.LATITUDE: self.latitude,
            SupplyCurveField.LONGITUDE: self.longitude,
            SupplyCurveField.COUNTRY: self.country,
            SupplyCurveField.STATE: self.state,
            SupplyCurveField.COUNTY: self.county,
            SupplyCurveField.ELEVATION: self.elevation,
            SupplyCurveField.TIMEZONE: self.timezone,
            SupplyCurveField.SC_POINT_GID: self.sc_point_gid,
            SupplyCurveField.SC_ROW_IND: self.sc_row_ind,
            SupplyCurveField.SC_COL_IND: self.sc_col_ind,
            SupplyCurveField.RES_GIDS: self.res_gid_set,
            SupplyCurveField.GEN_GIDS: self.gen_gid_set,
            SupplyCurveField.GID_COUNTS: self.gid_counts,
            SupplyCurveField.N_GIDS: self.n_gids,
            SupplyCurveField.OFFSHORE: self.offshore,
            SupplyCurveField.MEAN_CF_AC: (
                self.mean_cf if self.mean_cf_ac is None else self.mean_cf_ac
            ),
            SupplyCurveField.MEAN_CF_DC: self.mean_cf_dc,
            SupplyCurveField.MEAN_LCOE: self.mean_lcoe,
            SupplyCurveField.MEAN_RES: self.mean_res,
            SupplyCurveField.AREA_SQ_KM: self.area,
            SupplyCurveField.CAPACITY_AC_MW: (
                self.capacity if self.capacity_ac is None else self.capacity_ac
            ),
            SupplyCurveField.CAPACITY_DC_MW: self.capacity_dc,
            SupplyCurveField.EOS_MULT: 1,  # added later
            SupplyCurveField.REG_MULT: self.regional_multiplier,
            SupplyCurveField.SC_POINT_ANNUAL_ENERGY_MWH: (
                self.sc_point_annual_energy
            ),
            SupplyCurveField.COST_SITE_OCC_USD_PER_AC_MW: (
                self._compute_cost_per_ac_mw("capital_cost")
            ),
            SupplyCurveField.COST_BASE_OCC_USD_PER_AC_MW: (
                self._compute_cost_per_ac_mw("base_capital_cost")
            ),
            SupplyCurveField.COST_SITE_FOC_USD_PER_AC_MW: (
                self._compute_cost_per_ac_mw("fixed_operating_cost")
            ),
            SupplyCurveField.COST_BASE_FOC_USD_PER_AC_MW: (
                self._compute_cost_per_ac_mw("base_fixed_operating_cost")
            ),
            SupplyCurveField.COST_SITE_VOC_USD_PER_AC_MW: (
                self._compute_cost_per_ac_mw("variable_operating_cost")
            ),
            SupplyCurveField.COST_BASE_VOC_USD_PER_AC_MW: (
                self._compute_cost_per_ac_mw("base_variable_operating_cost")
            ),
            SupplyCurveField.FIXED_CHARGE_RATE: self.fixed_charge_rate,
        }

        if self._friction_layer is not None:
            ARGS[SupplyCurveField.MEAN_FRICTION] = self.mean_friction
            ARGS[SupplyCurveField.MEAN_LCOE_FRICTION] = self.mean_lcoe_friction

        if self._h5_dsets is not None:
            for dset, data in self.mean_h5_dsets_data.items():
                ARGS["mean_{}".format(dset)] = data

        if args is None:
            args = list(ARGS.keys())

        summary = {}
        for arg in args:
            if arg in ARGS:
                summary[arg] = ARGS[arg]
            else:
                warn(
                    'Cannot find "{}" as an available SC self summary '
                    "output",
                    OutputWarning,
                )

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
        summary[SupplyCurveField.RAW_LCOE] = eos.raw_lcoe
        summary[SupplyCurveField.MEAN_LCOE] = eos.scaled_lcoe
        summary[SupplyCurveField.EOS_MULT] = eos.capital_cost_scalar
        cost = summary[SupplyCurveField.COST_SITE_OCC_USD_PER_AC_MW]
        if cost is not None:
            summary[SupplyCurveField.COST_SITE_OCC_USD_PER_AC_MW] = (
                cost * summary[SupplyCurveField.EOS_MULT]
            )
        return summary

    @classmethod
    def summarize(
        cls,
        gid,
        excl_fpath,
        gen_fpath,
        tm_dset,
        gen_index,
        excl_dict=None,
        inclusion_mask=None,
        res_class_dset=None,
        res_class_bin=None,
        excl_area=None,
        power_density=None,
        cf_dset="cf_mean-means",
        lcoe_dset="lcoe_fcr-means",
        h5_dsets=None,
        resolution=64,
        exclusion_shape=None,
        close=False,
        friction_layer=None,
        args=None,
        data_layers=None,
        cap_cost_scale=None,
        recalc_lcoe=True,
        zone_mask=None,
    ):
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
            Dictionary of exclusion keyword arugments of the format
            {layer_dset_name: {kwarg: value}} where layer_dset_name is a
            dataset in the exclusion h5 file and kwarg is a keyword argument to
            the reV.supply_curve.exclusions.LayerMask class.
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
        excl_area : float | None, optional
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath, by default None
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
        recalc_lcoe : bool
            Flag to re-calculate the LCOE from the multi-year mean capacity
            factor and annual energy production data. This requires several
            datasets to be aggregated in the gen input: system_capacity,
            fixed_charge_rate, capital_cost, fixed_operating_cost,
            and variable_operating_cost.
        zone_mask : np.ndarray | None, optional
            2D array defining zone within the supply curve to be evaluated,
            where values of 0 or False will be excluded. The shape of this
            array will be checked against the input resolution. If not
            specified, no zone mask will be applied.

        Returns
        -------
        summary : dict
            Dictionary of summary outputs for this sc point.
        """
        kwargs = {
            "excl_dict": excl_dict,
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
            "friction_layer": friction_layer,
            "recalc_lcoe": recalc_lcoe,
            "zone_mask": zone_mask,
        }

        with cls(
            gid, excl_fpath, gen_fpath, tm_dset, gen_index, **kwargs
        ) as point:
            summary = point.point_summary(args=args)

            if data_layers is not None:
                summary = point.agg_data_layers(summary, data_layers)

            if cap_cost_scale is not None:
                summary = point.economies_of_scale(cap_cost_scale, summary)

        for arg, val in summary.items():
            if val is None:
                summary[arg] = np.nan

        return summary


def _infer_cf_dset_ac(cf_dset):
    """Infer AC dataset name from input. """
    parts = cf_dset.split("-")
    if len(parts) == 1:
        return f"{cf_dset}_ac"

    cf_name = "-".join(parts[:-1])
    return f"{cf_name}_ac-{parts[-1]}"
