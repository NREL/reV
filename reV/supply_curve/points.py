# -*- coding: utf-8 -*-
"""
reV supply curve extent and points base frameworks.
"""
from abc import ABC
import logging
import numpy as np
import pandas as pd
from scipy import stats
from warnings import warn

from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.exclusions import ExclusionMask, ExclusionMaskFromDict
from reV.utilities.exceptions import (SupplyCurveError, SupplyCurveInputError,
                                      EmptySupplyCurvePointError, InputWarning)

from rex.resource import Resource
from rex.utilities.utilities import get_chunk_ranges

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
        self._rows, self._cols = self._parse_slices(
            gid, resolution, exclusion_shape)

    def _parse_slices(self, gid, resolution, exclusion_shape):
        """Parse inputs for the definition of this SC point.

        Parameters
        ----------
        gid : int | None
            gid for supply curve point to analyze.
        resolution : int | None
            SC resolution, must be input in combination with gid.
        exclusion_shape : tuple
            Shape of the exclusions extent (rows, cols). Inputing this will
            speed things up considerably.

        Returns
        -------
        rows : slice
            Row slice to index the high-res layer (exclusions) for the gid in
            the agg layer (supply curve).
        cols : slice
            Col slice to index the high-res layer (exclusions) for the gid in
            the agg layer (supply curve).
        """

        rows, cols = self.get_agg_slices(gid, exclusion_shape, resolution)

        return rows, cols

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
            msg = ('Gid {} out of bounds for extent shape {} and '
                   'resolution {}.'.format(gid, shape, resolution))
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

    def __init__(self, gid, excl, tm_dset, excl_dict=None, inclusion_mask=None,
                 resolution=64, excl_area=0.0081, exclusion_shape=None,
                 close=True):
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
        """

        self._excl_dict = excl_dict
        self._close = close
        self._excl_fpath, self._excls = self._parse_excl_file(excl)

        if exclusion_shape is None:
            exclusion_shape = self.exclusions.shape

        super().__init__(gid, exclusion_shape, resolution=resolution)

        self._gids = self._parse_techmap(tm_dset)

        self._incl_mask = inclusion_mask
        self._incl_mask_flat = None
        if inclusion_mask is not None:
            msg = ('Bad inclusion mask input shape of {} with stated '
                   'resolution of {}'.format(inclusion_mask.shape, resolution))
            assert len(inclusion_mask.shape) == 2, msg
            assert inclusion_mask.shape[0] <= resolution, msg
            assert inclusion_mask.shape[1] <= resolution, msg
            assert inclusion_mask.size == len(self._gids), msg
            self._incl_mask = inclusion_mask.copy()

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
            raise SupplyCurveInputError('SupplyCurvePoints needs an '
                                        'exclusions file path, or '
                                        'ExclusionMask handler but '
                                        'received: {}'
                                        .format(type(excl)))

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
            emsg = ('Supply curve point gid {} has no viable exclusion points '
                    'based on exclusions file: "{}"'
                    .format(self._gid, self._excl_fpath))
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
            self._excls = ExclusionMaskFromDict(self._excl_fpath,
                                                layers_dict=self._excl_dict)

        return self._excls

    @property
    def centroid(self):
        """Get the supply curve point centroid coordinate.

        Returns
        -------
        centroid : tuple
            SC point centroid (lat, lon).
        """
        decimals = 3

        if self._centroid is None:
            lats = self.exclusions.excl_h5['latitude', self.rows, self.cols]
            lons = self.exclusions.excl_h5['longitude', self.rows, self.cols]
            self._centroid = (np.round(lats.mean(), decimals=decimals),
                              np.round(lons.mean(), decimals=decimals))

        return self._centroid

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
        area = np.sum(self.include_mask_flat[mask]) * self._excl_area

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
                w = ('Exclusions data max value is > 1: {}'
                     .format(self._incl_mask.max()), InputWarning)
                logger.warning(w)
                warn(w)

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
    def summary(self):
        """
        Placeholder for Supply curve point's meta data summary
        """

    def _check_excl(self):
        """
        Check to see if supply curve point is fully excluded
        """

        if all(self.include_mask_flat[self.bool_mask] == 0):
            msg = ('Supply curve point gid {} is completely excluded!'
                   .format(self._gid))
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
            x = arr[:, self._gids[self.bool_mask]].astype('float32')
            incl = self.include_mask_flat[self.bool_mask]
            x *= incl
            mean = x.sum(axis=1) / incl.sum()

        else:
            x = arr[self._gids[self.bool_mask]].astype('float32')
            incl = self.include_mask_flat[self.bool_mask]

            if np.isnan(x).all():
                return np.nan
            elif drop_nan and np.isnan(x).any():
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
            x = arr[:, self._gids[self.bool_mask]].astype('float32')
            ax = 1
        else:
            x = arr[self._gids[self.bool_mask]].astype('float32')
            ax = 0

        x *= self.include_mask_flat[self.bool_mask]
        agg = x.sum(axis=ax)

        return agg

    @classmethod
    def sc_mean(cls, gid, excl, tm_dset, data, excl_dict=None, resolution=64,
                exclusion_shape=None, close=True):
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
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
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
        kwargs = {"excl_dict": excl_dict, "resolution": resolution,
                  "exclusion_shape": exclusion_shape, "close": close}
        with cls(gid, excl, tm_dset, **kwargs) as point:
            means = point.exclusion_weighted_mean(data)

        return means

    @classmethod
    def sc_sum(cls, gid, excl, tm_dset, data, excl_dict=None, resolution=64,
               exclusion_shape=None, close=True):
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
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
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
        kwargs = {"excl_dict": excl_dict, "resolution": resolution,
                  "exclusion_shape": exclusion_shape, "close": close}
        with cls(gid, excl, tm_dset, **kwargs) as point:
            agg = point.aggregate(data)

        return agg


class AggregationSupplyCurvePoint(SupplyCurvePoint):
    """Generic single SC point to aggregate data from an h5 file."""

    def __init__(self, gid, excl, agg_h5, tm_dset,
                 excl_dict=None, inclusion_mask=None,
                 resolution=64, excl_area=0.0081, exclusion_shape=None,
                 close=True, gen_index=None):
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
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        """
        super().__init__(gid, excl, tm_dset,
                         excl_dict=excl_dict,
                         inclusion_mask=inclusion_mask,
                         resolution=resolution,
                         excl_area=excl_area,
                         exclusion_shape=exclusion_shape,
                         close=close)

        self._h5_gid_set = None
        self._h5_fpath, self._h5 = self._parse_h5_file(agg_h5)

        if gen_index is not None:
            self._gids, _ = self._map_gen_gids(self._gids, gen_index)

        self._h5_gids = self._gids

        if (self._h5_gids != -1).sum() == 0:
            emsg = ('Supply curve point gid {} has no viable exclusion '
                    'points based on exclusions file: "{}"'
                    .format(self._gid, self._excl_fpath))
            raise EmptySupplyCurvePointError(emsg)

        self._check_excl()

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
        elif isinstance(h5, Resource):
            h5_fpath = h5.h5_file
        else:
            raise SupplyCurveInputError('SupplyCurvePoints needs a '
                                        '.h5 file path, or '
                                        'Resource handler but '
                                        'received: {}'
                                        .format(type(h5)))

        return h5_fpath, h5

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

    @property
    def h5(self):
        """
        h5 Resource handler object

        Returns
        -------
        _h5 : Resource
            Resource h5 handler object.
        """
        if self._h5 is None:
            self._h5 = Resource(self._h5_fpath,)

        return self._h5

    @property
    def country(self):
        """Get the SC point country based on the resource meta data."""
        country = None
        if 'country' in self.h5.meta:
            country = self.h5.meta.loc[self.h5_gid_set, 'country'].values
            country = stats.mode(country).mode[0]

        return country

    @property
    def state(self):
        """Get the SC point state based on the resource meta data."""
        state = None
        if 'state' in self.h5.meta:
            state = self.h5.meta.loc[self.h5_gid_set, 'state'].values
            state = stats.mode(state).mode[0]

        return state

    @property
    def county(self):
        """Get the SC point county based on the resource meta data."""
        county = None
        if 'county' in self.h5.meta:
            county = self.h5.meta.loc[self.h5_gid_set, 'county'].values
            county = stats.mode(county).mode[0]

        return county

    @property
    def elevation(self):
        """Get the SC point elevation based on the resource meta data."""
        elevation = None
        if 'elevation' in self.h5.meta:
            elevation = self.h5.meta.loc[self.h5_gid_set, 'elevation'].mean()

        return elevation

    @property
    def timezone(self):
        """Get the SC point timezone based on the resource meta data."""
        timezone = None
        if 'timezone' in self.h5.meta:
            timezone = self.h5.meta.loc[self.h5_gid_set, 'timezone'].values
            timezone = stats.mode(timezone).mode[0]

        return timezone

    @property
    def offshore(self):
        """Get the SC point offshore flag based on the resource meta data
        (if offshore column is present)."""
        offshore = None
        if 'offshore' in self.h5.meta:
            offshore = self.h5.meta.loc[self.h5_gid_set, 'offshore'].values
            offshore = stats.mode(offshore).mode[0]

        return offshore

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
    def gid_counts(self):
        """Get the sum of the inclusion values in each resource/generation gid
        corresponding to this sc point. The sum of the gid counts can be less
        than the value provided by n_gids if fractional exclusion/inclusions
        are provided.

        Returns
        -------
        gid_counts : list
        """
        gid_counts = [self.include_mask_flat[(self._h5_gids == gid)].sum()
                      for gid in self.h5_gid_set]

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
        meta = {'sc_point_gid': self.sc_point_gid,
                'source_gids': self.h5_gid_set,
                'gid_counts': self.gid_counts,
                'n_gids': self.n_gids,
                'area_sq_km': self.area,
                'latitude': self.latitude,
                'longitude': self.longitude,
                'country': self.country,
                'state': self.state,
                'county': self.county,
                'elevation': self.elevation,
                'timezone': self.timezone,
                }
        meta = pd.Series(meta)

        return meta

    @classmethod
    def run(cls, gid, excl, agg_h5, tm_dset, *agg_dset, agg_method='mean',
            excl_dict=None, inclusion_mask=None,
            resolution=64, excl_area=0.0081,
            exclusion_shape=None, close=True, gen_index=None):
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
            Dataset to aggreate, can supply multiple datasets
        agg_method : str
            Aggregation method, either mean or sum/aggregate
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
            agg_dset = (agg_dset, )

        kwargs = {"excl_dict": excl_dict,
                  "inclusion_mask": inclusion_mask,
                  "resolution": resolution,
                  "excl_area": excl_area,
                  "exclusion_shape": exclusion_shape,
                  "close": close,
                  "gen_index": gen_index}

        with cls(gid, excl, agg_h5, tm_dset, **kwargs) as point:
            if agg_method.lower().startswith('mean'):
                agg_method = point.exclusion_weighted_mean
            elif agg_method.lower().startswith(('sum', 'agg')):
                agg_method = point.aggregate
            elif 'wind_dir' in agg_method.lower():
                agg_method = point.mean_wind_dirs
            else:
                msg = ('Aggregation method must be either mean, '
                       'sum/aggregate, or wind_dir')
                logger.error(msg)
                raise ValueError(msg)

            out = {'meta': point.summary}

            for dset in agg_dset:
                ds = point.h5.open_dataset(dset)
                out[dset] = agg_method(ds)

        return out


class GenerationSupplyCurvePoint(AggregationSupplyCurvePoint):
    """Single supply curve point with associated reV generation"""

    def __init__(self, gid, excl, gen, tm_dset, gen_index,
                 excl_dict=None, inclusion_mask=None, resolution=64,
                 excl_area=0.0081, exclusion_shape=None, close=True):
        """
        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        excl : str | ExclusionMask
            Filepath to exclusions h5 or ExclusionMask file handler.
        gen : str | reV.handlers.Resource
            Filepath to .h5 reV generation output results or reV Resource file
            handler.
        tm_dset : str
            Dataset name in the exclusions file containing the
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
        """

        super().__init__(gid, excl, gen, tm_dset,
                         excl_dict=excl_dict,
                         inclusion_mask=inclusion_mask,
                         resolution=resolution,
                         excl_area=excl_area,
                         exclusion_shape=exclusion_shape,
                         close=close)

        self._res_gid_set = None
        self._gen_gid_set = None

        self._gen_fpath, self._gen = self._h5_fpath, self._h5

        self._gen_gids, self._res_gids = self._map_gen_gids(self._gids,
                                                            gen_index)
        self._gids = self._gen_gids
        if (self._gen_gids != -1).sum() == 0:
            emsg = ('Supply curve point gid {} has no viable exclusion '
                    'points based on exclusions file: "{}"'
                    .format(self._gid, self._excl_fpath))
            raise EmptySupplyCurvePointError(emsg)

        self._check_excl()

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
        x = flat_arr[self._gen_gids[self.bool_mask]].astype('float32')
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
        gid_counts = [self.include_mask_flat[(self._res_gids == gid)].sum()
                      for gid in self.res_gid_set]

        return gid_counts


class SupplyCurveExtent:
    """Supply curve full extent framework."""

    def __init__(self, f_excl, resolution=64):
        """
        Parameters
        ----------
        f_excl : str | list | tuple | ExclusionLayers
            File path(s) to the exclusions grid, or pre-initialized
            ExclusionLayers. The exclusions dictate the SC analysis extent.
        resolution : int
            Number of exclusion points per SC point along an axis.
            This number**2 is the total number of exclusion points per
            SC point.
        """

        logger.debug('Initializing SupplyCurveExtent with res {} from: {}'
                     .format(resolution, f_excl))

        if not isinstance(resolution, int):
            raise SupplyCurveInputError('Supply Curve resolution needs to be '
                                        'an integer but received: {}'
                                        .format(type(resolution)))

        if isinstance(f_excl, (str, list, tuple)):
            self._excl_fpath = f_excl
            self._excls = ExclusionLayers(f_excl)
        elif isinstance(f_excl, ExclusionLayers):
            self._excl_fpath = f_excl.h5_file
            self._excls = f_excl
        else:
            raise SupplyCurveInputError('SupplyCurvePoints needs an '
                                        'exclusions file path, or '
                                        'ExclusionLayers handler but '
                                        'received: {}'
                                        .format(type(f_excl)))

        self._excl_shape = self.exclusions.shape
        # limit the resolution to the exclusion shape.
        self._res = int(np.min(list(self.excl_shape) + [resolution]))

        self._n_rows = None
        self._n_cols = None
        self._cols_of_excl = None
        self._rows_of_excl = None
        self._excl_row_slices = None
        self._excl_col_slices = None
        self._latitude = None
        self._longitude = None
        self._points = None

        self._sc_col_ind, self._sc_row_ind = np.meshgrid(
            np.arange(self.n_cols), np.arange(self.n_rows))
        self._sc_col_ind = self._sc_col_ind.flatten()
        self._sc_row_ind = self._sc_row_ind.flatten()

        logger.debug('Initialized SupplyCurveExtent with shape {} from '
                     'exclusions with shape {}'
                     .format(self.shape, self.excl_shape))

    def __len__(self):
        """Total number of supply curve points."""
        return self.n_rows * self.n_cols

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if type is not None:
            raise

    def __getitem__(self, gid):
        """Get SC extent meta data corresponding to an SC point gid."""
        if gid >= len(self):
            raise KeyError('SC extent with {} points does not contain SC '
                           'point gid {}.'.format(len(self), gid))

        return self.points.loc[gid]

    def close(self):
        """Close all file handlers."""
        self._excls.close()

    @property
    def shape(self):
        """Get the Supply curve shape tuple (n_rows, n_cols).

        Returns
        -------
        shape : tuple
            2-entry tuple representing the full supply curve extent.
        """

        return (self.n_rows, self.n_cols)

    @property
    def exclusions(self):
        """Get the exclusions object.

        Returns
        -------
        _excls : ExclusionLayers
            ExclusionLayers h5 handler object.
        """
        return self._excls

    @property
    def resolution(self):
        """Get the 1D resolution.

        Returns
        -------
        _res : int
            Number of exclusion points per SC point along an axis.
            This number**2 is the total number of exclusion points per
            SC point.
        """
        return self._res

    @property
    def excl_shape(self):
        """Get the shape tuple of the exclusion file raster.

        Returns
        -------
        tuple
        """
        return self._excl_shape

    @property
    def excl_rows(self):
        """Get the unique row indices identifying the exclusion points.

        Returns
        -------
        excl_rows : np.ndarray
            Array of exclusion row indices.
        """
        return np.arange(self.excl_shape[0])

    @property
    def excl_cols(self):
        """Get the unique column indices identifying the exclusion points.

        Returns
        -------
        excl_cols : np.ndarray
            Array of exclusion column indices.
        """
        return np.arange(self.excl_shape[1])

    @property
    def rows_of_excl(self):
        """List representing the supply curve points rows and which
        exclusions rows belong to each supply curve row.

        Returns
        -------
        _rows_of_excl : list
            List representing the supply curve points rows. Each list entry
            contains the exclusion row indices that are included in the sc
            point.
        """
        if self._rows_of_excl is None:
            self._rows_of_excl = self._chunk_excl(self.excl_rows,
                                                  self.resolution)

        return self._rows_of_excl

    @property
    def cols_of_excl(self):
        """List representing the supply curve points columns and which
        exclusions columns belong to each supply curve column.

        Returns
        -------
        _cols_of_excl : list
            List representing the supply curve points columns. Each list entry
            contains the exclusion column indices that are included in the sc
            point.
        """
        if self._cols_of_excl is None:
            self._cols_of_excl = self._chunk_excl(self.excl_cols,
                                                  self.resolution)

        return self._cols_of_excl

    @property
    def excl_row_slices(self):
        """
        List representing the supply curve points rows and which
        exclusions rows belong to each supply curve row.

        Returns
        -------
        _excl_row_slices : list
            List representing the supply curve points rows. Each list entry
            contains the exclusion row slice that are included in the sc
            point.
        """
        if self._excl_row_slices is None:
            self._excl_row_slices = self._excl_slices(self.excl_rows,
                                                      self.resolution)

        return self._excl_row_slices

    @property
    def excl_col_slices(self):
        """
        List representing the supply curve points cols and which
        exclusions cols belong to each supply curve col.

        Returns
        -------
        _excl_col_slices : list
            List representing the supply curve points cols. Each list entry
            contains the exclusion col slice that are included in the sc
            point.
        """
        if self._excl_col_slices is None:
            self._excl_col_slices = self._excl_slices(self.excl_cols,
                                                      self.resolution)

        return self._excl_col_slices

    @property
    def n_rows(self):
        """Get the number of supply curve grid rows.

        Returns
        -------
        n_rows : int
            Number of row entries in the full supply curve grid.
        """
        if self._n_rows is None:
            self._n_rows = int(np.ceil(self.excl_shape[0] / self.resolution))

        return self._n_rows

    @property
    def n_cols(self):
        """Get the number of supply curve grid columns.

        Returns
        -------
        n_cols : int
            Number of column entries in the full supply curve grid.
        """
        if self._n_cols is None:
            self._n_cols = int(np.ceil(self.excl_shape[1] / self.resolution))

        return self._n_cols

    @property
    def latitude(self):
        """
        Get supply curve point latitudes

        Returns
        -------
        ndarray
        """
        if self._latitude is None:
            lats = []
            lons = []

            sc_cols, sc_rows = np.meshgrid(np.arange(self.n_cols),
                                           np.arange(self.n_rows))
            for r, c in zip(sc_rows.flatten(), sc_cols.flatten()):
                r = self.excl_row_slices[r]
                c = self.excl_col_slices[c]
                lats.append(self.exclusions['latitude', r, c].mean())
                lons.append(self.exclusions['longitude', r, c].mean())

            self._latitude = np.array(lats, dtype='float32')
            self._longitude = np.array(lons, dtype='float32')

        return self._latitude

    @property
    def longitude(self):
        """
        Get supply curve point longitudes

        Returns
        -------
        ndarray
        """
        if self._longitude is None:
            lats = []
            lons = []

            sc_cols, sc_rows = np.meshgrid(np.arange(self.n_cols),
                                           np.arange(self.n_rows))
            for r, c in zip(sc_rows.flatten(), sc_cols.flatten()):
                r = self.excl_row_slices[r]
                c = self.excl_col_slices[c]
                lats.append(self.exclusions['latitude', r, c].mean())
                lons.append(self.exclusions['longitude', r, c].mean())

            self._latitude = np.array(lats, dtype='float32')
            self._longitude = np.array(lons, dtype='float32')

        return self._longitude

    @property
    def lat_lon(self):
        """
        2D array of lat, lon coordinates for all sc points

        Returns
        -------
        ndarray
        """
        return np.dstack((self.latitude, self.longitude))[0]

    @property
    def row_indices(self):
        """Get a 1D array of row indices for every gid. That is, this property
        has length == len(gids) and row_indices[sc_gid] yields the row index of
        the target supply curve gid

        Returns
        -------
        ndarray
        """
        return self._sc_row_ind

    @property
    def col_indices(self):
        """Get a 1D array of col indices for every gid. That is, this property
        has length == len(gids) and col_indices[sc_gid] yields the col index of
        the target supply curve gid

        Returns
        -------
        ndarray
        """
        return self._sc_col_ind

    @property
    def points(self):
        """Get the summary dataframe of supply curve points.

        Returns
        -------
        _points : pd.DataFrame
            Supply curve points with columns for attributes of each sc point.
        """

        if self._points is None:
            self._points = pd.DataFrame({'row_ind': self.row_indices.copy(),
                                         'col_ind': self.col_indices.copy()})

            self._points.index.name = 'gid'

        return self._points

    @staticmethod
    def _chunk_excl(arr, resolution):
        """Split an array into a list of arrays with len == resolution.

        Parameters
        ----------
        arr : np.ndarray
            1D array to be split into chunks.
        resolution : int
            Resolution of the chunks.

        Returns
        -------
        chunks : list
            List of arrays, each with length equal to self.resolution
            (except for the last array in the list which is the remainder).
        """

        chunks = get_chunk_ranges(len(arr), resolution)
        chunks = list(map(lambda i: np.arange(*i), chunks))

        return chunks

    @staticmethod
    def _excl_slices(arr, resolution):
        """Split row or col ind into slices of excl rows or slices

        Parameters
        ----------
        arr : np.ndarray
            1D array to be split into slices
        resolution : int
            Resolution of the sc points

        Returns
        -------
        slices : list
            List of arr slices, each with length equal to self.resolution
            (except for the last array in the list which is the remainder).
        """

        slices = get_chunk_ranges(len(arr), resolution)
        slices = list(map(lambda i: slice(*i), slices))

        return slices

    def get_excl_slices(self, gid):
        """Get the row and column slices of the exclusions grid corresponding
        to the supply curve point gid.

        Parameters
        ----------
        gid : int
            Supply curve point gid.

        Returns
        -------
        row_slice : slice
            Exclusions grid row slice corresponding to the sc point gid.
        col_slice : slice
            Exclusions grid col slice corresponding to the sc point gid.
        """

        if gid >= len(self):
            raise SupplyCurveError('Requested gid "{}" is out of bounds for '
                                   'supply curve points with length "{}".'
                                   .format(gid, len(self)))

        row_slice = self.excl_row_slices[self.row_indices[gid]]
        col_slice = self.excl_col_slices[self.col_indices[gid]]

        return row_slice, col_slice

    def get_flat_excl_ind(self, gid):
        """Get the index values of the flattened exclusions grid corresponding
        to the supply curve point gid.

        Parameters
        ----------
        gid : int
            Supply curve point gid.

        Returns
        -------
        excl_ind : np.ndarray
            Index values of the flattened exclusions grid corresponding to
            the SC gid.
        """

        row_slice, col_slice = self.get_excl_slices(gid)
        excl_ind = self.exclusions.iarr[row_slice, col_slice].flatten()

        return excl_ind

    def get_excl_points(self, dset, gid):
        """Get the exclusions data corresponding to a supply curve gid.

        Parameters
        ----------
        dset : str | int
            Used as the first arg in the exclusions __getitem__ slice.
            String can be "meta", integer can be layer number.
        gid : int
            Supply curve point gid.

        Returns
        -------
        excl_points : pd.DataFrame
            Exclusions data reduced to just the exclusion points associated
            with the requested supply curve gid.
        """

        row_slice, col_slice = self.get_excl_slices(gid)

        return self.exclusions[dset, row_slice, col_slice]

    def get_coord(self, gid):
        """Get the centroid coordinate for the supply curve gid point.

        Parameters
        ----------
        gid : int
            Supply curve point gid.

        Returns
        -------
        coord : tuple
            Two entry coordinate tuple: (latitude, longitude)
        """

        lat = self.latitude[gid]
        lon = self.longitude[gid]

        return (lat, lon)

    def valid_sc_points(self, tm_dset):
        """
        Determine which sc_point_gids contain resource gids and are thus
        valid supply curve points

        Parameters
        ----------
        tm_dset : str
            Techmap dataset name

        Returns
        -------
        valid_gids : ndarray
            Vector of valid sc_point_gids that contain resource gis
        """

        logger.info('Getting valid SC points from "{}"...'.format(tm_dset))

        valid_bool = np.zeros(self.n_rows * self.n_cols)
        tm = self._excls[tm_dset]

        gid = 0
        for r in self.excl_row_slices:
            for c in self.excl_col_slices:
                if np.any(tm[r, c] != -1):
                    valid_bool[gid] = 1
                gid += 1

        valid_gids = np.where(valid_bool == 1)[0].astype(np.uint32)

        logger.info('Found {} valid SC points out of {} total possible '
                    '(valid SC points that map to valid resource gids)'
                    .format(len(valid_gids), len(valid_bool)))

        return valid_gids

    def get_slice_lookup(self, sc_point_gids):
        """
        Get exclusion slices for all requested supply curve point gids

        Parameters
        ----------
        sc_point_gids : list | ndarray
            List or 1D array of sc_point_gids to get exclusion slices for

        Returns
        -------
        dict
            lookup mapping sc_point_gid to exclusion slice
        """
        return {g: self.get_excl_slices(g) for g in sc_point_gids}
