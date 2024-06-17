# -*- coding: utf-8 -*-
"""
reV supply curve extent
"""

import logging

import numpy as np
import pandas as pd
from rex.utilities.utilities import get_chunk_ranges

from reV.handlers.exclusions import LATITUDE, LONGITUDE, ExclusionLayers
from reV.utilities import SupplyCurveField
from reV.utilities.exceptions import SupplyCurveError, SupplyCurveInputError

logger = logging.getLogger(__name__)


class SupplyCurveExtent:
    """Supply curve full extent framework. This class translates the 90m
    exclusion grid to the aggregated supply curve resolution."""

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

        logger.debug(
            "Initializing SupplyCurveExtent with res {} from: {}".format(
                resolution, f_excl
            )
        )

        if not isinstance(resolution, int):
            raise SupplyCurveInputError(
                "Supply Curve resolution needs to be "
                "an integer but received: {}".format(type(resolution))
            )

        if isinstance(f_excl, (str, list, tuple)):
            self._excl_fpath = f_excl
            self._excls = ExclusionLayers(f_excl)
        elif isinstance(f_excl, ExclusionLayers):
            self._excl_fpath = f_excl.h5_file
            self._excls = f_excl
        else:
            raise SupplyCurveInputError(
                "SupplyCurvePoints needs an "
                "exclusions file path, or "
                "ExclusionLayers handler but "
                "received: {}".format(type(f_excl))
            )

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
            np.arange(self.n_cols), np.arange(self.n_rows)
        )
        self._sc_col_ind = self._sc_col_ind.flatten()
        self._sc_row_ind = self._sc_row_ind.flatten()

        logger.debug(
            "Initialized SupplyCurveExtent with shape {} from "
            "exclusions with shape {}".format(self.shape, self.excl_shape)
        )

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
            raise KeyError(
                "SC extent with {} points does not contain SC "
                "point gid {}.".format(len(self), gid)
            )

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
            self._rows_of_excl = self._chunk_excl(
                self.excl_rows, self.resolution
            )

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
            self._cols_of_excl = self._chunk_excl(
                self.excl_cols, self.resolution
            )

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
            self._excl_row_slices = self._excl_slices(
                self.excl_rows, self.resolution
            )

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
            self._excl_col_slices = self._excl_slices(
                self.excl_cols, self.resolution
            )

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

            sc_cols, sc_rows = np.meshgrid(
                np.arange(self.n_cols), np.arange(self.n_rows)
            )
            for r, c in zip(sc_rows.flatten(), sc_cols.flatten()):
                r = self.excl_row_slices[r]
                c = self.excl_col_slices[c]
                lats.append(self.exclusions[LATITUDE, r, c].mean())
                lons.append(self.exclusions[LONGITUDE, r, c].mean())

            self._latitude = np.array(lats, dtype="float32")
            self._longitude = np.array(lons, dtype="float32")

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

            sc_cols, sc_rows = np.meshgrid(
                np.arange(self.n_cols), np.arange(self.n_rows)
            )
            for r, c in zip(sc_rows.flatten(), sc_cols.flatten()):
                r = self.excl_row_slices[r]
                c = self.excl_col_slices[c]
                lats.append(self.exclusions[LATITUDE, r, c].mean())
                lons.append(self.exclusions[LONGITUDE, r, c].mean())

            self._latitude = np.array(lats, dtype="float32")
            self._longitude = np.array(lons, dtype="float32")

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
            self._points = pd.DataFrame(
                {
                    "row_ind": self.row_indices.copy(),
                    "col_ind": self.col_indices.copy(),
                }
            )

            self._points.index.name = "gid"  # sc_point_gid

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

    def get_sc_row_col_ind(self, gid):
        """Get the supply curve grid row and column index values corresponding
        to a supply curve gid.

        Parameters
        ----------
        gid : int
            Supply curve point gid.

        Returns
        -------
        row_ind : int
            Row index that the gid is located at in the sc grid.
        col_ind : int
            Column index that the gid is located at in the sc grid.
        """
        row_ind = self.points.loc[gid, "row_ind"]
        col_ind = self.points.loc[gid, "col_ind"]
        return row_ind, col_ind

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
            raise SupplyCurveError(
                'Requested gid "{}" is out of bounds for '
                'supply curve points with length "{}".'.format(gid, len(self))
            )

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

        logger.info(
            "Found {} valid SC points out of {} total possible "
            "(valid SC points that map to valid resource gids)".format(
                len(valid_gids), len(valid_bool)
            )
        )

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
