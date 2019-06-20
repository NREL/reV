"""
reV Supply Curve Points
"""
import pandas as pd
import numpy as np
from reV.handlers.geotiff import Geotiff


class SupplyCurvePoints:
    """Supply curve points framework."""

    def __init__(self, exclusions, resolution=64):
        """
        Parameters
        ----------
        exclusions : str | reV.handlers.geotiff.Geotiff
            File path to the exclusions grid, or pre-extracted exclusions
            geotiff. The exclusions dictate the SC analysis extent.
        resolution : int
            Number of exclusion points per SC point along an axis.
            This number**2 is the total number of exclusion points per
            SC point.
        """

        if isinstance(exclusions, Geotiff):
            self._exclusions = exclusions
        else:
            self._exclusions = Geotiff(exclusions)

        self._res = resolution
        self._cols_of_excl = None
        self._rows_of_excl = None
        self._points = None

    @property
    def exclusions(self):
        """Get the exclusions object.

        Returns
        -------
        _exclusions : reV.handlers.geotiff.Geotiff
            Exclusions geotiff object.
        """
        return self._exclusions

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
    def excl_rows(self):
        """Get the unique row indices identifying the exclusion points.

        Returns
        -------
        excl_rows : np.ndarray
            Array of exclusion row indices.
        """
        return self._exclusions.meta['row_ind'].unique()

    @property
    def excl_cols(self):
        """Get the unique column indices identifying the exclusion points.

        Returns
        -------
        excl_cols : np.ndarray
            Array of exclusion column indices.
        """
        return self._exclusions.meta['col_ind'].unique()

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
            self._rows_of_excl = self._chunk_excl(self.excl_rows)
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
            self._cols_of_excl = self._chunk_excl(self.excl_cols)
        return self._cols_of_excl

    @property
    def points(self):
        """Get the summary dataframe of supply curve points.

        Returns
        -------
        _points : pd.DataFrame
            Supply curve points with columns for attributes of each sc point.
        """

        if self._points is None:
            sc_col_ind, sc_row_ind = np.meshgrid(
                np.arange(len(self.cols_of_excl)),
                np.arange(len(self.rows_of_excl)))
            self._points = pd.DataFrame({'row_ind': sc_row_ind.flatten(),
                                         'col_ind': sc_col_ind.flatten()})
            self._points.index.name = 'gid'
        return self._points

    def get_excl_mask(self, gid):
        """Get the mask of excl points corresponding to the supply curve gid.

        Parameters
        ----------
        gid : int
            Supply curve point gid.

        Returns
        -------
        mask : pd.Series
            Boolean mask to be applied to the exclusions meta.
        """

        sc_row_ind = self.points.loc[gid, 'row_ind']
        sc_col_ind = self.points.loc[gid, 'col_ind']
        excl_rows = self.rows_of_excl[sc_row_ind]
        excl_cols = self.rows_of_excl[sc_col_ind]
        mask = (self.exclusions.meta['row_ind'].isin(excl_rows) &
                self.exclusions.meta['col_ind'].isin(excl_cols))
        return mask

    def get_excl_points(self, gid):
        """Get the exclusions meta points corresponding to a supply curve gid.

        Parameters
        ----------
        gid : int
            Supply curve point gid.

        Returns
        -------
        excl_points : pd.DataFrame
            Exclusions meta df reduced to just the exclusion points associated
            with the input supply curve gid.
        """
        return self.exclusions.meta[self.get_excl_mask(gid)]

    def _chunk_excl(self, arr):
        """Split an array into a list of arrays with len == resolution.

        Parameters
        ----------
        arr : np.ndarray
            1D array to be split into chunks.

        Returns
        -------
        chunks : list
            List of arrays, each with length equal to self.resolution
            (except for the last array in the list which is the remainder).
        """

        chunks = []
        i = 0
        while True:
            if i == len(arr):
                break
            else:
                chunks.append(arr[i:i + self.resolution])
            i = np.min((len(arr), i + self.resolution))

        return chunks
