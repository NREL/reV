# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:43:34 2019

@author: gbuster
"""

import pandas as pd
import numpy as np
import xarray as xr
from pyproj import transform, Proj

from reV.handlers.parse_keys import parse_keys
from reV.utilities.exceptions import HandlerKeyError


class Geotiff:
    """GeoTIFF handler object."""

    def __init__(self, fpath, chunks=(128, 128)):
        """
        Parameters
        ----------
        fpath : str
            Path to .tiff file.
        chunks : tuple
            GeoTIFF chunk (tile) shape/size.
        """

        self._fpath = fpath
        self._meta = None
        self._iarr = None
        self._src = xr.open_rasterio(self._fpath, chunks=chunks)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if type is not None:
            raise

    def __len__(self):
        """Total number of pixels in the GeoTiff."""
        return self.n_rows * self.n_cols

    def __getitem__(self, keys):
        """Retrieve data from the GeoTIFF object.

        Example, get meta data and layer-0 data for rows 0 through 128 and
        columns 128 through 256.

            meta = geotiff['meta', 0:128, 128:256]
            data = geotiff[0, 0:128, 128:256]

        Parameters
        ----------
        keys : tuple
            Slicing args similar to a numpy array slice. See examples above.
        """
        ds, ds_slice = parse_keys(keys)

        if ds == 'meta':
            out = self._get_meta(*ds_slice)
        else:
            out = self._get_data(ds, *ds_slice)

        return out

    @staticmethod
    def _get_meta_inds(x_slice, y_slice, lon, lat):
        """Get the row and column indices associated with lat/lon slices.

        Parameters
        ----------
        x_slice : slice
            Column slice corresponding to the extracted lon values.
        y_slice : slice
            Row slice corresponding to the extracted lat values.
        lon : np.ndarray
            Extracted lon values (pre-meshgrid) associated with the x_slice.
        lat : np.ndarray
            Extracted lat values (pre-meshgrid) associated with the y_slice.

        Returns
        -------
        row_ind : np.ndarray
            1D array of the row indices corresponding to the lat/lon arrays
            once mesh-gridded and flattened
        col_ind : np.ndarray
            1D array of the col indices corresponding to the lat/lon arrays
            once mesh-gridded and flattened
        """

        if y_slice.start is None:
            y_slice = slice(0, y_slice.stop)
        if x_slice.start is None:
            x_slice = slice(0, x_slice.stop)

        col_ind = np.arange(x_slice.start, x_slice.start + len(lon))
        row_ind = np.arange(y_slice.start, y_slice.start + len(lat))
        col_ind = col_ind.astype(np.uint32)
        row_ind = row_ind.astype(np.uint32)
        col_ind, row_ind = np.meshgrid(col_ind, row_ind)
        col_ind = col_ind.flatten()
        row_ind = row_ind.flatten()

        return row_ind, col_ind

    def _get_meta(self, *ds_slice):
        """Get the geotiff meta dataframe in standard WGS84 projection.

        Parameters
        ----------
        *ds_slice : tuple
            Slicing args for meta data.

        Returns
        -------
        _meta : pd.DataFrame
            Flattened meta data with same format as reV resource meta data.
        """

        y_slice, x_slice = self._unpack_slices(*ds_slice)

        lon = self._src.coords['x'].values.astype(np.float32)[x_slice]
        lat = self._src.coords['y'].values.astype(np.float32)[y_slice]

        row_ind, col_ind = self._get_meta_inds(x_slice, y_slice, lon, lat)

        lon, lat = np.meshgrid(lon, lat)
        lon = lon.flatten()
        lat = lat.flatten()
        lon, lat = transform(Proj(self._src.attrs['crs']),
                             Proj({"init": "epsg:4326"}),
                             lon, lat)

        meta = pd.DataFrame({'latitude': lat.astype(np.float32),
                             'longitude': lon.astype(np.float32),
                             'row_ind': row_ind, 'col_ind': col_ind})
        return meta

    def _get_data(self, ds, *ds_slice):
        """Get the flattened geotiff layer data.

        Parameters
        ----------
        ds : int
            Layer to get data from
        *ds_slice : tuple
            Slicing args for data

        Returns
        -------
        data : np.ndarray
            1D array of flattened data corresponding to meta data.
        """

        y_slice, x_slice = self._unpack_slices(*ds_slice)
        data = self._src.data[ds, y_slice, x_slice].flatten().compute()
        if np.issubdtype(data.dtype, np.float64):
            data = data.astype(np.float32)
        return data

    @staticmethod
    def _unpack_slices(*yx_slice):
        """Get the flattened geotiff layer data.

        Parameters
        ----------
        *yx_slice : tuple
            Slicing args for data

        Returns
        -------
        y_slice : slice
            Row slice.
        x_slice : slice
            Col slice.
        """

        if len(yx_slice) == 1:
            y_slice = yx_slice[0]
            x_slice = slice(None, None, None)
        elif len(yx_slice) == 2:
            y_slice = yx_slice[0]
            x_slice = yx_slice[1]
        else:
            raise HandlerKeyError('Cannot do 3D slicing on GeoTiff meta.')

        return y_slice, x_slice

    @property
    def iarr(self):
        """Get an array of 1D index values for the flattened geotiff extent.

        Returns
        -------
        iarr : np.ndarray
            Uint array with same shape as geotiff extent, representing the 1D
            index values if the geotiff extent was flattened
            (with default flatten order 'C')
        """
        if self._iarr is None:
            self._iarr = np.arange(len(self), dtype=np.uint32)
            self._iarr = self._iarr.reshape(self.shape)
        return self._iarr

    @property
    def shape(self):
        """Get the Geotiff shape tuple (n_rows, n_cols).

        Returns
        -------
        shape : tuple
            2-entry tuple representing the full GeoTiff shape.
        """

        return (self.n_rows, self.n_cols)

    @property
    def n_rows(self):
        """Get the number of Geotiff rows.

        Returns
        -------
        n_rows : int
            Number of row entries in the full geotiff.
        """
        return len(self._src.coords['y'])

    @property
    def n_cols(self):
        """Get the number of Geotiff columns.

        Returns
        -------
        n_cols : int
            Number of column entries in the full geotiff.
        """
        return len(self._src.coords['x'])

    def close(self):
        """Close the xarray-rasterio source object"""
        self._src.close()
