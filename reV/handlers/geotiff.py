# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:43:34 2019

@author: gbuster
"""

import pandas as pd
import numpy as np
import xarray as xr
from pyproj import transform, Proj

from reV.handlers.sam_resource import parse_keys
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
        self._src = xr.open_rasterio(self._fpath, chunks=chunks)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if type is not None:
            raise

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

    def _get_meta(self, *yx_slice):
        """Get the geotiff meta dataframe in standard WGS84 projection.

        Returns
        -------
        _meta : pd.DataFrame
            Flattened meta data with same format as reV resource meta data.
        """

        if len(yx_slice) == 1:
            y_slice = yx_slice[0]
            x_slice = slice(None, None, None)
        elif len(yx_slice) == 2:
            y_slice = yx_slice[0]
            x_slice = yx_slice[1]
        else:
            raise HandlerKeyError('Cannot do 3D slicing on GeoTiff meta.')

        if y_slice.start is None:
            y_slice = slice(0, y_slice.stop)
        if x_slice.start is None:
            x_slice = slice(0, x_slice.stop)

        lon = self._src.coords['x'].values.astype(np.float32)[x_slice]
        lat = self._src.coords['y'].values.astype(np.float32)[y_slice]

        col_ind = np.arange(x_slice.start, x_slice.start + len(lon))
        row_ind = np.arange(y_slice.start, y_slice.start + len(lat))
        col_ind = col_ind.astype(np.uint32)
        row_ind = row_ind.astype(np.uint32)
        col_ind, row_ind = np.meshgrid(col_ind, row_ind)
        col_ind = col_ind.flatten()
        row_ind = row_ind.flatten()

        lon, lat = np.meshgrid(lon, lat)
        lon = lon.flatten()
        lat = lat.flatten()
        lon, lat = transform(Proj(self._src.attrs['crs']),
                             Proj({"init": "epsg:4326"}),
                             lon, lat)

        meta = pd.DataFrame({'latitude': lat, 'longitude': lon,
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
        if len(ds_slice) == 1:
            y_slice = ds_slice[0]
            x_slice = slice(None, None, None)
        elif len(ds_slice) == 2:
            y_slice = ds_slice[0]
            x_slice = ds_slice[1]
        else:
            raise HandlerKeyError('Cannot do 3D slicing of GeoTiff data '
                                  'within a layer')

        return self._src.data[ds, y_slice, x_slice].flatten()

    def close(self):
        """Close the xarray-rasterio source object"""
        self._src.close()
