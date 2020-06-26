# -*- coding: utf-8 -*-
"""
Exclusion layers handler
"""
import h5py
import logging
import json
import numpy as np

from reV.utilities.exceptions import HandlerKeyError

from rex.utilities.parse_keys import parse_keys
from rex.resource import ResourceDataset

logger = logging.getLogger(__name__)


class ExclusionLayers:
    """
    Handler of .h5 file and techmap for Exclusion Layers
    """
    def __init__(self, h5_file, hsds=False):
        """
        Parameters
        ----------
        h5_file : str
            .h5 file containing exclusion layers and techmap
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        self.h5_file = h5_file
        if hsds:
            import h5pyd
            self._h5 = h5pyd.File(self.h5_file, 'r')
        else:
            self._h5 = h5py.File(self.h5_file, 'r')

        self._iarr = None

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self.h5_file)

        return msg

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

        if type is not None:
            raise

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, keys):
        ds, ds_slice = parse_keys(keys)

        if ds.lower().startswith('lat'):
            out = self._get_latitude(*ds_slice)
        elif ds.lower().startswith('lon'):
            out = self._get_longitude(*ds_slice)
        else:
            out = self._get_layer(ds, *ds_slice)

        return out

    def close(self):
        """
        Close h5 instance
        """
        self._h5.close()

    @property
    def h5(self):
        """
        Open h5py File instance.

        Returns
        -------
        h5 : h5py.File | h5py.Group
            Open h5py File or Group instance
        """
        return self._h5

    @property
    def iarr(self):
        """Get an array of 1D index values for the flattened h5 excl extent.

        Returns
        -------
        iarr : np.ndarray
            Uint array with same shape as exclusion extent, representing the 1D
            index values if the geotiff extent was flattened
            (with default flatten order 'C')
        """
        if self._iarr is None:
            N = self.shape[0] * self.shape[1]
            self._iarr = np.arange(N, dtype=np.uint32)
            self._iarr = self._iarr.reshape(self.shape)

        return self._iarr

    @property
    def profile(self):
        """
        GeoTiff profile for exclusions

        Returns
        -------
        profile : dict
            Generic GeoTiff profile for exclusions in .h5 file
        """
        return json.loads(self.h5.attrs['profile'])

    @property
    def pixel_area(self):
        """Get pixel area in km2 from the transform profile of the excl file.

        Returns
        -------
        area : float
            Exclusion pixel area in km2. Will return None if the
            appropriate transform attribute is not found.
        """

        area = None
        if 'transform' in self.profile:
            transform = self.profile['transform']
            area = np.abs(transform[0] * transform[4])
            area /= 1000 ** 2

        return area

    @property
    def layers(self):
        """
        Available exclusions layers

        Returns
        -------
        layers : list
            List of exclusion layers
        """
        layers = [ds for ds in self.h5
                  if ds != 'meta']

        return layers

    @property
    def shape(self):
        """
        Exclusion shape (latitude, longitude)

        Returns
        -------
        shape : tuple
            Shape of exclusion array (latitude, longitude)
        """
        shape = self.h5.attrs.get('shape', None)
        if shape is None:
            shape = self.h5['latitude'].shape

        return tuple(shape)

    @property
    def latitude(self):
        """
        Latitude coordinates array

        Returns
        -------
        ndarray
        """
        return self['latitude']

    @property
    def longitude(self):
        """
        Longitude coordinates array

        Returns
        -------
        ndarray
        """
        return self['longitude']

    def get_layer_profile(self, layer):
        """
        Get profile for a specific exclusion layer

        Parameters
        ----------
        layer : str
            Layer to get profile for

        Returns
        -------
        profile : dict
            GeoTiff profile for single exclusion layer
        """
        profile = json.loads(self.h5[layer].attrs['profile'])

        return profile

    def get_layer_values(self, layer):
        """
        Get values for given layer in Geotiff format (bands, y, x)

        Parameters
        ----------
        layer : str
            Layer to get values for

        Returns
        -------
        values : ndarray
            GeoTiff values for single exclusion layer
        """
        values = self.h5[layer][...]

        return values

    def get_layer_description(self, layer):
        """
        Get description for given layer

        Parameters
        ----------
        layer : str
            Layer to get description for

        Returns
        -------
        description : str
            Description of layer
        """
        description = self.h5[layer].attrs['description']

        return description

    def get_nodata_value(self, layer):
        """
        Get the nodata value for a given layer

        Parameters
        ----------
        layer : str
            Layer to get nodata value for

        Returns
        -------
        nodata : int | float | None
            nodata value for layer or None if not found
        """
        profile = self.get_layer_profile(layer)
        nodata = profile.get('nodata', None)

        return nodata

    def _get_latitude(self, *ds_slice):
        """
        Extract latitude coordinates

        Parameters
        ----------
        ds_slice : tuple of int | list | slice
            Pandas slicing describing which sites and columns to extract

        Returns
        -------
        lat : ndarray
            Latitude coordinates
        """
        if 'latitude' not in self.h5:
            msg = ('"latitude" is missing from {}'
                   .format(self.h5_file))
            logger.error(msg)
            raise HandlerKeyError(msg)

        lat = ResourceDataset.extract(self.h5['latitude'], ds_slice)

        return lat

    def _get_longitude(self, *ds_slice):
        """
        Extract longitude coordinates

        Parameters
        ----------
        ds_slice : tuple of int | list | slice
            Pandas slicing describing which sites and columns to extract

        Returns
        -------
        lon : ndarray
            Longitude coordinates
        """
        if 'longitude' not in self.h5:
            msg = ('"longitude" is missing from {}'
                   .format(self.h5_file))
            logger.error(msg)
            raise HandlerKeyError(msg)

        lat = ResourceDataset.extract(self.h5['longitude'], ds_slice)

        return lat

    def _get_layer(self, layer_name, *ds_slice):
        """
        Extract data from given dataset

        Parameters
        ----------
        layer_name : str
            Exclusion layer to extract
        ds_slice : tuple of int | list | slice
            tuple describing slice of layer array to extract

        Returns
        -------
        layer_data : ndarray
            Array of exclusion data
        """
        if layer_name not in self.layers:
            msg = ('{} not in available layers: {}'
                   .format(layer_name, self.layers))
            logger.error(msg)
            raise HandlerKeyError(msg)

        if len(self.h5[layer_name].shape) == 3:
            slices = (0, ) + ds_slice
        else:
            slices = ds_slice

        layer_data = ResourceDataset.extract(self.h5[layer_name], slices)

        return layer_data
