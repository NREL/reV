# -*- coding: utf-8 -*-
"""
Exclusion layers handler
"""
import json
import logging

import numpy as np
from rex.multi_file_resource import MultiFileResource
from rex.resource import Resource
from rex.utilities.parse_keys import parse_keys

from reV.utilities.exceptions import HandlerKeyError, MultiFileExclusionError

logger = logging.getLogger(__name__)
LATITUDE, LONGITUDE = "latitude", "longitude"


class ExclusionLayers:
    """
    Handler of .h5 file and techmap for Exclusion Layers
    """

    def __init__(self, h5_file, hsds=False):
        """
        Parameters
        ----------
        h5_file : str | list | tuple
            .h5 file containing exclusion layers and techmap,
            or a list of h5 files
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """

        self.h5_file = h5_file

        if isinstance(h5_file, str):
            self._h5 = Resource(h5_file, hsds=hsds)
        elif isinstance(h5_file, (list, tuple)):
            self._h5 = MultiFileResource(h5_file, check_files=False)
            self._preflight_multi_file()
        else:
            msg = ('Expected str, list, or tuple for h5_file input but '
                   'received {}'.format(type(h5_file)))
            logger.error(msg)
            raise TypeError(msg)

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

    def __contains__(self, layer):
        return layer in self.layers

    def _preflight_multi_file(self):
        """Run simple multi-file exclusion checks."""
        lat_shape = self.h5.shapes[LATITUDE]
        lon_shape = self.h5.shapes[LONGITUDE]
        for layer in self.layers:
            lshape = self.h5.shapes[layer]
            lshape = lshape[1:] if len(lshape) > 2 else lshape
            if lshape != lon_shape or lshape != lat_shape:
                msg = ('Shape of layer "{}" is {} which does not match '
                       'latitude and longitude shapes of {} and {}. '
                       'Check your exclusion file inputs: {}'
                       .format(layer, self.h5.shapes[layer],
                               lat_shape, lon_shape, self.h5._h5_files))
                logger.error(msg)
                raise MultiFileExclusionError(msg)

        check_attrs = ('height', 'width', 'crs', 'transform')
        base_profile = {}
        for fp in self.h5_file:
            with ExclusionLayers(fp) as f:
                if not base_profile:
                    base_profile = f.profile
                else:
                    for attr in check_attrs:
                        if attr not in base_profile or attr not in f.profile:
                            msg = ('Multi-file exclusion inputs from {} '
                                   'dont have profiles with height, width, '
                                   'crs, and transform: {} and {}'
                                   .format(self.h5_file, base_profile,
                                           f.profile))
                            logger.error(msg)
                            raise MultiFileExclusionError(msg)

                        base_attr = base_profile[attr]
                        file_attr = f.profile[attr]
                        attrs_are_str = (isinstance(base_attr, str)
                                         and isinstance(file_attr, str))
                        if attr == 'crs' and attrs_are_str:
                            attrs_match = (set(base_attr.split(' '))
                                           == set(file_attr.split(' ')))
                        else:
                            attrs_match = base_profile[attr] == f.profile[attr]

                        if not attrs_match:
                            msg = ('Multi-file exclusion inputs from {} '
                                   'dont have matching "{}": {} and {}'
                                   .format(self.h5_file, attr,
                                           base_profile[attr],
                                           f.profile[attr]))
                            logger.error(msg)
                            raise MultiFileExclusionError(msg)

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
        h5 : rex.MultiFileResource | rex.Resource
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
        """
        return json.loads(self.h5.global_attrs['profile'])

    @property
    def crs(self):
        """
        GeoTiff projection crs

        Returns
        -------
        str
        """
        return self.profile['crs']

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
        """
        layers = self.h5.datasets

        return layers

    @property
    def shape(self):
        """
        Exclusion shape (latitude, longitude)

        Returns
        -------
        shape : tuple
        """
        shape = self.h5.attrs.get('shape', None)
        if shape is None:
            shape = self.h5.shapes[LATITUDE]

        return tuple(shape)

    @property
    def chunks(self):
        """
        Exclusion layers chunks default chunk size

        Returns
        -------
        chunks : tuple | None
            Chunk size of exclusion layers
        """
        chunks = self.h5.attrs.get('chunks', None)
        if chunks is None:
            chunks = self.h5.chunks[LATITUDE]

        return chunks

    @property
    def latitude(self):
        """
        Latitude coordinates array

        Returns
        -------
        ndarray
        """
        return self[LATITUDE]

    @property
    def longitude(self):
        """
        Longitude coordinates array

        Returns
        -------
        ndarray
        """
        return self[LONGITUDE]

    def get_layer_profile(self, layer):
        """
        Get profile for a specific exclusion layer

        Parameters
        ----------
        layer : str
            Layer to get profile for

        Returns
        -------
        profile : dict | None
            GeoTiff profile for single exclusion layer
        """
        profile = self.h5.get_attrs(dset=layer).get('profile', None)
        if profile is not None:
            profile = json.loads(profile)

        return profile

    def get_layer_crs(self, layer):
        """
        Get crs for a specific exclusion layer

        Parameters
        ----------
        layer : str
            Layer to get profile for

        Returns
        -------
        crs : str | None
            GeoTiff projection crs
        """
        profile = self.get_layer_profile(layer)
        if profile is not None:
            crs = profile['crs']
        else:
            crs = None

        return crs

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
        values = self.h5[layer]

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
        description = self.h5.get_attrs(dset=layer).get('description', None)

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
        if LATITUDE not in self.h5:
            msg = ('"latitude" is missing from {}'
                   .format(self.h5_file))
            logger.error(msg)
            raise HandlerKeyError(msg)

        ds_slice = (LATITUDE, ) + ds_slice

        lat = self.h5[ds_slice]

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
        if LONGITUDE not in self.h5:
            msg = ('"longitude" is missing from {}'
                   .format(self.h5_file))
            logger.error(msg)
            raise HandlerKeyError(msg)

        ds_slice = (LONGITUDE, ) + ds_slice

        lon = self.h5[ds_slice]

        return lon

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

        shape = self.h5.get_dset_properties(layer_name)[0]
        if len(shape) == 3:
            ds_slice = (layer_name, 0) + ds_slice
        else:
            ds_slice = (layer_name, ) + ds_slice

        layer_data = self.h5[ds_slice]

        return layer_data
