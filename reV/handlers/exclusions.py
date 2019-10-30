# -*- coding: utf-8 -*-
"""
Exclusion layers handler
"""
import h5py
import logging
import json
import pandas as pd

from reV.handlers.parse_keys import parse_keys
from reV.handlers.resource import Resource
from reV.utilities.exceptions import HandlerKeyError

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

        self._meta = None

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

        if ds.endswith('meta'):
            out = self._get_meta(ds, *ds_slice)
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
    def profile(self):
        """
        GeoTiff profile for exclusions

        Returns
        -------
        profile : dict
            Generic GeoTiff profile for exclusions in .h5 file
        """
        return json.loads(self._h5.attrs['profile'])

    @property
    def layers(self):
        """
        Available exclusions layers

        Returns
        -------
        layers : list
            List of exclusion layers
        """
        layers = [ds for ds in self._h5
                  if ds != 'meta']
        return layers

    @property
    def shape(self):
        """
        Resource shape (timesteps, sites)
        shape = (len(time_index), len(meta))

        Returns
        -------
        shape : tuple
            Shape of resource variable arrays (timesteps, sites)
        """
        return self._h5.attrs['shape']

    @property
    def meta(self):
        """
        Meta data DataFrame

        Returns
        -------
        meta : pandas.DataFrame
            Resource Meta Data
        """
        if self._meta is None:
            if 'meta' in self.h5:
                self._meta = self._get_meta(slice(None))
            else:
                msg = "'meta' is not a valid dataset"
                logger.exception(msg)
                raise HandlerKeyError(msg)

        return self._meta

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

    def _get_meta(self, *ds_slice):
        """
        Extract and convert meta to a pandas DataFrame

        Parameters
        ----------
        ds_slice : tuple of int | list | slice
            Pandas slicing describing which sites and columns to extract

        Returns
        -------
        meta : pandas.Dataframe
            Dataframe of (lat, lon) to (i, j) mapping
        """
        if 'meta' not in self._h5:
            msg = ('"meta" is missing from {}'
                   .format(self.h5_file))
            logger.error(msg)
            raise HandlerKeyError(msg)

        sites = ds_slice[0]
        if isinstance(sites, int):
            sites = slice(sites, sites + 1)

        meta = self.h5['meta'][sites]

        if isinstance(sites, slice):
            if sites.stop:
                sites = list(range(*sites.indices(sites.stop)))
            else:
                sites = list(range(len(meta)))

        meta = pd.DataFrame(meta, index=sites)
        if len(ds_slice) == 2:
            meta = meta[ds_slice[1]]

        return meta

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

        slices = (0, ) + ds_slice
        layer_data = Resource._extract_ds_slice(self.h5[layer_name],
                                                *slices)

        return layer_data
