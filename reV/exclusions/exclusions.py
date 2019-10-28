# -*- coding: utf-8 -*-
"""
Generate reV inclusion mask from exclusion layers
"""
from collections import OrderedDict
import logging
import numpy as np
from scipy import ndimage
from warnings import warn

from reV.handlers.exclusions import ExclusionLayers

logger = logging.getLogger(__name__)


class LayerMask:
    """
    Class to convert exclusion layer to inclusion layer mask
    """
    def __init__(self, layer, inclusion_range=(None, None),
                 exclude_values=None, include_values=None):
        """
        Parameters
        ----------
        layer : str
            Layer name
        inclusion_range : tuple
            (min threshold, max threshold) for values to include
        exclude_values : list
            list of values to exclude
            Note: Only supply exclusions OR inclusions
        include_values : list
            List of values to include
            Note: Only supply inclusions OR exclusions
        """
        self._layer = layer
        self._inclusion_range = inclusion_range
        self._exclude_values = exclude_values
        self._include_values = include_values
        self._mask = self._mask_type()

    def __repr__(self):
        msg = ("{} for {} exclusions"
               .format(self.__class__.__name__, self.layer))
        return msg

    def __getitem__(self, data):
        return self.mask_func(data)

    @property
    def layer(self):
        """
        Layer name to extract from exclusions .h5 file

        Returns
        -------
        _layer : str
        """
        return self._layer

    @property
    def min_value(self):
        """
        Minimum value to include

        Returns
        -------
        float
        """
        return self._inclusion_range[0]

    @property
    def max_value(self):
        """
        Maximum value to include

        Returns
        -------
        float
        """
        return self._inclusion_range[1]

    @property
    def exclude_values(self):
        """
        Values to exclude

        Returns
        -------
        _exclude_values : list
        """
        return self._exclude_values

    @property
    def include_values(self):
        """
        Values to include

        Returns
        -------
        _include_values : list
        """
        return self._include_values

    @property
    def mask_func(self):
        """
        Function to use for masking exclusion data

        Returns
        -------
        func
            Masking function
        """
        if self._mask == 'range':
            func = self._range_mask
        elif self._mask == 'exclude':
            func = self._exclusion_mask
        elif self._mask == 'include':
            func = self._inclusion_mask
        else:
            msg = ('{} is an invalid mask type: expecting '
                   '"range", "exclude", or "include"'
                   .format(self._mask))
            logger.error(msg)
            raise ValueError(msg)

        return func

    def _mask_type(self):
        """
        Ensure that the initialization arguments are valid and not
        contradictory

        Returns
        ------
        mask : str
            Mask type
        """
        masks = {'range': any(i is not None for i in self._inclusion_range),
                 'exclude': self._exclude_values is not None,
                 'include': self._include_values is not None}
        mask = None
        for k, v in masks.items():
            if v:
                if mask is None:
                    mask = k
                else:
                    msg = ('Only one approach can be used to create the '
                           'inclusion mask, but you supplied {} and {}'
                           .format(mask, k))
                    logger.error(msg)
                    raise RuntimeError(msg)

        return mask

    def _range_mask(self, data):
        """
        Mask exclusion layer based on value range

        Parameters
        ----------
        data : ndarray
            Exclusions data to create mask from

        Returns
        -------
        mask : ndarray
            Boolean mask of which values to include
        """
        mask = None
        if self.min_value is not None:
            mask = data >= self.min_value

        if self.max_value is not None:
            if mask is not None:
                mask *= data <= self.max_value
            else:
                mask = data <= self.max_value

        return mask

    @staticmethod
    def _value_mask(data, values, include=True):
        """
        Mask exclusion layer based on values to include or exclude

        Parameters
        ----------
        data : ndarray
            Exclusions data to create mask from
        values : list
            Values to include or exclude
        include : boolean
            Flag as to whether values should be included or excluded

        Returns
        -------
        mask : ndarray
            Boolean mask of which values to include
        """
        mask = np.isin(data, values)

        if not include:
            mask = ~mask

        return mask

    def _exclusion_mask(self, data):
        """
        Mask exclusion layer based on values to exclude

        Parameters
        ----------
        data : ndarray
            Exclusions data to create mask from

        Returns
        -------
        mask : ndarray
            Boolean mask of which values to include
        """
        mask = self._value_mask(data, self._exclude_values, include=False)

        return mask

    def _inclusion_mask(self, data):
        """
        Mask exclusion layer based on values to include

        Parameters
        ----------
        data : ndarray
            Exclusions data to create mask from

        Returns
        -------
        mask : ndarray
            Boolean mask of which values to include
        """
        mask = self._value_mask(data, self._include_values, include=True)

        return mask


class InclusionMask:
    """
    Class to create final inclusion mask
    """

    FILTER_KERNELS = {
        'queen': np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]]),
        'rook': np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]])}

    def __init__(self, excl_h5, *layers, contiguous_filter='queen',
                 hsds=False):
        """
        Parameters
        ----------
        excl_h5 : str
            Path to exclusions .h5 file
        layers : LayerMask
            Instance of LayerMask for each exclusion layer to combine
        contiguous_filter : str | None
            Contiguous filter method to use on final exclusion
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        self._layers = OrderedDict()
        self._excl_h5 = excl_h5
        self._hsds = hsds
        self._excl_layers = None

        for layer in layers:
            self.add_layer(layer)

        if contiguous_filter in [None, "queen", "rook"]:
            self._contiguous_filter = contiguous_filter
        else:
            raise ValueError('contiguous_filter must be "queen", "rook" '
                             ' or "None"')

    def __repr__(self):
        msg = ("{} from {} with {} input layers"
               .format(self.__class__.__name__, self.excl_h5, len(self)))
        return msg

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, *ds_slice):
        return self._generate_mask(*ds_slice)

    @property
    def excl_h5(self):
        """
        Path to .h5 file containing exclusion layers

        Returns
        -------
        _excl_h5 : str
         """
        return self._excl_h5

    @property
    def excl_layers(self):
        """
        List of available exclusion layers in exclusions .h5

        Returns
        -------
        _excl_layers : list
        """
        if self._excl_layers is None:
            with ExclusionLayers(self._excl_h5, hsds=self._hsds) as excl:
                self._excl_layers = excl.layers

        return self._excl_layers

    @property
    def layer_names(self):
        """
        List of layers to combines

        Returns
        -------
        list
         """
        return self._layers.keys()

    @property
    def layers(self):
        """
        List of LayerMask instances for each exclusion layer to combine

        Returns
        -------
         list
         """
        return self._layers.values()

    @property
    def mask(self):
        """
        Inclusion mask for entire exclusion domain

        Returns
        -------
        mask : ndarray
        """
        mask = self[...]
        return mask

    def add_layer(self, layer, replace=False):
        """
        Add layer to be combined

        Parameters
        ----------
        layer : LayerMask
            LayerMask instance to add to set of layers to be combined
        """
        layer_name = layer.layer

        if layer_name not in self.excl_layers:
            msg = "{} does not existin in {}".format(layer_name, self._excl_h5)
            logger.error(msg)
            raise ValueError(layer_name)

        if layer_name in self.layer_names:
            msg = "{} is already in {}".format(layer_name, self)
            if replace:
                msg += " replacing existing layer"
                logger.warning(msg)
                warn(msg)
            else:
                logger.error(msg)
                raise RuntimeError(msg)

        self._layers[layer_name] = layer

    def _generate_mask(self, *ds_slice):
        """
        Generate inclusion mask from exclusion layers

        Parameters
        ----------
        ds_slice : int | slice | list | ndarray
            What to extract from ds, each arg is for a sequential axis

        Returns
        -------
        mask : ndarray
            Inclusion mask
        """
        mask = None
        for layer in self.layers:
            with ExclusionLayers(self._excl_h5, hsds=self._hsds) as f:
                layer_slice = (layer.layer, ) + ds_slice
                layer_mask = layer[f[layer_slice]]
                if mask is None:
                    mask = layer_mask
                else:
                    mask *= layer_mask

        if self._contiguous_filter is not None:
            kernel = self.FILTER_KERNELS[self._contiguous_filter]
            mask = ndimage.convolve(mask, kernel, mode='constant', cval=0.0)

        return mask

    @classmethod
    def run(cls, excl_h5, *layers, contiguous_filter='queen', hsds=False):
        """
        Create inclusion mask from given layers

        Parameters
        ----------
        excl_h5 : str
            Path to exclusions .h5 file
        layers : LayerMask
            Instance of LayerMask for each exclusion layer to combine
        contiguous_filter : str | None
            Contiguous filter method to use on final exclusion
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS

        Returns
        -------
        mask : ndarray
            Full inclusion mask
        """
        mask = cls(excl_h5, *layers, contiguous_filter=contiguous_filter,
                   hsds=hsds)
        return mask.mask

    @classmethod
    def run_from_dict(cls, excl_h5, layers_dict,
                      contiguous_filter='queen', hsds=False):
        """
        Create inclusion mask from dictionary of LayerMask arguments

        Parameters
        ----------
        excl_h5 : str
            Path to exclusions .h5 file
        layers_dict : dcit
            Dictionary of LayerMask arugments {layer: {kwarg: value}}
        contiguous_filter : str | None
            Contiguous filter method to use on final exclusion
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS

        Returns
        -------
        mask : ndarray
            Full inclusion mask
        """
        layers = []
        for layer, kwargs in layers_dict:
            layers.append(LayerMask(layer, **kwargs))

        mask = cls(excl_h5, *layers, contiguous_filter=contiguous_filter,
                   hsds=hsds)
        return mask.mask
