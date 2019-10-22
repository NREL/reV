# -*- coding: utf-8 -*-
"""
Generate reV inclusion mask from exclusion layers
"""
import logging
import numpy as np
from scipy import ndimage

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
        masks = {'range': all(i is None for i in self._inclusion_range),
                 'exclude': self._exclude_values is None,
                 'include': self._include_values is None}
        mask = None
        for k, v in masks.items():
            if v:
                if mask is None:
                    mask = k
                else:
                    msg = ('Only one approach can be used to create the '
                           'inclusion mask, but you supplied {} and {}'
                           .format(mask, v))
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


class Inclusions:
    """
    Class to create final inclusion
    """

    FILTER_KERNELS = {
        'queen': np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]]),
        'rook': np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]])}

    def __init__(self, layer_configs=None, use_blocks=False,
                 contiguous_filter=None):
        """
        Parameters
        ----------
        layer_configs : dictionary | list | None
            Optional configs list for the addition of layers
        use_blocks : boolean
            Use blocks when applying layers to exclusions
        contiguous_filter : str | None
            Contiguous filter method to use on final exclusion
        """
        self._layers = []
        self._data = None
        self._profile = None
        # validate and set use_blocks argument
        if isinstance(use_blocks, bool):
            self._use_blocks = use_blocks
        else:
            raise TypeError('use_blocks argument must be a boolean')
        # validate and set contiguous filter argument
        if contiguous_filter in [None, "queen", "rook"]:
            self._contiguous_filter = contiguous_filter
        else:
            raise TypeError('contiguous_filter must be "queen" or "rook"')
        # validate and set layer_configs argument
        if isinstance(layer_configs, (list, type(None))):
            self._layer_configs = layer_configs
        else:
            raise TypeError('layer_configs argument must be a list or None')

    @property
    def layers(self):
        """ Get the layers for exclusion """
        return self._layers

    @property
    def data(self):
        """ Get the data for exclusion """
        return self._data

    @property
    def profile(self):
        """ Get the profile for exclusion """
        return self._profile

    @property
    def use_blocks(self):
        """ Get the blocks setting for exclusion """
        return self._use_blocks

    @property
    def contiguous_filter(self):
        """ Get the contiguous filter method for exclusion """
        return self._contiguous_filter

    def apply_filter(self, contiguous_filter=None):
        """ Read, process, and apply an input layer to the
        final output layer

        Parameters
        ----------
        contiguous_filter : str | None
            Contiguous filter method to use on final exclusion
        """

        if not contiguous_filter:
            contiguous_filter = self._contiguous_filter
        if contiguous_filter not in [None, "queen", "rook"]:
            raise TypeError('contiguous_filter must be "queen" or "rook"')

        if isinstance(self._data, type(None)):
            raise AttributeError('Exclusion has not been created yet'
                                 '(i.e. self.apply_layer())')
        if isinstance(contiguous_filter, type(None)):
            logger.info('No contiguous filter provided')
            return None

        mask = (self._data != 0).astype('int8')
        kernel = self.FILTER_KERNELS[contiguous_filter]
        mask = ndimage.convolve(mask, kernel, mode='constant', cval=0.0)

        return None
