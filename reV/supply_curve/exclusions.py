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
                 exclude_values=None, include_values=None,
                 use_as_weights=False, weight=1.0):
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
        use_as_weights : bool
            Use layer as final inclusion weights
        weight : float
            How much to weight the inclusion of each pixel, Default = 1
        """
        self._layer = layer
        self._inclusion_range = inclusion_range
        self._exclude_values = exclude_values
        self._include_values = include_values
        self._as_weights = use_as_weights
        self._weight = weight
        self._mask_type = self._check_mask_type()

    def __repr__(self):
        msg = ("{} for {} exclusion, of type {}"
               .format(self.__class__.__name__, self.layer, self.mask_type))
        return msg

    def __getitem__(self, data):
        return self._apply_mask(data)

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
    def mask_type(self):
        """
        Type of exclusion mask for this layer

        Returns
        -------
        str
        """
        return self._mask_type

    def _apply_mask(self, data):
        """
        Apply mask function

        Parameters
        ----------
        data : ndarray
            Exclusions data to create mask from

        Returns
        -------
        data : ndarray
            Masked exclusion data with weights applied
        """
        if not self._as_weights:
            if self.mask_type == 'range':
                func = self._range_mask
            elif self.mask_type == 'exclude':
                func = self._exclusion_mask
            elif self.mask_type == 'include':
                func = self._inclusion_mask
            else:
                msg = ('{} is an invalid mask type: expecting '
                       '"range", "exclude", or "include"'
                       .format(self.mask_type))
                logger.error(msg)
                raise ValueError(msg)

            data = func(data)

        data *= self._weight

        return data

    def _check_mask_type(self):
        """
        Ensure that the initialization arguments are valid and not
        contradictory

        Returns
        ------
        mask : str
            Mask type
        """
        mask = None
        if not self._as_weights:
            masks = {'range': any(i is not None
                                  for i in self._inclusion_range),
                     'exclude': self._exclude_values is not None,
                     'include': self._include_values is not None}
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
        mask = True
        if self.min_value is not None:
            mask = data >= self.min_value

        if self.max_value is not None:
            mask *= data <= self.max_value

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


class ExclusionMask:
    """
    Class to create final exclusion mask
    """

    FILTER_KERNELS = {
        'queen': np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]),
        'rook': np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])}

    def __init__(self, excl_h5, *layers, min_area=None,
                 kernel='queen', hsds=False):
        """
        Parameters
        ----------
        excl_h5 : str
            Path to exclusions .h5 file
        layers : LayerMask
            Instance of LayerMask for each exclusion layer to combine
        min_area : float | NoneType
            Minimum required contiguous area in sq-km
        kernel : str
            Contiguous filter method to use on final exclusion
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        self._layers = OrderedDict()
        self._excl_h5 = ExclusionLayers(excl_h5, hsds=hsds)
        self._excl_layers = None

        for layer in layers:
            self.add_layer(layer)

        if kernel in ["queen", "rook"]:
            self._min_area = min_area
            self._kernel = kernel
        else:
            raise ValueError('kernel must be "queen" or "rook"')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

        if type is not None:
            raise

    def __repr__(self):
        msg = ("{} from {} with {} input layers"
               .format(self.__class__.__name__, self.excl_h5.h5_file,
                       len(self)))
        return msg

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, *ds_slice):
        return self._generate_mask(*ds_slice)

    def close(self):
        """
        Close h5 instance
        """
        self.excl_h5.close()

    @property
    def shape(self):
        """
        Get the exclusions shape.

        Returns
        -------
        shape : tuple
            (rows, cols) shape tuple
        """
        return self.excl_h5.shape

    @property
    def excl_h5(self):
        """
        Open ExclusionLayers instance

        Returns
        -------
        _excl_h5 : ExclusionLayers
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
            self._excl_layers = self.excl_h5.layers

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

    @staticmethod
    def _area_filter(mask, min_area=1, kernel='queen', ex_area=0.0081):
        """
        Ensure the contiguous area of included pixels is greater than
        prescribed minimum in sq-km

        Parameters
        ----------
        mask : ndarray
            Inclusion mask
        min_area : float
            Minimum required contiguous area in sq-km
        kernel : str
            Kernel type, either 'queen' or 'rook'
        ex_area : float
            Area of each exclusion pixel in km^2, assumes 90m resolution

        Returns
        -------
        mask : ndarray
            Updated inclusion mask
        """
        s = ExclusionMask.FILTER_KERNELS[kernel]
        labels, _ = ndimage.label(mask, structure=s)
        l, c = np.unique(labels, return_counts=True)

        min_counts = np.ceil(min_area / ex_area)
        pos = c[1:] < min_counts
        bad_labels = l[1:][pos]

        mask[np.isin(labels, bad_labels)] = False

        return mask

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
        if len(ds_slice) == 1 & isinstance(ds_slice[0], tuple):
            ds_slice = ds_slice[0]

        for layer in self.layers:
            layer_slice = (layer.layer, ) + ds_slice
            layer_mask = layer[self.excl_h5[layer_slice]]
            if mask is None:
                mask = layer_mask
            else:
                mask *= layer_mask

        if self._min_area is not None:
            mask = self._area_filter(mask, min_area=self._min_area,
                                     kernel=self._kernel)

        return mask

    @classmethod
    def run(cls, excl_h5, *layers, min_area=None,
            kernel='queen', hsds=False):
        """
        Create inclusion mask from given layers

        Parameters
        ----------
        excl_h5 : str
            Path to exclusions .h5 file
        layers : LayerMask
            Instance of LayerMask for each exclusion layer to combine
        min_area : float | NoneType
            Minimum required contiguous area in sq-km
        kernel : str
            Contiguous filter method to use on final exclusion
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS

        Returns
        -------
        mask : ndarray
            Full inclusion mask
        """
        with cls(excl_h5, *layers, min_area=min_area,
                 kernel=kernel, hsds=hsds) as f:
            mask = f.mask

        return mask

    @classmethod
    def run_from_dict(cls, excl_h5, layers_dict, min_area=None,
                      kernel='queen', hsds=False):
        """
        Create inclusion mask from dictionary of LayerMask arguments

        Parameters
        ----------
        excl_h5 : str
            Path to exclusions .h5 file
        layers_dict : dcit
            Dictionary of LayerMask arugments {layer: {kwarg: value}}
        min_area : float | NoneType
            Minimum required contiguous area in sq-km
        kernel : str
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
        for layer, kwargs in layers_dict.items():
            layers.append(LayerMask(layer, **kwargs))

        with cls(excl_h5, *layers, min_area=min_area,
                 kernel=kernel, hsds=hsds) as f:
            mask = f.mask

        return mask

    @classmethod
    def from_dict(cls, excl_h5, layers_dict, min_area=None,
                  kernel='queen', hsds=False):
        """
        Create inclusion handler from dictionary of LayerMask arguments

        Parameters
        ----------
        excl_h5 : str
            Path to exclusions .h5 file
        layers_dict : dcit
            Dictionary of LayerMask arugments {layer: {kwarg: value}}
        min_area : float | NoneType
            Minimum required contiguous area in sq-km
        kernel : str
            Contiguous filter method to use on final exclusion
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS

        Returns
        -------
        incl_mask : InclusionMask
            Initialized inclusion mask object.
        """
        layers = []
        for layer, kwargs in layers_dict.items():
            layers.append(LayerMask(layer, **kwargs))

        incl_mask = cls(excl_h5, *layers, min_area=min_area,
                        kernel=kernel, hsds=hsds)
        return incl_mask
