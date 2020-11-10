# -*- coding: utf-8 -*-
"""
Generate reV inclusion mask from exclusion layers
"""
import logging
import numpy as np
from scipy import ndimage
from warnings import warn

from reV.handlers.exclusions import ExclusionLayers
from reV.utilities.exceptions import ExclusionLayerError

logger = logging.getLogger(__name__)


class LayerMask:
    """
    Class to convert exclusion layer to inclusion layer mask
    """
    def __init__(self, layer, inclusion_range=(None, None),
                 exclude_values=None, include_values=None,
                 inclusion_weights=None, force_include_values=None,
                 use_as_weights=False, weight=1.0,
                 exclude_nodata=False, nodata_value=None):
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
        inclusions_weights : dict
            Include given values with given weights
        force_include_values : list
            Force the inclusion of the given values
        use_as_weights : bool
            Use layer as final inclusion weights
        weight : float
            How much to weight the inclusion of each pixel, Default = 1
        exclude_nodata : bool
            Flag to exclude nodata values (nodata_value). If nodata_value=None
            the nodata_value is infered by ExclusionMask
        nodata_value : int | float | None
            Nodata value for the layer. If None, it will be infered when
            LayerMask is added to ExclusionMask
        """
        self._layer = layer
        self._inclusion_range = inclusion_range
        self._exclude_values = exclude_values
        self._inclusion_weights = inclusion_weights
        if force_include_values is not None:
            self._include_values = force_include_values
            self._force_include = True
        else:
            self._include_values = include_values
            self._force_include = False

        self._as_weights = use_as_weights
        self._exclude_nodata = exclude_nodata
        self.nodata_value = nodata_value

        if weight > 1 or weight < 0:
            msg = ('Invalide weight ({}) provided for layer {}:'
                   '\nWeight must fall between 0 and 1!'.format(weight, layer))
            logger.error(msg)
            raise ValueError(msg)

        self._weight = weight
        self._mask_type = self._check_mask_type()

    def __repr__(self):
        msg = ("{} for {} exclusion, of type {}"
               .format(self.__class__.__name__, self.layer, self.mask_type))

        return msg

    def __getitem__(self, data):
        """Get the multiplicative inclusion mask.

        Returns
        -------
        mask : ndarray
            Masked exclusion data with weights applied such that 1 is included,
            0 is excluded, 0.5 is half included.
        """
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
    def inclusion_weights(self):
        """
        Mapping of values to include and at what weights

        Returns
        -------
        _inclusion_weights : dict
        """
        return self._inclusion_weights

    @property
    def force_include(self):
        """
        Flag to force include mask

        Returns
        -------
        _force_include : bool
        """
        return self._force_include

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
            Masked exclusion data with weights applied such that 1 is included,
            0 is excluded, 0.5 is half included.
        """

        if not self._as_weights:
            if self.mask_type == 'range':
                func = self._range_mask
            elif self.mask_type == 'exclude':
                func = self._exclusion_mask
            elif self.mask_type == 'include':
                func = self._inclusion_mask
            elif self.mask_type == 'inclusion_weights':
                func = self._weights_mask
            else:
                msg = ('{} is an invalid mask type: expecting '
                       '"range", "exclude", "include", or "inclusion_weights"'
                       .format(self.mask_type))
                logger.error(msg)
                raise KeyError(msg)

            data = func(data)

        data = data.astype('float16') * self._weight

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
                     'include': self._include_values is not None,
                     'inclusion_weights': self._inclusion_weights is not None}
            for k, v in masks.items():
                if v:
                    if mask is None:
                        mask = k
                    else:
                        msg = ('Only one approach can be used to create the '
                               'inclusion mask, but you supplied {} and {}'
                               .format(mask, k))
                        logger.error(msg)
                        raise ExclusionLayerError(msg)

        if mask == 'inclusion_weights' and self._weight < 1:
            msg = ("Values are individually weighted when using "
                   "'inclusion_weights', the supplied weight of {} will be "
                   "ignored!".format(self._weight))
            self._weight = 1
            logger.warning(msg)
            warn(msg)

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
            Boolean mask of which values to include (True is include).
        """
        mask = True
        if self.min_value is not None:
            mask = data >= self.min_value

        if self.max_value is not None:
            mask *= data <= self.max_value

        if self._exclude_nodata and self.nodata_value is not None:
            mask = mask & (data != self.nodata_value)

        return mask

    def _value_mask(self, data, values, include=True):
        """
        Mask exclusion layer based on values to include or exclude

        Parameters
        ----------
        data : ndarray
            Exclusions data to create mask from
        values : list
            Values to include or exclude.
        include : boolean
            Flag as to whether values should be included or excluded.
            If True, output mask will be True where data == values.
            If False, output mask will be True where data != values.

        Returns
        -------
        mask : ndarray
            Boolean mask of which values to include (True is include)
        """
        mask = np.isin(data, values)

        if not include:
            mask = ~mask

        # only include if not nodata
        if self._exclude_nodata and self.nodata_value is not None:
            mask = mask & (data != self.nodata_value)

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
            Boolean mask of which values to include (True is include)
        """
        mask = self._value_mask(data, self.exclude_values, include=False)

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
            Boolean mask of which values to include (True is include)
        """
        mask = self._value_mask(data, self.include_values, include=True)

        return mask

    def _weights_mask(self, data):
        """
        Mask exclusion layer based on the weights for each inclusion value

        Parameters
        ----------
        data : ndarray
            Exclusions data to create mask from

        Returns
        -------
        mask : ndarray
            Percentage of value to include
        """
        mask = None
        for value, weight in self.inclusion_weights.items():
            if isinstance(value, str):
                value = float(value)

            if mask is None:
                mask = self._value_mask(data, [value], include=True) * weight
            else:
                mask += self._value_mask(data, [value], include=True) * weight

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

    def __init__(self, excl_h5, layers=None, min_area=None,
                 kernel='queen', hsds=False, check_layers=False):
        """
        Parameters
        ----------
        excl_h5 : str
            Path to exclusions .h5 file
        layers : list | NoneType
            list of LayerMask instances for each exclusion layer to combine
        min_area : float | NoneType
            Minimum required contiguous area in sq-km
        kernel : str
            Contiguous filter method to use on final exclusion
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        check_layers : bool
            Run a pre-flight check on each layer to ensure they contain
            un-excluded values
        """
        self._layers = {}
        self._excl_h5 = ExclusionLayers(excl_h5, hsds=hsds)
        self._excl_layers = None
        self._check_layers = check_layers

        if layers is not None:
            if not isinstance(layers, list):
                layers = [layers]

            for layer in layers:
                self.add_layer(layer)

        if kernel in ["queen", "rook"]:
            self._min_area = min_area
            self._kernel = kernel
            logger.debug('Initializing Exclusions mask with min area of {} '
                         'km2 and filter kernel "{}".'
                         .format(self._min_area, self._kernel))
        else:
            raise KeyError('kernel must be "queen" or "rook"')

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
        """Get the multiplicative inclusion mask.

        Parameters
        ----------
        ds_slice : int | slice | list | ndarray
            What to extract from ds, each arg is for a sequential axis.
            For example, (slice(0, 64), slice(0, 64)) will extract a 64x64
            exclusions mask.

        Returns
        -------
        mask : ndarray
            Multiplicative inclusion mask with all layers multiplied together
            ("and" operation) such that 1 is included, 0 is excluded,
            0.5 is half.
        """
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
        ndarray
        """
        mask = self[...]
        return mask

    @property
    def latitude(self):
        """
        Latitude coordinates array

        Returns
        -------
        ndarray
        """
        return self.excl_h5['latitude']

    @property
    def longitude(self):
        """
        Longitude coordinates array

        Returns
        -------
        ndarray
        """
        return self.excl_h5['longitude']

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
            raise KeyError(layer_name)

        if layer_name in self.layer_names:
            msg = "{} is already in {}".format(layer_name, self)
            if replace:
                msg += " replacing existing layer"
                logger.warning(msg)
                warn(msg)
            else:
                logger.error(msg)
                raise ExclusionLayerError(msg)

        layer.nodata_value = self.excl_h5.get_nodata_value(layer_name)
        if self._check_layers:
            if not layer[self.excl_h5[layer_name]].any():
                msg = ("Layer {} does not have any un-excluded pixels!"
                       .format(layer_name))
                logger.error(msg)
                raise ExclusionLayerError(msg)

        self._layers[layer_name] = layer

    @property
    def nodata_lookup(self):
        """Get a dictionary lookup of the nodata values for each layer name.

        Returns
        -------
        nodata : dict
            Lookup keyed by layer name and values are nodata values for the
            respective layers.
        """
        nodata = {}
        for layer_name in self.layer_names:
            nodata[layer_name] = self.excl_h5.get_nodata_value(layer_name)

        return nodata

    @staticmethod
    def _area_filter(mask, min_area=1, kernel='queen', excl_area=0.0081):
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
        excl_area : float
            Area of each exclusion pixel in km^2, assumes 90m resolution

        Returns
        -------
        mask : ndarray
            Updated inclusion mask
        """
        s = ExclusionMask.FILTER_KERNELS[kernel]
        labels, _ = ndimage.label(mask > 0, structure=s)
        l, c = np.unique(labels, return_counts=True)

        min_counts = np.ceil(min_area / excl_area)
        pos = c[1:] < min_counts
        bad_labels = l[1:][pos]

        mask[np.isin(labels, bad_labels)] = 0

        return mask

    def _increase_mask_slice(self, ds_slice, n=1):
        """Increase the mask slice, e.g. from 64x64 to 192x192, to help the
        contiguous area filter be more accurate.

        Parameters
        ----------
        ds_slice : tuple
            Two entry tuple with x and y slices. Anything else will be passed
            through unaffected.
        n : int
            Number of blocks to increase in each direction. For example,
            a 64x64 slice with n=1 will increase to 192x192
            (increases by 64xn in each direction).

        Returns
        -------
        new_slice : tuple
            Two entry tuple with x and y slices with increased dimensions.
        sub_slice : tuple
            Two entry tuple with x and y slices to retrieve the original
            slice out of the bigger slice.
        """
        new_slice = ds_slice
        sub_slice = (slice(None), slice(None))

        if isinstance(ds_slice, tuple) and len(ds_slice) == 2:
            y_slice = ds_slice[0]
            x_slice = ds_slice[1]
            if isinstance(x_slice, slice) and isinstance(y_slice, slice):
                y_diff = n * np.abs(y_slice.stop - y_slice.start)
                x_diff = n * np.abs(x_slice.stop - x_slice.start)

                y_new_start = int(np.max((0, (y_slice.start - y_diff))))
                x_new_start = int(np.max((0, (x_slice.start - x_diff))))

                y_new_stop = int(np.min((self.shape[0],
                                         (y_slice.stop + y_diff))))
                x_new_stop = int(np.min((self.shape[1],
                                         (x_slice.stop + x_diff))))

                new_slice = (slice(y_new_start, y_new_stop),
                             slice(x_new_start, x_new_stop))

                if y_new_start == y_slice.start:
                    y_sub_start = 0
                else:
                    y_sub_start = int(n * y_diff)
                if x_new_start == x_slice.start:
                    x_sub_start = 0
                else:
                    x_sub_start = int(n * x_diff)

                y_sub_stop = y_sub_start + y_diff
                x_sub_stop = x_sub_start + x_diff

                sub_slice = (slice(y_sub_start, y_sub_stop),
                             slice(x_sub_start, x_sub_stop))

        return new_slice, sub_slice

    def _generate_ones_mask(self, ds_slice):
        """
        Generate mask of all ones

        Parameters
        ----------
        ds_slice : tuple
            dataset slice of interest along axis 0 and 1

        Returns
        -------
        mask : ndarray
            Array of ones slices down by ds_slice
        """
        mask = np.ones(self.shape)[ds_slice]

        return mask

    def _force_include(self, mask, layers, ds_slice):
        """
        Apply force inclusion layers

        Parameters
        ----------
        mask : ndarray | None
            Mask to apply force inclusion layers to
        layers : list
            List of force inclusion layers
        ds_slice : int | slice | list | ndarray
            What to extract from ds, each arg is for a sequential axis.
            For example, (slice(0, 64), slice(0, 64)) will extract a 64x64
            exclusions mask.
        """
        for layer in layers:
            layer_slice = (layer.layer, ) + ds_slice
            layer_mask = layer[self.excl_h5[layer_slice]]
            if mask is None:
                mask = layer_mask
            else:
                mask = np.maximum(mask, layer_mask)

        return mask

    def _generate_mask(self, *ds_slice):
        """
        Generate multiplicative inclusion mask from exclusion layers.

        Parameters
        ----------
        ds_slice : int | slice | list | ndarray
            What to extract from ds, each arg is for a sequential axis.
            For example, (slice(0, 64), slice(0, 64)) will extract a 64x64
            exclusions mask.

        Returns
        -------
        mask : ndarray
            Multiplicative inclusion mask with all layers multiplied together
            ("and" operation) such that 1 is included, 0 is excluded,
            0.5 is half.
        """
        mask = None
        if len(ds_slice) == 1 & isinstance(ds_slice[0], tuple):
            ds_slice = ds_slice[0]

        if self._min_area is not None:
            ds_slice, sub_slice = self._increase_mask_slice(ds_slice, n=1)

        if self.layers:
            force_include = []
            for layer in self.layers:
                if layer.force_include:
                    force_include.append(layer)
                else:
                    layer_slice = (layer.layer, ) + ds_slice
                    layer_mask = layer[self.excl_h5[layer_slice]]
                    if mask is None:
                        mask = layer_mask
                    else:
                        mask = np.minimum(mask, layer_mask)

            mask = self._force_include(mask, force_include, ds_slice)

            if self._min_area is not None:
                mask = self._area_filter(mask, min_area=self._min_area,
                                         kernel=self._kernel)
                mask = mask[sub_slice]
        else:
            if self._min_area is not None:
                ds_slice = sub_slice

            mask = self._generate_ones_mask(ds_slice)

        return mask

    @classmethod
    def run(cls, excl_h5, layers=None, min_area=None,
            kernel='queen', hsds=False):
        """
        Create inclusion mask from given layers

        Parameters
        ----------
        excl_h5 : str
            Path to exclusions .h5 file
        layers : list | NoneType
            list of LayerMask instances for each exclusion layer to combine
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
        with cls(excl_h5, layers=layers, min_area=min_area,
                 kernel=kernel, hsds=hsds) as f:
            mask = f.mask

        return mask


class ExclusionMaskFromDict(ExclusionMask):
    """
    Class to initialize ExclusionMask from a dictionary defining layers
    """
    def __init__(self, excl_h5, layers_dict=None, min_area=None,
                 kernel='queen', hsds=False, check_layers=False):
        """
        Parameters
        ----------
        excl_h5 : str
            Path to exclusions .h5 file
        layers_dict : dict | NoneType
            Dictionary of LayerMask arugments {layer: {kwarg: value}}
        min_area : float | NoneType
            Minimum required contiguous area in sq-km
        kernel : str
            Contiguous filter method to use on final exclusion
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        check_layers : bool
            Run a pre-flight check on each layer to ensure they contain
            un-excluded values
        """
        if layers_dict is not None:
            layers = []
            for layer, kwargs in layers_dict.items():
                layers.append(LayerMask(layer, **kwargs))
        else:
            layers = None

        super().__init__(excl_h5, layers=layers, min_area=min_area,
                         kernel=kernel, hsds=hsds, check_layers=check_layers)

    @classmethod
    def run(cls, excl_h5, layers_dict=None, min_area=None,
            kernel='queen', hsds=False):
        """
        Create inclusion mask from given layers dictionary

        Parameters
        ----------
        excl_h5 : str
            Path to exclusions .h5 file
        layers_dict : dict | NoneType
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
        with cls(excl_h5, layers_dict=layers_dict, min_area=min_area,
                 kernel=kernel, hsds=hsds) as f:
            mask = f.mask

        return mask


class FrictionMask(ExclusionMask):
    """Class to handle exclusion-style friction layer."""

    def __init__(self, fric_h5, fric_dset, hsds=False, check_layers=False):
        """
        Parameters
        ----------
        fric_h5 : str
            Path to friction layer .h5 file (same format as exclusions file)
        fric_dset : str
            Friction layer dataset in fric_h5
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        check_layers : bool
            Run a pre-flight check on each layer to ensure they contain
            un-excluded values
        """
        self._fric_dset = fric_dset
        L = [LayerMask(fric_dset, use_as_weights=True, exclude_nodata=False)]
        super().__init__(fric_h5, layers=L, min_area=None, hsds=hsds,
                         check_layers=check_layers)

    def _generate_mask(self, *ds_slice):
        """
        Generate multiplicative friction layer mask.

        Parameters
        ----------
        ds_slice : int | slice | list | ndarray
            What to extract from ds, each arg is for a sequential axis.
            For example, (slice(0, 64), slice(0, 64)) will extract a 64x64
            exclusions mask.

        Returns
        -------
        mask : ndarray
            Multiplicative friction layer mask with nodata values set to 1.
        """

        mask = None
        if len(ds_slice) == 1 & isinstance(ds_slice[0], tuple):
            ds_slice = ds_slice[0]

        layer_slice = (self._layers[self._fric_dset].layer, ) + ds_slice
        mask = self._layers[self._fric_dset][self.excl_h5[layer_slice]]
        mask[(mask == self._layers[self._fric_dset].nodata_value)] = 1

        return mask

    @classmethod
    def run(cls, excl_h5, fric_dset, hsds=False):
        """
        Create inclusion mask from given layers dictionary

        Parameters
        ----------
        fric_h5 : str
            Path to friction layer .h5 file (same format as exclusions file)
        fric_dset : str
            Friction layer dataset in fric_h5
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS

        Returns
        -------
        mask : ndarray
            Full inclusion mask
        """
        L = [LayerMask(fric_dset, use_as_weights=True, exclude_nodata=False)]
        with cls(excl_h5, *L, min_area=None, hsds=hsds) as f:
            mask = f.mask

        return mask
