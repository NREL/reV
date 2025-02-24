# -*- coding: utf-8 -*-
"""
Generate reV inclusion mask from exclusion layers
"""
import logging
from warnings import warn

import numpy as np
from rex.utilities.loggers import log_mem
from scipy import ndimage

from reV.handlers.exclusions import ExclusionLayers, LATITUDE, LONGITUDE
from reV.utilities.exceptions import ExclusionLayerError, SupplyCurveInputError

logger = logging.getLogger(__name__)


class LayerMask:
    """
    Class to convert exclusion layer to inclusion layer mask
    """

    def __init__(self, layer,
                 exclude_values=None,
                 exclude_range=(None, None),
                 include_values=None,
                 include_range=(None, None),
                 include_weights=None,
                 force_include_values=None,
                 force_include_range=None,
                 use_as_weights=False,
                 weight=1.0,
                 exclude_nodata=False,
                 nodata_value=None,
                 extent=None,
                 **kwargs):
        """
        Parameters
        ----------
        layer : str
            Layer name.
        exclude_values : int | float | list, optional
            Single value or list of values to exclude.

            .. Important:: The keyword arguments `exclude_values`,
              `exclude_range`, `include_values`, `include_range`,
              `include_weights`, `force_include_values`, and
              `force_include_range` are all mutually exclusive. Users
              should supply value(s) for exactly one of these arguments.

            By default, ``None``.
        exclude_range : list | tuple, optional
            Two-item list of [min threshold, max threshold] (ends are
            inclusive) for values to exclude. Mutually exclusive
            with other inputs (see info in the description of
            `exclude_values`). By default, ``None``.
        include_values : int | float | list, optional
            Single value or list of values to include. Mutually
            exclusive with other inputs (see info in the description of
            `exclude_values`). By default, ``None``.
        include_range : list | tuple, optional
            Two-item list of [min threshold, max threshold] (ends are
            inclusive) for values to include. Mutually exclusive with
            other inputs (see info in the description of
            `exclude_values`). By default, ``None``.
        include_weights : dict, optional
            A dictionary of ``{value: weight}`` pairs, where the
            ``value`` in the layer that should be included with the
            given ``weight``. Mutually exclusive with other inputs (see
            info in the description of  `exclude_values`).
            By default, ``None``.
        force_include_values : int | float | list, optional
            Force the inclusion of the given value(s). This input
            completely replaces anything provided as `include_values`
            and is mutually exclusive with other inputs (eee info in
            the description of `exclude_values`). By default, ``None``.
        force_include_range : list | tuple, optional
            Force the inclusion of given values in the range
            [min threshold, max threshold] (ends are inclusive). This
            input completely replaces anything provided as
            `include_range` and is mutually exclusive with other inputs
            (see info in the description of `exclude_values`).
            By default, ``None``.
        use_as_weights : bool, optional
            Option to use layer as final inclusion weights (i.e.
            1 = fully included, 0.75 = 75% included, 0.5 = 50% included,
            etc.). If ``True``, all inclusion/exclusions specifications
            for the layer are ignored and the raw values (scaled by the
            `weight` input) are used as inclusion weights.
            By default, ``False``.
        weight : float, optional
            Weight applied to exclusion layer after it is calculated.
            Can be used, for example, to turn a binary exclusion layer
            (i.e. data with 0 or 1 values and ``exclude_values=1``
            input) into partial exclusions by setting the weight to
            a fraction (e.g. 0.5 for 50% exclusions). By default, ``1``.
        exclude_nodata : bool, optional
            Flag to exclude nodata values (`nodata_value`). If
            ``nodata_value=None`` the `nodata_value` is inferred by
            :class:`reV.supply_curve.exclusions.ExclusionMask`.
            By default, ``False``.
        nodata_value : int | float, optional
            Nodata value for the layer. If ``None``, the value will be
            inferred when LayerMask is added to
            :class:`reV.supply_curve.exclusions.ExclusionMask`.
            By default, ``None``.
        extent : dict, optional
            Optional dictionary with values that can be used to
            initialize this class (i.e. `layer`, `exclude_values`,
            `include_range`, etc.). This dictionary should contain the
            specifications to create a boolean mask that defines the
            extent to which the original mask should be applied.
            For example, suppose you specify the input the following
            way::

                input_dict = {
                    "viewsheds": {
                        "exclude_values": 1,
                        "extent": {
                            "layer": "federal_parks",
                            "include_range": [1, 5]
                        }
                    }
                }

                for layer_name, kwargs in input_dict.items():
                    layer = LayerMask(layer_name, **kwargs)
                    ...

            This would mean that you are masking out all viewshed layer
            values equal to 1, **but only where the "federal_parks"
            layer is equal to 1, 2, 3, 4, or 5**. Outside of these
            regions (i.e. outside of federal park regions), the viewshed
            exclusion is **NOT** applied. If the extent mask created by
            these options is not boolean, an error is thrown (i.e. do
            not specify `weight` or `use_as_weights`).
            By default ``None``, which applies the original layer mask
            to the full extent.
        **kwargs
            Optional inputs to maintain legacy kwargs of ``inclusion_*``
            instead of ``include_*``.
        """

        self._name = layer
        self._exclude_values = exclude_values
        self._exclude_range = exclude_range
        self._include_values = include_values
        self._include_range = include_range
        self._include_weights = include_weights
        self._force_include = False

        self._parse_legacy_kwargs(kwargs)

        if force_include_values is not None:
            self._include_values = force_include_values
            self._force_include = True
        if force_include_range is not None:
            self._include_range = force_include_range
            self._force_include = True

        self._as_weights = use_as_weights
        self._exclude_nodata = exclude_nodata
        self.nodata_value = nodata_value

        if weight > 1 or weight < 0:
            msg = ('Invalid weight ({}) provided for layer {}:'
                   '\nWeight must fall between 0 and 1!'.format(weight, layer))
            logger.error(msg)
            raise ValueError(msg)

        self._weight = weight
        self._mask_type = self._check_mask_type()
        self.extent = LayerMask(**extent) if extent is not None else None

    def __repr__(self):
        msg = ('{} for "{}" exclusion, of type "{}"'
               .format(self.__class__.__name__, self.name, self.mask_type))

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

    def _parse_legacy_kwargs(self, kwargs):
        """Parse legacy kwargs that start with inclusion_* instead of include_*

        Parameters
        ----------
        kwargs : dict
            Optional inputs to maintain legacy kwargs of inclusion_* instead of
            include_*
        """

        for k, v in kwargs.items():
            msg = None
            if k == 'inclusion_range':
                self._include_range = v
                msg = 'Please use "include_range" instead of "inclusion_range"'

            elif k == 'inclusion_weights':
                self._include_weights = v
                msg = ('Please use "include_weights" instead of '
                       '"inclusion_weights"')

            elif k == 'inclusion_values':
                self._include_values = v
                msg = ('Please use "include_values" instead of '
                       '"inclusion_values"')

            if msg is not None:
                warn(msg)
                logger.warning(msg)

    @property
    def name(self):
        """
        Layer name to extract from exclusions .h5 file

        Returns
        -------
        _name : str
        """
        return self._name

    @property
    def min_value(self):
        """Minimum value to include/exclude if include_range or exclude_range
        was input.

        Returns
        -------
        float
        """
        if 'excl' in self.mask_type:
            range_var = self._exclude_range
        else:
            range_var = self._include_range

        if all(isinstance(x, (int, float)) for x in range_var):
            return min(range_var)
        return range_var[0]

    @property
    def max_value(self):
        """Maximum value to include/exclude if include_range or exclude_range
        was input.

        Returns
        -------
        float
        """
        if 'excl' in self.mask_type:
            range_var = self._exclude_range
        else:
            range_var = self._include_range

        if all(isinstance(x, (int, float)) for x in range_var):
            return max(range_var)
        return range_var[1]

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
    def include_weights(self):
        """
        Mapping of values to include and at what weights

        Returns
        -------
        dict
        """
        return self._include_weights

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
            if self.mask_type == 'include_range':
                func = self._include_range_mask
            elif self.mask_type == 'exclude_range':
                func = self._exclude_range_mask
            elif self.mask_type == 'exclude':
                func = self._exclusion_mask
            elif self.mask_type == 'include':
                func = self._inclusion_mask
            elif self.mask_type == 'include_weights':
                func = self._weights_mask
            else:
                msg = ('{} is an invalid mask type: expecting '
                       '"include_range", "exclude_range", "exclude", '
                       '"include", or "include_weights"'
                       .format(self.mask_type))
                logger.error(msg)
                raise KeyError(msg)

            data = func(data)

        data = data.astype('float32') * self._weight

        return data

    def _check_mask_type(self):
        """
        Ensure that the initialization arguments are valid and not
        contradictory

        Returns
        -------
        mask : str
            Mask type
        """
        mask = None
        if not self._as_weights:
            masks = {'include_range': any(i is not None
                                          for i in self._include_range),
                     'exclude_range': any(i is not None
                                          for i in self._exclude_range),
                     'exclude': self._exclude_values is not None,
                     'include': self._include_values is not None,
                     'include_weights': self._include_weights is not None}
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

            if mask is None:
                msg = ('Exactly one approach must be specified to create the '
                       'inclusion mask for layer {!r}! Please specify one of: '
                       '`exclude_values`, `exclude_range`, `include_values`, '
                       '`include_range`, `include_weights`, '
                       '`force_include_values`, or `force_include_range`.'
                       .format(self.name))
                logger.error(msg)
                raise ExclusionLayerError(msg)

        if mask == 'include_weights' and self._weight < 1:
            msg = ("Values are individually weighted when using "
                   "'include_weights', the supplied weight of {} will be "
                   "ignored!".format(self._weight))
            self._weight = 1
            logger.warning(msg)
            warn(msg)

        return mask

    def _exclude_range_mask(self, data):
        """
        Mask exclusion layer based on exclude value range

        Parameters
        ----------
        data : ndarray
            Exclusions data to create mask from

        Returns
        -------
        mask : ndarray
            Boolean mask of which values to include (True is include).
        """
        mask = np.full(data.shape, False)
        if self.min_value is not None:
            mask = data < self.min_value

        if self.max_value is not None:
            mask |= data > self.max_value

        mask[data == self.nodata_value] = True
        if self._exclude_nodata:
            mask = mask & (data != self.nodata_value)

        return mask

    def _include_range_mask(self, data):
        """
        Mask exclusion layer based on include value range

        Parameters
        ----------
        data : ndarray
            Exclusions data to create mask from

        Returns
        -------
        mask : ndarray
            Boolean mask of which values to include (True is include).
        """
        mask = np.full(data.shape, True)
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
        for value, weight in self.include_weights.items():
            if isinstance(value, str):
                value = float(value)

            weight = np.array([weight], dtype='float32')

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
        excl_h5 : str | list | tuple
            Path to one or more exclusions .h5 files
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

            missing = [layer.name for layer in layers
                       if layer.name not in self.excl_layers]
            if any(missing):
                msg = ("ExclusionMask layers {} are missing from: {}"
                       .format(missing, self._excl_h5))
                logger.error(msg)
                raise KeyError(msg)

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
        return self.excl_h5[LATITUDE]

    @property
    def longitude(self):
        """
        Longitude coordinates array

        Returns
        -------
        ndarray
        """
        return self.excl_h5[LONGITUDE]

    def add_layer(self, layer, replace=False):
        """
        Add layer to be combined

        Parameters
        ----------
        layer : LayerMask
            LayerMask instance to add to set of layers to be combined
        """

        if layer.name not in self.excl_layers:
            msg = "{} does not exist in {}".format(layer.name, self._excl_h5)
            logger.error(msg)
            raise KeyError(msg)

        if layer.name in self.layer_names:
            msg = "{} is already in {}".format(layer.name, self)
            if replace:
                msg += " replacing existing layer"
                logger.warning(msg)
                warn(msg)
            else:
                logger.error(msg)
                raise ExclusionLayerError(msg)

        layer.nodata_value = self.excl_h5.get_nodata_value(layer.name)
        if self._check_layers:
            if not layer[self.excl_h5[layer.name]].any():
                msg = ("Layer {} is fully excluded!".format(layer.name))
                logger.error(msg)
                raise ExclusionLayerError(msg)

        self._layers[layer.name] = layer

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

    @classmethod
    def _area_filter(cls, mask, min_area, excl_area, kernel='queen'):
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
        s = cls.FILTER_KERNELS[kernel]
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

        ones_shape = ()
        for i, s in enumerate(self.shape):
            if i < len(ds_slice):
                ax_slice = ds_slice[i]
                if np.issubdtype(type(ax_slice), np.integer):
                    ones_shape += (ax_slice,)
                else:
                    ax = np.arange(s, dtype=np.int32)
                    ones_shape += (len(ax[ax_slice]), )
            else:
                ones_shape += (s, )

        mask = np.ones(ones_shape, dtype='float32')

        return mask

    def _add_layer_to_mask(self, mask, layer, ds_slice, check_layers,
                           combine_func):
        """Add layer mask to full mask."""
        layer_mask = self._compute_layer_mask(layer, ds_slice, check_layers)
        if mask is None:
            return layer_mask

        return combine_func(mask, layer_mask, dtype='float32')

    def _compute_layer_mask(self, layer, ds_slice, check_layers=False):
        """Compute mask for single layer, including extent."""
        layer_mask = self._masked_layer_data(layer, ds_slice)
        layer_mask = self._apply_layer_mask_extent(layer, layer_mask, ds_slice)

        logger.debug('Computed exclusions {} for {}. Layer has average value '
                     'of {:.2f}.'
                     .format(layer, ds_slice, layer_mask.mean()))
        log_mem(logger, log_level='DEBUG')

        if check_layers and not layer_mask.any():
            msg = "Layer {} is fully excluded!".format(layer.name)
            logger.error(msg)
            raise ExclusionLayerError(msg)

        return layer_mask

    def _apply_layer_mask_extent(self, layer, layer_mask, ds_slice):
        """Apply extent to layer mask, if any."""
        if layer.extent is None:
            return layer_mask

        layer_extent = self._masked_layer_data(layer.extent, ds_slice)
        if not np.array_equal(layer_extent, layer_extent.astype(bool)):
            msg = ("Extent layer must be boolean (i.e. 0 and 1 values "
                   "only)! Please check your extent definition for layer "
                   "{} to ensure you are producing a boolean layer!"
                   .format(layer.name))
            logger.error(msg)
            raise ExclusionLayerError(msg)

        logger.debug("Filtering mask for layer %s down to specified extent",
                     layer.name)
        layer_mask = np.where(layer_extent, layer_mask, 1)
        return layer_mask

    def _masked_layer_data(self, layer, ds_slice):
        """Extract masked data for layer."""
        return layer[self.excl_h5[(layer.name, ) + ds_slice]]

    def _generate_mask(self, *ds_slice, check_layers=False):
        """
        Generate multiplicative inclusion mask from exclusion layers.

        Parameters
        ----------
        ds_slice : int | slice | list | ndarray
            What to extract from ds, each arg is for a sequential axis.
            For example, (slice(0, 64), slice(0, 64)) will extract a 64x64
            exclusions mask.
        check_layers : bool
            Check each layer as each layer is extracted to ensure they contain
            un-excluded values. This should only really be True if ds_slice is
            for the full inclusion mask. Otherwise, this could raise an error
            for a fully excluded mask for just one excluded SC point.

        Returns
        -------
        mask : ndarray
            Multiplicative inclusion mask with all layers multiplied together
            ("and" operation) such that 1 is included, 0 is excluded,
            0.5 is half.
        """

        mask = None
        ds_slice, sub_slice = self._parse_ds_slice(ds_slice)

        if self.layers:
            force_include = []
            for layer in self.layers:
                if layer.force_include:
                    force_include.append(layer)
                else:
                    mask = self._add_layer_to_mask(mask, layer, ds_slice,
                                                   check_layers,
                                                   combine_func=np.minimum)
            for layer in force_include:
                mask = self._add_layer_to_mask(mask, layer, ds_slice,
                                               check_layers,
                                               combine_func=np.maximum)

            if self._min_area is not None:
                mask = self._area_filter(mask, self._min_area,
                                         self._excl_h5.pixel_area,
                                         kernel=self._kernel)
                mask = mask[sub_slice]
        else:
            if self._min_area is not None:
                ds_slice = sub_slice

            mask = self._generate_ones_mask(ds_slice)

        return mask

    def _parse_ds_slice(self, ds_slice):
        """Parse a dataset slice to make it the proper dimensions and also
        optionally increase the dataset slice to make the contiguous area
        filter more accurate

        Parameters
        ----------
        ds_slice : int | slice | list | ndarray
            What to extract from ds, each arg is for a sequential axis.
            For example, (slice(0, 64), slice(0, 64)) will extract a 64x64
            exclusions mask.

        Returns
        -------
        ds_slice : tuple
            Two entry tuple with x and y slices with increased dimensions.
        sub_slice : tuple
            Two entry tuple with x and y slices to retrieve the original
            slice out of the bigger slice.
        """

        if len(ds_slice) == 1 & isinstance(ds_slice[0], tuple):
            ds_slice = ds_slice[0]

        sub_slice = None
        if self._min_area is not None:
            ds_slice, sub_slice = self._increase_mask_slice(ds_slice, n=1)

        return ds_slice, sub_slice

    @classmethod
    def run(cls, excl_h5, layers=None, min_area=None,
            kernel='queen', hsds=False):
        """
        Create inclusion mask from given layers

        Parameters
        ----------
        excl_h5 : str | list | tuple
            Path to one or more exclusions .h5 files
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
        excl_h5 : str | list | tuple
            Path to one or more exclusions .h5 files
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
    def extract_inclusion_mask(cls, excl_fpath, tm_dset, excl_dict=None,
                               area_filter_kernel='queen', min_area=None):
        """
        Extract the full inclusion mask from excl_fpath using the given
        exclusion layers and whether or not to run a minimum area filter

        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        excl_dict : dict | None
            Dictionary of exclusion keyword arugments of the format
            {layer_dset_name: {kwarg: value}} where layer_dset_name is a
            dataset in the exclusion h5 file and kwarg is a keyword argument to
            the reV.supply_curve.exclusions.LayerMask class.
        area_filter_kernel : str, optional
            Contiguous area filter method to use on final exclusions mask,
            by default "queen"
        min_area : float, optional
            Minimum required contiguous area filter in sq-km,
            by default None

        Returns
        -------
        inclusion_mask : ndarray
            Pre-computed 2D inclusion mask (normalized with expected range:
            [0, 1], where 1 is included and 0 is excluded)
        """
        logger.info('Pre-extracting full exclusion mask, this could take '
                    'up to 30min for a large exclusion config...')
        with cls(excl_fpath, layers_dict=excl_dict, check_layers=False,
                 min_area=min_area, kernel=area_filter_kernel) as f:
            inclusion_mask = f._generate_mask(..., check_layers=True)
            tm_mask = f._excl_h5[tm_dset] == -1
            inclusion_mask[tm_mask] = 0

        logger.info('Finished extracting full exclusion mask.')
        logger.info('The full exclusion mask has {:.2f}% of area included.'
                    .format(100 * inclusion_mask.sum()
                            / inclusion_mask.size))

        if inclusion_mask.sum() == 0:
            msg = 'The exclusions inputs resulted in a fully excluded mask!'
            logger.error(msg)
            raise SupplyCurveInputError(msg)

        return inclusion_mask

    @classmethod
    def run(cls, excl_h5, layers_dict=None, min_area=None,
            kernel='queen', hsds=False):
        """
        Create inclusion mask from given layers dictionary

        Parameters
        ----------
        excl_h5 : str | list | tuple
            Path to one or more exclusions .h5 files
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

        layer_slice = (self._layers[self._fric_dset].name, ) + ds_slice
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
