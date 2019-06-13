"""
Basic reV Exclusions

Sample Usage:

    exclusions = Exclusions()

    layer = ExclusionLayer("ri_srtm_slope.tif", max_thresh=5)
    exclusions.add_layer(layer)

    layer = ExclusionLayer("ri_padus.tif", classes_exclude=[1])
    exclusions.add_layer(layer)

    exclusions.apply_all_layers()
    exclusions.apply_filter('queen')
    exclusions.export(fpath='exclusions.tif')

    --- OR ---

    exclusions = Exclusions([{"fpath": "ri_srtm_slope.tif",
                             "max_thresh": 5,
                            },
                            {"fpath": "ri_padus.tif",
                             "classes_exclude": [1],
                            }],
                            use_blocks = True,
                            contiguous_filter = 'queen')
    exclusions.build_from_config()
    exclusions.export(fpath='exclusions.tif')

    --- OR ---

    Exclusions.run(config = [{"fpath": "ri_srtm_slope.tif",
                              "max_thresh": 5,
                             },
                             {"fpath": "ri_padus.tif",
                              "classes_exclude": [1],
                             }],
                   use_blocks = True,
                   contiguous_filter = 'queen',
                   output_fpath = 'exclusions.tif')

"""
import logging
import rasterio
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class ExclusionLayer:
    """Base class for all input exclusion layers"""

    def __init__(self, fpath, band=1, layer_name=None, layer_type=None,
                 min_thresh=None, max_thresh=None,
                 classes_exclude=None, classes_include=None):
        """
        Parameters
        ----------
        fpath : str
            Layer file with path.
        band : integer
            GeoTiff band to read from file.
        layer_name : str | None
            Name of the exclusion layer (optional).
        layer_type : str | None
            Type of exclusion layer (optional).
        min_thresh : float | int | None
            Create exclusion based on a minimum value.
            Conflicts with classes_include and classes_exclude.
        max_thresh : float | int | None
            Create exclusion based on a maximum value.
            Conflicts with classes_include and classes_exclude.
        classes_exclude : integer list | None
            Create exclusion based on discrete classes to exclude.
            Conflicts with min_thresh, max_thresh, and classes_include.
        classes_include : integer list | None
            Create exclusion based on discrete classes to include.
            Conflicts with min_thresh, max_thresh, and classes_exclude.
        """
        self._fpath = fpath
        self._band = band
        self.name = layer_name
        self._type = layer_type
        self._min_thresh = min_thresh
        self._max_thresh = max_thresh
        self._classes_include = classes_include
        self._classes_exclude = classes_exclude

        meta_data = self.extract_tiff_meta(self._fpath, self._band)
        self._profile = meta_data[0]
        self._block_windows = meta_data[1]
        self._num_windows = meta_data[2]

    @property
    def fpath(self):
        """ Get the file path for layer """
        return self._fpath

    @property
    def band(self):
        """ Get the file band for layer """
        return self._band

    @property
    def type(self):
        """ Get the layer type """
        return self._type

    @property
    def min_thresh(self):
        """ Get the minimumum threshold for layer """
        return self._min_thresh

    @property
    def max_thresh(self):
        """ Get the maximum threshold for layer """
        return self._max_thresh

    @property
    def classes_include(self):
        """ Get the classes to include for layer """
        return self._classes_include

    @property
    def classes_exclude(self):
        """ Get the classes to exclude for layer """
        return self._classes_exclude

    @staticmethod
    def extract_tiff_meta(fpath, band):
        """ Read the meta data from a tiff file

        Parameters
        ----------
        fpath : str
            Layer file with path.
        band : integer
            GeoTiff band to read from file.

        Outputs
        -------
        profile : rasterio._profiles._Profile
            Meta data for the tiff file
        block_windows : tuples list
            Delineation of tiff windows
        num_windows : int
            The number of windows
        """
        with rasterio.open(fpath, 'r') as file:
            profile = file.profile
            block_windows = list(file.block_windows(band))
            num_windows = len(block_windows)
        return profile, block_windows, num_windows

    def _read(self):
        """ Read in entire dataset from Tiff

        Outputs
        -------
        data : numpy.ndarray
            Layer's entire data array
        """

        with rasterio.open(self._fpath, 'r') as file:
            data = file.read(self._band)
            return data

    def _read_window(self, window_index):
        """ Read in a single block dataset from Tiff

        Parameters
        ----------
        window_index : integer
            list index of window

        Outputs
        -------
        data : numpy.ndarray
            Layer's data array for the current layer window
        window_slice : tuple
            The tuple that slices out the current window
        """

        _, window = self._block_windows[window_index]
        col_slice = slice(window.col_off, window.col_off + window.width)
        row_slice = slice(window.row_off, window.row_off + window.height)
        window_slice = (row_slice, col_slice)
        with rasterio.open(self._fpath, 'r') as file:
            data = file.read(self._band, window=window)
        return data, window_slice


class Exclusions:
    """Base class for single output exclusion layer"""

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

    def build_from_config(self, layer_configs=None, contiguous_filter=None):
        """ Build and apply exclusion layers from config if it exists

        Parameters
        ----------
        layer_configs : dictionary | list | None
            Configs list for the addition of layers
            Provided as either an argument or predefined instance attribute
        """

        if not layer_configs:
            layer_configs = self._layer_configs
        if not contiguous_filter:
            contiguous_filter = self._contiguous_filter
        if layer_configs:
            for config in layer_configs:
                if isinstance(config, dict):
                    if 'fpath' not in config:
                        raise KeyError('fpath key is not defined in config')
                    layer = ExclusionLayer(**config)
                    self.add_layer(layer)
                else:
                    raise TypeError('Layer config must be a dictionary')
            self.apply_all_layers()
            if contiguous_filter is not None:
                self.apply_filter(contiguous_filter)
        else:
            raise AttributeError('Object has no configs: self._layer_configs')

    def add_layer(self, layer):
        """ Append a layer object to the list of exclusion layers """

        if isinstance(layer, ExclusionLayer):
            self._layers.append(layer)
            if len(self._layers) == 1:
                self._create_profile()
            self._check_layer_compatibility()
        else:
            logger.warning('Layers must be an instance of ExclusionLayer')
        return self._layers

    def _check_layer_compatibility(self):
        """ Validate that all layers are the same shape

        Outputs
        -------
        compatibility : boolean
            Validates whether all layers are compatible with eachother
        """

        compatibility = True
        widths = []
        heights = []
        for layer in self._layers:
            widths.append(layer._profile['width'])
            heights.append(layer._profile['height'])
        if len(set(widths)) > 1:
            compatibility = False
        if len(set(heights)) > 1:
            compatibility = False

        if not compatibility:
            logger.warning('Layers are not compatible')
        return compatibility

    def _create_profile(self):
        """ Create the Tiff profile meta for output file from first layer """

        try:
            # Start from the profile of the first layer
            self._profile = self._layers[0]._profile
            # Modifications for output
            self._profile['dtype'] = rasterio.uint8
            self._profile['nodata'] = None
            self._initialize_data()
        except IndexError:
            raise AttributeError('Exclusion has no layers '
                                 '(i.e. self.add_layer(layer))')

    def _initialize_data(self):
        """ Initiate the array for the output exclusion data """

        if self._profile:
            shape = (self._profile['height'], self._profile['width'])
            self._data = np.ones(shape=shape, dtype='uint8') * 100
        else:
            raise AttributeError('Profile has not been created yet '
                                 '(i.e. self._create_profile())')

    def _get_method_mask(self, data, layer):
        """ Generate a mask based on config method specified

        Parameters
        ----------
        data : np.array
            data to be masked
        layer : reV.exclusions.exclusions.ExclusionLayer
            exclusion layer object

        Outputs
        -------
        mask : numpy.ndarray
            float mask (0 to 1) to be applied to the final exclusion layer
        """
        mask = None
        if layer._classes_exclude:
            mask = self._mask_classes(data,
                                      layer._classes_exclude,
                                      True)
        elif layer._classes_include:
            mask = self._mask_classes(data,
                                      layer._classes_include,
                                      False)
        elif layer._min_thresh or layer._max_thresh:
            mask = self._mask_interval(data,
                                       layer._min_thresh,
                                       layer._max_thresh)
        return mask

    def apply_layer(self, layer):
        """ Read, process, and apply an input layer to the
        final output layer

        Parameters
        ----------
        layer : reV.exclusions.exclusions.ExclusionLayer
            exclusion layer object
        """

        if self._use_blocks:
            for window_index in range(layer._num_windows):
                layer_data, window_slice = layer._read_window(window_index)
                mask = self._get_method_mask(layer_data, layer)
                if mask is not None:
                    block_data = (self._data[window_slice] * mask)
                    self._data[window_slice] = block_data.astype('uint8')
                else:
                    logger.warning('Failed to mask {}'.format(layer.name))
        else:
            layer_data = layer._read()
            mask = self._get_method_mask(layer_data, layer)
            if mask is not None:
                self._data = (self._data * mask).astype('uint8')
            else:
                logger.warning('Failed to mask {}'.format(layer.name))

    def apply_all_layers(self):
        """ Read, process, and apply all input layers to the
        final output layer """

        for layer in self._layers:
            self.apply_layer(layer)

    @staticmethod
    def _mask_interval(data, min_thresh, max_thresh):
        """ Provide a mask between a range of thresholds

        Parameters
        ----------
        min_thresh : float | int | None
            Create exclusion based on a minimum value.
        max_thresh : float | int | None
            Create exclusion based on a maximum value.
        """

        mask = np.ones(shape=data.shape)
        if min_thresh:
            mask *= (data >= min_thresh).astype('int8')
        if max_thresh:
            mask *= (data <= max_thresh).astype('int8')
        return mask

    @staticmethod
    def _mask_classes(data, classes, exclude=True):
        """ Provide a mask from integer classes

        Parameters
        ----------
        classes : integer list | None
            Create exclusion based on discrete classes to include/exclude.
        exclude : boolean
            Use classes as inclusions or exclusions.
        """

        mask = np.isin(data, classes)
        if exclude:
            mask = ~mask
        return mask.astype('int8')

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
        self._data[mask == 0] = 0
        return None

    def export(self, fpath='exclusions.tif', band=1):
        """ Save the output exclusion layer as a Tiff

        Parameters
        ----------
        fpath : str
            Output file with path.
        band : integer
            GeoTiff band to write to file.
        """

        if self._profile:
            with rasterio.open(fpath, 'w', **self._profile) as file:
                file.write(self._data, band)
        else:
            raise AttributeError('Profile has not been created yet '
                                 '(i.e. self._create_profile())')

    @classmethod
    def run(cls, config, output_fpath,
            use_blocks=False, contiguous_filter=None):
        """ Apply Exclusions from config and save to disc.

        Parameters
        ----------
        config : dictionary | list | None
            Optional configs list for the addition of layers
        output_fpath : str
            Output file with path.
        use_blocks : boolean
            Use blocks when applying layers to exclusions
        contiguous_filter : str | None
            Contiguous filter method to use on final exclusion
        """

        exclusions = cls(config, use_blocks=use_blocks,
                         contiguous_filter=contiguous_filter)
        exclusions.build_from_config()
        exclusions.export(fpath=output_fpath)
        return None
