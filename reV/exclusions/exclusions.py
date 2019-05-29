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
    exclusions.export(fname='exclusions.tif')

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
    exclusions.export(fname='exclusions.tif')

    --- OR ---

    Exclusions.run(config = [{"fpath": "ri_srtm_slope.tif",
                              "max_thresh": 5,
                             },
                             {"fpath": "ri_padus.tif",
                              "classes_exclude": [1],
                             }],
                   use_blocks = True,
                   contiguous_filter = 'queen',
                   output_fname = 'exclusions.tif')

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
        self.fpath = fpath
        self.band = band
        self.name = layer_name
        self.type = layer_type
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.classes_include = classes_include
        self.classes_exclude = classes_exclude

        meta_data = self.extract_tiff_meta(self.fpath, self.band)
        self.profile = meta_data[0]
        self.block_windows = meta_data[1]
        self.num_windows = meta_data[2]

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
        profile : rasterio.profiles.Profile
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

    def read(self):
        """ Read in entire dataset from Tiff

        Outputs
        -------
        data : numpy.ndarray
            Layer's entire data array
        """

        with rasterio.open(self.fpath, 'r') as file:
            data = file.read(self.band)
            return data

    def read_window(self, window_index):
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

        _, window = self.block_windows[window_index]
        col_slice = slice(window.col_off, window.col_off + window.width)
        row_slice = slice(window.row_off, window.row_off + window.height)
        window_slice = (row_slice, col_slice)
        with rasterio.open(self.fpath, 'r') as file:
            data = file.read(self.band, window=window)
        return data, window_slice


class Exclusions:
    """Base class for single output exclusion layer"""

    contiguous_filter_kernels = {
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
        layer_configs : dictionary list | None
            Optional configs list for the addition of layers
        use_blocks : boolean
            Use blocks when applying layers to exclusions
        contiguous_filter : str | None
            Contiguous filter method to use on final exclusion
        """
        self.layers = []
        self.data = None
        self.profile = None
        # validate and set use_blocks argument
        if isinstance(use_blocks, bool):
            self.use_blocks = use_blocks
        else:
            raise TypeError('use_blocks argument must be a boolean')
        # validate and set contiguous filter argument
        if contiguous_filter in [None, "queen", "rook"]:
            self.contiguous_filter = contiguous_filter
        else:
            raise TypeError('contiguous_filter must be "queen" or "rook"')
        # validate and set layer_configs argument
        if isinstance(layer_configs, (list, type(None))):
            self.layer_configs = layer_configs
        else:
            raise TypeError('layer_configs argument must be a list or None')

    def build_from_config(self, layer_configs=None, contiguous_filter=None):
        """ Build and apply exclusion layers from config if it exists

        Parameters
        ----------
        layer_configs : dictionary list | None
            Configs list for the addition of layers
            Provided as either an argument or predefined instance attribute
        """

        if not layer_configs:
            layer_configs = self.layer_configs
        if not contiguous_filter:
            contiguous_filter = self.contiguous_filter
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
            raise AttributeError('Object has no configs: self.layer_configs')

    def add_layer(self, layer):
        """ Append a layer object to the list of exclusion layers """

        if isinstance(layer, ExclusionLayer):
            self.layers.append(layer)
            if len(self.layers) == 1:
                self.create_profile()
            self.check_layer_compatibility()
        else:
            logger.warning('Layers must be an instance of ExclusionLayer')
        return self.layers

    def check_layer_compatibility(self):
        """ Validate that all layers are the same shape

        Outputs
        -------
        compatibility : boolean
            Validates whether all layers are compatible with eachother
        """

        compatibility = True
        widths = []
        heights = []
        for layer in self.layers:
            widths.append(layer.profile['width'])
            heights.append(layer.profile['height'])
        if len(set(widths)) > 1:
            compatibility = False
        if len(set(heights)) > 1:
            compatibility = False

        if not compatibility:
            logger.warning('Layers are not compatible')
        return compatibility

    def create_profile(self):
        """ Create the Tiff profile meta for output file from first layer """

        try:
            # Start from the profile of the first layer
            self.profile = self.layers[0].profile
            # Modifications for output
            self.profile['dtype'] = rasterio.uint8
            self.profile['nodata'] = None
            self.initialize_data()
        except IndexError:
            raise AttributeError('Exclusion has no layers '
                                 '(i.e. self.add_layer(layer))')

    def initialize_data(self):
        """ Initiate the array for the output exclusion data """

        if self.profile:
            shape = (self.profile['height'], self.profile['width'])
            self.data = np.ones(shape=shape, dtype='uint8') * 100
        else:
            raise AttributeError('Profile has not been created yet '
                                 '(i.e. self.create_profile())')

    def get_method_mask(self, data, layer):
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
        if layer.classes_exclude:
            mask = self.mask_classes(data,
                                     layer.classes_exclude,
                                     True)
        elif layer.classes_include:
            mask = self.mask_classes(data,
                                     layer.classes_include,
                                     False)
        elif layer.min_thresh or layer.max_thresh:
            mask = self.mask_interval(data,
                                      layer.min_thresh,
                                      layer.max_thresh)
        return mask

    def apply_layer(self, layer):
        """ Read, process, and apply an input layer to the
        final output layer

        Parameters
        ----------
        layer : reV.exclusions.exclusions.ExclusionLayer
            exclusion layer object
        """

        if self.use_blocks:
            for window_index in range(layer.num_windows):
                layer_data, window_slice = layer.read_window(window_index)
                mask = self.get_method_mask(layer_data, layer)
                if mask is not None:
                    block_data = (self.data[window_slice] * mask)
                    self.data[window_slice] = block_data.astype('uint8')
                else:
                    logger.warning('Failed to mask {}'.format(layer.name))
        else:
            layer_data = layer.read()
            mask = self.get_method_mask(layer_data, layer)
            if mask is not None:
                self.data = (self.data * mask).astype('uint8')
            else:
                logger.warning('Failed to mask {}'.format(layer.name))

    def apply_all_layers(self):
        """ Read, process, and apply all input layers to the
        final output layer """

        for layer in self.layers:
            self.apply_layer(layer)

    @staticmethod
    def mask_interval(data, min_thresh, max_thresh):
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
    def mask_classes(data, classes, exclude=True):
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
            contiguous_filter = self.contiguous_filter
        if contiguous_filter not in [None, "queen", "rook"]:
            raise TypeError('contiguous_filter must be "queen" or "rook"')
        if isinstance(self.data, type(None)):
            raise AttributeError('Exclusion has not been created yet'
                                 '(i.e. self.apply_layer())')
        if isinstance(contiguous_filter, type(None)):
            logger.info('No contiguous filter provided')
            return None
        mask = (self.data != 0).astype('int8')
        kernel = self.contiguous_filter_kernels[contiguous_filter]
        mask = ndimage.convolve(mask, kernel, mode='constant', cval=0.0)
        self.data[mask == 0] = 0
        return None

    def export(self, fname='exclusions.tif', band=1):
        """ Save the output exclusion layer as a Tiff

        Parameters
        ----------
        fname : str
            Output file with path.
        band : integer
            GeoTiff band to write to file.
        """

        if self.profile:
            with rasterio.open(fname, 'w', **self.profile) as file:
                file.write(self.data, band)
        else:
            raise AttributeError('Profile has not been created yet '
                                 '(i.e. self.create_profile())')

    @classmethod
    def run(cls, config, output_fname,
            use_blocks=False, contiguous_filter=None):
        """ Apply Exclusions from config and save to disc.

        Parameters
        ----------
        layer_configs : dictionary list | None
            Optional configs list for the addition of layers
        output_fname : str
            Output file with path.
        use_blocks : boolean
            Use blocks when applying layers to exclusions
        contiguous_filter : str | None
            Contiguous filter method to use on final exclusion
        """

        exclusions = cls(config, use_blocks=use_blocks,
                         contiguous_filter=contiguous_filter)
        exclusions.build_from_config()
        exclusions.export(fname=output_fname)
        return None
