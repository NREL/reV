"""
Basic reV Exclusions

Sample Usage:

    exclusion = Exclusions([{"fpath": "ri_srtm_slope.tif",
                             "name": "SRTM Slope",
                             "type": "Slope",
                             "band": 1,
                             "max_thresh": 5,
                            },
                            {"fpath": "ri_padus.tif",
                             "name": "PAD-US",
                             "type": "Protected",
                             "band": 1,
                             "classes_exclude": [1],
                            }])

    exclusion.apply_all_layers()
    exclusion.export(fname='exclusions.tif')

"""
import logging
import rasterio
import numpy as np

logger = logging.getLogger(__name__)


class ExclusionLayer:
    """Base class for all input exclusion layers"""

    def __init__(self, config):
        self.input_file = config.get('fpath', None)
        self.name = config.get("name", None)
        self.type = config.get("type", None)
        self.band = config.get("band", 1)
        self.min_thresh = config.get("min_thresh", None)
        self.max_thresh = config.get("max_thresh", None)
        self.classes_include = config.get("classes_include", None)
        self.classes_exclude = config.get("classes_exclude", None)

        with rasterio.open(self.input_file, 'r') as file:
            self.profile = file.profile
            self.block_windows = list(file.block_windows(self.band))
            self.num_windows = len(self.block_windows)

    def read(self):
        """ Read in entire dataset from Tiff """

        with rasterio.open(self.input_file, 'r') as file:
            data = file.read(self.band)
            return data

    def read_window(self, window_index):
        """ Read in a single block dataset from Tiff

        Parameters
        ----------
        window_index : integer
            list index of window
        """

        _, window = self.block_windows[window_index]
        col_slice = slice(window.col_off, window.col_off + window.width)
        row_slice = slice(window.row_off, window.row_off + window.height)
        window_slice = (row_slice, col_slice)
        with rasterio.open(self.input_file, 'r') as file:
            data = file.read(self.band, window=window)
        return data, window_slice


class Exclusions:
    """Base class for single output exclusion layer"""

    def __init__(self, exclusions_config):
        self.config = exclusions_config
        self.add_layers()
        self.check_layer_compatibility()
        self.create_profile()
        self.initialize_data()

    def add_layers(self):
        """ Generate list of layer objects from the config """

        layers = []
        for layer_config in self.config:
            exclusion = ExclusionLayer(layer_config)
            layers.append(exclusion)
        if layers:
            self.layers = layers
        else:
            logger.warning('There are no configured exclusion '
                           'layers to process')

    def check_layer_compatibility(self):
        """ Validate that all layers are the same shape """

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

        if compatibility:
            return True
        else:
            logger.error('Layers are not compatible')

    def create_profile(self):
        """ Create the Tiff profile meta for output file """

        try:
            # Start from the profile of a layer
            self.profile = self.layers[0].profile
            # Modifications for output
            self.profile['dtype'] = rasterio.float32
            self.profile['nodata'] = None
        except AttributeError:
            raise AttributeError('Exclusion has no layers '
                                 '(i.e. self.add_layers())')

    def initialize_data(self):
        """ Initiate the array for the output exclusion data """

        try:
            shape = (self.profile['height'], self.profile['width'])
            self.data = np.ones(shape=shape).astype(np.float32)
        except AttributeError:
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

    def apply_layer(self, layer, use_blocks=False):
        """ Read, process, and apply an input layer to the
        final output layer

        Parameters
        ----------
        layer : reV.exclusions.exclusions.ExclusionLayer
            exclusion layer object
        use_blocks : boolean
            for large Tiffs, read dataset in by blocks
        """

        if use_blocks:
            for window_index in range(layer.num_windows):
                data, window_slice = layer.read_window(window_index)
                mask = self.get_method_mask(data, layer)
                if mask is not None:
                    self.data[window_slice] = self.data[window_slice] * mask
                else:
                    logger.warning('Failed to mask {}'.format(layer.name))
        else:
            data = layer.read()
            mask = self.get_method_mask(data, layer)
            if mask is not None:
                self.data *= mask
            else:
                logger.warning('Failed to mask {}'.format(layer.name))

    def apply_all_layers(self, use_blocks=False):
        """ Read, process, and apply all input layers to the
        final output layer

        Parameters
        ----------
        use_blocks : boolean
            for large Tiffs, read dataset in by blocks
        """

        for layer in self.layers:
            self.apply_layer(layer, use_blocks)

    @staticmethod
    def mask_interval(data, min_thresh, max_thresh):
        """ Provide a mask between a range of thresholds """

        mask = np.ones(shape=data.shape)
        if min_thresh:
            mask *= (data >= min_thresh).astype(int)
        if max_thresh:
            mask *= (data <= max_thresh).astype(int)
        return mask

    @staticmethod
    def mask_classes(data, classes, exclude=True):
        """ Provide a mask from integer classes """

        mask = np.isin(data, classes)
        if exclude:
            mask = ~mask
        return mask.astype(int)

    def export(self, fname='exclusions.tif', band=1):
        """ Save the output exclusion layer as a Tiff """

        try:
            with rasterio.open(fname, 'w', **self.profile) as file:
                file.write(self.data, band)
        except AttributeError:
            raise AttributeError('Profile has not been created yet '
                                 '(i.e. self.create_profile())')
