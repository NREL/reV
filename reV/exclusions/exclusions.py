"""
Basic reV Exclusions
"""
import logging
import rasterio
import numpy as np

logger = logging.getLogger(__name__)


class ExclusionLayer:
    """Base class for all input exclusion layers"""

    def __init__(self, fname, options):
        self.input_file = fname
        self.band = options.get("band", 1)
        self.min_thresh = options.get("min_thresh", None)
        self.max_thresh = options.get("max_thresh", None)
        self.classes_include = options.get("classes_include", None)
        self.classes_exclude = options.get("classes_exclude", None)

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
        with rasterio.open(self.input_file, 'r') as file:
            data = file.read(self.band, window=window)
            return data


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
        for fname in list(self.config.keys()):
            options = self.config[fname]
            exclusion = ExclusionLayer(fname, options)
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

        mask = None

        if use_blocks:
            for window_index in range(layer.num_windows):
                data = layer.read_window(window_index)
                # TODO
                return None
        else:
            data = layer.read()
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

        if mask is not None:
            self.data *= mask
        else:
            logger.warning('Layer did not have any exclusions configured.')

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
