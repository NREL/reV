"""
Basic reV Exclusions

Sample Usage:

    exclusions = Exclusions()

    layer = ExclusionLayer("ri_srtm_slope.tif", max_thresh=5)
    exclusions.add_layer(layer)

    layer = ExclusionLayer("ri_padus.tif", classes_exclude=[1])
    exclusions.add_layer(layer)

    exclusions.apply_all_layers()
    exclusions.export(fname='exclusions.tif')

    --- OR ---

    exclusions = Exclusions([{"fpath": "ri_srtm_slope.tif",
                             "max_thresh": 5,
                            },
                            {"fpath": "ri_padus.tif",
                             "classes_exclude": [1],
                            }])
    exclusions.export(fname='exclusions.tif')

"""
import logging
import rasterio
import numpy as np

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

        with rasterio.open(self.fpath, 'r') as file:
            self.profile = file.profile
            self.block_windows = list(file.block_windows(self.band))
            self.num_windows = len(self.block_windows)

    def read(self):
        """ Read in entire dataset from Tiff """

        with rasterio.open(self.fpath, 'r') as file:
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
        with rasterio.open(self.fpath, 'r') as file:
            data = file.read(self.band, window=window)
        return data, window_slice


class Exclusions:
    """Base class for single output exclusion layer"""

    def __init__(self, layer_configs=None, use_blocks=False):
        """
        Parameters
        ----------
        layer_configs : dictionary list | None
            Optional configs list for the auto addition of layers
        use_blocks : boolean
            Use blocks when applying layers to exclusions
        """
        self.layers = []
        self.data = None
        self.profile = None
        if isinstance(use_blocks, bool):
            self.use_blocks = use_blocks
        else:
            logger.error('use_blocks argument must be a boolean')
        if isinstance(layer_configs, (list, type(None))):
            self.layer_configs = layer_configs
        else:
            logger.error('layer_configs argument must be a list or None')
        if self.layer_configs:
            self.build_config()

    def build_config(self):
        """ Build exclusions from config if it exists """
        if self.layer_configs:
            for config in self.layer_configs:
                if isinstance(config, dict):
                    if 'fpath' not in config:
                        logger.error('fpath key is not defined in config')
                        break
                    layer = ExclusionLayer(**config)
                    self.add_layer(layer)
                else:
                    logger.error('Layer config must be a dictionary')
            self.apply_all_layers()
        else:
            logger.error('Object has no configs: self.layer_configs')

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
            self.profile['dtype'] = rasterio.uint8
            self.profile['nodata'] = None
            self.initialize_data()
        except IndexError:
            logger.error('Exclusion has no layers '
                         '(i.e. self.add_layer(layer))')

    def initialize_data(self):
        """ Initiate the array for the output exclusion data """

        if self.profile:
            shape = (self.profile['height'], self.profile['width'])
            self.data = np.ones(shape=shape, dtype='uint8') * 100
        else:
            logger.error('Profile has not been created yet '
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
                data, window_slice = layer.read_window(window_index)
                mask = self.get_method_mask(data, layer)
                if mask is not None:
                    block_data = (self.data[window_slice] * mask)
                    self.data[window_slice] = block_data.astype('uint8')
                else:
                    logger.warning('Failed to mask {}'.format(layer.name))
        else:
            data = layer.read()
            mask = self.get_method_mask(data, layer)
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

        if self.profile:
            with rasterio.open(fname, 'w', **self.profile) as file:
                file.write(self.data, band)
        else:
            logger.error('Profile has not been created yet '
                         '(i.e. self.create_profile())')
