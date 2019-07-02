# -*- coding: utf-8 -*-
"""
Classes to handle reV h5 output files.
"""
import h5py
import json
import logging
import numpy as np
import pandas as pd
import time

from reV.handlers.resource import Resource
from reV.handlers.sam_resource import parse_keys
from reV.utilities.exceptions import (HandlerRuntimeError, HandlerKeyError,
                                      HandlerValueError)

logger = logging.getLogger(__name__)


class Outputs(Resource):
    """
    Base class to handle reV output data in .h5 format
    """
    def __init__(self, h5_file, unscale=True, mode='r', str_decode=True):
        """
        Parameters
        ----------
        h5_file : str
            Path to .h5 resource file
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        mode : str
            Mode to instantiate h5py.File instance
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        """
        self._h5_file = h5_file
        self._h5 = h5py.File(h5_file, mode=mode)
        self._unscale = unscale
        self._mode = mode
        self._meta = None
        self._time_index = None
        self._str_decode = str_decode

    def __len__(self):
        _len = 0
        if 'meta' in self.dsets:
            _len = super().__len__()

        return _len

    def __getitem__(self, keys):
        ds, ds_slice = parse_keys(keys)
        if ds in self.dsets:
            if ds == 'time_index':
                out = self._get_time_index(*ds_slice)
            elif ds == 'meta':
                out = self._get_meta(*ds_slice)
            else:
                out = self._get_ds(ds, *ds_slice)
        else:
            msg = '{} is not a valid Dataset'.format(ds)
            raise HandlerKeyError(msg)

        return out

    def __setitem__(self, keys, arr):
        if self.writable:
            ds, ds_slice = parse_keys(keys)
            slice_test = ds_slice == (slice(None, None, None),)
            if ds == 'meta' and slice_test:
                self.meta = arr
            elif ds == 'time_index' and slice_test:
                self.time_index = arr
            else:
                self._set_ds_array(ds, arr, *ds_slice)

    @property
    def shape(self):
        """
        Variable array shape from time_index and meta

        Returns
        -------
        tuple
            shape of variables arrays == (time, locations)
        """
        _shape = None
        dsets = self.dsets
        if 'meta' in dsets:
            _shape = self._h5['meta'].shape
            if 'time_index' in dsets:
                _shape = self._h5['time_index'].shape + _shape

        return _shape

    @property
    def writable(self):
        """
        Check to see if h5py.File instance is writable

        Returns
        -------
        bool
            Flag if mode is writable
        """
        mode = ['a', 'w', 'w-', 'x']
        if self._mode not in mode:
            raise HandlerRuntimeError('mode must be writable: {}'
                                      .format(mode))

        return True

    def update_dset(self, dset, dset_array, dset_slice=None):
        """
        Check to see if dset needs to be updated on disk
        If so write dset_array to disk

        Parameters
        ----------
        dset : str
            dataset to update
        dset_array : ndarray
            dataset array
        dset_slice : tuple
            slice of dataset to update, it None update all
        """
        if dset_slice is None:
            dset_slice = (slice(None, None, None), )

        keys = (dset, ) + dset_slice

        arr = self.__getitem__(keys)
        if not np.array_equal(arr, dset_array):
            self._set_ds_array(dset, dset_array, *dset_slice)

    @Resource.meta.setter  # pylint: disable-msg=E1101
    def meta(self, meta):
        """
        Write meta data to disk, convert type if neccessary

        Parameters
        ----------
        meta : pandas.DataFrame | numpy.recarray
            Locational meta data
        """
        if isinstance(meta, pd.DataFrame):
            meta = self.to_records_array(meta)

        if 'meta' in self.dsets:
            self.update_dset('meta', meta)
        else:
            self._create_dset('meta', meta.shape, meta.dtype, data=meta)

        self._meta = meta

    @Resource.time_index.setter  # pylint: disable-msg=E1101
    def time_index(self, time_index):
        """
        Write time_index to dics, convert type if neccessary

        Parameters
        ----------
        time_index : pandas.DatetimeIndex | ndarray
            Temporal index of timesteps
        """
        if isinstance(time_index, pd.DatetimeIndex):
            time_index = np.array(time_index.astype(str), dtype='S20')

        if 'time_index' in self.dsets:
            self.update_dset('time_index', time_index)
        else:
            self._create_dset('time_index', time_index.shape, time_index.dtype,
                              data=time_index)

        self._time_index = time_index

    @property
    def SAM_configs(self):
        """
        SAM configuration JSONs used to create CF profiles

        Returns
        -------
        configs : dict
            Dictionary of SAM configuration JSONs
        """
        if 'meta' in self.dsets:
            configs = {k: json.loads(v)
                       for k, v in self._h5['meta'].attrs.items()}
        else:
            configs = {}

        return configs

    def get_config(self, config_name):
        """
        Get SAM config

        Parameters
        ----------
        config_name : str
            Name of config

        Returns
        -------
        config : dict
            SAM config JSON as a dictionary
        """
        if 'meta' in self.dsets:
            config = json.loads(self._h5['meta'].attrs[config_name])
        else:
            config = None

        return config

    def set_configs(self, SAM_configs):
        """
        Set SAM configuration JSONs as attributes of 'meta'

        Parameters
        ----------
        SAM_configs : dict
            Dictionary of SAM configuration JSONs
        """
        if self.writable:
            for key, config in SAM_configs.items():
                if isinstance(config, dict):
                    config = json.dumps(config)

                if not isinstance(key, str):
                    key = str(key)

                self._h5['meta'].attrs[key] = config

    @staticmethod
    def get_dtype(col):
        """
        Get column dtype for converstion to records array

        Parameters
        ----------
        col : pandas.Series
            Column from pandas DataFrame

        Returns
        -------
        str
            converted dtype for column
            -  float = float32
            -  int = int16 or int32 depending on data range
            -  object/str = U* max length of strings in col
        """
        dtype = col.dtype
        if np.issubdtype(dtype, np.floating):
            out = 'float32'
        elif np.issubdtype(dtype, np.integer):
            if col.max() < 32767:
                out = 'int16'
            else:
                out = 'int32'
        elif np.issubdtype(dtype, np.object_):
            size = int(col.str.len().max())
            out = 'S{:}'.format(size)
        else:
            out = dtype

        return out

    @staticmethod
    def to_records_array(df):
        """
        Convert pandas DataFrame to numpy Records Array

        Parameters
        ----------
        df : pandas.DataFrame
            Pandas DataFrame to be converted

        Returns
        -------
        numpy.rec.array
            Records array of input df
        """
        meta_arrays = []
        dtypes = []
        for c_name, c_data in df.iteritems():
            dtype = Outputs.get_dtype(c_data)
            if np.issubdtype(dtype, np.bytes_):
                data = c_data.str.encode('utf-8').values
            else:
                data = c_data.values

            arr = np.array(data, dtype=dtype)
            meta_arrays.append(arr)
            dtypes.append((c_name, dtype))

        return np.core.records.fromarrays(meta_arrays, dtype=dtypes)

    @staticmethod
    def _check_data_dtype(data, dtype, scale_factor):
        """
        Check data dtype and scale if needed

        Parameters
        ----------
        data : ndarray
            Data to be written to disc
        dtype : str
            dtype of data on disc
        scale_factor : int
            Scale factor to scale data to integer (if needed)

        Returns
        -------
        data : ndarray
            Data ready for writing to disc:
            - Scaled and converted to dtype
        """
        if not np.issubdtype(data.dtype, np.dtype(dtype)):
            # apply scale factor and dtype
            data = np.multiply(data, scale_factor)
            if np.issubdtype(dtype, np.integer):
                data = np.round(data)

            data = data.astype(dtype)

        return data

    def _set_ds_array(self, ds_name, arr, *ds_slice):
        """
        Write ds to disk

        Parameters
        ----------
        ds_name : str
            Dataset name
        arr : ndarray
            Dataset data array
        ds_slice : tuple
            Dataset slicing that corresponds to arr
        """
        if ds_name not in self._h5:
            msg = '{} must be initialized!'.format(ds_name)
            raise HandlerRuntimeError(msg)

        dtype = self._h5[ds_name].dtype
        scale_factor = self.get_scale(ds_name)

        self._h5[ds_name][ds_slice] = self._check_data_dtype(arr, dtype,
                                                             scale_factor)

    def _check_chunks(self, chunks, data=None):
        """
        Convert dataset chunk size into valid tuple based on variable array
        shape
        Parameters
        ----------
        chunks : tuple
            Desired dataset chunk size
        data : ndarray
            Dataset array being chunked

        Returns
        -------
        ds_chunks : tuple
            dataset chunk size
        """
        if chunks is not None:
            if data is not None:
                shape = data.shape
            else:
                shape = self.shape

            if chunks[0] is None:
                chunk_0 = shape[0]
            else:
                chunk_0 = np.min((shape[0], chunks[0]))

            if chunks[1] is None:
                chunk_1 = shape[1]
            else:
                chunk_1 = np.min((shape[1], chunks[1]))

            ds_chunks = (chunk_0, chunk_1)
        else:
            ds_chunks = None

        return ds_chunks

    def _create_dset(self, ds_name, shape, dtype, chunks=None, attrs=None,
                     data=None):
        """
        Initialize dataset

        Parameters
        ----------
        ds_name : str
            Dataset name
        shape : tuple
            Dataset shape
        dtype : str
            Dataset numpy dtype
        chunks : tuple
            Dataset chunk size
        attrs : dict
            Dataset attributes
        data : ndarray
            Dataset data array
        """
        if self.writable:
            chunks = self._check_chunks(chunks, data=data)
            ds = self._h5.create_dataset(ds_name, shape=shape, dtype=dtype,
                                         chunks=chunks)
            if attrs is not None:
                for key, value in attrs.items():
                    ds.attrs[key] = value

            if data is not None:
                ds[...] = data

    def _check_dset_shape(self, dset_data):
        """
        Check to ensure that dataset array is of the proper shape

        Parameters
        ----------
        dset_data : ndarray
            Dataset data array
        """
        dset_shape = dset_data.shape
        if len(dset_shape) == 1:
            shape = len(self)
            if shape:
                shape = (shape,)
                if dset_shape != shape:
                    raise HandlerValueError("data is not of the proper shape:"
                                            " {}".format(shape))
            else:
                raise HandlerRuntimeError("'meta' has not been loaded")
        else:
            shape = self.shape
            if shape:
                if dset_shape != shape:
                    raise HandlerValueError("data is not of the proper shape:"
                                            " {}".format(shape))
            else:
                raise HandlerRuntimeError("'meta' and 'time_index' have not "
                                          "been loaded")

    def _add_dset(self, dset_name, data, dtype, chunks=None, attrs=None):
        """
        Write dataset to disk. Dataset it created in .h5 file and data is
        scaled if needed.

        Parameters
        ----------
        dset_name : str
            Name of dataset to be added to h5 file.
        data : ndarray
            Data to be added to h5 file.
        dtype : str
            Intended dataset datatype after scaling.
        chunks : tuple
            Chunk size for capacity factor means dataset.
        attrs : dict
            Attributes to be set. May include 'scale_factor'.
        """
        self._check_dset_shape(data)

        if 'scale_factor' in attrs:
            scale_factor = attrs['scale_factor']
            data = self._check_data_dtype(data, dtype, scale_factor)
        else:
            raise HandlerRuntimeError("A scale_factor is needed to"
                                      "scale data to {}.".format(dtype))

        self._create_dset(dset_name, data.shape, dtype,
                          chunks=chunks, attrs=attrs, data=data)

    @classmethod
    def write_profiles(cls, h5_file, meta, time_index, dset_name, profiles,
                       attrs, dtype, SAM_configs=None, chunks=(None, 100),
                       **kwargs):
        """
        Write profiles to disk

        Parameters
        ----------
        h5_file : str
            Path to .h5 resource file
        meta : pandas.Dataframe
            Locational meta data
        time_index : pandas.DatetimeIndex
            Temporal timesteps
        dset_name : str
            Name of the target dataset (should identify the profiles).
        profiles : ndarray
            reV output result timeseries profiles
        attrs : dict
            Attributes to be set. May include 'scale_factor'.
        dtype : str
            Intended dataset datatype after scaling.
        SAM_configs : dict
            Dictionary of SAM configuration JSONs used to compute cf profiles
        chunks : tuple
            Chunk size for profiles dataset
        """
        logger.info("Saving profiles ({}) to {}".format(dset_name, h5_file))
        if profiles.shape != (len(time_index), len(meta)):
            raise HandlerValueError("Profile dimensions does not match"
                                    "'time_index' and 'meta'")
        ts = time.time()
        with cls(h5_file, mode=kwargs.get('mode', 'w-')) as f:
            # Save time index
            f['time_index'] = time_index
            logger.debug("\t- 'time_index' saved to disc")
            # Save meta
            f['meta'] = meta
            logger.debug("\t- 'meta' saved to disc")
            # Add SAM configurations as attributes to meta
            if SAM_configs is not None:
                f.set_configs(SAM_configs)
                logger.debug("\t- SAM configurations saved as attributes "
                             "on 'meta'")

            # Write dset to disk
            f._add_dset(dset_name, profiles, dtype,
                        chunks=chunks, attrs=attrs)
            logger.debug("\t- '{}' saved to disc".format(dset_name))

        tt = (time.time() - ts) / 60
        logger.info('{} is complete'.format(h5_file))
        logger.debug('\t- Saving to disc took {:.4f} minutes'
                     .format(tt))

    @classmethod
    def write_means(cls, h5_file, meta, dset_name, means, attrs, dtype,
                    SAM_configs=None, chunks=None, **kwargs):
        """
        Write means array to disk

        Parameters
        ----------
        h5_file : str
            Path to .h5 resource file
        meta : pandas.Dataframe
            Locational meta data
        dset_name : str
            Name of the target dataset (should identify the means).
        means : ndarray
            reV output means array.
        attrs : dict
            Attributes to be set. May include 'scale_factor'.
        dtype : str
            Intended dataset datatype after scaling.
        SAM_configs : dict
            Dictionary of SAM configuration JSONs used to compute cf means
        chunks : tuple
            Chunk size for capacity factor means dataset
        """
        logger.info("Saving means ({}) to {}".format(dset_name, h5_file))
        if len(means) != len(meta):
            msg = 'Number of means does not match meta'
            raise HandlerValueError(msg)

        ts = time.time()
        with cls(h5_file, mode=kwargs.get('mode', 'w-')) as f:
            # Save meta
            f['meta'] = meta
            logger.debug("\t- 'meta' saved to disc")
            # Add SAM configurations as attributes to meta
            if SAM_configs is not None:
                f.set_configs(SAM_configs)
                logger.debug("\t- SAM configurations saved as attributes "
                             "on 'meta'")

            # Write dset to disk
            f._add_dset(dset_name, means, dtype,
                        chunks=chunks, attrs=attrs)
            logger.debug("\t- '{}' saved to disc".format(dset_name))

        tt = (time.time() - ts) / 60
        logger.info('{} is complete'.format(h5_file))
        logger.debug('\t- Saving to disc took {:.4f} minutes'
                     .format(tt))

    @classmethod
    def add_dataset(cls, h5_file, dset_name, dset_data, attrs, dtype,
                    chunks=None, **kwargs):
        """
        Add dataset to h5_file

        Parameters
        ----------
        h5_file : str
            Path to .h5 resource file
        dset_name : str
            Name of dataset to be added to h5 file
        dset_data : ndarray
            Data to be added to h5 file
        attrs : dict
            Attributes to be set. May include 'scale_factor'.
        dtype : str
            Intended dataset datatype after scaling.
        """
        logger.info("Adding {} to {}".format(dset_name, h5_file))
        ts = time.time()
        with cls(h5_file, mode=kwargs.get('mode', 'a')) as f:
            f._add_dset(dset_name, dset_data, dtype,
                        chunks=chunks, attrs=attrs)

        tt = (time.time() - ts) / 60
        logger.info('{} added'.format(dset_name))
        logger.debug('\t- Saving to disc took {:.4f} minutes'
                     .format(tt))

    @classmethod
    def init_h5(cls, h5_file, dsets, shapes, attrs, chunks, dtypes,
                meta, time_index=None, configs=None):
        """Init a full output file with the final intended shape without data.

        Parameters
        ----------
        h5_file : str
            Full h5 output filepath.
        dsets : list
            List of strings of dataset names to initialize (does not include
            meta or time_index).
        shapes : dict
            Dictionary of dataset shapes (keys correspond to dsets).
        attrs : dict
            Dictionary of dataset attributes (keys correspond to dsets).
        chunks : dict
            Dictionary of chunk tuples (keys correspond to dsets).
        dtypes : dict
            dictionary of numpy datatypes (keys correspond to dsets).
        meta : pd.DataFrame
            Full meta data.
        time_index : pd.datetimeindex | None
            Full pandas datetime index. None implies that only 1D results
            (no site profiles) are being written.
        configs : dict | None
            Optional input configs to set as attr on meta.
        """

        logger.debug("Initializing output file: {}".format(h5_file))
        with cls(h5_file, mode='w') as f:
            f['meta'] = meta

            if time_index is not None:
                f['time_index'] = time_index

            for dset in dsets:
                if dset not in ('meta', 'time_index'):
                    # initialize each dset to disk
                    f._create_dset(dset, shapes[dset], dtypes[dset],
                                   chunks=chunks[dset], attrs=attrs[dset])

            if configs is not None:
                f.set_configs(configs)
                logger.debug("\t- Configurations saved as attributes "
                             "on 'meta'")

        logger.debug('Output file has been initialized.')
