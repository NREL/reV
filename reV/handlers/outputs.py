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

from reV.version import __version__
from reV.utilities.exceptions import (HandlerRuntimeError, HandlerKeyError,
                                      HandlerValueError)

from rex.resource import Resource
from rex.utilities.parse_keys import parse_keys, parse_slice
from rex.utilities.utilities import to_records_array

logger = logging.getLogger(__name__)


class Outputs(Resource):
    """
    Base class to handle reV output data in .h5 format

    Examples
    --------
    The reV Outputs handler can be used to initialize h5 files in the standard
    reV/rex resource data format.

    >>> from reV import Outputs
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> meta = pd.DataFrame({'latitude': np.ones(100),
    >>>                      'longitude': np.ones(100)})
    >>>
    >>> time_index = pd.date_range('20210101', '20220101', freq='1h',
    >>>                            closed='right')
    >>>
    >>> with Outputs('test.h5', 'w') as f:
    >>>     f.meta = meta
    >>>     f.time_index = time_index

    You can also use the Outputs handler to read output h5 files from disk.
    The Outputs handler will automatically parse the meta data and time index
    into the expected pandas objects (DataFrame and DatetimeIndex,
    respectively).

    >>> with Outputs('test.h5') as f:
    >>>     print(f.meta.head())
    >>>
         latitude  longitude
    gid
    0         1.0        1.0
    1         1.0        1.0
    2         1.0        1.0
    3         1.0        1.0
    4         1.0        1.0

    >>> with Outputs('test.h5') as f:
    >>>     print(f.time_index)
    DatetimeIndex(['2021-01-01 01:00:00+00:00', '2021-01-01 02:00:00+00:00',
                   '2021-01-01 03:00:00+00:00', '2021-01-01 04:00:00+00:00',
                   '2021-01-01 05:00:00+00:00', '2021-01-01 06:00:00+00:00',
                   '2021-01-01 07:00:00+00:00', '2021-01-01 08:00:00+00:00',
                   '2021-01-01 09:00:00+00:00', '2021-01-01 10:00:00+00:00',
                   ...
                   '2021-12-31 15:00:00+00:00', '2021-12-31 16:00:00+00:00',
                   '2021-12-31 17:00:00+00:00', '2021-12-31 18:00:00+00:00',
                   '2021-12-31 19:00:00+00:00', '2021-12-31 20:00:00+00:00',
                   '2021-12-31 21:00:00+00:00', '2021-12-31 22:00:00+00:00',
                   '2021-12-31 23:00:00+00:00', '2022-01-01 00:00:00+00:00'],
                  dtype='datetime64[ns, UTC]', length=8760, freq=None)

    There are a few ways to use the Outputs handler to write data to a file.
    Here is one example using the pre-initialized file we created earlier.
    Note that the Outputs handler will automatically scale float data using
    the "scale_factor" attribute. The Outputs handler will unscale the data
    while being read unless the unscale kwarg is explicityly set to False.
    This behavior is intended to reduce disk storage requirements for big
    data and can be disabled by setting dtype=np.float32 or dtype=np.float64
    when writing data.

    >>> Outputs.add_dataset(h5_file='test.h5', dset_name='dset1',
    >>>                     dset_data=np.ones((8760, 100)) * 42.42,
    >>>                     attrs={'scale_factor': 100}, dtype=np.int32)


    >>> with Outputs('test.h5') as f:
    >>>     print(f['dset1'])
    >>>     print(f['dset1'].dtype)
    [[42.42 42.42 42.42 ... 42.42 42.42 42.42]
     [42.42 42.42 42.42 ... 42.42 42.42 42.42]
     [42.42 42.42 42.42 ... 42.42 42.42 42.42]
     ...
     [42.42 42.42 42.42 ... 42.42 42.42 42.42]
     [42.42 42.42 42.42 ... 42.42 42.42 42.42]
     [42.42 42.42 42.42 ... 42.42 42.42 42.42]]
    float32

    >>> with Outputs('test.h5', unscale=False) as f:
    >>>     print(f['dset1'])
    >>>     print(f['dset1'].dtype)
    [[4242 4242 4242 ... 4242 4242 4242]
     [4242 4242 4242 ... 4242 4242 4242]
     [4242 4242 4242 ... 4242 4242 4242]
     ...
     [4242 4242 4242 ... 4242 4242 4242]
     [4242 4242 4242 ... 4242 4242 4242]
     [4242 4242 4242 ... 4242 4242 4242]]
    int32

    Note that the reV Outputs handler is specifically designed to read and
    write spatiotemporal data. It is therefore important to intialize the meta
    data and time index objects even if your data is only spatial or only
    temporal. Furthermore, the Outputs handler will always assume that 1D
    datasets represent scalar data (non-timeseries) that corresponds to the
    meta data shape, and that 2D datasets represent spatiotemporal data whose
    shape corresponds to (len(time_index), len(meta)). You can see these
    constraints here:

    >>> Outputs.add_dataset(h5_file='test.h5', dset_name='bad_shape',
                            dset_data=np.ones((1, 100)) * 42.42,
                            attrs={'scale_factor': 100}, dtype=np.int32)
    HandlerValueError: 2D data with shape (1, 100) is not of the proper
    spatiotemporal shape: (8760, 100)

    >>> Outputs.add_dataset(h5_file='test.h5', dset_name='bad_shape',
                            dset_data=np.ones((8760,)) * 42.42,
                            attrs={'scale_factor': 100}, dtype=np.int32)
    HandlerValueError: 1D data with shape (8760,) is not of the proper
    spatial shape: (100,)
    """

    def __init__(self, h5_file, mode='r', unscale=True, str_decode=True,
                 group=None):
        """
        Parameters
        ----------
        h5_file : str
            Path to .h5 resource file
        mode : str
            Mode to instantiate h5py.File instance
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        group : str
            Group within .h5 resource file to open
        """
        self.h5_file = h5_file
        self._h5 = h5py.File(h5_file, mode=mode)
        self._unscale = unscale
        self._mode = mode
        self._meta = None
        self._time_index = None
        self._str_decode = str_decode
        self._group = self._check_group(group)

        if self.writable:
            self.set_version_attr()

    def __len__(self):
        _len = 0
        if 'meta' in self.datasets:
            _len = self.h5['meta'].shape[0]

        return _len

    def __getitem__(self, keys):
        ds, ds_slice = parse_keys(keys)
        if ds in self.datasets:
            if ds.endswith('time_index'):
                out = self._get_time_index(ds, ds_slice)
            elif ds.endswith('meta'):
                out = self._get_meta(ds, ds_slice)
            else:
                out = self._get_ds(ds, ds_slice)
        else:
            msg = '{} is not a valid Dataset'.format(ds)
            raise HandlerKeyError(msg)

        return out

    def __setitem__(self, keys, arr):
        if self.writable:
            ds, ds_slice = parse_keys(keys)

            slice_test = False
            if isinstance(ds_slice, tuple):
                slice_test = ds_slice[0] == slice(None, None, None)

            if ds.endswith('meta') and slice_test:
                self._set_meta(ds, arr)
            elif ds.endswith('time_index') and slice_test:
                self._set_time_index(ds, arr)
            else:
                self._set_ds_array(ds, arr, ds_slice)

    def set_version_attr(self):
        """Set the version attribute to the h5 file."""
        self.h5.attrs['version'] = __version__
        self.h5.attrs['package'] = 'reV'

    @property
    def version(self):
        """
        Version of package used to create file

        Returns
        -------
        str
        """
        return self.h5.attrs['version']

    @property
    def package(self):
        """
        Package used to create file

        Returns
        -------
        str
        """
        return self.h5.attrs['package']

    @property
    def source(self):
        """
        Package and version used to create file

        Returns
        -------
        str
        """
        out = ("{}_{}"
               .format(self.h5.attrs['package'], self.h5.attrs['version']))
        return out

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
        dsets = self.datasets
        if 'meta' in dsets:
            _shape = self.h5['meta'].shape
            if 'time_index' in dsets:
                _shape = self.h5['time_index'].shape + _shape

        return _shape

    @property
    def writable(self):
        """
        Check to see if h5py.File instance is writable

        Returns
        -------
        is_writable : bool
            Flag if mode is writable
        """
        is_writable = True
        mode = ['a', 'w', 'w-', 'x']
        if self._mode not in mode:
            is_writable = False

        return is_writable

    @Resource.meta.setter  # pylint: disable-msg=E1101
    def meta(self, meta):
        """
        Write meta data to disk, convert type if neccessary

        Parameters
        ----------
        meta : pandas.DataFrame | numpy.recarray
            Locational meta data
        """
        self._set_meta('meta', meta)

    @Resource.time_index.setter  # pylint: disable-msg=E1101
    def time_index(self, time_index):
        """
        Write time_index to dics, convert type if neccessary

        Parameters
        ----------
        time_index : pandas.DatetimeIndex | ndarray
            Temporal index of timesteps
        """
        self._set_time_index('time_index', time_index)

    @property
    def SAM_configs(self):
        """
        SAM configuration JSONs used to create CF profiles

        Returns
        -------
        configs : dict
            Dictionary of SAM configuration JSONs
        """
        if 'meta' in self.datasets:
            configs = {k: json.loads(v)
                       for k, v in self.h5['meta'].attrs.items()}
        else:
            configs = {}

        return configs

    @property
    def run_attrs(self):
        """
        Runtime attributes stored at the global (file) level

        Returns
        -------
        global_attrs : dict
        """
        return self.global_attrs

    @run_attrs.setter
    def run_attrs(self, run_attrs):
        """
        Set runtime attributes as global (file) attributes

        Parameters
        ----------
        run_attrs : dict
            Dictionary of runtime attributes (args, kwargs)
        """
        if self.writable:
            for k, v in run_attrs.items():
                self.h5.attrs[k] = v

    def _check_group(self, group):
        """
        Ensure group is in .h5 file

        Parameters
        ----------
        group : str
            Group of interest
        """
        if group is not None:
            if group not in self._h5:
                try:
                    if self.writable:
                        self._h5.create_group(group)
                except Exception as ex:
                    msg = ('Cannot create group {}: {}'
                           .format(group, ex))
                    raise HandlerRuntimeError(msg) from ex

        return group

    def _set_meta(self, ds, meta, attrs=None):
        """
        Write meta data to disk

        Parameters
        ----------
        ds : str
            meta dataset name
        meta : pandas.DataFrame | numpy.recarray
            Locational meta data
        attrs : dict
            Attributes to add to the meta data dataset
        """
        self._meta = meta
        if isinstance(meta, pd.DataFrame):
            meta = to_records_array(meta)

        if ds in self.datasets:
            self.update_dset(ds, meta)
        else:
            self._create_dset(ds, meta.shape, meta.dtype, data=meta,
                              attrs=attrs)

    def _set_time_index(self, ds, time_index, attrs=None):
        """
        Write time index to disk

        Parameters
        ----------
        ds : str
            time index dataset name
        time_index : pandas.DatetimeIndex | ndarray
            Temporal index of timesteps
        attrs : dict
            Attributes to add to the meta data dataset
        """
        self._time_index = time_index
        if isinstance(time_index, pd.DatetimeIndex):
            time_index = time_index.astype(str)
            dtype = "S{}".format(len(time_index[0]))
            time_index = np.array(time_index, dtype=dtype)

        if ds in self.datasets:
            self.update_dset(ds, time_index)
        else:
            self._create_dset(ds, time_index.shape, time_index.dtype,
                              data=time_index, attrs=attrs)

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
        if 'meta' in self.datasets:
            config = json.loads(self.h5['meta'].attrs[config_name])
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

                self.h5['meta'].attrs[key] = config

    @staticmethod
    def _check_data_dtype(data, dtype, scale_factor=1):
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
            if scale_factor == 1:
                raise HandlerRuntimeError('A scale_factor is needed to'
                                          'scale "{}" data to "{}".'
                                          .format(data.dtype, dtype))

            # apply scale factor and dtype
            data = np.multiply(data, scale_factor)
            if np.issubdtype(dtype, np.integer):
                data = np.round(data)

            data = data.astype(dtype)

        return data

    def _set_ds_array(self, ds_name, arr, ds_slice):
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
        if ds_name not in self.datasets:
            msg = '{} must be initialized!'.format(ds_name)
            raise HandlerRuntimeError(msg)

        dtype = self.h5[ds_name].dtype
        scale_factor = self.get_scale_factor(ds_name)
        ds_slice = parse_slice(ds_slice)
        self.h5[ds_name][ds_slice] = self._check_data_dtype(arr, dtype,
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
                     data=None, replace=True):
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
        replace : bool
            If previous dataset exists with the same name, it will be replaced.
        """
        if self.writable:
            if ds_name in self.datasets and replace:
                del self.h5[ds_name]

            elif ds_name in self.datasets:
                old_shape, old_dtype, _ = self.get_dset_properties(ds_name)
                if old_shape != shape or old_dtype != dtype:
                    e = ('Trying to create dataset "{}", but already exists '
                         'with mismatched shape and dtype. New shape/dtype '
                         'is {}/{}, previous shape/dtype is {}/{}'
                         .format(ds_name, shape, dtype, old_shape, old_dtype))
                    logger.error(e)
                    raise HandlerRuntimeError(e)

            if ds_name not in self.datasets:
                chunks = self._check_chunks(chunks, data=data)
                ds = self.h5.create_dataset(ds_name, shape=shape, dtype=dtype,
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
                    raise HandlerValueError("1D data with shape {} is not of "
                                            "the proper spatial shape:"
                                            " {}".format(dset_shape, shape))
            else:
                raise HandlerRuntimeError("'meta' has not been loaded")
        else:
            shape = self.shape
            if shape:
                if dset_shape != shape:
                    raise HandlerValueError("2D data with shape {} is not of "
                                            "the proper spatiotemporal shape:"
                                            " {}".format(dset_shape, shape))
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

        if attrs is not None:
            scale_factor = attrs.get('scale_factor', 1)
        else:
            scale_factor = 1

        data = self._check_data_dtype(data, dtype, scale_factor=scale_factor)

        self._create_dset(dset_name, data.shape, dtype,
                          chunks=chunks, attrs=attrs, data=data)

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
            self._set_ds_array(dset, dset_array, dset_slice)

    def write_dataset(self, dset_name, data, dtype, chunks=None, attrs=None):
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
        self._add_dset(dset_name, data, dtype, chunks=chunks, attrs=attrs)

    @classmethod
    def write_profiles(cls, h5_file, meta, time_index, dset_name, profiles,
                       attrs, dtype, SAM_configs=None, chunks=(None, 100),
                       unscale=True, mode='w-', str_decode=True, group=None):
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
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        mode : str
            Mode to instantiate h5py.File instance
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        group : str
            Group within .h5 resource file to open
        """
        logger.info("Saving profiles ({}) to {}".format(dset_name, h5_file))
        if profiles.shape != (len(time_index), len(meta)):
            raise HandlerValueError("Profile dimensions does not match"
                                    "'time_index' and 'meta'")
        ts = time.time()
        kwargs = {"unscale": unscale, "mode": mode, "str_decode": str_decode,
                  "group": group}
        with cls(h5_file, **kwargs) as f:
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
                    SAM_configs=None, chunks=None, unscale=True, mode='w-',
                    str_decode=True, group=None):
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
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        mode : str
            Mode to instantiate h5py.File instance
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        group : str
            Group within .h5 resource file to open
        """
        logger.info("Saving means ({}) to {}".format(dset_name, h5_file))
        if len(means) != len(meta):
            msg = 'Number of means does not match meta'
            raise HandlerValueError(msg)

        ts = time.time()
        kwargs = {"unscale": unscale, "mode": mode, "str_decode": str_decode,
                  "group": group}
        with cls(h5_file, **kwargs) as f:
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
                    chunks=None, unscale=True, mode='a', str_decode=True,
                    group=None):
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
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        mode : str
            Mode to instantiate h5py.File instance
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        group : str
            Group within .h5 resource file to open
        """
        logger.info("Adding {} to {}".format(dset_name, h5_file))
        ts = time.time()
        kwargs = {"unscale": unscale, "mode": mode, "str_decode": str_decode,
                  "group": group}
        with cls(h5_file, **kwargs) as f:
            f._add_dset(dset_name, dset_data, dtype,
                        chunks=chunks, attrs=attrs)

        tt = (time.time() - ts) / 60
        logger.info('{} added'.format(dset_name))
        logger.debug('\t- Saving to disc took {:.4f} minutes'
                     .format(tt))

    @classmethod
    def init_h5(cls, h5_file, dsets, shapes, attrs, chunks, dtypes,
                meta, time_index=None, configs=None, unscale=True, mode='w',
                str_decode=True, group=None, run_attrs=None):
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
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        mode : str
            Mode to instantiate h5py.File instance
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        group : str
            Group within .h5 resource file to open
        run_attrs : dict | NoneType
            Runtime attributes (args, kwargs) to add as global (file)
            attributes
        """

        logger.debug("Initializing output file: {}".format(h5_file))
        kwargs = {"unscale": unscale, "mode": mode, "str_decode": str_decode,
                  "group": group}
        with cls(h5_file, **kwargs) as f:
            if run_attrs is not None:
                f.run_attrs = run_attrs

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
