"""
Classes to handle capacity factor profiles and annual averages
"""
import h5py
import json
import numpy as np
import os
import pandas as pd
from reV.exceptions import (ResourceRuntimeError, ResourceKeyError,
                            ResourceValueError)
from reV.handlers.resource import Resource, parse_keys


class CapacityFactor(Resource):
    """
    Base class to handle capacity factor data in .h5 format
    """
    def __init__(self, h5_file, unscale=True, mode='r'):
        """
        Parameters
        ----------
        h5_file : str
            Path to .h5 resource file
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        mode : str
            Mode to instantiate h5py.File instance
        """
        self._h5_file = h5_file
        self._h5 = h5py.File(h5_file, mode=mode)
        self._unscale = unscale
        self._mode = mode

    def __len__(self):
        _len = 0
        if 'time_index' in self.dsets:
            _len = super().__len__()

        return _len

    def __getitem__(self, keys):
        ds, ds_slice = parse_keys(keys)
        if ds in self.dsets:
            if ds == 'time_index':
                out = self._time_index(*ds_slice)
            elif ds == 'meta':
                out = self._meta(*ds_slice)
            else:
                out = self._get_ds(ds, *ds_slice)
        else:
            msg = '{} is not a valid Dataset'
            raise ResourceKeyError(msg)

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
        if 'time_index' in dsets and 'meta' in dsets:
            _shape = super().shape

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
            msg = 'mode must be writable: {}'.format(mode)
            raise ResourceRuntimeError(msg)

        return True

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
            ds_slice = slice(None, None, None)
            self._set_ds_array('meta', meta, ds_slice)
        else:
            self.create_ds('meta', meta.shape, meta.dtype, data=meta)

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
            ds_slice = slice(None, None, None)
            self._set_ds_array('time_index', time_index, ds_slice)
        else:
            self.create_ds('time_index', time_index.shape, time_index.dtype,
                           data=time_index)

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

                self._h5['meta'].attr[key] = config

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

    def to_records_array(self, df):
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
            dtype = self.get_dtype(c_data)
            if np.issubdtype(dtype, np.bytes_):
                data = c_data.str.encode('utf-8').values
            else:
                data = c_data.values

            arr = np.array(data, dtype=dtype)
            meta_arrays.append(arr)
            dtypes.append((c_name, dtype))

        return np.core.records.fromarrays(meta_arrays, dtype=dtypes)

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
            raise ResourceRuntimeError(msg)

        self._h5[ds_name][ds_slice] = arr

    def _chunks(self, chunks):
        """
        Convert dataset chunk size into valid tuple based on variable array
        shape

        Parameters
        ----------
        chunks : tuple
            Desired dataset chunk size

        Returns
        -------
        ds_chunks : tuple
            dataset chunk size
        """
        if chunks is not None:
            shape = self.shape
            if chunks[0] is None:
                chunk_0 = shape[0]
            else:
                chunk_0 = chunks[0]

            if chunks[1] is None:
                chunk_1 = shape[1]
            else:
                chunk_1 = chunks[1]

            ds_chunks = (chunk_0, chunk_1)
        else:
            ds_chunks = None

        return ds_chunks

    def create_ds(self, ds_name, shape, dtype, chunks=None, attrs=None,
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
            chunks = self._chunks(chunks)
            ds = self._h5.create_dataset(ds_name, shape=shape, dtype=dtype,
                                         chunks=chunks)
            if attrs is not None:
                for key, value in attrs.items():
                    ds.attrs[key] = value

            if data is not None:
                ds[...] = data

    @classmethod
    def write_profiles(cls, h5_file, meta, time_index, cf_profiles,
                       SAM_configs, **kwargs):
        """
        Write cf_profiles to disk

        Parameters
        ----------
        h5_file : str
            Path to .h5 resource file
        meta : pandas.Dataframe
            Locational meta data
        time_index : pandas.DatetimeIndex
            Temporal timesteps
        cf_profiles : ndarray
            Capacity factor profiles
        SAM_configs : dict
            Dictionary of SAM configuration JSONs used to compute cf profiles
        """
        if cf_profiles.shape != (len(time_index), len(meta)):
            msg = 'CF profile dimensions does not match time index and meta'
            raise ResourceValueError(msg)

        with cls(h5_file, mode=kwargs.get('mode', 'w-')) as cf:
            # Save time index
            cf['time_index'] = time_index
            # Save meta
            cf['meta'] = meta
            # Add SAM configurations as attributes to meta
            cf.set_configs(SAM_configs)
            # Save CF
            cf_attrs = {'scale_factor': 1000, 'units': 'unitless'}
            if np.issubdtype(cf_profiles.dtype, np.floating):
                cf_profiles = (cf_profiles * 1000).astype('uint16')

            cf.create_ds('cf_profiles', cf_profiles.shape, cf_profiles.dtype,
                         chunks=(None, 100), attrs=cf_attrs,
                         data=cf_profiles)

    @classmethod
    def write_means(cls, h5_file, meta, cf_means, SAM_configs, year=None,
                    **kwargs):
        """
        Write cf_profiles to disk

        Parameters
        ----------
        h5_file : str
            Path to .h5 resource file
        meta : pandas.Dataframe
            Locational meta data
        cf_means : ndarray
            Capacity factor means
        SAM_configs : dict
            Dictionary of SAM configuration JSONs used to compute cf means
        year : int | str
            Year for which cf means were computed
            If None, inferred from h5_file name
        """
        if len(cf_means) != len(meta):
            msg = 'Number of CF means do not match meta'
            raise ResourceValueError(msg)

        if year is None:
            # Assumes file name is of type *_{year}.h5
            year = os.path.basename(h5_file).slit('.')[0].split('_')[-1]

        with cls(h5_file, mode=kwargs.get('mode', 'w-')) as cf:
            # Save meta
            cf['meta'] = meta
            # Add SAM configurations as attributes to meta
            cf.set_configs(SAM_configs)
            # Save CF
            cf_attrs = {'scale_factor': 1000, 'units': 'unitless'}
            if np.issubdtype(cf_means.dtype, np.floating):
                cf_means = (cf_means * 1000).astype('uint16')

            cf.create_ds('cf_{}'.format(year), cf_means.shape, cf_means.dtype,
                         attrs=cf_attrs, data=cf_means)
