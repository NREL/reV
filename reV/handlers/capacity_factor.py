"""
Classes to handle capacity factor profiles and annual averages
"""
import h5py
import numpy as np
import pandas as pd
from reV.exceptions import ResourceRuntimeError
from reV.handler.resource import Resource, parse_keys


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
        self._unscale = unscale
        self._mode = mode

    def __enter__(self):
        self.open()
        return self

    def __len__(self):
        len = None
        if self.hasattr('_h5'):
            if 'time_index' in self._h5:
                len = super(CapacityFactor).len(self)

        return len

    def __setitem__(self, keys, arr):
        mode = ['a', 'w', 'w-', 'x']
        if self._mode not in mode:
            msg = 'mode must be writable: {}'.format(mode)
            raise ResourceRuntimeError(msg)

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
        shape = None
        if self.hasattr('_h5'):
            if 'time_index' in self._h5 and 'meta' in self._h5:
                shape = (len(self._h5['time_index'].shape[0]),
                         len(self._h5['meta'].shape[0]))

        return shape

    @meta.setter
    def meta(self, meta):
        """
        Write meta data to disc, convert type if neccessary

        Parameters
        ----------
        meta : pandas.DataFrame | numpy.recarray
            Locational meta data
        """
        if isinstance(meta, pd.DataFrame):
            meta = self.to_records_array(meta)

        ds_slice = slice(None, None, None)
        self._set_ds_array('meta', meta, ds_slice)

    @time_index.setter
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

        ds_slice = slice(None, None, None)
        self._set_ds_array('time_index', time_index, ds_slice)

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
        if np.issubdtype(dtype, np.float):
            out = 'float32'
        elif np.issubdtype(dtype, np.int):
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
        Write ds to disc
        """
        close = False
        if not self.hasattr('_h5'):
            close = True
            self.open()

        if ds_name not in self._h5:
            msg = '{} must be initialized!'.format(ds_name)
            raise ResourceRuntimeError(msg)

        self._h5[ds_name][ds_slice] = arr

        if close:
            self.close()

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

    def open(self):
        """
        Initialize h5py File instance
        """
        self._h5 = h5py.File(self._h5_path, mode=self._mode)

    def close(self):
        """
        Close h5 instance
        """
        if self.hasattr('_h5'):
            self._h5.close()
