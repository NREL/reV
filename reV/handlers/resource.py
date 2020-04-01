# -*- coding: utf-8 -*-
"""
Classes to handle resource data
"""
import h5py
import numpy as np
import os
import pandas as pd

from reV.handlers.parse_keys import parse_keys, parse_slice
from reV.utilities.exceptions import (HandlerKeyError, HandlerRuntimeError)


class ResourceDataset:
    """
    h5py.Dataset wrapper for Resource .h5 files
    """
    def __init__(self, ds, scale_attr='scale_factor', add_attr='add_offset',
                 unscale=True):
        """
        Parameters
        ----------
        ds : h5py.dataset
            Open .h5 dataset instance to extract data from
        scale_attr : str, optional
            Name of scale factor attribute, by default 'scale_factor'
        add_attr : str, optional
            Name of add offset attribute, by default 'add_offset'
        unscale : bool, optional
            Flag to unscale dataset data, by default True
        """
        self._ds = ds
        self._scale_factor = self._ds.attrs.get(scale_attr, 1)
        self._adder = self._ds.attrs.get(add_attr, 0)
        self._unscale = unscale

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self._ds.name)

        return msg

    def __getitem__(self, ds_slice):
        ds_slice = parse_slice(ds_slice)
        out = self._extract_ds_slice(ds_slice)
        if self._unscale:
            out = self._unscale_data(out)

        return out

    @property
    def shape(self):
        """
        Dataset shape

        Returns
        -------
        tuple
        """
        return self._ds.shape

    @property
    def size(self):
        """
        Dataset size

        Returns
        -------
        int
        """
        return self._ds.size

    @property
    def dtype(self):
        """
        Dataset dtype

        Returns
        -------
        str | numpy.dtype
        """
        return self._ds.dtype

    @property
    def chunks(self):
        """
        Dataset chunk size

        Returns
        -------
        tuple
        """
        return self._ds.chunks

    @property
    def scale_factor(self):
        """
        Dataset scale factor

        Returns
        -------
        float
        """
        return self._scale_factor

    @property
    def adder(self):
        """
        Dataset add offset

        Returns
        -------
        float
        """
        return self._adder

    @staticmethod
    def _check_slice(ds_slice):
        """
        Check ds_slice to see if it is an int, slice, or list. Return
        pieces required for fancy indexing based on input type.

        Parameters
        ----------
        ds_slice : slice | list | ndarray
            slice, list, or vector of points to extract

        Returns
        -------
        ds_slice : slice
            Slice that encompasses the entire range
        ds_idx : ndarray
            Adjusted list to extract points of interest from sliced array
        """
        ds_idx = None
        if isinstance(ds_slice, (list, np.ndarray)):
            in_slice = np.array(ds_slice)
            s = in_slice.min()
            e = in_slice.max() + 1
            ds_slice = slice(s, e, None)
            ds_idx = in_slice - s
        elif isinstance(ds_slice, slice):
            ds_idx = slice(None)

        return ds_slice, ds_idx

    def _extract_ds_slice(self, ds_slice):
        """
        Extact ds_slice from ds as efficiently as possible.

        Parameters
        ----------
        ds_slice : int | slice | list | ndarray
            What to extract from ds, each arg is for a sequential axis

        Returns
        -------
        out : ndarray
            Extracted array of data from ds
        """
        slices = ()
        idx_slice = ()
        for ax_slice in ds_slice:
            ax_slice, ax_idx = ResourceDataset._check_slice(ax_slice)
            slices += (ax_slice,)
            if ax_idx is not None:
                idx_slice += (ax_idx,)

        out = self._ds[slices]
        if idx_slice:
            out = out[idx_slice]

        return out

    def _unscale_data(self, data):
        """
        Unscale dataset data

        Parameters
        ----------
        data : ndarray
            Native dataset array

        Returns
        -------
        data : ndarray
            Unscaled dataset array
        """
        data = data.astype('float32')

        if self.adder != 0:
            data *= self.scale_factor
            data += self.adder
        else:
            data /= self.scale_factor

        return data

    @classmethod
    def extract(cls, ds, ds_slice, scale_attr='scale_factor',
                add_attr='add_offset', unscale=True):
        """
        Extract data from Resource Dataset

        Parameters
        ----------
        ds : h5py.dataset
            Open .h5 dataset instance to extract data from
        ds_slice : slice | list | ndarray
            slice, list, or vector of points to extract
        scale_attr : str, optional
            Name of scale factor attribute, by default 'scale_factor'
        add_attr : str, optional
            Name of add offset attribute, by default 'add_offset'
        unscale : bool, optional
            Flag to unscale dataset data, by default True
        """
        dset = cls(ds, scale_attr=scale_attr, add_attr=add_attr,
                   unscale=unscale)

        return dset[ds_slice]


class Resource:
    """
    Base class to handle resource .h5 files

    Examples
    --------

    Extracting the resource's Datetime Index

    >>> file = '$TESTDATADIR/nsrdb/ri_100_nsrdb_2012.h5'
    >>> with Resource(file) as res:
    >>>     ti = res.time_index
    >>>
    >>> ti
    DatetimeIndex(['2012-01-01 00:00:00', '2012-01-01 00:30:00',
                   '2012-01-01 01:00:00', '2012-01-01 01:30:00',
                   '2012-01-01 02:00:00', '2012-01-01 02:30:00',
                   '2012-01-01 03:00:00', '2012-01-01 03:30:00',
                   '2012-01-01 04:00:00', '2012-01-01 04:30:00',
                   ...
                   '2012-12-31 19:00:00', '2012-12-31 19:30:00',
                   '2012-12-31 20:00:00', '2012-12-31 20:30:00',
                   '2012-12-31 21:00:00', '2012-12-31 21:30:00',
                   '2012-12-31 22:00:00', '2012-12-31 22:30:00',
                   '2012-12-31 23:00:00', '2012-12-31 23:30:00'],
                  dtype='datetime64[ns]', length=17568, freq=None)

    Efficient slicing of the Datetime Index

    >>> with Resource(file) as res:
    >>>     ti = res['time_index', 1]
    >>>
    >>> ti
    2012-01-01 00:30:00

    >>> with Resource(file) as res:
    >>>     ti = res['time_index', :10]
    >>>
    >>> ti
    DatetimeIndex(['2012-01-01 00:00:00', '2012-01-01 00:30:00',
                   '2012-01-01 01:00:00', '2012-01-01 01:30:00',
                   '2012-01-01 02:00:00', '2012-01-01 02:30:00',
                   '2012-01-01 03:00:00', '2012-01-01 03:30:00',
                   '2012-01-01 04:00:00', '2012-01-01 04:30:00'],
                  dtype='datetime64[ns]', freq=None)

    >>> with Resource(file) as res:
    >>>     ti = res['time_index', [1, 2, 4, 8, 9]
    >>>
    >>> ti
    DatetimeIndex(['2012-01-01 00:30:00', '2012-01-01 01:00:00',
                   '2012-01-01 02:00:00', '2012-01-01 04:00:00',
                   '2012-01-01 04:30:00'],
                  dtype='datetime64[ns]', freq=None)

    Extracting resource's site metadata

    >>> with Resource(file) as res:
    >>>     meta = res.meta
    >>>
    >>> meta
            latitude  longitude   elevation  timezone    country ...
    0      41.29     -71.86    0.000000        -5           None ...
    1      41.29     -71.82    0.000000        -5           None ...
    2      41.25     -71.82    0.000000        -5           None ...
    3      41.33     -71.82   15.263158        -5  United States ...
    4      41.37     -71.82   25.360000        -5  United States ...
    ..       ...        ...         ...       ...            ... ...
    95     41.25     -71.66    0.000000        -5           None ...
    96     41.89     -71.66  153.720000        -5  United States ...
    97     41.45     -71.66   35.440000        -5  United States ...
    98     41.61     -71.66  140.200000        -5  United States ...
    99     41.41     -71.66   35.160000        -5  United States ...
    [100 rows x 10 columns]

    Efficient slicing of the metadata

    >>> with Resource(file) as res:
    >>>     meta = res['meta', 1]
    >>>
    >>> meta
       latitude  longitude  elevation  timezone country state county urban ...
    1     41.29     -71.82        0.0        -5    None  None   None  None ...

    >>> with Resource(file) as res:
    >>>     meta = res['meta', :5]
    >>>
    >>> meta
       latitude  longitude  elevation  timezone        country ...
    0     41.29     -71.86   0.000000        -5           None ...
    1     41.29     -71.82   0.000000        -5           None ...
    2     41.25     -71.82   0.000000        -5           None ...
    3     41.33     -71.82  15.263158        -5  United States ...
    4     41.37     -71.82  25.360000        -5  United States ...

    >>> with Resource(file) as res:
    >>>     tz = res['meta', :, 'timezone']
    >>>
    >>> tz
    0    -5
    1    -5
    2    -5
    3    -5
    4    -5
         ..
    95   -5
    96   -5
    97   -5
    98   -5
    99   -5
    Name: timezone, Length: 100, dtype: int64

    >>> with Resource(file) as res:
    >>>     lat_lon = res['meta', :, ['latitude', 'longitude']]
    >>>
    >>> lat_lon
        latitude  longitude
    0      41.29     -71.86
    1      41.29     -71.82
    2      41.25     -71.82
    3      41.33     -71.82
    4      41.37     -71.82
    ..       ...        ...
    95     41.25     -71.66
    96     41.89     -71.66
    97     41.45     -71.66
    98     41.61     -71.66
    99     41.41     -71.66
    [100 rows x 2 columns]

    Extracting resource variables (datasets)

    >>> with Resource(file) as res:
    >>>     wspd = res['wind_speed']
    >>>
    >>> wspd
    [[12. 12. 12. ... 12. 12. 12.]
     [12. 12. 12. ... 12. 12. 12.]
     [12. 12. 12. ... 12. 12. 12.]
     ...
     [14. 14. 14. ... 14. 14. 14.]
     [15. 15. 15. ... 15. 15. 15.]
     [15. 15. 15. ... 15. 15. 15.]]

    Efficient slicing of variables

    >>> with Resource(file) as res:
    >>>     wspd = res['wind_speed', :2]
    >>>
    >>> wspd
    [[12. 12. 12. 12. 12. 12. 53. 53. 53. 53. 53. 12. 53.  1.  1. 12. 12. 12.
       1.  1. 12. 53. 53. 53. 12. 12. 12. 12. 12.  1. 12. 12.  1. 12. 12. 53.
      12. 53.  1. 12.  1. 53. 53. 12. 12. 12. 12.  1.  1.  1. 12. 12.  1.  1.
      12. 12. 53. 53. 53. 12. 12. 53. 53. 12. 12. 12. 12. 12. 12.  1. 53.  1.
      53. 12. 12. 53. 53.  1.  1.  1. 53. 12.  1.  1. 53. 53. 53. 12. 12. 12.
      12. 12. 12. 12.  1. 12.  1. 12. 12. 12.]
     [12. 12. 12. 12. 12. 12. 53. 53. 53. 53. 53. 12. 53.  1.  1. 12. 12. 12.
       1.  1. 12. 53. 53. 53. 12. 12. 12. 12. 12.  1. 12. 12.  1. 12. 12. 53.
      12. 53.  1. 12.  1. 53. 53. 12. 12. 12. 12.  1.  1.  1. 12. 12.  1.  1.
      12. 12. 53. 53. 53. 12. 12. 53. 53. 12. 12. 12. 12. 12. 12.  1. 53.  1.
      53. 12. 12. 53. 53.  1.  1.  1. 53. 12.  1.  1. 53. 53. 53. 12. 12. 12.
      12. 12. 12. 12.  1. 12.  1. 12. 12. 12.]]

    >>> with Resource(file) as res:
    >>>     wspd = res['wind_speed', :, [2, 3]]
    >>>
    >>> wspd
    [[12. 12.]
     [12. 12.]
     [12. 12.]
     ...
     [14. 14.]
     [15. 15.]
     [15. 15.]]
    """
    SCALE_ATTR = 'scale_factor'
    ADD_ATTR = 'add_offset'
    UNIT_ATTR = 'units'

    def __init__(self, h5_file, unscale=True, hsds=False, str_decode=True,
                 group=None):
        """
        Parameters
        ----------
        h5_file : str
            Path to .h5 resource file
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        group : str
            Group within .h5 resource file to open
        """
        self.h5_file = h5_file
        if hsds:
            import h5pyd
            self._h5 = h5pyd.File(self.h5_file, 'r')
        else:
            self._h5 = h5py.File(self.h5_file, 'r')

        self._group = group
        self._unscale = unscale
        self._meta = None
        self._time_index = None
        self._str_decode = str_decode
        self._i = 0

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self.h5_file)

        return msg

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

        if type is not None:
            raise

    def __len__(self):
        return self.h5['meta'].shape[0]

    def __getitem__(self, keys):
        ds, ds_slice = parse_keys(keys)

        if ds.endswith('time_index'):
            out = self._get_time_index(ds, ds_slice)
        elif ds.endswith('meta'):
            out = self._get_meta(ds, ds_slice)
        elif 'SAM' in ds:
            site = ds_slice[0]
            if isinstance(site, int):
                out = self._get_SAM_df(ds, site)
            else:
                msg = "Can only extract SAM DataFrame for a single site"
                raise HandlerRuntimeError(msg)
        else:
            out = self._get_ds(ds, ds_slice)

        return out

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self.datasets):
            self._i = 0
            raise StopIteration

        dset = self.datasets[self._i]
        self._i += 1

        return dset

    def __contains__(self, dset):
        return dset in self.datasets

    @staticmethod
    def _get_datasets(h5_obj, group=None):
        """
        Search h5 file instance for Datasets

        Parameters
        ----------
        h5_obj : h5py.File | h5py.Group
            Open h5py File or Group instance to search

        Returns
        -------
        dsets : list
            List of datasets in h5_obj
        """
        dsets = []
        for name in h5_obj:
            sub_obj = h5_obj[name]
            if isinstance(sub_obj, h5py.Group):
                dsets.extend(Resource._get_datasets(sub_obj, group=name))
            else:
                dset_name = name
                if group is not None:
                    dset_name = "{}/{}".format(group, dset_name)

                dsets.append(dset_name)

        return dsets

    @property
    def h5(self):
        """
        Open h5py File instance. If _group is not None return open Group

        Returns
        -------
        h5 : h5py.File | h5py.Group
            Open h5py File or Group instance
        """
        h5 = self._h5
        if self._group is not None:
            h5 = h5[self._group]

        return h5

    @property
    def datasets(self):
        """
        Datasets available

        Returns
        -------
        list
            List of datasets
        """
        return self._get_datasets(self.h5)

    @property
    def dsets(self):
        """
        Datasets available

        Returns
        -------
        list
            List of datasets
        """
        return self.datasets

    @property
    def groups(self):
        """
        Groups available

        Returns
        -------
        groups : list
            List of groups
        """
        groups = []
        for name in self.h5:
            if isinstance(self.h5[name], h5py.Group):
                groups.append(name)

        return groups

    @property
    def shape(self):
        """
        Resource shape (timesteps, sites)
        shape = (len(time_index), len(meta))

        Returns
        -------
        shape : tuple
            Shape of resource variable arrays (timesteps, sites)
        """
        _shape = (self.h5['time_index'].shape[0], self.h5['meta'].shape[0])
        return _shape

    @property
    def meta(self):
        """
        Meta data DataFrame

        Returns
        -------
        meta : pandas.DataFrame
            Resource Meta Data
        """
        if self._meta is None:
            if 'meta' in self.h5:
                self._meta = self._get_meta('meta', slice(None))
            else:
                raise HandlerKeyError("'meta' is not a valid dataset")

        return self._meta

    @property
    def time_index(self):
        """
        DatetimeIndex

        Returns
        -------
        time_index : pandas.DatetimeIndex
            Resource datetime index
        """
        if self._time_index is None:
            if 'time_index' in self.h5:
                self._time_index = self._get_time_index('time_index',
                                                        slice(None))
            else:
                raise HandlerKeyError("'time_index' is not a valid dataset!")

        return self._time_index

    @property
    def global_attrs(self):
        """
        Global (file) attributes

        Returns
        -------
        global_attrs : dict
        """
        return dict(self.h5.attrs)

    @staticmethod
    def df_str_decode(df):
        """Decode a dataframe with byte string columns into ordinary str cols.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with some columns being byte strings.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with str columns instead of byte str columns.
        """
        for col in df:
            if (np.issubdtype(df[col].dtype, np.object_)
                    and isinstance(df[col].values[0], bytes)):
                df[col] = df[col].copy().str.decode('utf-8', 'ignore')

        return df

    def open_dataset(self, ds_name):
        """
        Open resource dataset

        Parameters
        ----------
        ds_name : str
            Dataset name to open

        Returns
        -------
        ds : ResourceDataset
            Handler for open resource dataset
        """
        if ds_name not in self.datasets:
            raise HandlerKeyError('{} not in {}'
                                  .format(ds_name, self.datasets))

        ds = ResourceDataset(self.h5[ds_name], scale_attr=self.SCALE_ATTR,
                             add_attr=self.ADD_ATTR, unscale=self._unscale)

        return ds

    def get_attrs(self, dset=None):
        """
        Get h5 attributes either from file or dataset

        Parameters
        ----------
        dset : str
            Dataset to get attributes for, if None get file (global) attributes

        Returns
        -------
        attrs : dict
            Dataset or file attributes
        """
        if dset is None:
            attrs = dict(self.h5.attrs)
        else:
            attrs = dict(self.h5[dset].attrs)

        return attrs

    def get_dset_properties(self, dset):
        """
        Get dataset properties (shape, dtype, chunks)

        Parameters
        ----------
        dset : str
            Dataset to get scale factor for

        Returns
        -------
        shape : tuple
            Dataset array shape
        dtype : str
            Dataset array dtype
        chunks : tuple
            Dataset chunk size
        """
        ds = self.h5[dset]
        shape, dtype, chunks = ds.shape, ds.dtype, ds.chunks
        if isinstance(chunks, dict):
            chunks = tuple(chunks.get('dims', (None, 100)))

        return shape, dtype, chunks

    def get_scale(self, dset):
        """
        Get dataset scale factor

        Parameters
        ----------
        dset : str
            Dataset to get scale factor for

        Returns
        -------
        float
            Dataset scale factor, used to unscale int values to floats
        """
        return self.h5[dset].attrs.get(self.SCALE_ATTR, 1)

    def get_units(self, dset):
        """
        Get dataset units

        Parameters
        ----------
        dset : str
            Dataset to get units for

        Returns
        -------
        str
            Dataset units, None if not defined
        """
        return self.h5[dset].attrs.get(self.UNIT_ATTR, None)

    def get_meta_arr(self, rec_name, rows=slice(None)):
        """Get a meta array by name (faster than DataFrame extraction).

        Parameters
        ----------
        rec_name : str
            Named record from the meta data to retrieve.
        rows : slice
            Rows of the record to extract.

        Returns
        -------
        arr : np.ndarray
            Extracted array from the meta data record name.
        """
        if 'meta' in self.h5:
            meta_arr = self.h5['meta'][rec_name, rows]
            if self._str_decode and np.issubdtype(meta_arr.dtype, np.bytes_):
                meta_arr = np.char.decode(meta_arr, encoding='utf-8')
        else:
            raise HandlerKeyError("'meta' is not a valid dataset")

        return meta_arr

    def _get_time_index(self, ds, ds_slice):
        """
        Extract and convert time_index to pandas Datetime Index

        Parameters
        ----------
        ds : str
            Dataset to extract time_index from
        ds_slice : int | list | slice
            tuple describing slice of time_index to extract

        Returns
        -------
        time_index : pandas.DatetimeIndex
            Vector of datetime stamps
        """
        ds_slice = parse_slice(ds_slice)
        time_index = self.h5[ds][ds_slice[0]]
        # time_index: np.array
        return pd.to_datetime(time_index.astype(str))

    def _get_meta(self, ds, ds_slice):
        """
        Extract and convert meta to a pandas DataFrame

        Parameters
        ----------
        ds : str
            Dataset to extract meta from
        ds_slice : int | list | slice
            Pandas slicing describing which sites and columns to extract

        Returns
        -------
        meta : pandas.Dataframe
            Dataframe of location meta data
        """
        ds_slice = parse_slice(ds_slice)
        sites = ds_slice[0]
        if isinstance(sites, int):
            sites = slice(sites, sites + 1)

        meta = self.h5[ds][sites]

        if isinstance(sites, slice):
            if sites.stop:
                sites = list(range(*sites.indices(sites.stop)))
            else:
                sites = list(range(len(meta)))

        meta = pd.DataFrame(meta, index=sites)
        if self._str_decode:
            meta = self.df_str_decode(meta)

        if len(ds_slice) == 2:
            meta = meta[ds_slice[1]]

        return meta

    def _get_SAM_df(self, ds_name, site):
        """
        Placeholder for get_SAM_df method that it resource specific

        Parameters
        ----------
        ds_name : str
            'Dataset' name == SAM
        site : int
            Site to extract SAM DataFrame for
        """

    def _get_ds(self, ds_name, ds_slice):
        """
        Extract data from given dataset

        Parameters
        ----------
        ds_name : str
            Variable dataset to be extracted
        ds_slice : int | list | slice
            tuple describing slice of dataset array to extract

        Returns
        -------
        ds : ndarray
            ndarray of variable timeseries data
            If unscale, returned in native units else in scaled units
        """
        if ds_name not in self.datasets:
            raise HandlerKeyError('{} not in {}'
                                  .format(ds_name, self.datasets))

        ds = self.h5[ds_name]
        ds_slice = parse_slice(ds_slice)
        out = ResourceDataset.extract(ds, ds_slice, scale_attr=self.SCALE_ATTR,
                                      add_attr=self.ADD_ATTR,
                                      unscale=self._unscale)

        return out

    def close(self):
        """
        Close h5 instance
        """
        self._h5.close()

    def _preload_SAM(self, project_points, **kwargs):
        """
        Placeholder method to pre-load project_points for SAM

        Parameters
        ----------
        project_points : reV.config.ProjectPoints
            Projects points to be pre-loaded from Resource for SAM
        kwargs : dict
            internal kwargs
        """

    @classmethod
    def preload_SAM(cls, h5_file, project_points, **kwargs):
        """
        Placeholder for classmethod that will pre-load project_points for SAM

        Parameters
        ----------
        h5_file : str
            h5_file to extract resource from
        project_points : reV.config.ProjectPoints
            Projects points to be pre-loaded from Resource for SAM
        kwargs : dict
            kwargs to init resource class
        """


class MultiH5:
    """
    Class to handle multiple h5 file handlers
    """

    def __init__(self, h5_dir, prefix='', suffix='.h5'):
        """
        Parameters
        ----------
        h5_dir : str
            Path to directory containing 5min .h5 files
        prefix : str
            Prefix for resource .h5 files
        suffix : str
            Suffix for resource .h5 files
        """
        self.h5_dir = h5_dir
        self._dset_map = self._map_file_dsets(h5_dir, prefix=prefix,
                                              suffix=suffix)
        self._h5_map = self._map_file_instances(set(self._dset_map.values()))

        self._i = 0

    def __repr__(self):
        msg = ("{} for {}:\n Contains {} files and {} datasets"
               .format(self.__class__.__name__, self.h5_dir,
                       len(self), len(self._dset_map)))
        return msg

    def __len__(self):
        return len(self._h5_map)

    def __getitem__(self, dset):
        if dset in self:
            path = self._dset_map[dset]
            h5 = self._h5_map[path]
            ds = h5[dset]

        return ds

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self.datasets):
            self._i = 0
            raise StopIteration

        dset = self.datasets[self._i]
        self._i += 1

        return dset

    def __contains__(self, dset):
        return dset in self.datasets

    @property
    def attrs(self):
        """
        Global .h5 file attributes sourced from first .h5 file

        Returns
        -------
        attrs : dict
            .h5 file attributes sourced from first .h5 file
        """
        path = self.h5_files[0]
        attrs = dict(self._h5_map[path].attrs)
        return attrs

    @property
    def datasets(self):
        """
        Available datasets

        Returns
        -------
        list
            List of dataset present in .h5 files
        """
        return sorted(self._dset_map)

    @property
    def h5_files(self):
        """
        .h5 files data is being sourced from

        Returns
        -------
        list
            List of .h5 files data is being sourced form
        """
        return sorted(self._h5_map)

    @staticmethod
    def _get_dsets(h5_path):
        """
        Get datasets in given .h5 file

        Parameters
        ----------
        h5_path : str
            Path to .h5 file to get variables for

        Returns
        -------
        unique_dsets : list
            List of unique datasets in .h5 file
        shared_dsets : list
            List of shared datasets in .h5 file
        """
        unique_dsets = []
        shared_dsets = []
        with h5py.File(h5_path, mode='r') as f:
            for dset in f:
                if dset not in ['meta', 'time_index', 'coordinates']:
                    unique_dsets.append(dset)
                else:
                    shared_dsets.append(dset)

        return unique_dsets, shared_dsets

    @staticmethod
    def _map_file_dsets(h5_dir, prefix='', suffix='.h5'):
        """
        Map 5min variables to their .h5 files in given directory

        Parameters
        ----------
        h5_dir : str
            Path to directory containing 5min .h5 files
        prefix : str
            Prefix for resource .h5 files
        suffix : str
            Suffix for resource .h5 files

        Returns
        -------
        dset_map : dict
            Dictionary mapping datasets to file paths
        """
        dset_map = {}
        for file in sorted(os.listdir(h5_dir)):
            if file.startswith(prefix) and file.endswith(suffix):
                path = os.path.join(h5_dir, file)
                unique_dsets, shared_dsets = MultiH5._get_dsets(path)
                for dset in shared_dsets:
                    if dset not in dset_map:
                        dset_map[dset] = path

                for dset in unique_dsets:
                    dset_map[dset] = path

        return dset_map

    @staticmethod
    def _map_file_instances(h5_files):
        """
        Open all .h5 files and map the open h5py instances to the
        associated file paths

        Parameters
        ----------
        h5_files : list
            List of .h5 files to open

        Returns
        -------
        h5_map : dict
            Dictionary mapping file paths to open resource instances
        """
        h5_map = {}
        for f_path in h5_files:
            h5_map[f_path] = h5py.File(f_path, mode='r')

        return h5_map

    def close(self):
        """
        Close all h5py.File instances
        """
        for f in self._h5_map.values():
            f.close()


class MultiFileResource(Resource):
    """
    Class to handle fine spatial resolution resource data stored in
    multiple .h5 files

    See Also
    --------
    resource.Resource : Parent class

    Examples
    --------
    Due to the size of the 2018 NSRDB and 5min WTK, datasets are stored in
    multiple files. MultiFileResource and it's sub-classes allow for
    interaction with all datasets as if they are in a single file.

    MultiFileResource can take a directory containing all files to source
    data from, or a filepath with a wildcard (*) indicating the filename
    format.

    >>> file = '$TESTDATADIR/wtk/wtk_2010_*m.h5'
    >>> with MultiFileResource(file) as res:
    >>>     print(self._h5_files)
    ['$TESTDATADIR/wtk_2010_200m.h5',
     '$TESTDATADIR/wtk_2010_100m.h5']

    >>> file_100m = '$TESTDATADIR/wtk_2010_100m.h5'
    >>> with Resource(file_100m) as res:
    >>>     print(res.datasets)
    ['coordinates', 'meta', 'pressure_100m', 'temperature_100m', 'time_index',
     'winddirection_100m', 'windspeed_100m']

    >>> file_200m = '$TESTDATADIR/wtk_2010_200m.h5'
    >>> with Resource(file_200m) as res:
    >>>     print(res.datasets)
    ['coordinates', 'meta', 'pressure_200m', 'temperature_200m', 'time_index',
     'winddirection_200m', 'windspeed_200m']

    >>> with MultiFileResource(file) as res:
    >>>     print(res.datasets)
    ['coordinates', 'meta', 'pressure_100m', 'pressure_200m',
     'temperature_100m', 'temperature_200m', 'time_index',
     'winddirection_100m', 'winddirection_200m', 'windspeed_100m',
     'windspeed_200m']

    >>> with MultiFileResource(file) as res:
    >>>     wspd = res['windspeed_100m']
    >>>
    >>> wspd
    [[15.13 15.17 15.21 ... 15.3  15.32 15.31]
     [15.09 15.13 15.16 ... 15.26 15.29 15.31]
     [15.09 15.12 15.15 ... 15.24 15.23 15.26]
     ...
     [10.29 11.08 11.51 ... 14.43 14.41 14.19]
     [11.   11.19 11.79 ... 13.27 11.93 11.8 ]
     [12.16 12.44 13.09 ... 11.94 10.88 11.12]]
    """
    PREFIX = ''
    SUFFIX = '.h5'

    def __init__(self, h5_path, unscale=True, str_decode=True):
        """
        Parameters
        ----------
        h5_path : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        """
        self.h5_dir, prefix, suffix = self.multi_args(h5_path)
        if prefix is None:
            prefix = self.PREFIX

        if suffix is None:
            suffix = self.SUFFIX

        self._unscale = unscale
        self._meta = None
        self._time_index = None
        self._str_decode = str_decode
        self._group = None
        # Map variables to their .h5 files
        self._h5 = MultiH5(self.h5_dir, prefix=prefix, suffix=suffix)
        self._h5_files = sorted(list(self._h5._h5_map.keys()))
        self.h5_file = self._h5_files[0]

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self.h5_dir)
        return msg

    @staticmethod
    def multi_args(h5_path):
        """
        Get multi-h5 directory arguments for multi file resource paths.

        Parameters
        ----------
        h5_path : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix

        Returns
        -------
        h5_dir : str
            Directory containing multi-file resource files.
        prefix : str
            File prefix for files in h5_dir.
        suffix : str
            File suffix for files in h5_dir.
        """
        h5_dir = h5_path
        prefix = None
        suffix = None

        if '*' in h5_path:
            h5_dir, fn = os.path.split(h5_path)
            prefix, suffix = fn.split('*')
        elif os.path.isfile(h5_path):
            raise RuntimeError("MultiFileResource cannot handle a single file"
                               " use Resource instead.")

        return h5_dir, prefix, suffix
