# -*- coding: utf-8 -*-
"""Functions used for pytests"""

import os

import numpy as np
import pandas as pd
from packaging import version
from rex.outputs import Outputs as RexOutputs

from reV.utilities import ResourceMetaField


def pd_date_range(*args, **kwargs):
    """A simple wrapper on the pd.date_range() method that handles the closed
    vs. inclusive kwarg change in pd 1.4.0"""
    incl = version.parse(pd.__version__) >= version.parse('1.4.0')

    if incl and 'closed' in kwargs:
        kwargs['inclusive'] = kwargs.pop('closed')
    elif not incl and 'inclusive' in kwargs:
        kwargs['closed'] = kwargs.pop('inclusive')
        if kwargs['closed'] == 'both':
            kwargs['closed'] = None

    return pd.date_range(*args, **kwargs)


def write_chunk(meta, times, data, features, out_file):
    """Write data chunk to an h5 file

    Parameters
    ----------
    meta : dict
        Dictionary of meta data for this chunk. Includes flattened lat and lon
        arrays
    times : pd.DatetimeIndex
        times in this chunk
    features : list
        List of feature names in this chunk
    out_file : str
        Name of output file
    """
    with RexOutputs(out_file, 'w') as fh:
        fh.meta = meta
        fh.time_index = times
        for feature in features:
            flat_data = data.reshape((-1, len(times)))
            flat_data = np.transpose(flat_data, (1, 0))
            fh.add_dataset(out_file, feature, flat_data, dtype=np.float32)


def make_fake_h5_chunks(td, features, shuffle=False):
    """Make fake h5 chunks to test collection

    Parameters
    ----------
    td : tempfile.TemporaryDirectory
        Test TemporaryDirectory
    features : list
        List of dsets to write to chunks
    shuffle : bool
        Whether to shuffle gids

    Returns
    -------
    out_pattern : str
        Pattern for output file names
    data : ndarray
        Full non-chunked data array
    features : list
        List of feature names in output
    s_slices : list
        List of spatial slices used to chunk full data array
    times : pd.DatetimeIndex
        Times in output
    """
    shape = (50, 50, 48)
    data = np.random.uniform(0, 20, shape)
    lat = np.linspace(90, 0, 50)
    lon = np.linspace(-180, 0, 50)
    lon, lat = np.meshgrid(lon, lat)
    gids = np.arange(np.product(lat.shape))
    if shuffle:
        np.random.shuffle(gids)
    gids = gids.reshape(shape[:-1])
    times = pd_date_range('20220101', '20220103', freq='3600s',
                          inclusive='left')
    s_slices = [slice(0, 25), slice(25, 50)]
    out_pattern = os.path.join(td, 'chunks_{i}_{j}.h5')

    for i, s1 in enumerate(s_slices):
        for j, s2 in enumerate(s_slices):
            out_file = out_pattern.format(i=i, j=j)
            meta = pd.DataFrame(
                {ResourceMetaField.LATITUDE: lat[s1, s2].flatten(),
                 ResourceMetaField.LONGITUDE: lon[s1, s2].flatten(),
                 ResourceMetaField.GID: gids[s1, s2].flatten()})
            write_chunk(meta=meta, times=times, data=data[s1, s2],
                        features=features, out_file=out_file)

    out = (out_pattern.format(i='*', j='*'), data, features, s_slices, times)
    return out
