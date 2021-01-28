# -*- coding: utf-8 -*-
"""
PyTest file for reV LCOE economies of scale
"""
import h5py
import numpy as np
import pandas as pd
import pytest
import os
import tempfile

from reV.version import __version__
from reV.handlers.outputs import Outputs
from reV.utilities.exceptions import HandlerRuntimeError, HandlerValueError


arr1 = np.ones(100)
arr2 = np.ones((8760, 100))
arr3 = np.ones((8760, 100), dtype=float) * 42.42
meta = pd.DataFrame({'latitude': np.ones(100),
                     'longitude': np.zeros(100)})
time_index = pd.date_range('20210101', '20220101', freq='1h', closed='right')


def test_create():
    """Test simple output file creation"""

    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, 'outputs.h5')

        with Outputs(fp, 'w') as f:
            f.meta = meta
            f.time_index = time_index

        with h5py.File(fp, 'r') as f:
            test_meta = pd.DataFrame(f['meta'][...])
            test_ti = f['time_index'][...]
            assert test_meta.shape == (100, 2)
            assert len(test_ti) == 8760

            assert f.attrs['package'] == 'reV'
            assert f.attrs['version'] == __version__


def test_add_dset():
    """Test the addition of datasets to a pre-existing h5 file"""

    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, 'outputs.h5')

        with Outputs(fp, 'w') as f:
            f.meta = meta
            f.time_index = time_index

        with pytest.raises(HandlerRuntimeError):
            f.add_dataset(fp, 'dset1', arr1, None, int, chunks=None,
                          unscale=True, mode='a', str_decode=True,
                          group=None)

        with pytest.raises(HandlerRuntimeError):
            Outputs.add_dataset(fp, 'dset2', arr2, {'scale_factor': 1}, int,
                                chunks=(None, 10), unscale=True, mode='a',
                                str_decode=True, group=None)

        Outputs.add_dataset(fp, 'dset1', arr1.astype(int), None, int,
                            chunks=None, unscale=True, mode='a',
                            str_decode=True, group=None)

        with h5py.File(fp, 'r') as f:
            assert 'dset1' in f
            data = f['dset1'][...]
            assert data.dtype == int
            assert np.allclose(arr1, data)

        Outputs.add_dataset(fp, 'dset2', arr2.astype(np.int16),
                            {'scale_factor': 1}, np.int16, chunks=(None, 10),
                            unscale=True, mode='a', str_decode=True,
                            group=None)

        with h5py.File(fp, 'r') as f:
            assert 'dset1' in f
            assert 'dset2' in f
            assert f['dset1'].chunks is None
            assert f['dset2'].chunks == (8760, 10)
            assert np.allclose(f['dset2'][...], arr2)

        Outputs.add_dataset(fp, 'dset3', arr3, {'scale_factor': 100}, np.int32,
                            chunks=(100, 25),
                            unscale=True, mode='a', str_decode=True,
                            group=None)

        with h5py.File(fp, 'r') as f:
            assert 'dset1' in f
            assert 'dset2' in f
            assert 'dset3' in f
            assert f['dset1'].chunks is None
            assert f['dset2'].chunks == (8760, 10)
            assert f['dset3'].chunks == (100, 25)
            assert f['dset3'].attrs['scale_factor'] == 100
            assert f['dset3'].dtype == np.int32
            assert np.allclose(f['dset3'][...], arr3 * 100)


def test_bad_shape():
    """Negative test for bad data shapes"""

    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, 'outputs.h5')

        with Outputs(fp, 'w') as f:
            f.meta = meta
            f.time_index = time_index

        with pytest.raises(HandlerValueError):
            Outputs.add_dataset(fp, 'dset3', np.ones(10), None, float)

        with pytest.raises(HandlerValueError):
            Outputs.add_dataset(fp, 'dset3', np.ones((10, 10)), None, float)


def execute_pytest(capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
