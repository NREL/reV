# -*- coding: utf-8 -*-
"""
pytests for resource handlers
"""
import h5py
import json
import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import tempfile
import shutil

from reV.utilities.exceptions import MultiFileExclusionError

from reV import TESTDATADIR
from reV.handlers.exclusions import ExclusionLayers


def check_crs(truth, test):
    """
    Compare crs'
    """
    truth = dict(i.split("=")
                 for i in truth.split(' '))
    truth = pd.DataFrame(truth, index=[0, ])
    truth = truth.apply(pd.to_numeric, errors='ignore')

    test = dict(i.split("=")
                for i in test.split(' '))
    test = pd.DataFrame(test, index=[0, ])
    test = test.apply(pd.to_numeric, errors='ignore')

    cols = list(set(truth.columns) & set(test.columns))
    assert_frame_equal(truth[cols], test[cols],
                       check_dtype=False, check_exact=False)


@pytest.mark.parametrize(('layer', 'ds_slice'), [
    ('ri_padus', None),
    ('ri_padus', (100, 100)),
    ('ri_padus', (slice(None, 100), slice(None, 100))),
    ('ri_padus', (np.random.choice(range(200), 100, replace=False),
                  np.random.choice(range(200), 100, replace=False))),
    ('ri_srtm_slope', None),
    ('ri_srtm_slope', (100, 100)),
    ('ri_srtm_slope', (slice(None, 100), slice(None, 100))),
    ('ri_srtm_slope', (np.random.choice(range(200), 100, replace=False),
                       np.random.choice(range(200), 100, replace=False)))])
def test_extraction(layer, ds_slice):
    """
    Test extraction of Exclusions Layers

    Parameters
    ----------
    layer : str
        Layer to extract
    ds_slice : tuple
        Slices to extract
    """
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')
    with h5py.File(excl_h5, mode='r') as f:
        truth = f[layer][0]
        if ds_slice is not None:
            truth = truth[ds_slice]

    with ExclusionLayers(excl_h5) as f:
        if ds_slice is None:
            test = f[layer]
        else:
            keys = (layer,) + ds_slice
            test = f[keys]

    assert np.allclose(truth, test)


def test_profile():
    """
    Test profile extraction
    """
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')
    with h5py.File(excl_h5, mode='r') as f:
        truth = json.loads(f.attrs['profile'])

    with ExclusionLayers(excl_h5) as excl:
        test = excl.profile

        assert truth['transform'] == test['transform']
        check_crs(truth['crs'], test['crs'])
        for layer in excl.layers:
            test = excl.get_layer_profile(layer)
            if test is not None:
                assert truth['transform'] == test['transform']
                check_crs(truth['crs'], test['crs'])


def test_crs():
    """
    Test crs extraction
    """
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')
    with h5py.File(excl_h5, mode='r') as f:
        truth = json.loads(f.attrs['profile'])['crs']

    with ExclusionLayers(excl_h5) as excl:
        test = excl.crs

        check_crs(truth, test)
        for layer in excl.layers:
            test = excl.get_layer_crs(layer)
            if test is not None:
                check_crs(truth, test)


def test_shape():
    """
    Test shape attr extraction
    """
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')
    with h5py.File(excl_h5, mode='r') as f:
        truth = f.attrs['shape']

    with ExclusionLayers(excl_h5) as excl:
        test = excl.shape

        assert np.allclose(truth, test)


@pytest.mark.parametrize(('layer', 'ds_slice'), [
    ('excl_test', None),
    ('excl_test', (100, 100)),
    ('excl_test', (slice(10, 100), slice(40, 50))),
    ('ri_padus', (slice(None, 100), slice(None, 100))),
    ('ri_srtm_slope', (100, 100)),
    ('ri_srtm_slope', (slice(None, 100), slice(None, 100)))])
def test_multi_h5(layer, ds_slice):
    """Test the exclusion handler with multiple source files"""
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')
    with tempfile.TemporaryDirectory() as td:
        excl_temp_1 = os.path.join(td, 'excl1.h5')
        excl_temp_2 = os.path.join(td, 'excl2.h5')
        shutil.copy(excl_h5, excl_temp_1)
        shutil.copy(excl_h5, excl_temp_2)

        with h5py.File(excl_temp_1, 'a') as f:
            shape = f['ri_srtm_slope'].shape
            attrs = dict(f['ri_srtm_slope'].attrs)
            test_dset = 'excl_test'
            data = np.ones(shape) * 0.5
            f.create_dataset(test_dset, shape, data=data)
            for k, v in attrs.items():
                f[test_dset].attrs[k] = v

            # make sure ri_srtm_slope can be pulled from the other file
            del f['ri_srtm_slope']

        fp_temp = excl_temp_1
        if layer == 'ri_srtm_slope':
            fp_temp = excl_temp_2

        with h5py.File(fp_temp, mode='r') as f:
            truth = f[layer][0]
            if ds_slice is not None:
                truth = truth[ds_slice]

        with ExclusionLayers([excl_temp_1, excl_temp_2]) as f:
            if ds_slice is None:
                test = f[layer]
            else:
                keys = (layer,) + ds_slice
                test = f[keys]

        assert np.allclose(truth, test)


def test_multi_h5_bad_shape():
    """Test the exclusion handler with multiple source files and a poorly
    shaped dataset"""
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')
    with tempfile.TemporaryDirectory() as td:
        excl_temp_1 = os.path.join(td, 'excl1.h5')
        excl_temp_2 = os.path.join(td, 'excl2.h5')
        shutil.copy(excl_h5, excl_temp_1)
        shutil.copy(excl_h5, excl_temp_2)

        with h5py.File(excl_temp_2, 'a') as f:
            bad_dset = 'excl_test_bad'
            bad_shape = (1, 100, 300)
            attrs = dict(f['ri_srtm_slope'].attrs)
            data = np.ones(bad_shape)
            f.create_dataset(bad_dset, bad_shape, data=data)
            for k, v in attrs.items():
                f[bad_dset].attrs[k] = v

        with pytest.raises(MultiFileExclusionError):
            ExclusionLayers([excl_temp_1, excl_temp_2])


def test_multi_h5_bad_crs():
    """Test the exclusion handler with multiple source files and one file
    with a bad crs attribute"""
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')
    with tempfile.TemporaryDirectory() as td:
        excl_temp_1 = os.path.join(td, 'excl1.h5')
        excl_temp_2 = os.path.join(td, 'excl2.h5')
        shutil.copy(excl_h5, excl_temp_1)
        shutil.copy(excl_h5, excl_temp_2)

        with h5py.File(excl_temp_2, 'a') as f:
            attrs = dict(f.attrs)
            attrs['profile'] = json.loads(attrs['profile'])
            attrs['profile']['crs'] = 'bad_crs'
            attrs['profile'] = json.dumps(attrs['profile'])
            for k, v in attrs.items():
                f.attrs[k] = v

        with pytest.raises(MultiFileExclusionError):
            ExclusionLayers([excl_temp_1, excl_temp_2])


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
