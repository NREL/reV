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
