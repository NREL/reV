# -*- coding: utf-8 -*-
"""
pytests for resource handlers
"""
import h5py
import numpy as np
import os
import pytest

from reV import TESTDATADIR
from reV.handlers.exclusions import ExclusionLayers


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
