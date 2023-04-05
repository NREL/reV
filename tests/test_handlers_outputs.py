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
from rex.utilities.utilities import pd_date_range


arr1 = np.ones(100)
arr2 = np.ones((8760, 100))
arr3 = np.ones((8760, 100), dtype=float) * 42.42
meta = pd.DataFrame({'latitude': np.ones(100),
                     'longitude': np.zeros(100)})
time_index = pd_date_range('20210101', '20220101', freq='1h', closed='right')


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
