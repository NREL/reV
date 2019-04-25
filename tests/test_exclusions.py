#!/usr/bin/env python
# pylint: skip-file
"""Exclusions unit test module
"""
import os
import pytest
import rasterio
import numpy as np

from reV import TESTDATADIR
from reV.exclusions.exclusions import Exclusions


def test_exclusions_output():
    """Validate exclusions output
    """
    f_path = os.path.join(TESTDATADIR, 'ri_exclusions')

    with rasterio.open(os.path.join(f_path, "exclusions.tif"), 'r') as file:
        valid_exclusions_data = file.read(1)

    layer_configs = [{"fpath": os.path.join(f_path, "ri_srtm_slope.tif"),
                      "max_thresh": 5},
                     {"fpath": os.path.join(f_path, "ri_padus.tif"),
                      "classes_exclude": [1]}]

    exclusions = Exclusions(layer_configs)

    assert np.array_equal(exclusions.data, valid_exclusions_data)


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
