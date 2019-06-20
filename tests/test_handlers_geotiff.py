# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:37:05 2019

@author: gbuster
"""
import pytest
import os
from reV.handlers.geotiff import Geotiff
from reV import TESTDATADIR


def test_geotiff_meta():
    """Test correct un-projection of the geotiff meta."""
    fpath = os.path.join(TESTDATADIR, 'ri_exclusions/exclusions.tif')

    with Geotiff(fpath) as f:
        meta = f.meta
        data = f.get_flat_data(layer=0)

    # assert approximate RI boundaries
    assert meta['latitude'].min() > 40.8
    assert meta['latitude'].max() < 42.1
    assert meta['longitude'].min() > -72.0
    assert meta['longitude'].max() < -70.8
    assert len(meta) == len(data)
    assert len(data.shape) == 1


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
