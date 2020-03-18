# -*- coding: utf-8 -*-
"""
pytests for SolarPosition calculator
"""
import numpy as np
import os
import pytest

from reV.handlers.rev_resource import NSRDB
from reV.utilities.solar_position import SolarPosition
from reV import TESTDATADIR

NSRDB_DIR = os.path.join(TESTDATADIR, 'sza')


def extract_nsrdb(year, site=None):
    """
    Extract time_index, meta, and sza from NSRDB

    Parameters
    ----------
    year : int
        Year to extract from
    site : int
        Site to extract, if None extract all

    Returns
    -------
    time_index : pandas.DatetimeIndex
        Datetime Stamps
    lat_lon : ndarray
        Site(s) (lat, lon)
    sza : ndarray
        Solar Zenith Angles
    """
    path = os.path.join(NSRDB_DIR, 'nsrdb_sza_{}.h5'.format(year))
    with NSRDB(path) as f:
        time_index = f.time_index
        lat_lon = f.meta[['latitude', 'longitude']].values
        sza = f['solar_zenith_angle']

    if site:
        lat_lon = lat_lon[site]
        sza = sza[:, site]

    return time_index, lat_lon, sza


@pytest.mark.parametrize(('year', 'site'),
                         [(2012, None),
                          (2012, True),
                          (2013, None),
                          (2013, True)])
def test_sza(year, site):
    """
    Compare a solar zenith angles between code and NSRDB

    Parameters
    ----------
    year : int
        NSRDB year to compare with
    site : bool
        If true randomly select a site, else compare all sites
    """
    if site:
        site = np.random.randint(101)
    else:
        site = None

    time_index, lat_lon, nsrdb_sza = extract_nsrdb(year, site=site)

    sza = SolarPosition(time_index, lat_lon).zenith

    assert sza.shape == nsrdb_sza.shape, 'Shapes do not match!'
    msg = 'Zenith angle differ by more than ~1 degree'
    assert np.allclose(sza, nsrdb_sza, rtol=0.01, atol=1), msg


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
