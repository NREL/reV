# -*- coding: utf-8 -*-
# pylint: skip-file
"""reV SAM unit test module
"""
import os
from pkg_resources import get_distribution
from packaging import version
import pytest
import numpy as np
import warnings

from reV.SAM.defaults import (DefaultPvWattsv5, DefaultPvWattsv8,
                              DefaultWindPower)
from reV.SAM.generation import PvWattsv5, PvWattsv7, PvWattsv8
from reV import TESTDATADIR
from reV.config.project_points import ProjectPoints
from reV.SAM.version_checker import PySamVersionChecker
from reV.utilities.exceptions import PySAMVersionWarning
from reV.utilities.exceptions import InputError

from rex.renewable_resource import NSRDB
from rex.utilities.utilities import pd_date_range


@pytest.fixture
def res():
    """Initialize a SAM resource object to test SAM functions on."""
    res_file = TESTDATADIR + '/nsrdb/ri_100_nsrdb_2012.h5'
    rev2_points = TESTDATADIR + '/project_points/ri.csv'

    sam_files = [TESTDATADIR + '/SAM/naris_pv_1axis_inv13.json',
                 TESTDATADIR + '/SAM/naris_pv_1axis_inv13.json']
    sam_files = {'sam_param_{}'.format(i): k for i, k in
                 enumerate(sam_files)}

    pp = ProjectPoints(rev2_points, sam_files, 'pv')
    res = NSRDB.preload_SAM(res_file, pp.sites)

    return res


def test_res_length(res):
    """Test the method to ensure resource array length with truncation."""
    for res_df, meta in res:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_dropped = PvWattsv5.ensure_res_len(res_df.values, res_df.index)
        break

    compare = np.allclose(res_dropped[:9000, :], res_df.values[:9000, :])


def test_leap_year(res):
    """Test the method to ensure resource array length with dropping leap day.
    """
    for res_df, meta in res:
        res_dropped = PvWattsv5.drop_leap(res_df)
        break

    compare = np.allclose(res_dropped.iloc[-9000:, :].values,
                          res_df.iloc[-9000:, :].values)
    assert compare


def test_leap_year_freq():
    """
    Test ensure_res_len with leap year data
    """
    time_index = pd_date_range('2012', '2013', freq='h', closed='left')
    arr = np.arange(len(time_index) * 10).reshape(len(time_index), 10)
    out = PvWattsv5.ensure_res_len(arr, time_index)
    assert np.allclose(arr[:8760], out)

    mask = time_index.month == 2
    mask &= time_index.day == 29
    time_index = time_index[~mask]
    arr = np.arange(len(time_index) * 10).reshape(len(time_index), 10)
    out = PvWattsv5.ensure_res_len(arr, time_index)
    assert np.allclose(arr, out)


@pytest.mark.parametrize('site_index', range(5))
def test_PV_lat_tilt(res, site_index):
    """Test the method to set tilt based on latitude."""
    rev2_points = TESTDATADIR + '/project_points/ri.csv'
    sam_files = [TESTDATADIR + '/SAM/naris_pv_1axis_inv13.json',
                 TESTDATADIR + '/SAM/naris_pv_1axis_inv13.json']
    sam_files = {'sam_param_{}'.format(i): k for i, k in
                 enumerate(sam_files)}
    pp = ProjectPoints(rev2_points, sam_files, 'pv')

    for i, [res_df, meta] in enumerate(res):
        if i == site_index:
            # get SAM inputs from project_points based on the current site
            site = res_df.name
            config, inputs = pp[site]
            inputs['tilt'] = 'latitude'
            # iterate through requested sites.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sim = PvWattsv5(resource=res_df, meta=meta,
                                sam_sys_inputs=inputs,
                                output_request=('cf_mean',))
            break
        else:
            pass

    assert sim.sam_sys_inputs['tilt'] == meta['latitude']


@pytest.mark.parametrize('dt', ('1h', '30min', '5min'))
def test_time_interval(dt):
    """Test the method to get the 'time interval' from the time index obj."""
    baseline = {'1h': 1, '30min': 2, '5min': 12}
    ti = pd_date_range('1-1-{y}'.format(y=2012), '1-1-{y}'.format(y=2013),
                       freq=dt)[:-1]
    interval = PvWattsv5.get_time_interval(ti)
    assert interval == baseline[dt]


def test_pysam_version_checker_pv():
    """Test that the pysam version checker passes through PV config untouched
    without warning.
    """
    pv_config = {'gcr': 0.4, 'system_capacity': 1}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sam_sys_inputs = PySamVersionChecker.run('pvwattsv5', pv_config)
        assert 'gcr' in sam_sys_inputs
        assert 'system_capacity' in sam_sys_inputs


def test_pysam_version_checker_wind():
    """Check that the pysam version checker recognizes outdated config keys
    from pysam v1 and fixes them and raises warning.
    """
    wind_config = {'wind_farm_losses_percent': 10, 'system_capacity': 1}

    pysam_version = str(get_distribution('nrel-pysam')).split(' ')[1]
    pysam_version = version.parse(pysam_version)

    if pysam_version > version.parse('2.1.0'):
        with pytest.warns(PySAMVersionWarning) as record:
            sam_sys_inputs = PySamVersionChecker.run('windpower', wind_config)

        assert 'old SAM v1 keys' in str(record[0].message)
        assert 'turb_generic_loss' in sam_sys_inputs
        assert 'system_capacity' in sam_sys_inputs
        assert 'wind_farm_losses_percent'


def test_nan_resource():
    """Test that the reV-SAM interface will raise an error if there is NaN
    data in the resource input."""
    res_file = TESTDATADIR + '/nsrdb/ri_100_nsrdb_2012.h5'
    sam_files = TESTDATADIR + '/SAM/naris_pv_1axis_inv13.json'
    pp = ProjectPoints(0, sam_files, 'pv')
    res = NSRDB.preload_SAM(res_file, pp.sites)

    for res_df, meta in res:
        res_df.iloc[10, 0] = np.nan
        site = res_df.name
        _, inputs = pp[site]
        with pytest.raises(InputError):
            PvWattsv7(resource=res_df, meta=meta, sam_sys_inputs=inputs)
        with pytest.raises(InputError):
            PvWattsv8(resource=res_df, meta=meta, sam_sys_inputs=inputs)


def test_default_pvwattsv5():
    """Test default pvwattsv5 execution and compare baseline annual energy"""
    default = DefaultPvWattsv5.default()
    assert round(default.Outputs.annual_energy, -1) == 6830


def test_default_pvwattsv8():
    """Test default pvwattsv8 execution and compare baseline annual energy"""
    default = DefaultPvWattsv8.default()
    assert round(default.Outputs.annual_energy, -1) == 173276030


def test_default_windpower():
    """Test default windpower execution and compare baseline annual energy"""
    default = DefaultWindPower.default()
    assert round(default.Outputs.annual_energy, -1) == 201595970


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
