# -*- coding: utf-8 -*-
"""
pytests for resource handlers
"""
from datetime import datetime
import numpy as np
import os
import pandas as pd
import pytest

from reV import TESTDATADIR
from reV.handlers.resource import NSRDB, WindResource
from reV.utilities.exceptions import HandlerKeyError


@pytest.fixture
def NSRDB_res():
    """
    Init NSRDB resource handler
    """
    path = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_2012.h5')
    return NSRDB(path)


@pytest.fixture
def WindResource_res():
    """
    Init WindResource resource handler
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    return WindResource(path)


@pytest.fixture
def wind_group():
    """
    Init WindResource resource handler
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012_group.h5')
    return WindResource(path, group='group')


def check_res(res_cls):
    """
    Run test on len and shape methods
    """
    time_index = res_cls.time_index
    meta = res_cls.meta
    res_shape = (len(time_index), len(meta))

    assert len(res_cls) == len(meta)
    assert res_cls.shape == res_shape


def check_meta(res_cls):
    """
    Run tests on meta data
    """
    meta = res_cls['meta']
    assert isinstance(meta, pd.DataFrame)
    meta_shape = meta.shape
    # single site
    meta = res_cls['meta', 50]
    assert isinstance(meta, pd.DataFrame)
    assert meta.shape == (1, meta_shape[1])
    # site slice
    meta = res_cls['meta', :10]
    assert isinstance(meta, pd.DataFrame)
    assert meta.shape == (10, meta_shape[1])
    # site list
    sites = sorted(np.random.choice(meta_shape[0], 20, replace=False))
    meta = res_cls['meta', sites]
    assert isinstance(meta, pd.DataFrame)
    assert meta.shape == (len(sites), meta_shape[1])
    # select columns
    meta = res_cls['meta', :, ['latitude', 'longitude']]
    assert isinstance(meta, pd.DataFrame)
    assert meta.shape == (meta_shape[0], 2)


def check_time_index(res_cls):
    """
    Run tests on time_index
    """
    time_index = res_cls['time_index']
    time_shape = time_index.shape
    assert isinstance(time_index, pd.DatetimeIndex)
    # single timestep
    time_index = res_cls['time_index', 50]
    assert isinstance(time_index, datetime)
    # time slice
    time_index = res_cls['time_index', 100:200]
    assert isinstance(time_index, pd.DatetimeIndex)
    assert time_index.shape == (100,)
    # list of timesteps
    steps = sorted(np.random.choice(time_shape[0], 50, replace=False))
    time_index = res_cls['time_index', steps]
    assert isinstance(time_index, pd.DatetimeIndex)
    assert time_index.shape == (50,)


def check_dset(res_cls, ds_name):
    """
    Run tests on dataset ds_name
    """
    time_index = res_cls['time_index']
    meta = res_cls['meta']
    ds_shape = (len(time_index), len(meta))
    arr = res_cls[ds_name]
    ds = res_cls[ds_name]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == ds_shape
    assert np.allclose(arr, ds)
    # single site all time
    ds = res_cls[ds_name, :, 1]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[0],)
    # single time all sites
    ds = res_cls[ds_name, 10]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[1],)
    assert np.allclose(arr[10], ds)
    # single value
    ds = res_cls[ds_name, 10, 10]
    assert isinstance(ds, (np.integer, np.floating))
    assert np.allclose(arr[10, 10], ds)
    # site slice
    ds = res_cls[ds_name, :, 10:20]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[0], 10)
    assert np.allclose(arr[:, 10:20], ds)
    # time slice
    ds = res_cls[ds_name, 10:20]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (10, ds_shape[1])
    assert np.allclose(arr[10:20], ds)
    # slice in time and space
    ds = res_cls[ds_name, 100:200, 20:30]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (100, 10)
    assert np.allclose(arr[100:200, 20:30], ds)
    # site list
    sites = sorted(np.random.choice(ds_shape[1], 20, replace=False))
    ds = res_cls[ds_name, :, sites]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[0], 20)
    assert np.allclose(arr[:, sites], ds)
    # site list single time
    ds = res_cls[ds_name, 0, sites]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (20,)
    assert np.allclose(arr[0, sites], ds)
    # time list
    times = sorted(np.random.choice(ds_shape[0], 100, replace=False))
    ds = res_cls[ds_name, times]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (100, ds_shape[1])
    assert np.allclose(arr[times], ds)
    # time list single site
    ds = res_cls[ds_name, times, 0]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (100,)
    assert np.allclose(arr[times, 0], ds)
    # time and site lists
    ds = res_cls[ds_name, times[:20], sites]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (20,)
    assert np.allclose(arr[times[:20], sites], ds)
    with pytest.raises(IndexError):
        assert res_cls[ds_name, times, sites]


def check_scale(res_cls, ds_name):
    """
    Test unscaling of variable
    """
    native_value = res_cls[ds_name, 0, 0]
    scaled_value = res_cls.h5[ds_name][0, 0]
    scale_factor = res_cls.get_scale(ds_name)
    if scale_factor != 1:
        assert native_value != scaled_value

    assert native_value == (scaled_value / scale_factor)


def check_interp(res_cls, var, h):
    """
    Test linear interpolation of Wind variables
    """
    ds_name = '{}_{}m'.format(var, h)
    ds_value = res_cls[ds_name, 0, 0]

    (h1, h2), _ = res_cls.get_nearest_h(h, res_cls.heights[var])

    ds_name = '{}_{}m'.format(var, h1)
    h1_value = res_cls[ds_name, 0, 0]
    ds_name = '{}_{}m'.format(var, h2)
    h2_value = res_cls[ds_name, 0, 0]
    interp_value = (h2_value - h1_value) / (h2 - h1) * (h - h1) + h1_value

    assert ds_value == interp_value


class TestNSRDB:
    """
    NSRDB Resource handler tests
    """
    @staticmethod
    def test_res(NSRDB_res):
        """
        test NSRDB class calls
        """
        check_res(NSRDB_res)
        NSRDB_res.close()

    @staticmethod
    def test_meta(NSRDB_res):
        """
        test extraction of NSRDB meta data
        """
        check_meta(NSRDB_res)
        NSRDB_res.close()

    @staticmethod
    def test_time_index(NSRDB_res):
        """
        test extraction of NSRDB time_index
        """
        check_time_index(NSRDB_res)
        NSRDB_res.close()

    @staticmethod
    def test_ds(NSRDB_res, ds_name='dni'):
        """
        test extraction of a variable array
        """
        check_dset(NSRDB_res, ds_name)
        NSRDB_res.close()

    @staticmethod
    def test_unscale_dni(NSRDB_res):
        """
        test unscaling of dni values
        """
        check_scale(NSRDB_res, 'dni')
        NSRDB_res.close()

    @staticmethod
    def test_unscale_pressure(NSRDB_res):
        """
        test unscaling of pressure values
        """
        check_scale(NSRDB_res, 'surface_pressure')
        NSRDB_res.close()

    # @staticmethod
    # def test_units(NSRDB_res):
    #     """
    #     test unit attributes
    #     """
    #     assert NSRDB_res.get_units('dni') == 'W/m2'
    #     assert NSRDB_res.get_units('wind_speed') == 'm/s'
    #     NSRDB_res.close()


class TestWindResource:
    """
    WindResource Resource handler tests
    """
    @staticmethod
    def test_res(WindResource_res):
        """
        test WindResource class calls
        """
        check_res(WindResource_res)
        WindResource_res.close()

    @staticmethod
    def test_meta(WindResource_res):
        """
        test extraction of WindResource meta data
        """
        check_meta(WindResource_res)
        WindResource_res.close()

    @staticmethod
    def test_time_index(WindResource_res):
        """
        test extraction of WindResource time_index
        """
        check_time_index(WindResource_res)
        WindResource_res.close()

    @staticmethod
    def test_ds(WindResource_res, ds_name='windspeed_100m'):
        """
        test extraction of a variable array
        """
        check_dset(WindResource_res, ds_name)
        WindResource_res.close()

    @staticmethod
    def test_new_hubheight(WindResource_res, ds_name='windspeed_90m'):
        """
        test extraction of a variable array
        """
        check_dset(WindResource_res, ds_name)
        WindResource_res.close()

    @staticmethod
    def test_unscale_windspeed(WindResource_res):
        """
        test unscaling of windspeed values
        """
        check_scale(WindResource_res, 'windspeed_100m')
        WindResource_res.close()

    @staticmethod
    def test_unscale_pressure(WindResource_res):
        """
        test unscaling of pressure values
        """
        check_scale(WindResource_res, 'pressure_100m')
        WindResource_res.close()

    @staticmethod
    def test_interpolation(WindResource_res, h=90):
        """
        test variable interpolation
        """
        ignore = ['winddirection', 'precipitationrate', 'relativehumidity']
        for var in WindResource_res.heights.keys():
            if var not in ignore:
                check_interp(WindResource_res, var, h)

        WindResource_res.close()

    @staticmethod
    def test_extrapolation(WindResource_res, h=110):
        """
        test variable interpolation
        """
        for var in ['temperature', 'pressure']:
            check_interp(WindResource_res, var, h)

        WindResource_res.close()

    # @staticmethod
    # def test_units(WindResource_res):
    #     """
    #     test unit attributes
    #     """
    #     assert WindResource_res.get_units('windspeed_100m') == 'm s-1'
    #     assert WindResource_res.get_units('temperature_100m') == 'C'
    #     WindResource_res.close()


class TestGroupResource:
    """
    WindResource Resource handler tests
    """
    @staticmethod
    def test_group():
        """
        test WindResource class calls
        """
        path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012_group.h5')
        with pytest.raises(HandlerKeyError):
            with WindResource(path) as res:
                check_res(res)

    @staticmethod
    def test_res(wind_group):
        """
        test WindResource class calls
        """
        check_res(wind_group)
        wind_group.close()

    @staticmethod
    def test_meta(wind_group):
        """
        test extraction of WindResource meta data
        """
        check_meta(wind_group)
        wind_group.close()

    @staticmethod
    def test_time_index(wind_group):
        """
        test extraction of WindResource time_index
        """
        check_time_index(wind_group)
        wind_group.close()

    @staticmethod
    def test_ds(wind_group, ds_name='windspeed_100m'):
        """
        test extraction of a variable array
        """
        check_dset(wind_group, ds_name)
        wind_group.close()

    @staticmethod
    def test_new_hubheight(wind_group, ds_name='windspeed_90m'):
        """
        test extraction of a variable array
        """
        check_dset(wind_group, ds_name)
        wind_group.close()

    @staticmethod
    def test_unscale_windspeed(wind_group):
        """
        test unscaling of windspeed values
        """
        check_scale(wind_group, 'windspeed_100m')
        wind_group.close()

    @staticmethod
    def test_unscale_pressure(wind_group):
        """
        test unscaling of pressure values
        """
        check_scale(wind_group, 'pressure_100m')
        wind_group.close()

    @staticmethod
    def test_interpolation(wind_group, h=90):
        """
        test variable interpolation
        """
        ignore = ['winddirection', 'precipitationrate', 'relativehumidity']
        for var in wind_group.heights.keys():
            if var not in ignore:
                check_interp(wind_group, var, h)

        wind_group.close()

    @staticmethod
    def test_extrapolation(wind_group, h=110):
        """
        test variable interpolation
        """
        for var in ['temperature', 'pressure']:
            check_interp(wind_group, var, h)

        wind_group.close()

    # @staticmethod
    # def test_units(wind_group):
    #     """
    #     test unit attributes
    #     """
    #     assert wind_group.get_units('windspeed_100m') == 'm s-1'
    #     assert wind_group.get_units('temperature_100m') == 'C'
    #     wind_group.close()


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
