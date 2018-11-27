"""
pytests for resource handlers
"""
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from reV.handlers.resource import NSRDB, WTK


@pytest.fixture
def NSRDB_res():
    """
    Init NSRDB resource handler
    """
    return NSRDB('./data/nsrdb/ri_100_nsrdb_2012.h5')


@pytest.fixture
def WTK_res():
    """
    Init WTK resource handler
    """
    return WTK('./data/wtk/ri_100_wtk_2012.h5')


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
    ds = res_cls[ds_name]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == ds_shape
    # single site all time
    ds = res_cls[ds_name, :, 1]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[0],)
    # single time all sites
    ds = res_cls[ds_name, 10]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[1],)
    # single value
    ds = res_cls[ds_name, 10, 10]
    assert isinstance(ds, (int, float))
    # site slice
    ds = res_cls[ds_name, :, 10:20]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[0], 10)
    # time slice
    ds = res_cls[ds_name, 10:20]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (10, ds_shape[1])
    # slice in time and space
    ds = res_cls[ds_name, 100:200, 20:30]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (100, 10)
    # site list
    sites = sorted(np.random.choice(ds_shape[1], 20, replace=False))
    ds = res_cls[ds_name, :, sites]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[0], 20)
    # time list
    times = sorted(np.random.choice(ds_shape[0], 100, replace=False))
    ds = res_cls[ds_name, times]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (100, ds_shape[1])


def check_scale(res_cls, ds_name):
    """
    Test unscaling of variable
    """
    native_value = res_cls[ds_name, 0, 0]
    scaled_value = res_cls._h5[ds_name][0, 0]
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


class TestWTK:
    """
    WTK Resource handler tests
    """
    @staticmethod
    def test_meta(WTK_res):
        """
        test extraction of WTK meta data
        """
        check_meta(WTK_res)
        WTK_res.close()

    @staticmethod
    def test_time_index(WTK_res):
        """
        test extraction of WTK time_index
        """
        check_time_index(WTK_res)
        WTK_res.close()

    @staticmethod
    def test_ds(WTK_res, ds_name='windspeed_100m'):
        """
        test extraction of a variable array
        """
        check_dset(WTK_res, ds_name)
        WTK_res.close()

    @staticmethod
    def test_new_hubheight(WTK_res, ds_name='windspeed_90m'):
        """
        test extraction of a variable array
        """
        check_dset(WTK_res, ds_name)
        WTK_res.close()

    @staticmethod
    def test_unscale_windspeed(WTK_res):
        """
        test unscaling of windspeed values
        """
        check_scale(WTK_res, 'windspeed_100m')
        WTK_res.close()

    @staticmethod
    def test_unscale_pressure(WTK_res):
        """
        test unscaling of pressure values
        """
        check_scale(WTK_res, 'pressure_100m')
        WTK_res.close()

    @staticmethod
    def test_interpolation(WTK_res, h=90):
        """
        test variable interpolation
        """
        for var in WTK_res.heights.keys():
            if 'winddirection' not in var:
                check_interp(WTK_res, var, h)

        WTK_res.close()

    @staticmethod
    def test_extrapolation(WTK_res, h=110):
        """
        test variable interpolation
        """
        for var in ['temperature', 'pressure']:
            check_interp(WTK_res, var, h)

        WTK_res.close()
