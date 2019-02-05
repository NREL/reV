"""
pytests for resource handlers
"""
import h5py
import numpy as np
import os

from reV.handlers.collection import Collector
from reV.utilities.loggers import init_logger
from reV import TESTDATADIR

TEMP_DIR = os.path.join(TESTDATADIR, 'ri_gen_collect')
H5_DIR = os.path.join(TESTDATADIR, 'gen_out')
POINTS_PATH = os.path.join(TESTDATADIR, 'config', 'project_points_100.csv')
PURGE_OUT = True

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


def manual_collect(h5_dir, file_prefix, dset):
    """
    Manually collect dset from .h5 files w/ file_prefix in h5_dir

    Parameters
    ----------
    h5_dir : str
        Directory containing .h5 files to collect from
    file_prefix : str
        File prefix on .h5 file to collect from
    dset : str
        Dataset to collect

    Results
    -------
    arr : ndarray
        Collected dataset array
    """
    h5_files = Collector.find_h5_files(h5_dir, file_prefix)
    arr = []
    for file in h5_files:
        with h5py.File(file, 'r') as f:
            arr.append(f[dset][...])

    return np.hstack(arr)


def test_collection():
    """
    Test collection on 'cf_profile' ensuring output array is correct
    """
    init_logger('reV.handlers.collection')
    profiles = manual_collect(H5_DIR, 'peregrine_2012', 'cf_profile')
    h5_file = os.path.join(TEMP_DIR, 'cf_profiles.h5')
    Collector.collect_profiles(h5_file, H5_DIR, POINTS_PATH, 'cf_profile',
                               dset_out=None,
                               file_prefix='peregrine_2012',
                               parallel=False)
    with h5py.File(h5_file) as f:
        cf_profiles = f['cf_profile'][...]

    assert np.array_equal(profiles, cf_profiles)

    if PURGE_OUT:
        os.remove(h5_file)


def test_parallel_collection():
    """
    Test collection on 'cf_profile' in series and parallel
    """
    init_logger('reV.handlers.collection')
    h5_file = os.path.join(TEMP_DIR, 'cf_profiles.h5')
    Collector.collect_profiles(h5_file, H5_DIR, POINTS_PATH, 'cf_profile',
                               dset_out=None,
                               file_prefix='peregrine_2012',
                               parallel=False)
    with h5py.File(h5_file) as f:
        series_profiles = f['cf_profile'][...]

    Collector.collect_profiles(h5_file, H5_DIR, POINTS_PATH, 'cf_profile',
                               dset_out=None,
                               file_prefix='peregrine_2012',
                               parallel=True)
    with h5py.File(h5_file) as f:
        parallel_profiles = f['cf_profile'][...]

    assert np.array_equal(series_profiles, parallel_profiles)

    if PURGE_OUT:
        os.remove(h5_file)


def test_collect_means():
    """
    Test means collection:
    """
    init_logger('reV.handlers.collection')
    h5_file = os.path.join(TEMP_DIR, 'cf_means.h5')
    Collector.collect_means(h5_file, H5_DIR, POINTS_PATH, 'cf_mean',
                            dset_out=None,
                            file_prefix='peregrine_2012',
                            parallel=False)
    if PURGE_OUT:
        os.remove(h5_file)


def test_profiles_means():
    """
    Test adding means to pre-collected profiles
    """
    init_logger('reV.handlers.collection')
    h5_file = os.path.join(TEMP_DIR, 'cf.h5')
    Collector.collect_profiles(h5_file, H5_DIR, POINTS_PATH, 'cf_profile',
                               dset_out=None,
                               file_prefix='peregrine_2012',
                               parallel=False)
    Collector.add_dataset(h5_file, H5_DIR, 'cf_mean',
                          dset_out=None,
                          file_prefix='peregrine_2012',
                          parallel=False)

    with h5py.File(h5_file, 'r') as f:
        assert 'cf_profile' in f
        assert 'cf_mean' in f

    if PURGE_OUT:
        os.remove(h5_file)


def test_means_lcoe():
    """
    Test adding means to pre-collected profiles
    """
    init_logger('reV.handlers.collection')
    h5_file = os.path.join(TEMP_DIR, 'cf_lcoe.h5')
    Collector.collect_means(h5_file, H5_DIR, POINTS_PATH, 'cf_mean',
                            dset_out=None,
                            file_prefix='peregrine_2012',
                            parallel=False)
    Collector.add_dataset(h5_file, H5_DIR, 'lcoe_fcr',
                          dset_out=None,
                          file_prefix='peregrine_2012',
                          parallel=False)

    with h5py.File(h5_file, 'r') as f:
        assert 'cf_mean' in f
        assert 'lcoe_fcr' in f

    if PURGE_OUT:
        os.remove(h5_file)
