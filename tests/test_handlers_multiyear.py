# -*- coding: utf-8 -*-
"""
pytests for MultiYear collection and computation
"""
import h5py
import numpy as np
import os
import pytest

from reV.handlers.outputs import Outputs
from reV.handlers.multi_year import MultiYear
from reV.utilities.loggers import init_logger
from reV import TESTDATADIR

TEMP_DIR = os.path.join(TESTDATADIR, 'ri_gen_collect')
H5_DIR = os.path.join(TESTDATADIR, 'gen_out')
YEARS = [2012, 2013]
H5_FILES = [os.path.join(H5_DIR, 'gen_ri_pv_{}_x000.h5'.format(year))
            for year in YEARS]
PURGE_OUT = True

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


def manual_means(h5_files, dset):
    """
    Manually calculate means for dset in .h5 files

    Parameters
    ----------
    h5_files : list
        List of .h5 files to compute means from
    dset : str
        Dataset to collect

    Results
    -------
    arr : ndarray
        Multi-year means
    """
    arr = []
    for file in h5_files:
        with Outputs(file, mode='r') as f:
            arr.append(f[dset])

    return np.mean(arr, axis=0)


def manual_stdev(h5_files, dset):
    """
    Manually calculate standard deviation for dset in .h5 files

    Parameters
    ----------
    h5_files : list
        List of .h5 files to compute means from
    dset : str
        Dataset to collect

    Results
    -------
    arr : ndarray
        Multi-year standard deviation
    """
    arr = []
    for file in h5_files:
        with Outputs(file, mode='r') as f:
            arr.append(f[dset])

    return np.std(arr, axis=0)


def compare_arrays(my_arr, test_arr, desc):
    """
    Compare arrays

    Parameters
    ----------
    my_arr : ndarray
        'Truth' array
    test_arr : ndarray
        Test array
    desc : str
        Description of array being tested
    """
    diff = np.mean(np.abs(my_arr - test_arr))
    msg = "{} differ from actuals by {:.4f}".format(desc, diff)
    assert np.allclose(my_arr, test_arr, atol=1.e-3), msg


@pytest.mark.parametrize('dset', ['cf_profile', 'cf_mean'])
def test_my_collection(dset):
    """
    Collect the desired dset

    Parameters
    ----------
    dset : str
        dset to collect from H5_Files
    """
    init_logger('reV.handlers.multi_year')
    my_out = os.path.join(TEMP_DIR, "{}-MY.h5".format(dset))
    my_dsets = ['meta', ]
    my_dsets.extend(['{}-{}'.format(dset, year) for year in YEARS])
    if 'profile' in dset:
        MultiYear.collect_profiles(my_out, H5_FILES, dset)
        my_dsets.extend(["time_index-{}".format(year) for year in YEARS])
    else:
        MultiYear.collect_means(my_out, H5_FILES, dset)
        my_dsets.extend(["{}-{}".format(dset, val)
                         for val in ['means', 'std']])

    with h5py.File(my_out, 'r') as f:
        out_dsets = list(f)

    msg = "Missing datasets after collection"
    assert np.in1d(my_dsets, out_dsets).all(), msg

    if PURGE_OUT:
        os.remove(my_out)


def test_my_means(dset='cf_mean'):
    """
    Test computation of multi-year means

    Parameters
    ----------
    dset : str
        dset to compute means from
    """
    init_logger('reV.handlers.multi_year')
    my_means = manual_means(H5_FILES, dset)

    my_out = os.path.join(TEMP_DIR, "{}-MY.h5".format(dset))
    with MultiYear(my_out, mode='w') as my:
        my.collect(H5_FILES, dset)
        dset_means = my.means(dset)

    compare_arrays(my_means, dset_means, "Computed Means")

    with MultiYear(my_out, mode='r') as my:
        dset_means = my.means(dset)

    compare_arrays(my_means, dset_means, "Saved Means")

    if PURGE_OUT:
        os.remove(my_out)


def test_update(dset='cf_mean'):
    """
    Test computation of multi-year means

    Parameters
    ----------
    dset : str
        dset to compute means from
    """
    init_logger('reV.handlers.multi_year')

    my_out = os.path.join(TEMP_DIR, "{}-MY.h5".format(dset))
    # Collect 2012 and compute 'means'
    files = H5_FILES[:1]
    MultiYear.collect_means(my_out, files, dset)
    my_means = manual_means(files, dset)
    my_std = manual_stdev(files, dset)
    with MultiYear(my_out, mode='r') as my:
        dset_means = my.means(dset)
        dset_std = my.std(dset)

    compare_arrays(my_means, dset_means, "2012 Means")
    compare_arrays(my_std, dset_std, "2013 STD")

    # Add 2013
    files = H5_FILES
    MultiYear.collect_means(my_out, files, dset)
    my_means = manual_means(files, dset)
    my_std = manual_stdev(files, dset)
    with MultiYear(my_out, mode='r') as my:
        dset_means = my.means(dset)
        dset_std = my.std(dset)

    compare_arrays(my_means, dset_means, "Updated Means")
    compare_arrays(my_std, dset_std, "Updated STD")

    if PURGE_OUT:
        os.remove(my_out)


def test_my_stdev(dset='cf_mean'):
    """
    Test computation of multi-year means

    Parameters
    ----------
    dset : str
        dset to compute means from
    """
    init_logger('reV.handlers.multi_year')
    my_std = manual_stdev(H5_FILES, dset)

    my_out = os.path.join(TEMP_DIR, "{}-MY.h5".format(dset))
    with MultiYear(my_out, mode='w') as my:
        my.collect(H5_FILES, dset)
        dset_std = my.std(dset)

    compare_arrays(my_std, dset_std, "Computed STD")

    with MultiYear(my_out, mode='r') as my:
        dset_std = my.std(dset)

    compare_arrays(my_std, dset_std, "Saved STD")

    if PURGE_OUT:
        os.remove(my_out)
