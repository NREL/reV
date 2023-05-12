# -*- coding: utf-8 -*-
"""
pytests for MultiYear collection and computation
"""
import h5py
import numpy as np
import os
import shutil
import pytest
import tempfile
import json
import traceback

from reV.handlers.cli_multi_year import main
from reV.handlers.outputs import Outputs
from reV.handlers.multi_year import MultiYear
from reV.config.multi_year import MultiYearConfig
from reV import TESTDATADIR

from rex import Resource
from rex.utilities.loggers import init_logger

H5_DIR = os.path.join(TESTDATADIR, 'gen_out')
YEARS = [2012, 2013]
H5_FILES = [os.path.join(H5_DIR, 'gen_ri_pv_{}_x000.h5'.format(year))
            for year in YEARS]
H5_PATTERN = os.path.join(H5_DIR, 'gen_ri_pv_201*_x000.h5')

logger = init_logger('reV.handlers.multi_year', log_level='DEBUG')


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


@pytest.mark.parametrize(('source', 'dset', 'group'), [
    (H5_FILES, 'cf_profile', None),
    (H5_FILES, 'cf_mean', None),
    (H5_FILES, 'cf_profile', 'pytest'),
    (H5_FILES, 'cf_mean', 'pytest'),
    (H5_PATTERN, 'cf_profile', None),
    (H5_PATTERN, 'cf_mean', None),
    (H5_PATTERN, 'cf_profile', 'pytest'),
    (H5_PATTERN, 'cf_mean', 'pytest'),
])
def test_my_collection(source, dset, group):
    """
    Collect the desired dset

    Parameters
    ----------
    source : list | str
        h5 source files, either an explicit list or a file*pattern
    dset : str
        dset to collect from H5_Files
    group : str | NoneType
        group to collect datasets into
    """
    with tempfile.TemporaryDirectory() as temp:
        my_out = os.path.join(temp, "{}-MY.h5".format(dset))
        my_dsets = ['meta', ]
        my_dsets.extend(['{}-{}'.format(dset, year) for year in YEARS])
        if 'profile' in dset:
            MultiYear.collect_profiles(my_out, source, dset, group=group)
            my_dsets.extend(["time_index-{}".format(year) for year in YEARS])
        else:
            MultiYear.collect_means(my_out, source, dset, group=group)
            my_dsets.extend(["{}-{}".format(dset, val)
                             for val in ['means', 'stdev']])

        if group is not None:
            my_dsets = ['{}/{}'.format(group, ds) for ds in my_dsets]

        with Outputs(my_out, mode='r') as f:
            out_dsets = f.datasets

        msg = "Missing datasets after collection"
        assert np.in1d(my_dsets, out_dsets).all(), msg


def test_cli(runner, clear_loggers):
    """Test multi year collection cli with pass through of some datasets."""

    with tempfile.TemporaryDirectory() as temp:
        config = {"log_directory": temp,
                  "execution_control": {"option": "local"},
                  "groups": {"none": {"dsets": ["cf_mean"],
                                      "pass_through_dsets": ['pass_through_1',
                                                             'pass_through_2'],
                                      "source_dir": temp,
                                      "source_prefix": "gen_ri_pv"}},
                  "log_level": "INFO"}

        dirname = os.path.basename(temp)
        fn = "{}_{}.h5".format(dirname, MultiYearConfig.NAME)
        my_out = os.path.join(temp, fn)
        temp_h5_files = [os.path.join(temp, os.path.basename(fp))
                         for fp in H5_FILES]
        for fp, fp_temp in zip(H5_FILES, temp_h5_files):
            shutil.copy(fp, fp_temp)

        pass_through_dsets = config['groups']['none']['pass_through_dsets']
        for fp in temp_h5_files:
            for i, dset in enumerate(pass_through_dsets):
                with h5py.File(fp, 'a') as f:
                    shape = f['meta'].shape
                    arr = np.arange(shape[0]) * (i + 1)
                    f.create_dataset(dset, shape, data=arr)

        fp_config = os.path.join(temp, 'config.json')
        with open(fp_config, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config', '-c', fp_config])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        with Resource(my_out) as res:
            assert 'cf_mean-2012' in res.dsets
            assert 'cf_mean-2013' in res.dsets
            assert 'cf_mean-means' in res.dsets
            assert 'cf_mean-stdev' in res.dsets
            assert 'pass_through_1' in res.dsets
            assert 'pass_through_2' in res.dsets
            assert 'pass_through_1-means' not in res.dsets
            assert 'pass_through_2-means' not in res.dsets
            assert np.allclose(res['pass_through_1'],
                               1 * np.arange(len(res.meta)))
            assert np.allclose(res['pass_through_2'],
                               2 * np.arange(len(res.meta)))

        clear_loggers()


@pytest.mark.parametrize(('dset', 'group'), [
    ('cf_mean', None),
    ('cf_mean', 'pytest')])
def test_my_means(dset, group):
    """
    Test computation of multi-year means

    Parameters
    ----------
    dset : str
        dset to compute means from
    group : str | NoneType
        group to collect datasets into
    """
    my_means = manual_means(H5_FILES, dset)

    with tempfile.TemporaryDirectory() as temp:
        my_out = os.path.join(temp, "{}-MY.h5".format(dset))
        with MultiYear(my_out, mode='w', group=group) as my:
            my.collect(H5_FILES, dset)
            dset_means = my.means(dset)

        compare_arrays(my_means, dset_means, "Computed Means")

        with MultiYear(my_out, mode='r', group=group) as my:
            dset_means = my.means(dset)

        compare_arrays(my_means, dset_means, "Saved Means")


@pytest.mark.parametrize(('dset', 'group'), [
    ('cf_mean', None),
    ('cf_mean', 'pytest')])
def test_update(dset, group):
    """
    Test computation of multi-year means

    Parameters
    ----------
    dset : str
        dset to compute means from
    group : str | NoneType
        group to collect datasets into
    """
    with tempfile.TemporaryDirectory() as temp:
        my_out = os.path.join(temp, "{}-MY.h5".format(dset))
        # Collect 2012 and compute 'means'
        files = H5_FILES[:1]
        MultiYear.collect_means(my_out, files, dset, group=group)
        my_means = manual_means(files, dset)
        my_std = manual_stdev(files, dset)
        with MultiYear(my_out, mode='r', group=group) as my:
            dset_means = my.means(dset)
            dset_std = my.stdev(dset)

        compare_arrays(my_means, dset_means, "2012 Means")
        compare_arrays(my_std, dset_std, "2012 STDEV")

        # Add 2013
        files = H5_FILES
        MultiYear.collect_means(my_out, files, dset, group=group)
        my_means = manual_means(files, dset)
        my_std = manual_stdev(files, dset)
        with MultiYear(my_out, mode='r', group=group) as my:
            dset_means = my.means(dset)
            dset_std = my.stdev(dset)

        compare_arrays(my_means, dset_means, "Updated Means")
        compare_arrays(my_std, dset_std, "Updated STDEV")


@pytest.mark.parametrize(('dset', 'group'), [
    ('cf_mean', None),
    ('cf_mean', 'pytest')])
def test_my_stdev(dset, group):
    """
    Test computation of multi-year means

    Parameters
    ----------
    dset : str
        dset to compute means from
    group : str | NoneType
        group to collect datasets into
    """
    my_std = manual_stdev(H5_FILES, dset)

    with tempfile.TemporaryDirectory() as temp:
        my_out = os.path.join(temp, "{}-MY.h5".format(dset))
        with MultiYear(my_out, mode='w', group=group) as my:
            my.collect(H5_FILES, dset)
            dset_std = my.stdev(dset)

        compare_arrays(my_std, dset_std, "Computed STDEV")

        with MultiYear(my_out, mode='r', group=group) as my:
            dset_std = my.stdev(dset)

        compare_arrays(my_std, dset_std, "Saved STDEV")


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
