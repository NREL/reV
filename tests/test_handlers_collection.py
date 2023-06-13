# -*- coding: utf-8 -*-
"""
pytests for output collection
"""
from glob import glob
import h5py
import numpy as np
import os
import pytest
import tempfile
import json
import traceback
import shutil

from reV.cli import main
from reV.handlers.collection import Collector
from reV import TESTDATADIR
from reV.utilities.pytest_utils import make_fake_h5_chunks

from rex import Resource
from rex.utilities.loggers import init_logger

TEMP_DIR = os.path.join(TESTDATADIR, 'ri_gen_collect')
H5_DIR = os.path.join(TESTDATADIR, 'gen_out')
H5_PATTERN_1 = os.path.join(H5_DIR, 'peregrine_2012*.h5')
H5_PATTERN_2 = os.path.join(H5_DIR, 'pv_gen_2018*.h5')
POINTS_PATH = os.path.join(TESTDATADIR, 'config', 'project_points_100.csv')


def manual_collect(collect_pattern, dset):
    """
    Manually collect dset from .h5 files in h5_dir

    Parameters
    ----------
    collect_pattern : str
        /Filepath/pattern*.h5 to collect
    dset : str
        Dataset to collect

    Results
    -------
    arr : ndarray
        Collected dataset array
    """
    h5_files = Collector.find_h5_files(collect_pattern)
    arr = []
    for file in h5_files:
        with h5py.File(file, 'r') as f:
            arr.append(f[dset][...])

    return np.hstack(arr)


def test_unordered_collection():
    """
    Test collection of multiple dsets from chunks with unordered gids
    """

    with tempfile.TemporaryDirectory() as TEMP_DIR:
        init_logger('reV.handlers.collection')
        features = ['cf_profile', 'ac']
        out = make_fake_h5_chunks(TEMP_DIR, features, shuffle=True)
        out_files = out[0]

        for feature in features:
            profiles = manual_collect(out_files, feature)
            h5_file = os.path.join(TEMP_DIR, 'collection.h5')
            Collector.collect(h5_file, out_files, None, feature, dset_out=None)
            with h5py.File(h5_file, 'r') as f:
                cf_profiles = f[feature][...]

            diff = np.mean(np.abs(profiles - cf_profiles))
            msg = "Arrays differ by {:.4f}".format(diff)
            assert np.allclose(profiles, cf_profiles), msg


def test_collection():
    """
    Test collection on 'cf_profile' ensuring output array is correct
    """
    with tempfile.TemporaryDirectory() as TEMP_DIR:
        init_logger('reV.handlers.collection')
        profiles = manual_collect(H5_PATTERN_1, 'cf_profile')
        h5_file = os.path.join(TEMP_DIR, 'collection.h5')
        Collector.collect(h5_file, H5_PATTERN_1, POINTS_PATH, 'cf_profile',
                          dset_out=None)
        with h5py.File(h5_file, 'r') as f:
            cf_profiles = f['cf_profile'][...]

        diff = np.mean(np.abs(profiles - cf_profiles))
        msg = "Arrays differ by {:.4f}".format(diff)
        assert np.allclose(profiles, cf_profiles), msg

        source_file = os.path.join(H5_DIR, "peregrine_2012_node00_x000.h5")
        with h5py.File(source_file, 'r') as f_s:
            def check_attrs(name, object):
                object_s = f_s[name]
                for k, v in object.attrs.items():
                    v_s = object_s.attrs[k]
                    assert v == v_s

            with h5py.File(h5_file, 'r') as f:
                f.visititems(check_attrs)


def test_collect_means():
    """
    Test means collection:
    """
    with tempfile.TemporaryDirectory() as TEMP_DIR:
        init_logger('reV.handlers.collection')
        h5_file = os.path.join(TEMP_DIR, 'cf_means.h5')
        Collector.collect(h5_file, H5_PATTERN_1, POINTS_PATH, 'cf_mean',
                          dset_out=None)


def test_profiles_means():
    """
    Test adding means to pre-collected profiles
    """
    with tempfile.TemporaryDirectory() as TEMP_DIR:
        init_logger('reV.handlers.collection')
        h5_file = os.path.join(TEMP_DIR, 'cf.h5')
        Collector.collect(h5_file, H5_PATTERN_1, POINTS_PATH, 'cf_profile',
                          dset_out=None)
        Collector.add_dataset(h5_file, H5_PATTERN_1, 'cf_mean', dset_out=None)

        with h5py.File(h5_file, 'r') as f:
            assert 'cf_profile' in f
            assert 'cf_mean' in f
            data = f['cf_profile'][...]

        node_file = os.path.join(H5_DIR, 'peregrine_2012_node01_x001.h5')
        with h5py.File(node_file, 'r') as f:
            source_data = f['cf_profile'][...]

        assert np.allclose(source_data, data[:, -1 * source_data.shape[1]:])


def test_low_mem_collect():
    """Test memory limited multi chunk collection"""
    with tempfile.TemporaryDirectory() as TEMP_DIR:
        init_logger('reV.handlers.collection', log_level='DEBUG')
        h5_file = os.path.join(TEMP_DIR, 'cf.h5')
        Collector.collect(h5_file, H5_PATTERN_1, POINTS_PATH, 'cf_profile',
                          dset_out=None,
                          mem_util_lim=0.00002)

        with h5py.File(h5_file, 'r') as f:
            assert 'cf_profile' in f
            data = f['cf_profile'][...]

        node_file = os.path.join(H5_DIR, 'peregrine_2012_node01_x001.h5')
        with h5py.File(node_file, 'r') as f:
            source_data = f['cf_profile'][...]

        assert np.allclose(source_data, data[:, -1 * source_data.shape[1]:])


def test_means_lcoe():
    """
    Test adding means to pre-collected profiles
    """
    with tempfile.TemporaryDirectory() as TEMP_DIR:
        init_logger('reV.handlers.collection')
        h5_file = os.path.join(TEMP_DIR, 'cf_lcoe.h5')
        Collector.collect(h5_file, H5_PATTERN_1, POINTS_PATH, 'cf_mean',
                          dset_out=None)
        Collector.add_dataset(h5_file, H5_PATTERN_1, 'lcoe_fcr', dset_out=None)

        with h5py.File(h5_file, 'r') as f:
            assert 'cf_mean' in f
            assert 'lcoe_fcr' in f


def test_collect_zero_gid():
    """Test memory limited multi chunk collection"""
    with tempfile.TemporaryDirectory() as TEMP_DIR:
        init_logger('reV.handlers.collection', log_level='DEBUG')
        h5_file = os.path.join(TEMP_DIR, 'cf.h5')
        node_file = os.path.join(H5_DIR, 'peregrine_2012_node00_x000.h5')
        Collector.collect(h5_file, node_file, [0], 'cf_profile', dset_out=None)

        with h5py.File(h5_file, 'r') as f:
            assert 'cf_profile' in f
            data = f['cf_profile'][...]

        with h5py.File(node_file, 'r') as f:
            source_data = f['cf_profile'][...]

        assert np.allclose(source_data[:, 0], data[:, 0])
        assert not np.allclose(source_data[:, 1:], data[:, 1:])


def test_cli(runner, clear_loggers):
    """Test the collection command line interface"""

    with tempfile.TemporaryDirectory() as td:
        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local"
            },
            "log_level": "DEBUG",
            "dsets": [
                "cf_profile"
            ],
            "project_points": None,
            "collect_pattern": H5_PATTERN_1,
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['collect', '-c', config_path])

        if result.exit_code != 0:
            print('Collect cli failed')
            print('Temp dir files: ', os.listdir(td))
            log_file = os.path.join(td, 'collect.log')
            with open(log_file, 'r') as f:
                print(f.read())
            log_file = os.path.join(td, 'collect_peregrine_2012.log')
            with open(log_file, 'r') as f:
                print(f.read())
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)

        for fn in os.listdir(os.path.join(H5_DIR, 'chunk_files')):
            shutil.move(os.path.join(H5_DIR, 'chunk_files/', fn),
                        os.path.join(H5_DIR, fn))
        shutil.rmtree(os.path.join(H5_DIR, 'chunk_files/'))
        clear_loggers()


def test_collect_duplicates():
    """Test the collection of duplicate gids as in the case with reV-gen
    with a gid_map input."""
    with tempfile.TemporaryDirectory() as td:

        source_fps = sorted(glob(H5_DIR + '/pv_gen_2018*.h5'))
        assert len(source_fps) > 1

        h5_file = os.path.join(td, 'collection.h5')
        Collector.collect(h5_file, H5_PATTERN_2, None, 'cf_profile',
                          dset_out=None)

        with Resource(h5_file) as res:
            test_cf = res['cf_profile']
            test_meta = res.meta

        i0 = 0
        for fp in source_fps:
            with Resource(fp) as res:
                truth_cf = res['cf_profile']
                truth_meta = res.meta

            collect_slice = slice(i0, i0 + len(truth_meta))

            assert np.allclose(test_cf[:, collect_slice], truth_cf)
            for col in ('latitude', 'longitude', 'gid'):
                test_meta_col = test_meta[col].values[collect_slice]
                assert np.allclose(test_meta_col, truth_meta[col].values)

            i0 += len(truth_meta)


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
