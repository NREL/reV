# -*- coding: utf-8 -*-
# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""
import json
import os
import shutil
import tempfile
import traceback

import numpy as np
import pandas as pd
import pytest
from rex.utilities.utilities import safe_json_load

from reV import TESTDATADIR
from reV.cli import main
from reV.config.project_points import ProjectPoints
from reV.generation.base import LCOE_REQUIRED_OUTPUTS
from reV.generation.generation import Gen
from reV.handlers.outputs import Outputs
from reV.utilities import SiteDataField

RTOL = 0.0
ATOL = 0.04


def get_r1_profiles(year=2012, tech='pv'):
    """Get the first 100 reV 1.0 ri generation profiles."""

    if tech == 'pv':
        rev1 = os.path.join(TESTDATADIR, 'ri_pv', 'profile_outputs',
                            'pv_{}_0.h5'.format(year))
    elif tech == 'wind':
        rev1 = os.path.join(TESTDATADIR, 'ri_wind', 'profile_outputs',
                            'wind_{}_0.h5'.format(year))

    with Outputs(rev1) as cf:
        data = cf['cf_profile'][...] / 10000

    return data


@pytest.mark.parametrize('tech', ['pv', 'wind'])
def test_gen_from_config(runner, tech, clear_loggers):
    """Gen PV CF profiles with write to disk and compare against rev1."""
    with tempfile.TemporaryDirectory() as td:

        run_dir = os.path.join(td, 'generation')
        os.mkdir(run_dir)

        if tech == 'pv':
            fconfig = 'local_pv.json'
            project_points = os.path.join(TESTDATADIR, 'config', '..',
                                          'config', "project_points_10.csv")
            resource_file = os.path.join(TESTDATADIR,
                                         'nsrdb/ri_100_nsrdb_{}.h5')
            sam_files = {"sam gen pv_1":
                         os.path.join(TESTDATADIR,
                                      "SAM/../SAM/naris_pv_1axis_inv13.json")}
        elif tech == 'wind':
            fconfig = 'local_wind.json'
            project_points = os.path.join(TESTDATADIR, 'config', '..',
                                          'config', "wtk_pp_2012_10.csv")
            resource_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_{}.h5')
            sam_files = {"wind0":
                         os.path.join(TESTDATADIR,
                                      "SAM/wind_gen_standard_losses_0.json")}

        config = os.path.join(TESTDATADIR,
                              'config/{}'.format(fconfig)).replace('\\', '/')
        config = safe_json_load(config)
        config['project_points'] = project_points
        config['resource_file'] = resource_file
        config['sam_files'] = sam_files
        config['log_directory'] = run_dir

        config_path = os.path.join(run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['generation', '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        assert len(os.listdir(td)) == 1

        # get reV 2.0 generation profiles from disk
        rev2_profiles = None
        flist = os.listdir(run_dir)
        for fname in flist:
            if fname.endswith('.h5'):
                path = os.path.join(run_dir, fname)
                with Outputs(path, 'r') as cf:
                    msg = 'cf_profile not written to disk'
                    assert 'cf_profile' in cf.datasets, msg
                    rev2_profiles = cf['cf_profile']

                    msg = 'monthly_energy not written to disk'
                    assert 'monthly_energy' in cf.datasets, msg
                    monthly = cf['monthly_energy']
                    assert monthly.shape == (12, 10)

                    for output in LCOE_REQUIRED_OUTPUTS:
                        if tech == 'pv':
                            assert output in cf.datasets
                        else:
                            assert output not in cf.datasets

                    assert "generation_config_fp" in cf.h5.attrs
                    assert "generation_config" in cf.h5.attrs
                    assert (json.loads(cf.h5.attrs["generation_config"])
                            == config)

                break

        if rev2_profiles is None:
            msg = ('reV gen from config failed for "{}"! Could not find '
                   'output file in flist: {}'.format(tech, flist))
            raise RuntimeError(msg)

        # get reV 1.0 generation profiles
        points = ProjectPoints(project_points, sam_files, tech=tech)
        rev1_profiles = get_r1_profiles(year=2012, tech=tech)
        rev1_profiles = rev1_profiles[:, points.sites]

        result = np.allclose(rev1_profiles, rev2_profiles,
                             rtol=RTOL, atol=ATOL)

        clear_loggers()
        msg = ('reV generation from config input failed for "{}" module!'
               .format(tech))
        assert result is True, msg


@pytest.mark.parametrize('tech', ['pv', 'wind'])
def test_sam_config(tech):
    """
    Test running generation from a SAM JSON file or SAM config dictionary
    """
    if tech == 'pv':
        res_file = TESTDATADIR + '/nsrdb/ri_100_nsrdb_2012.h5'
        sam_file = TESTDATADIR + '/SAM/naris_pv_1axis_inv13.json'
        sam_config = {'default': safe_json_load(sam_file)}

        points = slice(0, 100)
        points_config = pd.DataFrame({SiteDataField.GID: range(0, 100),
                                      SiteDataField.CONFIG: ['default'] * 100})

        gen_json = Gen('pvwattsv5', points, sam_file, res_file,
                       output_request=('cf_profile',), sites_per_worker=50)
        gen_json.run(max_workers=2)

        gen_dict = Gen('pvwattsv5', points_config, sam_config, res_file,
                       output_request=('cf_profile',), sites_per_worker=50)
        gen_dict.run(max_workers=2)

        msg = ("reV {} generation run from JSON and SAM config dictionary do "
               "not match".format(tech))
        assert np.allclose(gen_json.out['cf_profile'],
                           gen_dict.out['cf_profile']), msg
    elif tech == 'wind':
        sam_file = TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json'
        res_file = TESTDATADIR + '/wtk/ri_100_wtk_2012.h5'
        sam_config = {'default': safe_json_load(sam_file)}

        points = slice(0, 10)
        points_config = pd.DataFrame({SiteDataField.GID: range(0, 10),
                                      SiteDataField.CONFIG: ['default'] * 10})

        gen_json = Gen('windpower', points, sam_file, res_file,
                       output_request=('cf_profile',), sites_per_worker=3)
        gen_json.run(max_workers=2)

        gen_dict = Gen('windpower', points_config, sam_config, res_file,
                       output_request=('cf_profile',), sites_per_worker=3)
        gen_dict.run(max_workers=2)

        msg = ("reV {} generation run from JSON and SAM config dictionary do "
               "not match".format(tech))
        assert np.allclose(gen_json.out['cf_profile'],
                           gen_dict.out['cf_profile']), msg


def test_cli_run_spaces_in_config(runner, clear_loggers):
    """
    Test running generation from CLI with spaces in SAM config keys
    """
    wind_key = "My wind SAM config_"
    with tempfile.TemporaryDirectory() as td:

        run_dir = os.path.join(td, 'generation')
        os.mkdir(run_dir)

        fconfig = 'local_wind.json'
        project_points = os.path.join(TESTDATADIR, 'config', '..',
                                      'config', "wtk_pp_2012_10_spaces.csv")
        resource_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_{}.h5')
        sam_files = {wind_key: os.path.join(TESTDATADIR, "SAM",
                                            "wind_gen_standard_losses_0.json")}

        config = os.path.join(TESTDATADIR,
                              'config/{}'.format(fconfig)).replace('\\', '/')
        config = safe_json_load(config)
        config['project_points'] = project_points
        config['resource_file'] = resource_file
        config['sam_files'] = sam_files
        config['log_directory'] = run_dir

        config_path = os.path.join(run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['generation', '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        assert len(os.listdir(td)) == 1

        # get reV 2.0 generation profiles from disk
        rev2_profiles = None
        flist = os.listdir(run_dir)
        for fname in flist:
            if fname.endswith('.h5'):
                path = os.path.join(run_dir, fname)
                with Outputs(path, 'r') as cf:

                    msg = 'cf_profile not written to disk'
                    assert 'cf_profile' in cf.datasets, msg
                    rev2_profiles = cf['cf_profile']

                    msg = 'monthly_energy not written to disk'
                    assert 'monthly_energy' in cf.datasets, msg
                    monthly = cf['monthly_energy']
                    assert monthly.shape == (12, 10)

                    for output in LCOE_REQUIRED_OUTPUTS:
                        assert output not in cf.datasets

                break

        if rev2_profiles is None:
            msg = ('reV gen from config failed for "wind"! Could not find '
                   'output file in flist: {}'.format(flist))
            raise RuntimeError(msg)

        # get reV 1.0 generation profiles
        points = ProjectPoints(project_points, sam_files, tech='wind')
        rev1_profiles = get_r1_profiles(year=2012, tech='wind')
        rev1_profiles = rev1_profiles[:, points.sites]

        result = np.allclose(rev1_profiles, rev2_profiles,
                             rtol=RTOL, atol=ATOL)

        clear_loggers()
        msg = 'reV generation from config input failed for "wind" module!'
        assert result is True, msg


@pytest.mark.parametrize('expected_log_message',
                         ["Running serial generation for",
                          "Running parallel generation for"])
def test_gen_mw_config_input(runner, clear_loggers, expected_log_message):
    """Test max_workers input from gen config"""
    with tempfile.TemporaryDirectory() as td:

        run_dir = os.path.join(td, 'generation')
        os.mkdir(run_dir)
        project_points = os.path.join(TESTDATADIR, 'config', '..',
                                      'config', "wtk_pp_2012_10.csv")
        resource_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_{}.h5')
        sam_files = {"wind0":
                     os.path.join(TESTDATADIR,
                                  "SAM/wind_gen_standard_losses_0.json")}

        config = os.path.join(TESTDATADIR, 'config', 'local_wind.json')
        config = safe_json_load(config)
        if "parallel" in expected_log_message:
            config["execution_control"].pop("max_workers")
        else:
            config["execution_control"]["max_workers"] = 1

        config['project_points'] = project_points
        config['resource_file'] = resource_file
        config['sam_files'] = sam_files
        config['log_directory'] = run_dir

        config_path = os.path.join(run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['generation', '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        log_file = os.path.join(run_dir, 'generation_generation.log')
        with open(log_file) as fh:
            assert expected_log_message in fh.read()
        clear_loggers()


def test_year_in_dir_name(runner, clear_loggers):
    """Test generation run with year in project dir"""
    with tempfile.TemporaryDirectory() as td:

        run_dir = os.path.join(td, 'generation_2012_2013')
        os.mkdir(run_dir)
        project_points = os.path.join(TESTDATADIR, 'config', '..',
                                      'config', "wtk_pp_2012_10.csv")
        resource_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_{}.h5')
        sam_files = {"wind0":
                     os.path.join(TESTDATADIR,
                                  "SAM/wind_gen_standard_losses_0.json")}

        config = os.path.join(TESTDATADIR, 'config', 'local_wind.json')
        config = safe_json_load(config)

        config['project_points'] = project_points
        config['resource_file'] = resource_file
        config['sam_files'] = sam_files
        config['log_directory'] = run_dir
        config["analysis_years"] = [2012, 2013]

        config_path = os.path.join(run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['generation', '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        h5_files = [fn for fn in os.listdir(run_dir)
                    if "generation_2012_2013" in fn and ".h5" in fn]
        assert len(h5_files) == 2
        clear_loggers()


def test_year_in_path(runner, clear_loggers):
    """Test generation run with year in multiple places in resource path"""
    with tempfile.TemporaryDirectory() as td:

        run_dir = os.path.join(td, 'generation')
        os.mkdir(run_dir)

        data_dir_2012 = os.path.join(run_dir, '2012')
        data_dir_2013 = os.path.join(run_dir, '2013')
        os.mkdir(data_dir_2012)
        os.mkdir(data_dir_2013)

        resource_fp = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_{}.h5')
        shutil.copy(resource_fp.format(2012),
                    os.path.join(data_dir_2012, 'ri_100_wtk_2012.h5'))
        shutil.copy(resource_fp.format(2013),
                    os.path.join(data_dir_2013, 'ri_100_wtk_2013.h5'))

        project_points = os.path.join(TESTDATADIR, 'config', '..',
                                      'config', "wtk_pp_2012_10.csv")
        resource_file = os.path.join(run_dir, '{}/ri_100_wtk_{}.h5')
        sam_files = {"wind0":
                     os.path.join(TESTDATADIR,
                                  "SAM/wind_gen_standard_losses_0.json")}

        config = os.path.join(TESTDATADIR, 'config', 'local_wind.json')
        config = safe_json_load(config)

        config['project_points'] = project_points
        config['resource_file'] = resource_file
        config['sam_files'] = sam_files
        config['log_directory'] = run_dir
        config["analysis_years"] = [2012, 2013]

        config_path = os.path.join(run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['generation', '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        h5_files = [fn for fn in os.listdir(run_dir)
                    if "generation" in fn and ".h5" in fn]
        assert len(h5_files) == 2
        clear_loggers()


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
