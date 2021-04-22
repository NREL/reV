# -*- coding: utf-8 -*-
# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""
from click.testing import CliRunner
import json
import numpy as np
import os
import pandas as pd
import pytest
import tempfile
import traceback

from rex.utilities.loggers import LOGGERS
from reV.cli import main
from reV.config.sam_analysis_configs import GenConfig
from reV.generation.generation import Gen
from reV import TESTDATADIR
from reV.handlers.outputs import Outputs

from rex.utilities.utilities import safe_json_load

RTOL = 0.0
ATOL = 0.04


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


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


@pytest.mark.parametrize('tech', ['pv', 'wind'])  # noqa: C901
def test_gen_from_config(runner, tech):  # noqa: C901
    """Gen PV CF profiles with write to disk and compare against rev1."""
    with tempfile.TemporaryDirectory() as td:
        job_name = 'config_test_{}'.format(tech)

        if tech == 'pv':
            fconfig = 'local_pv.json'
            project_points = os.path.join(TESTDATADIR, 'config',
                                          "project_points_10.csv")
            resource_file = os.path.join(TESTDATADIR,
                                         'nsrdb/ri_100_nsrdb_{}.h5')
            sam_files = {"sam_gen_pv_1":
                         os.path.join(TESTDATADIR,
                                      "SAM/naris_pv_1axis_inv13.json")}
        elif tech == 'wind':
            fconfig = 'local_wind.json'
            project_points = os.path.join(TESTDATADIR, 'config',
                                          "wtk_pp_2012_10.csv")
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
        config['directories']['log_directory'] = td
        config['directories']['output_directory'] = td

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        config_obj = GenConfig(config_path)

        result = runner.invoke(main, ['-n', job_name,
                                      '-c', config_path,
                                      'generation'])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        # get reV 2.0 generation profiles from disk
        rev2_profiles = None
        flist = os.listdir(config_obj.dirout)
        for fname in flist:
            if job_name in fname and fname.endswith('.h5'):
                path = os.path.join(config_obj.dirout, fname)
                with Outputs(path, 'r') as cf:

                    msg = 'cf_profile not written to disk'
                    assert 'cf_profile' in cf.datasets, msg
                    rev2_profiles = cf['cf_profile']

                    msg = 'monthly_energy not written to disk'
                    assert 'monthly_energy' in cf.datasets, msg
                    monthly = cf['monthly_energy']
                    assert monthly.shape == (12, 10)

                break

        if rev2_profiles is None:
            msg = ('reV gen from config failed for "{}"! Could not find '
                   'output file in flist: {}'.format(tech, flist))
            raise RuntimeError(msg)

        # get reV 1.0 generation profiles
        rev1_profiles = get_r1_profiles(year=config_obj.years[0], tech=tech)
        rev1_profiles = \
            rev1_profiles[:, config_obj.parse_points_control().sites]

        result = np.allclose(rev1_profiles, rev2_profiles,
                             rtol=RTOL, atol=ATOL)

        LOGGERS.clear()
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
        points_config = pd.DataFrame({"gid": range(0, 100),
                                      "config": ['default'] * 100})

        gen_json = Gen.reV_run('pvwattsv5', points, sam_file, res_file,
                               output_request=('cf_profile',),
                               max_workers=2, sites_per_worker=50)

        gen_dict = Gen.reV_run('pvwattsv5', points_config, sam_config,
                               res_file,
                               output_request=('cf_profile',),
                               max_workers=2, sites_per_worker=50)

        msg = ("reV {} generation run from JSON and SAM config dictionary do "
               "not match".format(tech))
        assert np.allclose(gen_json.out['cf_profile'],
                           gen_dict.out['cf_profile']), msg
    elif tech == 'wind':
        sam_file = TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json'
        res_file = TESTDATADIR + '/wtk/ri_100_wtk_2012.h5'
        sam_config = {'default': safe_json_load(sam_file)}

        points = slice(0, 10)
        points_config = pd.DataFrame({"gid": range(0, 10),
                                      "config": ['default'] * 10})

        gen_json = Gen.reV_run('windpower', points, sam_file, res_file,
                               output_request=('cf_profile',),
                               max_workers=2, sites_per_worker=3)

        gen_dict = Gen.reV_run('windpower', points_config, sam_config,
                               res_file,
                               output_request=('cf_profile',),
                               max_workers=2, sites_per_worker=3)

        msg = ("reV {} generation run from JSON and SAM config dictionary do "
               "not match".format(tech))
        assert np.allclose(gen_json.out['cf_profile'],
                           gen_dict.out['cf_profile']), msg


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
