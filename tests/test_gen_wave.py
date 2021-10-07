# -*- coding: utf-8 -*-
# pylint: skip-file
"""
PyTest file for MHK Wave generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import pytest
import numpy as np

from reV.generation.generation import Gen
from reV.SAM.defaults import DefaultMhkWave
from reV import TESTDATADIR

from rex import Resource, safe_json_load

BASELINE = os.path.join(TESTDATADIR, 'gen_out', 'ri_wave_2010.h5')


def test_mhkwave_defaults():
    """Test mhkwave module defaults"""
    sam_files = TESTDATADIR + '/SAM/mhkwave_default.json'
    baseline = safe_json_load(sam_files)
    baseline = np.array(baseline['wave_power_matrix'])

    defaults = DefaultMhkWave.default()
    test = np.array(defaults.MHKWave.wave_power_matrix)

    assert np.allclose(baseline, test)


def test_mhkwave():
    """Test mhkwave module in reV"""
    sam_files = TESTDATADIR + '/SAM/mhkwave_default.json'
    res_file = TESTDATADIR + '/wave/ri_wave_2010.h5'
    points = slice(0, 100)
    output_request = ('cf_mean', 'cf_profile')

    test = Gen.reV_run('mhkwave', points, sam_files, res_file,
                       max_workers=1, sites_per_worker=3,
                       out_fpath=None, output_request=output_request)

    with Resource(BASELINE) as f:
        assert np.allclose(test.out['cf_mean'], f['cf_mean'],
                           atol=0.01, rtol=0.01)
        assert np.allclose(test.out['cf_profile'], f['cf_profile'],
                           atol=0.01, rtol=0.01)


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
