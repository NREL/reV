# -*- coding: utf-8 -*-
# pylint: disable=all
"""
PyTest file for solar water heating generation
This is intended to be run with PySAM 1.2.1

Created on 2/6/2020
@author: Mike Bannister
"""
import os

import numpy as np
import pytest
from rex import Resource

from reV import TESTDATADIR
from reV.generation.generation import Gen

RTOL = 0.01
ATOL = 0


def test_gen_swh_non_leap_year():
    """Test generation for solar water heating for non leap year (2013)"""
    BASELINE = os.path.join(TESTDATADIR, 'gen_out', 'gen_ri_swh_2013.h5')
    points = slice(0, 1)
    sam_files = TESTDATADIR + '/SAM/swh_default.json'
    res_file = TESTDATADIR + '/nsrdb/ri_100_nsrdb_{}.h5'.format(2013)

    output_request = ('T_amb', 'T_cold', 'T_deliv', 'T_hot', 'draw',
                      'beam', 'diffuse', 'I_incident', 'I_transmitted',
                      'annual_Q_deliv', 'Q_deliv', 'cf_mean', 'solar_fraction',
                      'gen_profile')

    # run reV 2.0 generation
    gen = Gen('solarwaterheat', points, sam_files, res_file,
              output_request=output_request, sites_per_worker=1,
              scale_outputs=True)
    gen.run(max_workers=1)

    with Resource(BASELINE) as f:
        for dset in output_request:
            truth = f[dset]
            test = gen.out[dset]
            if len(test.shape) == 2:
                truth = np.mean(truth, axis=1)
                test = np.mean(test, axis=1)

            msg = ('{} outputs do not match baseline value! Values differ '
                   'at most by: {}'.format(dset, np.max(np.abs(truth - test))))
            assert np.allclose(truth, test, rtol=RTOL, atol=ATOL), msg


def test_gen_swh_leap_year():
    """Test generation for solar water heating for a leap year (2012)"""

    BASELINE = os.path.join(TESTDATADIR, 'gen_out', 'gen_ri_swh_2012.h5')
    points = slice(0, 1)
    sam_files = TESTDATADIR + '/SAM/swh_default.json'
    res_file = TESTDATADIR + '/nsrdb/ri_100_nsrdb_{}.h5'.format(2012)

    output_request = ('T_amb', 'T_cold', 'T_deliv', 'T_hot', 'draw',
                      'beam', 'diffuse', 'I_incident', 'I_transmitted',
                      'annual_Q_deliv', 'Q_deliv', 'cf_mean', 'solar_fraction')

    # run reV 2.0 generation
    gen = Gen('solarwaterheat', points, sam_files, res_file,
              output_request=output_request, sites_per_worker=1,
              scale_outputs=True)
    gen.run(max_workers=1)

    with Resource(BASELINE) as f:
        for dset in output_request:
            truth = f[dset]
            test = gen.out[dset]
            if len(test.shape) == 2:
                truth = np.mean(truth, axis=0)
                test = np.mean(test, axis=0)

            msg = ('{} outputs do not match baseline value! Values differ '
                   'at most by: {}'.format(dset, np.max(np.abs(truth - test))))
            assert np.allclose(truth, test, rtol=RTOL, atol=ATOL), msg


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
