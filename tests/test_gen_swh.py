# -*- coding: utf-8 -*-
"""
PyTest file for solar water heating generation
This is intended to be run with PySAM 1.2.1

Created on 2/6/2020
@author: Mike Bannister
"""
import numpy as np
import os
import pytest
import pickle

from reV.generation.generation import Gen
from reV import TESTDATADIR

PICKLEFILE = TESTDATADIR + '/SAM/swh_profiles_2013.pkl'


def save_outputs(out):
    """ Save appropriate outputs to pickle file """
    new_out = {'gen_profile': out['gen_profile']}
    pickle.dump(new_out, open(PICKLEFILE, 'wb'))


def my_assert(x, y, digits):
    """
    Sum time series data for comparison to rounded known value. This is an
    arbitrary method of comparison but it's simple.
    """
    if isinstance(x, np.ndarray):
        x = float(x.sum())
    assert round(x, digits) == round(y, digits)


def test_gen_swh_non_leap_year():
    """Test generation for solar water heating for non leap year (2013)"""

    points = slice(0, 1)
    sam_files = TESTDATADIR + '/SAM/swh_default.json'
    res_file = TESTDATADIR + '/nsrdb/ri_100_nsrdb_{}.h5'.format(2013)

    output_request = ('T_amb', 'T_cold', 'T_deliv', 'T_hot', 'draw',
                      'beam', 'diffuse', 'I_incident', 'I_transmitted',
                      'annual_Q_deliv', 'Q_deliv', 'cf_mean', 'solar_fraction',
                      'gen_profile')

    # run reV 2.0 generation
    gen = Gen.reV_run(tech='solarwaterheat', points=points,
                      sam_files=sam_files, res_file=res_file, max_workers=1,
                      output_request=output_request,
                      sites_per_worker=1, fout=None, scale_outputs=True)

    # Some results will be different with PySAM 2 vs 1.2.1
    my_assert(gen.out['T_amb'], 180621, 0)
    my_assert(gen.out['T_cold'], 410491.1066, 0)
    my_assert(gen.out['T_deliv'], 813060.4364, 0)
    my_assert(gen.out['T_hot'], 813419.981, 0)
    my_assert(gen.out['Q_deliv'], 5390.47749, 1)

    # Verify series are in correct order and have been rolled correctly
    profiles = pickle.load(open(PICKLEFILE, 'rb'))
    for k in profiles.keys():
        assert np.array_equal(profiles[k], gen.out[k])


def test_gen_swh_leap_year():
    """Test generation for solar water heating for a leap year (2012)"""

    points = slice(0, 1)
    sam_files = TESTDATADIR + '/SAM/swh_default.json'
    res_file = TESTDATADIR + '/nsrdb/ri_100_nsrdb_{}.h5'.format(2012)

    output_request = ('T_amb', 'T_cold', 'T_deliv', 'T_hot', 'draw',
                      'beam', 'diffuse', 'I_incident', 'I_transmitted',
                      'annual_Q_deliv', 'Q_deliv', 'cf_mean', 'solar_fraction')

    # run reV 2.0 generation
    gen = Gen.reV_run(tech='solarwaterheat', points=points,
                      sam_files=sam_files, res_file=res_file, max_workers=1,
                      output_request=output_request,
                      sites_per_worker=1, fout=None, scale_outputs=True)

    # Some results will be different with PySAM 2 vs 1.2.1, in particular,
    # solar_fraction and cf_mean
    my_assert(gen.out['T_amb'], 204459, 0)
    my_assert(gen.out['T_cold'], 433511.47, 0)
    my_assert(gen.out['T_deliv'], 836763.3482, 0)
    my_assert(gen.out['T_hot'], 836961.2498, 0)
    my_assert(gen.out['draw'], 145999.90, 0)
    my_assert(gen.out['beam'], 3047259, 0)
    my_assert(gen.out['diffuse'], 1222013, 0)
    my_assert(gen.out['I_incident'], 3279731.523, -1)
    my_assert(gen.out['I_transmitted'], 2769482.776, -1)
    my_assert(gen.out['annual_Q_deliv'], 2697.09, 1)
    my_assert(gen.out['Q_deliv'], 5394.188339, 1)
    my_assert(gen.out['solar_fraction'], 0.6875, 4)


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
