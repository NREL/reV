# -*- coding: utf-8 -*-
"""
PyTest file for trough physical heat
Created on 2/6/2020

@author: Mike Bannister
"""

import os
import pytest
import logging

from reV.generation.generation import Gen
from reV import TESTDATADIR


def test_gen_tph(caplog):
    """Test generation for trough physical heat"""

    caplog.set_level(logging.DEBUG)
    points = slice(0, 1)
    # json is from "Process heat parabolic trough" in SAM
    # sam_files = TESTDATADIR + '/SAM/process_heat_parabolic_trough.json'
    sam_files = TESTDATADIR + '/SAM/trough_default.json'
    # Next file came from CSP parabolic trough (pysical)
    # sam_files = TESTDATADIR + '/SAM/process_heat_parabolic_trough2.json'
    res_file = TESTDATADIR + '/nsrdb/ri_100_nsrdb_{}.h5'.format(2012)
    output_request = ('T_field_cold_in', 'T_field_hot_out',
                      'm_dot_field_delivered', 'm_dot_field_recirc',
                      'q_dot_htf_sf_out', 'q_dot_to_heat_sink',
                      'q_dot_rec_inc', 'qinc_costh', 'dni_costh', 'beam',
                      'cf_mean')
    # run reV 2.0 generation
    gen = Gen.reV_run(tech='troughphysicalheat', points=points,
                      sam_files=sam_files, res_file=res_file, max_workers=1,
                      output_request=output_request,
                      sites_per_worker=1, fout=None, scale_outputs=False)

    for var in output_request:
        print(var, gen.out[var])

#    cf_mean = gen.out['cf_mean']
#    cf_profile = gen.out['cf_profile']
#    gen_profile = gen.out['gen_profile']
#
#    assert cf_mean == 0.0
#    assert cf_profile.max() == 1.0
#    assert gen_profile.max() > 1e5


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
