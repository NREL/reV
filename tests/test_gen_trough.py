# -*- coding: utf-8 -*-
# pylint: disable=all
"""
PyTest file for trough physical heat
This is intended to be run with PySAM 1.2.1

Created on 2/6/2020
@author: Mike Bannister
"""
import numpy as np
import os
import pytest

from reV.generation.generation import Gen
from reV import TESTDATADIR
from rex import Resource

BASELINE = os.path.join(TESTDATADIR, 'gen_out', 'gen_ri_trough_2012.h5')
RTOL = 0.01
ATOL = 0


def test_gen_tph():
    """Test generation for trough physical heat"""

    points = slice(0, 1)
    sam_files = TESTDATADIR + '/SAM/trough_default.json'
    res_file = TESTDATADIR + '/nsrdb/ri_100_nsrdb_{}.h5'.format(2012)

    # T_field_cold_in
    #   sequence: Field timestep-averaged inlet temperature [C]
    # m_dot_field_delivered
    #   sequence: Field total mass flow delivered [kg/s]
    # q_dot_htf_sf_out
    #   sequence: Field thermal power leaving in HTF [MWt]

    output_request = ('T_field_cold_in', 'T_field_hot_out',
                      'm_dot_field_delivered', 'm_dot_field_recirc',
                      'q_dot_htf_sf_out', 'q_dot_to_heat_sink',
                      'q_dot_rec_inc', 'qinc_costh', 'dni_costh', 'beam',
                      'cf_mean', 'annual_gross_energy',
                      'annual_thermal_consumption', 'annual_energy')

    # run reV 2.0 generation
    gen = Gen.reV_run('troughphysicalheat', points, sam_files, res_file,
                      max_workers=1,
                      output_request=output_request,
                      sites_per_worker=1, out_fpath=None, scale_outputs=True)

    with Resource(BASELINE) as f:
        for dset in output_request:
            truth = f[dset]
            test = gen.out[dset]
            msg = '{} outputs do not match baseline value!'.format(dset)
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
