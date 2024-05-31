# -*- coding: utf-8 -*-
# pylint: disable=all
"""
PyTest file for trough physical heat
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

BASELINE = os.path.join(TESTDATADIR, 'gen_out', 'gen_ri_trough_2012.h5')
RTOL = 0.1
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
    gen = Gen('troughphysicalheat', points, sam_files, res_file,
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
