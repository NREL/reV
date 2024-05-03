# -*- coding: utf-8 -*-
"""
PyTest file for linear Fresnel

Created on 2/6/2020
@author: Mike Bannister
"""
import os

import numpy as np
import pytest
from rex import Resource

from reV import TESTDATADIR
from reV.generation.generation import Gen

BASELINE = os.path.join(TESTDATADIR, 'gen_out', 'gen_ri_linear_2012.h5')
RTOL = 0.01
ATOL = 0.05  # Increased from 0 -> 0.05 when upgrading to PySAM 3+


def test_gen_linear():
    """Test generation for linear Fresnel"""

    points = slice(0, 1)
    sam_files = TESTDATADIR + '/SAM/linear_default.json'
    res_file = TESTDATADIR + '/nsrdb/ri_100_nsrdb_{}.h5'.format(2012)

    # q_dot_to_heat_sink
    #     sequence: Heat sink thermal power [MWt]
    # gen
    #     sequence: System power generated [kW]
    # m_dot_field
    #     sequence: Field total mass flow rate [kg/s]
    # q_dot_sf_out
    #     sequence: Field thermal power leaving in steam [MWt]
    # W_dot_heat_sink_pump
    #     sequence: Heat sink pumping power [MWe]
    # m_dot_loop
    #     sequence: Receiver mass flow rate [kg/s]
    output_request = ('q_dot_to_heat_sink', 'gen', 'm_dot_field',
                      'q_dot_sf_out', 'W_dot_heat_sink_pump', 'm_dot_loop',
                      'q_dot_rec_inc', MetaKeyName.CF_MEAN, 'gen_profile',
                      'annual_field_energy', 'annual_thermal_consumption',)

    # run reV 2.0 generation
    gen = Gen('lineardirectsteam', points, sam_files, res_file,
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
