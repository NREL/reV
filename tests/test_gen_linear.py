# -*- coding: utf-8 -*-
"""
PyTest file for linear Fresnel
This is intended to be run with PySAM 1.2.1

Created on 2/6/2020
@author: Mike Bannister
"""
import os
import pytest
import numpy as np
import json

from reV.generation.generation import Gen
from reV import TESTDATADIR

BASELINE = os.path.join(TESTDATADIR, 'SAM/output_linear_direct_steam.json')
RTOL = 0
ATOL = 0.001


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
                      'q_dot_rec_inc', 'cf_mean', 'gen_profile',
                      'annual_field_energy', 'annual_thermal_consumption',)

    # run reV 2.0 generation
    gen = Gen.reV_run(tech='lineardirectsteam', points=points,
                      sam_files=sam_files, res_file=res_file, max_workers=1,
                      output_request=output_request,
                      sites_per_worker=1, fout=None, scale_outputs=True)

    def my_assert(test, *truth, digits=0):
        if isinstance(test, np.ndarray):
            test = float(test.sum())

        check = any(round(test, digits) == round(true, digits)
                    for true in truth)
        assert check

    # Some results may be different with PySAM 2 vs 1.2.1
    my_assert(gen.out['q_dot_to_heat_sink'], 10874.82934, digits=0)
    my_assert(gen.out['gen'], 10439836.56, digits=-2)
    my_assert(gen.out['m_dot_field'], 15146.1688, digits=1)
    my_assert(gen.out['q_dot_sf_out'], 10946.40988, digits=0)
    my_assert(gen.out['W_dot_heat_sink_pump'], 0.173017451, digits=6)
    my_assert(gen.out['annual_field_energy'], 5219916.0, 5219921.5, digits=0)
    my_assert(gen.out['annual_thermal_consumption'], 3178, digits=0)

    # Verify series are in correct order and have been rolled correctly
    if os.path.exists(BASELINE):
        with open(BASELINE, 'r') as f:
            profiles = json.load(f)
        for k in profiles.keys():
            assert np.allclose(profiles[k], gen.out[k], rtol=RTOL, atol=ATOL)
    else:
        with open(BASELINE, 'w') as f:
            out = {k: v.tolist() for k, v in gen.out.items()}
            json.dump(out, f)


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
