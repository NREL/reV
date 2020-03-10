# -*- coding: utf-8 -*-
"""
PyTest file for linear Fresnel
This is intended to be run with PySAM 1.2.1

Created on 2/6/2020
@author: Mike Bannister
"""

import os
import logging
import pytest
import numpy as np

from reV.generation.generation import Gen
from reV import TESTDATADIR


def test_gen_linear(caplog):
    """Test generation for linear Fresnel"""

    caplog.set_level(logging.DEBUG)
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
                      'q_dot_rec_inc', 'cf_mean', 'annual_field_energy',
                      'annual_thermal_consumption')

    # run reV 2.0 generation
    gen = Gen.reV_run(tech='lineardirectsteam', points=points,
                      sam_files=sam_files, res_file=res_file, max_workers=1,
                      output_request=output_request,
                      sites_per_worker=1, fout=None, scale_outputs=True)

    def my_assert(x, y, digits):
        if isinstance(x, np.ndarray):
            x = float(x.sum())
        assert round(x, digits) == round(y, digits)

    # Some results may be different with PySAM 2 vs 1.2.1
    my_assert(gen.out['q_dot_to_heat_sink'], 10898.58, 0)
    my_assert(gen.out['gen'], 10462639.51, -2)
    my_assert(gen.out['m_dot_field'], 15153.68676, 1)
    my_assert(gen.out['q_dot_sf_out'], 10970, 0)
    my_assert(gen.out['W_dot_heat_sink_pump'], 0.17348091, 6)
    my_assert(gen.out['annual_field_energy'], 5231320, 0)
    my_assert(gen.out['annual_thermal_consumption'], 3195, 0)


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
