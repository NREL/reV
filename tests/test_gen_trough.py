# -*- coding: utf-8 -*-
"""
PyTest file for trough physical heat
This is intended to be run with PySAM 1.2.1

Created on 2/6/2020
@author: Mike Bannister
"""
import numpy as np
import os
import pytest
import logging

from reV.generation.generation import Gen
from reV import TESTDATADIR


def test_gen_tph(caplog):
    """Test generation for trough physical heat"""

    caplog.set_level(logging.DEBUG)
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
    gen = Gen.reV_run(tech='troughphysicalheat', points=points,
                      sam_files=sam_files, res_file=res_file, max_workers=1,
                      output_request=output_request,
                      sites_per_worker=1, fout=None, scale_outputs=True)

    def my_assert(x, y, digits):
        if isinstance(x, np.ndarray):
            x = float(x.sum())
        assert round(x, digits) == round(y, digits)

    # Some results may be different with PySAM 2 vs 1.2.1
    my_assert(gen.out['T_field_cold_in'], 1591237.0, 0)
    my_assert(gen.out['T_field_hot_out'], 1912855.0, 0)
    my_assert(gen.out['m_dot_field_delivered'], 127769.113, 0)
    my_assert(gen.out['m_dot_field_recirc'], 94023.448, 0)
    my_assert(gen.out['q_dot_htf_sf_out'], 30396.0, 0)
    my_assert(gen.out['q_dot_to_heat_sink'], 30031.35, 0)
    my_assert(gen.out['annual_thermal_consumption'], 16239.159, 0)
    my_assert(gen.out['beam'], 3052417.0, 0)
    my_assert(gen.out['annual_gross_energy'], 15015675.0, 0)


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
