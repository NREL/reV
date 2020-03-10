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
import logging

from reV.generation.generation import Gen
from reV import TESTDATADIR


def test_gen_swh(caplog):
    """Test generation for solar water heating"""

    caplog.set_level(logging.DEBUG)
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

    #  for var in output_request:
    #      if isinstance(gen.out[var], np.ndarray):
    #          print(var, gen.out[var].sum())
    #      else:
    #          print(var, gen.out[var])

    def my_assert(x, y, digits):
        if isinstance(x, np.ndarray):
            x = float(x.sum())
        assert round(x, digits) == round(y, digits)

    # Some results will be different with PySAM 2 vs 1.2.1, in particular,
    # solar_fraction and cf_mean
    my_assert(gen.out['T_amb'], 204374, 0)
    my_assert(gen.out['T_deliv'], 837587.6528, 0)
    my_assert(gen.out['T_hot'], 837785.36, 0)
    my_assert(gen.out['draw'], 145999.90, 0)
    my_assert(gen.out['beam'], 3052417, 0)
    my_assert(gen.out['diffuse'], 1221626, 0)
    my_assert(gen.out['I_incident'], 3284008.791, 0)
    my_assert(gen.out['I_transmitted'], 2773431.416, 0)
    my_assert(gen.out['annual_Q_deliv'], 2701.62, 1)
    my_assert(gen.out['Q_deliv'], 5403.240911, 0)
    my_assert(gen.out['solar_fraction'], 0.6887506, 4)


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
