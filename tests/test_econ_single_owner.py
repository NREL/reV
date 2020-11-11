# -*- coding: utf-8 -*-
# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import pytest
import numpy as np

from reV.econ.econ import Econ
from reV import TESTDATADIR


RTOL = 0.01
ATOL = 0.001

# current baseline results
OUT_BASELINE = {'ppa_price': np.array([75.7973, 74.618286, 73.2412,
                                       74.28226, 74.26111, 70.87696,
                                       68.88366, 69.298996, 69.13398,
                                       66.936386], dtype=np.float32),
                'project_return_aftertax_npv': -0.00982841 * np.ones((10, )),
                'lcoe_nom': np.array([75.7973, 74.618286, 73.2412, 74.28226,
                                      74.26111, 70.87696, 68.88366, 69.298996,
                                      69.13398, 66.936386], dtype=np.float32),
                'lcoe_real': np.array([63.769077, 62.777153, 61.618607,
                                       62.49445, 62.476658, 59.62954,
                                       57.952553, 58.30198, 58.16315,
                                       56.314293], dtype=np.float32),
                'size_of_equity': np.array(10 * [1343680]),
                'wacc': np.array(10 * [8.280014]),
                }


def test_single_owner():
    """Gen PV CF profiles with write to disk and compare against rev1."""
    cf_file = os.path.join(TESTDATADIR, 'gen_out/wind_2012_x000.h5')
    sam_files = os.path.join(TESTDATADIR, 'SAM/wind_single_owner.json')

    output_request = ('ppa_price', 'project_return_aftertax_npv', 'lcoe_nom',
                      'lcoe_real', 'size_of_equity', 'wacc')

    obj = Econ.reV_run(points=slice(0, 10), sam_files=sam_files,
                       cf_file=cf_file, year=2012,
                       output_request=output_request,
                       max_workers=1, sites_per_worker=10,
                       points_range=None, fout=None)

    for k, v in obj.out.items():
        msg = 'Array for "{}" is bad!'.format(k)
        result = np.allclose(v, OUT_BASELINE[k], rtol=RTOL, atol=ATOL)
        assert result, msg

    return obj.out


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
