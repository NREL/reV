# -*- coding: utf-8 -*-
# pylint: skip-file
"""
PyTest file for wind econ with offshore ORCA.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import pytest
import numpy as np

pytest.importorskip("ORCA")
from reV.econ.econ import Econ
from reV import TESTDATADIR


RTOL = 0.01
ATOL = 0.001
PURGE_OUT = True


@pytest.mark.parametrize('rut_id', range(5))
def test_ORCA(rut_id):
    """Test LCOE results from subset of Rutgers wind analysis."""

    # results from Rutgers analysis for each ensemble ID
    # updated based on 2019 ORCA model v0.9 (10-21-2019)
    baseline = {0: [129.78041, 129.44212, 59.77386, 59.468113, 59.165474],
                1: [130.4546, 129.78186, 59.927917, 59.468113, 59.31641],
                2: [129.78041, 129.10416, 59.620598, 59.31641, 59.165474],
                3: [125.87725, 125.49972, 57.840874, 57.554535, 57.412426],
                4: [126.51138, 125.81905, 57.840874, 57.554535, 57.27102],
                }

    points = TESTDATADIR + '/ORCA/rutgers_pp_slim.csv'
    cf_file = TESTDATADIR + '/gen_out/rut_{}_node00_x000.h5'.format(rut_id)
    sam_files = {'6MW_offshore': TESTDATADIR + '/ORCA/6MW_offshore.json',
                 # 't200_t186': TESTDATADIR + '/ORCA/t200_t186.json',
                 't233_t217': TESTDATADIR + '/ORCA/t233_t217.json',
                 # 't325_t302': TESTDATADIR + '/ORCA/t325_t302.json',
                 }
    site_data = TESTDATADIR + '/ORCA/orca_site_data.csv'

    obj = Econ.reV_run(points=points, sam_files=sam_files, cf_file=cf_file,
                       cf_year=None, site_data=site_data,
                       output_request='lcoe_fcr', n_workers=1,
                       sites_per_split=25, points_range=None,
                       fout=None, dirout=None)

    lcoe = list(obj.out['lcoe_fcr'])

    msg = ('LCOE does not match (new, baseline): \n{} \n{}'
           .format(lcoe, baseline[rut_id]))
    result = np.allclose(lcoe, baseline[rut_id], rtol=RTOL, atol=ATOL)

    assert result, msg


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
