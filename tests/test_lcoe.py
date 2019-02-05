# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import h5py
import pytest
import numpy as np

from reV.econ.econ import Econ
from reV import TESTDATADIR


RTOL = 0.01
ATOL = 0.001
PURGE_OUT = True


@pytest.mark.parametrize('year', ('2012', '2013'))
def test_lcoe(year):
    """Gen PV CF profiles with write to disk and compare against rev1."""
    cf_file = TESTDATADIR + '/gen_out/gen_ri_pv_{}_x000.h5'.format(year)
    sam_files = TESTDATADIR + '/SAM/i_lcoe_naris_pv_1axis_inv13.json'
    r1f = TESTDATADIR + '/ri_pv/scalar_outputs/project_outputs.h5'
    points = slice(0, 100)
    obj = Econ.run_direct(points=points, sam_files=sam_files, cf_file=cf_file,
                          cf_year=year, output_request='lcoe_fcr',
                          n_workers=1, sites_per_split=25,
                          points_range=None, fout=None, return_obj=True)
    lcoe = [c['lcoe_fcr'] for c in obj.out.values()]

    with h5py.File(r1f) as f:
        year_rows = {'2012': 0, '2013': 1}
        r1_lcoe = f['pv']['lcoefcr'][year_rows[str(year)], 0:100] * 1000

    result = np.allclose(lcoe, r1_lcoe, rtol=RTOL, atol=ATOL)

    assert result


@pytest.mark.parametrize('rut_id', range(5))
def test_ORCA(rut_id):
    """Test LCOE results from subset of Rutgers wind analysis."""

    # results from Rutgers analysis for each ensemble ID
    baseline = {0: [160.1777458, 159.5480359, 59.725, 59.384, 59.161],
                1: [160.7288996, 160.0425542, 59.822, 59.466, 59.213],
                2: [159.9368503, 159.2512835, 59.595, 59.257, 59.066],
                3: [155.3335381, 154.6299536, 57.831, 57.494, 57.402],
                4: [155.9064394, 155.048778, 57.802, 57.439, 57.261],
                }

    points = TESTDATADIR + '/ORCA/rutgers_pp_slim.csv'
    cf_file = TESTDATADIR + '/gen_out/rut_{}_node00_x000.h5'.format(rut_id)
    sam_files = {'6MW_offshore': TESTDATADIR + '/ORCA/6MW_offshore.json',
                 't200_t186': TESTDATADIR + '/ORCA/t200_t186.json',
                 't233_t217': TESTDATADIR + '/ORCA/t233_t217.json',
                 't325_t302': TESTDATADIR + '/ORCA/t325_t302.json',
                 }
    site_data = TESTDATADIR + '/ORCA/orca_site_data.csv'

    obj = Econ.run_direct(points=points, sam_files=sam_files, cf_file=cf_file,
                          cf_year=None, site_data=site_data,
                          output_request='lcoe_fcr', n_workers=1,
                          sites_per_split=25, points_range=None,
                          fout=None, dirout=None, return_obj=True)

    lcoe = [c['lcoe_fcr'] for c in obj.out.values()]
    result = np.allclose(lcoe, baseline[rut_id], rtol=RTOL, atol=ATOL)

    assert result


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
