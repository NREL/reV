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

from reV.lcoe.lcoe import LCOE
from reV import __testdatadir__ as TESTDATADIR


RTOL = 0.0
ATOL = 0.04
PURGE_OUT = True


@pytest.mark.parametrize('year', [('2012'), ('2013')])
def test_lcoe(year):
    """Gen PV CF profiles with write to disk and compare against rev1."""
    cf_file = TESTDATADIR + '/gen_out/gen_ri_pv_{}_x000.h5'.format(year)
    sam_files = TESTDATADIR + '/SAM/i_lcoe_naris_pv_1axis_inv13.json'
    r1f = TESTDATADIR + '/ri_pv/scalar_outputs/project_outputs.h5'
    dirout = os.path.join(TESTDATADIR, 'lcoe_out')
    fout = 'lcoe_ri_pv_{}.h5'.format(year)
    points = slice(0, 100)
    obj = LCOE.run_direct(points=points, sam_files=sam_files, cf_file=cf_file,
                          cf_year=year, n_workers=1, sites_per_split=25,
                          points_range=None, fout=fout, dirout=dirout,
                          return_obj=True)
    lcoe = [c['lcoe_fcr'] for c in obj.out.values()]

    with h5py.File(r1f) as f:
        year_rows = {'2012': 0, '2013': 1}
        r1_lcoe = f['pv']['lcoefcr'][year_rows[str(year)], 0:100] * 1000

    result = np.allclose(lcoe, r1_lcoe, rtol=RTOL, atol=ATOL)
    if result and PURGE_OUT:
        # remove output files if test passes.
        flist = os.listdir(dirout)
        for fname in flist:
            os.remove(os.path.join(dirout, fname))

    assert result is True


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
