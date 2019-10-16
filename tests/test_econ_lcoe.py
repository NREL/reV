# -*- coding: utf-8 -*-
# pylint: skip-file
"""
PyTest file for lcoe econ run in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import h5py
import pytest
import numpy as np

from reV.econ.econ import Econ
from reV import TESTDATADIR
from reV.handlers.outputs import Outputs


RTOL = 0.01
ATOL = 0.001
PURGE_OUT = True


@pytest.mark.parametrize('year', ('2012', '2013'))
def test_lcoe(year):
    """Gen PV CF profiles with write to disk and compare against rev1."""
    cf_file = os.path.join(TESTDATADIR,
                           'gen_out/gen_ri_pv_{}_x000.h5'.format(year))
    sam_files = os.path.join(TESTDATADIR,
                             'SAM/i_lcoe_naris_pv_1axis_inv13.json')
    r1f = os.path.join(TESTDATADIR,
                       'ri_pv/scalar_outputs/project_outputs.h5')
    points = slice(0, 100)
    obj = Econ.reV_run(points=points, sam_files=sam_files, cf_file=cf_file,
                       cf_year=year, output_request='lcoe_fcr',
                       n_workers=1, sites_per_split=25,
                       points_range=None, fout=None, return_obj=True)
    lcoe = list(obj.out['lcoe_fcr'])

    with h5py.File(r1f) as f:
        year_rows = {'2012': 0, '2013': 1}
        r1_lcoe = f['pv']['lcoefcr'][year_rows[str(year)], 0:100] * 1000

    result = np.allclose(lcoe, r1_lcoe, rtol=RTOL, atol=ATOL)

    assert result


@pytest.mark.parametrize('year', ('2012', '2013'))
def test_fout(year):
    """Gen PV CF profiles with write to disk and compare against rev1."""
    cf_file = os.path.join(TESTDATADIR,
                           'gen_out/gen_ri_pv_{}_x000.h5'.format(year))
    sam_files = os.path.join(TESTDATADIR,
                             'SAM/i_lcoe_naris_pv_1axis_inv13.json')
    r1f = os.path.join(TESTDATADIR,
                       'ri_pv/scalar_outputs/project_outputs.h5')
    dirout = os.path.join(TESTDATADIR, 'lcoe_out')
    fout = 'lcoe_out_{}.h5'.format(year)
    fpath = os.path.join(dirout, fout)
    points = slice(0, 100)
    Econ.reV_run(points=points, sam_files=sam_files, cf_file=cf_file,
                 cf_year=year, output_request='lcoe_fcr',
                 n_workers=1, sites_per_split=25,
                 points_range=None, fout=fout, dirout=dirout,
                 return_obj=False)

    with Outputs(fpath) as f:
        lcoe = f['lcoe_fcr']

    with h5py.File(r1f) as f:
        year_rows = {'2012': 0, '2013': 1}
        r1_lcoe = f['pv']['lcoefcr'][year_rows[str(year)], 0:100] * 1000
    result = np.allclose(lcoe, r1_lcoe, rtol=RTOL, atol=ATOL)

    if PURGE_OUT:
        os.remove(fpath)

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
