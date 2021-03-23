# -*- coding: utf-8 -*-
"""
PyTest file for CSP generation in Rhode Island (lol).

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""
import numpy as np
import os
import pytest

from reV.generation.generation import Gen
from reV import TESTDATADIR


def test_gen_csp():
    """Test generation for CSP"""
    points = slice(0, 1)
    sam_files = TESTDATADIR + '/SAM/i_csp_tcsmolten_salt.json'
    res_file = TESTDATADIR + '/nsrdb/ri_100_nsrdb_{}.h5'.format(2012)

    # run reV 2.0 generation
    gen = Gen.reV_run('tcsmoltensalt', points, sam_files, res_file,
                      max_workers=1,
                      output_request=('cf_mean', 'cf_profile', 'gen_profile'),
                      sites_per_worker=1, fout=None, scale_outputs=False)

    cf_mean = gen.out['cf_mean']
    cf_profile = gen.out['cf_profile']
    gen_profile = gen.out['gen_profile']

    assert np.isclose(cf_mean, 0.2679, atol=0.001)
    assert np.isclose(cf_profile.max(), 0.001)
    assert gen_profile.max() > 1e5


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
