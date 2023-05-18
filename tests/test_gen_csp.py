# -*- coding: utf-8 -*-
# pylint: disable=all
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
from rex import Resource

BASELINE = os.path.join(TESTDATADIR, 'gen_out', 'gen_ri_csp_2012.h5')
RTOL = 0.1
ATOL = 0.0


def test_gen_csp():
    """Test generation for CSP"""
    points = slice(0, 1)
    sam_files = TESTDATADIR + '/SAM/i_csp_tcsmolten_salt.json'
    res_file = TESTDATADIR + '/nsrdb/ri_100_nsrdb_{}.h5'.format(2012)

    # run reV 2.0 generation
    output_request = ('cf_mean', 'cf_profile', 'gen_profile')
    gen = Gen('tcsmoltensalt', points, sam_files, res_file,
               output_request=output_request, sites_per_worker=1)
    gen.reV_run(max_workers=1, scale_outputs=True)

    with Resource(BASELINE) as f:
        for dset in output_request:
            truth = f[dset]
            test = gen.out[dset]
            if len(test.shape) == 2:
                truth = np.mean(truth, axis=0)
                test = np.mean(test, axis=0)

            msg = ('{} outputs do not match baseline value! Values differ '
                   'at most by: {}'.format(dset, np.max(np.abs(truth - test))))
            assert np.allclose(truth, test, rtol=RTOL, atol=ATOL), msg


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
