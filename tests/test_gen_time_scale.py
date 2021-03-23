# -*- coding: utf-8 -*-
# pylint: skip-file
"""
Test resource up and down scaling
"""

import os
import h5py
import pytest
import numpy as np

from reV.generation.generation import Gen
from reV import TESTDATADIR


def test_time_index_step():
    """Test reV time_index_step option to upscale resource"""
    # get full file paths.
    baseline = os.path.join(TESTDATADIR, 'gen_out',
                            'gen_profiles_hr_2017.h5')
    sam_files = os.path.join(TESTDATADIR, 'SAM',
                             'naris_pv_1axis_inv13_hr.json')
    res_file = os.path.join(TESTDATADIR, 'nsrdb', 'nsrdb_surfrad_2017.h5')

    # run reV 2.0 generation
    gen = Gen.reV_run('pvwattsv5', slice(0, None), sam_files, res_file,
                      output_request=('cf_mean', 'cf_profile'),
                      max_workers=1, sites_per_worker=100,
                      fout=None)
    gen_outs = gen.out['cf_profile'].astype(np.int32)

    if not os.path.exists(baseline):
        with h5py.File(baseline, 'w') as f:
            f.create_dataset('cf_profile', data=gen_outs, dtype=gen_outs.dtype)
    else:
        with h5py.File(baseline, 'r') as f:
            baseline = f['cf_profile'][...].astype(np.int32)

        assert np.allclose(gen_outs, baseline)


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
