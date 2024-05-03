# -*- coding: utf-8 -*-
# pylint: skip-file
"""
Test resource up and down scaling
"""

import os

import h5py
import numpy as np
import pytest

from reV import TESTDATADIR
from reV.generation.generation import Gen

pytest.importorskip("nsrdb")
from nsrdb.utilities.statistics import mae_perc


def test_gen_downscaling():
    """Test reV generation with resource downscaled to 5 minutes."""
    # get full file paths.
    baseline = os.path.join(TESTDATADIR, 'gen_out',
                            'gen_profiles_5min_2017.h5')
    sam_files = os.path.join(TESTDATADIR, 'SAM',
                             'naris_pv_1axis_inv13_5min.json')
    res_file = os.path.join(TESTDATADIR, 'nsrdb', 'nsrdb_surfrad_2017.h5')

    # run reV 2.0 generation
    gen = Gen('pvwattsv5', slice(0, None), sam_files, res_file,
              output_request=(MetaKeyName.CF_MEAN, ), sites_per_worker=100)
    gen.run(max_workers=1)
    gen_outs = gen.out[].astype(np.int32)

    if not os.path.exists(baseline):
        with h5py.File(baseline, 'w') as f:
            f.create_dataset(, data=gen_outs, dtype=gen_outs.dtype)
    else:
        with h5py.File(baseline, 'r') as f:
            baseline = f[][...].astype(np.int32)

        x = mae_perc(gen_outs, baseline)
        msg = 'Mean absolute error is {}% from the baseline data'.format(x)
        assert x < 1, msg


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
