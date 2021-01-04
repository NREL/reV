# -*- coding: utf-8 -*-
"""Test for local pipeline execution.

Created on Thu Oct  3 13:24:16 2019

@author: gbuster
"""
import h5py
import os
import pytest
import numpy as np
import shutil

from rex.utilities.loggers import LOGGERS
from reV import TESTDATADIR
from reV.pipeline.pipeline import Pipeline

PURGE_OUT = True


def test_pipeline_local():
    """Test the reV pipeline execution on a local machine."""

    pipeline_dir = os.path.join(TESTDATADIR, 'pipeline/')
    log_dir = os.path.join(pipeline_dir, 'logs/')
    out_dir = os.path.join(pipeline_dir, 'outputs/')
    fpipeline = os.path.join(pipeline_dir, 'config_pipeline.json')
    fbaseline = os.path.join(pipeline_dir, 'baseline_pipeline_multi-year.h5')

    Pipeline.run(fpipeline, monitor=True)

    fpath_out = Pipeline.parse_previous(out_dir, 'multi-year',
                                        target_module='multi-year')[0]

    dsets = ['generation/cf_mean-means', 'econ/lcoe_fcr-means']
    with h5py.File(fpath_out, 'r') as f_new:
        with h5py.File(fbaseline, 'r') as f_base:
            for dset in dsets:
                msg = 'Local pipeline failed for "{}"'.format(dset)
                assert np.allclose(f_new[dset][...], f_base[dset][...]), msg

    if PURGE_OUT:
        shutil.rmtree(log_dir)
        shutil.rmtree(out_dir)

    LOGGERS.clear()


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
