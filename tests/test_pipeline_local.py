# -*- coding: utf-8 -*-
"""Test for local pipeline execution.

Created on Thu Oct  3 13:24:16 2019

@author: gbuster
"""
import os
import pytest
import numpy as np
import tempfile

from rex import Resource
from rex.utilities.loggers import LOGGERS
from reV import TESTDATADIR
from reV.pipeline.pipeline import Pipeline


def test_pipeline_local():
    """Test the reV pipeline execution on a local machine."""
    with tempfile.TemporaryDirectory() as TEMP_DIR:
        pipeline_dir = os.path.join(TESTDATADIR, 'pipeline/')
        out_dir = os.path.join(TEMP_DIR, 'outputs/')
        fpipeline = os.path.join(TEMP_DIR, 'config_pipeline.json')
        fbaseline = os.path.join(pipeline_dir,
                                 'baseline_pipeline_multi-year.h5')

        Pipeline.run(fpipeline, monitor=True)

        fpath_out = Pipeline.parse_previous(out_dir, 'multi-year',
                                            target_module='multi-year')[0]

        dsets = ['generation/cf_mean-means', 'econ/lcoe_fcr-means']
        with Resource(fpath_out, 'r') as f_new:
            with Resource(fbaseline, 'r') as f_base:
                for dset in dsets:
                    if dset in ['meta', 'time_index']:
                        test = f_new.h5[dset]
                        truth = f_base.h5[dset]
                    else:
                        test = f_new[dset]
                        truth = f_base[dset]

                    msg = 'Local pipeline failed for "{}"'.format(dset)
                    assert np.allclose(truth, test, rtol=0.01, atol=0), msg

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
