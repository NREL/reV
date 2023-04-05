# -*- coding: utf-8 -*-
"""
QA/QC tests
"""
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from reV import TESTDATADIR
from reV.qa_qc.summary import SummarizeH5, SummarizeSupplyCurve

H5_FILE = os.path.join(TESTDATADIR, 'gen_out', 'ri_wind_gen_profiles_2010.h5')
SC_TABLE = os.path.join(TESTDATADIR, 'sc_out', 'sc_full_out_1.csv')
SUMMARY_DIR = os.path.join(TESTDATADIR, 'qa_qc')


@pytest.mark.parametrize('dataset', ['cf_mean', 'cf_profile', None])
def test_summarize(dataset):
    """Run QA/QC Summarize and compare with baseline"""

    summary = SummarizeH5(H5_FILE)

    if dataset is None:
        baseline = os.path.join(SUMMARY_DIR,
                                'ri_wind_gen_profiles_2010_summary.csv')
        baseline = pd.read_csv(baseline)
        test = summary.summarize_means()
    elif dataset == 'cf_mean':
        baseline = os.path.join(SUMMARY_DIR, 'cf_mean_summary.csv')
        baseline = pd.read_csv(baseline, index_col=0)
        test = summary.summarize_dset(
            dataset, process_size=None, max_workers=1)
    elif dataset == 'cf_profile':
        baseline = os.path.join(SUMMARY_DIR, 'cf_profile_summary.csv')
        baseline = pd.read_csv(baseline, index_col=0)
        test = summary.summarize_dset(
            dataset, process_size=None, max_workers=1)

    test = test.fillna('None')
    baseline = baseline.fillna('None')

    assert_frame_equal(test, baseline, check_dtype=False, atol=1e-5,
                       check_index_type=False)


def test_sc_summarize():
    """Run QA/QC Summarize and compare with baseline"""
    test = SummarizeSupplyCurve(SC_TABLE).supply_curve_summary()
    baseline = os.path.join(SUMMARY_DIR,
                            'sc_full_out_1_summary.csv')

    if os.path.exists(baseline):
        baseline = pd.read_csv(baseline, index_col=0)
    else:
        test.to_csv(baseline)

    assert_frame_equal(test, baseline, check_dtype=False,
                       check_index_type=False)


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
