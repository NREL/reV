# -*- coding: utf-8 -*-
"""
Aggregation tests
"""
import numpy as np
import os
from pandas.testing import assert_frame_equal
import pytest

from reV.supply_curve.aggregation import Aggregation
from reV import TESTDATADIR

from rex.resource import Resource

EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
GEN = os.path.join(TESTDATADIR, 'gen_out/ri_wind_gen_profiles_2010.h5')
TM_DSET = 'techmap_wtk'
AGG_DSET = ('cf_mean', 'cf_profile')

EXCL_DICT = {'ri_srtm_slope': {'inclusion_range': (None, 5),
                               'exclude_nodata': True},
             'ri_padus': {'exclude_values': [1],
                          'exclude_nodata': True},
             'ri_reeds_regions': {'inclusion_range': (None, 400),
                                  'exclude_nodata': True}}

RTOL = 0.01
ATOL = 0.001


def check_agg(agg_out, baseline_h5):
    """
    Compare agg_out to baseline data

    Parameters
    ----------
    agg_out : dict
        Aggregation data
    baseline_h5 : str
        h5 file containing baseline data
    """
    with Resource(baseline_h5) as f:
        for dset, test in agg_out.items():
            truth = f[dset]
            if dset == 'meta':
                for c in ['source_gids', 'gid_counts']:
                    test[c] = test[c].astype(str)

                assert_frame_equal(truth, test, check_dtype=False)
            else:
                assert np.allclose(truth, test, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(('excl_dict', 'baseline_name'),
                         [(None, 'baseline_agg.h5'),
                         (EXCL_DICT, 'baseline_agg_excl.h5')])
def test_aggregation_seriel(excl_dict, baseline_name):
    """
    test aggregation run in seriel
    """
    agg_out = Aggregation.run(EXCL, GEN, TM_DSET, *AGG_DSET,
                              excl_dict=excl_dict, max_workers=1)
    baseline_h5 = os.path.join(TESTDATADIR, "sc_out", baseline_name)
    check_agg(agg_out, baseline_h5)


@pytest.mark.parametrize(('excl_dict', 'baseline_name'),
                         [(None, 'baseline_agg.h5'),
                          (EXCL_DICT, 'baseline_agg_excl.h5')])
def test_aggregation_parallel(excl_dict, baseline_name):
    """
    test aggregation run in parallel
    """
    agg_out = Aggregation.run(EXCL, GEN, TM_DSET, *AGG_DSET,
                              excl_dict=excl_dict, max_workers=None,
                              chunk_point_len=10)
    baseline_h5 = os.path.join(TESTDATADIR, "sc_out", baseline_name)
    check_agg(agg_out, baseline_h5)


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
