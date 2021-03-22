# -*- coding: utf-8 -*-
# pylint: disable=all
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
                truth.index.name = None
                for c in ['source_gids', 'gid_counts']:
                    test[c] = test[c].astype(str)

                assert_frame_equal(truth, test, check_dtype=False, rtol=0.0001)
            else:
                assert np.allclose(truth, test, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(('excl_dict', 'baseline_name'),
                         [(None, 'baseline_agg.h5'),
                         (EXCL_DICT, 'baseline_agg_excl.h5')])
def test_aggregation_serial(excl_dict, baseline_name):
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


@pytest.mark.parametrize(('pre_extract_inclusions', 'max_workers'),
                         [(True, 1),
                          (True, None),
                          (False, 1),
                          (False, None)])
def test_pre_extract_inclusions(pre_extract_inclusions, max_workers):
    """
    test aggregation w/ and w/out pre-extracting inclusions
    """
    agg_out = Aggregation.run(EXCL, GEN, TM_DSET, *AGG_DSET,
                              excl_dict=EXCL_DICT, max_workers=max_workers,
                              chunk_point_len=10,
                              pre_extract_inclusions=pre_extract_inclusions)
    baseline_h5 = os.path.join(TESTDATADIR, "sc_out", "baseline_agg_excl.h5")
    check_agg(agg_out, baseline_h5)


@pytest.mark.parametrize('excl_dict', [None, EXCL_DICT])
def test_gid_counts(excl_dict):
    """
    test counting of exclusion gids during aggregation
    """
    agg_out = Aggregation.run(EXCL, GEN, TM_DSET, *AGG_DSET,
                              excl_dict=excl_dict, max_workers=None,
                              chunk_point_len=10)

    for i, row in agg_out['meta'].iterrows():
        n_gids = row['n_gids']
        gid_counts = np.sum(row['gid_counts'])
        area = row['area_sq_km']

        msg = ('For sc_gid {}: the sum of gid_counts ({}), does not match '
               'n_gids ({})'.format(i, n_gids, gid_counts))
        assert n_gids == gid_counts, msg
        assert area <= n_gids * 0.0081


def compute_mean_wind_dirs(res_path, dset, gids, fracs):
    """
    Compute mean wind directions for given dset and gids
    """
    with Resource(res_path) as f:
        wind_dirs = np.radians(f[dset, :, gids])

    sin = np.mean(np.sin(wind_dirs) * fracs, axis=1)
    cos = np.mean(np.cos(wind_dirs) * fracs, axis=1)
    mean_wind_dirs = np.degrees(np.arctan2(sin, cos))

    mask = mean_wind_dirs < 0
    mean_wind_dirs[mask] += 360

    return mean_wind_dirs


@pytest.mark.parametrize('excl_dict', [None, EXCL_DICT])
def test_mean_wind_dirs(excl_dict):
    """
    Test mean wind direction aggregation
    """
    RES = os.path.join(TESTDATADIR, 'wtk/wind_dirs_2012.h5')
    DSET = 'winddirection_100m'
    agg_out = Aggregation.run(EXCL, RES, TM_DSET, DSET,
                              agg_method='wind_dir',
                              excl_dict=excl_dict, max_workers=None,
                              chunk_point_len=10)

    mean_wind_dirs = agg_out['winddirection_100m']
    out_meta = agg_out['meta']
    for i, row in out_meta.iterrows():
        test = mean_wind_dirs[:, i]

        gids = row['source_gids']
        fracs = row['gid_counts'] / row['n_gids']

        truth = compute_mean_wind_dirs(RES, DSET, gids, fracs)

        msg = ('mean wind directions do now match for sc_gid {}'
               .format(i))
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
