# -*- coding: utf-8 -*-
"""reVX representative profile tests.
"""
import os
import pytest
import pandas as pd
import numpy as np

from reV.rep_profiles.rep_profiles import Region, RepProfiles, Methods
from reV import TESTDATADIR
from reV.handlers.resource import Resource


GEN_FPATH = os.path.join(TESTDATADIR, 'gen_out/gen_ri_pv_2012_x000.h5')


def test_rep_region_interval():
    """Test the rep profile with a weird interval of gids"""
    sites = np.arange(40) * 2
    rev_summary = pd.DataFrame({'gen_gids': sites,
                                'res_gids': sites})
    r = Region(GEN_FPATH, rev_summary)
    assert r.i_rep == 14


def test_rep_methods():
    """Test integrated rep methods against baseline rep profile result"""
    sites = np.arange(100)
    rev_summary = pd.DataFrame({'gen_gids': sites,
                                'res_gids': sites})

    r = Region(GEN_FPATH, rev_summary, rep_method='meanoid', err_method='rmse')
    assert r.i_rep == 15

    r = Region(GEN_FPATH, rev_summary, rep_method='meanoid', err_method='mbe')
    assert r.i_rep == 13

    r = Region(GEN_FPATH, rev_summary, rep_method='meanoid', err_method='mae')
    assert r.i_rep == 15

    r = Region(GEN_FPATH, rev_summary, rep_method='median', err_method='rmse')
    assert r.i_rep == 15

    r = Region(GEN_FPATH, rev_summary, rep_method='median', err_method='mbe')
    assert r.i_rep == 13


def test_meanoid():
    """Test the simple meanoid method"""
    sites = np.arange(100)
    rev_summary = pd.DataFrame({'gen_gids': sites,
                                'res_gids': sites})
    r = Region(GEN_FPATH, rev_summary)
    gids = r._get_region_attr(r._rev_summary, 'gen_gids')
    all_profiles = r._get_profiles(gids)

    meanoid = Methods.meanoid(all_profiles)

    with Resource(GEN_FPATH) as res:
        truth_profiles = res['cf_profile', :, sites]
    truth = truth_profiles.mean(axis=1).reshape(meanoid.shape)
    assert np.allclose(meanoid, truth)


def test_mult_regions():
    """Test a multi-region rep profile calc serial vs. parallel and against
    baseline results."""
    sites = np.arange(100)
    zeros = np.zeros((100,))
    regions = (['r0'] * 7) + (['r1'] * 33) + (['r2'] * 60)
    rev_summary = pd.DataFrame({'gen_gids': sites,
                                'res_gids': sites,
                                'res_class': zeros,
                                'region': regions})
    p1, m1 = RepProfiles.run(GEN_FPATH, rev_summary, 'region', parallel=True)
    p2, m2 = RepProfiles.run(GEN_FPATH, rev_summary, 'region', parallel=False)

    assert np.allclose(m1['rep_res_gid'].values, m2['rep_res_gid'].values)
    assert np.allclose(p1, p2)
    assert m1.loc[0, 'rep_res_gid'] == 4
    assert m1.loc[1, 'rep_res_gid'] == 15
    assert m1.loc[2, 'rep_res_gid'] == 60


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
