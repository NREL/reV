# -*- coding: utf-8 -*-
"""reVX representative profile tests.
"""
import os
import pytest
import pandas as pd
import numpy as np
import json

from reV.rep_profiles.rep_profiles import (RegionRepProfile, RepProfiles,
                                           RepresentativeMethods,
                                           AggregatedRepProfiles)
from reV import TESTDATADIR
from reV.handlers.resource import Resource


GEN_FPATH = os.path.join(TESTDATADIR, 'gen_out/gen_ri_pv_2012_x000.h5')
PURGE_OUT = True


def test_rep_region_interval():
    """Test the rep profile with a weird interval of gids"""
    sites = np.arange(40) * 2
    rev_summary = pd.DataFrame({'gen_gids': sites,
                                'res_gids': sites})
    r = RegionRepProfile(GEN_FPATH, rev_summary, weight=None)
    assert r.i_reps[0] == 14


def test_rep_methods():
    """Test integrated rep methods against baseline rep profile result"""
    sites = np.arange(100)
    rev_summary = pd.DataFrame({'gen_gids': sites,
                                'res_gids': sites})

    r = RegionRepProfile(GEN_FPATH, rev_summary, rep_method='meanoid',
                         err_method='rmse', weight=None)
    assert r.i_reps[0] == 15

    r = RegionRepProfile(GEN_FPATH, rev_summary, rep_method='meanoid',
                         err_method='mbe', weight=None)
    assert r.i_reps[0] == 13

    r = RegionRepProfile(GEN_FPATH, rev_summary, rep_method='meanoid',
                         err_method='mae', weight=None)
    assert r.i_reps[0] == 15

    r = RegionRepProfile(GEN_FPATH, rev_summary, rep_method='median',
                         err_method='rmse', weight=None)
    assert r.i_reps[0] == 15

    r = RegionRepProfile(GEN_FPATH, rev_summary, rep_method='median',
                         err_method='mbe', weight=None)
    assert r.i_reps[0] == 13


def test_meanoid():
    """Test the simple meanoid method"""
    sites = np.arange(100)
    rev_summary = pd.DataFrame({'gen_gids': sites,
                                'res_gids': sites})
    r = RegionRepProfile(GEN_FPATH, rev_summary)

    meanoid = RepresentativeMethods.meanoid(r.source_profiles)

    with Resource(GEN_FPATH) as res:
        truth_profiles = res['cf_profile', :, sites]
    truth = truth_profiles.mean(axis=1).reshape(meanoid.shape)
    assert np.allclose(meanoid, truth)


def test_weighted_meanoid():
    """Test a meanoid weighted by gid_counts vs. a non-weighted meanoid."""

    sites = np.arange(100)
    rev_summary = pd.DataFrame({'gen_gids': sites,
                                'res_gids': sites,
                                'gid_counts': [1] * 50 + [0] * 50})
    r = RegionRepProfile(GEN_FPATH, rev_summary)
    weights = r._get_region_attr(r._rev_summary, 'gid_counts')

    w_meanoid = RepresentativeMethods.meanoid(r.source_profiles,
                                              weights=weights)

    sites = np.arange(50)
    rev_summary = pd.DataFrame({'gen_gids': sites,
                                'res_gids': sites})
    r = RegionRepProfile(GEN_FPATH, rev_summary)

    meanoid = RepresentativeMethods.meanoid(r.source_profiles, weights=None)

    assert np.allclose(meanoid, w_meanoid)


def test_integrated():
    """Test a multi-region rep profile calc serial vs. parallel and against
    baseline results."""
    sites = np.arange(100)
    ones = np.ones((100,))
    zeros = np.zeros((100,))
    regions = (['r0'] * 7) + (['r1'] * 33) + (['r2'] * 60)
    rev_summary = pd.DataFrame({'gen_gids': sites,
                                'res_gids': sites,
                                'res_class': zeros,
                                'weight': ones,
                                'region': regions})
    p1, m1, _ = RepProfiles.run(GEN_FPATH, rev_summary, 'region',
                                max_workers=1, weight='weight')
    p2, m2, _ = RepProfiles.run(GEN_FPATH, rev_summary, 'region',
                                max_workers=None, weight='weight')

    assert np.allclose(m1['rep_res_gid'].values.astype(int),
                       m2['rep_res_gid'].values.astype(int))
    assert np.allclose(p1[0], p2[0])
    assert m1.loc[0, 'rep_res_gid'] == 4
    assert m1.loc[1, 'rep_res_gid'] == 15
    assert m1.loc[2, 'rep_res_gid'] == 60


def test_sc_points():
    """Test rep profiles for each SC point."""
    sites = np.arange(10)
    rev_summary = pd.DataFrame({'sc_gid': sites,
                                'gen_gids': sites,
                                'res_gids': sites})
    p1 = RepProfiles.run(GEN_FPATH, rev_summary, 'sc_gid',
                         weight=None, max_workers=1)[0]

    with Resource(GEN_FPATH) as res:
        truth = res['cf_profile', :, slice(0, 10)]

    assert np.allclose(p1[0], truth)


def test_agg_profile():
    """Test aggregation of weighted meanoid profile for each SC point."""

    gen_fpath = os.path.join(TESTDATADIR, 'offshore/ri_offshore_archive.h5')

    rev_sc_fpath = os.path.join(TESTDATADIR, 'sc_out/ri_wind_farm_sc.csv')
    rev_summary = pd.read_csv(rev_sc_fpath, index_col=0).iloc[0:2]

    profiles = AggregatedRepProfiles.run(gen_fpath, rev_summary,
                                         cf_dset='cf_profile',
                                         scaled_precision=False,
                                         max_workers=None)[0]

    for index in rev_summary.index:

        gen_gids = json.loads(rev_summary.loc[index, 'gen_gids'])
        weights = np.array(json.loads(rev_summary.loc[index, 'gid_counts']))

        with Resource(gen_fpath) as res:
            raw_profiles = res['cf_profile', :, gen_gids]
            last = res['cf_profile', :, gen_gids[-1]]

        assert np.allclose(raw_profiles[:, -1], last)

        truth = raw_profiles * weights
        assert len(truth) == 8760
        truth = truth.sum(axis=1)
        assert len(truth) == 8760
        truth = truth / weights.sum()

        assert np.allclose(profiles[0][:, index], truth)


def test_many_regions():
    """Test multiple complicated regions."""
    sites = np.arange(100)
    zeros = np.zeros((100,))
    region1 = (['r0'] * 7) + (['r1'] * 33) + (['r2'] * 60)
    region2 = (['a0'] * 20) + (['b1'] * 10) + (['c2'] * 20) + (['d3'] * 50)
    rev_summary = pd.DataFrame({'gen_gids': sites,
                                'res_gids': sites,
                                'res_class': zeros,
                                'region1': region1,
                                'region2': region2})
    reg_cols = ['region1', 'region2']
    p1, m1, _ = RepProfiles.run(GEN_FPATH, rev_summary, reg_cols,
                                weight=None)

    assert p1[0].shape == (17520, 6)
    assert len(m1) == 6

    for r1 in set(region1):
        assert r1 in m1['region1'].values

    for r2 in set(region2):
        assert r2 in m1['region2'].values


def test_write_to_file():
    """Test rep profiles with file write."""

    sites = np.arange(100)
    zeros = np.zeros((100,))
    regions = (['r0'] * 7) + (['r1'] * 33) + (['r2'] * 60)
    rev_summary = pd.DataFrame({'gen_gids': sites,
                                'res_gids': sites,
                                'res_class': zeros,
                                'region': regions})
    fout = os.path.join(TESTDATADIR, 'sc_out/temp_rep_profiles.h5')
    p1, m1, _ = RepProfiles.run(GEN_FPATH, rev_summary, 'region',
                                fout=fout, n_profiles=3, weight=None)
    with Resource(fout) as res:
        disk_profiles = res['rep_profiles_0']
        disk_meta = res.meta
        assert 'rep_profiles_2' in res.dsets
        assert not np.array_equal(res['rep_profiles_0'], res['rep_profiles_1'])

    assert np.allclose(p1[0], disk_profiles)
    assert len(disk_meta) == 3

    for i in m1.index:
        v1 = json.loads(m1.loc[i, 'rep_gen_gid'])
        v2 = json.loads(disk_meta.loc[i, 'rep_gen_gid'])
        assert v1 == v2

    if PURGE_OUT:
        os.remove(fout)


def test_file_options():
    """Test rep profiles with file write."""

    sites = np.arange(100)
    zeros = np.zeros((100,))
    regions = (['r0'] * 7) + (['r1'] * 33) + (['r2'] * 60)
    rev_summary = pd.DataFrame({'gen_gids': sites,
                                'res_gids': sites,
                                'res_class': zeros,
                                'region': regions})
    fout = os.path.join(TESTDATADIR, 'sc_out/temp_rep_profiles.h5')
    p1, _, _ = RepProfiles.run(GEN_FPATH, rev_summary, 'region',
                               fout=fout, n_profiles=3,
                               save_rev_summary=False,
                               scaled_precision=True,
                               weight=None)
    with Resource(fout) as res:
        dtype = res.get_dset_properties('rep_profiles_0')[1]
        attrs = res.get_attrs('rep_profiles_0')
        disk_profiles = res['rep_profiles_0']
        disk_dsets = res.dsets

    assert np.issubdtype(dtype, np.integer)
    assert attrs['scale_factor'] == 1000
    assert np.allclose(p1[0], disk_profiles)
    assert 'rev_summary' not in disk_dsets

    if PURGE_OUT:
        os.remove(fout)


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
