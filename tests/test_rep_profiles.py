# -*- coding: utf-8 -*-
"""reVX representative profile tests.
"""
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from rex.resource import Resource

from reV import TESTDATADIR
from reV.rep_profiles.rep_profiles import (
    RegionRepProfile,
    RepProfiles,
    RepresentativeMethods,
)
from reV.utilities import MetaKeyName

GEN_FPATH = os.path.join(TESTDATADIR, 'gen_out/gen_ri_pv_2012_x000.h5')


def test_rep_region_interval():
    """Test the rep profile with a weird interval of gids"""
    sites = np.arange(40) * 2
    rev_summary = pd.DataFrame({MetaKeyName.GEN_GIDS: sites,
                                MetaKeyName.RES_GIDS: sites})
    r = RegionRepProfile(GEN_FPATH, rev_summary, weight=None)
    assert r.i_reps[0] == 14


def test_rep_methods():
    """Test integrated rep methods against baseline rep profile result"""
    sites = np.arange(100)
    rev_summary = pd.DataFrame({MetaKeyName.GEN_GIDS: sites,
                                MetaKeyName.RES_GIDS: sites})

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
    rev_summary = pd.DataFrame({MetaKeyName.GEN_GIDS: sites,
                                MetaKeyName.RES_GIDS: sites})
    r = RegionRepProfile(GEN_FPATH, rev_summary, weight=None)

    meanoid = RepresentativeMethods.meanoid(r.source_profiles)

    with Resource(GEN_FPATH) as res:
        truth_profiles = res['cf_profile', :, sites]
    truth = truth_profiles.mean(axis=1).reshape(meanoid.shape)
    assert np.allclose(meanoid, truth)


def test_weighted_meanoid():
    """Test a meanoid weighted by gid_counts vs. a non-weighted meanoid."""

    sites = np.arange(100)
    rev_summary = pd.DataFrame({MetaKeyName.GEN_GIDS: sites,
                                MetaKeyName.RES_GIDS: sites,
                                MetaKeyName.GID_COUNTS: [1] * 50 + [0] * 50})
    r = RegionRepProfile(GEN_FPATH, rev_summary)
    weights = r._get_region_attr(r._rev_summary, MetaKeyName.GID_COUNTS)

    w_meanoid = RepresentativeMethods.meanoid(r.source_profiles,
                                              weights=weights)

    sites = np.arange(50)
    rev_summary = pd.DataFrame({MetaKeyName.GEN_GIDS: sites,
                                MetaKeyName.RES_GIDS: sites})
    r = RegionRepProfile(GEN_FPATH, rev_summary, weight=None)

    meanoid = RepresentativeMethods.meanoid(r.source_profiles, weights=None)

    assert np.allclose(meanoid, w_meanoid)


def test_integrated():
    """Test a multi-region rep profile calc serial vs. parallel and against
    baseline results."""
    sites = np.arange(100)
    ones = np.ones((100,))
    zeros = np.zeros((100,))
    regions = (['r0'] * 7) + (['r1'] * 33) + (['r2'] * 60)
    timezone = np.random.choice([-4, -5, -6, -7], 100)
    rev_summary = pd.DataFrame({MetaKeyName.GEN_GIDS: sites,
                                MetaKeyName.RES_GIDS: sites,
                                'res_class': zeros,
                                'weight': ones,
                                'region': regions,
                                MetaKeyName.TIMEZONE: timezone})
    rp = RepProfiles(GEN_FPATH, rev_summary, 'region', weight='weight')
    rp.run(max_workers=1)
    p1, m1 = rp.profiles, rp.meta
    rp.run(max_workers=None)
    p2, m2 = rp.profiles, rp.meta

    assert np.allclose(m1['rep_res_gid'].values.astype(int),
                       m2['rep_res_gid'].values.astype(int))
    assert np.allclose(p1[0], p2[0])
    assert m1.loc[0, 'rep_res_gid'] == 4
    assert m1.loc[1, 'rep_res_gid'] == 15
    assert m1.loc[2, 'rep_res_gid'] == 60


def test_sc_points():
    """Test rep profiles for each SC point."""
    sites = np.arange(10)
    timezone = np.random.choice([-4, -5, -6, -7], 10)
    rev_summary = pd.DataFrame({MetaKeyName.SC_GID: sites,
                                MetaKeyName.GEN_GIDS: sites,
                                MetaKeyName.RES_GIDS: sites,
                                MetaKeyName.TIMEZONE: timezone})

    rp = RepProfiles(GEN_FPATH, rev_summary, MetaKeyName.SC_GID, weight=None)
    rp.run(max_workers=1)

    with Resource(GEN_FPATH) as res:
        truth = res['cf_profile', :, slice(0, 10)]

    assert np.allclose(rp.profiles[0], truth)


def test_agg_profile():
    """Test aggregation of weighted meanoid profile for each SC point."""
    # make up a rev aggregation summary to pull profiles from GEN_FPATH
    gen_gids = [[1, 2], [2, 3, 6], [10, 11, 12], [77, 73]]
    res_gids = [[10, 9], [0, 1, 2], [10, 11, 12], [54, 61]]
    gid_counts = [[10, 1], [50, 3, 1], [123, 432, 452], [50, 50]]
    gen_gids = [json.dumps(x) for x in gen_gids]
    res_gids = [json.dumps(x) for x in res_gids]
    gid_counts = [json.dumps(x) for x in gid_counts]
    timezone = np.random.choice([-4, -5, -6, -7], 4)
    rev_summary = pd.DataFrame({MetaKeyName.SC_GID: np.arange(4),
                                MetaKeyName.GEN_GIDS: gen_gids,
                                MetaKeyName.RES_GIDS: res_gids,
                                MetaKeyName.GID_COUNTS: gid_counts,
                                MetaKeyName.TIMEZONE: timezone})

    rp = RepProfiles(GEN_FPATH, rev_summary, MetaKeyName.SC_GID, cf_dset='cf_profile',
                     err_method=None)
    rp.run(scaled_precision=False, max_workers=1)

    for index in rev_summary.index:
        gen_gids = json.loads(rev_summary.loc[index, MetaKeyName.GEN_GIDS])
        res_gids = json.loads(rev_summary.loc[index, MetaKeyName.RES_GIDS])
        weights = np.array(json.loads(rev_summary.loc[index, MetaKeyName.GID_COUNTS]))

        with Resource(GEN_FPATH) as res:
            meta = res.meta

            raw_profiles = []
            for gid in res_gids:
                iloc = np.where(meta[MetaKeyName.GID] == gid)[0][0]
                prof = np.expand_dims(res['cf_profile', :, iloc], 1)
                raw_profiles.append(prof)

            last = raw_profiles[-1].flatten()
            raw_profiles = np.hstack(raw_profiles)

        assert np.allclose(raw_profiles[:, -1], last)

        truth = raw_profiles * weights
        assert len(truth) == len(raw_profiles)
        truth = truth.sum(axis=1)
        assert len(truth) == len(raw_profiles)
        truth = truth / weights.sum()

        assert np.allclose(rp.profiles[0][:, index], truth)

    passthrough_cols = [MetaKeyName.GEN_GIDS, MetaKeyName.RES_GIDS, MetaKeyName.GID_COUNTS]
    for col in passthrough_cols:
        assert col in rp.meta

    assert_frame_equal(
        rev_summary[passthrough_cols],
        rp.meta[passthrough_cols]
    )


@pytest.mark.parametrize("use_weights", [True, False])
def test_many_regions(use_weights):
    """Test multiple complicated regions."""
    sites = np.arange(100)
    zeros = np.zeros((100,))
    region1 = (['r0'] * 7) + (['r1'] * 33) + (['r2'] * 60)
    region2 = (['a0'] * 20) + (['b1'] * 10) + (['c2'] * 20) + (['d3'] * 50)
    timezone = np.random.choice([-4, -5, -6, -7], 100)
    rev_summary = pd.DataFrame({MetaKeyName.GEN_GIDS: sites,
                                MetaKeyName.RES_GIDS: sites,
                                'res_class': zeros,
                                'region1': region1,
                                'region2': region2,
                                'weight': sites + 1,
                                MetaKeyName.TIMEZONE: timezone})
    reg_cols = ['region1', 'region2']
    if use_weights:
        rp = RepProfiles(GEN_FPATH, rev_summary, reg_cols, weight="weight")
    else:
        rp = RepProfiles(GEN_FPATH, rev_summary, reg_cols, weight=None)
    rp.run()

    assert rp.profiles[0].shape == (17520, 6)
    assert len(rp.meta) == 6

    for r1 in set(region1):
        assert r1 in rp.meta['region1'].values

    for r2 in set(region2):
        assert r2 in rp.meta['region2'].values


def test_many_regions_with_list_weights():
    """Test multiple complicated regions with multiple weights per row."""
    sites = [list(np.random.choice(np.arange(100), np.random.randint(10) + 10))
             for __ in np.arange(100)]
    weights = [str(list(np.random.choice(np.arange(100), len(row))))
               for row in sites]
    sites = [str(row) for row in sites]
    zeros = np.zeros((100,))
    region1 = (['r0'] * 7) + (['r1'] * 33) + (['r2'] * 60)
    region2 = (['a0'] * 20) + (['b1'] * 10) + (['c2'] * 20) + (['d3'] * 50)
    timezone = np.random.choice([-4, -5, -6, -7], 100)
    rev_summary = pd.DataFrame({MetaKeyName.GEN_GIDS: sites,
                                MetaKeyName.RES_GIDS: sites,
                                'res_class': zeros,
                                'region1': region1,
                                'region2': region2,
                                'weights': weights,
                                MetaKeyName.TIMEZONE: timezone})
    reg_cols = ['region1', 'region2']
    rp = RepProfiles(GEN_FPATH, rev_summary, reg_cols, weight='weights')
    rp.run()

    assert rp.profiles[0].shape == (17520, 6)
    assert len(rp.meta) == 6

    for r1 in set(region1):
        assert r1 in rp.meta['region1'].values

    for r2 in set(region2):
        assert r2 in rp.meta['region2'].values


def test_write_to_file():
    """Test rep profiles with file write."""
    with tempfile.TemporaryDirectory() as td:
        sites = np.arange(100)
        zeros = np.zeros((100,))
        regions = (['r0'] * 7) + (['r1'] * 33) + (['r2'] * 60)
        timezone = np.random.choice([-4, -5, -6, -7], 100)
        rev_summary = pd.DataFrame({MetaKeyName.GEN_GIDS: sites,
                                    MetaKeyName.RES_GIDS: sites,
                                    'res_class': zeros,
                                    'region': regions,
                                    MetaKeyName.TIMEZONE: timezone})
        fout = os.path.join(td, 'temp_rep_profiles.h5')
        rp = RepProfiles(GEN_FPATH, rev_summary, 'region', n_profiles=3,
                         weight=None)
        rp.run(fout=fout)
        with Resource(fout) as res:
            disk_profiles = res['rep_profiles_0']
            disk_meta = res.meta
            assert 'rep_profiles_2' in res.datasets
            test = np.array_equal(res['rep_profiles_0'],
                                  res['rep_profiles_1'])
            assert not test

        assert np.allclose(rp.profiles[0], disk_profiles)
        assert len(disk_meta) == 3

        for i in rp.meta.index:
            v1 = json.loads(rp.meta.loc[i, 'rep_gen_gid'])
            v2 = json.loads(disk_meta.loc[i, 'rep_gen_gid'])
            assert v1 == v2


def test_file_options():
    """Test rep profiles with file write."""
    with tempfile.TemporaryDirectory() as td:
        sites = np.arange(100)
        zeros = np.zeros((100,))
        regions = (['r0'] * 7) + (['r1'] * 33) + (['r2'] * 60)
        timezone = np.random.choice([-4, -5, -6, -7], 100)
        rev_summary = pd.DataFrame({MetaKeyName.GEN_GIDS: sites,
                                    MetaKeyName.RES_GIDS: sites,
                                    'res_class': zeros,
                                    'region': regions,
                                    MetaKeyName.TIMEZONE: timezone})
        fout = os.path.join(td, 'temp_rep_profiles.h5')
        rp = RepProfiles(GEN_FPATH, rev_summary, 'region', n_profiles=3,
                         weight=None)
        rp.run(fout=fout, save_rev_summary=False, scaled_precision=True)
        with Resource(fout) as res:
            dtype = res.get_dset_properties('rep_profiles_0')[1]
            attrs = res.get_attrs('rep_profiles_0')
            disk_profiles = res['rep_profiles_0']
            disk_dsets = res.datasets

        assert np.issubdtype(dtype, np.integer)
        assert attrs['scale_factor'] == 1000
        assert np.allclose(rp.profiles[0], disk_profiles)
        assert 'rev_summary' not in disk_dsets


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
