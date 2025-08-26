# -*- coding: utf-8 -*-
"""reVX representative profile tests.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from rex.resource import Resource

from reV import TESTDATADIR
from reV.cli import main
from reV.rep_profiles.rep_profiles import (
    RegionRepProfile,
    RepProfiles,
    RepresentativeMethods,
)
from reV.utilities import ResourceMetaField, SupplyCurveField, ModuleName

GEN_FPATH = os.path.join(TESTDATADIR, "gen_out/gen_ri_pv_2012_x000.h5")


def test_rep_region_interval():
    """Test the rep profile with a weird interval of gids"""
    sites = np.arange(40) * 2
    rev_summary = pd.DataFrame({SupplyCurveField.GEN_GIDS: sites,
                                SupplyCurveField.RES_GIDS: sites})
    r = RegionRepProfile(GEN_FPATH, rev_summary, weight=None)
    assert r.i_reps[0] == 14


def test_rep_methods():
    """Test integrated rep methods against baseline rep profile result"""
    sites = np.arange(100)
    rev_summary = pd.DataFrame({SupplyCurveField.GEN_GIDS: sites,
                                SupplyCurveField.RES_GIDS: sites})

    r = RegionRepProfile(
        GEN_FPATH,
        rev_summary,
        rep_method="meanoid",
        err_method="rmse",
        weight=None,
    )
    assert r.i_reps[0] == 15

    r = RegionRepProfile(
        GEN_FPATH,
        rev_summary,
        rep_method="meanoid",
        err_method="mbe",
        weight=None,
    )
    assert r.i_reps[0] == 13

    r = RegionRepProfile(
        GEN_FPATH,
        rev_summary,
        rep_method="meanoid",
        err_method="mae",
        weight=None,
    )
    assert r.i_reps[0] == 15

    r = RegionRepProfile(
        GEN_FPATH,
        rev_summary,
        rep_method="median",
        err_method="rmse",
        weight=None,
    )
    assert r.i_reps[0] == 15

    r = RegionRepProfile(
        GEN_FPATH,
        rev_summary,
        rep_method="median",
        err_method="mbe",
        weight=None,
    )
    assert r.i_reps[0] == 13


def test_meanoid():
    """Test the simple meanoid method"""
    sites = np.arange(100)
    rev_summary = pd.DataFrame({SupplyCurveField.GEN_GIDS: sites,
                                SupplyCurveField.RES_GIDS: sites})
    r = RegionRepProfile(GEN_FPATH, rev_summary, weight=None)

    meanoid = RepresentativeMethods.meanoid(r.source_profiles)

    with Resource(GEN_FPATH) as res:
        truth_profiles = res["cf_profile", :, sites]
    truth = truth_profiles.mean(axis=1).reshape(meanoid.shape)
    assert np.allclose(meanoid, truth)


def test_weighted_meanoid():
    """Test a meanoid weighted by gid_counts vs. a non-weighted meanoid."""

    sites = np.arange(100)
    rev_summary = pd.DataFrame({SupplyCurveField.GEN_GIDS: sites,
                                SupplyCurveField.RES_GIDS: sites,
                                SupplyCurveField.GID_COUNTS: (
                                    [1] * 50 + [0] * 50
                                )})
    r = RegionRepProfile(GEN_FPATH, rev_summary)
    weights = r._get_region_attr(r._rev_summary, SupplyCurveField.GID_COUNTS)

    w_meanoid = RepresentativeMethods.meanoid(
        r.source_profiles, weights=weights
    )

    sites = np.arange(50)
    rev_summary = pd.DataFrame({SupplyCurveField.GEN_GIDS: sites,
                                SupplyCurveField.RES_GIDS: sites})
    r = RegionRepProfile(GEN_FPATH, rev_summary, weight=None)

    meanoid = RepresentativeMethods.meanoid(r.source_profiles, weights=None)

    assert np.allclose(meanoid, w_meanoid)


def test_integrated():
    """Test a multi-region rep profile calc serial vs. parallel and against
    baseline results."""
    sites = np.arange(100)
    ones = np.ones((100,))
    zeros = np.zeros((100,))
    regions = (["r0"] * 7) + (["r1"] * 33) + (["r2"] * 60)
    timezone = np.random.choice([-4, -5, -6, -7], 100)
    rev_summary = pd.DataFrame({SupplyCurveField.GEN_GIDS: sites,
                                SupplyCurveField.RES_GIDS: sites,
                                'res_class': zeros,
                                'weight': ones,
                                'region': regions,
                                SupplyCurveField.TIMEZONE: timezone})
    rp = RepProfiles(GEN_FPATH, rev_summary, 'region', weight='weight')
    rp.run(max_workers=1)
    p1, m1 = rp.profiles, rp.meta
    rp.run(max_workers=None)
    p2, m2 = rp.profiles, rp.meta

    assert np.allclose(
        m1["rep_res_gid"].values.astype(int),
        m2["rep_res_gid"].values.astype(int),
    )
    assert np.allclose(p1[0], p2[0])
    assert m1.loc[0, "rep_res_gid"] == 4
    assert m1.loc[1, "rep_res_gid"] == 15
    assert m1.loc[2, "rep_res_gid"] == 60


def test_sc_points():
    """Test rep profiles for each SC point."""
    sites = np.arange(10)
    timezone = np.random.choice([-4, -5, -6, -7], 10)
    rev_summary = pd.DataFrame({SupplyCurveField.SC_GID: sites,
                                SupplyCurveField.GEN_GIDS: sites,
                                SupplyCurveField.RES_GIDS: sites,
                                SupplyCurveField.TIMEZONE: timezone})

    rp = RepProfiles(GEN_FPATH, rev_summary,
                     SupplyCurveField.SC_GID, weight=None)
    rp.run(max_workers=1)

    with Resource(GEN_FPATH) as res:
        truth = res["cf_profile", :, slice(0, 10)]

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
    rev_summary = pd.DataFrame({SupplyCurveField.SC_GID: np.arange(4),
                                SupplyCurveField.GEN_GIDS: gen_gids,
                                SupplyCurveField.RES_GIDS: res_gids,
                                SupplyCurveField.GID_COUNTS: gid_counts,
                                SupplyCurveField.TIMEZONE: timezone})

    rp = RepProfiles(GEN_FPATH, rev_summary, SupplyCurveField.SC_GID,
                     cf_dset='cf_profile', err_method=None)
    rp.run(scaled_precision=False, max_workers=1)

    for index in rev_summary.index:
        gen_gids = json.loads(rev_summary.loc[index,
                                              SupplyCurveField.GEN_GIDS])
        res_gids = json.loads(rev_summary.loc[index,
                                              SupplyCurveField.RES_GIDS])
        weights = np.array(
            json.loads(rev_summary.loc[index, SupplyCurveField.GID_COUNTS]))

        with Resource(GEN_FPATH) as res:
            meta = res.meta

            raw_profiles = []
            for gid in res_gids:
                iloc = np.where(meta[ResourceMetaField.GID] == gid)[0][0]
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

    passthrough_cols = [SupplyCurveField.GEN_GIDS, SupplyCurveField.RES_GIDS,
                        SupplyCurveField.GID_COUNTS]
    for col in passthrough_cols:
        assert col in rp.meta

    assert_frame_equal(
        rev_summary[passthrough_cols], rp.meta[passthrough_cols]
    )


@pytest.mark.parametrize("use_weights", [True, False])
def test_many_regions(use_weights):
    """Test multiple complicated regions."""
    sites = np.arange(100)
    zeros = np.zeros((100,))
    region1 = (["r0"] * 7) + (["r1"] * 33) + (["r2"] * 60)
    region2 = (["a0"] * 20) + (["b1"] * 10) + (["c2"] * 20) + (["d3"] * 50)
    timezone = np.random.choice([-4, -5, -6, -7], 100)
    rev_summary = pd.DataFrame({SupplyCurveField.GEN_GIDS: sites,
                                SupplyCurveField.RES_GIDS: sites,
                                'res_class': zeros,
                                'region1': region1,
                                'region2': region2,
                                'weight': sites + 1,
                                SupplyCurveField.TIMEZONE: timezone})
    reg_cols = ['region1', 'region2']
    if use_weights:
        rp = RepProfiles(GEN_FPATH, rev_summary, reg_cols, weight="weight")
    else:
        rp = RepProfiles(GEN_FPATH, rev_summary, reg_cols, weight=None)
    rp.run()

    assert rp.profiles[0].shape == (17520, 6)
    assert len(rp.meta) == 6

    for r1 in set(region1):
        assert r1 in rp.meta["region1"].values

    for r2 in set(region2):
        assert r2 in rp.meta["region2"].values


def test_many_regions_with_list_weights():
    """Test multiple complicated regions with multiple weights per row."""
    sites = [np.random.choice(np.arange(100), np.random.randint(10) + 10)
             .tolist() for __ in np.arange(100)]
    weights = [str(np.random.choice(np.arange(100), len(row)).tolist())
               for row in sites]
    sites = [str(row) for row in sites]
    zeros = np.zeros((100,))
    region1 = (["r0"] * 7) + (["r1"] * 33) + (["r2"] * 60)
    region2 = (["a0"] * 20) + (["b1"] * 10) + (["c2"] * 20) + (["d3"] * 50)
    timezone = np.random.choice([-4, -5, -6, -7], 100)
    rev_summary = pd.DataFrame({SupplyCurveField.GEN_GIDS: sites,
                                SupplyCurveField.RES_GIDS: sites,
                                'res_class': zeros,
                                'region1': region1,
                                'region2': region2,
                                'weights': weights,
                                SupplyCurveField.TIMEZONE: timezone})
    reg_cols = ['region1', 'region2']
    rp = RepProfiles(GEN_FPATH, rev_summary, reg_cols, weight='weights')
    rp.run()

    assert rp.profiles[0].shape == (17520, 6)
    assert len(rp.meta) == 6

    for r1 in set(region1):
        assert r1 in rp.meta["region1"].values

    for r2 in set(region2):
        assert r2 in rp.meta["region2"].values


def test_write_to_file():
    """Test rep profiles with file write."""
    with tempfile.TemporaryDirectory() as td:
        sites = np.arange(100)
        zeros = np.zeros((100,))
        regions = (["r0"] * 7) + (["r1"] * 33) + (["r2"] * 60)
        timezone = np.random.choice([-4, -5, -6, -7], 100)
        rev_summary = pd.DataFrame({SupplyCurveField.GEN_GIDS: sites,
                                    SupplyCurveField.RES_GIDS: sites,
                                    'res_class': zeros,
                                    'region': regions,
                                    SupplyCurveField.TIMEZONE: timezone})
        fout = os.path.join(td, 'temp_rep_profiles.h5')
        rp = RepProfiles(GEN_FPATH, rev_summary, 'region', n_profiles=3,
                         weight=None)
        rp.run(fout=fout)
        with Resource(fout) as res:
            disk_profiles = res["rep_profiles_0"]
            disk_meta = res.meta
            assert "rep_profiles_2" in res.datasets
            test = np.array_equal(res["rep_profiles_0"], res["rep_profiles_1"])
            assert not test

        assert np.allclose(rp.profiles[0], disk_profiles)
        assert len(disk_meta) == 3

        for i in rp.meta.index:
            v1 = json.loads(rp.meta.loc[i, "rep_gen_gid"])
            v2 = json.loads(disk_meta.loc[i, "rep_gen_gid"])
            assert v1 == v2


def test_file_options():
    """Test rep profiles with file write."""
    with tempfile.TemporaryDirectory() as td:
        sites = np.arange(100)
        zeros = np.zeros((100,))
        regions = (["r0"] * 7) + (["r1"] * 33) + (["r2"] * 60)
        timezone = np.random.choice([-4, -5, -6, -7], 100)
        rev_summary = pd.DataFrame({SupplyCurveField.GEN_GIDS: sites,
                                    SupplyCurveField.RES_GIDS: sites,
                                    'res_class': zeros,
                                    'region': regions,
                                    SupplyCurveField.TIMEZONE: timezone})
        fout = os.path.join(td, 'temp_rep_profiles.h5')
        rp = RepProfiles(GEN_FPATH, rev_summary, 'region', n_profiles=3,
                         weight=None)
        rp.run(fout=fout, save_rev_summary=False, scaled_precision=True)
        with Resource(fout) as res:
            dtype = res.get_dset_properties("rep_profiles_0")[1]
            attrs = res.get_attrs("rep_profiles_0")
            disk_profiles = res["rep_profiles_0"]
            disk_dsets = res.datasets

        assert np.issubdtype(dtype, np.integer)
        assert attrs["scale_factor"] == 1000
        assert np.allclose(rp.profiles[0], disk_profiles)
        assert "rev_summary" not in disk_dsets


def test_rep_profiles_cli(runner, clear_loggers):
    """Test rep profiles CLI"""
    with tempfile.TemporaryDirectory() as td:
        sites = np.arange(100)
        zeros = np.zeros((100,))
        regions = (["r0"] * 7) + (["r1"] * 33) + (["r2"] * 60)
        timezone = np.random.choice([-4, -5, -6, -7], 100)
        rev_summary = pd.DataFrame({SupplyCurveField.GEN_GIDS: sites,
                                    SupplyCurveField.RES_GIDS: sites,
                                    'res_class': zeros,
                                    'region': regions,
                                    SupplyCurveField.TIMEZONE: timezone})
        summary_fp = os.path.join(td, 'rev_summary.csv')
        rev_summary.to_csv(summary_fp, index=False)

        config = {
            "log_directory": td,
            "log_level": "INFO",
            "execution_control": {"option": "local"},
            "gen_fpath": GEN_FPATH,
            "rev_summary": summary_fp,
            "reg_cols": "region",
            "n_profiles": 3,
            "weight": None,
            "scaled_precision": True,
            "save_rev_summary": False,
        }

        config_path = os.path.join(td, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = runner.invoke(
            main, [str(ModuleName.REP_PROFILES), "-c", config_path]
        )

        if result.exit_code != 0:
            import traceback

            msg = "Failed with error {}".format(
                traceback.print_exception(*result.exc_info)
            )
            clear_loggers()
            raise RuntimeError(msg)

        dirname = os.path.basename(td)
        fn_out = "{}_{}.h5".format(dirname, ModuleName.REP_PROFILES)
        out_fpath = os.path.join(td, fn_out)
        with Resource(out_fpath) as res:
            dtype = res.get_dset_properties("rep_profiles_0")[1]
            attrs = res.get_attrs("rep_profiles_0")
            disk_dsets = res.datasets

            assert "gen_fpath" in res.h5.attrs
            assert "rep-profiles_config_fp" in res.h5.attrs
            assert "rep-profiles_config" in res.h5.attrs

            assert Path(res.h5.attrs["gen_fpath"]) == Path(GEN_FPATH)

            config_fp = Path(config_path).expanduser().resolve()
            assert Path(res.h5.attrs["rep-profiles_config_fp"]) == config_fp
            assert json.loads(res.h5.attrs["rep-profiles_config"]) == config

        assert np.issubdtype(dtype, np.integer)
        assert attrs["scale_factor"] == 1000
        assert "rev_summary" not in disk_dsets


def execute_pytest(capture="all", flags="-rapP"):
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
    pytest.main(["-q", "--show-capture={}".format(capture), fname, flags])


if __name__ == "__main__":
    execute_pytest()
