# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Created on Wed Jun 19 15:37:05 2019

@author: gbuster
"""

import json
import os
import shutil
import tempfile
import traceback

import h5py
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from rex import Outputs, Resource

from reV import TESTDATADIR
from reV.cli import main
from reV.econ.utilities import lcoe_fcr
from reV.supply_curve.sc_aggregation import (
    SupplyCurveAggregation,
    _warn_about_large_datasets,
)
from reV.supply_curve.cli_sc_aggregation import _validate_res_fpath
from reV.handlers.exclusions import LATITUDE
from reV.utilities import ModuleName, SupplyCurveField
from reV.supply_curve.extent import SupplyCurveExtent


EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
RES = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_2012.h5')
GEN = os.path.join(TESTDATADIR, 'gen_out/ri_my_pv_gen.h5')
ONLY_GEN = os.path.join(TESTDATADIR, 'gen_out/ri_my_pv_only_gen.h5')
ONLY_ECON = os.path.join(TESTDATADIR, 'gen_out/ri_my_pv_only_econ.h5')
AGG_BASELINE = os.path.join(TESTDATADIR, 'sc_out/baseline_agg_summary.csv')
TM_DSET = 'techmap_nsrdb'
RES_CLASS_DSET = 'ghi_mean-means'
RES_CLASS_BINS = [0, 4, 100]
DATA_LAYERS = {
    "pct_slope": {"dset": "ri_srtm_slope", "method": "mean"},
    "reeds_region": {"dset": "ri_reeds_regions", "method": "mode"},
    "padus": {"dset": "ri_padus", "method": "mode"},
}
EXCL_DICT = {
    "ri_srtm*slope": {"inclusion_range": (None, 5), "exclude_nodata": True},
    "ri_padus": {"exclude_values": [1], "exclude_nodata": True},
}
RTOL = 0.001
LEGACY_SC_COL_MAP = SupplyCurveField.map_from_legacy()


def test_agg_extent(resolution=64):
    """Get the SC points aggregation summary and test that there are expected
    columns and that all resource gids were found"""

    sca = SupplyCurveAggregation(
        EXCL,
        TM_DSET,
        excl_dict=EXCL_DICT,
        res_class_dset=None,
        res_class_bins=None,
        data_layers=DATA_LAYERS,
        resolution=resolution,
    )
    summary = sca.summarize(GEN)

    all_res_gids = []
    for gids in summary[SupplyCurveField.RES_GIDS]:
        all_res_gids += gids

    assert SupplyCurveField.SC_COL_IND in summary
    assert SupplyCurveField.SC_ROW_IND in summary
    assert SupplyCurveField.GEN_GIDS in summary
    assert len(set(all_res_gids)) == 177


def test_parallel_agg(resolution=64):
    """Test that parallel aggregation yields the same results as serial
    aggregation."""

    gids = list(range(50, 70))
    sca = SupplyCurveAggregation(
        EXCL,
        TM_DSET,
        excl_dict=EXCL_DICT,
        res_class_dset=None,
        res_class_bins=None,
        data_layers=DATA_LAYERS,
        gids=gids,
        resolution=resolution,
    )
    summary_serial = sca.summarize(GEN, max_workers=1)
    summary_parallel = sca.summarize(
        GEN, max_workers=None, sites_per_worker=10
    )

    assert all(summary_serial == summary_parallel)


def test_agg_summary():
    """Test the aggregation summary method against a baseline file."""

    sca = SupplyCurveAggregation(
        EXCL,
        TM_DSET,
        excl_dict=EXCL_DICT,
        res_class_dset=RES_CLASS_DSET,
        res_class_bins=RES_CLASS_BINS,
        data_layers=DATA_LAYERS,
    )
    summary = sca.summarize(GEN, max_workers=1)

    if not os.path.exists(AGG_BASELINE):
        summary.to_csv(AGG_BASELINE)
        raise Exception(
            "Aggregation summary baseline file did not exist. "
            "Created: {}".format(AGG_BASELINE)
        )

    else:
        for c in [SupplyCurveField.RES_GIDS, SupplyCurveField.GEN_GIDS,
                  SupplyCurveField.GID_COUNTS]:
            summary[c] = summary[c].astype(str)

    s_baseline = pd.read_csv(AGG_BASELINE)
    s_baseline = s_baseline.rename(columns=LEGACY_SC_COL_MAP)
    s_baseline = s_baseline.set_index(s_baseline.columns[0])

    summary = summary.fillna("None")
    s_baseline = s_baseline.fillna("None")

    assert SupplyCurveField.CAPACITY_AC_MW in summary
    assert SupplyCurveField.CAPACITY_DC_MW in summary
    assert SupplyCurveField.MEAN_CF_AC in summary
    assert SupplyCurveField.MEAN_CF_DC in summary
    assert SupplyCurveField.REG_MULT in summary
    assert SupplyCurveField.EOS_MULT in summary

    # dc outputs are `None` because old gen file does not have correct
    # output dsets
    assert not summary[SupplyCurveField.CAPACITY_AC_MW].isna().any()
    assert not summary[SupplyCurveField.CAPACITY_DC_MW].isna().all()
    assert not summary[SupplyCurveField.MEAN_CF_AC].isna().any()
    assert not summary[SupplyCurveField.MEAN_CF_DC].isna().all()

    assert np.allclose(summary[SupplyCurveField.REG_MULT], 1)
    assert np.allclose(summary[SupplyCurveField.EOS_MULT], 1)

    summary = summary[list(s_baseline.columns)]
    assert_frame_equal(summary, s_baseline, check_dtype=False, rtol=0.0001)


@pytest.mark.parametrize("pd", [None, 45])
def test_agg_summary_solar_ac(pd):
    """Test the aggregation summary method for solar ac outputs."""

    with Outputs(GEN, "r") as out:
        cf_means_dc = out["cf_mean-means"]

    with tempfile.TemporaryDirectory() as td:
        gen = os.path.join(td, "gen.h5")
        shutil.copy(GEN, gen)
        Outputs.add_dataset(
            gen, "dc_ac_ratio", np.array([1.3] * 188), np.float32
        )
        Outputs.add_dataset(
            gen, "cf_mean_ac-means", cf_means_dc * 1.3, np.float32
        )

        with Outputs(gen, "r") as out:
            assert "dc_ac_ratio" in out.datasets
            assert "cf_mean_ac-means" in out.datasets

        sca = SupplyCurveAggregation(
            EXCL,
            TM_DSET,
            excl_dict=EXCL_DICT,
            res_class_dset=RES_CLASS_DSET,
            res_class_bins=RES_CLASS_BINS,
            data_layers=DATA_LAYERS,
            power_density=pd,
        )
        summary = sca.summarize(gen, max_workers=1)

    assert SupplyCurveField.CAPACITY_AC_MW in summary
    assert SupplyCurveField.CAPACITY_DC_MW in summary
    assert SupplyCurveField.MEAN_CF_AC in summary
    assert SupplyCurveField.MEAN_CF_DC in summary

    assert not summary[SupplyCurveField.CAPACITY_AC_MW].isna().any()
    assert not summary[SupplyCurveField.CAPACITY_DC_MW].isna().any()
    assert not summary[SupplyCurveField.MEAN_CF_AC].isna().any()
    assert not summary[SupplyCurveField.MEAN_CF_DC].isna().any()

    assert np.allclose(summary[SupplyCurveField.CAPACITY_DC_MW] / 1.3,
                       summary[SupplyCurveField.CAPACITY_AC_MW])
    assert np.allclose(summary[SupplyCurveField.CAPACITY_DC_MW]
                       * summary[SupplyCurveField.MEAN_CF_DC],
                       summary[SupplyCurveField.CAPACITY_AC_MW]
                       * summary[SupplyCurveField.MEAN_CF_AC])
    assert np.allclose(summary[SupplyCurveField.CAPACITY_DC_MW]
                       * summary[SupplyCurveField.MEAN_CF_DC]
                       * 8760,
                       summary[SupplyCurveField.SC_POINT_ANNUAL_ENERGY_MWH])
    assert np.allclose(summary[SupplyCurveField.CAPACITY_AC_MW]
                       * summary[SupplyCurveField.MEAN_CF_AC]
                       * 8760,
                       summary[SupplyCurveField.SC_POINT_ANNUAL_ENERGY_MWH])


def test_multi_file_excl():
    """Test sc aggregation with multiple exclusion file inputs."""

    excl_dict = {
        "ri_srtm_slope": {
            "inclusion_range": (None, 5),
            "exclude_nodata": True,
        },
        "ri_padus": {"exclude_values": [1], "exclude_nodata": True},
        "excl_test": {"include_values": [1], "weight": 0.5},
    }

    with tempfile.TemporaryDirectory() as td:
        excl_temp_1 = os.path.join(td, "excl1.h5")
        excl_temp_2 = os.path.join(td, "excl2.h5")
        shutil.copy(EXCL, excl_temp_1)
        shutil.copy(EXCL, excl_temp_2)

        with h5py.File(excl_temp_1, 'a') as f:
            shape = f[LATITUDE].shape
            attrs = dict(f['ri_srtm_slope'].attrs)
            data = np.ones(shape)
            test_dset = "excl_test"
            f.create_dataset(test_dset, shape, data=data)
            for k, v in attrs.items():
                f[test_dset].attrs[k] = v
            del f["ri_srtm_slope"]

        sca = SupplyCurveAggregation(
            (excl_temp_1, excl_temp_2),
            TM_DSET,
            excl_dict=excl_dict,
            res_class_dset=RES_CLASS_DSET,
            res_class_bins=RES_CLASS_BINS,
        )
        summary = sca.summarize(GEN)

        s_baseline = pd.read_csv(AGG_BASELINE)
        s_baseline = s_baseline.rename(columns=LEGACY_SC_COL_MAP)
        s_baseline = s_baseline.set_index(s_baseline.columns[0])

        summary = summary.fillna("None")
        s_baseline = s_baseline.fillna("None")

        assert np.allclose(summary[SupplyCurveField.AREA_SQ_KM] * 2,
                           s_baseline[SupplyCurveField.AREA_SQ_KM])


@pytest.mark.parametrize("pre_extract", (True, False))
def test_pre_extract_inclusions(pre_extract):
    """Test the aggregation summary w/ and w/out pre-extracting inclusions"""

    sca = SupplyCurveAggregation(
        EXCL,
        TM_DSET,
        excl_dict=EXCL_DICT,
        res_class_dset=RES_CLASS_DSET,
        res_class_bins=RES_CLASS_BINS,
        data_layers=DATA_LAYERS,
        pre_extract_inclusions=pre_extract,
    )
    summary = sca.summarize(GEN, max_workers=1)

    if not os.path.exists(AGG_BASELINE):
        summary.to_csv(AGG_BASELINE)
        raise Exception(
            "Aggregation summary baseline file did not exist. "
            "Created: {}".format(AGG_BASELINE)
        )

    else:
        for c in [SupplyCurveField.RES_GIDS, SupplyCurveField.GEN_GIDS,
                  SupplyCurveField.GID_COUNTS]:
            summary[c] = summary[c].astype(str)

    s_baseline = pd.read_csv(AGG_BASELINE)
    s_baseline = s_baseline.rename(columns=LEGACY_SC_COL_MAP)
    s_baseline = s_baseline.set_index(s_baseline.columns[0])

    summary = summary.fillna("None")
    s_baseline = s_baseline.fillna("None")
    summary = summary[list(s_baseline.columns)]

    assert_frame_equal(summary, s_baseline, check_dtype=False, rtol=0.0001)


def test_agg_gen_econ():
    """Test the aggregation summary method with separate gen and econ
    input files."""

    sca = SupplyCurveAggregation(
        EXCL,
        TM_DSET,
        excl_dict=EXCL_DICT,
        res_class_dset=RES_CLASS_DSET,
        res_class_bins=RES_CLASS_BINS,
        data_layers=DATA_LAYERS,
    )
    summary_base = sca.summarize(GEN, max_workers=1)

    sca = SupplyCurveAggregation(
        EXCL,
        TM_DSET,
        econ_fpath=ONLY_ECON,
        excl_dict=EXCL_DICT,
        res_class_dset=RES_CLASS_DSET,
        res_class_bins=RES_CLASS_BINS,
        data_layers=DATA_LAYERS,
    )
    summary_econ = sca.summarize(ONLY_GEN, max_workers=1)

    assert_frame_equal(summary_base, summary_econ)


def test_agg_extra_dsets():
    """Test aggregation with extra datasets to aggregate."""
    h5_dsets = ["lcoe_fcr-2012", "lcoe_fcr-2013", "lcoe_fcr-stdev"]
    sca = SupplyCurveAggregation(
        EXCL,
        TM_DSET,
        h5_dsets=h5_dsets,
        econ_fpath=ONLY_ECON,
        excl_dict=EXCL_DICT,
        res_class_dset=RES_CLASS_DSET,
        res_class_bins=RES_CLASS_BINS,
        data_layers=DATA_LAYERS,
    )
    summary = sca.summarize(ONLY_GEN, max_workers=1)

    for dset in h5_dsets:
        assert "mean_{}".format(dset) in summary.columns

    check = (summary['mean_lcoe_fcr-2012']
             == summary[SupplyCurveField.MEAN_LCOE])
    assert not any(check)
    check = (summary['mean_lcoe_fcr-2013']
             == summary[SupplyCurveField.MEAN_LCOE])
    assert not any(check)

    avg = (summary['mean_lcoe_fcr-2012'] + summary['mean_lcoe_fcr-2013']) / 2
    assert np.allclose(avg.values, summary[SupplyCurveField.MEAN_LCOE].values)


def test_agg_extra_2D_dsets():
    """Test that warning is thrown for 2D datasets."""
    dset = "cf_profile"
    fp = os.path.join(TESTDATADIR, "gen_out/pv_gen_2018_node00.h5")
    with pytest.warns(UserWarning) as records:
        with Resource(fp) as res:
            _warn_about_large_datasets(res, dset)

    messages = [r.message.args[0] for r in records]
    assert any(
        "Generation dataset {!r} is not 1-dimensional (shape: {})".format(
            dset, (17520, 50)
        )
        in msg
        for msg in messages
    )
    assert any(
        "You may run into memory errors during aggregation" in msg
        for msg in messages
    )


def test_agg_scalar_excl():
    """Test the aggregation summary with exclusions of 0.5"""

    gids_subset = list(range(0, 20))
    excl_dict_1 = {"ri_padus": {"exclude_values": [1]}}
    sca = SupplyCurveAggregation(
        EXCL,
        TM_DSET,
        excl_dict=excl_dict_1,
        res_class_dset=RES_CLASS_DSET,
        res_class_bins=RES_CLASS_BINS,
        data_layers=DATA_LAYERS,
        gids=gids_subset,
    )
    summary_base = sca.summarize(GEN, max_workers=1)

    excl_dict_2 = {"ri_padus": {"exclude_values": [1], "weight": 0.5}}
    sca = SupplyCurveAggregation(
        EXCL,
        TM_DSET,
        excl_dict=excl_dict_2,
        res_class_dset=RES_CLASS_DSET,
        res_class_bins=RES_CLASS_BINS,
        data_layers=DATA_LAYERS,
        gids=gids_subset,
    )
    summary_with_weights = sca.summarize(GEN, max_workers=1)

    dsets = [SupplyCurveField.AREA_SQ_KM, SupplyCurveField.CAPACITY_AC_MW]
    for dset in dsets:
        diff = summary_base[dset].values / summary_with_weights[dset].values
        msg = ("Fractional exclusions failed for {} which has values {} and {}"
               .format(dset, summary_base[dset].values,
                       summary_with_weights[dset].values))
        assert all(diff == 2), msg

    for i in summary_base.index:
        counts_full = summary_base.loc[i, SupplyCurveField.GID_COUNTS]
        counts_half = summary_with_weights.loc[i, SupplyCurveField.GID_COUNTS]

        for j, counts in enumerate(counts_full):
            msg = ("GID counts for fractional exclusions failed for index {}!"
                   .format(i))
            assert counts == 2 * counts_half[j], msg


def test_data_layer_methods():
    """Test aggregation of data layers with different methods"""
    data_layers = {
        "pct_slope_mean": {"dset": "ri_srtm_slope", "method": "mean"},
        "pct_slope_max": {"dset": "ri_srtm_slope", "method": "max"},
        "pct_slope_min": {"dset": "ri_srtm_slope", "method": "min"},
        "pct_slope_sum": {"dset": "ri_srtm_slope", "method": "sum"},
        "pct_slope_mode": {"dset": "ri_srtm_slope", "method": "mode"},
        "reeds_region_cat": {"dset": "ri_reeds_regions", "method": "category"},
        "padus_cat": {"dset": "ri_padus", "method": "category"},
    }

    sca = SupplyCurveAggregation(
        EXCL,
        TM_DSET,
        excl_dict=EXCL_DICT,
        res_class_dset=RES_CLASS_DSET,
        res_class_bins=RES_CLASS_BINS,
        data_layers=data_layers,
    )
    summary = sca.summarize(GEN, max_workers=1)

    for i in summary.index.values:
        # Check categorical data layers
        counts = summary.loc[i, SupplyCurveField.GID_COUNTS]
        rr = summary.loc[i, 'reeds_region_cat']
        assert isinstance(rr, str)
        rr = json.loads(rr)
        assert isinstance(rr, dict)
        rr_sum = sum(list(rr.values()))
        padus = summary.loc[i, "padus_cat"]
        assert isinstance(padus, str)
        padus = json.loads(padus)
        assert isinstance(padus, dict)
        padus_sum = sum(list(padus.values()))
        try:
            assert padus_sum == sum(counts)
            assert padus_sum >= rr_sum
        except AssertionError:
            e = "Categorical data layer aggregation failed:\n{}".format(
                summary.loc[i]
            )
            raise RuntimeError(e)

        # Check min/mean/max of the same data layer
        n = summary.loc[i, SupplyCurveField.N_GIDS]
        slope_mean = summary.loc[i, 'pct_slope_mean']
        slope_max = summary.loc[i, 'pct_slope_max']
        slope_min = summary.loc[i, 'pct_slope_min']
        slope_sum = summary.loc[i, 'pct_slope_sum']
        slope_mode = summary.loc[i, 'pct_slope_mode']
        if n > 3:  # sc points with <= 3 90m pixels can have min == mean == max
            assert slope_min < slope_mean < slope_max <= slope_sum
        else:
            assert slope_min <= slope_mean <= slope_max <= slope_sum

        assert slope_min <= slope_mode <= slope_max



@pytest.mark.parametrize(
    "cap_cost_scale",
    ["1", f"2 * np.multiply(1000, {SupplyCurveField.CAPACITY_AC_MW}) ** -0.3"]
)
@pytest.mark.parametrize("voc", [0, 0.1])
def test_recalc_lcoe(cap_cost_scale, voc):
    """Test supply curve aggregation with the re-calculation of lcoe using the
    multi-year mean capacity factor"""

    data = {"capital_cost": 34900000,
            "fixed_operating_cost": 280000,
            "fixed_charge_rate": 0.09606382995843887,
            "variable_operating_cost": voc,
            'system_capacity': 20000}
    annual_cf = [0.24, 0.26, 0.37, 0.15]
    annual_lcoe = []
    years = list(range(2012, 2016))

    with tempfile.TemporaryDirectory() as td:
        gen_temp = os.path.join(td, "ri_my_pv_gen.h5")
        shutil.copy(GEN, gen_temp)

        with h5py.File(gen_temp, "a") as res:
            for k in [d for d in list(res) if d != "meta"]:
                del res[k]
            for k, v in data.items():
                arr = np.full(res["meta"].shape, v)
                res.create_dataset(k, res["meta"].shape, data=arr)

            arr = np.full(res["meta"].shape, data["capital_cost"])
            res.create_dataset("base_capital_cost", res["meta"].shape,
                               data=arr)

            arr = np.full(res["meta"].shape, data["fixed_operating_cost"])
            res.create_dataset("base_fixed_operating_cost", res["meta"].shape,
                               data=arr)

            arr = np.full(res["meta"].shape, data["variable_operating_cost"])
            res.create_dataset("base_variable_operating_cost",
                               res["meta"].shape, data=arr)

            for year, cf in zip(years, annual_cf):
                lcoe = lcoe_fcr(data["fixed_charge_rate"],
                                data["capital_cost"],
                                data["fixed_operating_cost"],
                                data['system_capacity'] * cf * 8760,
                                data["variable_operating_cost"])
                cf_arr = np.full(res['meta'].shape, cf)
                lcoe_arr = np.full(res['meta'].shape, lcoe)
                annual_lcoe.append(lcoe)

                res.create_dataset(
                    "cf_mean-{}".format(year), res["meta"].shape, data=cf_arr
                )
                res.create_dataset(
                    "lcoe_fcr-{}".format(year),
                    res["meta"].shape,
                    data=lcoe_arr,
                )

            cf_arr = np.full(res["meta"].shape, np.mean(annual_cf))
            lcoe_arr = np.full(res["meta"].shape, np.mean(annual_lcoe))
            res.create_dataset("cf_mean-means", res["meta"].shape, data=cf_arr)
            res.create_dataset(
                "lcoe_fcr-means", res["meta"].shape, data=lcoe_arr
            )

        h5_dsets = ["capital_cost",
                    "fixed_operating_cost",
                    "fixed_charge_rate",
                    "variable_operating_cost",
                    'system_capacity']

        base = SupplyCurveAggregation(
            EXCL,
            TM_DSET,
            excl_dict=EXCL_DICT,
            res_class_dset=None,
            res_class_bins=None,
            data_layers=DATA_LAYERS,
            h5_dsets=h5_dsets,
            gids=list(np.arange(10)),
            recalc_lcoe=False,
            cap_cost_scale=cap_cost_scale,
        )
        summary_base = base.summarize(gen_temp, max_workers=1)

        sca = SupplyCurveAggregation(
            EXCL,
            TM_DSET,
            excl_dict=EXCL_DICT,
            res_class_dset=None,
            res_class_bins=None,
            data_layers=DATA_LAYERS,
            h5_dsets=h5_dsets,
            gids=list(np.arange(10)),
            recalc_lcoe=True,
            cap_cost_scale=cap_cost_scale,
        )
        summary = sca.summarize(gen_temp, max_workers=1)

    assert not np.allclose(summary_base[SupplyCurveField.MEAN_LCOE],
                           summary[SupplyCurveField.MEAN_LCOE])

    assert np.allclose(summary[SupplyCurveField.EOS_MULT],
                       summary[SupplyCurveField.COST_SITE_CC_USD_PER_AC_MW]
                       / summary[SupplyCurveField.COST_BASE_CC_USD_PER_AC_MW])
    assert np.allclose(data['capital_cost'] / data['system_capacity'] * 1000,
                       summary[SupplyCurveField.COST_BASE_CC_USD_PER_AC_MW])

    expected_recalc_lcoe = lcoe_fcr(data["fixed_charge_rate"],
                                    data["capital_cost"],
                                    data["fixed_operating_cost"],
                                    data["system_capacity"]
                                    * np.array(annual_cf).mean()
                                    * 8760,
                                    data["variable_operating_cost"])
    if cap_cost_scale == "1":
        assert np.allclose(summary[SupplyCurveField.MEAN_LCOE],
                           expected_recalc_lcoe)
    else:
        assert not np.allclose(summary[SupplyCurveField.MEAN_LCOE],
                               expected_recalc_lcoe)

    fcr = summary[SupplyCurveField.FIXED_CHARGE_RATE]
    cap_cost = (summary[SupplyCurveField.COST_SITE_CC_USD_PER_AC_MW]
                * summary[SupplyCurveField.CAPACITY_AC_MW])
    foc = (summary[SupplyCurveField.COST_SITE_FOC_USD_PER_AC_MW]
           * summary[SupplyCurveField.CAPACITY_AC_MW])
    voc = summary[SupplyCurveField.COST_SITE_VOC_USD_PER_AC_MWH] / 1000
    aep_kwh = summary[SupplyCurveField.SC_POINT_ANNUAL_ENERGY_MWH] * 1000

    lcoe = lcoe_fcr(fcr, cap_cost, foc, aep_kwh, voc)
    assert np.allclose(lcoe, summary[SupplyCurveField.MEAN_LCOE])

    cap_cost = (summary[SupplyCurveField.COST_BASE_CC_USD_PER_AC_MW]
                * summary[SupplyCurveField.CAPACITY_AC_MW]
                * summary[SupplyCurveField.REG_MULT]
                * summary[SupplyCurveField.EOS_MULT])
    foc = (summary[SupplyCurveField.COST_BASE_FOC_USD_PER_AC_MW]
           * summary[SupplyCurveField.CAPACITY_AC_MW])
    voc = summary[SupplyCurveField.COST_BASE_VOC_USD_PER_AC_MWH] / 1000

    lcoe = lcoe_fcr(fcr, cap_cost, foc, aep_kwh, voc)
    assert np.allclose(lcoe, summary[SupplyCurveField.MEAN_LCOE])


@pytest.mark.parametrize("tm_dset", ("techmap_ri", "techmap_ri_new"))
@pytest.mark.parametrize("pre_extract", (True, False))
def test_cli_basic_agg(runner, clear_loggers, tm_dset, pre_extract):
    """Test basic sc agg cli invocation"""
    with tempfile.TemporaryDirectory() as td:
        excl_fp = os.path.join(td, "excl.h5")
        shutil.copy(EXCL, excl_fp)
        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local",
                "max_workers": 1,
            },
            "log_level": "INFO",
            "excl_fpath": excl_fp,
            "gen_fpath": None,
            "econ_fpath": None,
            "tm_dset": tm_dset,
            "res_fpath": RES,
            "excl_dict": EXCL_DICT,
            "resolution": 32,
            "pre_extract_inclusions": pre_extract,
        }
        config_path = os.path.join(td, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = runner.invoke(
            main, [ModuleName.SUPPLY_CURVE_AGGREGATION, "-c", config_path]
        )
        clear_loggers()

        if result.exit_code != 0:
            msg = "Failed with error {}".format(
                traceback.print_exception(*result.exc_info)
            )
            raise RuntimeError(msg)

        fn_list = os.listdir(td)
        dirname = os.path.basename(td)
        out_csv_fn = "{}_{}.csv".format(
            dirname, ModuleName.SUPPLY_CURVE_AGGREGATION
        )
        assert out_csv_fn in fn_list


def test_validate_res_fpath():
    """Test the validate_res_fpath function"""
    assert _validate_res_fpath({"test": 1}) == {"test": 1, "res_fpath": None}

    with tempfile.TemporaryDirectory() as td:
        test_file = os.path.join(td, "gen.h5")
        config = {"res_fpath": test_file}
        with pytest.raises(FileNotFoundError) as error:
            _validate_res_fpath(config)

        expected_msg = ("gen.h5 is not a valid file path, and HSDS cannot "
                        "be checked for a file at this path!")
        assert expected_msg in str(error)

        with open(test_file, 'w'):
            pass

        assert _validate_res_fpath(config) == config


def test_validate_res_fpath_with_year_pattern():
    """Test the validate_res_fpath function with {} substitute for year."""

    with tempfile.TemporaryDirectory() as td:
        tf = os.path.join(td, "gen_{}.h5")
        config = {"res_fpath": tf}
        with pytest.raises(FileNotFoundError) as error:
            _validate_res_fpath(config)
        assert "Could not find any files that match the pattern" in str(error)
        assert "gen_<year>.h5" in str(error)

        with open(tf.format(2012), 'w'):
            pass

        config = {"res_fpath": tf}
        assert _validate_res_fpath(config) == {"res_fpath": tf.format(2012)}

        with open(tf.format(2010), 'w'):
            pass

        config = {"res_fpath": tf}
        assert _validate_res_fpath(config) == {"res_fpath": tf.format(2010)}


@pytest.mark.parametrize("zone_config,max_workers,pre_extract_inclusions", [
    ("one_full", None, False),
    (1, None, False),
    (2, None, False),
    (3, None, False),
    (1, 1, False),  # test with run_serial
    (1, None, True),  # test with pre_extract_exclusions
])
def test_agg_zones(zone_config, max_workers, pre_extract_inclusions):
    """Test sc aggregation with zones within each sc site."""

    resolution = 64
    gids = [1, 2, 3]

    with tempfile.TemporaryDirectory() as td:
        excl_temp = os.path.join(td, "excl.h5")
        shutil.copy(EXCL, excl_temp)
        with SupplyCurveExtent(excl_temp, resolution=resolution) as sc:
            slice_lookup = sc.get_slice_lookup(gids)

        with h5py.File(excl_temp, 'a') as f:
            shape = f[LATITUDE].shape
            attrs = dict(f['ri_smod'].attrs)
            profile = json.loads(attrs["profile"])
            profile["dtype"] = "uint32"
            profile["nodata"] = 0
            attrs["profile"] = json.dumps(profile)
            data = np.zeros(shape, dtype=np.uint32)
            if zone_config == "one_full":
                # each entire cell is one zone
                for gid, gid_slice in slice_lookup.items():
                    data[gid_slice] = gid + 10
                # use the standard test dataset
                baseline = AGG_BASELINE
                excl_dict = EXCL_DICT.copy()
                res_class_bins = RES_CLASS_BINS
                apply_legacy_remap = True
            else:
                excl_dict = {
                    k: v for k, v in EXCL_DICT.items() if k == "ri_srtm_slope"
                }
                res_class_bins = None
                baseline = os.path.join(
                    TESTDATADIR,
                    f"sc_out/baseline_agg_summary_zones_{zone_config}.csv"
                )
                apply_legacy_remap = False
                for gid, gid_slice in slice_lookup.items():
                    gid_rows, gid_cols = gid_slice
                    for z in range(0, zone_config):
                        zone_rows = slice(
                            gid_rows.stop - (z + 1) * 4,
                            gid_rows.stop - z * 4
                        )
                        zone_cols = slice(gid_cols.stop - 4, gid_cols.stop)
                        data[(zone_rows, zone_cols)] = gid + 10 * (z + 1)

            test_dset = "parcels"
            f.create_dataset(test_dset, shape, data=data)
            for k, v in attrs.items():
                f[test_dset].attrs[k] = v

        sca = SupplyCurveAggregation(
            excl_temp,
            TM_DSET,
            excl_dict=excl_dict,
            res_class_dset=RES_CLASS_DSET,
            res_class_bins=res_class_bins,
            zones_dset=test_dset,
            resolution=resolution,
            power_density=36.0,
            gids=gids,
            pre_extract_inclusions=pre_extract_inclusions,
        )
        summary = sca.summarize(GEN, max_workers=max_workers)

    s_baseline = pd.read_csv(baseline)
    if apply_legacy_remap:
        s_baseline = s_baseline.rename(columns=LEGACY_SC_COL_MAP)
    s_baseline = s_baseline.set_index(s_baseline.columns[0])
    s_baseline_subset = s_baseline[
        s_baseline["sc_point_gid"].isin(gids)
    ].copy()
    list_cols = ["res_gids", "gen_gids", "gid_counts"]
    # convert columns containing lists of integers as strings to lists
    # of integers
    for list_col in list_cols:
        s_baseline_subset[list_col] = s_baseline_subset[list_col].apply(
            json.loads
        )

    summary = summary.fillna("None")
    s_baseline_subset = s_baseline_subset.fillna("None")

    compare_cols = list(
        set(s_baseline_subset.columns).intersection(summary.columns)
    )
    assert_frame_equal(
        summary[compare_cols],
        s_baseline_subset[compare_cols], check_dtype=False, rtol=0.0001
    )


def test_cli_agg_zones(runner, clear_loggers):
    """
    Test SC aggregation with zones within each SC site via the CLI
    """

    resolution = 64
    gids = [1, 2, 3]

    with tempfile.TemporaryDirectory() as td:
        excl_temp = os.path.join(td, "excl.h5")
        shutil.copy(EXCL, excl_temp)
        with SupplyCurveExtent(excl_temp, resolution=resolution) as sc:
            slice_lookup = sc.get_slice_lookup(gids)

        with h5py.File(excl_temp, 'a') as f:
            shape = f[LATITUDE].shape
            attrs = dict(f['ri_smod'].attrs)
            profile = json.loads(attrs["profile"])
            profile["dtype"] = "uint32"
            profile["nodata"] = 0
            attrs["profile"] = json.dumps(profile)
            data = np.zeros(shape, dtype=np.uint32)
            # each entire cell is one zone
            for gid, gid_slice in slice_lookup.items():
                data[gid_slice] = gid + 10
            test_dset = "parcels"
            f.create_dataset(test_dset, shape, data=data)
            for k, v in attrs.items():
                f[test_dset].attrs[k] = v

        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local",
                "max_workers": 1,
            },
            "log_level": "INFO",
            "excl_fpath": excl_temp,
            "gen_fpath": GEN,
            "econ_fpath": None,
            "tm_dset": TM_DSET,
            "res_fpath": None,
            "res_class_dset": RES_CLASS_DSET,
            "res_class_bins": RES_CLASS_BINS,
            "excl_dict": EXCL_DICT,
            "resolution": resolution,
            "zones_dset": test_dset,
            "pre_extract_inclusions": False,
            "gids": gids,
            "power_density": 36,
        }

        config_path = os.path.join(td, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = runner.invoke(
            main, [ModuleName.SUPPLY_CURVE_AGGREGATION, "-c", config_path]
        )
        clear_loggers()

        if result.exit_code != 0:
            msg = "Failed with error {}".format(
                traceback.print_exception(*result.exc_info)
            )
            raise RuntimeError(msg)

        fn_list = os.listdir(td)
        dirname = os.path.basename(td)
        out_csv_fn = "{}_{}.csv".format(
            dirname, ModuleName.SUPPLY_CURVE_AGGREGATION
        )
        assert out_csv_fn in fn_list

        summary = pd.read_csv(os.path.join(td, out_csv_fn))

    s_baseline = pd.read_csv(AGG_BASELINE)
    s_baseline = s_baseline.rename(columns=LEGACY_SC_COL_MAP)
    s_baseline_subset = s_baseline[
        s_baseline["sc_point_gid"].isin(gids)
    ].copy()

    summary = summary.fillna("None")
    s_baseline_subset = s_baseline_subset.fillna("None")

    compare_cols = list(
        set(s_baseline_subset.columns).intersection(summary.columns)
    )
    assert_frame_equal(
        summary[compare_cols],
        s_baseline_subset[compare_cols], check_dtype=False, rtol=0.0001
    )


def test_basic_col_desc_cli(runner, clear_loggers):
    """Test basic sc-col-descriptions cli invocation"""
    with tempfile.TemporaryDirectory() as td:
        out_fp = os.path.join(td, "test.csv")
        result = runner.invoke(main, ["sc-col-descriptions", "-of", out_fp])
        clear_loggers()

        if result.exit_code != 0:
            msg = "Failed with error {}".format(
                traceback.print_exception(*result.exc_info)
            )
            raise RuntimeError(msg)

        assert os.path.exists(out_fp)

        desc = pd.read_csv(out_fp)
        assert len(desc) > 0


def test_col_desc_cli_for_sc(runner, clear_loggers):
    """Test sc-col-descriptions for an input SC file via the CLI"""
    test_sc = os.path.join(TESTDATADIR, "sc_out", "ri_sc_simple_lc.csv")
    with tempfile.TemporaryDirectory() as td:
        to_run_sc = os.path.join(td, "ri_sc_simple_lc.csv")
        shutil.copy(test_sc, to_run_sc)

        result = runner.invoke(main, ["sc-col-descriptions", "-sc", to_run_sc])
        clear_loggers()

        if result.exit_code != 0:
            msg = "Failed with error {}".format(
                traceback.print_exception(*result.exc_info)
            )
            raise RuntimeError(msg)

        expected_fp = os.path.join(td, "ri_sc_simple_lc_column_lookup.csv")
        assert os.path.exists(expected_fp)

        desc = pd.read_csv(expected_fp)
        sc = pd.read_csv(to_run_sc)

        assert len(desc) > 0
        assert all(col in sc.columns for col in desc["reV Column"])


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

    fname = __file__
    pytest.main(["-q", "--show-capture={}".format(capture), fname, flags])


if __name__ == "__main__":
    execute_pytest()
