# -*- coding: utf-8 -*-
"""
Supply Curve computation integrated tests
"""

import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from reV import TESTDATADIR
from reV.supply_curve.supply_curve import SupplyCurve
from reV.utilities import SupplyCurveField
from reV.utilities.exceptions import SupplyCurveInputError

TRANS_COSTS_1 = {
    "line_tie_in_cost": 200,
    "line_cost": 1000,
    "station_tie_in_cost": 50,
    "center_tie_in_cost": 10,
    "sink_tie_in_cost": 100,
    "available_capacity": 0.3,
}


TRANS_COSTS_2 = {
    "line_tie_in_cost": 3000,
    "line_cost": 2000,
    "station_tie_in_cost": 500,
    "center_tie_in_cost": 100,
    "sink_tie_in_cost": 1e6,
    "available_capacity": 0.9,
}

path = os.path.join(TESTDATADIR, "sc_out/baseline_agg_summary.csv")
LEGACY_SC_COL_MAP = SupplyCurveField.map_from_legacy()
SC_POINTS = pd.read_csv(path).rename(columns=LEGACY_SC_COL_MAP)

path = os.path.join(TESTDATADIR, "sc_out/baseline_agg_summary_friction.csv")
SC_POINTS_FRICTION = pd.read_csv(path).rename(columns=LEGACY_SC_COL_MAP)

path = os.path.join(TESTDATADIR, "trans_tables/ri_transmission_table.csv")
TRANS_TABLE = pd.read_csv(path).rename(columns=LEGACY_SC_COL_MAP)

path = os.path.join(TESTDATADIR, "trans_tables/transmission_multipliers.csv")
MULTIPLIERS = pd.read_csv(path).rename(columns=LEGACY_SC_COL_MAP)

SC_FULL_COLUMNS = (
    "trans_gid",
    "trans_type",
    "trans_capacity",
    "trans_cap_cost_per_mw",
    "dist_km",
    "lcot",
    "total_lcoe",
)


def baseline_verify(sc_full, fpath_baseline):
    """Verify numerical columns in a CSV against a baseline file."""
    if isinstance(sc_full, str) and os.path.exists(sc_full):
        sc_full = pd.read_csv(sc_full)
        assert not any("Unnamed" in col_name for col_name in sc_full.columns)

    if os.path.exists(fpath_baseline):
        baseline = pd.read_csv(fpath_baseline)
        baseline = baseline.rename(columns=LEGACY_SC_COL_MAP)
        # double check useful for when tables are changing
        # but lcoe should be the same
        check = np.allclose(baseline["total_lcoe"], sc_full["total_lcoe"])
        if not check:
            diff = np.abs(
                baseline["total_lcoe"].values - sc_full["total_lcoe"]
            )
            rel_diff = 100 * diff / baseline["total_lcoe"].values
            msg = (
                "Total LCOE values differed from baseline. "
                "Maximum difference is {:.1f} ({:.1f}%), "
                "mean difference is {:.1f} ({:.1f}%). "
                "In total, {:.1f}% of all SC point connections changed".format(
                    diff.max(),
                    rel_diff.max(),
                    diff.mean(),
                    rel_diff.mean(),
                    100 * (diff > 0).sum() / len(diff),
                )
            )
            raise RuntimeError(msg)

        assert_frame_equal(
            baseline, sc_full[baseline.columns], check_dtype=False
        )

    else:
        sc_full.to_csv(fpath_baseline, index=False)


@pytest.mark.parametrize(
    ("i", "trans_costs"), ((1, TRANS_COSTS_1), (2, TRANS_COSTS_2))
)
def test_integrated_sc_full(i, trans_costs):
    """Run the full SC test and verify results against baseline file."""
    tcosts = trans_costs.copy()
    avail_cap_frac = tcosts.pop("available_capacity", 1)
    sc = SupplyCurve(SC_POINTS, TRANS_TABLE, sc_features=MULTIPLIERS)
    with tempfile.TemporaryDirectory() as td:
        out_fpath = os.path.join(td, "sc")
        sc_full = sc.run(
            out_fpath,
            fixed_charge_rate=0.1,
            simple=False,
            transmission_costs=tcosts,
            avail_cap_frac=avail_cap_frac,
            columns=SC_FULL_COLUMNS,
        )
        fpath_baseline = os.path.join(
            TESTDATADIR, "sc_out/sc_full_out_{}.csv".format(i)
        )
        baseline_verify(sc_full, fpath_baseline)


@pytest.mark.parametrize(
    ("i", "trans_costs"), ((1, TRANS_COSTS_1), (2, TRANS_COSTS_2))
)
def test_integrated_sc_simple(i, trans_costs):
    """Run the simple SC test and verify results against baseline file."""
    tcosts = trans_costs.copy()
    tcosts.pop("available_capacity", 1)
    sc = SupplyCurve(SC_POINTS, TRANS_TABLE, sc_features=MULTIPLIERS)
    with tempfile.TemporaryDirectory() as td:
        out_fpath = os.path.join(td, "sc")
        sc_simple = sc.run(
            out_fpath,
            fixed_charge_rate=0.1,
            simple=True,
            transmission_costs=tcosts,
        )

        fpath_baseline = os.path.join(
            TESTDATADIR, "sc_out/sc_simple_out_{}.csv".format(i)
        )
        baseline_verify(sc_simple, fpath_baseline)


def test_integrated_sc_full_friction():
    """Run the full SC algorithm with friction"""
    tcosts = TRANS_COSTS_1.copy()
    avail_cap_frac = tcosts.pop("available_capacity", 1)
    sc = SupplyCurve(SC_POINTS_FRICTION, TRANS_TABLE, sc_features=MULTIPLIERS)
    with tempfile.TemporaryDirectory() as td:
        out_fpath = os.path.join(td, "sc")
        sc_full = sc.run(out_fpath, fixed_charge_rate=0.1, simple=False,
                         transmission_costs=tcosts,
                         avail_cap_frac=avail_cap_frac,
                         columns=SC_FULL_COLUMNS,
                         sort_on=SupplyCurveField.TOTAL_LCOE_FRICTION)

        sc_full = pd.read_csv(sc_full)
        assert SupplyCurveField.MEAN_LCOE_FRICTION in sc_full
        assert SupplyCurveField.TOTAL_LCOE_FRICTION in sc_full
        test = sc_full[SupplyCurveField.MEAN_LCOE_FRICTION] + sc_full['lcot']
        assert np.allclose(test, sc_full[SupplyCurveField.TOTAL_LCOE_FRICTION])

        fpath_baseline = os.path.join(
            TESTDATADIR, "sc_out/sc_full_out_friction.csv"
        )
        baseline_verify(sc_full, fpath_baseline)


def test_integrated_sc_simple_friction():
    """Run the simple SC algorithm with friction"""
    tcosts = TRANS_COSTS_1.copy()
    tcosts.pop("available_capacity", 1)
    sc = SupplyCurve(SC_POINTS_FRICTION, TRANS_TABLE, sc_features=MULTIPLIERS)
    with tempfile.TemporaryDirectory() as td:
        out_fpath = os.path.join(td, "sc")
        sc_simple = sc.run(out_fpath, fixed_charge_rate=0.1, simple=True,
                           transmission_costs=tcosts,
                           sort_on=SupplyCurveField.TOTAL_LCOE_FRICTION)
        sc_simple = pd.read_csv(sc_simple)
        assert SupplyCurveField.MEAN_LCOE_FRICTION in sc_simple
        assert SupplyCurveField.TOTAL_LCOE_FRICTION in sc_simple
        test = (sc_simple[SupplyCurveField.MEAN_LCOE_FRICTION]
                + sc_simple['lcot'])
        assert np.allclose(test,
                           sc_simple[SupplyCurveField.TOTAL_LCOE_FRICTION])

        fpath_baseline = os.path.join(
            TESTDATADIR, "sc_out/sc_simple_out_friction.csv"
        )
        baseline_verify(sc_simple, fpath_baseline)


def test_sc_warning1():
    """Run the full SC test with missing connections and verify warning."""
    mask = TRANS_TABLE[SupplyCurveField.SC_POINT_GID].isin(list(range(10)))
    trans_table = TRANS_TABLE[~mask]
    tcosts = TRANS_COSTS_1.copy()
    avail_cap_frac = tcosts.pop("available_capacity", 1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sc = SupplyCurve(SC_POINTS, trans_table, sc_features=MULTIPLIERS)
        with tempfile.TemporaryDirectory() as td:
            out_fpath = os.path.join(td, "sc")
            sc.run(
                out_fpath,
                fixed_charge_rate=0.1,
                simple=False,
                transmission_costs=tcosts,
                avail_cap_frac=avail_cap_frac,
                columns=SC_FULL_COLUMNS,
            )

        s1 = str(list(range(10))).replace("]", "").replace("[", "")
        s2 = str(w[0].message)
        msg = (
            "Warning failed! Should have had missing sc_gids 0 through 9: "
            "{}".format(s2)
        )
        assert s1 in s2, msg


def test_sc_warning2():
    """Run the full SC test without PCA load centers and verify warning."""
    mask = TRANS_TABLE["category"] == "PCALoadCen"
    trans_table = TRANS_TABLE[~mask]
    tcosts = TRANS_COSTS_1.copy()
    avail_cap_frac = tcosts.pop("available_capacity", 1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sc = SupplyCurve(SC_POINTS, trans_table, sc_features=MULTIPLIERS)
        with tempfile.TemporaryDirectory() as td:
            out_fpath = os.path.join(td, "sc")
            sc.run(
                out_fpath,
                fixed_charge_rate=0.1,
                simple=False,
                transmission_costs=tcosts,
                avail_cap_frac=avail_cap_frac,
                columns=SC_FULL_COLUMNS,
            )
        s1 = "Unconnected sc_gid"
        s2 = str(w[0].message)
        msg = "Warning failed! Should have Unconnected sc_gid: " "{}".format(
            s2
        )
        assert s1 in s2, msg


def test_parallel():
    """Test a parallel compute against a serial compute"""

    tcosts = TRANS_COSTS_1.copy()
    avail_cap_frac = tcosts.pop("available_capacity", 1)
    sc = SupplyCurve(SC_POINTS, TRANS_TABLE, sc_features=MULTIPLIERS)
    with tempfile.TemporaryDirectory() as td:
        out_fpath = os.path.join(td, "sc")
        sc_full_parallel = sc.run(
            out_fpath,
            fixed_charge_rate=0.1,
            simple=False,
            transmission_costs=tcosts,
            avail_cap_frac=avail_cap_frac,
            columns=SC_FULL_COLUMNS,
            max_workers=4,
        )
        sc_full_serial = sc.run(
            out_fpath,
            fixed_charge_rate=0.1,
            simple=False,
            transmission_costs=tcosts,
            avail_cap_frac=avail_cap_frac,
            columns=SC_FULL_COLUMNS,
            max_workers=1,
        )
        sc_full_parallel = pd.read_csv(sc_full_parallel)
        sc_full_serial = pd.read_csv(sc_full_serial)

    assert_frame_equal(sc_full_parallel, sc_full_serial)


def verify_trans_cap(sc_table, trans_tables,
                     cap_col=SupplyCurveField.CAPACITY_AC_MW):
    """
    Verify that sc_points are connected to features in the correct capacity
    bins
    """

    trans_features = []
    for path in trans_tables:
        df = pd.read_csv(path)
        trans_features.append(df[["trans_gid", "max_cap"]])

    trans_features = pd.concat(trans_features)

    if isinstance(sc_table, str) and os.path.exists(sc_table):
        sc_table = pd.read_csv(sc_table)

    if "max_cap" in sc_table and "max_cap" in trans_features:
        sc_table = sc_table.drop("max_cap", axis=1)

    test = sc_table.merge(trans_features, on='trans_gid', how='left')
    mask = test[cap_col] > test['max_cap']
    cols = [SupplyCurveField.SC_GID, 'trans_gid', cap_col, 'max_cap']
    msg = ("SC points connected to transmission features with "
           "max_cap < sc_cap:\n{}"
           .format(test.loc[mask, cols]))
    assert any(mask), msg


def test_least_cost_full():
    """
    Test full supply curve sorting with least-cost path transmission tables
    """
    trans_tables = [
        os.path.join(TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv")
        for cap in [100, 200, 400, 1000]
    ]
    sc = SupplyCurve(SC_POINTS, trans_tables, sc_features=None)
    with tempfile.TemporaryDirectory() as td:
        out_fpath = os.path.join(td, "sc")
        sc_full = sc.run(out_fpath, fixed_charge_rate=0.1, simple=False,
                         avail_cap_frac=0.1,
                         columns=[*list(SC_FULL_COLUMNS), "max_cap"])

        fpath_baseline = os.path.join(TESTDATADIR, "sc_out/sc_full_lc.csv")
        baseline_verify(sc_full, fpath_baseline)
        verify_trans_cap(sc_full, trans_tables)


def test_least_cost_simple():
    """
    Test simple supply curve sorting with least-cost path transmission tables
    """
    trans_tables = [
        os.path.join(TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv")
        for cap in [100, 200, 400, 1000]
    ]
    sc = SupplyCurve(SC_POINTS, trans_tables)
    with tempfile.TemporaryDirectory() as td:
        out_fpath = os.path.join(td, "sc")
        sc_simple = sc.run(out_fpath, fixed_charge_rate=0.1, simple=True)

        fpath_baseline = os.path.join(TESTDATADIR, "sc_out/sc_simple_lc.csv")
        baseline_verify(sc_simple, fpath_baseline)
        verify_trans_cap(sc_simple, trans_tables)


def test_simple_trans_table():
    """
    Run the simple SC test using a simple transmission table
    and verify results against baseline file.
    """
    trans_table = os.path.join(
        TESTDATADIR, "trans_tables", "ri_simple_transmission_table.csv"
    )
    sc = SupplyCurve(SC_POINTS, trans_table)
    with tempfile.TemporaryDirectory() as td:
        out_fpath = os.path.join(td, "sc")
        sc_simple = sc.run(out_fpath, fixed_charge_rate=0.1, simple=True)

        fpath_baseline = os.path.join(
            TESTDATADIR, "sc_out/ri_sc_simple_lc.csv"
        )
        baseline_verify(sc_simple, fpath_baseline)


def test_substation_conns():
    """
    Ensure missing trans lines are caught by SupplyCurveInputError
    """
    tcosts = TRANS_COSTS_1.copy()
    avail_cap_frac = tcosts.pop("available_capacity", 1)
    drop_lines = np.where(TRANS_TABLE["category"] == "TransLine")[0]
    drop_lines = np.random.choice(drop_lines, 10, replace=False)
    trans_table = TRANS_TABLE.drop(labels=drop_lines)

    with pytest.raises(SupplyCurveInputError):
        sc = SupplyCurve(SC_POINTS, trans_table, sc_features=MULTIPLIERS)
        with tempfile.TemporaryDirectory() as td:
            out_fpath = os.path.join(td, "sc")
            sc.run(
                out_fpath,
                fixed_charge_rate=0.1,
                simple=False,
                columns=SC_FULL_COLUMNS,
                avail_cap_frac=avail_cap_frac,
                max_workers=4,
            )


def test_multi_parallel_trans():
    """When SC points exceed maximum available transmission capacity, they
    should connect to the max capacity trans feature with multiple parallel
    lines.

    This function tests this feature. Previously (before 20220517), the
    SC point would just connect to the biggest line possible, but the cost per
    capacity would become small due to the large capacity and the fixed cost
    per line (least cost path lines were specified at cost per km at a given
    discrete capacity rating). This is documented in
    https://github.com/NREL/reV/issues/336
    """

    columns = (
        "trans_gid",
        "trans_type",
        "n_parallel_trans",
        "lcot",
        "total_lcoe",
        "trans_cap_cost_per_mw",
        "max_cap",
    )

    trans_tables = [
        os.path.join(TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv")
        for cap in [100, 200, 400, 1000]
    ]
    sc = SupplyCurve(SC_POINTS, trans_tables)
    sc_1 = sc.simple_sort(fcr=0.1, columns=columns)

    trans_tables = [
        os.path.join(TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv")
        for cap in [100]
    ]
    sc = SupplyCurve(SC_POINTS, trans_tables)
    sc_2 = sc.simple_sort(fcr=0.1, columns=columns)

    assert not (set(SC_POINTS[SupplyCurveField.SC_GID])
                - set(sc_1[SupplyCurveField.SC_GID]))
    assert not (set(SC_POINTS[SupplyCurveField.SC_GID])
                - set(sc_2[SupplyCurveField.SC_GID]))
    assert not (set(SC_POINTS[SupplyCurveField.SC_POINT_GID])
                - set(sc_1[SupplyCurveField.SC_POINT_GID]))
    assert not (set(SC_POINTS[SupplyCurveField.SC_POINT_GID])
                - set(sc_2[SupplyCurveField.SC_POINT_GID]))
    assert not (set(sc_1[SupplyCurveField.SC_POINT_GID])
                - set(SC_POINTS[SupplyCurveField.SC_POINT_GID]))
    assert not (set(sc_2[SupplyCurveField.SC_POINT_GID])
                - set(SC_POINTS[SupplyCurveField.SC_POINT_GID]))

    assert (sc_2.n_parallel_trans > 1).any()

    mask_2 = sc_2["n_parallel_trans"] > 1

    for gid in sc_2.loc[mask_2, SupplyCurveField.SC_GID]:
        nx_1 = sc_1.loc[(sc_1[SupplyCurveField.SC_GID] == gid),
                        'n_parallel_trans'].values[0]
        nx_2 = sc_2.loc[(sc_2[SupplyCurveField.SC_GID] == gid),
                        'n_parallel_trans'].values[0]
        assert nx_2 >= nx_1
        if nx_1 != nx_2:
            lcot_1 = sc_1.loc[(sc_1[SupplyCurveField.SC_GID] == gid),
                              'lcot'].values[0]
            lcot_2 = sc_2.loc[(sc_2[SupplyCurveField.SC_GID] == gid),
                              'lcot'].values[0]
            assert lcot_2 > lcot_1


# pylint: disable=no-member
def test_least_cost_full_with_reinforcement():
    """
    Test full supply curve sorting with reinforcement costs in the least-cost
    path transmission tables
    """
    with tempfile.TemporaryDirectory() as td:
        trans_tables = []
        for cap in [100, 200, 400, 1000]:
            in_table = os.path.join(
                TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv"
            )
            in_table = pd.read_csv(in_table)
            out_fp = os.path.join(td, f"costs_RI_{cap}MW.csv")
            in_table["reinforcement_cost_per_mw"] = 0
            in_table["reinforcement_dist_km"] = 0
            in_table.to_csv(out_fp, index=False)
            trans_tables.append(out_fp)

        out_fpath = os.path.join(td, "sc")
        sc = SupplyCurve(SC_POINTS, trans_tables)
        sc_full = sc.run(
            out_fpath,
            fixed_charge_rate=0.1,
            simple=False,
            avail_cap_frac=0.1,
            columns=list(SC_FULL_COLUMNS) + ["max_cap"],
        )
        sc_full = pd.read_csv(sc_full)

        fpath_baseline = os.path.join(TESTDATADIR,
                                      'sc_out/sc_full_lc.csv')
        baseline_verify(sc_full, fpath_baseline)
        verify_trans_cap(sc_full, trans_tables)

        trans_tables = []
        for cap in [100, 200, 400, 1000]:
            in_table = os.path.join(
                TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv"
            )
            in_table = pd.read_csv(in_table)
            out_fp = os.path.join(td, f"costs_RI_{cap}MW.csv")
            in_table["reinforcement_cost_per_mw"] = 1e6
            in_table["reinforcement_dist_km"] = 10
            in_table.to_csv(out_fp, index=False)
            trans_tables.append(out_fp)

        out_fpath = os.path.join(td, "sc_r")
        sc = SupplyCurve(SC_POINTS, trans_tables)
        sc_full_r = sc.run(
            out_fpath,
            fixed_charge_rate=0.1,
            simple=False,
            avail_cap_frac=0.1,
            columns=list(SC_FULL_COLUMNS) + ["max_cap"],
        )
        sc_full_r = pd.read_csv(sc_full_r)
        verify_trans_cap(sc_full, trans_tables)

        assert np.allclose(sc_full.trans_gid, sc_full_r.trans_gid)
        assert not np.allclose(sc_full.total_lcoe, sc_full_r.total_lcoe)


# pylint: disable=no-member
def test_least_cost_simple_with_reinforcement():
    """
    Test simple supply curve sorting with reinforcement costs in the
    least-cost path transmission tables
    """
    with tempfile.TemporaryDirectory() as td:
        trans_tables = []
        for cap in [100, 200, 400, 1000]:
            in_table = os.path.join(
                TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv"
            )
            in_table = pd.read_csv(in_table)
            out_fp = os.path.join(td, f"costs_RI_{cap}MW.csv")
            in_table["reinforcement_cost_per_mw"] = 0
            in_table["reinforcement_dist_km"] = 0
            in_table.to_csv(out_fp, index=False)
            trans_tables.append(out_fp)

        out_fpath = os.path.join(td, "sc")
        sc = SupplyCurve(SC_POINTS, trans_tables)
        sc_simple = sc.run(out_fpath, fixed_charge_rate=0.1,
                           simple=True)
        sc_simple = pd.read_csv(sc_simple)

        fpath_baseline = os.path.join(TESTDATADIR,
                                      'sc_out/sc_simple_lc.csv')
        baseline_verify(sc_simple, fpath_baseline)
        verify_trans_cap(sc_simple, trans_tables)

        trans_tables = []
        for cap in [100, 200, 400, 1000]:
            in_table = os.path.join(
                TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv"
            )
            in_table = pd.read_csv(in_table)
            out_fp = os.path.join(td, f"costs_RI_{cap}MW.csv")
            in_table["reinforcement_cost_per_mw"] = 1e6
            in_table["reinforcement_dist_km"] = 10
            in_table.to_csv(out_fp, index=False)
            trans_tables.append(out_fp)

        out_fpath = os.path.join(td, "sc_r")
        sc = SupplyCurve(SC_POINTS, trans_tables)
        sc_simple_r = sc.run(out_fpath, fixed_charge_rate=0.1, simple=True)
        sc_simple_r = pd.read_csv(sc_simple_r)

        verify_trans_cap(sc_simple_r, trans_tables)

        assert np.allclose(sc_simple.trans_gid, sc_simple_r.trans_gid)
        assert not np.allclose(sc_simple.total_lcoe,
                               sc_simple_r.total_lcoe)


# pylint: disable=no-member
@pytest.mark.parametrize("r_costs", [True, False])
def test_least_cost_simple_with_trans_cap_cost_per_mw(r_costs):
    """
    Test simple supply curve with only "trans_cap_cost_per_mw" entry
    """

    with tempfile.TemporaryDirectory() as td:
        trans_tables = []
        for cap in [100, 200, 400, 1000]:
            in_table = os.path.join(
                TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv"
            )
            in_table = pd.read_csv(in_table)
            out_fp = os.path.join(td, f"costs_RI_{cap}MW.csv")
            t_gids = in_table["trans_gid"].values
            if r_costs:
                sort_on = "lcoe_no_reinforcement"
                in_table["reinforcement_cost_per_mw"] = t_gids[::-1]
            else:
                sort_on = "total_lcoe"
                in_table["reinforcement_cost_per_mw"] = 0
            in_table["reinforcement_dist_km"] = 0
            in_table["trans_cap_cost_per_mw"] = t_gids
            in_table = in_table.drop(columns=["trans_cap_cost", "max_cap"])
            in_table.to_csv(out_fp, index=False)
            trans_tables.append(out_fp)

        out_fpath = os.path.join(td, "sc")
        sc = SupplyCurve(SC_POINTS, trans_tables)
        sc_simple = sc.run(out_fpath, fixed_charge_rate=0.1,
                           simple=True, sort_on=sort_on)
        sc_simple = pd.read_csv(sc_simple)
        assert (sc_simple["trans_gid"] == 42445).all()

        if not r_costs:
            lcot = 4244.5 / (sc_simple[SupplyCurveField.MEAN_CF_AC] * 8760)
            assert np.allclose(lcot, sc_simple["lcot"], atol=0.001)


# pylint: disable=no-member
def test_least_cost_simple_with_reinforcement_floor():
    """
    Test simple supply curve sorting with reinforcement costs in the
    least-cost path transmission tables
    """

    with tempfile.TemporaryDirectory() as td:
        trans_tables = []
        for cap in [100, 200, 400, 1000]:
            in_table = os.path.join(
                TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv"
            )
            in_table = pd.read_csv(in_table)
            out_fp = os.path.join(td, f"costs_RI_{cap}MW.csv")
            in_table["reinforcement_cost_per_mw"] = 0
            in_table["reinforcement_dist_km"] = 0
            in_table["reinforcement_cost_floored_per_mw"] = 0
            in_table.to_csv(out_fp, index=False)
            trans_tables.append(out_fp)

        out_fpath = os.path.join(td, "sc")
        sc = SupplyCurve(SC_POINTS, trans_tables)
        sc_simple = sc.run(out_fpath, fixed_charge_rate=0.1,
                           simple=True)
        sc_simple = pd.read_csv(sc_simple)

        fpath_baseline = os.path.join(TESTDATADIR,
                                      'sc_out/sc_simple_lc.csv')
        baseline_verify(sc_simple, fpath_baseline)
        verify_trans_cap(sc_simple, trans_tables)

        trans_tables = []
        for cap in [100, 200, 400, 1000]:
            in_table = os.path.join(
                TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv"
            )
            in_table = pd.read_csv(in_table)
            out_fp = os.path.join(td, f"costs_RI_{cap}MW.csv")
            in_table["reinforcement_cost_per_mw"] = 0
            in_table["reinforcement_dist_km"] = 0
            in_table["reinforcement_cost_floored_per_mw"] = 2000
            in_table.to_csv(out_fp, index=False)
            trans_tables.append(out_fp)

        out_fpath = os.path.join(td, "sc_r")
        sc = SupplyCurve(SC_POINTS, trans_tables)
        sc_simple_r = sc.run(out_fpath, fixed_charge_rate=0.1, simple=True,
                             sort_on="lcot_floored_reinforcement")
        sc_simple_r = pd.read_csv(sc_simple_r)

        baseline_verify(sc_simple, fpath_baseline)
        verify_trans_cap(sc_simple, trans_tables)


def test_least_cost_full_pass_through():
    """
    Test the full supply curve sorting passes through variables correctly
    """
    check_cols = {'poi_lat', 'poi_lon', 'reinforcement_poi_lat',
                  'reinforcement_poi_lon', SupplyCurveField.EOS_MULT,
                  SupplyCurveField.REG_MULT,
                  'reinforcement_cost_per_mw', 'reinforcement_dist_km'}
    with tempfile.TemporaryDirectory() as td:
        trans_tables = []
        for cap in [100, 200, 400, 1000]:
            in_table = os.path.join(
                TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv"
            )
            in_table = pd.read_csv(in_table)
            out_fp = os.path.join(td, f"costs_RI_{cap}MW.csv")
            in_table["reinforcement_cost_per_mw"] = 0
            for col in check_cols:
                in_table[col] = 0
            in_table.to_csv(out_fp, index=False)
            trans_tables.append(out_fp)

        out_fpath = os.path.join(td, "sc")
        sc = SupplyCurve(SC_POINTS, trans_tables)
        sc_full = sc.run(
            out_fpath,
            fixed_charge_rate=0.1,
            simple=False,
            avail_cap_frac=0.1,
            columns=list(SC_FULL_COLUMNS) + ["max_cap"],
        )
        sc_full = pd.read_csv(sc_full)

        for col in check_cols:
            assert col in sc_full
            assert np.allclose(sc_full[col], 0)


def test_least_cost_simple_pass_through():
    """
    Test the simple supply curve sorting passes through variables correctly
    """
    check_cols = {'poi_lat', 'poi_lon', 'reinforcement_poi_lat',
                  'reinforcement_poi_lon', SupplyCurveField.EOS_MULT,
                  SupplyCurveField.REG_MULT,
                  'reinforcement_cost_per_mw', 'reinforcement_dist_km'}
    with tempfile.TemporaryDirectory() as td:
        trans_tables = []
        for cap in [100, 200, 400, 1000]:
            in_table = os.path.join(
                TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv"
            )
            in_table = pd.read_csv(in_table)
            out_fp = os.path.join(td, f"costs_RI_{cap}MW.csv")
            in_table["reinforcement_cost_per_mw"] = 0
            for col in check_cols:
                in_table[col] = 0
            in_table.to_csv(out_fp, index=False)
            trans_tables.append(out_fp)

        out_fpath = os.path.join(td, "sc")
        sc = SupplyCurve(SC_POINTS, trans_tables)
        sc_simple = sc.run(out_fpath, fixed_charge_rate=0.1, simple=True)
        sc_simple = pd.read_csv(sc_simple)

        for col in check_cols:
            assert col in sc_simple
            assert np.allclose(sc_simple[col], 0)


def test_least_cost_simple_with_ac_capacity_column():
    """
    Test simple supply curve sorting with reinforcement costs in the
    least-cost path transmission tables and AC capacity column as capacity
    """
    with tempfile.TemporaryDirectory() as td:
        trans_tables = []
        for cap in [100, 200, 400, 1000]:
            in_table = os.path.join(
                TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv"
            )
            in_table = pd.read_csv(in_table)
            out_fp = os.path.join(td, f"costs_RI_{cap}MW.csv")
            in_table["reinforcement_cost_per_mw"] = 1e6
            in_table["reinforcement_dist_km"] = 10
            in_table.to_csv(out_fp, index=False)
            trans_tables.append(out_fp)

        sc = SupplyCurve(SC_POINTS, trans_tables)
        sc_simple = sc.simple_sort(fcr=0.1)
        verify_trans_cap(sc_simple, trans_tables)

        trans_tables = []
        for cap in [100, 200, 400, 1000]:
            in_table = os.path.join(
                TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv"
            )
            in_table = pd.read_csv(in_table)
            out_fp = os.path.join(td, f"costs_RI_{cap}MW.csv")
            in_table["reinforcement_cost_per_mw"] = 1e6
            in_table["reinforcement_dist_km"] = 10
            in_table.to_csv(out_fp, index=False)
            trans_tables.append(out_fp)

        sc = SC_POINTS.copy()
        sc[SupplyCurveField.CAPACITY_DC_MW] = (
            sc[SupplyCurveField.CAPACITY_AC_MW].values
        )
        sc[SupplyCurveField.CAPACITY_AC_MW] = (
            sc[SupplyCurveField.CAPACITY_DC_MW] / 1.02
        )
        sc = SupplyCurve(sc, trans_tables,
                         sc_capacity_col=SupplyCurveField.CAPACITY_AC_MW)
        sc_simple_ac_cap = sc.simple_sort(fcr=0.1)
        verify_trans_cap(sc_simple_ac_cap, trans_tables,
                         cap_col=SupplyCurveField.CAPACITY_AC_MW)

        assert np.allclose(
            sc_simple["trans_cap_cost_per_mw"] * 1.02,
            sc_simple_ac_cap["trans_cap_cost_per_mw"],
        )
        assert np.allclose(
            sc_simple["reinforcement_cost_per_mw"],
            sc_simple_ac_cap["reinforcement_cost_per_mw"],
        )

        # Final reinforcement costs are slightly cheaper for AC capacity
        assert np.all(sc_simple["lcot"] > sc_simple_ac_cap["lcot"])
        assert np.all(sc_simple["total_lcoe"] > sc_simple_ac_cap["total_lcoe"])
