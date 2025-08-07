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
from reV.supply_curve.supply_curve import SupplyCurve, _REQUIRED_OUTPUT_COLS
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
    SupplyCurveField.TRANS_GID,
    SupplyCurveField.TRANS_TYPE,
    SupplyCurveField.TRANS_CAPACITY,
    SupplyCurveField.TOTAL_TRANS_CAP_COST_PER_MW,
    SupplyCurveField.DIST_SPUR_KM,
    SupplyCurveField.LCOT,
    SupplyCurveField.TOTAL_LCOE,
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
        check = np.allclose(baseline[SupplyCurveField.TOTAL_LCOE],
                            sc_full[SupplyCurveField.TOTAL_LCOE])
        if not check:
            diff = np.abs(
                baseline[SupplyCurveField.TOTAL_LCOE].values
                - sc_full[SupplyCurveField.TOTAL_LCOE].values
            )
            rel_diff = (
                100 * diff / baseline[SupplyCurveField.TOTAL_LCOE].values
            )
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

        try:
            assert_frame_equal(baseline, sc_full[baseline.columns],
                               check_dtype=False)
        except AssertionError:
            baseline = baseline.drop(columns=[SupplyCurveField.TRANS_GID,
                                              SupplyCurveField.TRANS_CAPACITY,
                                              SupplyCurveField.DIST_SPUR_KM],
                                     errors="ignore")

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


@pytest.mark.parametrize(("i", "trans_costs"),
                         ((1, TRANS_COSTS_1), (2, TRANS_COSTS_2)))
@pytest.mark.parametrize("drop_ac_cap", (True, False))
def test_integrated_sc_simple(i, trans_costs, drop_ac_cap):
    """Run the simple SC test and verify results against baseline file."""
    tcosts = trans_costs.copy()
    tcosts.pop("available_capacity", 1)
    tt = TRANS_TABLE.copy()
    if drop_ac_cap:
        tt = tt.drop(columns="ac_cap")

    sc = SupplyCurve(SC_POINTS, tt, sc_features=MULTIPLIERS)
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
        test = (sc_full[SupplyCurveField.MEAN_LCOE_FRICTION]
                + sc_full[SupplyCurveField.LCOT])
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
                + sc_simple[SupplyCurveField.LCOT])
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
    mask = TRANS_TABLE[SupplyCurveField.TRANS_TYPE] == "PCALoadCen"
    trans_table = TRANS_TABLE[~mask]
    tcosts = TRANS_COSTS_1.copy()
    avail_cap_frac = tcosts.pop("available_capacity", 1)
    with warnings.catch_warnings(record=True) as caught_warnings:
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
        msg = "Warning failed! Should have Unconnected sc_gid in warning!"
        assert any(s1 in str(w.message) for w in caught_warnings), msg


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
        df = pd.read_csv(path).rename(columns=LEGACY_SC_COL_MAP)
        trans_features.append(df[[SupplyCurveField.TRANS_GID, "max_cap"]])

    trans_features = pd.concat(trans_features)

    if isinstance(sc_table, str) and os.path.exists(sc_table):
        sc_table = pd.read_csv(sc_table).rename(columns=LEGACY_SC_COL_MAP)

    if "max_cap" in sc_table and "max_cap" in trans_features:
        sc_table = sc_table.drop("max_cap", axis=1)

    test = sc_table.merge(trans_features,
                          on=SupplyCurveField.TRANS_GID, how='left')
    mask = test[cap_col] > test['max_cap']
    cols = [SupplyCurveField.SC_GID,
            SupplyCurveField.TRANS_GID,
            cap_col,
            'max_cap']
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
    drop_lines = np.where(TRANS_TABLE[SupplyCurveField.TRANS_TYPE]
                          == "TransLine")[0]
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
        SupplyCurveField.TRANS_GID,
        SupplyCurveField.TRANS_TYPE,
        SupplyCurveField.N_PARALLEL_TRANS,
        SupplyCurveField.LCOT,
        SupplyCurveField.TOTAL_LCOE,
        SupplyCurveField.TOTAL_TRANS_CAP_COST_PER_MW,
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

    assert (sc_2[SupplyCurveField.N_PARALLEL_TRANS] > 1).any()

    mask_2 = sc_2[SupplyCurveField.N_PARALLEL_TRANS] > 1

    for gid in sc_2.loc[mask_2, SupplyCurveField.SC_GID]:
        nx_1 = sc_1.loc[(sc_1[SupplyCurveField.SC_GID] == gid),
                        SupplyCurveField.N_PARALLEL_TRANS].values[0]
        nx_2 = sc_2.loc[(sc_2[SupplyCurveField.SC_GID] == gid),
                        SupplyCurveField.N_PARALLEL_TRANS].values[0]
        assert nx_2 >= nx_1
        if nx_1 != nx_2:
            lcot_1 = sc_1.loc[(sc_1[SupplyCurveField.SC_GID] == gid),
                              SupplyCurveField.LCOT].values[0]
            lcot_2 = sc_2.loc[(sc_2[SupplyCurveField.SC_GID] == gid),
                              SupplyCurveField.LCOT].values[0]
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

        assert np.allclose(sc_full[SupplyCurveField.TRANS_GID],
                           sc_full_r[SupplyCurveField.TRANS_GID])
        assert not np.allclose(sc_full[SupplyCurveField.TOTAL_LCOE],
                               sc_full_r[SupplyCurveField.TOTAL_LCOE])


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
            in_table["poi_lat"] = 1
            in_table["poi_lon"] = 2
            in_table["reinforcement_poi_lat"] = 3
            in_table["reinforcement_poi_lon"] = 4
            in_table["reinforcement_cost_per_mw"] = 1e6
            in_table["reinforcement_dist_km"] = 10
            in_table.to_csv(out_fp, index=False)
            trans_tables.append(out_fp)

        out_fpath = os.path.join(td, "sc_r")
        sc = SupplyCurve(SC_POINTS, trans_tables)
        sc_simple_r = sc.run(out_fpath, fixed_charge_rate=0.1, simple=True)
        sc_simple_r = pd.read_csv(sc_simple_r)

        verify_trans_cap(sc_simple_r, trans_tables)

        assert np.allclose(sc_simple[SupplyCurveField.TRANS_GID],
                           sc_simple_r[SupplyCurveField.TRANS_GID])
        assert not np.allclose(sc_simple[SupplyCurveField.TOTAL_LCOE],
                               sc_simple_r[SupplyCurveField.TOTAL_LCOE])

        check_cols = [SupplyCurveField.POI_LAT,
                      SupplyCurveField.POI_LON,
                      SupplyCurveField.REINFORCEMENT_POI_LAT,
                      SupplyCurveField.REINFORCEMENT_POI_LON,
                      SupplyCurveField.REINFORCEMENT_COST_PER_MW,
                      SupplyCurveField.REINFORCEMENT_DIST_KM]
        nan_cols = [SupplyCurveField.POI_LAT,
                    SupplyCurveField.POI_LON,
                    SupplyCurveField.REINFORCEMENT_POI_LAT,
                    SupplyCurveField.REINFORCEMENT_POI_LON]
        for col in check_cols:
            assert col in sc_simple
            if col in nan_cols:
                assert sc_simple[col].isna().all()
            else:
                assert np.allclose(sc_simple[col], 0)

            assert col in sc_simple_r
            assert (sc_simple_r[col] > 0).all()

        assert np.allclose(
            sc_simple[SupplyCurveField.TOTAL_TRANS_CAP_COST_PER_MW]
            * 0.1
            / sc_simple[SupplyCurveField.MEAN_CF_AC]
            / 8760,
            sc_simple[SupplyCurveField.LCOT],
            atol=0.001
        )


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
                sort_on = SupplyCurveField.TOTAL_LCOE
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
        assert (sc_simple[SupplyCurveField.TRANS_GID] == 42445).all()

        assert np.allclose(
            sc_simple[SupplyCurveField.TOTAL_TRANS_CAP_COST_PER_MW]
            * 0.1
            / sc_simple[SupplyCurveField.MEAN_CF_AC]
            / 8760,
            sc_simple[SupplyCurveField.LCOT],
            atol=0.001
        )

        if not r_costs:
            lcot = 4244.5 / (sc_simple[SupplyCurveField.MEAN_CF_AC] * 8760)
            assert np.allclose(lcot, sc_simple[SupplyCurveField.LCOT],
                               atol=0.001)


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


@pytest.mark.parametrize("cols_exist", [True, False])
def test_least_cost_full_pass_through(cols_exist):
    """
    Test the full supply curve sorting passes through variables correctly
    """
    with tempfile.TemporaryDirectory() as td:
        trans_tables = []
        for cap in [100, 200, 400, 1000]:
            in_table = os.path.join(
                TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv"
            )
            in_table = pd.read_csv(in_table)
            out_fp = os.path.join(td, f"costs_RI_{cap}MW.csv")
            if cols_exist:
                for col in _REQUIRED_OUTPUT_COLS:
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

        for col in _REQUIRED_OUTPUT_COLS:
            assert col in sc_full
            if cols_exist:
                assert np.allclose(sc_full[col], 0)
            else:
                assert sc_full[col].isna().all()


@pytest.mark.parametrize("cols_exist", [True, False])
def test_least_cost_simple_pass_through(cols_exist):
    """
    Test the simple supply curve sorting passes through variables correctly
    """
    with tempfile.TemporaryDirectory() as td:
        trans_tables = []
        for cap in [100, 200, 400, 1000]:
            in_table = os.path.join(
                TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv"
            )
            in_table = pd.read_csv(in_table)
            out_fp = os.path.join(td, f"costs_RI_{cap}MW.csv")
            if cols_exist:
                for col in _REQUIRED_OUTPUT_COLS:
                    in_table[col] = 0
            in_table.to_csv(out_fp, index=False)
            trans_tables.append(out_fp)

        out_fpath = os.path.join(td, "sc")
        sc = SupplyCurve(SC_POINTS, trans_tables)
        sc_simple = sc.run(out_fpath, fixed_charge_rate=0.1, simple=True)
        sc_simple = pd.read_csv(sc_simple)

        for col in _REQUIRED_OUTPUT_COLS:
            assert col in sc_simple
            if cols_exist:
                assert np.allclose(sc_simple[col], 0)
            else:
                assert sc_simple[col].isna().all()


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

        tcc_no_r_simple = (
            sc_simple[SupplyCurveField.TOTAL_TRANS_CAP_COST_PER_MW]
            - sc_simple[SupplyCurveField.REINFORCEMENT_COST_PER_MW]
        )
        tcc_no_r_simple_ac_cap = (
            sc_simple_ac_cap[SupplyCurveField.TOTAL_TRANS_CAP_COST_PER_MW]
            - sc_simple_ac_cap[SupplyCurveField.REINFORCEMENT_COST_PER_MW]
        )
        assert np.allclose(tcc_no_r_simple * 1.02, tcc_no_r_simple_ac_cap)
        assert np.allclose(
            sc_simple[SupplyCurveField.REINFORCEMENT_COST_PER_MW],
            sc_simple_ac_cap[SupplyCurveField.REINFORCEMENT_COST_PER_MW],
        )

        # sc_simple_ac_cap lower capacity so higher cost per unit
        assert np.all(sc_simple[SupplyCurveField.LCOT]
                      < sc_simple_ac_cap[SupplyCurveField.LCOT])
        assert np.all(sc_simple[SupplyCurveField.TOTAL_LCOE]
                      < sc_simple_ac_cap[SupplyCurveField.TOTAL_LCOE])


def test_parsing_poi_info():
    """Test that POI info is parsed correctly"""
    sc = pd.DataFrame({SupplyCurveField.SC_GID: [0],
                       SupplyCurveField.SC_ROW_IND: [0],
                       SupplyCurveField.SC_COL_IND: [0],
                       SupplyCurveField.CAPACITY_AC_MW: [10],
                       SupplyCurveField.MEAN_CF_AC: [0.3],
                       SupplyCurveField.MEAN_LCOE: [4]})
    lcp = pd.DataFrame({SupplyCurveField.SC_ROW_IND: [0],
                        SupplyCurveField.SC_COL_IND: [0],
                        SupplyCurveField.TRANS_GID: [1000]})
    pois = pd.DataFrame({"POI_name": ["A", "B", "C"],
                         "POI_limit": [100, 200, 10],
                         "POI_cost_MW": [1000, 2000, 3000]})

    sc = SupplyCurve(sc, lcp, poi_info=pois)
    assert sc._poi_info[SupplyCurveField.TRANS_GID].to_list() == [0, 1, 2]
    assert sc._poi_info["POI_name"].to_list() == ["A", "B", "C"]
    assert sc._poi_info["ac_cap"].to_list() == [100, 200, 10]
    assert sc._poi_info["POI_cost_MW"].to_list() == [1000, 2000, 3000]
    assert (sc._poi_info[SupplyCurveField.TRANS_TYPE] == "loadcen").any()


def test_trans_gid_pulled_from_poi_info():
    """Test that the trans gid value is pulled from POI info"""

    sc = pd.DataFrame({SupplyCurveField.SC_GID: [0],
                       SupplyCurveField.SC_ROW_IND: [0],
                       SupplyCurveField.SC_COL_IND: [0],
                       SupplyCurveField.CAPACITY_AC_MW: [10],
                       SupplyCurveField.MEAN_CF_AC: [0.3],
                       SupplyCurveField.MEAN_LCOE: [4]})
    lcp = pd.DataFrame({SupplyCurveField.SC_ROW_IND: [0],
                        SupplyCurveField.SC_COL_IND: [0],
                        "POI_name": ["B"]})

    pois = pd.DataFrame({"POI_name": ["A", "B", "C"],
                         "POI_limit": [100, 200, 10],
                         "POI_cost_MW": [1000, 2000, 3000]})

    sc = SupplyCurve(sc, lcp, poi_info=pois)
    assert (sc._trans_table[SupplyCurveField.TRANS_GID] == 1).all()


def test_basic_1_poi_to_1_sc_connection():
    """Test the most basic case of POI connection"""
    sc = pd.DataFrame({SupplyCurveField.SC_GID: [0],
                       SupplyCurveField.SC_ROW_IND: [0],
                       SupplyCurveField.SC_COL_IND: [0],
                       SupplyCurveField.CAPACITY_AC_MW: [10],
                       SupplyCurveField.MEAN_CF_AC: [0.3],
                       SupplyCurveField.MEAN_LCOE: [4]})
    lcp = pd.DataFrame({SupplyCurveField.SC_ROW_IND: [0],
                        SupplyCurveField.SC_COL_IND: [0],
                        "POI_name": ["B"],
                        "cost": [4000],
                        SupplyCurveField.DIST_SPUR_KM: [10]})

    pois = pd.DataFrame({"POI_name": ["A", "B", "C"],
                         "POI_limit": [100, 200, 10],
                         "POI_cost_MW": [1000, 2000, 3000]})

    sc = SupplyCurve(sc, lcp, poi_info=pois)
    out = sc.poi_sort(fcr=1, scale_with_capacity=True)

    # Full capacity was connected
    assert out[SupplyCurveField.CAPACITY_AC_MW].to_list() == [10]

    truth_lcot = (4000 / 10 + 2000) / (8760 * 0.3)
    assert np.allclose(out[SupplyCurveField.LCOT], truth_lcot,
                       atol=1e-6, rtol=1e-6)


def test_basic_1_poi_to_many_sc_connection():
    """Test a basic case of POI connection"""
    sc = pd.DataFrame({SupplyCurveField.SC_GID: [0, 10],
                       SupplyCurveField.SC_ROW_IND: [0, 1],
                       SupplyCurveField.SC_COL_IND: [0, 1],
                       SupplyCurveField.CAPACITY_AC_MW: [10, 90],
                       SupplyCurveField.MEAN_CF_AC: [0.3, 0.3],
                       SupplyCurveField.MEAN_LCOE: [4, 5]})
    lcp = pd.DataFrame({SupplyCurveField.SC_ROW_IND: [0, 1],
                        SupplyCurveField.SC_COL_IND: [0, 1],
                        "POI_name": ["B", "B"],
                        "cost": [4000, 4000],
                        SupplyCurveField.DIST_SPUR_KM: [10, 20]})

    pois = pd.DataFrame({"POI_name": ["A", "B", "C"],
                         "POI_limit": [100, 200, 10],
                         "POI_cost_MW": [1000, 2000, 3000]})

    sc = SupplyCurve(sc, lcp, poi_info=pois)
    out = sc.poi_sort(fcr=1, scale_with_capacity=True)

    # Full capacity was connected
    assert out[SupplyCurveField.CAPACITY_AC_MW].to_list() == [10, 90]


def test_too_large_sc_connection():
    """Test connecting some but not all capacity to POI"""
    sc = pd.DataFrame({SupplyCurveField.SC_GID: [0, 10, 15, 20],
                       SupplyCurveField.SC_ROW_IND: [0, 1, 2, 1],
                       SupplyCurveField.SC_COL_IND: [0, 1, 2, 1],
                       SupplyCurveField.CAPACITY_AC_MW: [25, 100, 1000, 1],
                       SupplyCurveField.MEAN_CF_AC: [0.3, 0.3, 0.3, 0.3],
                       SupplyCurveField.MEAN_LCOE: [4, 5, 1, 10]})
    lcp = pd.DataFrame({SupplyCurveField.SC_ROW_IND: [0, 1, 2, 1],
                        SupplyCurveField.SC_COL_IND: [0, 1, 2, 1],
                        "POI_name": ["B", "B", "C", "B"],
                        "cost": [4000, 4000, 100, 10_000],
                        SupplyCurveField.DIST_SPUR_KM: [10, 20, 1, 40]})

    pois = pd.DataFrame({"POI_name": ["A", "B", "C"],
                         "POI_limit": [100, 100, 10],
                         "POI_cost_MW": [1000, 2000, 3000]})

    sc = SupplyCurve(sc, lcp, poi_info=pois)
    out = sc.poi_sort(fcr=1, scale_with_capacity=True)

    # Full capacity was connected
    assert out[SupplyCurveField.CAPACITY_AC_MW].to_list() == [25, 75, 10]
    assert 20 not in set(out[SupplyCurveField.SC_GID])

    cost_per_mw = (np.array([4000, 4000, 100]) / np.array([25, 75, 10])
                   + np.array([2000, 2000, 3000]))
    truth_lcot = cost_per_mw / (8760 * 0.3)
    assert np.allclose(out[SupplyCurveField.LCOT], truth_lcot,
                       atol=1e-6, rtol=1e-6)

    for trans_gid, cap in zip([1, 2], [100, 10]):
        mask = out[SupplyCurveField.TRANS_GID] == trans_gid
        assert np.isclose(
            out.loc[mask, SupplyCurveField.CAPACITY_AC_MW].sum(), cap)


def test_poi_connection_respects_limit():
    """Test connecting to POI respects POI limit on capacity"""
    sc = pd.DataFrame({SupplyCurveField.SC_GID: [0, 10, 15, 20],
                       SupplyCurveField.SC_ROW_IND: [0, 1, 2, 1],
                       SupplyCurveField.SC_COL_IND: [0, 1, 2, 1],
                       SupplyCurveField.CAPACITY_AC_MW: [25, 100, 1000, 1],
                       SupplyCurveField.MEAN_CF_AC: [0.3, 0.3, 0.3, 0.3],
                       SupplyCurveField.MEAN_LCOE: [4, 4, 1, 10]})
    lcp = pd.DataFrame({SupplyCurveField.SC_ROW_IND: [0, 1, 2],
                        SupplyCurveField.SC_COL_IND: [0, 1, 2],
                        "POI_name": ["B", "B", "C"],
                        "cost": [4000, 4500, 100],
                        SupplyCurveField.DIST_SPUR_KM: [10, 20, 1]})

    pois = pd.DataFrame({"POI_name": ["A", "B", "C"],
                         "POI_limit": [100, 50, 10],
                         "POI_cost_MW": [1000, 2000, 3000]})

    sc = SupplyCurve(sc, lcp, poi_info=pois)
    out = sc.poi_sort(fcr=1, scale_with_capacity=True)

    # Full capacity was connected
    assert out[SupplyCurveField.CAPACITY_AC_MW].to_list() == [50, 10]
    assert set(out[SupplyCurveField.SC_GID]) == {10, 15}

    cost_per_mw = (np.array([4500, 100]) / np.array([50, 10])
                   + np.array([2000, 3000]))
    truth_lcot = cost_per_mw / (8760 * 0.3)
    assert np.allclose(out[SupplyCurveField.LCOT], truth_lcot,
                       atol=1e-6, rtol=1e-6)

    for trans_gid, cap in zip([1, 2], [50, 10]):
        mask = out[SupplyCurveField.TRANS_GID] == trans_gid
        assert np.isclose(
            out.loc[mask, SupplyCurveField.CAPACITY_AC_MW].sum(), cap)


def test_poi_connection_respects_selects_cheapest_lcoe():
    """Test connecting to POI selects best connection"""
    sc = pd.DataFrame({SupplyCurveField.SC_GID: [0, 10, 15, 20],
                       SupplyCurveField.SC_ROW_IND: [0, 1, 2, 1],
                       SupplyCurveField.SC_COL_IND: [0, 1, 2, 1],
                       SupplyCurveField.CAPACITY_AC_MW: [25, 100, 1000, 1],
                       SupplyCurveField.MEAN_CF_AC: [0.3, 0.3, 0.3, 0.3],
                       SupplyCurveField.MEAN_LCOE: [4, 4, 1, 10]})
    lcp = pd.DataFrame({SupplyCurveField.SC_ROW_IND: [0, 1, 2],
                        SupplyCurveField.SC_COL_IND: [0, 1, 2],
                        "POI_name": ["B", "B", "C"],
                        "cost": [4000, 4500, 100],
                        SupplyCurveField.DIST_SPUR_KM: [10, 20, 1]})

    pois = pd.DataFrame({"POI_name": ["A", "B", "C"],
                         "POI_limit": [100, 25, 10],
                         "POI_cost_MW": [1000, 2000, 3000]})

    sc = SupplyCurve(sc, lcp, poi_info=pois)
    out = sc.poi_sort(fcr=1, scale_with_capacity=True)

    # Full capacity was connected
    assert out[SupplyCurveField.CAPACITY_AC_MW].to_list() == [25, 10]
    assert set(out[SupplyCurveField.SC_GID]) == {0, 15}

    cost_per_mw = (np.array([4000, 100]) / np.array([25, 10])
                   + np.array([2000, 3000]))
    truth_lcot = cost_per_mw / (8760 * 0.3)
    assert np.allclose(out[SupplyCurveField.LCOT], truth_lcot,
                       atol=1e-6, rtol=1e-6)

    for trans_gid, cap in zip([0, 1, 2], [0, 25, 10]):
        mask = out[SupplyCurveField.TRANS_GID] == trans_gid
        assert np.isclose(
            out.loc[mask, SupplyCurveField.CAPACITY_AC_MW].sum(), cap)


def test_too_large_sc_connection_allowed():
    """Test connecting overflow capacity to POI"""
    sc = pd.DataFrame({SupplyCurveField.SC_GID: [0, 10, 15, 20],
                       SupplyCurveField.SC_ROW_IND: [0, 1, 2, 1],
                       SupplyCurveField.SC_COL_IND: [0, 1, 2, 1],
                       SupplyCurveField.CAPACITY_AC_MW: [25, 100, 1000, 1],
                       SupplyCurveField.MEAN_CF_AC: [0.3, 0.3, 0.3, 0.3],
                       SupplyCurveField.MEAN_LCOE: [4, 5, 1, 10]})
    lcp = pd.DataFrame({SupplyCurveField.SC_ROW_IND: [0, 1, 2, 1],
                        SupplyCurveField.SC_COL_IND: [0, 1, 2, 1],
                        "POI_name": ["B", "B", "C", "B"],
                        "cost": [4000, 4000, 100, 10_000],
                        SupplyCurveField.DIST_SPUR_KM: [10, 20, 1, 40]})

    pois = pd.DataFrame({"POI_name": ["A", "B", "C"],
                         "POI_limit": [100, 100, 10],
                         "POI_cost_MW": [1000, 2000, 3000]})

    sc = SupplyCurve(sc, lcp, poi_info=pois)
    out = sc.poi_sort(fcr=1, max_cap_tie_in_cost_per_mw=1_000_000,
                      scale_with_capacity=True)

    # Full capacity was connected
    assert (out[SupplyCurveField.CAPACITY_AC_MW].to_list()
            == [25, 75, 25, 10, 990, 1])
    assert set(out[SupplyCurveField.SC_GID]) == {0, 10, 15, 20}

    cost_per_mw = (np.array([4000, 4000, 4000, 100, 100, 4000])
                   / np.array([25, 75, 25, 10, 990, 1])
                   + np.array([2000, 2000, 1_000_000, 3000, 1_000_000,
                               1_000_000]))
    truth_lcot = cost_per_mw / (8760 * 0.3)
    assert np.allclose(out[SupplyCurveField.LCOT], truth_lcot,
                       atol=1e-6, rtol=1e-6)

    for trans_gid, cap in zip([0, 1, 2], [0, 126, 1000]):
        mask = out[SupplyCurveField.TRANS_GID] == trans_gid
        assert np.isclose(
            out.loc[mask, SupplyCurveField.CAPACITY_AC_MW].sum(), cap)
