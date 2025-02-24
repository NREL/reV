# -*- coding: utf-8 -*-
"""
Supply Curve computation integrated tests
"""
import os

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from reV import TESTDATADIR
from reV.supply_curve.supply_curve import CompetitiveWindFarms, SupplyCurve
from reV.utilities import SupplyCurveField

TRANS_COSTS = {'line_tie_in_cost': 200, 'line_cost': 1000,
               'station_tie_in_cost': 50, 'center_tie_in_cost': 10,
               'sink_tie_in_cost': 100}
AVAIL_CAP_FRAC = 0.3

SC_POINTS = os.path.join(TESTDATADIR, 'sc_out/baseline_agg_summary.csv')
TRANS_TABLE = os.path.join(
    TESTDATADIR, 'trans_tables/ri_transmission_table.csv')
MULTIPLIERS = os.path.join(
    TESTDATADIR, 'trans_tables/transmission_multipliers.csv')
WIND_DIRS = os.path.join(TESTDATADIR, 'comp_wind_farms/wind_dirs.csv')


@pytest.mark.parametrize('downwind', [False, True])
def test_competitive_wind_dirs(downwind):
    """Run CompetitiveWindFarms and verify results against baseline file."""

    sc_points = CompetitiveWindFarms.run(WIND_DIRS, SC_POINTS,
                                         n_dirs=2,
                                         sort_on=SupplyCurveField.MEAN_LCOE,
                                         downwind=downwind)

    if downwind:
        baseline = os.path.join(TESTDATADIR, 'comp_wind_farms',
                                'sc_points_downwind.csv')
    else:
        baseline = os.path.join(TESTDATADIR, 'comp_wind_farms',
                                'sc_points_upwind.csv')

    if not os.path.exists(baseline):
        sc_points.to_csv(baseline, index=False)
    else:
        baseline = pd.read_csv(baseline)
        baseline = baseline.rename(columns=SupplyCurveField.map_from_legacy())

    sc_points = sc_points.sort_values(by=SupplyCurveField.SC_GID)
    sc_points = sc_points.reset_index(drop=True)
    baseline = baseline.sort_values(by=SupplyCurveField.SC_GID)
    baseline = baseline.reset_index(drop=True)

    assert_frame_equal(sc_points, baseline, check_dtype=False)


@pytest.mark.parametrize('downwind', [False, True])
def test_sc_full_wind_dirs(downwind):
    """Run the full SC test and verify results against baseline file."""

    sc = SupplyCurve(SC_POINTS, TRANS_TABLE, sc_features=MULTIPLIERS)
    sc_out = sc.full_sort(fcr=0.1, transmission_costs=TRANS_COSTS,
                          avail_cap_frac=AVAIL_CAP_FRAC, wind_dirs=WIND_DIRS,
                          downwind=downwind)

    if downwind:
        baseline = os.path.join(TESTDATADIR, 'comp_wind_farms',
                                'sc_full_downwind.csv')
    else:
        baseline = os.path.join(TESTDATADIR, 'comp_wind_farms',
                                'sc_full_upwind.csv')

    if not os.path.exists(baseline):
        sc_out.to_csv(baseline, index=False)
    else:
        baseline = pd.read_csv(baseline)
        baseline = baseline.rename(columns=SupplyCurveField.map_from_legacy())

    assert_frame_equal(sc_out[baseline.columns], baseline, check_dtype=False)


@pytest.mark.parametrize('downwind', [False, True])
def test_sc_simple_wind_dirs(downwind):
    """Run the simple SC test and verify results against baseline file."""
    sc = SupplyCurve(SC_POINTS, TRANS_TABLE, sc_features=MULTIPLIERS)
    sc_out = sc.simple_sort(fcr=0.1, transmission_costs=TRANS_COSTS,
                            wind_dirs=WIND_DIRS, downwind=downwind)

    if downwind:
        baseline = os.path.join(TESTDATADIR, 'comp_wind_farms',
                                'sc_simple_downwind.csv')
    else:
        baseline = os.path.join(TESTDATADIR, 'comp_wind_farms',
                                'sc_simple_upwind.csv')

    if not os.path.exists(baseline):
        sc_out.to_csv(baseline, index=False)
    else:
        baseline = pd.read_csv(baseline)
        baseline = baseline.rename(columns=SupplyCurveField.map_from_legacy())

    assert_frame_equal(sc_out[baseline.columns], baseline, check_dtype=False)


def test_upwind_exclusion():
    """
    Ensure all upwind neighbors are excluded
    """
    cwf = CompetitiveWindFarms(WIND_DIRS, SC_POINTS, n_dirs=2)

    sc_out = os.path.join(TESTDATADIR, 'comp_wind_farms',
                          'sc_full_upwind.csv')
    sc_out = pd.read_csv(sc_out).sort_values('total_lcoe')
    sc_out = sc_out.rename(columns=SupplyCurveField.map_from_legacy())

    sc_point_gids = sc_out[SupplyCurveField.SC_POINT_GID].values.tolist()
    for _, row in sc_out.iterrows():
        sc_gid = row[SupplyCurveField.SC_GID]
        sc_point_gids.remove(row[SupplyCurveField.SC_POINT_GID])
        sc_point_gid = cwf[SupplyCurveField.SC_POINT_GID, sc_gid]
        for gid in cwf['upwind', sc_point_gid]:
            msg = 'Upwind gid {} was not excluded!'.format(gid)
            assert gid not in sc_point_gids, msg


def test_upwind_downwind_exclusion():
    """
    Ensure all upwind and downwind neighbors are excluded
    """
    cwf = CompetitiveWindFarms(WIND_DIRS, SC_POINTS, n_dirs=2)

    sc_out = os.path.join(TESTDATADIR, 'comp_wind_farms',
                          'sc_full_downwind.csv')
    sc_out = pd.read_csv(sc_out).sort_values('total_lcoe')
    sc_out = sc_out.rename(columns=SupplyCurveField.map_from_legacy())

    sc_point_gids = sc_out[SupplyCurveField.SC_POINT_GID].values.tolist()
    for _, row in sc_out.iterrows():
        sc_gid = row[SupplyCurveField.SC_GID]
        sc_point_gids.remove(row[SupplyCurveField.SC_POINT_GID])
        sc_point_gid = cwf[SupplyCurveField.SC_POINT_GID, sc_gid]
        for gid in cwf['upwind', sc_point_gid]:
            msg = 'Upwind gid {} was not excluded!'.format(gid)
            assert gid not in sc_point_gids, msg

        for gid in cwf['downwind', sc_point_gid]:
            msg = 'downwind gid {} was not excluded!'.format(gid)
            assert gid not in sc_point_gids, msg


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
