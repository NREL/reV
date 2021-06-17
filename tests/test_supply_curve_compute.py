# -*- coding: utf-8 -*-
"""
Supply Curve computation integrated tests
"""
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import warnings
import numpy as np

from reV import TESTDATADIR
from reV.supply_curve.supply_curve import SupplyCurve


TRANS_COSTS_1 = {'line_tie_in_cost': 200, 'line_cost': 1000,
                 'station_tie_in_cost': 50, 'center_tie_in_cost': 10,
                 'sink_tie_in_cost': 100, 'available_capacity': 0.3}


TRANS_COSTS_2 = {'line_tie_in_cost': 3000, 'line_cost': 2000,
                 'station_tie_in_cost': 500, 'center_tie_in_cost': 100,
                 'sink_tie_in_cost': 1e6, 'available_capacity': 0.9}


@pytest.fixture
def sc_points():
    """Get the supply curve aggregation summary table"""
    path = os.path.join(TESTDATADIR, 'sc_out/baseline_agg_summary.csv')
    sc_points = pd.read_csv(path)
    return sc_points


@pytest.fixture
def sc_points_friction():
    """Get the supply curve aggregation summary table with friction surface"""
    path = os.path.join(TESTDATADIR,
                        'sc_out/baseline_agg_summary_friction.csv')
    sc_points = pd.read_csv(path)
    return sc_points


@pytest.fixture
def trans_table():
    """Get the transmission mapping table"""
    path = os.path.join(TESTDATADIR, 'trans_tables/ri_transmission_table.csv')
    trans_table = pd.read_csv(path)
    return trans_table


@pytest.fixture
def multipliers():
    """Get table of transmission multipliers"""
    path = os.path.join(TESTDATADIR,
                        'trans_tables/transmission_multipliers.csv')
    multipliers = pd.read_csv(path)
    return multipliers


def baseline_verify(sc_full, fpath_baseline):
    """Verify numerical columns in a CSV against a baseline file."""

    if os.path.exists(fpath_baseline):
        baseline = pd.read_csv(fpath_baseline)
        assert_frame_equal(baseline, sc_full, check_dtype=False)
    else:
        sc_full.to_csv(fpath_baseline, index=False)


@pytest.mark.parametrize(('i', 'tcosts'), ((1, TRANS_COSTS_1),
                                           (2, TRANS_COSTS_2)))
def test_integrated_sc_full(i, tcosts, sc_points, trans_table, multipliers):
    """Run the full SC test and verify results against baseline file."""

    sc_full = SupplyCurve.full(sc_points, trans_table, fcr=0.1,
                               sc_features=multipliers,
                               transmission_costs=tcosts)
    fpath_baseline = os.path.join(TESTDATADIR,
                                  'sc_out/sc_full_out_{}.csv'.format(i))
    baseline_verify(sc_full, fpath_baseline)


@pytest.mark.parametrize(('i', 'tcosts'), ((1, TRANS_COSTS_1),
                                           (2, TRANS_COSTS_2)))
def test_integrated_sc_simple(i, tcosts, sc_points, trans_table, multipliers):
    """Run the simple SC test and verify results against baseline file."""

    sc_simple = SupplyCurve.simple(sc_points, trans_table, fcr=0.1,
                                   sc_features=multipliers,
                                   transmission_costs=tcosts)

    fpath_baseline = os.path.join(TESTDATADIR,
                                  'sc_out/sc_simple_out_{}.csv'.format(i))
    baseline_verify(sc_simple, fpath_baseline)


def test_integrated_sc_full_friction(sc_points_friction, trans_table,
                                     multipliers):
    """Run the full SC algorithm with friction"""

    sc_full = SupplyCurve.full(sc_points_friction, trans_table, fcr=0.1,
                               sc_features=multipliers,
                               transmission_costs=TRANS_COSTS_1,
                               sort_on='total_lcoe_friction')

    assert 'mean_lcoe_friction' in sc_full
    assert 'total_lcoe_friction' in sc_full
    test = sc_full['mean_lcoe_friction'] + sc_full['lcot']
    assert np.allclose(test, sc_full['total_lcoe_friction'])

    fpath_baseline = os.path.join(TESTDATADIR,
                                  'sc_out/sc_full_out_friction.csv')
    baseline_verify(sc_full, fpath_baseline)


def test_integrated_sc_simple_friction(sc_points_friction, trans_table,
                                       multipliers):
    """Run the simple SC algorithm with friction"""

    sc_simple = SupplyCurve.simple(sc_points_friction, trans_table, fcr=0.1,
                                   sc_features=multipliers,
                                   transmission_costs=TRANS_COSTS_1,
                                   sort_on='total_lcoe_friction')

    assert 'mean_lcoe_friction' in sc_simple
    assert 'total_lcoe_friction' in sc_simple
    test = sc_simple['mean_lcoe_friction'] + sc_simple['lcot']
    assert np.allclose(test, sc_simple['total_lcoe_friction'])

    fpath_baseline = os.path.join(TESTDATADIR,
                                  'sc_out/sc_simple_out_friction.csv')
    baseline_verify(sc_simple, fpath_baseline)


def test_sc_warning1(sc_points, trans_table, multipliers):
    """Run the full SC test with missing connections and verify warning."""
    mask = trans_table['sc_point_gid'].isin(list(range(10)))
    trans_table = trans_table[~mask]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        SupplyCurve.full(sc_points, trans_table, fcr=0.1,
                         sc_features=multipliers,
                         transmission_costs=TRANS_COSTS_1)

        s1 = str(list(range(10))).replace(']', '').replace('[', '')
        s2 = str(w[0].message)
        msg = ('Warning failed! Should have had missing sc_gids 0 through 9: '
               '{}'.format(s2))
        assert s1 in s2, msg


def test_sc_warning2(sc_points, trans_table, multipliers):
    """Run the full SC test without PCA load centers and verify warning."""
    mask = trans_table['category'] == 'PCALoadCen'
    trans_table = trans_table[~mask]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        SupplyCurve.full(sc_points, trans_table, fcr=0.1,
                         sc_features=multipliers,
                         transmission_costs=TRANS_COSTS_1)

        s1 = 'Unconnected sc_gid'
        s2 = str(w[0].message)
        msg = ('Warning failed! Should have Unconnected sc_gid: '
               '{}'.format(s2))
        assert s1 in s2, msg


def test_parallel(sc_points, trans_table, multipliers):
    """Test a parallel compute against a serial compute"""

    sc_full_parallel = SupplyCurve.full(sc_points, trans_table, fcr=0.1,
                                        sc_features=multipliers,
                                        transmission_costs=TRANS_COSTS_1,
                                        max_workers=4)

    sc_full_serial = SupplyCurve.full(sc_points, trans_table, fcr=0.1,
                                      sc_features=multipliers,
                                      transmission_costs=TRANS_COSTS_1,
                                      max_workers=1)

    assert_frame_equal(sc_full_parallel, sc_full_serial)


def verify_trans_cap(sc_table, trans_tables):
    """
    Verify that sc_points are connected to features in the correct capacity
    bins
    """
    trans_features = []
    for path in trans_tables:
        df = pd.read_csv(path)
        trans_features.append(df[['trans_gid', 'min_cap', 'max_cap']])

    trans_features = pd.concat(trans_features)

    by = ['min_cap', 'max_cap']
    for (min_cap, max_cap), df in trans_features.groupby(by=by):
        mask = sc_table['capacity'] > min_cap
        mask &= sc_table['capacity'] <= max_cap
        msg = ("SC points were not connected to a transmission feature with "
               "capacity between {} and {}".format(min_cap, max_cap))
        assert all(sc_table.loc[mask, 'trans_gid'].isin(df['trans_gid'])), msg


def test_least_cost_full(sc_points):
    """
    Test full supply curve sorting with least-cost path transmission tables
    """
    trans_tables = [os.path.join(TESTDATADIR, 'trans_tables',
                                 f'costs_RI_{cap}MW.csv')
                    for cap in [100, 200, 400, 1000]]
    sc_full = SupplyCurve.full(sc_points, trans_tables, fcr=0.1)
    fpath_baseline = os.path.join(TESTDATADIR, 'sc_out/sc_full_lc.csv')
    baseline_verify(sc_full, fpath_baseline)
    verify_trans_cap(sc_full, trans_tables)


def test_least_cost_simple(sc_points):
    """
    Test simple supply curve sorting with least-cost path transmission tables
    """
    trans_tables = [os.path.join(TESTDATADIR, 'trans_tables',
                                 f'costs_RI_{cap}MW.csv')
                    for cap in [100, 200, 400, 1000]]
    sc_simple = SupplyCurve.simple(sc_points, trans_tables, fcr=0.1)
    fpath_baseline = os.path.join(TESTDATADIR, 'sc_out/sc_simple_lc.csv')
    baseline_verify(sc_simple, fpath_baseline)
    verify_trans_cap(sc_simple, trans_tables)


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
