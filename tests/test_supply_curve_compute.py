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
from reV.utilities.exceptions import SupplyCurveInputError

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
        # double check useful for when tables are changing
        # but lcoe should be the same
        assert np.allclose(baseline['total_lcoe'], sc_full['total_lcoe'])
        assert_frame_equal(baseline, sc_full, check_dtype=False)
    else:
        sc_full.to_csv(fpath_baseline, index=False)


@pytest.mark.parametrize(('i', 'trans_costs'), ((1, TRANS_COSTS_1),
                                                (2, TRANS_COSTS_2)))
def test_integrated_sc_full(i, trans_costs, sc_points, trans_table,
                            multipliers):
    """Run the full SC test and verify results against baseline file."""
    tcosts = trans_costs.copy()
    avail_cap_frac = tcosts.pop('available_capacity', 1)
    sc_full = SupplyCurve.full(sc_points, trans_table, fcr=0.1,
                               sc_features=multipliers,
                               transmission_costs=tcosts,
                               avail_cap_frac=avail_cap_frac)
    fpath_baseline = os.path.join(TESTDATADIR,
                                  'sc_out/sc_full_out_{}.csv'.format(i))
    baseline_verify(sc_full, fpath_baseline)


@pytest.mark.parametrize(('i', 'trans_costs'), ((1, TRANS_COSTS_1),
                                                (2, TRANS_COSTS_2)))
def test_integrated_sc_simple(i, trans_costs, sc_points, trans_table,
                              multipliers):
    """Run the simple SC test and verify results against baseline file."""
    tcosts = trans_costs.copy()
    tcosts.pop('available_capacity', 1)
    sc_simple = SupplyCurve.simple(sc_points, trans_table, fcr=0.1,
                                   sc_features=multipliers,
                                   transmission_costs=tcosts)

    fpath_baseline = os.path.join(TESTDATADIR,
                                  'sc_out/sc_simple_out_{}.csv'.format(i))
    baseline_verify(sc_simple, fpath_baseline)


def test_integrated_sc_full_friction(sc_points_friction, trans_table,
                                     multipliers):
    """Run the full SC algorithm with friction"""
    tcosts = TRANS_COSTS_1.copy()
    avail_cap_frac = tcosts.pop('available_capacity', 1)
    sc_full = SupplyCurve.full(sc_points_friction, trans_table, fcr=0.1,
                               sc_features=multipliers,
                               transmission_costs=tcosts,
                               avail_cap_frac=avail_cap_frac,
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
    tcosts = TRANS_COSTS_1.copy()
    tcosts.pop('available_capacity', 1)
    sc_simple = SupplyCurve.simple(sc_points_friction, trans_table, fcr=0.1,
                                   sc_features=multipliers,
                                   transmission_costs=tcosts,
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
    tcosts = TRANS_COSTS_1.copy()
    avail_cap_frac = tcosts.pop('available_capacity', 1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        SupplyCurve.full(sc_points, trans_table, fcr=0.1,
                         sc_features=multipliers,
                         transmission_costs=tcosts,
                         avail_cap_frac=avail_cap_frac)

        s1 = str(list(range(10))).replace(']', '').replace('[', '')
        s2 = str(w[0].message)
        msg = ('Warning failed! Should have had missing sc_gids 0 through 9: '
               '{}'.format(s2))
        assert s1 in s2, msg


def test_sc_warning2(sc_points, trans_table, multipliers):
    """Run the full SC test without PCA load centers and verify warning."""
    mask = trans_table['category'] == 'PCALoadCen'
    trans_table = trans_table[~mask]
    tcosts = TRANS_COSTS_1.copy()
    avail_cap_frac = tcosts.pop('available_capacity', 1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        SupplyCurve.full(sc_points, trans_table, fcr=0.1,
                         sc_features=multipliers,
                         transmission_costs=tcosts,
                         avail_cap_frac=avail_cap_frac)

        s1 = 'Unconnected sc_gid'
        s2 = str(w[0].message)
        msg = ('Warning failed! Should have Unconnected sc_gid: '
               '{}'.format(s2))
        assert s1 in s2, msg


def test_parallel(sc_points, trans_table, multipliers):
    """Test a parallel compute against a serial compute"""

    tcosts = TRANS_COSTS_1.copy()
    avail_cap_frac = tcosts.pop('available_capacity', 1)
    sc_full_parallel = SupplyCurve.full(sc_points, trans_table, fcr=0.1,
                                        sc_features=multipliers,
                                        transmission_costs=tcosts,
                                        avail_cap_frac=avail_cap_frac,
                                        max_workers=4)

    sc_full_serial = SupplyCurve.full(sc_points, trans_table, fcr=0.1,
                                      sc_features=multipliers,
                                      transmission_costs=tcosts,
                                      avail_cap_frac=avail_cap_frac,
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
        trans_features.append(df[['trans_gid', 'max_cap']])

    trans_features = pd.concat(trans_features)

    test = sc_table.merge(trans_features, on='trans_gid', how='left')
    mask = test['capacity'] > test['max_cap']
    cols = ['sc_gid', 'trans_gid', 'capacity', 'max_cap']
    msg = ("SC points connected to transmission features with "
           "max_cap < sc_cap:\n{}"
           .format(test.loc[mask, cols]))
    assert any(mask), msg


def test_least_cost_full(sc_points):
    """
    Test full supply curve sorting with least-cost path transmission tables
    """
    trans_tables = [os.path.join(TESTDATADIR, 'trans_tables',
                                 f'costs_RI_{cap}MW.csv')
                    for cap in [100, 200, 400, 1000]]
    sc_full = SupplyCurve.full(sc_points, trans_tables, fcr=0.1,
                               avail_cap_frac=0.1)
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


def test_simple_trans_table(sc_points):
    """
    Run the simple SC test using a simple transmission table
    and verify results against baseline file.
    """
    trans_table = os.path.join(TESTDATADIR,
                               'trans_tables',
                               'ri_simple_transmission_table.csv')
    sc_simple = SupplyCurve.simple(sc_points, trans_table, fcr=0.1)

    fpath_baseline = os.path.join(TESTDATADIR, 'sc_out/ri_sc_simple_lc.csv')
    baseline_verify(sc_simple, fpath_baseline)


def test_substation_conns(sc_points, trans_table, multipliers):
    """
    Ensure missing trans lines are caught by SupplyCurveInputError
    """
    tcosts = TRANS_COSTS_1.copy()
    avail_cap_frac = tcosts.pop('available_capacity', 1)
    drop_lines = np.where(trans_table['category'] == 'TransLine')[0]
    drop_lines = np.random.choice(drop_lines, 10, replace=False)
    trans_table = trans_table.drop(labels=drop_lines)

    with pytest.raises(SupplyCurveInputError):
        SupplyCurve.full(sc_points, trans_table, fcr=0.1,
                         sc_features=multipliers,
                         transmission_costs=tcosts,
                         avail_cap_frac=avail_cap_frac,
                         max_workers=4)


def test_multi_parallel_trans(sc_points):
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

    path = os.path.join(TESTDATADIR, 'sc_out/baseline_agg_summary.csv')
    sc_points = pd.read_csv(path)

    columns = ('trans_gid', 'trans_type', 'n_parallel_trans',
               'lcot', 'total_lcoe', 'trans_cap_cost_per_mw',
               'max_cap')

    trans_tables = [os.path.join(TESTDATADIR, 'trans_tables',
                                 f'costs_RI_{cap}MW.csv')
                    for cap in [100, 200, 400, 1000]]
    sc_1 = SupplyCurve.simple(sc_points, trans_tables, fcr=0.1,
                              columns=columns)

    trans_tables = [os.path.join(TESTDATADIR, 'trans_tables',
                                 f'costs_RI_{cap}MW.csv')
                    for cap in [100]]
    sc_2 = SupplyCurve.simple(sc_points, trans_tables, fcr=0.1,
                              columns=columns)

    assert not set(sc_points['sc_gid']) - set(sc_1['sc_gid'])
    assert not set(sc_points['sc_gid']) - set(sc_2['sc_gid'])
    assert not set(sc_points['sc_point_gid']) - set(sc_1['sc_point_gid'])
    assert not set(sc_points['sc_point_gid']) - set(sc_2['sc_point_gid'])
    assert not set(sc_1['sc_point_gid']) - set(sc_points['sc_point_gid'])
    assert not set(sc_2['sc_point_gid']) - set(sc_points['sc_point_gid'])

    assert (sc_2.n_parallel_trans > 1).any()

    mask_2 = sc_2['n_parallel_trans'] > 1

    for gid in sc_2.loc[mask_2, 'sc_gid']:
        nx_1 = sc_1.loc[(sc_1['sc_gid'] == gid), 'n_parallel_trans'].values[0]
        nx_2 = sc_2.loc[(sc_2['sc_gid'] == gid), 'n_parallel_trans'].values[0]
        assert nx_2 >= nx_1
        if nx_1 != nx_2:
            lcot_1 = sc_1.loc[(sc_1['sc_gid'] == gid), 'lcot'].values[0]
            lcot_2 = sc_2.loc[(sc_2['sc_gid'] == gid), 'lcot'].values[0]
            assert lcot_2 > lcot_1


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
