# -*- coding: utf-8 -*-
"""
Supply Curve computation integrated tests
"""
import os
import numpy as np
import pandas as pd
import pytest
import warnings

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
        for c in baseline.columns:
            numeric = (np.issubdtype(sc_full[c].values.dtype, np.floating)
                       | np.issubdtype(sc_full[c].values.dtype, np.integer))
            if numeric:
                np.seterr(divide='ignore', invalid='ignore')
                diff = np.divide(baseline[c].values,
                                 (baseline[c].values - sc_full[c].values))
                diff = np.nan_to_num(diff)
                msg = 'Bad column: {}, max diff: {}'.format(c, diff.max())
                assert np.allclose(baseline[c].values,
                                   sc_full[c].values, rtol=0.01), msg
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
