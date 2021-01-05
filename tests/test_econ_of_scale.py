# -*- coding: utf-8 -*-
"""
PyTest file for reV LCOE economies of scale
"""
import numpy as np
import pytest
import os

from reV.generation.generation import Gen
from reV.econ.economies_of_scale import EconomiesOfScale
from reV import TESTDATADIR


def test_pass_through_lcoe_args():
    """Test that the kwarg works to pass through LCOE input args from the SAM
    input to the reV output."""
    year = 2012
    rev2_points = slice(0, 3)
    res_file = TESTDATADIR + '/wtk/ri_100_wtk_{}.h5'.format(year)
    sam_files = TESTDATADIR + '/SAM/i_windpower_lcoe.json'

    output_request = ('cf_mean', 'lcoe_fcr')

    # run reV 2.0 generation
    gen = Gen.reV_run(tech='windpower', points=rev2_points,
                      sam_files=sam_files, res_file=res_file, max_workers=1,
                      sites_per_worker=1, fout=None,
                      pass_through_lcoe_args=True,
                      output_request=output_request)

    checks = [x in gen.out for x in Gen.LCOE_ARGS]
    assert all(checks)
    assert 'lcoe_fcr' in gen.out
    assert 'cf_mean' in gen.out


def test_lcoe_calc_noscale():
    """Test the EconomiesOfScale LCOE calculator without cap cost scalar"""
    eqn = None
    # from pvwattsv7 defaults
    data = {'aep': 35188456.00,
            'capital_cost': 53455000.00,
            'foc': 360000.00,
            'voc': 0,
            'fcr': 0.096}
    true_lcoe = 15.62  # cents/kWh
    true_lcoe *= 10  # $/MWh

    eos = EconomiesOfScale(eqn, data)
    assert eos.raw_capital_cost == eos.scaled_capital_cost
    assert eos.raw_capital_cost == data['capital_cost']
    assert np.allclose(eos.raw_lcoe, true_lcoe, rtol=0.001)
    assert np.allclose(eos.scaled_lcoe, true_lcoe, rtol=0.001)

    eqn = 1
    eos = EconomiesOfScale(eqn, data)
    assert eos.raw_capital_cost == eos.scaled_capital_cost
    assert eos.raw_capital_cost == data['capital_cost']
    assert np.allclose(eos.raw_lcoe, true_lcoe, rtol=0.001)
    assert np.allclose(eos.scaled_lcoe, true_lcoe, rtol=0.001)

    aep = data.pop('aep')
    data['mean_cf'] = 0.201
    data['capacity'] = 20
    eos = EconomiesOfScale(eqn, data)
    assert np.allclose(aep, eos.aep, rtol=0.001)
    assert eos.raw_capital_cost == eos.scaled_capital_cost
    assert eos.raw_capital_cost == data['capital_cost']
    assert np.allclose(eos.raw_lcoe, true_lcoe, rtol=0.002)
    assert np.allclose(eos.scaled_lcoe, true_lcoe, rtol=0.002)


def test_econ_scale_eqn():
    """Test the EconomiesOfScale LCOE calculator with capacity scaling"""
    eqn = 0.5
    # from pvwattsv7 defaults
    data = {'aep': 35188456.00,
            'capital_cost': 53455000.00,
            'foc': 360000.00,
            'voc': 0,
            'fcr': 0.096}
    true_raw_lcoe = 15.62  # cents/kWh
    true_raw_lcoe *= 10  # cents/kWh -> $/MWh

    # Back out the fcr * capital_cost term ($)
    x = (true_raw_lcoe / 1000 - data['voc']) * data['aep'] - data['foc']
    # pylint: disable=eval-used
    x *= eval(str(eqn))
    true_scaled_lcoe = (x + data['foc']) / data['aep'] + data['voc']
    true_scaled_lcoe *= 1000  # $/kWh -> $/MWh
    eos = EconomiesOfScale(eqn, data)
    assert np.allclose(true_scaled_lcoe, eos.scaled_lcoe, rtol=0.001)
    assert np.allclose(true_raw_lcoe, eos.raw_lcoe, rtol=0.001)

    eqn = '2 * capacity ** -0.3'
    aep = data.pop('aep')
    data['mean_cf'] = 0.201
    data['capacity'] = 20
    x = (true_raw_lcoe / 1000 - data['voc']) * aep - data['foc']
    # pylint: disable=eval-used
    x *= eval(str(eqn), globals(), data)  # x is fcr*capital_cost in $
    true_scaled_lcoe = (x + data['foc']) / aep + data['voc']
    true_scaled_lcoe *= 1000  # $/kWh -> $/MWh
    eos = EconomiesOfScale(eqn, data)
    assert np.allclose(aep, eos.aep, rtol=0.001)
    assert np.allclose(true_scaled_lcoe, eos.scaled_lcoe, rtol=0.002)
    assert np.allclose(true_raw_lcoe, eos.raw_lcoe, rtol=0.002)

    data['capacity'] = np.arange(20, 400, 20)
    eos = EconomiesOfScale(eqn, data)
    assert all(np.diff(eos.scaled_lcoe) < 0)


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
