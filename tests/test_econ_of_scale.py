# -*- coding: utf-8 -*-
"""
PyTest file for reV LCOE economies of scale
"""
import h5py
import numpy as np
import pytest
import os
import shutil
import tempfile

from reV.generation.generation import Gen
from reV.econ.economies_of_scale import EconomiesOfScale
from reV.supply_curve.sc_aggregation import SupplyCurveAggregation
from reV import TESTDATADIR

EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
GEN = os.path.join(TESTDATADIR, 'gen_out/ri_my_pv_gen.h5')
TM_DSET = 'techmap_nsrdb'
RES_CLASS_DSET = 'ghi_mean-means'
RES_CLASS_BINS = [0, 4, 100]
DATA_LAYERS = {'pct_slope': {'dset': 'ri_srtm_slope',
                             'method': 'mean'},
               'reeds_region': {'dset': 'ri_reeds_regions',
                                'method': 'mode'},
               'padus': {'dset': 'ri_padus',
                         'method': 'mode'}}

EXCL_DICT = {'ri_srtm_slope': {'inclusion_range': (None, 5),
                               'exclude_nodata': True},
             'ri_padus': {'exclude_values': [1],
                          'exclude_nodata': True}}

RTOL = 0.001


def test_pass_through_lcoe_args():
    """Test that the kwarg works to pass through LCOE input args from the SAM
    input to the reV output."""
    year = 2012
    rev2_points = slice(0, 3)
    res_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_{}.h5'.format(year))
    sam_files = os.path.join(TESTDATADIR, 'SAM/i_windpower_lcoe.json')

    output_request = ('cf_mean',
                      'lcoe_fcr',
                      'capital_cost',
                      'fixed_charge_rate',
                      'variable_operating_cost',
                      'fixed_operating_cost')

    # run reV 2.0 generation
    gen = Gen.reV_run(tech='windpower', points=rev2_points,
                      sam_files=sam_files, res_file=res_file, max_workers=1,
                      sites_per_worker=1, fout=None,
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


def test_sc_agg_econ_scale():
    """Test supply curve aggregation with LCOE scaling based on plant capacity.
    """
    data = {'capital_cost': 53455000,
            'fixed_operating_cost': 360000,
            'fixed_charge_rate': 0.096,
            'variable_operating_cost': 0}

    with tempfile.TemporaryDirectory() as td:
        gen_temp = os.path.join(td, 'ri_my_pv_gen.h5')
        shutil.copy(GEN, gen_temp)

        with h5py.File(gen_temp, 'a') as res:
            for k, v in data.items():
                arr = np.full(res['meta'].shape, v)
                res.create_dataset(k, res['meta'].shape, data=arr)
                res[k].attrs['scale_factor'] = 1.0

        eqn = '2 * capacity ** -0.3'
        s = SupplyCurveAggregation.summary(EXCL, gen_temp, TM_DSET,
                                           excl_dict=EXCL_DICT,
                                           res_class_dset=RES_CLASS_DSET,
                                           res_class_bins=RES_CLASS_BINS,
                                           data_layers=DATA_LAYERS,
                                           gids=list(np.arange(10)),
                                           max_workers=1, cap_cost_scale=eqn)

        aep = s['capacity'] * s['mean_cf'] * 8760 * 1000

        true_raw_lcoe = ((data['fixed_charge_rate'] * data['capital_cost']
                          + data['fixed_operating_cost'])
                         / aep + data['variable_operating_cost'])
        true_raw_lcoe *= 1000  # convert $/kwh -> $/MWh

        # Back out the fcr * capital_cost term ($)
        x = ((s['raw_lcoe'] / 1000 - data['variable_operating_cost'])
             * aep - data['fixed_operating_cost'])
        eval_inputs = {k: s[k].values.flatten() for k in s.columns}
        # pylint: disable=eval-used
        scalars = eval(str(eqn), globals(), eval_inputs)
        s['scalars'] = scalars
        x *= scalars
        true_scaled_lcoe = ((x + data['fixed_operating_cost'])
                            / aep + data['variable_operating_cost'])
        true_scaled_lcoe *= 1000  # convert $/kwh -> $/MWh

        assert np.allclose(true_scaled_lcoe, s['mean_lcoe'])
        assert np.allclose(true_raw_lcoe, s['raw_lcoe'])
        s = s.sort_values('capacity')
        assert all(s['mean_lcoe'].diff()[1:] < 0)
        for i in s.index.values:
            if s.loc[i, 'scalars'] < 1:
                assert s.loc[i, 'mean_lcoe'] < s.loc[i, 'raw_lcoe']
            else:
                assert s.loc[i, 'mean_lcoe'] >= s.loc[i, 'raw_lcoe']


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
