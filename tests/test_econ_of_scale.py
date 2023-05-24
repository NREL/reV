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

from rex import Resource
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
                      'system_capacity',
                      'capital_cost',
                      'fixed_charge_rate',
                      'variable_operating_cost',
                      'fixed_operating_cost')

    # run reV 2.0 generation
    gen = Gen('windpower', rev2_points, sam_files, res_file,
              sites_per_worker=1, output_request=output_request)
    gen.run(max_workers=1,)

    checks = [x in gen.out for x in Gen.LCOE_ARGS]
    assert all(checks)
    assert 'lcoe_fcr' in gen.out
    assert 'cf_mean' in gen.out


def test_lcoe_calc_simple():
    """Test the EconomiesOfScale LCOE calculator without cap cost scalar"""
    eqn = None
    # from pvwattsv7 defaults
    data = {'aep': 35188456.00,
            'capital_cost': 53455000.00,
            'foc': 360000.00,
            'voc': 0,
            'fcr': 0.096}

    true_lcoe = ((data['fcr'] * data['capital_cost'] + data['foc'])
                 / (data['aep'] / 1000))
    data['mean_lcoe'] = true_lcoe

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

    eqn = 2
    true_scaled = ((data['fcr'] * eqn * data['capital_cost'] + data['foc'])
                   / (data['aep'] / 1000))
    eos = EconomiesOfScale(eqn, data)
    assert eqn * eos.raw_capital_cost == eos.scaled_capital_cost
    assert eos.raw_capital_cost == data['capital_cost']
    assert np.allclose(eos.raw_lcoe, true_lcoe, rtol=0.001)
    assert np.allclose(eos.scaled_lcoe, true_scaled, rtol=0.001)

    data['system_capacity'] = 2
    eqn = '1 / system_capacity'
    true_scaled = ((data['fcr'] * 0.5 * data['capital_cost'] + data['foc'])
                   / (data['aep'] / 1000))
    eos = EconomiesOfScale(eqn, data)
    assert 0.5 * eos.raw_capital_cost == eos.scaled_capital_cost
    assert eos.raw_capital_cost == data['capital_cost']
    assert np.allclose(eos.raw_lcoe, true_lcoe, rtol=0.001)
    assert np.allclose(eos.scaled_lcoe, true_scaled, rtol=0.001)


def test_econ_of_scale_baseline():
    """Test an economies of scale calculation with scalar = 1 to ensure we can
    reproduce the lcoe values
    """
    data = {'capital_cost': 39767200,
            'fixed_operating_cost': 260000,
            'fixed_charge_rate': 0.096,
            'system_capacity': 20000,
            'variable_operating_cost': 0}

    with tempfile.TemporaryDirectory() as td:
        gen_temp = os.path.join(td, 'ri_my_pv_gen.h5')
        shutil.copy(GEN, gen_temp)

        # overwrite the LCOE values since i dont know what econ inputs
        # the original test file was run with
        with Resource(GEN) as res:
            cf = res['cf_mean-means']

        lcoe = (1000 * (data['fixed_charge_rate'] * data['capital_cost']
                        + data['fixed_operating_cost'])
                / (cf * data['system_capacity'] * 8760))

        with h5py.File(gen_temp, 'a') as res:
            res['lcoe_fcr-means'][...] = lcoe
            for k, v in data.items():
                arr = np.full(res['meta'].shape, v)
                res.create_dataset(k, res['meta'].shape, data=arr)
                res[k].attrs['scale_factor'] = 1.0

        base = SupplyCurveAggregation.summary(EXCL, gen_temp, TM_DSET,
                                              excl_dict=EXCL_DICT,
                                              res_class_dset=RES_CLASS_DSET,
                                              res_class_bins=RES_CLASS_BINS,
                                              data_layers=DATA_LAYERS,
                                              gids=list(np.arange(10)),
                                              max_workers=1)

        s = SupplyCurveAggregation.summary(EXCL, gen_temp, TM_DSET,
                                           excl_dict=EXCL_DICT,
                                           res_class_dset=RES_CLASS_DSET,
                                           res_class_bins=RES_CLASS_BINS,
                                           data_layers=DATA_LAYERS,
                                           gids=list(np.arange(10)),
                                           max_workers=1,
                                           cap_cost_scale='1')

        assert np.allclose(base['mean_lcoe'], s['mean_lcoe'])
        assert (s['capital_cost_scalar'] == 1).all()


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

        eqn = '2 * np.multiply(1000, capacity) ** -0.3'
        base = SupplyCurveAggregation.summary(EXCL, gen_temp, TM_DSET,
                                              excl_dict=EXCL_DICT,
                                              res_class_dset=RES_CLASS_DSET,
                                              res_class_bins=RES_CLASS_BINS,
                                              data_layers=DATA_LAYERS,
                                              gids=list(np.arange(10)),
                                              max_workers=1)
        s = SupplyCurveAggregation.summary(EXCL, gen_temp, TM_DSET,
                                           excl_dict=EXCL_DICT,
                                           res_class_dset=RES_CLASS_DSET,
                                           res_class_bins=RES_CLASS_BINS,
                                           data_layers=DATA_LAYERS,
                                           gids=list(np.arange(10)),
                                           max_workers=1, cap_cost_scale=eqn)

        # check that econ of scale saved the raw lcoe and that it reduced all
        # of the mean lcoe values from baseline
        assert np.allclose(s['raw_lcoe'], base['mean_lcoe'])
        assert all(s['mean_lcoe'] < base['mean_lcoe'])

        aep = ((s['mean_fixed_charge_rate'] * s['mean_capital_cost']
                + s['mean_fixed_operating_cost']) / s['raw_lcoe'])

        true_raw_lcoe = ((data['fixed_charge_rate'] * data['capital_cost']
                          + data['fixed_operating_cost'])
                         / aep + data['variable_operating_cost'])

        eval_inputs = {k: s[k].values.flatten() for k in s.columns}
        # pylint: disable=eval-used
        scalars = eval(str(eqn), globals(), eval_inputs)
        s['scalars'] = scalars
        true_scaled_lcoe = ((data['fixed_charge_rate']
                             * scalars * data['capital_cost']
                             + data['fixed_operating_cost'])
                            / aep + data['variable_operating_cost'])

        assert np.allclose(scalars, s['capital_cost_scalar'])

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
