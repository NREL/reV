# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Created on Wed Jun 19 15:37:05 2019

@author: gbuster
"""
import json
import os
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import tempfile
import shutil
import h5py
import json
import shutil
import traceback

from reV.cli import main, gaps_cli
from reV.econ.utilities import lcoe_fcr
from reV.supply_curve.sc_aggregation import SupplyCurveAggregation
from reV.config.supply_curve_configs import SupplyCurveAggregationConfig
from reV.utilities import ModuleName
from reV import TESTDATADIR


EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
RES = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_2012.h5')
GEN = os.path.join(TESTDATADIR, 'gen_out/ri_my_pv_gen.h5')
ONLY_GEN = os.path.join(TESTDATADIR, 'gen_out/ri_my_pv_only_gen.h5')
ONLY_ECON = os.path.join(TESTDATADIR, 'gen_out/ri_my_pv_only_econ.h5')
AGG_BASELINE = os.path.join(TESTDATADIR, 'sc_out/baseline_agg_summary.csv')
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


def test_agg_extent(resolution=64):
    """Get the SC points aggregation summary and test that there are expected
    columns and that all resource gids were found"""

    sca = SupplyCurveAggregation(EXCL, TM_DSET, excl_dict=EXCL_DICT,
                                 res_class_dset=None, res_class_bins=None,
                                 data_layers=DATA_LAYERS,
                                 resolution=resolution)
    summary = sca.summarize(GEN)

    all_res_gids = []
    for gids in summary['res_gids']:
        all_res_gids += gids

    assert 'sc_col_ind' in summary
    assert 'sc_row_ind' in summary
    assert 'gen_gids' in summary
    assert len(set(all_res_gids)) == 177


def test_parallel_agg(resolution=64):
    """Test that parallel aggregation yields the same results as serial
    aggregation."""

    gids = list(range(50, 70))
    sca = SupplyCurveAggregation(EXCL, TM_DSET, excl_dict=EXCL_DICT,
                                 res_class_dset=None, res_class_bins=None,
                                 data_layers=DATA_LAYERS, gids=gids,
                                 resolution=resolution)
    summary_serial = sca.summarize(GEN, max_workers=1)
    summary_parallel = sca.summarize(GEN, max_workers=None,
                                     sites_per_worker=10)

    assert all(summary_serial == summary_parallel)


def test_agg_summary():
    """Test the aggregation summary method against a baseline file."""

    sca = SupplyCurveAggregation(EXCL, TM_DSET, excl_dict=EXCL_DICT,
                                 res_class_dset=RES_CLASS_DSET,
                                 res_class_bins=RES_CLASS_BINS,
                                 data_layers=DATA_LAYERS)
    summary = sca.summarize(GEN, max_workers=1)

    if not os.path.exists(AGG_BASELINE):
        summary.to_csv(AGG_BASELINE)
        raise Exception('Aggregation summary baseline file did not exist. '
                        'Created: {}'.format(AGG_BASELINE))

    else:
        for c in ['res_gids', 'gen_gids', 'gid_counts']:
            summary[c] = summary[c].astype(str)

        s_baseline = pd.read_csv(AGG_BASELINE, index_col=0)

        summary = summary.fillna('None')
        s_baseline = s_baseline.fillna('None')

        assert_frame_equal(summary, s_baseline, check_dtype=False, rtol=0.0001)


def test_multi_file_excl():
    """Test sc aggregation with multple exclusion file inputs."""

    excl_dict = {'ri_srtm_slope': {'inclusion_range': (None, 5),
                                   'exclude_nodata': True},
                 'ri_padus': {'exclude_values': [1],
                              'exclude_nodata': True},
                 'excl_test': {'include_values': [1],
                               'weight': 0.5},
                 }

    with tempfile.TemporaryDirectory() as td:
        excl_temp_1 = os.path.join(td, 'excl1.h5')
        excl_temp_2 = os.path.join(td, 'excl2.h5')
        shutil.copy(EXCL, excl_temp_1)
        shutil.copy(EXCL, excl_temp_2)

        with h5py.File(excl_temp_1, 'a') as f:
            shape = f['latitude'].shape
            attrs = dict(f['ri_srtm_slope'].attrs)
            data = np.ones(shape)
            test_dset = 'excl_test'
            f.create_dataset(test_dset, shape, data=data)
            for k, v in attrs.items():
                f[test_dset].attrs[k] = v
            del f['ri_srtm_slope']

        sca = SupplyCurveAggregation((excl_temp_1, excl_temp_2), TM_DSET,
                                     excl_dict=excl_dict,
                                     res_class_dset=RES_CLASS_DSET,
                                     res_class_bins=RES_CLASS_BINS)
        summary = sca.summarize(GEN)

        s_baseline = pd.read_csv(AGG_BASELINE, index_col=0)

        summary = summary.fillna('None')
        s_baseline = s_baseline.fillna('None')

        assert np.allclose(summary['area_sq_km'] * 2, s_baseline['area_sq_km'])


@pytest.mark.parametrize('pre_extract', (True, False))
def test_pre_extract_inclusions(pre_extract):
    """Test the aggregation summary w/ and w/out pre-extracting inclusions"""

    sca = SupplyCurveAggregation(EXCL, TM_DSET, excl_dict=EXCL_DICT,
                                 res_class_dset=RES_CLASS_DSET,
                                 res_class_bins=RES_CLASS_BINS,
                                 data_layers=DATA_LAYERS,
                                 pre_extract_inclusions=pre_extract)
    summary = sca.summarize(GEN, max_workers=1)

    if not os.path.exists(AGG_BASELINE):
        summary.to_csv(AGG_BASELINE)
        raise Exception('Aggregation summary baseline file did not exist. '
                        'Created: {}'.format(AGG_BASELINE))

    else:
        for c in ['res_gids', 'gen_gids', 'gid_counts']:
            summary[c] = summary[c].astype(str)

        s_baseline = pd.read_csv(AGG_BASELINE, index_col=0)

        summary = summary.fillna('None')
        s_baseline = s_baseline.fillna('None')

        assert_frame_equal(summary, s_baseline, check_dtype=False, rtol=0.0001)


def test_agg_gen_econ():
    """Test the aggregation summary method with separate gen and econ
    input files."""

    sca = SupplyCurveAggregation(EXCL, TM_DSET, excl_dict=EXCL_DICT,
                                 res_class_dset=RES_CLASS_DSET,
                                 res_class_bins=RES_CLASS_BINS,
                                 data_layers=DATA_LAYERS)
    summary_base = sca.summarize(GEN, max_workers=1)

    sca = SupplyCurveAggregation(EXCL, TM_DSET, econ_fpath=ONLY_ECON,
                                 excl_dict=EXCL_DICT,
                                 res_class_dset=RES_CLASS_DSET,
                                 res_class_bins=RES_CLASS_BINS,
                                 data_layers=DATA_LAYERS)
    summary_econ = sca.summarize(ONLY_GEN, max_workers=1)

    assert_frame_equal(summary_base, summary_econ)


def test_agg_extra_dsets():
    """Test aggregation with extra datasets to aggregate."""
    h5_dsets = ['lcoe_fcr-2012', 'lcoe_fcr-2013', 'lcoe_fcr-stdev']
    sca = SupplyCurveAggregation(EXCL, TM_DSET, h5_dsets=h5_dsets,
                                 econ_fpath=ONLY_ECON, excl_dict=EXCL_DICT,
                                 res_class_dset=RES_CLASS_DSET,
                                 res_class_bins=RES_CLASS_BINS,
                                 data_layers=DATA_LAYERS)
    summary = sca.summarize(ONLY_GEN, max_workers=1)

    for dset in h5_dsets:
        assert 'mean_{}'.format(dset) in summary.columns

    check = summary['mean_lcoe_fcr-2012'] == summary['mean_lcoe']
    assert not any(check)
    check = summary['mean_lcoe_fcr-2013'] == summary['mean_lcoe']
    assert not any(check)

    avg = (summary['mean_lcoe_fcr-2012'] + summary['mean_lcoe_fcr-2013']) / 2
    assert np.allclose(avg.values, summary['mean_lcoe'].values)


def test_agg_scalar_excl():
    """Test the aggregation summary with exclusions of 0.5"""

    gids_subset = list(range(0, 20))
    excl_dict_1 = {'ri_padus': {'exclude_values': [1]}}
    sca = SupplyCurveAggregation(EXCL, TM_DSET, excl_dict=excl_dict_1,
                                 res_class_dset=RES_CLASS_DSET,
                                 res_class_bins=RES_CLASS_BINS,
                                 data_layers=DATA_LAYERS, gids=gids_subset)
    summary_base = sca.summarize(GEN, max_workers=1)

    excl_dict_2 = {'ri_padus': {'exclude_values': [1],
                                'weight': 0.5}}
    sca = SupplyCurveAggregation(EXCL, TM_DSET, excl_dict=excl_dict_2,
                                 res_class_dset=RES_CLASS_DSET,
                                 res_class_bins=RES_CLASS_BINS,
                                 data_layers=DATA_LAYERS, gids=gids_subset)
    summary_with_weights = sca.summarize(GEN, max_workers=1)

    dsets = ['area_sq_km', 'capacity']
    for dset in dsets:
        diff = (summary_base[dset].values / summary_with_weights[dset].values)
        msg = ('Fractional exclusions failed for {} which has values {} and {}'
               .format(dset, summary_base[dset].values,
                       summary_with_weights[dset].values))
        assert all(diff == 2), msg

    for i in summary_base.index:
        counts_full = summary_base.loc[i, 'gid_counts']
        counts_half = summary_with_weights.loc[i, 'gid_counts']

        for j, counts in enumerate(counts_full):
            msg = ('GID counts for fractional exclusions failed for index {}!'
                   .format(i))
            assert counts == 2 * counts_half[j], msg


def test_data_layer_methods():
    """Test aggregation of data layers with different methods"""
    data_layers = {'pct_slope_mean': {'dset': 'ri_srtm_slope',
                                      'method': 'mean'},
                   'pct_slope_max': {'dset': 'ri_srtm_slope',
                                     'method': 'max'},
                   'pct_slope_min': {'dset': 'ri_srtm_slope',
                                     'method': 'min'},
                   'reeds_region': {'dset': 'ri_reeds_regions',
                                    'method': 'category'},
                   'padus': {'dset': 'ri_padus',
                             'method': 'category'}}

    sca = SupplyCurveAggregation(EXCL, TM_DSET, excl_dict=EXCL_DICT,
                                 res_class_dset=RES_CLASS_DSET,
                                 res_class_bins=RES_CLASS_BINS,
                                 data_layers=data_layers)
    summary = sca.summarize(GEN, max_workers=1)

    for i in summary.index.values:

        # Check categorical data layers
        counts = summary.loc[i, 'gid_counts']
        rr = summary.loc[i, 'reeds_region']
        assert isinstance(rr, str)
        rr = json.loads(rr)
        assert isinstance(rr, dict)
        rr_sum = sum(list(rr.values()))
        padus = summary.loc[i, 'padus']
        assert isinstance(padus, str)
        padus = json.loads(padus)
        assert isinstance(padus, dict)
        padus_sum = sum(list(padus.values()))
        try:
            assert padus_sum == sum(counts)
            assert padus_sum >= rr_sum
        except AssertionError:
            e = ('Categorical data layer aggregation failed:\n{}'
                 .format(summary.loc[i]))
            raise RuntimeError(e)

        # Check min/mean/max of the same data layer
        n = summary.loc[i, 'n_gids']
        slope_mean = summary.loc[i, 'pct_slope_mean']
        slope_max = summary.loc[i, 'pct_slope_max']
        slope_min = summary.loc[i, 'pct_slope_min']
        if n > 3:  # sc points with <= 3 90m pixels can have min == mean == max
            assert slope_min < slope_mean < slope_max
        else:
            assert slope_min <= slope_mean <= slope_max


def test_recalc_lcoe():
    """Test supply curve aggregation with the re-calculation of lcoe using the
    multi-year mean capacity factor"""

    data = {'capital_cost': 34900000,
            'fixed_operating_cost': 280000,
            'fixed_charge_rate': 0.09606382995843887,
            'variable_operating_cost': 0,
            'system_capacity': 20000}
    annual_cf = [0.24, 0.26, 0.37, 0.15]
    annual_lcoe = []
    years = list(range(2012, 2016))

    with tempfile.TemporaryDirectory() as td:
        gen_temp = os.path.join(td, 'ri_my_pv_gen.h5')
        shutil.copy(GEN, gen_temp)

        with h5py.File(gen_temp, 'a') as res:
            for k in [d for d in list(res) if d != 'meta']:
                del res[k]
            for k, v in data.items():
                arr = np.full(res['meta'].shape, v)
                res.create_dataset(k, res['meta'].shape, data=arr)
            for year, cf in zip(years, annual_cf):
                lcoe = lcoe_fcr(data['fixed_charge_rate'],
                                data['capital_cost'],
                                data['fixed_operating_cost'],
                                data['system_capacity'] * cf * 8760,
                                data['variable_operating_cost'])
                cf_arr = np.full(res['meta'].shape, cf)
                lcoe_arr = np.full(res['meta'].shape, lcoe)
                annual_lcoe.append(lcoe)

                res.create_dataset('cf_mean-{}'.format(year),
                                   res['meta'].shape, data=cf_arr)
                res.create_dataset('lcoe_fcr-{}'.format(year),
                                   res['meta'].shape, data=lcoe_arr)

            cf_arr = np.full(res['meta'].shape, np.mean(annual_cf))
            lcoe_arr = np.full(res['meta'].shape, np.mean(annual_lcoe))
            res.create_dataset('cf_mean-means',
                               res['meta'].shape, data=cf_arr)
            res.create_dataset('lcoe_fcr-means',
                               res['meta'].shape, data=lcoe_arr)

        h5_dsets = ('capital_cost', 'fixed_operating_cost',
                    'fixed_charge_rate', 'variable_operating_cost',
                    'system_capacity')

        base = SupplyCurveAggregation(EXCL, TM_DSET, excl_dict=EXCL_DICT,
                                      res_class_dset=None, res_class_bins=None,
                                      data_layers=DATA_LAYERS,
                                      h5_dsets=h5_dsets,
                                      gids=list(np.arange(10)),
                                      recalc_lcoe=False)
        summary_base = base.summarize(gen_temp, max_workers=1)

        sca = SupplyCurveAggregation(EXCL, TM_DSET, excl_dict=EXCL_DICT,
                                     res_class_dset=None, res_class_bins=None,
                                     data_layers=DATA_LAYERS,
                                     h5_dsets=h5_dsets,
                                     gids=list(np.arange(10)),
                                     recalc_lcoe=True)
        summary = sca.summarize(gen_temp, max_workers=1)

    assert not np.allclose(summary_base['mean_lcoe'], summary['mean_lcoe'])


def test_cli_basic_agg(runner, clear_loggers):
    with tempfile.TemporaryDirectory() as td:
        excl_fp = os.path.join(td, 'excl.h5')
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
            "tm_dset": "techmap_ri",
            "res_fpath": RES,
            'excl_dict': EXCL_DICT,
            'resolution': 32
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(gaps_cli, ['supply-curve-aggregation',
                                          '-c', config_path])
        clear_loggers()

        if result.exit_code != 0:
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)

        fn_list = os.listdir(td)
        dirname = os.path.basename(td)
        out_csv_fn = ('{}_{}.csv'
                      .format(dirname, ModuleName.SUPPLY_CURVE_AGGREGATION)
                      .replace("-", "_"))
        assert out_csv_fn in fn_list


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
