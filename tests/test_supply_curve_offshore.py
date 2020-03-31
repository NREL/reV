# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:37:05 2019

@author: gbuster
"""
import json
import numpy as np
import pytest
import os
from reV.handlers.outputs import Outputs
from reV.supply_curve.sc_aggregation import SupplyCurveAggregation
from reV import TESTDATADIR


EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
GEN = os.path.join(TESTDATADIR, 'offshore/ri_offshore_archive.h5')
TM_DSET = 'techmap_wtk'
RES_CLASS_DSET = 'ws_mean'
CF_DSET = 'cf_mean'
LCOE_DSET = 'lcoe_fcr'
RES_CLASS_BINS = [0, 6, 7, 8, 9, 100]
DATA_LAYERS = {'pct_slope': {'dset': 'ri_srtm_slope',
                             'method': 'mean'},
               'reeds_region': {'dset': 'ri_reeds_regions',
                                'method': 'mode'},
               'padus': {'dset': 'ri_padus',
                         'method': 'mode'}}

EXCL_DICT = {'ri_srtm_slope': {'inclusion_range': (None, 5),
                               'exclude_nodata': True},
             'ri_padus': {'exclude_values': [1],
                          'exclude_nodata': True},
             'ri_reeds_regions': {'inclusion_range': (None, 400),
                                  'exclude_nodata': True}}

RTOL = 0.001


def test_sc_agg_offshore():
    """Test the SC offshore aggregation and check offshore SC points against
    known offshore gen points."""

    s = SupplyCurveAggregation.summary(EXCL, GEN, TM_DSET,
                                       excl_dict=EXCL_DICT,
                                       res_class_dset=RES_CLASS_DSET,
                                       res_class_bins=RES_CLASS_BINS,
                                       cf_dset=CF_DSET, lcoe_dset=LCOE_DSET,
                                       data_layers=DATA_LAYERS,
                                       max_workers=1)

    assert 'sub_type' in s, 'sub_type was not passed through to offshore agg!'

    with Outputs(GEN, mode='r') as out:
        meta = out.meta

    offshore_mask = (meta.offshore == 1)
    offshore_gids = meta.loc[offshore_mask, 'gid'].values.tolist()

    for sc_gid in s.index:
        if s.at[sc_gid, 'offshore']:
            assert int(s.at[sc_gid, 'sc_point_gid']) > 1e7
            assert int(s.at[sc_gid, 'sc_point_gid']) in offshore_gids
            assert all(np.array(json.loads(s.at[sc_gid, 'res_gids'])) < 3e6)
            assert s.at[sc_gid, 'elevation'] == 0.0
            assert s.at[sc_gid, 'capacity'] == 600
            assert np.isnan(s.at[sc_gid, 'pct_slope'])
        else:
            for res_gid in s.at[sc_gid, 'res_gids']:
                assert res_gid not in offshore_gids
    for gid in offshore_gids:
        assert gid in s['sc_point_gid'].values


def plot_sc_offshore(plot_var='mean_lcoe'):
    """Plot the supply curve map colored by plot_var."""

    s = SupplyCurveAggregation.summary(EXCL, GEN, TM_DSET,
                                       excl_dict=EXCL_DICT,
                                       res_class_dset=RES_CLASS_DSET,
                                       res_class_bins=RES_CLASS_BINS,
                                       cf_dset=CF_DSET, lcoe_dset=LCOE_DSET,
                                       data_layers=DATA_LAYERS,
                                       max_workers=1)
    import matplotlib.pyplot as plt

    plt.scatter(s['longitude'], s['latitude'], c=s[plot_var], marker='s')
    plt.axis('equal')
    plt.colorbar(label=plot_var)


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
