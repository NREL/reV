# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:37:05 2019

@author: gbuster
"""
import pandas as pd
import numpy as np
import pytest
import os
from reV.supply_curve.aggregation import Aggregation
from reV import TESTDATADIR
from reV.utilities.exceptions import FileInputError


EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
GEN = os.path.join(TESTDATADIR, 'gen_out/ri_my_pv_gen.h5')
AGG_BASELINE = os.path.join(TESTDATADIR, 'sc_out/baseline_agg_summary.csv')
FVPD = os.path.join(TESTDATADIR, 'variable_power_density/vpd.csv')
FVPDI = os.path.join(TESTDATADIR, 'variable_power_density/vpd_incomplete.csv')
TM_DSET = 'techmap_nsrdb'
RES_CLASS_DSET = 'ghi_mean-means'
RES_CLASS_BINS = [0, 4, 100]
DATA_LAYERS = {'pct_slope': {'dset': 'ri_srtm_slope',
                             'method': 'mean'},
               'reeds_region': {'dset': 'ri_reeds_regions',
                                'method': 'mode'}}

EXCL_DICT = {'ri_srtm_slope': {'inclusion_range': (None, 5)},
             'ri_padus': {'exclude_values': [1]}}

RTOL = 0.001


def test_vpd():
    """Test variable power density"""

    s = Aggregation.summary(EXCL, GEN, TM_DSET, EXCL_DICT,
                            res_class_dset=RES_CLASS_DSET,
                            res_class_bins=RES_CLASS_BINS,
                            data_layers=DATA_LAYERS,
                            max_workers=1, power_density=FVPD)

    vpd = pd.read_csv(FVPD, index_col=0)
    for i in s.index:
        capacity = s.loc[i, 'capacity']
        area = s.loc[i, 'area_sq_km']
        res_gids = np.array(s.loc[i, 'res_gids'])
        gid_counts = np.array(s.loc[i, 'gid_counts'])
        vpd_per_gid = vpd.loc[res_gids, 'power_density'].values
        truth = area * (vpd_per_gid * gid_counts).sum() / gid_counts.sum()

        diff = 100 * (capacity - truth) / truth

        msg = ('Variable power density failed! Index {} has cap {} and '
               'truth {}'.format(i, capacity, truth))
        assert diff < 1, msg


def test_vpd_fractional_excl():
    """Test variable power density with fractional exclusions"""

    gids_subset = list(range(0, 20))
    excl_dict_1 = {'ri_padus': {'exclude_values': [1]}}
    s1 = Aggregation.summary(EXCL, GEN, TM_DSET, excl_dict_1,
                             res_class_dset=RES_CLASS_DSET,
                             res_class_bins=RES_CLASS_BINS,
                             data_layers=DATA_LAYERS,
                             power_density=FVPD,
                             max_workers=1, gids=gids_subset)

    excl_dict_2 = {'ri_padus': {'exclude_values': [1],
                                'weight': 0.5}}
    s2 = Aggregation.summary(EXCL, GEN, TM_DSET, excl_dict_2,
                             res_class_dset=RES_CLASS_DSET,
                             res_class_bins=RES_CLASS_BINS,
                             data_layers=DATA_LAYERS,
                             power_density=FVPD,
                             max_workers=1, gids=gids_subset)

    for i in s1.index:
        cap_full = s1.loc[i, 'capacity']
        cap_half = s2.loc[i, 'capacity']

        msg = ('Variable power density for fractional exclusions failed! '
               'Index {} has cap full {} and cap half {}'
               .format(i, cap_full, cap_half))
        assert (cap_full / cap_half) == 2, msg


def test_vpd_incomplete():
    """Test an incomplete VPD input and make sure an exception is raised"""
    try:
        Aggregation.summary(EXCL, GEN, TM_DSET, EXCL_DICT,
                            res_class_dset=RES_CLASS_DSET,
                            res_class_bins=RES_CLASS_BINS,
                            data_layers=DATA_LAYERS,
                            max_workers=1, power_density=FVPDI)
    except FileInputError as e:
        if '1314958' in str(e):
            pass
    else:
        raise Exception('Test with incomplete VPD input did not throw error!')


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
