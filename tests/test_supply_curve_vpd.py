# -*- coding: utf-8 -*-
"""
Test Variable Power Density
@author: gbuster
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from reV import TESTDATADIR
from reV.supply_curve.sc_aggregation import SupplyCurveAggregation
from reV.utilities import SupplyCurveField
from reV.utilities.exceptions import FileInputError

EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
GEN = os.path.join(TESTDATADIR, 'gen_out/ri_my_pv_gen.h5')
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
    vpd = pd.read_csv(FVPD)
    vpd = vpd.rename(columns=SupplyCurveField.map_from_legacy())
    vpd = vpd.set_index(vpd.columns[0])

    with tempfile.TemporaryDirectory() as td:
        tmp_path = os.path.join(td, "vpd.csv")
        vpd.to_csv(tmp_path)
        sca = SupplyCurveAggregation(EXCL, TM_DSET, excl_dict=EXCL_DICT,
                                     res_class_dset=RES_CLASS_DSET,
                                     res_class_bins=RES_CLASS_BINS,
                                     data_layers=DATA_LAYERS,
                                     power_density=tmp_path)
        summary = sca.summarize(GEN, max_workers=1)

    for i in summary.index:
        capacity = summary.loc[i, SupplyCurveField.CAPACITY_AC_MW]
        area = summary.loc[i, SupplyCurveField.AREA_SQ_KM]
        res_gids = np.array(summary.loc[i, SupplyCurveField.RES_GIDS])
        gid_counts = np.array(summary.loc[i, SupplyCurveField.GID_COUNTS])
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
    excl_dict_2 = {'ri_padus': {'exclude_values': [1],
                                'weight': 0.5}}

    vpd = pd.read_csv(FVPD)
    vpd = vpd.rename(columns=SupplyCurveField.map_from_legacy())
    vpd = vpd.set_index(vpd.columns[0])

    with tempfile.TemporaryDirectory() as td:
        tmp_path = os.path.join(td, "vpd.csv")
        vpd.to_csv(tmp_path)

        sca_1 = SupplyCurveAggregation(EXCL, TM_DSET, excl_dict=excl_dict_1,
                                       res_class_dset=RES_CLASS_DSET,
                                       res_class_bins=RES_CLASS_BINS,
                                       data_layers=DATA_LAYERS,
                                       power_density=tmp_path,
                                       gids=gids_subset)
        summary_1 = sca_1.summarize(GEN, max_workers=1)

        sca_2 = SupplyCurveAggregation(EXCL, TM_DSET, excl_dict=excl_dict_2,
                                       res_class_dset=RES_CLASS_DSET,
                                       res_class_bins=RES_CLASS_BINS,
                                       data_layers=DATA_LAYERS,
                                       power_density=tmp_path,
                                       gids=gids_subset)
        summary_2 = sca_2.summarize(GEN, max_workers=1)

    for i in summary_1.index:
        cap_full = summary_1.loc[i, SupplyCurveField.CAPACITY_AC_MW]
        cap_half = summary_2.loc[i, SupplyCurveField.CAPACITY_AC_MW]

        msg = ('Variable power density for fractional exclusions failed! '
               'Index {} has cap full {} and cap half {}'
               .format(i, cap_full, cap_half))
        assert (cap_full / cap_half) == 2, msg


def test_vpd_incomplete():
    """Test an incomplete VPD input and make sure an exception is raised"""
    sca = SupplyCurveAggregation(EXCL, TM_DSET, excl_dict=EXCL_DICT,
                                 res_class_dset=RES_CLASS_DSET,
                                 res_class_bins=RES_CLASS_BINS,
                                 data_layers=DATA_LAYERS,
                                 power_density=FVPDI)
    try:
        sca.summarize(GEN, max_workers=1)
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
