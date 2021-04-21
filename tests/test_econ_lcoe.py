# -*- coding: utf-8 -*-
# pylint: skip-file
"""
PyTest file for lcoe econ run in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""
import h5py
import numpy as np
import os
import pandas as pd
import shutil
from pandas.testing import assert_frame_equal
import pytest
import tempfile

from reV.econ.econ import Econ
from reV import TESTDATADIR
from reV.handlers.outputs import Outputs

RTOL = 0.01
ATOL = 0.001


@pytest.mark.parametrize(('year', 'max_workers', 'spw'),
                         [('2012', 1, 25),
                          ('2012', 2, 25),
                          ('2012', os.cpu_count() * 2, 25),
                          ('2012', 1, 1),
                          ('2013', 1, 25),
                          ('2013', 2, 25),
                          ('2013', os.cpu_count() * 2, 25),
                          ('2013', 1, 1)])
def test_lcoe(year, max_workers, spw):
    """Gen PV CF profiles with write to disk and compare against rev1."""
    cf_file = os.path.join(TESTDATADIR,
                           'gen_out/gen_ri_pv_{}_x000.h5'.format(year))
    sam_files = os.path.join(TESTDATADIR,
                             'SAM/i_lcoe_naris_pv_1axis_inv13.json')
    r1f = os.path.join(TESTDATADIR,
                       'ri_pv/scalar_outputs/project_outputs.h5')
    points = slice(0, 100)
    obj = Econ.reV_run(points, sam_files, cf_file,
                       year=year, output_request='lcoe_fcr',
                       max_workers=max_workers, sites_per_worker=spw,
                       points_range=None, out_fpath=None)
    lcoe = list(obj.out['lcoe_fcr'])

    with h5py.File(r1f, mode='r') as f:
        year_rows = {'2012': 0, '2013': 1}
        r1_lcoe = f['pv']['lcoefcr'][year_rows[str(year)], 0:100] * 1000

    result = np.allclose(lcoe, r1_lcoe, rtol=RTOL, atol=ATOL)

    assert result


@pytest.mark.parametrize('year', ('2012', '2013'))
def test_fout(year):
    """Gen PV CF profiles with write to disk and compare against rev1."""
    with tempfile.TemporaryDirectory() as dirout:
        cf_file = os.path.join(TESTDATADIR,
                               'gen_out/gen_ri_pv_{}_x000.h5'.format(year))
        sam_files = os.path.join(TESTDATADIR,
                                 'SAM/i_lcoe_naris_pv_1axis_inv13.json')
        r1f = os.path.join(TESTDATADIR,
                           'ri_pv/scalar_outputs/project_outputs.h5')
        fout = 'lcoe_out_{}.h5'.format(year)
        fpath = os.path.join(dirout, fout)
        points = slice(0, 100)
        Econ.reV_run(points, sam_files, cf_file,
                     year=year, output_request='lcoe_fcr',
                     max_workers=1, sites_per_worker=25,
                     points_range=None, out_fpath=fpath)

        with Outputs(fpath) as f:
            lcoe = f['lcoe_fcr']

        with h5py.File(r1f, mode='r') as f:
            year_rows = {'2012': 0, '2013': 1}
            r1_lcoe = f['pv']['lcoefcr'][year_rows[str(year)], 0:100] * 1000

        result = np.allclose(lcoe, r1_lcoe, rtol=RTOL, atol=ATOL)

        assert result


@pytest.mark.parametrize('year', ('2012', '2013'))
def test_append_data(year):
    """Gen PV CF profiles with write to disk and compare against rev1."""
    with tempfile.TemporaryDirectory() as dirout:
        original_file = os.path.join(TESTDATADIR,
                                     'gen_out/gen_ri_pv_{}_x000.h5'
                                     .format(year))
        cf_file = os.path.join(dirout,
                               'copy_gen_ri_pv_{}_x000.h5'.format(year))
        shutil.copy(original_file, cf_file)
        sam_files = os.path.join(TESTDATADIR,
                                 'SAM/i_lcoe_naris_pv_1axis_inv13.json')
        r1f = os.path.join(TESTDATADIR,
                           'ri_pv/scalar_outputs/project_outputs.h5')
        points = slice(0, 100)
        Econ.reV_run(points, sam_files, cf_file,
                     year=year, output_request='lcoe_fcr',
                     max_workers=1, sites_per_worker=25,
                     points_range=None, append=True)

        with Outputs(cf_file) as f:
            new_dsets = f.dsets
            cf_profile = f['cf_profile']
            lcoe = f['lcoe_fcr']
            meta = f.meta
            ti = f.time_index

        with Outputs(original_file) as f:
            og_dsets = f.dsets
            og_profiles = f['cf_profile']
            og_meta = f.meta
            og_ti = f.time_index

        with h5py.File(r1f, mode='r') as f:
            year_rows = {'2012': 0, '2013': 1}
            r1_lcoe = f['pv']['lcoefcr'][year_rows[str(year)], 0:100] * 1000

        assert np.allclose(lcoe, r1_lcoe, rtol=RTOL, atol=ATOL)
        assert np.allclose(cf_profile, og_profiles)
        assert_frame_equal(meta, og_meta)
        assert all(ti == og_ti)
        assert all([d in new_dsets for d in og_dsets])


@pytest.mark.parametrize('node', (0, 1))
def test_append_multi_node(node):
    """Test econ multi node with append flag ON using a
    real reV run from 8/17/2020"""
    with tempfile.TemporaryDirectory() as dirout:
        original_file = os.path.join(
            TESTDATADIR, 'gen_out/pv_atb20_gen_1998_node0{}.h5'.format(node))
        cf_file = os.path.join(
            dirout,
            'gen_out/copy_pv_atb20_gen_1998_node0{}.h5'.format(node))
        shutil.copy(original_file, cf_file)
        sam_files = {'default': os.path.join(
            TESTDATADIR, 'SAM/pv_tracking_atb2020.json')}
        year = 1998
        points = os.path.join(
            TESTDATADIR,
            'config/nsrdb_projpoints_atb2020_capcostmults_subset.csv')
        site_data = os.path.join(
            TESTDATADIR,
            'config/nsrdb_sitedata_atb2020_capcostmults_subset.csv')
        econ = Econ.reV_run(points, sam_files, cf_file, year=year,
                            output_request=('lcoe_fcr', 'capital_cost'),
                            max_workers=1, sites_per_worker=25,
                            points_range=None, append=True,
                            site_data=site_data)

        with Outputs(original_file) as out:
            data_baseline = out['lcoe_fcr']
        with Outputs(cf_file) as out:
            meta = out.meta
            data_test = out['lcoe_fcr']
            test_cap_cost = out['capital_cost']

        assert np.allclose(data_baseline, data_test)

        site_data = pd.read_csv(site_data)
        sd_cap_cost = \
            site_data.loc[site_data.gid.isin(meta.gid), 'capital_cost']
        assert np.allclose(test_cap_cost, sd_cap_cost)
        assert np.allclose(econ.out['capital_cost'], sd_cap_cost)


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
