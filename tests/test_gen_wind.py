# -*- coding: utf-8 -*-
# pylint: skip-file
"""
PyTest file for Wind generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import shutil
import h5py
import pytest
import pandas as pd
import numpy as np
import tempfile

from reV.generation.generation import Gen
from reV.config.project_points import ProjectPoints
from reV import TESTDATADIR

from rex import Resource, WindResource, Outputs

RTOL = 0
ATOL = 0.001


class wind_results:
    """Class to retrieve results from the rev 1.0 pv files"""

    def __init__(self, f):
        self._h5 = h5py.File(f, 'r')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._h5.close()

        if type is not None:
            raise

    @property
    def years(self):
        """Get a list of year strings."""
        if not hasattr(self, '_years'):
            year_index = self._h5['wind']['year_index'][...]
            self._years = [y.decode() for y in year_index]
        return self._years

    def get_cf_mean(self, site, year):
        """Get a cf mean based on site and year"""
        iy = self.years.index(year)
        out = self._h5['wind']['cf_mean'][iy, site]
        return out


def is_num(n):
    """Check if n is a number"""
    try:
        float(n)
        return True
    except Exception:
        return False


def to_list(gen_out):
    """Generation output handler that converts to the rev 1.0 format."""
    if isinstance(gen_out, list) and len(gen_out) == 1:
        out = [c['cf_mean'] for c in gen_out[0].values()]

    if isinstance(gen_out, dict):
        out = [c['cf_mean'] for c in gen_out.values()]

    return out


@pytest.mark.parametrize(('f_rev1_out', 'rev2_points', 'year', 'max_workers'),
                         [
    ('project_outputs.h5', slice(0, 10), '2012', 1),
    ('project_outputs.h5', slice(0, 100, 10), '2013', 2)])
def test_wind_gen_slice(f_rev1_out, rev2_points, year, max_workers):
    """Test reV 2.0 generation for PV and benchmark against reV 1.0 results."""
    # get full file paths.
    rev1_outs = os.path.join(TESTDATADIR, 'ri_wind', 'scalar_outputs',
                             f_rev1_out)
    sam_files = TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json'
    res_file = TESTDATADIR + '/wtk/ri_100_wtk_{}.h5'.format(year)

    # run reV 2.0 generation
    pp = ProjectPoints(rev2_points, sam_files, 'windpower', res_file=res_file)
    gen = Gen('windpower', rev2_points, sam_files, res_file,
              sites_per_worker=3)
    gen.reV_run(max_workers=max_workers)
    gen_outs = list(gen.out['cf_mean'])

    # initialize the rev1 output hander
    with wind_results(rev1_outs) as wind:
        # get reV 1.0 results
        cf_mean_list = wind.get_cf_mean(pp.sites, year)

    # benchmark the results
    msg = 'Wind cf_means results did not match reV 1.0 results!'
    assert np.allclose(gen_outs, cf_mean_list, rtol=RTOL, atol=ATOL), msg
    assert np.allclose(pp.sites, gen.meta.index.values), 'bad gen meta!'
    assert np.allclose(pp.sites, gen.meta['gid'].values), 'bad gen meta!'

    labels = ['latitude', 'longitude']
    with Resource(res_file) as res:
        for i, (gen_gid, site_meta) in enumerate(gen.meta.iterrows()):
            res_gid = site_meta['gid']
            assert gen_gid == res_gid
            test_coords = site_meta[labels].values.astype(float)
            true_coords = res.meta.loc[res_gid, labels].values.astype(float)
            assert np.allclose(test_coords, true_coords)
            assert site_meta['gid'] == res_gid


@pytest.mark.parametrize('gid_map',
                         [{0: 0, 1: 1, 2: 1, 3: 3, 4: 4},
                          {0: 4, 1: 3, 2: 2, 3: 1, 4: 0},
                          {10: 14, 11: 13, 12: 12, 13: 11, 20: 0},
                          {0: 59, 1: 1, 2: 1, 3: 0, 4: 4},
                          {0: 59, 1: 1, 2: 0, 3: 0, 4: 4},
                          {0: 1, 1: 1, 2: 0, 3: 0, 4: 0},
                          {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                          ])
def test_gid_map(gid_map):
    """Test gid mapping feature where the unique gen_gids are mapped to
    non-unique res_gids
    """
    points_base = sorted(list(set(gid_map.values())))
    points_test = sorted(list(set(gid_map.keys())))
    year = 2012
    max_workers = 1
    sam_files = TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json'
    res_file = TESTDATADIR + '/wtk/ri_100_wtk_{}.h5'.format(year)

    output_request = ('cf_mean', 'cf_profile', 'ws_mean', 'windspeed',
                      'monthly_energy')

    baseline = Gen('windpower', points_base, sam_files, res_file,
                   sites_per_worker=3, output_request=output_request)
    baseline.reV_run(max_workers=max_workers)

    map_test = Gen('windpower', points_test, sam_files, res_file,
                   sites_per_worker=3, output_request=output_request,
                   gid_map=gid_map)
    map_test.reV_run(max_workers=max_workers)

    write_gid_test = Gen('windpower', points_test, sam_files, res_file,
                         sites_per_worker=3, output_request=output_request,
                         gid_map=gid_map, write_mapped_gids=True)
    write_gid_test.reV_run(max_workers=max_workers)

    for key in output_request:
        assert np.allclose(map_test.out[key], write_gid_test.out[key])

    for map_test_gid, write_test_gid in zip(map_test.meta['gid'],
                                            write_gid_test.meta['gid']):
        assert map_test_gid == gid_map[write_test_gid]

    if len(baseline.out['cf_mean']) == len(map_test.out['cf_mean']):
        assert not np.allclose(baseline.out['cf_mean'],
                               map_test.out['cf_mean'])

    for gen_gid_test, res_gid in gid_map.items():
        gen_gid_test = points_test.index(gen_gid_test)
        gen_gid_base = points_base.index(res_gid)
        for key in output_request:
            if len(map_test.out[key].shape) == 2:
                assert np.allclose(baseline.out[key][:, gen_gid_base],
                                   map_test.out[key][:, gen_gid_test])
            else:
                assert np.allclose(baseline.out[key][gen_gid_base],
                                   map_test.out[key][gen_gid_test])

    labels = ['latitude', 'longitude']
    with Resource(res_file) as res:
        for i, (gen_gid, site_meta) in enumerate(baseline.meta.iterrows()):
            res_gid = site_meta['gid']
            test_coords = site_meta[labels].values.astype(float)
            true_coords = res.meta.loc[res_gid, labels].values.astype(float)
            assert np.allclose(test_coords, true_coords)
            assert site_meta['gid'] == res_gid

        for i, (gen_gid, site_meta) in enumerate(map_test.meta.iterrows()):
            res_gid = gid_map[gen_gid]
            test_coords = site_meta[labels].values.astype(float)
            true_coords = res.meta.loc[res_gid, labels].values.astype(float)
            assert np.allclose(test_coords, true_coords)
            assert site_meta['gid'] == res_gid


def test_wind_gen_new_outputs(points=slice(0, 10), year=2012, max_workers=1):
    """Test reV 2.0 generation for wind with new outputs."""
    # get full file paths.
    sam_files = TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json'
    res_file = TESTDATADIR + '/wtk/ri_100_wtk_{}.h5'.format(year)

    output_request = ('cf_mean', 'cf_profile', 'monthly_energy')

    # run reV 2.0 generation
    gen = Gen('windpower', points, sam_files, res_file, sites_per_worker=3,
              output_request=output_request)
    gen.reV_run(max_workers=max_workers)

    assert gen.out['cf_mean'].shape == (10, )
    assert gen.out['cf_profile'].shape == (8760, 10)
    assert gen.out['monthly_energy'].shape == (12, 10)

    assert gen._out['cf_mean'].dtype == np.float32
    assert gen._out['cf_profile'].dtype == np.uint16
    assert gen._out['monthly_energy'].dtype == np.float32


def test_windspeed_pass_through(rev2_points=slice(0, 10), year=2012,
                                max_workers=1):
    """Test a windspeed output request so that resource array is passed
    through to output dict."""

    sam_files = TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json'
    res_file = TESTDATADIR + '/wtk/ri_100_wtk_{}.h5'.format(year)

    output_requests = ('cf_mean', 'windspeed')

    # run reV 2.0 generation
    gen = Gen('windpower', rev2_points, sam_files, res_file,
              sites_per_worker=3, output_request=output_requests)
    gen.reV_run(max_workers=max_workers)
    assert 'windspeed' in gen.out
    assert gen.out['windspeed'].shape == (8760, 10)
    assert gen._out['windspeed'].max() == 2597
    assert gen._out['windspeed'].min() == 1


def test_multi_file_5min_wtk():
    """Test running reV gen from a multi-h5 directory with prefix and suffix"""
    points = slice(0, 10)
    max_workers = 1
    sam_files = TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json'
    res_file = TESTDATADIR + '/wtk/wtk_{}_*m.h5'.format(2010)
    # run reV 2.0 generation
    gen = Gen('windpower', points, sam_files, res_file, sites_per_worker=3)
    gen.reV_run(max_workers=max_workers)
    gen_outs = list(gen._out['cf_mean'])
    assert len(gen_outs) == 10
    assert np.mean(gen_outs) > 0.55


def test_wind_gen_site_data(points=slice(0, 5), year=2012, max_workers=1):
    """Test site specific SAM input config via site_data arg"""
    sam_files = TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json'
    res_file = TESTDATADIR + '/wtk/ri_100_wtk_{}.h5'.format(year)

    output_request = ('cf_mean', 'turb_generic_loss')

    baseline = Gen('windpower', points, sam_files, res_file,
                   sites_per_worker=3, output_request=output_request)
    baseline.reV_run(max_workers=max_workers)

    site_data = pd.DataFrame({'gid': np.arange(2),
                              'turb_generic_loss': np.zeros(2)})
    test = Gen('windpower', points, sam_files, res_file, sites_per_worker=3,
               output_request=output_request, site_data=site_data)
    test.reV_run(max_workers=max_workers)

    assert all(test.out['cf_mean'][0:2] > baseline.out['cf_mean'][0:2])
    assert np.allclose(test.out['cf_mean'][2:], baseline.out['cf_mean'][2:])
    assert np.allclose(test.out['turb_generic_loss'][0:2], np.zeros(2))
    assert np.allclose(test.out['turb_generic_loss'][2:], 16.7 * np.ones(3))


def test_multi_resolution_wtk():
    """Test windpower analysis with wind at 5min and t+p at hourly"""
    with tempfile.TemporaryDirectory() as td:

        points = slice(0, 2)
        max_workers = 1
        sam_files = TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json'
        source_fp_100m = TESTDATADIR + '/wtk/wtk_2010_100m.h5'
        source_fp_200m = TESTDATADIR + '/wtk/wtk_2010_200m.h5'

        fp_hr_100m = os.path.join(td, 'wtk_2010_100m_hr.h5')
        fp_hr_200m = os.path.join(td, 'wtk_2010_200m_hr.h5')
        fp_hr = os.path.join(td, 'wtk_2010_*hr.h5')
        fp_lr = os.path.join(td, 'wtk_2010_lr.h5')
        shutil.copy(source_fp_100m, fp_hr_100m)
        shutil.copy(source_fp_200m, fp_hr_200m)

        lr_dsets = ['temperature_100m', 'pressure_100m']
        with WindResource(fp_hr_100m) as hr_res:
            ti = hr_res.time_index
            meta = hr_res.meta
            lr_data = [hr_res[dset] for dset in lr_dsets]
            lr_attrs = hr_res.attrs
            lr_chunks = hr_res.chunks
            lr_dtypes = hr_res.dtypes

        t_slice = slice(None, None, 12)
        s_slice = slice(None, None, 10)
        lr_ti = ti[t_slice]
        lr_meta = meta.iloc[s_slice]
        lr_data = [d[t_slice, s_slice] for d in lr_data]
        lr_shapes = {d: (len(lr_ti), len(lr_meta)) for d in lr_dsets}

        Outputs.init_h5(fp_lr, lr_dsets, lr_shapes, lr_attrs, lr_chunks,
                        lr_dtypes, lr_meta, lr_ti)
        for name, arr in zip(lr_dsets, lr_data):
            Outputs.add_dataset(fp_lr, name, arr, lr_dtypes[name],
                                attrs=lr_attrs[name], chunks=lr_chunks[name])

        for fp in (fp_hr_100m, fp_hr_200m):
            with h5py.File(fp, 'a') as f:
                for dset in lr_dsets:
                    if dset in f:
                        del f[dset]

        # run reV 2.0 generation
        gen = Gen('windpower', points, sam_files, fp_hr,
                  low_res_resource_file=fp_lr,
                  sites_per_worker=3)
        gen.reV_run(max_workers=max_workers)
        gen_outs = list(gen._out['cf_mean'])
        assert len(gen_outs) == 2
        assert np.mean(gen_outs) > 0.55


def test_wind_bias_correct():
    """Test rev generation with bias correction."""
    sam_files = TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json'
    res_file = TESTDATADIR + '/wtk/ri_100_wtk_2012.h5'

    # run reV 2.0 generation
    points = slice(0, 10)
    pp = ProjectPoints(points, sam_files, 'windpower', res_file=res_file)
    gen_base = Gen('windpower', points, sam_files, res_file,
                   output_request=('cf_mean', 'cf_profile', 'ws_mean'),
                   sites_per_worker=3)
    gen_base.reV_run(max_workers=1)
    outs_base = np.array(list(gen_base.out['cf_mean']))

    bc_df = pd.DataFrame({'gid': np.arange(100), 'scalar': 1, 'adder': 2})
    gen = Gen('windpower', points, sam_files, res_file,
              output_request=('cf_mean', 'cf_profile', 'ws_mean'),
              sites_per_worker=3, bias_correct=bc_df)
    gen.reV_run(max_workers=1)
    outs_bc = np.array(list(gen.out['cf_mean']))
    assert all(outs_bc > outs_base)
    assert np.allclose(gen_base.out['ws_mean'] + 2, gen.out['ws_mean'])

    bc_df = pd.DataFrame({'gid': np.arange(100), 'scalar': 1, 'adder': -100})
    gen = Gen('windpower', points, sam_files, res_file,
              output_request=('cf_mean', 'cf_profile', 'ws_mean'),
              sites_per_worker=3, bias_correct=bc_df)
    gen.reV_run(max_workers=1)
    for k, arr in gen.out.items():
        assert (np.array(arr) == 0).all()


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
