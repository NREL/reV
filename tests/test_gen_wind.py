# -*- coding: utf-8 -*-
# pylint: skip-file
"""
PyTest file for Wind generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import h5py
import pytest
import pandas as pd
import numpy as np
import tempfile

from reV.generation.generation import Gen
from reV.config.project_points import ProjectPoints
from reV import TESTDATADIR

from rex import Resource

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
    gen = Gen.reV_run('windpower', rev2_points, sam_files, res_file,
                      max_workers=max_workers, sites_per_worker=3,
                      out_fpath=None)
    gen_outs = list(gen.out['cf_mean'])

    # initialize the rev1 output hander
    with wind_results(rev1_outs) as wind:
        # get reV 1.0 results
        cf_mean_list = wind.get_cf_mean(pp.sites, year)

    # benchmark the results
    msg = 'Wind cf_means results did not match reV 1.0 results!'
    assert np.allclose(gen_outs, cf_mean_list, rtol=RTOL, atol=ATOL), msg


@pytest.mark.parametrize('gid_map',
                         [{0: 0, 1: 1, 2: 1, 3: 3, 4: 4},
                          {0: 4, 1: 3, 2: 2, 3: 1, 4: 0},
                          {0: 59, 1: 1, 2: 1, 3: 0, 4: 4},
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

    baseline = Gen.reV_run('windpower', points_base, sam_files, res_file,
                           max_workers=max_workers, sites_per_worker=3,
                           out_fpath=None, output_request=output_request)

    test = Gen.reV_run('windpower', points_test, sam_files, res_file,
                       max_workers=max_workers, sites_per_worker=3,
                       out_fpath=None, output_request=output_request,
                       gid_map=gid_map)

    if len(baseline.out['cf_mean']) == len(test.out['cf_mean']):
        assert not np.allclose(baseline.out['cf_mean'], test.out['cf_mean'])

    for gen_gid_test, res_gid in gid_map.items():
        gen_gid_base = points_base.index(res_gid)
        for key in output_request:
            if len(test.out[key].shape) == 2:
                assert np.allclose(baseline.out[key][:, gen_gid_base],
                                   test.out[key][:, gen_gid_test])
            else:
                assert np.allclose(baseline.out[key][gen_gid_base],
                                   test.out[key][gen_gid_test])

    labels = ['latitude', 'longitude']
    with Resource(res_file) as res:
        for i, (gen_gid, site_meta) in enumerate(test.meta.iterrows()):
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
    gen = Gen.reV_run('windpower', points, sam_files, res_file,
                      max_workers=max_workers, sites_per_worker=3,
                      out_fpath=None, output_request=output_request)

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
    gen = Gen.reV_run('windpower', rev2_points, sam_files, res_file,
                      max_workers=max_workers, sites_per_worker=3,
                      out_fpath=None, output_request=output_requests)
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
    gen = Gen.reV_run('windpower', points, sam_files, res_file,
                      max_workers=max_workers,
                      sites_per_worker=3, out_fpath=None)
    gen_outs = list(gen._out['cf_mean'])
    assert len(gen_outs) == 10
    assert np.mean(gen_outs) > 0.55


def test_wind_gen_site_data(points=slice(0, 5), year=2012, max_workers=1):
    """Test site specific SAM input config via site_data arg"""
    sam_files = TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json'
    res_file = TESTDATADIR + '/wtk/ri_100_wtk_{}.h5'.format(year)

    output_request = ('cf_mean', 'turb_generic_loss')

    baseline = Gen.reV_run('windpower', points, sam_files, res_file,
                           max_workers=max_workers, sites_per_worker=3,
                           out_fpath=None, output_request=output_request)

    site_data = pd.DataFrame({'gid': np.arange(2),
                              'turb_generic_loss': np.zeros(2)})
    test = Gen.reV_run('windpower', points, sam_files, res_file,
                       max_workers=max_workers, sites_per_worker=3,
                       out_fpath=None, output_request=output_request,
                       site_data=site_data)

    assert all(test.out['cf_mean'][0:2] > baseline.out['cf_mean'][0:2])
    assert np.allclose(test.out['cf_mean'][2:], baseline.out['cf_mean'][2:])
    assert np.allclose(test.out['turb_generic_loss'][0:2], np.zeros(2))
    assert np.allclose(test.out['turb_generic_loss'][2:], 16.7 * np.ones(3))


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
