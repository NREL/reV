"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import h5py
import pytest
import numpy as np

from reV.generation.generation import Gen
from reV.config.config import ProjectPoints
from reV import __testdatadir__ as TESTDATA
from reV.handlers.capacity_factor import CapacityFactor


TOL = 0.001


class pv_results:
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
            year_index = self._h5['pv']['year_index'][...]
            self._years = [y.decode() for y in year_index]
        return self._years

    def get_cf_mean(self, site, year):
        """Get a cf mean based on site and year"""
        iy = self.years.index(year)
        out = self._h5['pv']['cf_mean'][iy, site]
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


@pytest.mark.parametrize('f_rev1_out, rev2_points, year, n_workers', [
    ('project_outputs.h5', slice(0, 10), '2012', 1),
    ('project_outputs.h5', slice(0, None, 10), '2013', 1),
    ('project_outputs.h5', slice(3, 25, 2), '2012', 2),
    ('project_outputs.h5', slice(40, None, 10), '2013', 2)])
def test_pv_gen_slice(f_rev1_out, rev2_points, year, n_workers):
    """Test reV 2.0 generation for PV and benchmark against reV 1.0 results."""
    # get full file paths.
    rev1_outs = os.path.join(TESTDATA, 'ri_pv', 'scalar_outputs', f_rev1_out)
    sam_files = TESTDATA + '/SAM/naris_pv_1axis_inv13.json'
    res_file = TESTDATA + '/nsrdb/ri_100_nsrdb_{}.h5'.format(year)

    # run reV 2.0 generation
    pp = ProjectPoints(rev2_points, sam_files, 'pv', res_file=res_file)
    gen = Gen.direct('pv', rev2_points, sam_files, res_file,
                     n_workers=n_workers, sites_per_split=3, fout=None,
                     return_obj=True)

    gen_outs = to_list(gen.out)

    # initialize the rev1 output hander
    with pv_results(rev1_outs) as pv:
        # get reV 1.0 results
        cf_mean_list = pv.get_cf_mean(pp.sites, year)

    # benchmark the results
    result = np.allclose(gen_outs, cf_mean_list, rtol=TOL, atol=TOL)
    return result


def test_pv_gen_csv1(f_rev1_out='project_outputs.h5',
                     rev2_points=TESTDATA + '/project_points/ri.csv',
                     res_file=TESTDATA + '/nsrdb/ri_100_nsrdb_2012.h5'):
    """Test project points csv input with dictionary-based sam files."""
    rev1_outs = os.path.join(TESTDATA, 'ri_pv', 'scalar_outputs', f_rev1_out)
    sam_files = {'sam_param_0': TESTDATA + '/SAM/naris_pv_1axis_inv13.json',
                 'sam_param_1': TESTDATA + '/SAM/naris_pv_1axis_inv13.json'}
    pp = ProjectPoints(rev2_points, sam_files, 'pv')

    # run reV 2.0 generation
    gen = Gen.direct('pv', rev2_points, sam_files, res_file, fout=None,
                     return_obj=True)
    gen_outs = to_list(gen.out)

    # initialize the rev1 output hander
    with pv_results(rev1_outs) as pv:
        # get reV 1.0 results
        cf_mean_list = pv.get_cf_mean(pp.sites, '2012')

    # benchmark the results
    result = np.allclose(gen_outs, cf_mean_list, rtol=TOL, atol=TOL)
    return result


def test_pv_gen_csv2(f_rev1_out='project_outputs.h5',
                     rev2_points=TESTDATA + '/project_points/ri.csv',
                     res_file=TESTDATA + '/nsrdb/ri_100_nsrdb_2012.h5'):
    """Test project points csv input with list-based sam files."""
    rev1_outs = os.path.join(TESTDATA, 'ri_pv', 'scalar_outputs', f_rev1_out)
    sam_files = [TESTDATA + '/SAM/naris_pv_1axis_inv13.json',
                 TESTDATA + '/SAM/naris_pv_1axis_inv13.json']
    pp = ProjectPoints(rev2_points, sam_files, 'pv')
    gen = Gen.direct('pv', rev2_points, sam_files, res_file, fout=None,
                     return_obj=True)
    gen_outs = to_list(gen.out)

    # initialize the rev1 output hander
    with pv_results(rev1_outs) as pv:
        # get reV 1.0 results
        cf_mean_list = pv.get_cf_mean(pp.sites, '2012')

    # benchmark the results
    result = np.allclose(gen_outs, cf_mean_list, rtol=TOL, atol=TOL)
    return result


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

    pytest.main(['-q', '--show-capture={}'.format(capture),
                 'test_ri_pv_gen.py', flags])


@pytest.mark.parametrize('year', [('2012'), ('2013')])
def test_pv_gen_profiles(year):
    """Gen PV CF profiles with write to disk and compare against rev1."""
    res_file = TESTDATA + '/nsrdb/ri_100_nsrdb_{}.h5'.format(year)
    sam_files = TESTDATA + '/SAM/naris_pv_1axis_inv13.json'
    rev2_out_dir = os.path.join(TESTDATA, 'ri_pv_reV2')
    rev2_out = 'gen_ri_pv_{}.h5'.format(year)

    # run reV 2.0 generation and write to disk
    Gen.direct('pv', slice(0, 100), sam_files, res_file, fout=rev2_out,
               n_workers=2, sites_per_split=50, dirout=rev2_out_dir,
               return_obj=False)

    # get reV 2.0 generation profiles from disk
    with CapacityFactor(os.path.join(rev2_out_dir, rev2_out), 'r') as cf:
        rev2_profiles = cf['cf_profiles']

    # get reV 1.0 generation profiles
    rev1_profiles = get_r1_profiles(year=year)

    result = np.allclose(rev1_profiles, rev2_profiles, rtol=TOL, atol=TOL)
    return result


def get_r1_profiles(year=2012):
    """Get the first 100 reV 1.0 ri pv generation profiles."""
    rev1 = os.path.join(TESTDATA, 'ri_pv', 'profile_outputs',
                        'pv_{}_0.h5'.format(year))
    with CapacityFactor(rev1) as cf:
        data = cf['cf_profile'][...] / 10000
    return data


if __name__ == '__main__':
    execute_pytest()
