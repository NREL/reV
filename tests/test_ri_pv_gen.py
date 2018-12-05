"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import h5py
import pytest

import utilities as ut
from reV.generation.generation import Gen
from reV.config.config import ProjectPoints
from reV import __testdatadir__ as TESTDATA


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

    # initialize the generation module
    bad_data = 0
    # run reV 2.0 generation
    pp = ProjectPoints(rev2_points, sam_files, 'pv', res_file=res_file)
    gen_outs = Gen.direct('pv', rev2_points, sam_files, res_file,
                          n_workers=n_workers, sites_per_split=3)

    gen_outs = to_list(gen_outs)

    # initialize the rev1 output hander
    with pv_results(rev1_outs) as pv:
        # get reV 1.0 results
        cf_mean_list = pv.get_cf_mean(pp.sites, year)

        # benchmark the results and count the number of bad results
        count = ut.compare_arrays(gen_outs, cf_mean_list)
        bad_data += count

    if bad_data == 0:
        return True


def test_pv_gen_csv1(f_rev1_out='project_outputs.h5',
                     rev2_points=TESTDATA + '/project_points/ri.csv',
                     res_file=TESTDATA + '/nsrdb/ri_100_nsrdb_2012.h5'):
    """Test project points csv input with dictionary-based sam files."""
    bad_data = 0
    rev1_outs = os.path.join(TESTDATA, 'ri_pv', 'scalar_outputs', f_rev1_out)
    sam_files = {'sam_param_0': TESTDATA + '/SAM/naris_pv_1axis_inv13.json',
                 'sam_param_1': TESTDATA + '/SAM/naris_pv_1axis_inv13.json'}
    pp = ProjectPoints(rev2_points, sam_files, 'pv')

    # run reV 2.0 generation
    gen_outs = Gen.direct('pv', rev2_points, sam_files, res_file)
    gen_outs = to_list(gen_outs)

    # initialize the rev1 output hander
    with pv_results(rev1_outs) as pv:
        # get reV 1.0 results
        cf_mean_list = pv.get_cf_mean(pp.sites, '2012')

        # benchmark the results and count the number of bad results
        count = ut.compare_arrays(gen_outs, cf_mean_list)
        bad_data += count

    if bad_data == 0:
        return True


def test_pv_gen_csv2(f_rev1_out='project_outputs.h5',
                     rev2_points=TESTDATA + '/project_points/ri.csv',
                     res_file=TESTDATA + '/nsrdb/ri_100_nsrdb_2012.h5'):
    """Test project points csv input with list-based sam files."""
    bad_data = 0
    rev1_outs = os.path.join(TESTDATA, 'ri_pv', 'scalar_outputs', f_rev1_out)
    sam_files = [TESTDATA + '/SAM/naris_pv_1axis_inv13.json',
                 TESTDATA + '/SAM/naris_pv_1axis_inv13.json']
    pp = ProjectPoints(rev2_points, sam_files, 'pv')
    gen_outs = Gen.direct('pv', rev2_points, sam_files, res_file)
    gen_outs = to_list(gen_outs)

    # initialize the rev1 output hander
    with pv_results(rev1_outs) as pv:
        # get reV 1.0 results
        cf_mean_list = pv.get_cf_mean(pp.sites, '2012')

        # benchmark the results and count the number of bad results
        count = ut.compare_arrays(gen_outs, cf_mean_list)
        bad_data += count

    if bad_data == 0:
        return True


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


if __name__ == '__main__':
    execute_pytest()
