# pylint: skip-file
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
from reV.config.project_points import ProjectPoints
from reV import TESTDATADIR


RTOL = 0.0
ATOL = 0.001
PURGE_OUT = True


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


@pytest.mark.parametrize('f_rev1_out, rev2_points, year, n_workers', [
    ('project_outputs.h5', slice(0, 10), '2012', 1),
    ('project_outputs.h5', slice(0, 100, 10), '2013', 1)])
def test_wind_gen_slice(f_rev1_out, rev2_points, year, n_workers):
    """Test reV 2.0 generation for PV and benchmark against reV 1.0 results."""
    # get full file paths.
    rev1_outs = os.path.join(TESTDATADIR, 'ri_wind', 'scalar_outputs',
                             f_rev1_out)
    sam_files = TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json'
    res_file = TESTDATADIR + '/wtk/ri_100_wtk_{}.h5'.format(year)

    # run reV 2.0 generation
    pp = ProjectPoints(rev2_points, sam_files, 'wind', res_file=res_file)
    gen = Gen.run_direct('wind', rev2_points, sam_files, res_file,
                         n_workers=n_workers, sites_per_split=3, fout=None,
                         return_obj=True)
    gen_outs = list(gen.out['cf_mean'] / 1000)

    # initialize the rev1 output hander
    with wind_results(rev1_outs) as wind:
        # get reV 1.0 results
        cf_mean_list = wind.get_cf_mean(pp.sites, year)

    # benchmark the results
    result = np.allclose(gen_outs, cf_mean_list, rtol=RTOL, atol=ATOL)
    msg = 'Wind cf_means results did not match reV 1.0 results!'
    assert result is True, msg


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
