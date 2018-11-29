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
from reV import __testdata__ as TESTDATA


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


def run_gen(gen, res_files, option='serial'):
    """Run reV generation with a gen instance and specified res_files."""
    if option == 'serial':
        outs = gen.execute_serial(res_files=res_files)
    elif option == 'parallel':
        outs = gen.execute_parallel(res_files=res_files)
    elif option == 'hpc':
        outs = gen.execute_hpc(res_files=res_files)
    else:
        raise ValueError('Run option not recognized: {}'
                         .format(gen.execution_control.option))

    gen_outs = to_list(outs)

    return gen_outs


@pytest.mark.parametrize('f_rev1_out, f_rev2_config, year, option', [
    ('project_outputs.h5', 'ri_rev2_pv_gen.ini', '2012', 'serial'),
    ('project_outputs.h5', 'ri_rev2_pv_gen.ini', '2013', 'serial'),
    ('project_outputs.h5', 'ri_rev2_pv_gen.ini', '2012', 'parallel'),
    ('project_outputs.h5', 'ri_rev2_pv_gen.ini', '2013', 'parallel')])
def test_pv_gen(f_rev1_out, f_rev2_config, year, option):
    """Test reV 2.0 generation for PV and benchmark against reV 1.0 results."""
    # get full file paths.
    rev1_outs = os.path.join(TESTDATA, 'ri_pv', 'scalar_outputs', f_rev1_out)
    rev2_config = os.path.join(TESTDATA, 'config_ini', f_rev2_config)

    # initialize the generation module
    gen = Gen(rev2_config)
    bad_data = 0

    # initialize the rev1 output hander
    with pv_results(rev1_outs) as pv:
        # get the res file corresponding to the year
        res_files = [r for r in gen.res_files if year in r]

        # run reV 2.0 generation
        gen_outs = run_gen(gen, res_files, option=option)

        # get reV 1.0 results
        N = len(gen_outs)
        cf_mean_list = pv.get_cf_mean(slice(0, N), year)

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
