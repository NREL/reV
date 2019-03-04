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
from subprocess import Popen, PIPE
import shlex

from reV.utilities.exceptions import HandlerKeyError
from reV.SAM.SAM import SAM
from reV.config.project_points import ProjectPoints
from reV.config.analysis_configs import GenConfig
from reV import TESTDATADIR
from reV.handlers.outputs import Outputs


RTOL = 0.0
ATOL = 0.04
PURGE_OUT = True


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


def test_clearsky():
    res_file = os.path.join(TESTDATADIR, 'nsrdb/', 'ri_100_nsrdb_2012.h5')
    sam_config_dict = {0: os.path.join(TESTDATADIR, 'SAM/'
                                       'naris_pv_1axis_inv13_cs.json')}
    pp = ProjectPoints(slice(0, 10), sam_config_dict, 'pv', res_file=res_file)
    try:
        # Get the SAM resource object
        SAM.get_sam_res(res_file, pp, pp.tech)
        assert False
    except HandlerKeyError as e:
        # Should look for clearsky_dni and not find it in RI data
        assert True


@pytest.mark.parametrize('tech', ['pv', 'wind'])
def test_gen_from_config(tech):
    """Gen PV CF profiles with write to disk and compare against rev1."""

    job_name = 'config_test'

    if tech == 'pv':
        fconfig = 'local_pv.json'
    elif tech == 'wind':
        fconfig = 'local_wind.json'

    config = os.path.join(TESTDATADIR, 'config/{}'.format(fconfig))\
        .replace('\\', '/')

    cmd = 'python -m reV.cli -n "{}" -c {} generation'.format(job_name, config)
    cmd = shlex.split(cmd)

    # use subprocess to submit command and get piped o/e
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    stderr = stderr.decode('ascii').rstrip()
    stdout = stdout.decode('ascii').rstrip()

    config_obj = GenConfig(config)

    if stderr:
        ferr = os.path.join(config_obj.dirout, 'test_config.e')
        with open(ferr, 'w') as f:
            f.write(stderr)

    # get reV 2.0 generation profiles from disk
    flist = os.listdir(config_obj.dirout)
    for fname in flist:
        if job_name in fname and fname.endswith('.h5'):
            with Outputs(os.path.join(config_obj.dirout, fname), 'r') as cf:
                rev2_profiles = cf['cf_profile']
            break

    # get reV 1.0 generation profiles
    rev1_profiles = get_r1_profiles(year=config_obj.years[0], tech=tech)
    rev1_profiles = rev1_profiles[:, config_obj.points_control.sites]

    result = np.allclose(rev1_profiles, rev2_profiles, rtol=RTOL, atol=ATOL)

    if result and PURGE_OUT:
        # remove output files if test passes.
        flist = os.listdir(config_obj.dirout)
        for fname in flist:
            os.remove(os.path.join(config_obj.dirout, fname))

    assert result is True


def get_r1_profiles(year=2012, tech='pv'):
    """Get the first 100 reV 1.0 ri generation profiles."""

    if tech == 'pv':
        rev1 = os.path.join(TESTDATADIR, 'ri_pv', 'profile_outputs',
                            'pv_{}_0.h5'.format(year))
    elif tech == 'wind':
        rev1 = os.path.join(TESTDATADIR, 'ri_wind', 'profile_outputs',
                            'wind_{}_0.h5'.format(year))

    with Outputs(rev1) as cf:
        data = cf['cf_profile'][...] / 10000
    return data


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
