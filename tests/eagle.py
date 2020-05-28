# -*- coding: utf-8 -*-
# pylint: skip-file
"""
Test file for qsub on Eagle. Will not run with pytest
(in case using on local machine).

RUN THIS FILE AS A SCRIPT

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import h5py
import pytest
import numpy as np
import time

from reV import TESTDATADIR
from reV.handlers.outputs import Outputs
from rex.utilities.execution import SLURM
from rex.utilities.loggers import init_logger
from reV.generation.cli_gen import get_node_cmd


RTOL = 0.0
ATOL = 0.04


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


@pytest.mark.parametrize('year', [('2012')])
def test_eagle(year):
    """Gen PV CF profiles with write to disk and compare against rev1."""
    res_file = TESTDATADIR + '/nsrdb/ri_100_nsrdb_{}.h5'.format(year)
    sam_files = TESTDATADIR + '/SAM/naris_pv_1axis_inv13.json'
    rev2_out_dir = os.path.join(TESTDATADIR, 'ri_pv_reV2')
    rev2_out = 'gen_ri_pv_smart_{}.h5'.format(year)

    if not os.path.exists(rev2_out_dir):
        os.mkdir(rev2_out_dir)

    name = 'etest'
    points = slice(0, 100)
    verbose = True

    log_level = 'DEBUG'
    log_file = os.path.join(rev2_out_dir, '{}.log'.format(name))
    modules = [__name__, 'reV.utilities', 'reV.generation']
    for mod in modules:
        init_logger(mod, log_level=log_level, log_file=log_file)

    cmd = get_node_cmd(name=name, tech='pvwattsv5',
                       points=points, points_range=None,
                       sam_files=sam_files, res_file=res_file,
                       sites_per_worker=None, max_workers=None,
                       fout=rev2_out, dirout=rev2_out_dir, logdir=rev2_out_dir,
                       output_request=('cf_profile', 'cf_mean'),
                       verbose=verbose)

    # create and submit the SLURM job
    slurm = SLURM(cmd, alloc='rev', memory=96, walltime=0.1,
                  name=name, stdout_path=rev2_out_dir)

    while True:
        status = slurm.check_status(name, var='name')
        if status == 'CG':
            break
        else:
            time.sleep(5)

    # get reV 2.0 generation profiles from disk
    flist = os.listdir(rev2_out_dir)
    for fname in flist:
        if '.h5' in fname:
            if rev2_out.strip('.h5') in fname:
                full_f = os.path.join(rev2_out_dir, fname)
                with Outputs(full_f, 'r') as cf:
                    rev2_profiles = cf['cf_profile']
                break

    # get reV 1.0 generation profiles
    rev1_profiles = get_r1_profiles(year=year)
    rev1_profiles = rev1_profiles[:, points]

    result = np.allclose(rev1_profiles, rev2_profiles, rtol=RTOL, atol=ATOL)
    if result:
        # remove output files if test passes.
        flist = os.listdir(rev2_out_dir)
        for fname in flist:
            os.remove(os.path.join(rev2_out_dir, fname))

    assert result is True


def get_r1_profiles(year=2012):
    """Get the first 100 reV 1.0 ri pv generation profiles."""
    rev1 = os.path.join(TESTDATADIR, 'ri_pv', 'profile_outputs',
                        'pv_{}_0.h5'.format(year))
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
