# -*- coding: utf-8 -*-
# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import pytest
import numpy as np
from subprocess import Popen, PIPE
import shlex

from reV.config.sam_analysis_configs import GenConfig
from reV import TESTDATADIR
from reV.handlers.outputs import Outputs


RTOL = 0.0
ATOL = 0.04
PURGE_OUT = True


@pytest.mark.parametrize('tech', ['pv', 'wind'])
def test_gen_from_config(tech):
    """Gen PV CF profiles with write to disk and compare against rev1."""

    job_name = 'config_test_{}'.format(tech)

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

                msg = 'cf_profile not written to disk'
                assert 'cf_profile' in cf.datasets, msg
                rev2_profiles = cf['cf_profile']

                msg = 'monthly_energy not written to disk'
                assert 'monthly_energy' in cf.datasets, msg
                monthly = cf['monthly_energy']
                assert monthly.shape == (12, 10)

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

    msg = ('reV generation from config input failed for "{}" module!'
           .format(tech))
    assert result is True, msg


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
