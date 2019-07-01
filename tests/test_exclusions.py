# -*- coding: utf-8 -*-
# pylint: skip-file
"""Exclusions unit test module
"""
import os
import pytest
import rasterio
import numpy as np
import shlex
from subprocess import Popen, PIPE

from reV import TESTDATADIR
from reV.config.analysis_configs import ExclConfig
from reV.exclusions.exclusions import Exclusions


PURGE = True


def test_exclusions_output():
    """Validate exclusions output
    """
    f_path = os.path.join(TESTDATADIR, 'ri_exclusions')

    with rasterio.open(os.path.join(f_path, "exclusions.tif"), 'r') as file:
        valid_exclusions_data = file.read(1)

    layer_configs = [{"fpath": os.path.join(f_path, "ri_srtm_slope.tif"),
                      "max_thresh": 5},
                     {"fpath": os.path.join(f_path, "ri_padus.tif"),
                      "classes_exclude": [1]}]

    exclusions = Exclusions(layer_configs, contiguous_filter='queen')
    exclusions.build_from_config()

    assert np.array_equal(exclusions.data, valid_exclusions_data)


def test_excl_from_cli():
    """Validate exclusions run from CLI call.
    """
    f_config = os.path.join(TESTDATADIR, 'config', 'local_exclusions.json')

    cmd = ('python -m reV.cli -n "{}" -c "{}" exclusions'
           .format('test_excl', f_config))
    cmd = shlex.split(cmd)

    # use subprocess to submit command and get piped o/e
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    stderr = stderr.decode('ascii').rstrip()
    stdout = stdout.decode('ascii').rstrip()

    config_obj = ExclConfig(f_config)

    if stderr:
        print(stderr)
        ferr = os.path.join(config_obj.dirout, 'test_config.e')
        with open(ferr, 'w') as f:
            f.write(stderr)
        assert False, 'Check stderr file: "{}"'.format(ferr)

    f_test_out = os.path.join(config_obj.dirout, "exclusions.tif")
    f_baseline = os.path.join(TESTDATADIR, 'ri_exclusions', "exclusions.tif")

    with rasterio.open(f_test_out, 'r') as file:
        test_exclusions_data = file.read(1)
    with rasterio.open(f_baseline, 'r') as file:
        valid_exclusions_data = file.read(1)

    assert np.array_equal(test_exclusions_data, valid_exclusions_data)

    if PURGE:
        for fname in os.listdir(config_obj.dirout):
            os.remove(os.path.join(config_obj.dirout, fname))


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
