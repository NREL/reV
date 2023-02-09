# -*- coding: utf-8 -*-
# pylint: disable=all
"""
PyTest file for geothermal generation.
"""
import os
import json
import shutil
from tempfile import TemporaryDirectory

import pytest
import pandas as pd
import numpy as np

from reV.generation.generation import Gen
from reV import TESTDATADIR
from rex import Outputs

RTOL = 0.1
ATOL = 0.01


@pytest.mark.parametrize("depth", [2000, 4500])
def test_gen_geothermal(depth):
    """Test generation for geothermal module"""
    points = slice(0, 1)
    sam_files = TESTDATADIR + '/SAM/geothermal_default.json'

    meta = pd.DataFrame({"latitude": [41.29], "longitude": [-71.86],
                         "timezone": [-5]})
    meta.index.name = "gid"

    with TemporaryDirectory() as td:
        geo_sam_file = os.path.join(td, "geothermal_sam.json")
        geo_res_file = os.path.join(td, "test_geo.h5")
        with open(sam_files, "r") as fh:
            geo_config = json.load(fh)

        geo_config["resource_depth"] = depth
        with open(geo_sam_file, "w") as fh:
            json.dump(geo_config, fh)

        with Outputs(geo_res_file, 'w') as f:
            f.meta = meta
            f.time_index = pd.date_range(start='1/1/2018', end='1/1/2019',
                                         freq='H')[:-1]

        Outputs.add_dataset(
            geo_res_file, 'temperature_2000m', np.array([150]),
            np.float32, attrs={"units": "C"},
        )
        Outputs.add_dataset(
            geo_res_file, 'potential_MW_2000m', np.array([200]),
            np.float32, attrs={"units": "MW"},
        )


        output_request = ('annual_energy', 'cf_mean', 'cf_profile',
                          'gen_profile', 'lcoe_fcr', 'nameplate')
        gen = Gen.reV_run('geothermal', points, geo_sam_file, geo_res_file,
                          max_workers=1, output_request=output_request,
                          sites_per_worker=1, out_fpath=None,
                          scale_outputs=True)

        truth_vals = {"annual_energy": 1.74e+09, "cf_mean": 0.993,
                      "cf_profile": 0.993, "gen_profile": 198653.64,
                      "lcoe_fcr": 12.52, "nameplate": 200_000,
                      "resource_temp": 150}
        for dset in output_request:
            truth = truth_vals[dset]
            test = gen.out[dset]
            if len(test.shape) == 2:
                test = np.mean(test, axis=0)

            msg = ('{} outputs do not match baseline value! Values differ '
                   'at most by: {}'
                   .format(dset, np.max(np.abs(truth - test))))
            assert np.allclose(truth, test, rtol=RTOL, atol=ATOL), msg


def test_gen_geothermal_temp_too_low():
    """Test generation for geothermal module when temp too low"""
    points = slice(0, 1)
    sam_files = TESTDATADIR + '/SAM/geothermal_default.json'

    meta = pd.DataFrame({"latitude": [41.29], "longitude": [-71.86],
                         "timezone": [-5]})
    meta.index.name = "gid"

    with TemporaryDirectory() as td:
        geo_sam_file = os.path.join(td, "geothermal_sam.json")
        geo_res_file = os.path.join(td, "test_geo.h5")
        shutil.copy(sam_files, geo_sam_file)

        with Outputs(geo_res_file, 'w') as f:
            f.meta = meta
            f.time_index = pd.date_range(start='1/1/2018', end='1/1/2019',
                                         freq='H')[:-1]

        Outputs.add_dataset(
            geo_res_file, 'temperature_2000m', np.array([60]),
            np.float32, attrs={"units": "C"},
        )
        Outputs.add_dataset(
            geo_res_file, 'potential_MW_2000m', np.array([200]),
            np.float32, attrs={"units": "MW"},
        )


        output_request = ('annual_energy', 'cf_mean', 'cf_profile',
                          'gen_profile', 'lcoe_fcr', 'nameplate')
        gen = Gen.reV_run('geothermal', points, geo_sam_file, geo_res_file,
                          max_workers=1, output_request=output_request,
                          sites_per_worker=1, out_fpath=None,
                          scale_outputs=True)

        truth_vals = {"annual_energy": 0, "cf_mean": 0, "cf_profile": 0,
                      "gen_profile": 0, "lcoe_fcr": 0, "nameplate": 0,
                      "resource_temp": 60}
        for dset in output_request:
            truth = truth_vals[dset]
            test = gen.out[dset]
            if len(test.shape) == 2:
                test = np.mean(test, axis=0)

            msg = ('{} outputs do not match baseline value! Values differ '
                   'at most by: {}'
                   .format(dset, np.max(np.abs(truth - test))))
            assert np.allclose(truth, test, rtol=RTOL, atol=ATOL), msg


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
