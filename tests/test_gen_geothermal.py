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

DEFAULT_GEO_SAM_FILE = TESTDATADIR + '/SAM/geothermal_default.json'
RTOL = 0.1
ATOL = 0.01


@pytest.mark.parametrize("depth", [2000, 4500])
def test_gen_geothermal(depth):
    """Test generation for geothermal module"""
    points = slice(0, 1)

    meta = pd.DataFrame({"latitude": [41.29], "longitude": [-71.86],
                         "timezone": [-5]})
    meta.index.name = "gid"

    with TemporaryDirectory() as td:
        geo_sam_file = os.path.join(td, "geothermal_sam.json")
        geo_res_file = os.path.join(td, "test_geo.h5")
        with open(DEFAULT_GEO_SAM_FILE, "r") as fh:
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

    meta = pd.DataFrame({"latitude": [41.29], "longitude": [-71.86],
                         "timezone": [-5]})
    meta.index.name = "gid"

    with TemporaryDirectory() as td:
        geo_sam_file = os.path.join(td, "geothermal_sam.json")
        geo_res_file = os.path.join(td, "test_geo.h5")
        shutil.copy(DEFAULT_GEO_SAM_FILE, geo_sam_file)

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


def test_per_kw_cost_inputs():
    """Test per_kw cost inputs for geothermal module"""
    points = slice(0, 1)

    meta = pd.DataFrame({"latitude": [41.29], "longitude": [-71.86],
                         "timezone": [-5]})
    meta.index.name = "gid"

    with TemporaryDirectory() as td:
        geo_sam_file = os.path.join(td, "geothermal_sam.json")
        geo_res_file = os.path.join(td, "test_geo.h5")
        with open(DEFAULT_GEO_SAM_FILE, "r") as fh:
            geo_config = json.load(fh)

        geo_config["resource_depth"] = 2000
        geo_config.pop("capital_cost", None)
        geo_config.pop("fixed_operating_cost", None)
        geo_config["capital_cost_per_kw"] = 3_000
        geo_config["fixed_operating_cost_per_kw"] = 200
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
            geo_res_file, 'potential_MW_2000m', np.array([100]),
            np.float32, attrs={"units": "MW"},
        )

        output_request = ('capital_cost', 'fixed_operating_cost', 'lcoe_fcr')
        gen = Gen.reV_run('geothermal', points, geo_sam_file, geo_res_file,
                          max_workers=1, output_request=output_request,
                          sites_per_worker=1, out_fpath=None,
                          scale_outputs=True)

        truth_vals = {"capital_cost": 383_086_656,
                      "fixed_operating_cost": 25539104,
                      "lcoe_fcr": 72.5092}
        for dset in output_request:
            truth = truth_vals[dset]
            test = gen.out[dset]
            msg = ('{} outputs do not match baseline value! Values differ '
                   'at most by: {}'
                   .format(dset, np.max(np.abs(truth - test))))
            assert np.allclose(truth, test, rtol=1e-6, atol=ATOL), msg


def test_drill_cost_inputs():
    """Test per_kw cost inputs for geothermal module"""
    points = slice(0, 1)

    meta = pd.DataFrame({"latitude": [41.29], "longitude": [-71.86],
                         "timezone": [-5]})
    meta.index.name = "gid"

    with TemporaryDirectory() as td:
        geo_sam_file = os.path.join(td, "geothermal_sam.json")
        geo_res_file = os.path.join(td, "test_geo.h5")
        with open(DEFAULT_GEO_SAM_FILE, "r") as fh:
            geo_config = json.load(fh)

        geo_config["resource_depth"] = 2000
        geo_config.pop("capital_cost", None)
        geo_config.pop("fixed_operating_cost", None)
        geo_config["capital_cost_per_kw"] = 3_000
        geo_config["fixed_operating_cost_per_kw"] = 200
        geo_config["drill_cost_per_well"] = 2_500_000
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
            geo_res_file, 'potential_MW_2000m', np.array([100]),
            np.float32, attrs={"units": "MW"},
        )

        output_request = ('capital_cost', 'fixed_operating_cost', 'lcoe_fcr')
        gen = Gen.reV_run('geothermal', points, geo_sam_file, geo_res_file,
                          max_workers=1, output_request=output_request,
                          sites_per_worker=1, out_fpath=None,
                          scale_outputs=True)

        truth_vals = {"capital_cost": 466_134_733,
                      "fixed_operating_cost": 25539104,
                      "lcoe_fcr": 81.8643}
        for dset in output_request:
            truth = truth_vals[dset]
            test = gen.out[dset]
            msg = ('{} outputs do not match baseline value! Values differ '
                   'at most by: {}'
                   .format(dset, np.max(np.abs(truth - test))))
            assert np.allclose(truth, test, rtol=1e-6, atol=ATOL), msg


def test_gen_with_nameplate_input():
    """Test generation for geothermal module with user nameplate input"""
    points = slice(0, 1)

    meta = pd.DataFrame({"latitude": [41.29], "longitude": [-71.86],
                         "timezone": [-5]})
    meta.index.name = "gid"

    with TemporaryDirectory() as td:
        geo_sam_file = os.path.join(td, "geothermal_sam.json")
        geo_res_file = os.path.join(td, "test_geo.h5")
        with open(DEFAULT_GEO_SAM_FILE, "r") as fh:
            geo_config = json.load(fh)

        geo_config["resource_depth"] = 2000
        geo_config["nameplate"] = 40000
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
            geo_res_file, 'potential_MW_2000m', np.array([20]),
            np.float32, attrs={"units": "MW"},
        )

        output_request = ('annual_energy', 'cf_mean', 'cf_profile',
                          'gen_profile', 'lcoe_fcr', 'nameplate')
        gen = Gen.reV_run('geothermal', points, geo_sam_file, geo_res_file,
                          max_workers=1, output_request=output_request,
                          sites_per_worker=1, out_fpath=None,
                          scale_outputs=True)

        truth_vals = {"annual_energy": 3.47992e+08, "cf_mean": 0.993,
                      "cf_profile": 0.993, "gen_profile": 39725.117,
                      "lcoe_fcr": 62.613, "nameplate": 40_000,
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


def test_gen_egs_too_high_egs_plant_design_temp():
    """Test generation for EGS too high plant design temp"""
    points = slice(0, 1)

    meta = pd.DataFrame({"latitude": [41.29], "longitude": [-71.86],
                         "timezone": [-5]})
    meta.index.name = "gid"

    with TemporaryDirectory() as td:
        geo_sam_file = os.path.join(td, "geothermal_sam.json")
        geo_res_file = os.path.join(td, "test_geo.h5")
        with open(DEFAULT_GEO_SAM_FILE, "r") as fh:
            geo_config = json.load(fh)

        geo_config["resource_depth"] = 2000
        geo_config["resource_type"] = 1
        geo_config["design_temp"] = 200
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
            geo_res_file, 'potential_MW_2000m', np.array([20]),
            np.float32, attrs={"units": "MW"},
        )

        output_request = ('design_temp',)
        with pytest.warns(UserWarning):
            gen = Gen.reV_run('geothermal', points, geo_sam_file, geo_res_file,
                              max_workers=1, output_request=output_request,
                              sites_per_worker=1, out_fpath=None,
                              scale_outputs=True)

        truth_vals = {"design_temp": 150}
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
