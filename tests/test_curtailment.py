# -*- coding: utf-8 -*-
# pylint: skip-file
"""
Created on Fri Mar  1 15:24:13 2019

@author: gbuster
"""

import os
import json
import shutil
import tempfile
import traceback
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from rex import Resource
from rex.utilities import safe_json_load
from rex.utilities.solar_position import SolarPosition

from reV import TESTDATADIR
from reV.cli import main
from reV.config.project_points import ProjectPoints
from reV.generation.generation import Gen
from reV.SAM.SAM import RevPySam
from reV.utilities import ResourceMetaField
from reV.utilities.curtailment import curtail


def get_curtailment(year, curt_fn="curtailment.json"):
    """Get the curtailed and non-curtailed resource objects, and project
    points"""
    res_file = os.path.join(
        TESTDATADIR, "wtk/", "ri_100_wtk_{}.h5".format(year)
    )
    sam_files = os.path.join(
        TESTDATADIR, "SAM/wind_gen_standard_losses_0.json"
    )
    curtailment = os.path.join(TESTDATADIR, "config/", curt_fn)
    pp = ProjectPoints(
        slice(0, 100), sam_files, "windpower", curtailment=curtailment
    )

    resource = RevPySam.get_sam_res(res_file, pp, "windpower")
    non_curtailed_res = deepcopy(resource)

    curtailment_config = list(pp.curtailment.values())[0]
    out = curtail(resource, curtailment_config, resource.sites, random_seed=0)

    return out, non_curtailed_res, pp


def _compute_res_curtailment_df(year, site, curt_fn):
    """Compute curtailment df for easy testing"""
    out, non_curtailed_res, __ = get_curtailment(year, curt_fn=curt_fn)
    ti = non_curtailed_res.time_index

    sza = SolarPosition(
        non_curtailed_res.time_index,
        non_curtailed_res.meta[
            [ResourceMetaField.LATITUDE, ResourceMetaField.LONGITUDE]
        ].values,
    ).zenith

    # optional output df to help check results
    i = site
    df = pd.DataFrame(
        {
            "i": range(len(sza)),
            "curtailed_wind": out._res_arrays["windspeed"][:, i],
            "original_wind": non_curtailed_res._res_arrays["windspeed"][:, i],
            "temperature": out._res_arrays["temperature"][:, i],
        },
        index=ti,
    )

    if str(year) == "2012":
        drop_day = (ti.month == 12) & (ti.day == 31)
        df = df.drop(df.index[drop_day])

    return df


def _compute_res_curtailment_check(year, curt_fn, drop_leap=True):
    """Compute wind resource curtailment check array"""
    out, non_curtailed_res, pp = get_curtailment(year, curt_fn=curt_fn)
    curtailment_config = list(pp.curtailment.values())[0]
    ti = non_curtailed_res.time_index

    sza = SolarPosition(
        non_curtailed_res.time_index,
        non_curtailed_res.meta[
            [ResourceMetaField.LATITUDE, ResourceMetaField.LONGITUDE]
        ].values,
    ).zenith

    # was it in a curtailment month?
    check1 = np.isin(non_curtailed_res.time_index.month,
                     curtailment_config.months)
    check1 = np.tile(
        np.expand_dims(check1, axis=1), non_curtailed_res.shape[1]
    )

    # was the non-curtailed wind speed threshold met?
    check2 = (
        non_curtailed_res._res_arrays["windspeed"]
        < curtailment_config.wind_speed
    )

    # was it nighttime?
    check3 = sza > curtailment_config.dawn_dusk

    # was the temperature threshold met?
    check4 = out._res_arrays["temperature"] > curtailment_config.temperature

    # thresholds for curtailment
    check_curtailment = check1 & check2 & check3 & check4

    if str(year) == "2012" and drop_leap:
        drop_day = (ti.month == 12) & (ti.day == 31)
        check_curtailment = check_curtailment[~drop_day, :]

    return check_curtailment, out


@pytest.mark.parametrize(
    ("year", "site"), [("2012", 0), ("2012", 10), ("2013", 0), ("2013", 10)]
)
def test_cf_curtailment(year, site):
    """Run Wind generation and ensure that the cf_profile is zero when
    curtailment is expected.

    Note that the probability of curtailment must be 1 for this to succeed.
    """

    res_file = os.path.join(TESTDATADIR, "wtk/ri_100_wtk_{}.h5".format(year))
    sam_files = os.path.join(
        TESTDATADIR, "SAM/wind_gen_standard_losses_0.json"
    )

    curt_fn = "curtailment.json"
    curtailment = os.path.join(TESTDATADIR, "config", curt_fn)
    points = slice(site, site + 1)

    # run reV 2.0 generation
    gen = Gen(
        "windpower",
        points,
        sam_files,
        res_file,
        output_request=("cf_profile",),
        curtailment=curtailment,
        sites_per_worker=50,
        scale_outputs=True,
    )
    gen.run(max_workers=1)
    check_curtailment, __ = _compute_res_curtailment_check(year,
                                                           curt_fn=curt_fn)

    # was capacity factor NOT curtailed?
    check_cf = gen.out["cf_profile"].flatten() != 0

    # Were all thresholds met and windspeed NOT curtailed?
    check = check_curtailment[:, site] & check_cf

    msg = (
        "All curtailment thresholds were met and cf_profile "
        "was not curtailed!"
    )
    assert np.sum(check) == 0, msg


@pytest.mark.parametrize("year", ["2012", "2013"])
def test_curtailment_res_mean(year):
    """Run Wind generation and ensure that the cf_profile is zero when
    curtailment is expected.

    Note that the probability of curtailment must be 1 for this to succeed.
    """

    res_file = os.path.join(TESTDATADIR, "wtk/ri_100_wtk_{}.h5".format(year))
    sam_files = os.path.join(
        TESTDATADIR, "SAM/wind_gen_standard_losses_0.json"
    )

    curtailment = os.path.join(TESTDATADIR, "config/", "curtailment.json")
    points = slice(0, 100)
    output_request = ("cf_mean", "ws_mean")
    pc = Gen.get_pc(
        points,
        None,
        sam_files,
        "windpower",
        sites_per_worker=50,
        res_file=res_file,
        curtailment=curtailment,
    )

    resources = RevPySam.get_sam_res(
        res_file, pc.project_points, pc.project_points.tech, output_request
    )
    truth = resources["mean_windspeed"]

    # run reV 2.0 generation
    gen = Gen(
        "windpower",
        points,
        sam_files,
        res_file,
        output_request=output_request,
        curtailment=curtailment,
        sites_per_worker=50,
        scale_outputs=True,
    )
    gen.run(max_workers=1)
    test = gen.out["ws_mean"]

    assert np.allclose(truth, test, rtol=0.001)


@pytest.mark.parametrize(("year", "site"), [("2012", 10), ("2013", 10)])
def test_random(year, site):
    """Run wind generation and ensure that no curtailment, 100% probability
    curtailment, and 50% probability curtailment result in expected decreases
    in the annual cf_mean.
    """
    res_file = os.path.join(TESTDATADIR, "wtk/ri_100_wtk_{}.h5".format(year))
    sam_files = os.path.join(
        TESTDATADIR, "SAM/wind_gen_standard_losses_0.json"
    )
    results = []
    no_curtail = None
    curtailment = {
        "dawn_dusk": "nautical",
        "months": [4, 5, 6, 7],
        "precipitation": None,
        "probability": 1,
        "temperature": None,
        "wind_speed": 10.0,
    }
    prob_curtail = {
        "dawn_dusk": "nautical",
        "months": [4, 5, 6, 7],
        "precipitation": None,
        "probability": 0.5,
        "temperature": None,
        "wind_speed": 10.0,
    }

    for c in [no_curtail, curtailment, prob_curtail]:
        points = slice(site, site + 1)

        # run reV 2.0 generation and write to disk
        gen = Gen(
            "windpower",
            points,
            sam_files,
            res_file,
            output_request=("cf_profile",),
            curtailment=None if c is None else {"test": c},
            sites_per_worker=50,
            scale_outputs=True,
        )
        gen.run(max_workers=1)

        results.append(gen.out["cf_mean"])

    assert results[0] > results[1], "Curtailment did not decrease cf_mean!"

    expected = (results[0] + results[1]) / 2
    diff = expected - results[2]
    msg = (
        "Curtailment with 50% probability did not result in 50% less "
        "curtailment! No curtailment, curtailment, and 50% curtailment "
        "have the following cf_means: {}".format(results)
    )
    assert diff <= 2, msg


@pytest.mark.parametrize("year", ["2012", "2013"])
@pytest.mark.parametrize("site", [50])
@pytest.mark.parametrize("curt_fn", ["curtailment.json"])
def test_res_curtailment(year, site, curt_fn):
    """Test wind resource curtailment."""
    check_curtailment, out = _compute_res_curtailment_check(year,
                                                            curt_fn=curt_fn,
                                                            drop_leap=False)

    # was windspeed NOT curtailed?
    check5 = out._res_arrays["windspeed"] != 0

    # Were all thresholds met and windspeed NOT curtailed?
    check = check_curtailment & check5

    msg = (
        "All curtailment thresholds were met and windspeed "
        "was not curtailed!"
    )

    assert np.sum(check) == 0, msg

def test_date_range():
    """Test curtailment based on a date range vs. months list"""
    year = 2012
    cres_m = get_curtailment(year, curt_fn="curtailment.json")[0]
    cres_dr = get_curtailment(year, curt_fn="curtailment_date_range.json")[0]
    for df_res, site in cres_m:
        gid = int(site.name)
        assert np.allclose(df_res["windspeed"], cres_dr[gid]["windspeed"])


def test_eqn_curtailment(plot=False):
    """Test equation-based curtailment strategies."""
    year = 2012
    curt_fn = "curtailment_eqn.json"
    curtailed, non_curtailed_res, pp = get_curtailment(year, curt_fn=curt_fn)
    c_config = safe_json_load(os.path.join(TESTDATADIR, "config/", curt_fn))
    c_eqn = c_config["equation"]

    c_res = curtailed[0]
    nc_res = non_curtailed_res[0]
    c_mask = (c_res.windspeed == 0) & (nc_res.windspeed > 0)

    temperature = nc_res["temperature"].values
    wind_speed = nc_res["windspeed"].values

    eval_mask = eval(c_eqn)

    # All curtailed windspeeds should satisfy the eqn eval but maybe not the
    # other way around due to dawn/dusk/sza
    assert all(eval_mask[np.where(c_mask)[0]] == True)  # noqa: E712

    if plot:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        ax.scatter(
            nc_res.loc[c_mask, "windspeed"], nc_res.loc[c_mask, "temperature"]
        )
        ax.grid("on")
        ax.set_xlim([0, 7])
        ax.set_ylim([0, 30])
        ax.set_legend(["Curtailed"])
        plt.savefig("equation_based_curtailment.png")


def test_points_missing_curtailment():
    """Test that points with missing curtailment values throw a warning"""

    res_file = os.path.join(TESTDATADIR, "wtk/ri_100_wtk_2012.h5")
    sam_files = os.path.join(TESTDATADIR,
                             "SAM/wind_gen_standard_losses_0.json")
    curt_fn = "curtailment.json"
    curtail_config = os.path.join(TESTDATADIR, "config", curt_fn)

    curtailment = {"test_c1": curtail_config, "test_c2": curtail_config}
    points = slice(0, 1)
    # run reV 2.0 generation
    gen = Gen("windpower", points, sam_files, res_file,
              output_request=("cf_profile", "windspeed"),
              curtailment=curtailment,
              sites_per_worker=50, scale_outputs=True)

    with pytest.warns(UserWarning) as warn:
        gen.run(max_workers=1)

    expected_msgs = ("One or more curtailment configurations not found "
                     "in project points and are thus ignored",
                     "test_c1", "test_c2")
    for msg in expected_msgs:
        assert any(msg in str(record) for record in warn)

    df = _compute_res_curtailment_df(2012, 0, curt_fn=curt_fn)
    assert np.allclose(gen.out["windspeed"][:, 0], df["original_wind"])


def test_basic_spatial_curtailment():
    """Test basic execution of spatial curtailment"""

    res_file = os.path.join(TESTDATADIR, "wtk/ri_100_wtk_2012.h5")
    sam_files = os.path.join(TESTDATADIR,
                             "SAM/wind_gen_standard_losses_0.json")
    curt_fn = "curtailment.json"
    curtail_config = os.path.join(TESTDATADIR, "config", curt_fn)

    points = pd.DataFrame({"gid": [0, 1], "curtailment": ["default", None]})
    out_req = ("cf_profile", "windspeed", "cf_mean")
    # run reV 2.0 generation
    gen = Gen("windpower", points, sam_files, res_file, output_request=out_req,
              curtailment=curtail_config, sites_per_worker=50,
              scale_outputs=True)

    gen.run(max_workers=1)

    df = _compute_res_curtailment_df(2012, 0, curt_fn=curt_fn)
    assert not np.allclose(gen.out["windspeed"][:, 0], df["original_wind"])
    assert np.allclose(gen.out["windspeed"][:, 0], df["curtailed_wind"])

    df = _compute_res_curtailment_df(2012, 1, curt_fn=curt_fn)
    assert np.allclose(gen.out["windspeed"][:, 1], df["original_wind"])
    assert not np.allclose(gen.out["windspeed"][:, 1], df["curtailed_wind"])

    non_curtailed_gen = Gen("windpower", points, sam_files, res_file,
                            output_request=out_req, sites_per_worker=50,
                            scale_outputs=True)

    non_curtailed_gen.run(max_workers=1)

    assert non_curtailed_gen.out["cf_mean"][0] > gen.out["cf_mean"][0]
    assert non_curtailed_gen.out["cf_mean"][1] == gen.out["cf_mean"][1]


def test_multiple_spatial_curtailment():
    """Test execution with multiple spatial curtailments"""

    res_file = os.path.join(TESTDATADIR, "wtk/ri_100_wtk_2012.h5")
    sam_files = os.path.join(TESTDATADIR,
                             "SAM/wind_gen_standard_losses_0.json")
    curtail_config = {"c 1": os.path.join(TESTDATADIR, "config",
                                          "curtailment.json"),
                      "c 2": os.path.join(TESTDATADIR, "config",
                                          "curtailment_date_range.json")}

    points = pd.DataFrame({"gid": [0, 10, 25, 33, 49],
                           "curtailment": ["c 1", None, "c 2", "c 1", None]})
    # run reV 2.0 generation
    gen = Gen("windpower", points, sam_files, res_file,
              output_request=("cf_profile", "windspeed"),
              curtailment=curtail_config,
              sites_per_worker=50, scale_outputs=True)

    gen.run(max_workers=1)

    for (gid, g_ind) in [(10, 1), (49, 4)]:
        df = _compute_res_curtailment_df(2012, gid, curt_fn="curtailment.json")
        assert np.allclose(gen.out["windspeed"][:, g_ind], df["original_wind"])
        assert not np.allclose(gen.out["windspeed"][:, g_ind],
                               df["curtailed_wind"])

    for (gid, g_ind) in [(0, 0), (33, 3)]:
        df = _compute_res_curtailment_df(2012, gid, curt_fn="curtailment.json")
        assert not np.allclose(gen.out["windspeed"][:, g_ind],
                               df["original_wind"])
        assert np.allclose(gen.out["windspeed"][:, g_ind],
                           df["curtailed_wind"])

    df = _compute_res_curtailment_df(2012, 25,
                                     curt_fn="curtailment_date_range.json")
    assert not np.allclose(gen.out["windspeed"][:, 2], df["original_wind"])
    assert np.allclose(gen.out["windspeed"][:, 2], df["curtailed_wind"])


@pytest.mark.parametrize("points", [pd.DataFrame({"gid": [0]}),
                                    pd.DataFrame({"gid": [0],
                                                  "curtailment": [None]}),
                                    pd.DataFrame({"gid": [0],
                                                  "curtailment": ["default"]
                                                  })])
def test_default_curtailment(points):
    """Test basic execution of spatial curtailment"""

    res_file = os.path.join(TESTDATADIR, "wtk/ri_100_wtk_2012.h5")
    sam_files = os.path.join(TESTDATADIR,
                             "SAM/wind_gen_standard_losses_0.json")
    curt_fn = "curtailment.json"
    curtail_config = os.path.join(TESTDATADIR, "config", curt_fn)

    # run reV 2.0 generation
    gen = Gen("windpower", points, sam_files, res_file,
              output_request=("cf_profile", "windspeed"),
              curtailment=curtail_config,
              sites_per_worker=50, scale_outputs=True)

    gen.run(max_workers=1)

    df = _compute_res_curtailment_df(2012, 0, curt_fn=curt_fn)
    assert not np.allclose(gen.out["windspeed"][:, 0], df["original_wind"])
    assert np.allclose(gen.out["windspeed"][:, 0], df["curtailed_wind"])


def test_multiple_spatial_curtailment_cli(runner, clear_loggers):
    """Test execution with multiple spatial curtailments from CLI"""

    res_file = os.path.join(TESTDATADIR, "wtk/ri_100_wtk_2012.h5")
    sam_files = os.path.join(TESTDATADIR,
                             "SAM/wind_gen_standard_losses_0.json")
    source_cc1 = os.path.join(TESTDATADIR, "config", "curtailment.json")
    source_cc2 = os.path.join(TESTDATADIR, "config",
                              "curtailment_date_range.json")
    gen_config = os.path.join(TESTDATADIR, 'config', 'local_wind.json')
    c1_key = "test config 1"
    c2_key = "Test Config 2"

    with tempfile.TemporaryDirectory() as td:
        cc1 = os.path.join(td, "curtailment.json")
        cc2 = os.path.join(td, "curtailment date range.json")
        shutil.copy(source_cc1, cc1)
        shutil.copy(source_cc2, cc2)

        curtail_config = {c1_key: cc1, c2_key: cc2}

        points_fp = os.path.join(td, "points.csv")
        points = pd.DataFrame({"gid": [0, 10, 25, 33, 49],
                               "curtailment": [c1_key, None, c2_key, c1_key,
                                               None]})
        points.to_csv(points_fp, index=False)

        config = safe_json_load(gen_config)
        config["execution_control"]["max_workers"] = 1
        config["execution_control"]["sites_per_worker"] = 50
        config['project_points'] = points_fp
        config['resource_file'] = res_file
        config['sam_files'] = sam_files
        config['log_directory'] = os.path.join(td, 'logs')
        config['output_request'] = ["cf_profile", "windspeed"]
        config['curtailment'] = curtail_config

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['generation', '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        h5_files = [fn for fn in os.listdir(td)
                    if "generation_2012" in fn and ".h5" in fn]
        assert len(h5_files) == 1

        out_file = os.path.join(td, h5_files[0])
        with Resource(out_file) as res:
            wind_speeds = res["windspeed"]

        clear_loggers()

    for (gid, g_ind) in [(10, 1), (49, 4)]:
        df = _compute_res_curtailment_df(2012, gid, curt_fn="curtailment.json")
        assert np.allclose(wind_speeds[:, g_ind], df["original_wind"])
        assert not np.allclose(wind_speeds[:, g_ind], df["curtailed_wind"])

    for (gid, g_ind) in [(0, 0), (33, 3)]:
        df = _compute_res_curtailment_df(2012, gid, curt_fn="curtailment.json")
        assert not np.allclose(wind_speeds[:, g_ind], df["original_wind"])
        assert np.allclose(wind_speeds[:, g_ind], df["curtailed_wind"])

    df = _compute_res_curtailment_df(2012, 25,
                                     curt_fn="curtailment_date_range.json")
    assert not np.allclose(wind_speeds[:, 2], df["original_wind"])
    assert np.allclose(wind_speeds[:, 2], df["curtailed_wind"])


def execute_pytest(capture="all", flags="-rapP"):
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
    pytest.main(["-q", "--show-capture={}".format(capture), fname, flags])


if __name__ == "__main__":
    execute_pytest()
