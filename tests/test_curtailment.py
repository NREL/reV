# -*- coding: utf-8 -*-
# pylint: skip-file
"""
Created on Fri Mar  1 15:24:13 2019

@author: gbuster
"""

import os
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from rex.utilities import safe_json_load
from rex.utilities.solar_position import SolarPosition

from reV import TESTDATADIR
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

    out = curtail(resource, pp.curtailment, random_seed=0)

    return out, non_curtailed_res, pp


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

    curtailment = os.path.join(TESTDATADIR, "config/", "curtailment.json")
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
    results, check_curtailment = test_res_curtailment(year, site=site)
    results["cf_profile"] = gen.out["cf_profile"].flatten()

    # was capacity factor NOT curtailed?
    check_cf = gen.out["cf_profile"].flatten() != 0

    # Were all thresholds met and windspeed NOT curtailed?
    check = check_curtailment & check_cf

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
            curtailment=c,
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


@pytest.mark.parametrize(("year", "site"), [("2012", 50), ("2013", 50)])
def test_res_curtailment(year, site):
    """Test wind resource curtailment."""
    out, non_curtailed_res, pp = get_curtailment(year)

    sza = SolarPosition(
        non_curtailed_res.time_index,
        non_curtailed_res.meta[
            [ResourceMetaField.LATITUDE, ResourceMetaField.LONGITUDE]
        ].values,
    ).zenith

    ti = non_curtailed_res.time_index

    # was it in a curtailment month?
    check1 = np.isin(non_curtailed_res.time_index.month, pp.curtailment.months)
    check1 = np.tile(
        np.expand_dims(check1, axis=1), non_curtailed_res.shape[1]
    )

    # was the non-curtailed wind speed threshold met?
    check2 = (
        non_curtailed_res._res_arrays["windspeed"] < pp.curtailment.wind_speed
    )

    # was it nighttime?
    check3 = sza > pp.curtailment.dawn_dusk

    # was the temperature threshold met?
    check4 = out._res_arrays["temperature"] > pp.curtailment.temperature

    # thresholds for curtailment
    check_curtailment = check1 & check2 & check3 & check4

    # was windspeed NOT curtailed?
    check5 = out._res_arrays["windspeed"] != 0

    # Were all thresholds met and windspeed NOT curtailed?
    check = check_curtailment & check5

    msg = (
        "All curtailment thresholds were met and windspeed "
        "was not curtailed!"
    )

    assert np.sum(check) == 0, msg

    # optional output df to help check results
    i = site
    df = pd.DataFrame(
        {
            "i": range(len(sza)),
            "curtailed_wind": out._res_arrays["windspeed"][:, i],
            "original_wind": non_curtailed_res._res_arrays["windspeed"][:, i],
            "temperature": out._res_arrays["temperature"][:, i],
            "sza": sza[:, i],
            "wind_curtail": check2[:, i],
            "month_curtail": check1[:, i],
            "sza_curtail": check3[:, i],
            "temp_curtail": check4[:, i],
        },
        index=ti,
    )

    if str(year) == "2012":
        drop_day = (ti.month == 12) & (ti.day == 31)
        df = df.drop(df.index[drop_day])
        check_curtailment = check_curtailment[~drop_day, :]

    return df, check_curtailment[:, site]


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
