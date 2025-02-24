# -*- coding: utf-8 -*-
# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import json
import shutil
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest
from rex.utilities.exceptions import ResourceRuntimeError

from reV import TESTDATADIR
from reV.config.project_points import ProjectPoints
from reV.generation.generation import Gen
from reV.handlers.outputs import Outputs
from reV.utilities import SiteDataField
from reV.utilities.exceptions import ConfigError, ExecutionError

RTOL = 0.0
ATOL = 0.04


class pv_results:
    """Class to retrieve results from the rev 1.0 pv files"""

    def __init__(self, f):
        self._h5 = h5py.File(f, "r")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._h5.close()

        if type is not None:
            raise

    @property
    def years(self):
        """Get a list of year strings."""
        if not hasattr(self, "_years"):
            year_index = self._h5["pv"]["year_index"][...]
            self._years = [y.decode() for y in year_index]
        return self._years

    def get_cf_mean(self, site, year):
        """Get a cf mean based on site and year"""
        iy = self.years.index(year)
        out = self._h5["pv"]["cf_mean"][iy, site]
        return out


def is_num(n):
    """Check if n is a number"""
    try:
        float(n)
        return True
    except Exception:
        return False


def _to_list(gen_out):
    """Generation output handler that converts to the rev 1.0 format."""
    if isinstance(gen_out, list) and len(gen_out) == 1:
        out = [c["cf_mean"] for c in gen_out[0].values()]

    if isinstance(gen_out, dict):
        out = [c["cf_mean"] for c in gen_out.values()]

    return out


@pytest.mark.parametrize(
    ("f_rev1_out", "rev2_points", "year", "max_workers"),
    [
        ("project_outputs.h5", slice(0, 10), "2012", 1),
        ("project_outputs.h5", slice(0, None, 10), "2013", 1),
        ("project_outputs.h5", slice(3, 25, 2), "2012", 2),
        ("project_outputs.h5", slice(40, None, 10), "2013", 2),
    ],
)
def test_pv_gen_slice(f_rev1_out, rev2_points, year, max_workers):
    """Test reV 2.0 generation for PV and benchmark against reV 1.0 results."""
    # get full file paths.
    rev1_outs = os.path.join(
        TESTDATADIR, "ri_pv", "scalar_outputs", f_rev1_out
    )
    sam_files = TESTDATADIR + "/SAM/naris_pv_1axis_inv13.json"
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)

    # run reV 2.0 generation
    pp = ProjectPoints(rev2_points, sam_files, "pvwattsv5", res_file=res_file)
    gen = Gen(
        "pvwattsv5", rev2_points, sam_files, res_file, sites_per_worker=3
    )
    gen.run(max_workers=max_workers)
    gen_outs = list(gen.out["cf_mean"])

    # initialize the rev1 output hander
    with pv_results(rev1_outs) as pv:
        # get reV 1.0 results
        cf_mean_list = pv.get_cf_mean(pp.sites, year)

    # benchmark the results
    assert np.allclose(gen_outs, cf_mean_list, rtol=RTOL, atol=ATOL)


def test_pv_gen_csv1(
    f_rev1_out="project_outputs.h5",
    rev2_points=TESTDATADIR + "/project_points/ri.csv",
    res_file=TESTDATADIR + "/nsrdb/ri_100_nsrdb_2012.h5",
):
    """Test project points csv input with dictionary-based sam files."""
    rev1_outs = os.path.join(
        TESTDATADIR, "ri_pv", "scalar_outputs", f_rev1_out
    )
    sam_files = {
        "sam_param_0": TESTDATADIR + "/SAM/naris_pv_1axis_inv13.json",
        "sam_param_1": TESTDATADIR + "/SAM/naris_pv_1axis_inv13.json",
    }
    pp = ProjectPoints(rev2_points, sam_files, "pvwattsv5")

    # run reV 2.0 generation
    gen = Gen("pvwattsv5", rev2_points, sam_files, res_file)
    gen.run()
    gen_outs = list(gen.out["cf_mean"])

    # initialize the rev1 output hander
    with pv_results(rev1_outs) as pv:
        # get reV 1.0 results
        cf_mean_list = pv.get_cf_mean(pp.sites, "2012")

    # benchmark the results
    assert np.allclose(gen_outs, cf_mean_list, rtol=RTOL, atol=ATOL)


def test_pv_gen_csv2(
    f_rev1_out="project_outputs.h5",
    rev2_points=TESTDATADIR + "/project_points/ri.csv",
    res_file=TESTDATADIR + "/nsrdb/ri_100_nsrdb_2012.h5",
):
    """Test project points csv input with list-based sam files."""
    rev1_outs = os.path.join(
        TESTDATADIR, "ri_pv", "scalar_outputs", f_rev1_out
    )
    sam_files = [
        TESTDATADIR + "/SAM/naris_pv_1axis_inv13.json",
        TESTDATADIR + "/SAM/naris_pv_1axis_inv13.json",
    ]
    sam_files = {"sam_param_{}".format(i): k for i, k in enumerate(sam_files)}
    pp = ProjectPoints(rev2_points, sam_files, "pvwattsv5")
    gen = Gen("pvwattsv5", rev2_points, sam_files, res_file)
    gen.run()
    gen_outs = list(gen.out["cf_mean"])

    # initialize the rev1 output hander
    with pv_results(rev1_outs) as pv:
        # get reV 1.0 results
        cf_mean_list = pv.get_cf_mean(pp.sites, "2012")

    # benchmark the results
    assert np.allclose(gen_outs, cf_mean_list, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("year", [("2012"), ("2013")])
def test_pv_gen_profiles(year):
    """Gen PV CF profiles with write to disk and compare against rev1."""
    with tempfile.TemporaryDirectory() as td:
        res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
        sam_files = TESTDATADIR + "/SAM/naris_pv_1axis_inv13.json"
        fn = "gen_ri_pv_generation_{}.h5".format(year)
        rev2_out = os.path.join(td, fn)

        points = slice(0, 100)

        # run reV 2.0 generation and write to disk
        gen = Gen(
            "pvwattsv5",
            points,
            sam_files,
            res_file,
            output_request=("cf_profile",),
            sites_per_worker=50,
        )
        gen.run(max_workers=2, out_fpath=rev2_out)

        with Outputs(rev2_out, "r") as cf:
            rev2_profiles = cf["cf_profile"]

        # get reV 1.0 generation profiles
        rev1_profiles = get_r1_profiles(year=year)
        rev1_profiles = rev1_profiles[:, points]

        assert np.allclose(rev1_profiles, rev2_profiles, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("year", [("2012"), ("2013")])
def test_smart(year):
    """Gen PV CF profiles with write to disk and compare against rev1."""
    with tempfile.TemporaryDirectory() as td:
        res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
        sam_files = TESTDATADIR + "/SAM/naris_pv_1axis_inv13.json"
        fn = "gen_ri_pv_generation_smart_{}.h5".format(year)
        rev2_out = os.path.join(td, fn)

        points = slice(0, 10)

        # run reV 2.0 generation and write to disk
        gen = Gen(
            "pvwattsv5",
            points,
            sam_files,
            res_file,
            output_request=("cf_profile",),
            sites_per_worker=50,
        )
        gen.run(max_workers=2, out_fpath=rev2_out)

        with Outputs(rev2_out, "r") as cf:
            rev2_profiles = cf["cf_profile"]

        # get reV 1.0 generation profiles
        rev1_profiles = get_r1_profiles(year=year)
        rev1_profiles = rev1_profiles[:, points]

        assert np.allclose(rev1_profiles, rev2_profiles, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("model", ["pvwattsv5", "pvwattsv7"])
def test_multi_file_nsrdb_2018(model):
    """Test running reV gen from a multi-h5 directory with prefix and suffix"""
    points = slice(0, 10)
    max_workers = 1
    sam_files = TESTDATADIR + "/SAM/naris_pv_1axis_inv13.json"
    res_file = TESTDATADIR + "/nsrdb/nsrdb_*{}.h5".format(2018)
    # run reV 2.0 generation
    gen = Gen(model, points, sam_files, res_file,
              output_request=('cf_mean', 'cf_profile'),
              sites_per_worker=3)
    gen.run(max_workers=max_workers)

    means_outs = list(gen.out["cf_mean"])
    assert len(means_outs) == 10
    assert np.mean(means_outs) > 0.14

    profiles_out = gen.out["cf_profile"]
    assert profiles_out.shape == (105120, 10)
    assert np.mean(profiles_out) > 0.14


def get_r1_profiles(year=2012):
    """Get the first 100 reV 1.0 ri pv generation profiles."""
    rev1 = os.path.join(
        TESTDATADIR, "ri_pv", "profile_outputs", "pv_{}_0.h5".format(year)
    )
    with Outputs(rev1) as cf:
        data = cf["cf_profile"][...] / 10000

    return data


def test_pv_name_error():
    """Test reV 2.0 generation for PV and benchmark against reV 1.0 results."""

    year = 2012
    rev2_points = slice(0, 3)
    sam_files = TESTDATADIR + "/SAM/naris_pv_1axis_inv13.json"
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)

    # run reV 2.0 generation
    with pytest.raises(KeyError) as record:
        gen = Gen("pv", rev2_points, sam_files, res_file, sites_per_worker=1)
        gen.run(max_workers=1)
        assert "Did not recognize" in record[0].message


def test_southern_hemisphere():
    """Test reV pvwatts in the southern hemisphere with correct azimuth"""

    rev2_points = slice(0, 1)
    res_file = TESTDATADIR + "/nsrdb/brazil_solar.h5"
    sam_files = TESTDATADIR + "/SAM/i_pvwatts_fixed_lat_tilt.json"
    output_request = (
        "cf_mean",
        "cf_profile",
        "dni_mean",
        "dhi_mean",
        "ghi_mean",
        "ac",
        "dc",
        "azimuth",
    )

    gen = Gen(
        "pvwattsv7",
        rev2_points,
        sam_files,
        res_file,
        sites_per_worker=1,
        output_request=output_request,
    )
    gen.run(max_workers=1)
    assert gen.out["azimuth"] == 0

    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_2012.h5"
    gen = Gen(
        "pvwattsv7",
        rev2_points,
        sam_files,
        res_file,
        sites_per_worker=1,
        output_request=output_request,
    )
    gen.run(max_workers=1)
    assert gen.out["azimuth"] == 180


def test_pvwattsv7_baseline():
    """Test reV pvwattsv7 generation against baseline data"""

    baseline_cf_mean = np.array([0.1516, 0.1517, 0.1573])

    year = 2012
    rev2_points = slice(0, 3)
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
    sam_files = TESTDATADIR + "/SAM/i_pvwattsv7.json"

    output_request = (
        "cf_mean",
        "cf_profile",
        "dni_mean",
        "dhi_mean",
        "ghi_mean",
        "ac",
        "dc",
    )

    # run reV 2.0 generation
    gen = Gen(
        "pvwattsv7",
        rev2_points,
        sam_files,
        res_file,
        sites_per_worker=1,
        output_request=output_request,
    )
    gen.run(max_workers=1)

    msg = "PVWattsv7 cf_mean results {} did not match baseline: {}".format(
        gen.out["cf_mean"], baseline_cf_mean
    )
    assert np.allclose(
        gen.out["cf_mean"], baseline_cf_mean, rtol=0.005, atol=0.0
    ), msg

    for req in output_request:
        assert req in gen.out
        assert (gen.out[req] != 0).sum() > 0


def test_pvwatts_v5_v7():
    """Test reV pvwatts generation for v5 vs. v7"""

    year = 2012
    rev2_points = slice(0, 3)
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
    sam_files = TESTDATADIR + "/SAM/naris_pv_1axis_inv13.json"

    # run reV 2.0 generation
    gen7 = Gen(
        "pvwattsv7", rev2_points, sam_files, res_file, sites_per_worker=1
    )
    gen7.run(max_workers=1)

    gen5 = Gen(
        "pvwattsv5", rev2_points, sam_files, res_file, sites_per_worker=1
    )
    gen5.run(max_workers=1)

    msg = 'PVwatts v5 and v7 did not match within test tolerance'
    assert np.allclose(gen7.out['cf_mean'],
                       gen5.out['cf_mean'], atol=3), msg


def test_pvwatts_v8_lifetime():
    """Test reV pvwatts v8 generation/LCOE with system lifetime outputs."""
    baseline_cf_mean = np.array([0.1821, 0.1826, 0.189])

    year = 2012
    rev2_points = slice(0, 3)
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
    sam_files = TESTDATADIR + "/SAM/i_pvwattsv8_degradation.json"

    output_request = (
        "cf_mean",
        "cf_profile",
        "dni_mean",
        "dhi_mean",
        "ghi_mean",
    )

    # run reV 2.0 generation with valid output request
    gen = Gen(
        "pvwattsv8",
        rev2_points,
        sam_files,
        res_file,
        sites_per_worker=1,
        output_request=output_request,
    )
    gen.run(max_workers=1)

    msg = ('PVWATTSV8 cf_mean with system lifetime results {} did not match '
           'baseline: {}'.format(gen.out['cf_mean'],
                                 baseline_cf_mean))
    assert np.allclose(gen.out['cf_mean'], baseline_cf_mean,
                       rtol=0.005, atol=0.0), msg

    for req in output_request:
        assert req in gen.out
        assert (gen.out[req] != 0).sum() > 0


def test_pvwatts_v8_lifetime_invalid_request():
    """Test pvwatts v8 generation error with invalid lifetime outputs."""

    year = 2012
    rev2_points = slice(0, 3)
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
    sam_files = TESTDATADIR + "/SAM/i_pvwattsv8_degradation.json"

    output_request_invalid = (
        "cf_mean",
        "cf_profile",
        "dni_mean",
        "dhi_mean",
        "ghi_mean",
        "ac",
        "dc",
    )

    # run reV 2.0 generation with invalid output request
    with pytest.raises(ConfigError) as record:
        gen = Gen(
            "pvwattsv8",
            rev2_points,
            sam_files,
            res_file,
            sites_per_worker=1,
            output_request=output_request_invalid,
        )
        gen.run(max_workers=1)
        msg_pattern = (
            "reV can only handle the following output arrays when "
            "modeling with `system_use_lifetime_output`"
        )
        assert msg_pattern in record[0].message


def test_bifacial():
    """Test pvwattsv7 with bifacial panel with albedo."""
    year = 2012
    rev2_points = slice(0, 1)
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
    sam_files = TESTDATADIR + "/SAM/i_pvwattsv7.json"
    # run reV 2.0 generation
    gen = Gen(
        "pvwattsv7", rev2_points, sam_files, res_file, sites_per_worker=1
    )
    gen.run(max_workers=1)

    sam_files = TESTDATADIR + "/SAM/i_pvwattsv7_bifacial.json"
    # run reV 2.0 generation
    output_request = ("cf_mean", "cf_profile", "surface_albedo")
    gen_bi = Gen(
        "pvwattsv7",
        rev2_points,
        sam_files,
        res_file,
        sites_per_worker=1,
        output_request=output_request,
    )
    gen_bi.run(max_workers=1)

    assert "surface_albedo" in gen_bi.out
    assert all(gen_bi.out["cf_mean"] > gen.out["cf_mean"])
    assert np.isclose(gen.out["cf_mean"][0], 0.151, atol=0.005)
    assert np.isclose(gen_bi.out["cf_mean"][0], 0.162, atol=0.005)


def test_gen_input_mods():
    """Test that the gen workers do not modify the top level input SAM
    config"""
    year = 2012
    rev2_points = slice(0, 5)
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
    sam_files = TESTDATADIR + "/SAM/i_pvwatts_fixed_lat_tilt.json"

    # run reV 2.0 generation
    gen = Gen(
        "pvwattsv7", rev2_points, sam_files, res_file, sites_per_worker=1
    )
    gen.run(max_workers=1)
    for i in range(5):
        inputs = gen.project_points[i][1]
        assert inputs['tilt'] == "latitude"


def test_gen_input_pass_through():
    """Test the ability for reV gen to pass through inputs from the sam
    config."""
    output_request = ("cf_mean", "gcr", "azimuth")
    year = 2012
    rev2_points = slice(0, 2)
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
    sam_files = TESTDATADIR + "/SAM/i_pvwatts_fixed_lat_tilt.json"
    # run reV 2.0 generation
    gen = Gen(
        "pvwattsv7",
        rev2_points,
        sam_files,
        res_file,
        sites_per_worker=1,
        output_request=output_request,
    )
    gen.run(max_workers=1)
    assert "gcr" in gen.out
    assert "azimuth" in gen.out

    output_request = ("cf_mean", "gcr", "azimuth", "tilt")
    with pytest.raises(ExecutionError):
        gen = Gen(
            "pvwattsv7",
            rev2_points,
            sam_files,
            res_file,
            sites_per_worker=1,
            output_request=output_request,
        )
        gen.run(max_workers=1)


def test_gen_pv_site_data():
    """Test site specific SAM input config via site_data arg"""
    output_request = ("cf_mean", "gcr", "azimuth", "losses")
    year = 2012
    rev2_points = slice(0, 5)
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
    sam_files = TESTDATADIR + "/SAM/i_pvwatts_fixed_lat_tilt.json"
    # run reV 2.0 generation
    baseline = Gen(
        "pvwattsv7",
        rev2_points,
        sam_files,
        res_file,
        sites_per_worker=1,
        output_request=output_request,
    )
    baseline.run(max_workers=1)

    site_data = pd.DataFrame({SiteDataField.GID: np.arange(2),
                              'losses': np.ones(2)})
    test = Gen('pvwattsv7', rev2_points, sam_files, res_file,
               sites_per_worker=1, output_request=output_request,
               site_data=site_data)
    test.run(max_workers=1)

    assert all(test.out['cf_mean'][0:2] > baseline.out['cf_mean'][0:2])
    assert np.allclose(test.out['cf_mean'][2:], baseline.out['cf_mean'][2:])
    assert np.allclose(test.out['losses'][0:2], np.ones(2))
    assert np.allclose(test.out['losses'][2:], 14.07566 * np.ones(3))


def test_clipping():
    """Test reV pvwattsv7 generation against baseline data"""
    year = 2012
    rev2_points = slice(0, 3)
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
    sam_files = TESTDATADIR + "/SAM/i_pvwattsv7.json"

    output_request = ("ac", "dc", "clipped_power")

    # run reV 2.0 generation
    gen = Gen(
        "pvwattsv7",
        rev2_points,
        sam_files,
        res_file,
        sites_per_worker=1,
        output_request=output_request,
    )
    gen.run(max_workers=1)

    ac = gen.out["ac"]
    dc = gen.out["dc"]
    clipped = gen.out["clipped_power"]

    mask = ac < ac.max()
    dc_ac = dc[~mask] - ac[~mask]
    assert all(clipped[mask] == 0)
    assert np.allclose(clipped[~mask], dc_ac, rtol=RTOL, atol=ATOL)


def test_detailed_pv_baseline():
    """Test the detailed pv module against baseline values that are similar to
    the pvwattsv7 cf mean values"""
    baseline_cf_mean = np.array([0.1623, 0.1624, 0.1680])

    year = 2012
    rev2_points = slice(0, 3)
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
    sam_files = TESTDATADIR + "/SAM/i_pvsamv1.json"

    output_request = (
        "cf_mean",
        "cf_profile",
        "dni_mean",
        "dhi_mean",
        "ghi_mean",
        "ac",
        "dc",
    )

    # run reV 2.0 generation
    gen = Gen(
        "pvsamv1",
        rev2_points,
        sam_files,
        res_file,
        sites_per_worker=1,
        output_request=output_request,
    )
    gen.run(max_workers=1)

    msg = "PVSAMv1 cf_mean results {} did not match baseline: {}".format(
        gen.out["cf_mean"], baseline_cf_mean
    )
    assert np.allclose(
        gen.out["cf_mean"], baseline_cf_mean, rtol=0.005, atol=0.0
    ), msg

    for req in output_request:
        assert req in gen.out
        assert (gen.out[req] != 0).sum() > 0


def test_detailed_pv_bifacial():
    """Test the detailed pv module with bifacial configs"""
    baseline_cf_mean = np.array([0.1745, 0.17455, 0.1799])

    year = 2012
    rev2_points = slice(0, 3)
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
    sam_files = TESTDATADIR + "/SAM/i_pvsamv1_bifacial.json"

    output_request = (
        "cf_mean",
        "cf_profile",
        "dni_mean",
        "dhi_mean",
        "ghi_mean",
        "ac",
        "dc",
        "surface_albedo",
    )

    # run reV 2.0 generation
    gen = Gen(
        "pvsamv1",
        rev2_points,
        sam_files,
        res_file,
        sites_per_worker=1,
        output_request=output_request,
    )
    gen.run(max_workers=1)

    msg = "PVSAMv1 cf_mean results {} did not match baseline: {}".format(
        gen.out["cf_mean"], baseline_cf_mean
    )
    assert np.allclose(
        gen.out["cf_mean"], baseline_cf_mean, rtol=0.005, atol=0.0
    ), msg

    for req in output_request:
        assert req in gen.out
        assert (gen.out[req] != 0).sum() > 0


def test_pv_clearsky():
    """Test basic clearsky functionality"""
    year = 2012
    rev2_points = slice(0, 3)
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
    sam_files = TESTDATADIR + "/SAM/naris_pv_1axis_inv13.json"
    sam_files_cs = TESTDATADIR + "/SAM/naris_pv_1axis_inv13_cs.json"

    output_request = (
        "cf_mean",
        "cf_profile",
        "dni_mean",
        "dhi_mean",
        "ghi_mean",
        "ac",
        "dc",
    )
    output_request_cs = (
        "cf_mean",
        "cf_profile",
        "clearsky_dni_mean",
        "clearsky_dhi_mean",
        "clearsky_ghi_mean",
        "ac",
        "dc",
    )

    with tempfile.TemporaryDirectory() as td:
        res_cs = os.path.join(td, "res_cs_{}.h5".format(year))
        shutil.copy(res_file, res_cs)
        with Outputs(res_cs, mode="a") as f:
            f.write_dataset("clearsky_ghi", f["ghi"], np.float32)
            f.write_dataset("clearsky_dni", f["dni"], np.float32)
            f.write_dataset("clearsky_dhi", f["dhi"], np.float32)

        with pytest.raises(ResourceRuntimeError):
            gen = Gen(
                "pvwattsv7",
                rev2_points,
                sam_files_cs,
                res_file,
                sites_per_worker=1,
                output_request=output_request_cs,
            )
            gen.run(max_workers=1)

        gen_cs = Gen(
            "pvwattsv7",
            rev2_points,
            sam_files_cs,
            res_cs,
            sites_per_worker=1,
            output_request=output_request_cs,
        )
        gen_cs.run(max_workers=1)
        gen = Gen(
            "pvwattsv7",
            rev2_points,
            sam_files,
            res_file,
            sites_per_worker=1,
            output_request=output_request,
        )
        gen.run(max_workers=1)

        for k, v in gen.out.items():
            if k in ("ghi_mean", "dni_mean", "dhi_mean"):
                k = "clearsky_" + k
            assert np.allclose(v, gen_cs.out[k])


def test_irrad_bias_correct():
    """Test reV generation with bias correction"""
    year = 2012
    points = slice(0, 3)
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
    sam_files = TESTDATADIR + "/SAM/i_pvwattsv7.json"

    output_request = (
        "cf_mean",
        "cf_profile",
        "dni_mean",
        "dhi_mean",
        "ghi_mean",
        "ac",
        "dc",
    )

    gen_base = Gen(
        "pvwattsv7",
        points,
        sam_files,
        res_file,
        sites_per_worker=1,
        output_request=output_request,
    )
    gen_base.run(max_workers=1)

    bc_df = pd.DataFrame({SiteDataField.GID: np.arange(1, 10),
                          'method': 'lin_irrad', 'scalar': 1, 'adder': 50})
    gen = Gen('pvwattsv7', points, sam_files, res_file,
              sites_per_worker=1, output_request=output_request,
              bias_correct=bc_df)
    gen.run(max_workers=1)

    assert (gen_base.out['cf_mean'][0] == gen.out['cf_mean'][0]).all()
    assert (gen_base.out['ghi_mean'][0] == gen.out['ghi_mean'][0]).all()
    assert np.allclose(gen_base.out['cf_profile'][:, 0],
                       gen.out['cf_profile'][:, 0])

    assert (gen_base.out['cf_mean'][1:] < gen.out['cf_mean'][1:]).all()
    assert (gen_base.out['ghi_mean'][1:] < gen.out['ghi_mean'][1:]).all()
    mask = (gen_base.out['cf_profile'][:, 1:] <= gen.out['cf_profile'][:, 1:])
    assert (mask.sum() / mask.size) > 0.99

    bc_df = pd.DataFrame({SiteDataField.GID: np.arange(100),
                          'method': 'lin_irrad', 'scalar': 1, 'adder': -1500})
    gen = Gen('pvwattsv7', points, sam_files, res_file, sites_per_worker=1,
              output_request=output_request, bias_correct=bc_df)
    gen.run(max_workers=2)
    for arr in gen.out.values():
        assert (arr == 0).all()


def test_ac_outputs():
    """Test reV pvwattsv8 AC outputs"""
    baseline_cf_mean = np.array([0.1517, 0.1518, 0.1570])

    year = 2012
    rev2_points = slice(0, 3)
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(year)
    sam_files = TESTDATADIR + "/SAM/i_pvwattsv8.json"

    output_request = (
        "cf_mean",
        "cf_mean_ac",
        "cf_profile",
        "cf_profile_ac",
        "system_capacity",
        "system_capacity_ac",
        "ac",
        "dc",
        "dc_ac_ratio",
    )

    # run reV 2.0 generation
    gen = Gen(
        "pvwattsv8",
        rev2_points,
        sam_files,
        res_file,
        sites_per_worker=1,
        output_request=output_request,
    )
    gen.run(max_workers=1)

    msg = "PVWattsv8 cf_mean results {} did not match baseline: {}".format(
        gen.out["cf_mean"], baseline_cf_mean
    )
    assert np.allclose(
        gen.out["cf_mean"], baseline_cf_mean, rtol=0.005, atol=0.0
    ), msg

    for req in ['cf_mean', ]:
        ac_req = '{}_ac'.format(req)
        assert req in gen.out
        assert ac_req in gen.out
        assert (gen.out[req] <= gen.out[ac_req]).all()

    assert (gen.out["dc"] >= gen.out["ac"]).all()
    assert np.allclose(
        gen.out["system_capacity"] / gen.out["dc_ac_ratio"],
        gen.out["system_capacity_ac"],
    )

    assert not np.isclose(gen.out["cf_profile"], 1).any()
    assert np.isclose(gen.out["cf_profile_ac"], 1).any()


def test_pv_regional_mults():
    """Test reV pvwattsv8 regional multiplier outputs"""

    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_2012.h5"
    points = pd.DataFrame({"gid": [0, 1, 2],
                           "capital_cost_multiplier": [0.6, 0.8, 1]})

    output_request = ("cf_mean", "cf_mean_ac", "cf_profile", "cf_profile_ac",
                      "system_capacity", "system_capacity_ac", "ac", "dc",
                      "dc_ac_ratio", "lcoe_fcr")

    with open(TESTDATADIR + "/SAM/i_pvwattsv8.json") as fh:
        sam_config = json.load(fh)

    costs = {"capital_cost": 39767200, "fixed_charge_rate": 0.096,
             "fixed_operating_cost": 260000, "variable_operating_cost": 10,
             "system_capacity": 20_000}
    sam_config.update(costs)
    sam_files = {"default": sam_config}

    # run reV 2.0 generation
    gen = Gen("pvwattsv8", points, sam_files, res_file, sites_per_worker=1,
              output_request=output_request)
    gen.run(max_workers=1)

    # SAM config unchanged
    assert sam_config["capital_cost"] == 39767200
    assert sam_config["fixed_operating_cost"] == 260000
    assert sam_config["variable_operating_cost"] == 10

    assert np.allclose(gen.out["base_fixed_operating_cost"], 260000)
    assert np.allclose(gen.out["fixed_operating_cost"], 260000)
    assert np.allclose(gen.out["base_variable_operating_cost"], 10)
    assert np.allclose(gen.out["variable_operating_cost"], 10)

    assert np.allclose(gen.out["capital_cost_multiplier"],
                       points["capital_cost_multiplier"])

    assert np.allclose(gen.out["base_capital_cost"], 39767200)
    cc = sam_config["capital_cost"] * points["capital_cost_multiplier"]
    assert np.allclose(gen.out["capital_cost"], cc)

    cost = (cc * sam_config["fixed_charge_rate"]
            + sam_config["fixed_operating_cost"])
    aep = gen.out["cf_mean"] * sam_config["system_capacity"] / 1000 * 8760
    lcoe_truth = cost / aep + sam_config["variable_operating_cost"] * 1000

    assert np.allclose(gen.out["lcoe_fcr"], lcoe_truth)


@pytest.mark.parametrize(
    "bad_input",
    [
        ("latitude", -91),
        ("latitude", 91),
        ("longitude", -181),
        ("longitude", 181),
    ],
)
def test_bad_loc_inputs(bad_input):
    """Test that error is thrown for bad lat/lon inputs."""
    res_file = TESTDATADIR + "/nsrdb/ri_100_nsrdb_{}.h5".format(2012)
    sam_files = TESTDATADIR + "/SAM/naris_pv_1axis_inv13.json"
    col, val = bad_input
    with tempfile.TemporaryDirectory() as td:
        res_file_bad = os.path.join(td, "res_bad_loc.h5")
        shutil.copy(res_file, res_file_bad)
        with Outputs(res_file_bad, mode="a") as f:
            meta = f.meta.copy()
            meta.loc[0, col] = val
            f.meta = meta

        gen = Gen(
            "pvwattsv8",
            slice(0, 3),
            sam_files,
            res_file_bad,
            output_request=("cf_profile",),
            sites_per_worker=50,
        )
        with pytest.raises(ValueError):
            gen.run(max_workers=1)


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
