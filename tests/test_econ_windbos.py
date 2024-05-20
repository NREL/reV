# -*- coding: utf-8 -*-
"""
PyTest file for SAM/reV econ Wind Balance of System cost model.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from reV import TESTDATADIR
from reV.econ.econ import Econ
from reV.generation.generation import Gen
from reV.SAM.windbos import WindBos
from reV.utilities import SiteDataField

RTOL = 0.000001
ATOL = 0.001
OUT_DIR = os.path.join(TESTDATADIR, "ri_wind_reV2/")

DEFAULTS = {
    "tech_model": "windbos",
    "financial_model": "none",
    "machine_rating": 1500.0,
    "rotor_diameter": 77.0,
    "hub_height": 80.0,
    "number_of_turbines": 32,
    "interconnect_voltage": 137.0,
    "distance_to_interconnect": 5.0,
    "site_terrain": 0,
    "turbine_layout": 1,
    "soil_condition": 0,
    "construction_time": 6,
    "om_building_size": 3000.0,
    "quantity_test_met_towers": 1,
    "quantity_permanent_met_towers": 1,
    "weather_delay_days": 6,
    "crane_breakdowns": 2,
    "access_road_entrances": 2,
    "turbine_capital_cost": 0.0,
    "turbine_cost_per_kw": 1094.0,  # not in SDK tool, but important
    "tower_top_mass": 88.0,  # tonnes, tuned to default foundation cost
    "delivery_assist_required": 0,
    "pad_mount_transformer_required": 1,
    "new_switchyard_required": 1,
    "rock_trenching_required": 10,
    "mv_thermal_backfill": 0.0,
    "mv_overhead_collector": 0.0,
    "performance_bond": 0,
    "contingency": 3.0,
    "warranty_management": 0.02,
    "sales_and_use_tax": 5.0,
    "sales_tax_basis": 0.0,
    "overhead": 5.0,
    "profit_margin": 5.0,
    "development_fee": 5.0,  # Millions of dollars
    "turbine_transportation": 0.0,
}


# baseline single owner + windbos results using 2012 RI WTK data.
BASELINE = {
    "project_return_aftertax_npv": np.array(
        [
            7876459.5,
            7875551.5,
            7874505.0,
            7875270.0,
            7875349.5,
            7872819.5,
            7871078.5,
            7871352.5,
            7871153.5,
            7869134.5,
        ]
    ),
    "lcoe_real": np.array(
        [
            71.007614,
            69.21741,
            67.24552,
            68.67675,
            68.829605,
            64.257965,
            61.39551,
            61.83265,
            61.51415,
            58.43599,
        ]
    ),
    "lcoe_nom": np.array(
        [
            89.433525,
            87.17878,
            84.6952,
            86.497826,
            86.69034,
            80.932396,
            77.327156,
            77.87773,
            77.476585,
            73.59966,
        ]
    ),
    "flip_actual_irr": np.array(
        [
            10.999977,
            10.999978,
            10.999978,
            10.999978,
            10.999978,
            10.999978,
            10.999979,
            10.999979,
            10.999979,
            10.99998,
        ]
    ),
    "total_installed_cost": np.array(10 * [88892234.91311586]),
    "turbine_cost": np.array(10 * [52512000.0]),
    "sales_tax_cost": np.array(10 * [0.0]),
    "bos_cost": np.array(10 * [36380234.91311585]),
}


# baseline single owner + windbos results when sweeping sales tax basis
BASELINE_SITE_BOS = {
    "total_installed_cost": np.array(
        [88892230.0, 88936680.0, 88981130.0, 89025576.0, 89070020.0]
    ),
    "turbine_cost": np.array(5 * [52512000.0]),
    "sales_tax_cost": np.array(
        [0.0, 44446.117, 88892.234, 133338.36, 177784.47]
    ),
    "bos_cost": np.array(5 * [36380234.91311585]),
}


def test_sam_windbos():
    """Test SAM SSC from dict with windbos"""
    from PySAM.PySSC import ssc_sim_from_dict

    out = ssc_sim_from_dict(DEFAULTS)

    tcost = (
        (DEFAULTS["turbine_cost_per_kw"] + DEFAULTS["turbine_capital_cost"])
        * DEFAULTS["machine_rating"]
        * DEFAULTS["number_of_turbines"]
    )
    total_installed_cost = tcost + out["project_total_budgeted_cost"]

    assert np.allclose(total_installed_cost, 88892240.00, atol=ATOL, rtol=RTOL)


def test_rev_windbos():
    """Test baseline windbos calc with single owner defaults"""
    fpath = TESTDATADIR + "/SAM/i_singleowner_windbos.json"
    with open(fpath) as f:
        inputs = json.load(f)
    wb = WindBos(inputs)
    assert np.allclose(wb.turbine_cost, 52512000.00, atol=ATOL, rtol=RTOL)
    assert np.allclose(wb.bos_cost, 36380236.00, atol=ATOL, rtol=RTOL)
    assert np.allclose(
        wb.total_installed_cost, 88892240.00, atol=ATOL, rtol=RTOL
    )


def test_standalone_json():
    """Test baseline windbos calc with standalone json file"""
    fpath = TESTDATADIR + "/SAM/i_singleowner_windbos.json"
    with open(fpath) as f:
        inputs = json.load(f)
    wb1 = WindBos(inputs)
    fpath = TESTDATADIR + "/SAM/i_windbos.json"
    with open(fpath) as f:
        inputs = json.load(f)
    wb2 = WindBos(inputs)

    for k, v in wb1.output.items():
        assert v == wb2.output[k]


def test_rev_windbos_perf_bond():
    """Test windbos calc with performance bonds"""
    fpath = TESTDATADIR + "/SAM/i_singleowner_windbos.json"
    with open(fpath) as f:
        inputs = json.load(f)
    inputs["performance_bond"] = 10.0
    wb = WindBos(inputs)
    assert np.allclose(wb.turbine_cost, 52512000.00, atol=ATOL, rtol=RTOL)
    assert np.allclose(wb.bos_cost, 36686280.00, atol=ATOL, rtol=RTOL)
    assert np.allclose(
        wb.total_installed_cost, 89198280.00, atol=ATOL, rtol=RTOL
    )


def test_rev_windbos_transport():
    """Test windbos calc with turbine transport costs"""
    fpath = TESTDATADIR + "/SAM/i_singleowner_windbos.json"
    with open(fpath) as f:
        inputs = json.load(f)
    inputs["turbine_transportation"] = 100.0
    wb = WindBos(inputs)
    assert np.allclose(wb.turbine_cost, 52512000.00, atol=ATOL, rtol=RTOL)
    assert np.allclose(wb.bos_cost, 37720412.00, atol=ATOL, rtol=RTOL)
    assert np.allclose(
        wb.total_installed_cost, 90232416.00, atol=ATOL, rtol=RTOL
    )


def test_rev_windbos_sales():
    """Test windbos calc with turbine transport costs"""
    fpath = TESTDATADIR + "/SAM/i_singleowner_windbos.json"
    with open(fpath) as f:
        inputs = json.load(f)
    inputs["sales_tax_basis"] = 5.0
    wb = WindBos(inputs)
    assert np.allclose(wb.turbine_cost, 52512000.00, atol=ATOL, rtol=RTOL)
    assert np.allclose(wb.bos_cost, 36380236.00, atol=ATOL, rtol=RTOL)
    assert np.allclose(
        wb.total_installed_cost, 89114464.00, atol=ATOL, rtol=RTOL
    )


def test_run_gen_econ(points=slice(0, 10), year=2012, max_workers=1):
    """Test full reV2 gen->econ pipeline with windbos inputs and benchmark
    against baseline results."""
    with tempfile.TemporaryDirectory() as td:
        # get full file paths.
        sam_files = os.path.join(TESTDATADIR, "SAM/i_singleowner_windbos.json")
        res_file = os.path.join(
            TESTDATADIR, "wtk/ri_100_wtk_{}.h5".format(year)
        )
        fn_gen = "windbos_generation_{}.h5".format(year)
        cf_file = os.path.join(td, fn_gen)

        # run reV 2.0 generation
        gen = Gen(
            "windpower",
            points,
            sam_files,
            res_file,
            output_request=('cf_mean',),
            sites_per_worker=3,
        )
        gen.run(max_workers=max_workers, out_fpath=cf_file)

        econ_outs = (
            "lcoe_nom",
            "lcoe_real",
            "flip_actual_irr",
            "project_return_aftertax_npv",
            "total_installed_cost",
            "turbine_cost",
            "sales_tax_cost",
            "bos_cost",
        )
        e = Econ(
            points,
            sam_files,
            cf_file,
            site_data=None,
            output_request=econ_outs,
            sites_per_worker=3,
        )
        e.run(max_workers=max_workers)

        for k in econ_outs:
            msg = "Failed for {}".format(k)
            test = np.allclose(e.out[k], BASELINE[k], atol=ATOL, rtol=RTOL)
            assert test, msg

        return e


def test_run_bos(points=slice(0, 5), max_workers=1):
    """Test full reV2 gen->econ pipeline with windbos inputs and benchmark
    against baseline results."""

    # get full file paths.
    sam_files = TESTDATADIR + "/SAM/i_singleowner_windbos.json"
    site_data = pd.DataFrame(
        {SiteDataField.GID: range(5), "sales_tax_basis": range(5)}
    )

    econ_outs = (
        "total_installed_cost",
        "turbine_cost",
        "sales_tax_cost",
        "bos_cost",
    )
    e = Econ(
        points,
        sam_files,
        None,
        site_data=site_data,
        output_request=econ_outs,
        sites_per_worker=3,
    )
    e.run(max_workers=max_workers)

    for k in econ_outs:
        check = np.allclose(
            e.out[k], BASELINE_SITE_BOS[k], atol=ATOL, rtol=RTOL
        )
        msg = "Failed for {}".format(k)
        assert check, msg

    return e


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
