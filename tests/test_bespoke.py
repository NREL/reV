# -*- coding: utf-8 -*-
"""reV bespoke wind plant optimization tests"""

import copy
import json
import os
import shutil
import tempfile
import traceback
from glob import glob

import h5py
import numpy as np
import pandas as pd
import pytest
from gaps.collection import Collector
from rex import Resource

from reV import TESTDATADIR
from reV.econ.utilities import lcoe_fcr
from reV.bespoke.bespoke import BespokeSinglePlant, BespokeWindPlants
from reV.bespoke.place_turbines import PlaceTurbines, _compute_nn_conn_dist
from reV.cli import main
from reV.handlers.outputs import Outputs
from reV.losses.power_curve import PowerCurveLossesMixin
from reV.losses.scheduled import ScheduledLossesMixin
from reV.SAM.generation import WindPower
from reV.supply_curve.supply_curve import SupplyCurve
from reV.supply_curve.tech_mapping import TechMapping
from reV.utilities import (
    ModuleName,
    SiteDataField,
    SupplyCurveField,
)

pytest.importorskip("shapely")

SAM = os.path.join(TESTDATADIR, "SAM/i_windpower.json")
EXCL = os.path.join(TESTDATADIR, "ri_exclusions/ri_exclusions.h5")
RES = os.path.join(TESTDATADIR, "wtk/ri_100_wtk_{}.h5")
TM_DSET = "techmap_wtk_ri_100"
AGG_DSET = ("cf_mean", "cf_profile")

DATA_LAYERS = {
    "pct_slope": {"dset": "ri_srtm_slope", "method": "mean", "fpath": EXCL},
    "reeds_region": {
        "dset": "ri_reeds_regions",
        "method": "mode",
        "fpath": EXCL,
    },
    "padus": {"dset": "ri_padus", "method": "mode", "fpath": EXCL},
}

# Note that this differs from the
EXCL_DICT = {
    "ri_srtm_slope": {"include_range": (None, 5), "exclude_nodata": False},
    "ri_padus": {"exclude_values": [1], "exclude_nodata": False},
    "ri_reeds_regions": {
        "include_range": (None, 400),
        "exclude_nodata": False,
    },
}

with open(SAM) as f:
    SAM_SYS_INPUTS = json.load(f)

SAM_SYS_INPUTS["wind_farm_wake_model"] = 2
SAM_SYS_INPUTS["wind_farm_losses_percent"] = 0
del SAM_SYS_INPUTS["wind_resource_filename"]
TURB_RATING = np.max(SAM_SYS_INPUTS["wind_turbine_powercurve_powerout"])
SAM_CONFIGS = {"default": SAM_SYS_INPUTS}


CAP_COST_FUN = (
    "140 * system_capacity "
    "* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))"
)
FOC_FUN = (
    "60 * system_capacity "
    "* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))"
)
VOC_FUN = "3"
BOS_FUN = '0'
OBJECTIVE_FUNCTION = (
    "(0.0975 * capital_cost + fixed_operating_cost) "
    "/ aep + variable_operating_cost"
)
EXPECTED_META_COLUMNS = ["gid",  # needed for H5 collection to work properly
                         SupplyCurveField.SC_POINT_GID,
                         SupplyCurveField.TURBINE_X_COORDS,
                         SupplyCurveField.TURBINE_Y_COORDS,
                         SupplyCurveField.POSSIBLE_X_COORDS,
                         SupplyCurveField.POSSIBLE_Y_COORDS,
                         SupplyCurveField.N_TURBINES,
                         SupplyCurveField.RES_GIDS,
                         SupplyCurveField.MEAN_RES,
                         SupplyCurveField.CAPACITY_AC_MW,
                         SupplyCurveField.CAPACITY_DC_MW,
                         SupplyCurveField.MEAN_CF_AC,
                         SupplyCurveField.MEAN_CF_DC,
                         SupplyCurveField.WAKE_LOSSES,
                         SupplyCurveField.SC_POINT_ANNUAL_ENERGY_MWH,
                         SupplyCurveField.EOS_MULT,
                         SupplyCurveField.REG_MULT,
                         SupplyCurveField.COST_BASE_CC_USD_PER_AC_MW,
                         SupplyCurveField.COST_SITE_CC_USD_PER_AC_MW,
                         SupplyCurveField.COST_BASE_FOC_USD_PER_AC_MW,
                         SupplyCurveField.COST_SITE_FOC_USD_PER_AC_MW,
                         SupplyCurveField.COST_BASE_VOC_USD_PER_AC_MWH,
                         SupplyCurveField.COST_SITE_VOC_USD_PER_AC_MWH,
                         SupplyCurveField.FIXED_CHARGE_RATE,
                         SupplyCurveField.INCLUDED_AREA,
                         SupplyCurveField.INCLUDED_AREA_CAPACITY_DENSITY,
                         SupplyCurveField.CONVEX_HULL_AREA,
                         SupplyCurveField.CONVEX_HULL_CAPACITY_DENSITY,
                         SupplyCurveField.FULL_CELL_CAPACITY_DENSITY,
                         SupplyCurveField.BESPOKE_AEP,
                         SupplyCurveField.BESPOKE_OBJECTIVE,
                         SupplyCurveField.BESPOKE_CAPITAL_COST,
                         SupplyCurveField.BESPOKE_FIXED_OPERATING_COST,
                         SupplyCurveField.BESPOKE_VARIABLE_OPERATING_COST,
                         SupplyCurveField.BESPOKE_BALANCE_OF_SYSTEM_COST]


def test_turbine_placement(gid=33):
    """Test turbine placement with zero available area."""
    output_request = ("system_capacity", "cf_mean", "cf_profile")
    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format("*")

        sam_sys_inputs = copy.deepcopy(SAM_SYS_INPUTS)
        sam_sys_inputs["fixed_operating_cost_multiplier"] = 2
        sam_sys_inputs["variable_operating_cost_multiplier"] = 5

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 sam_sys_inputs,
                                 OBJECTIVE_FUNCTION,
                                 CAP_COST_FUN,
                                 FOC_FUN,
                                 VOC_FUN,
                                 '10 * nn_conn_dist_m',
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        place_optimizer = bsp.plant_optimizer
        assert place_optimizer.turbine_x is None
        assert place_optimizer.turbine_y is None
        assert place_optimizer.nturbs is None
        assert place_optimizer.capacity is None
        assert place_optimizer.area is None
        assert place_optimizer.aep is None
        assert place_optimizer.capital_cost is None
        assert place_optimizer.fixed_operating_cost is None
        assert place_optimizer.variable_operating_cost is None
        assert place_optimizer.balance_of_system_cost is None
        assert place_optimizer.objective is None
        place_optimizer.place_turbines(max_time=5)

        assert place_optimizer.nturbs == len(place_optimizer.turbine_x)
        assert (
            place_optimizer.capacity
            == place_optimizer.nturbs * place_optimizer.turbine_capacity
        )
        assert place_optimizer.area == place_optimizer.full_polygons.area / 1e6
        assert (
            place_optimizer.capacity_density
            == place_optimizer.capacity / place_optimizer.area / 1e3
        )

        place_optimizer.wind_plant["wind_farm_xCoordinates"] = (
            place_optimizer.turbine_x
        )
        place_optimizer.wind_plant["wind_farm_yCoordinates"] = (
            place_optimizer.turbine_y
        )
        place_optimizer.wind_plant["system_capacity"] = (
            place_optimizer.capacity
        )
        place_optimizer.wind_plant.assign_inputs()
        place_optimizer.wind_plant.execute()

        assert (
            place_optimizer.aep == place_optimizer.wind_plant.annual_energy()
        )

        # pylint: disable=W0641
        system_capacity = place_optimizer.capacity
        # pylint: disable=W0641
        aep = place_optimizer.aep
        # pylint: disable=W0123
        capital_cost = eval(CAP_COST_FUN, globals(), locals())
        fixed_operating_cost = eval(FOC_FUN, globals(), locals()) * 2
        variable_operating_cost = eval(VOC_FUN, globals(), locals()) * 5
        balance_of_system_cost = 10 * _compute_nn_conn_dist(
            place_optimizer.turbine_x, place_optimizer.turbine_y
        )
        # pylint: disable=W0123
        assert place_optimizer.objective == eval(
            OBJECTIVE_FUNCTION, globals(), locals()
        )
        assert place_optimizer.capital_cost == capital_cost
        assert place_optimizer.fixed_operating_cost == fixed_operating_cost
        assert (place_optimizer.variable_operating_cost
                == variable_operating_cost)
        assert place_optimizer.balance_of_system_cost == balance_of_system_cost

        bsp.close()


def test_zero_area(gid=33):
    """Test turbine placement with zero available area."""
    output_request = ("system_capacity", "cf_mean", "cf_profile")

    objective_function = (
        "(0.0975 * capital_cost + fixed_operating_cost) "
        "/ (aep + 1E-6) + variable_operating_cost"
    )

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format("*")

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function, CAP_COST_FUN,
                                 FOC_FUN, VOC_FUN, BOS_FUN,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        optimizer = bsp.plant_optimizer
        optimizer.include_mask = np.zeros_like(optimizer.include_mask)
        optimizer.place_turbines(max_time=5)

        # pylint: disable=W0123
        assert len(optimizer.turbine_x) == 0
        assert len(optimizer.turbine_y) == 0
        assert optimizer.nturbs == 0
        assert optimizer.capacity == 0
        assert optimizer.area == 0
        assert optimizer.capacity_density == 0
        assert optimizer.objective == eval(VOC_FUN)
        assert optimizer.capital_cost == 0
        assert optimizer.fixed_operating_cost == 0

        bsp.close()


def test_correct_turb_location(gid=33):
    """Test turbine location is reported correctly."""
    output_request = ("system_capacity", "cf_mean", "cf_profile")

    objective_function = (
        "(0.0975 * capital_cost + fixed_operating_cost) "
        "/ (aep + 1E-6) + variable_operating_cost"
    )

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format("*")

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function, CAP_COST_FUN,
                                 FOC_FUN, VOC_FUN, BOS_FUN,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        include_mask = np.zeros_like(bsp.include_mask)
        include_mask[1, -2] = 1
        pt = PlaceTurbines(bsp.wind_plant_pd, bsp.objective_function,
                           bsp.capital_cost_function,
                           bsp.fixed_operating_cost_function,
                           bsp.variable_operating_cost_function,
                           bsp.balance_of_system_cost_function,
                           include_mask, pixel_side_length=90,
                           min_spacing=45)

        pt.define_exclusions()
        pt.initialize_packing()

        assert pt.x_locations[0] == 62 * 90
        assert pt.y_locations[0] == 62 * 90

        bsp.close()


def test_correct_turb_chb(gid=33):
    """Test turbine convex hull buffered correctly"""
    output_request = ("system_capacity", "cf_mean", "cf_profile")

    objective_function = (
        "(0.0975 * capital_cost + fixed_operating_cost) "
        "/ (aep + 1E-6) + variable_operating_cost"
    )

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format("*")

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function, CAP_COST_FUN,
                                 FOC_FUN, VOC_FUN, BOS_FUN,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        include_mask = np.zeros_like(bsp.include_mask)
        include_mask[1, -2] = 1
        pt = PlaceTurbines(bsp.wind_plant_pd, bsp.objective_function,
                           bsp.capital_cost_function,
                           bsp.fixed_operating_cost_function,
                           bsp.variable_operating_cost_function,
                           bsp.balance_of_system_cost_function,
                           include_mask, pixel_side_length=90,
                           min_spacing=45)

        pt.define_exclusions()
        pt.initialize_packing()
        pt.optimized_design_variables = pt.x_locations >= 0

        pt_buffered = PlaceTurbines(bsp.wind_plant_pd, bsp.objective_function,
                                    bsp.capital_cost_function,
                                    bsp.fixed_operating_cost_function,
                                    bsp.variable_operating_cost_function,
                                    bsp.balance_of_system_cost_function,
                                    include_mask, pixel_side_length=90,
                                    min_spacing=45, convex_hull_buffer=100)

        pt_buffered.define_exclusions()
        pt_buffered.initialize_packing()
        pt_buffered.optimized_design_variables = pt.x_locations >= 0

        assert pt.convex_hull_area > 0
        assert pt_buffered.convex_hull_area > pt.convex_hull_area

        bsp.close()


def test_packing_algorithm(gid=33):
    """Test turbine placement with zero available area."""
    output_request = ()
    cap_cost_fun = ""
    foc_fun = ""
    voc_fun = ""
    bos_fun = ""
    objective_function = ""
    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format("*")

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )
        bsp = BespokeSinglePlant(
            gid,
            excl_fp,
            res_fp,
            TM_DSET,
            SAM_SYS_INPUTS,
            objective_function,
            cap_cost_fun,
            foc_fun,
            voc_fun,
            bos_fun,
            ga_kwargs={"max_time": 5},
            excl_dict=EXCL_DICT,
            output_request=output_request,
        )

        optimizer = bsp.plant_optimizer
        optimizer.define_exclusions()
        optimizer.initialize_packing()

        assert len(optimizer.x_locations) < 165
        assert len(optimizer.x_locations) > 145
        assert np.sum(optimizer.include_mask) == (
            optimizer.safe_polygons.area / (optimizer.pixel_side_length**2)
        )

        bsp.close()


def test_bespoke_points():
    """Test the bespoke points input options"""
    # pylint: disable=W0612
    points = pd.DataFrame(
        {
            SiteDataField.GID: [33, 34, 35],
            SiteDataField.CONFIG: ["default"] * 3,
        }
    )
    pp = BespokeWindPlants._parse_points(points, {"default": SAM})
    assert len(pp) == 3
    for gid in pp.gids:
        assert pp[gid][0] == "default"

    points = pd.DataFrame({SiteDataField.GID: [33, 34, 35]})
    pp = BespokeWindPlants._parse_points(points, {"default": SAM})
    assert len(pp) == 3
    assert SiteDataField.CONFIG in pp.df.columns
    for gid in pp.gids:
        assert pp[gid][0] == "default"


def test_single(gid=33):
    """Test a single wind plant bespoke optimization run"""
    output_request = ("system_capacity", "cf_mean", "cf_profile")
    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format("*")

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 OBJECTIVE_FUNCTION, CAP_COST_FUN,
                                 FOC_FUN, VOC_FUN, BOS_FUN,
                                 ga_kwargs={'max_time': 5},
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        out = bsp.run_plant_optimization()
        out = bsp.run_wind_plant_ts()

        assert "cf_profile-2012" in out
        assert "cf_profile-2013" in out
        assert "cf_mean-2012" in out
        assert "cf_mean-2013" in out
        assert "cf_mean-means" in out
        assert "annual_energy-2012" in out
        assert "annual_energy-2013" in out
        assert "annual_energy-means" in out

        assert (
            TURB_RATING * bsp.meta[SupplyCurveField.N_TURBINES].values[0]
            == out["system_capacity"]
        )
        x_coords = json.loads(
            bsp.meta[SupplyCurveField.TURBINE_X_COORDS].values[0]
        )
        y_coords = json.loads(
            bsp.meta[SupplyCurveField.TURBINE_Y_COORDS].values[0]
        )
        assert bsp.meta[SupplyCurveField.N_TURBINES].values[0] == len(x_coords)
        assert bsp.meta[SupplyCurveField.N_TURBINES].values[0] == len(y_coords)

        for y in (2012, 2013):
            cf = out[f"cf_profile-{y}"]
            assert cf.min() == 0
            assert cf.max() == 1
            assert np.allclose(cf.mean(), out[f"cf_mean-{y}"])

        # simple windpower obj for comparison
        wp_sam_config = bsp.sam_sys_inputs
        wp_sam_config["wind_farm_wake_model"] = 0
        wp_sam_config["wake_int_loss"] = 0
        wp_sam_config["wind_farm_xCoordinates"] = [0]
        wp_sam_config["wind_farm_yCoordinates"] = [0]
        wp_sam_config["system_capacity"] = TURB_RATING
        res_df = bsp.res_df[(bsp.res_df.index.year == 2012)].copy()
        wp = WindPower(
            res_df, bsp.meta, wp_sam_config, output_request=bsp._out_req
        )
        wp.run()

        # make sure the wind resource was loaded correctly
        res_ideal = np.array(wp["wind_resource_data"]["data"])
        bsp_2012 = bsp.wind_plant_ts[2012]
        res_bsp = np.array(bsp_2012["wind_resource_data"]["data"])
        ws_ideal = res_ideal[:, 2]
        ws_bsp = res_bsp[:, 2]
        assert np.allclose(ws_ideal, ws_bsp)

        # make sure that the zero-losses analysis has greater CF
        cf_bespoke = out["cf_profile-2012"]
        cf_ideal = wp.outputs["cf_profile"]
        diff = cf_ideal - cf_bespoke
        assert all(diff > -0.00001)
        assert diff.mean() > 0.02

        bsp.close()


def test_extra_outputs(gid=33):
    """Test running bespoke single farm optimization with lcoe requests"""
    output_request = ("system_capacity", "cf_mean", "cf_profile", "lcoe_fcr")

    objective_function = (
        "(fixed_charge_rate * capital_cost + fixed_operating_cost) "
        "/ aep + variable_operating_cost"
    )

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format("*")

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )

        with pytest.raises(KeyError):
            bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                     SAM_SYS_INPUTS,
                                     objective_function, CAP_COST_FUN,
                                     FOC_FUN, VOC_FUN, BOS_FUN,
                                     ga_kwargs={'max_time': 5},
                                     excl_dict=EXCL_DICT,
                                     output_request=output_request,
                                     )

        sam_sys_inputs = copy.deepcopy(SAM_SYS_INPUTS)
        sam_sys_inputs["fixed_charge_rate"] = 0.0975
        test_eos_cap = 200_000
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 sam_sys_inputs,
                                 objective_function, CAP_COST_FUN,
                                 FOC_FUN, VOC_FUN, BOS_FUN,
                                 ga_kwargs={'max_time': 5},
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 data_layers=copy.deepcopy(DATA_LAYERS),
                                 eos_mult_baseline_cap_mw=test_eos_cap * 1e-3
                                 )

        out = bsp.run_plant_optimization()
        out = bsp.run_wind_plant_ts()
        bsp.agg_data_layers()

        assert "lcoe_fcr-2012" in out
        assert "lcoe_fcr-2013" in out
        assert "lcoe_fcr-means" in out

        assert SupplyCurveField.CAPACITY_AC_MW in bsp.meta
        assert SupplyCurveField.MEAN_CF_AC in bsp.meta
        assert SupplyCurveField.MEAN_LCOE in bsp.meta

        assert "pct_slope" in bsp.meta
        assert "reeds_region" in bsp.meta
        assert "padus" in bsp.meta

        out = None
        data_layers = copy.deepcopy(DATA_LAYERS)
        for layer in data_layers:
            data_layers[layer].pop("fpath", None)

        for layer in data_layers:
            assert "fpath" not in data_layers[layer]

        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 sam_sys_inputs,
                                 objective_function, CAP_COST_FUN,
                                 FOC_FUN, VOC_FUN, BOS_FUN,
                                 ga_kwargs={'max_time': 5},
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 data_layers=data_layers,
                                 )

        out = bsp.run_plant_optimization()
        out = bsp.run_wind_plant_ts()
        bsp.agg_data_layers()

        assert "lcoe_fcr-2012" in out
        assert "lcoe_fcr-2013" in out
        assert "lcoe_fcr-means" in out

        assert SupplyCurveField.CAPACITY_AC_MW in bsp.meta
        assert SupplyCurveField.MEAN_CF_AC in bsp.meta
        assert SupplyCurveField.MEAN_LCOE in bsp.meta

        assert "pct_slope" in bsp.meta
        assert "reeds_region" in bsp.meta
        assert "padus" in bsp.meta

        assert SupplyCurveField.EOS_MULT in bsp.meta
        assert SupplyCurveField.REG_MULT in bsp.meta
        assert np.allclose(bsp.meta[SupplyCurveField.REG_MULT], 1)

        n_turbs = round(test_eos_cap / TURB_RATING)
        test_eos_cap_kw = n_turbs * TURB_RATING
        baseline_cost = (
            140
            * test_eos_cap_kw
            * np.exp(-test_eos_cap_kw / 1e5 * 0.1 + (1 - 0.1))
        )
        eos_mult = (
            bsp.plant_optimizer.capital_cost
            / bsp.plant_optimizer.capacity
            / (baseline_cost / test_eos_cap_kw)
        )
        assert np.allclose(bsp.meta[SupplyCurveField.EOS_MULT], eos_mult)

        bsp.close()


def test_bespoke():
    """Test bespoke optimization with multiple plants, parallel processing, and
    file output."""
    output_request = (
        "system_capacity",
        "cf_mean",
        "cf_profile",
        "extra_unused_data",
        "winddirection",
        "windspeed",
        "ws_mean",
    )

    with tempfile.TemporaryDirectory() as td:
        out_fpath_request = os.path.join(td, "wind")
        out_fpath_truth = os.path.join(td, "wind_bespoke.h5")
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format("*")
        # both 33 and 35 are included, 37 is fully excluded
        points = pd.DataFrame(
            {
                SiteDataField.GID: [33, 35],
                SiteDataField.CONFIG: ["default"] * 2,
                "extra_unused_data": [0, 42],
                "capital_cost_multiplier": [1, 2],
                "fixed_operating_cost_multiplier": [3, 4],
                "variable_operating_cost_multiplier": [5, 6]
            }
        )
        fully_excluded_points = pd.DataFrame(
            {
                SiteDataField.GID: [37],
                SiteDataField.CONFIG: ["default"],
                "extra_unused_data": [0],
            }
        )

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )
        sam_configs = copy.deepcopy(SAM_CONFIGS)
        sam_configs["default"]["fixed_charge_rate"] = 0.0975

        # test no outputs
        with pytest.warns(UserWarning) as record:
            assert not os.path.exists(out_fpath_truth)
            bsp = BespokeWindPlants(excl_fp, res_fp, TM_DSET,
                                    OBJECTIVE_FUNCTION, CAP_COST_FUN,
                                    FOC_FUN, VOC_FUN, BOS_FUN,
                                    fully_excluded_points,
                                    sam_configs, ga_kwargs={'max_time': 5},
                                    excl_dict=EXCL_DICT,
                                    output_request=output_request)
            test_fpath = bsp.run(max_workers=2, out_fpath=out_fpath_request)
            assert out_fpath_truth == test_fpath
            assert "points are excluded" in str(record[0].message)

        assert not os.path.exists(out_fpath_truth)
        bsp = BespokeWindPlants(excl_fp, res_fp, TM_DSET, OBJECTIVE_FUNCTION,
                                CAP_COST_FUN, FOC_FUN, VOC_FUN, BOS_FUN,
                                points, sam_configs, ga_kwargs={'max_time': 5},
                                excl_dict=EXCL_DICT,
                                output_request=output_request)
        test_fpath = bsp.run(max_workers=2, out_fpath=out_fpath_request)
        assert out_fpath_truth == test_fpath
        assert os.path.exists(out_fpath_truth)
        with Resource(out_fpath_truth) as f:
            meta = f.meta.reset_index()
            assert len(meta) <= len(points)
            for col in EXPECTED_META_COLUMNS:
                assert col in meta

            dsets_1d = (
                "system_capacity",
                "cf_mean-2012",
                "annual_energy-2012",
                "cf_mean-means",
                "extra_unused_data-2012",
                "ws_mean",
                "annual_wake_loss_internal_percent-means"
            )
            for dset in dsets_1d:
                assert dset in list(f)
                assert isinstance(f[dset], np.ndarray)
                assert len(f[dset].shape) == 1
                assert len(f[dset]) == len(meta)
                assert f[dset].any()  # not all zeros

            assert np.allclose(meta[SupplyCurveField.MEAN_RES], f["ws_mean"],
                               atol=0.01)
            assert np.allclose(
                f["annual_energy-means"] / 1000,
                meta[SupplyCurveField.SC_POINT_ANNUAL_ENERGY_MWH]
            )

            dsets_2d = (
                "cf_profile-2012",
                "cf_profile-2013",
                "windspeed-2012",
                "windspeed-2013",
            )
            for dset in dsets_2d:
                assert dset in list(f)
                assert isinstance(f[dset], np.ndarray)
                assert len(f[dset].shape) == 2
                assert len(f[dset]) == 8760
                assert f[dset].shape[1] == len(meta)
                assert f[dset].any()  # not all zeros

        assert not np.allclose(
            meta[SupplyCurveField.COST_SITE_CC_USD_PER_AC_MW],
            meta[SupplyCurveField.COST_BASE_CC_USD_PER_AC_MW])
        assert not np.allclose(
            meta[SupplyCurveField.COST_SITE_FOC_USD_PER_AC_MW],
            meta[SupplyCurveField.COST_BASE_FOC_USD_PER_AC_MW])
        assert not np.allclose(
            meta[SupplyCurveField.COST_SITE_VOC_USD_PER_AC_MWH],
            meta[SupplyCurveField.COST_BASE_VOC_USD_PER_AC_MWH])

        fcr = meta[SupplyCurveField.FIXED_CHARGE_RATE]
        cap_cost = (meta[SupplyCurveField.COST_SITE_CC_USD_PER_AC_MW]
                    * meta[SupplyCurveField.CAPACITY_AC_MW])
        foc = (meta[SupplyCurveField.COST_SITE_FOC_USD_PER_AC_MW]
               * meta[SupplyCurveField.CAPACITY_AC_MW])
        voc = meta[SupplyCurveField.COST_SITE_VOC_USD_PER_AC_MWH]
        aep = meta[SupplyCurveField.SC_POINT_ANNUAL_ENERGY_MWH]
        lcoe_site = lcoe_fcr(fcr, cap_cost, foc, aep, voc)

        cap_cost = (meta[SupplyCurveField.COST_BASE_CC_USD_PER_AC_MW]
                    * meta[SupplyCurveField.CAPACITY_AC_MW]
                    * meta[SupplyCurveField.REG_MULT]
                    * meta[SupplyCurveField.EOS_MULT])
        foc = (meta[SupplyCurveField.COST_BASE_FOC_USD_PER_AC_MW]
               * meta[SupplyCurveField.CAPACITY_AC_MW]
               * np.array([3, 4]))
        voc = (meta[SupplyCurveField.COST_BASE_VOC_USD_PER_AC_MWH]
               * np.array([5, 6]))
        lcoe_base = lcoe_fcr(fcr, cap_cost, foc, aep, voc)

        assert np.allclose(lcoe_site, lcoe_base)
        assert np.allclose(meta[SupplyCurveField.REG_MULT], [1, 2])

        out_fpath_pre = os.path.join(td, 'bespoke_out_pre.h5')
        bsp = BespokeWindPlants(excl_fp, res_fp, TM_DSET, OBJECTIVE_FUNCTION,
                                CAP_COST_FUN, FOC_FUN, VOC_FUN, BOS_FUN,
                                points, SAM_CONFIGS, ga_kwargs={'max_time': 1},
                                excl_dict=EXCL_DICT,
                                output_request=output_request,
                                pre_load_data=True)
        bsp.run(max_workers=1, out_fpath=out_fpath_pre)

        with Resource(out_fpath_truth) as f1, Resource(out_fpath_pre) as f2:
            assert np.allclose(
                f1["winddirection-2012"], f2["winddirection-2012"]
            )
            assert np.allclose(f1["ws_mean"], f2["ws_mean"])


def test_collect_bespoke():
    """Test the collection of multiple chunked bespoke files."""
    with tempfile.TemporaryDirectory() as td:
        source_dir = os.path.join(TESTDATADIR, "bespoke/")
        source_pattern = source_dir + "/test_bespoke*.h5"
        source_fps = sorted(glob(source_pattern))
        assert len(source_fps) > 1

        h5_file = os.path.join(td, "collection.h5")
        collector = Collector(h5_file, source_pattern, None)
        collector.collect("cf_profile-2012")

        with Resource(h5_file) as fout:
            meta = fout.meta.rename(columns=SupplyCurveField.map_from_legacy())
            assert all(
                meta[SupplyCurveField.SC_POINT_GID].values
                == sorted(meta[SupplyCurveField.SC_POINT_GID].values)
            )
            ti = fout.time_index
            assert len(ti) == 8760
            assert "time_index-2012" in fout
            assert "time_index-2013" in fout
            data = fout["cf_profile-2012"]

        for fp in source_fps:
            with Resource(fp) as source:
                src_meta = source.meta.rename(
                    columns=SupplyCurveField.map_from_legacy())
                assert all(
                    np.isin(
                        src_meta[SupplyCurveField.SC_POINT_GID].values,
                        meta[SupplyCurveField.SC_POINT_GID].values,
                    )
                )
                for isource, gid in enumerate(
                    src_meta[SupplyCurveField.SC_POINT_GID].values
                ):
                    gid_mask = (meta[SupplyCurveField.SC_POINT_GID].values
                                == gid)
                    iout = np.where(gid_mask)[0]
                    truth = source["cf_profile-2012", :, isource].flatten()
                    test = data[:, iout].flatten()
                    assert np.allclose(truth, test)


def test_consistent_eval_namespace(gid=33):
    """Test that all the same variables are available for every eval."""
    output_request = ("system_capacity", "cf_mean", "cf_profile")
    cap_cost_fun = "2000"
    foc_fun = "0"
    voc_fun = "0"
    bos_fun = "0"
    objective_function = (
        "n_turbines + id(self.wind_plant) "
        "+ system_capacity + capital_cost + aep"
    )
    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format("*")

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )
        bsp = BespokeSinglePlant(
            gid,
            excl_fp,
            res_fp,
            TM_DSET,
            SAM_SYS_INPUTS,
            objective_function,
            cap_cost_fun,
            foc_fun,
            voc_fun,
            bos_fun,
            ga_kwargs={"max_time": 5},
            excl_dict=EXCL_DICT,
            output_request=output_request,
        )
        _ = bsp.run_plant_optimization()

        assert (bsp.meta[SupplyCurveField.BESPOKE_AEP].values[0]
                == bsp.plant_optimizer.aep)
        assert (bsp.meta[SupplyCurveField.BESPOKE_OBJECTIVE].values[0]
                == bsp.plant_optimizer.objective)

        bsp.close()


def test_bespoke_supply_curve():
    """Test supply curve compute from a bespoke output that acts as the
    traditional reV-sc-aggregation output table."""

    bespoke_sample_fout = os.path.join(
        TESTDATADIR, "bespoke/test_bespoke_node00.h5"
    )

    normal_path = os.path.join(TESTDATADIR, "sc_out/baseline_agg_summary.csv")
    normal_sc_points = pd.read_csv(normal_path)
    normal_sc_points = normal_sc_points.rename(
        SupplyCurveField.map_from_legacy(), axis=1
    )

    with tempfile.TemporaryDirectory() as td:
        bespoke_sc_fp = os.path.join(td, "bespoke_out.h5")
        shutil.copy(bespoke_sample_fout, bespoke_sc_fp)
        with h5py.File(bespoke_sc_fp, "a") as f:
            del f["meta"]
        with Outputs(bespoke_sc_fp, mode="a") as f:
            bespoke_meta = normal_sc_points.copy()
            bespoke_meta = bespoke_meta.drop(SupplyCurveField.SC_GID, axis=1)
            f.meta = bespoke_meta

        # this is basically copied from test_supply_curve_compute.py
        trans_tables = [
            os.path.join(TESTDATADIR, "trans_tables", f"costs_RI_{cap}MW.csv")
            for cap in [100, 200, 400, 1000]
        ]

        sc = SupplyCurve(bespoke_sc_fp, trans_tables)
        sc_full = sc.full_sort(fcr=0.1, avail_cap_frac=0.1)

        assert SupplyCurveField.SC_GID in sc_full

        assert all(
            gid in sc_full[SupplyCurveField.SC_GID]
            for gid in normal_sc_points[SupplyCurveField.SC_GID]
        )
        for _, inp_row in normal_sc_points.iterrows():
            sc_gid = inp_row[SupplyCurveField.SC_GID]
            assert sc_gid in sc_full[SupplyCurveField.SC_GID]
            test_ind = np.where(sc_full[SupplyCurveField.SC_GID] == sc_gid)[0]
            assert len(test_ind) == 1
            test_row = sc_full.iloc[test_ind]
            assert (
                test_row[SupplyCurveField.TOTAL_LCOE].values[0]
                > inp_row[SupplyCurveField.MEAN_LCOE]
            )

    fpath_baseline = os.path.join(TESTDATADIR, "sc_out/sc_full_lc.csv")
    sc_baseline = pd.read_csv(fpath_baseline)
    sc_baseline = sc_baseline.rename(
        columns=SupplyCurveField.map_from_legacy()
    )
    assert np.allclose(sc_baseline[SupplyCurveField.TOTAL_LCOE],
                       sc_full[SupplyCurveField.TOTAL_LCOE])


def test_bespoke_wind_plant_with_power_curve_losses():
    """Test bespoke ``wind_plant`` with power curve losses."""
    output_request = ("system_capacity", "cf_mean", "cf_profile")
    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format("*")

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )
        bsp = BespokeSinglePlant(33, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 OBJECTIVE_FUNCTION,
                                 CAP_COST_FUN,
                                 FOC_FUN,
                                 VOC_FUN,
                                 BOS_FUN,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        optimizer = bsp.plant_optimizer
        optimizer.wind_plant["wind_farm_xCoordinates"] = [1000, -1000]
        optimizer.wind_plant["wind_farm_yCoordinates"] = [1000, -1000]
        cap = 2 * optimizer.turbine_capacity
        optimizer.wind_plant["system_capacity"] = cap

        optimizer.wind_plant.assign_inputs()
        optimizer.wind_plant.execute()
        aep = optimizer.wind_plant["annual_energy"]
        bsp.close()

        sam_inputs = copy.deepcopy(SAM_SYS_INPUTS)
        sam_inputs[PowerCurveLossesMixin.POWER_CURVE_CONFIG_KEY] = {
            "target_losses_percent": 10,
            "transformation": "exponential_stretching",
        }
        bsp = BespokeSinglePlant(33, excl_fp, res_fp, TM_DSET,
                                 sam_inputs,
                                 OBJECTIVE_FUNCTION,
                                 CAP_COST_FUN,
                                 FOC_FUN,
                                 VOC_FUN,
                                 BOS_FUN,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request)
        optimizer2 = bsp.plant_optimizer
        optimizer2.wind_plant["wind_farm_xCoordinates"] = [1000, -1000]
        optimizer2.wind_plant["wind_farm_yCoordinates"] = [1000, -1000]
        cap = 2 * optimizer2.turbine_capacity
        optimizer2.wind_plant["system_capacity"] = cap

        optimizer2.wind_plant.assign_inputs()
        optimizer2.wind_plant.execute()
        aep_losses = optimizer2.wind_plant["annual_energy"]
        bsp.close()

    assert aep > aep_losses, f"{aep}, {aep_losses}"

    err_msg = "{:0.3f} != 0.9".format(aep_losses / aep)
    assert np.isclose(aep_losses / aep, 0.9), err_msg


def test_bespoke_run_with_icing_cutoff():
    """Test bespoke run with icing cutoff enabled."""
    output_request = ("system_capacity", "cf_mean", "cf_profile")
    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format("*")

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )
        bsp = BespokeSinglePlant(
            33,
            excl_fp,
            res_fp,
            TM_DSET,
            SAM_SYS_INPUTS,
            OBJECTIVE_FUNCTION,
            CAP_COST_FUN,
            FOC_FUN,
            VOC_FUN,
            BOS_FUN,
            ga_kwargs={"max_time": 5},
            excl_dict=EXCL_DICT,
            output_request=output_request,
        )

        out = bsp.run_plant_optimization()
        out = bsp.run_wind_plant_ts()
        bsp.close()

        sam_inputs_ice = copy.deepcopy(SAM_SYS_INPUTS)
        sam_inputs_ice["en_icing_cutoff"] = 1
        sam_inputs_ice["en_low_temp_cutoff"] = 1
        sam_inputs_ice["icing_cutoff_rh"] = 90  # High values to ensure diff
        sam_inputs_ice["icing_cutoff_temp"] = 10
        sam_inputs_ice["low_temp_cutoff"] = 0
        bsp = BespokeSinglePlant(
            33,
            excl_fp,
            res_fp,
            TM_DSET,
            sam_inputs_ice,
            OBJECTIVE_FUNCTION,
            CAP_COST_FUN,
            FOC_FUN,
            VOC_FUN,
            BOS_FUN,
            ga_kwargs={"max_time": 5},
            excl_dict=EXCL_DICT,
            output_request=output_request,
        )

        out_ice = bsp.run_plant_optimization()
        out_ice = bsp.run_wind_plant_ts()
        bsp.close()

    ae_dsets = [
        "annual_energy-2012",
        "annual_energy-2013",
        "annual_energy-means",
    ]
    for dset in ae_dsets:
        assert not np.isclose(out[dset], out_ice[dset])
        assert out[dset] > out_ice[dset]


def test_bespoke_run_with_power_curve_losses():
    """Test bespoke run with power curve losses."""
    output_request = ("system_capacity", "cf_mean", "cf_profile",
                      "annual_energy", "annual_gross_energy",
                      "annual_wake_loss_internal_percent",
                      "annual_wake_loss_internal_kWh",
                      "annual_wake_loss_total_percent")

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format("*")

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )
        bsp = BespokeSinglePlant(
            33,
            excl_fp,
            res_fp,
            TM_DSET,
            SAM_SYS_INPUTS,
            OBJECTIVE_FUNCTION,
            CAP_COST_FUN,
            FOC_FUN,
            VOC_FUN,
            BOS_FUN,
            ga_kwargs={"max_time": 5},
            excl_dict=EXCL_DICT,
            output_request=output_request,
        )

        out = bsp.run_plant_optimization()
        out = bsp.run_wind_plant_ts()
        bsp.close()

        sam_inputs = copy.deepcopy(SAM_SYS_INPUTS)
        sam_inputs[PowerCurveLossesMixin.POWER_CURVE_CONFIG_KEY] = {
            "target_losses_percent": 10,
            "transformation": "exponential_stretching",
        }
        bsp = BespokeSinglePlant(
            33,
            excl_fp,
            res_fp,
            TM_DSET,
            sam_inputs,
            OBJECTIVE_FUNCTION,
            CAP_COST_FUN,
            FOC_FUN,
            VOC_FUN,
            BOS_FUN,
            ga_kwargs={"max_time": 5},
            excl_dict=EXCL_DICT,
            output_request=output_request,
        )

        out_losses = bsp.run_plant_optimization()
        out_losses = bsp.run_wind_plant_ts()
        bsp.close()

    ae_dsets = [
        "annual_gross_energy-2012",
        "annual_gross_energy-2013",
        "annual_gross_energy-means",
    ]
    for dset in ae_dsets:
        assert not np.isclose(out[dset], out_losses[dset])
        err_msg = "{:0.3f} != 0.9".format(out_losses[dset] / out[dset])
        assert np.isclose(out_losses[dset] / out[dset], 0.9), err_msg


def test_bespoke_run_with_scheduled_losses():
    """Test bespoke run with scheduled losses."""
    output_request = ("system_capacity", "cf_mean", "cf_profile")
    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format("*")

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )
        bsp = BespokeSinglePlant(33, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 OBJECTIVE_FUNCTION, CAP_COST_FUN,
                                 FOC_FUN, VOC_FUN, BOS_FUN,
                                 ga_kwargs={'max_time': 5},
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request)

        out = bsp.run_plant_optimization()
        out = bsp.run_wind_plant_ts()
        bsp.close()

        sam_inputs = copy.deepcopy(SAM_SYS_INPUTS)
        sam_inputs[ScheduledLossesMixin.OUTAGE_CONFIG_KEY] = [
            {
                "name": "Environmental",
                "count": 115,
                "duration": 2,
                "percentage_of_capacity_lost": 100,
                "allowed_months": [
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                ],
            }
        ]
        sam_inputs["adjust_timeindex"] = [0] * 8760  # only needed for testing
        output_request = ("system_capacity", "cf_mean", "cf_profile",
                          "adjust_timeindex")

        bsp = BespokeSinglePlant(33, excl_fp, res_fp, TM_DSET,
                                 sam_inputs,
                                 OBJECTIVE_FUNCTION,
                                 CAP_COST_FUN,
                                 FOC_FUN,
                                 VOC_FUN,
                                 BOS_FUN,
                                 ga_kwargs={'max_time': 5},
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request)

        out_losses = bsp.run_plant_optimization()
        out_losses = bsp.run_wind_plant_ts()
        bsp.close()

    ae_dsets = [
        "annual_energy-2012",
        "annual_energy-2013",
        "annual_energy-means",
    ]
    for dset in ae_dsets:
        assert not np.isclose(out[dset], out_losses[dset])
        assert out[dset] > out_losses[dset]

    assert not np.allclose(
        out_losses["adjust_timeindex-2012"],
        out_losses["adjust_timeindex-2013"]
    )


def test_bespoke_aep_is_zero_if_no_turbines_placed():
    """Test that bespoke aep output is zero if no turbines placed."""
    output_request = ("system_capacity", "cf_mean", "cf_profile")

    objective_function = "aep"

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format("*")

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )
        bsp = BespokeSinglePlant(33, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function,
                                 CAP_COST_FUN,
                                 FOC_FUN,
                                 VOC_FUN,
                                 BOS_FUN,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        optimizer = bsp.plant_optimizer
        optimizer.define_exclusions()
        optimizer.initialize_packing()
        optimizer.wind_plant["wind_farm_xCoordinates"] = []
        optimizer.wind_plant["wind_farm_yCoordinates"] = []
        optimizer.wind_plant["system_capacity"] = 0

        aep = optimizer.optimization_objective(x=[])
        bsp.close()

    assert aep == 0


def test_bespoke_prior_run():
    """Test a follow-on bespoke timeseries generation run based on a prior
    plant layout optimization.

    Also added another minor test with extrapolation of t/p datasets from a
    single vertical level (e.g., with Sup3rCC data)
    """
    sam_sys_inputs = copy.deepcopy(SAM_SYS_INPUTS)
    sam_sys_inputs["fixed_charge_rate"] = 0.096
    sam_configs = {"default": sam_sys_inputs}
    output_request = (
        "system_capacity",
        "cf_mean",
        "cf_profile",
        "extra_unused_data",
        "lcoe_fcr",
    )
    with tempfile.TemporaryDirectory() as td:
        out_fpath1 = os.path.join(td, "bespoke_out2.h5")
        out_fpath2 = os.path.join(td, "bespoke_out1.h5")
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))

        # test t/p extrapolation from single level (e.g. with Sup3rCC data)
        del_dsets = ("pressure_100m", "pressure_200m", "temperature_80m")
        for y in (2012, 2013):
            with h5py.File(res_fp.format(y), "a") as h5:
                for dset in del_dsets:
                    del h5[dset]

        res_fp_all = res_fp.format("*")
        res_fp_2013 = res_fp.format("2013")

        # gids 33 and 35 are included, 37 is fully excluded
        points = pd.DataFrame(
            {
                SiteDataField.GID: [33],
                SiteDataField.CONFIG: ["default"],
                "extra_unused_data": [42],
            }
        )

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )

        assert not os.path.exists(out_fpath1)
        assert not os.path.exists(out_fpath2)

        bsp = BespokeWindPlants(excl_fp, res_fp_all, TM_DSET,
                                OBJECTIVE_FUNCTION, CAP_COST_FUN,
                                FOC_FUN, VOC_FUN, BOS_FUN, points, sam_configs,
                                ga_kwargs={'max_time': 1}, excl_dict=EXCL_DICT,
                                output_request=output_request)
        bsp.run(max_workers=1, out_fpath=out_fpath1)

        assert os.path.exists(out_fpath1)
        assert not os.path.exists(out_fpath2)

        bsp = BespokeWindPlants(excl_fp, res_fp_2013, TM_DSET,
                                OBJECTIVE_FUNCTION, CAP_COST_FUN,
                                FOC_FUN, VOC_FUN, BOS_FUN, points, sam_configs,
                                ga_kwargs={'max_time': 1}, excl_dict=EXCL_DICT,
                                output_request=output_request,
                                prior_run=out_fpath1)
        bsp.run(max_workers=1, out_fpath=out_fpath2)
        assert os.path.exists(out_fpath2)

        with Resource(out_fpath1) as f1:
            meta1 = f1.meta
            data1 = {k: f1[k] for k in f1.dsets}

        with Resource(out_fpath2) as f2:
            meta2 = f2.meta
            data2 = {k: f2[k] for k in f2.dsets}

        cols = [
            SupplyCurveField.TURBINE_X_COORDS,
            SupplyCurveField.TURBINE_Y_COORDS,
            SupplyCurveField.CAPACITY_AC_MW,
            SupplyCurveField.N_GIDS,
            SupplyCurveField.GID_COUNTS,
            SupplyCurveField.RES_GIDS,
            SupplyCurveField.N_TURBINES,
            SupplyCurveField.EOS_MULT,
            SupplyCurveField.REG_MULT,
            SupplyCurveField.INCLUDED_AREA,
            SupplyCurveField.INCLUDED_AREA_CAPACITY_DENSITY,
            SupplyCurveField.CONVEX_HULL_AREA,
            SupplyCurveField.CONVEX_HULL_CAPACITY_DENSITY,
            SupplyCurveField.FULL_CELL_CAPACITY_DENSITY,
            SupplyCurveField.COST_BASE_CC_USD_PER_AC_MW,
            SupplyCurveField.COST_SITE_CC_USD_PER_AC_MW,
            SupplyCurveField.COST_BASE_FOC_USD_PER_AC_MW,
            SupplyCurveField.COST_SITE_FOC_USD_PER_AC_MW,
            SupplyCurveField.COST_BASE_VOC_USD_PER_AC_MWH,
            SupplyCurveField.COST_SITE_VOC_USD_PER_AC_MWH,
            SupplyCurveField.FIXED_CHARGE_RATE,
            SupplyCurveField.BESPOKE_AEP,
            SupplyCurveField.BESPOKE_OBJECTIVE,
            SupplyCurveField.BESPOKE_CAPITAL_COST,
            SupplyCurveField.BESPOKE_FIXED_OPERATING_COST,
            SupplyCurveField.BESPOKE_VARIABLE_OPERATING_COST,
            SupplyCurveField.BESPOKE_BALANCE_OF_SYSTEM_COST,
        ]
        pd.testing.assert_frame_equal(meta1[cols], meta2[cols])

        # multi-year means should not match the 2nd run with 2013 only.
        # 2013 values should match exactly
        assert not np.allclose(data1["cf_mean-means"], data2["cf_mean-means"])
        assert np.allclose(data1["cf_mean-2013"], data2["cf_mean-2013"],
                           rtol=1e-6, atol=1e-9)

        assert not np.allclose(data1["annual_energy-means"],
                               data2["annual_energy-means"])
        assert np.allclose(data1["annual_energy-2013"],
                           data2["annual_energy-2013"],rtol=1e-6, atol=1e-9)

        assert not np.allclose(
            data1["annual_wake_loss_internal_percent-means"],
            data2["annual_wake_loss_internal_percent-means"]
        )
        assert np.allclose(data1["annual_wake_loss_internal_percent-2013"],
                           data2["annual_wake_loss_internal_percent-2013"],
                           rtol=1e-6, atol=1e-9)


def test_gid_map():
    """Test bespoke run with resource gid map - used to swap resource gids with
    new resource data files for example so you can run forecasted resource with
    the same spatial configuration."""

    output_request = (
        "system_capacity",
        "cf_mean",
        "cf_profile",
        "extra_unused_data",
        "winddirection",
        "ws_mean",
    )
    with tempfile.TemporaryDirectory() as td:
        out_fpath1 = os.path.join(td, "bespoke_out2.h5")
        out_fpath2 = os.path.join(td, "bespoke_out1.h5")
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2013), res_fp.format(2013))

        res_fp_2013 = res_fp.format("2013")

        # gids 33 and 35 are included, 37 is fully excluded
        points = pd.DataFrame(
            {
                SiteDataField.GID: [33],
                SiteDataField.CONFIG: ["default"],
                "extra_unused_data": [42],
            }
        )

        gid_map = pd.DataFrame(
            {SiteDataField.GID: [3, 4, 13, 12, 11, 10, 9]}
        )
        new_gid = 50
        gid_map["gid_map"] = new_gid
        fp_gid_map = os.path.join(td, "gid_map.csv")
        gid_map.to_csv(fp_gid_map)

        TechMapping.run(
            excl_fp, RES.format(2013), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )

        assert not os.path.exists(out_fpath1)
        assert not os.path.exists(out_fpath2)

        bsp = BespokeWindPlants(excl_fp, res_fp_2013, TM_DSET,
                                OBJECTIVE_FUNCTION, CAP_COST_FUN,
                                FOC_FUN, VOC_FUN, BOS_FUN, points, SAM_CONFIGS,
                                ga_kwargs={'max_time': 1}, excl_dict=EXCL_DICT,
                                output_request=output_request)
        bsp.run(max_workers=1, out_fpath=out_fpath1)

        assert os.path.exists(out_fpath1)
        assert not os.path.exists(out_fpath2)

        bsp = BespokeWindPlants(excl_fp, res_fp_2013, TM_DSET,
                                OBJECTIVE_FUNCTION, CAP_COST_FUN,
                                FOC_FUN, VOC_FUN, BOS_FUN, points, SAM_CONFIGS,
                                ga_kwargs={'max_time': 1}, excl_dict=EXCL_DICT,
                                output_request=output_request,
                                gid_map=fp_gid_map)
        bsp.run(max_workers=1, out_fpath=out_fpath2)
        assert os.path.exists(out_fpath2)

        with Resource(out_fpath1) as f1:
            meta1 = f1.meta
            data1 = {k: f1[k] for k in f1.dsets}

        with Resource(out_fpath2) as f2:
            meta2 = f2.meta
            data2 = {k: f2[k] for k in f2.dsets}

        hh = SAM_CONFIGS["default"]["wind_turbine_hub_ht"]
        with Resource(res_fp_2013) as f3:
            ws = f3[f"windspeed_{hh}m", :, new_gid]

        cols = [
            SupplyCurveField.N_GIDS,
            SupplyCurveField.GID_COUNTS,
            SupplyCurveField.RES_GIDS,
        ]
        pd.testing.assert_frame_equal(meta1[cols], meta2[cols])

        assert not np.allclose(data1["cf_mean-2013"], data2["cf_mean-2013"])
        assert not np.allclose(data1["ws_mean"], data2["ws_mean"], atol=0.2)
        assert np.allclose(ws.mean(), data2["ws_mean"], atol=0.01)

        out_fpath_pre = os.path.join(td, 'bespoke_out_pre.h5')
        bsp = BespokeWindPlants(excl_fp, res_fp_2013, TM_DSET,
                                OBJECTIVE_FUNCTION, CAP_COST_FUN,
                                FOC_FUN, VOC_FUN, BOS_FUN, points, SAM_CONFIGS,
                                ga_kwargs={'max_time': 1}, excl_dict=EXCL_DICT,
                                output_request=output_request,
                                gid_map=fp_gid_map, pre_load_data=True)
        bsp.run(max_workers=1, out_fpath=out_fpath_pre)

        with Resource(out_fpath2) as f1, Resource(out_fpath_pre) as f2:
            assert np.allclose(
                f1["winddirection-2013"], f2["winddirection-2013"]
            )
            assert np.allclose(f1["ws_mean"], f2["ws_mean"])


def test_bespoke_bias_correct():
    """Test bespoke run with bias correction on windspeed data."""
    output_request = (
        "system_capacity",
        "cf_mean",
        "cf_profile",
        "extra_unused_data",
        "ws_mean",
    )
    with tempfile.TemporaryDirectory() as td:
        out_fpath1 = os.path.join(td, "bespoke_out2.h5")
        out_fpath2 = os.path.join(td, "bespoke_out1.h5")
        res_fp = os.path.join(td, "ri_100_wtk_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2013), res_fp.format(2013))

        res_fp_2013 = res_fp.format("2013")

        # gids 33 and 35 are included, 37 is fully excluded
        points = pd.DataFrame(
            {
                SiteDataField.GID: [33],
                SiteDataField.CONFIG: ["default"],
                "extra_unused_data": [42],
            }
        )

        # intentionally leaving out WTK gid 13 which only has 5 included 90m
        # pixels in order to check that this is dynamically patched.
        bias_correct = pd.DataFrame(
            {SiteDataField.GID: [3, 4, 12, 11, 10, 9]}
        )
        bias_correct["method"] = "lin_ws"
        bias_correct["scalar"] = 0.5
        fp_bc = os.path.join(td, "bc.csv")
        bias_correct.to_csv(fp_bc)

        TechMapping.run(
            excl_fp, RES.format(2013), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )

        assert not os.path.exists(out_fpath1)
        assert not os.path.exists(out_fpath2)

        bsp = BespokeWindPlants(excl_fp, res_fp_2013, TM_DSET,
                                OBJECTIVE_FUNCTION, CAP_COST_FUN,
                                FOC_FUN, VOC_FUN, BOS_FUN, points, SAM_CONFIGS,
                                ga_kwargs={'max_time': 1}, excl_dict=EXCL_DICT,
                                output_request=output_request)
        bsp.run(max_workers=1, out_fpath=out_fpath1)

        assert os.path.exists(out_fpath1)
        assert not os.path.exists(out_fpath2)

        bsp = BespokeWindPlants(excl_fp, res_fp_2013, TM_DSET,
                                OBJECTIVE_FUNCTION, CAP_COST_FUN,
                                FOC_FUN, VOC_FUN, BOS_FUN, points, SAM_CONFIGS,
                                ga_kwargs={'max_time': 1}, excl_dict=EXCL_DICT,
                                output_request=output_request,
                                bias_correct=fp_bc)
        bsp.run(max_workers=1, out_fpath=out_fpath2)
        assert os.path.exists(out_fpath2)

        with Resource(out_fpath1) as f1:
            meta1 = f1.meta
            data1 = {k: f1[k] for k in f1.dsets}

        with Resource(out_fpath2) as f2:
            meta2 = f2.meta
            data2 = {k: f2[k] for k in f2.dsets}

        cols = [
            SupplyCurveField.N_GIDS,
            SupplyCurveField.GID_COUNTS,
            SupplyCurveField.RES_GIDS,
        ]
        pd.testing.assert_frame_equal(meta1[cols], meta2[cols])

        assert data1["cf_mean-2013"] * 0.5 > data2["cf_mean-2013"]
        assert np.allclose(data1["ws_mean"] * 0.5, data2["ws_mean"], atol=0.01)


def test_cli(runner, clear_loggers):
    """Test bespoke CLI"""
    output_request = (
        "system_capacity",
        "cf_mean",
        "cf_profile",
        "winddirection",
        "windspeed",
        "ws_mean",
    )

    with tempfile.TemporaryDirectory() as td:
        dirname = os.path.basename(td)
        fn_out = "{}_{}.h5".format(dirname, ModuleName.BESPOKE)
        out_fpath = os.path.join(td, fn_out)

        res_fp_1 = os.path.join(td, "ri_100_wtk_{}.h5")
        res_fp_2 = os.path.join(td, "another_name_{}.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp_1.format(2012))
        shutil.copy(RES.format(2013), res_fp_2.format(2013))
        res_fp = [res_fp_1.format(2012), res_fp_2.format("*")]

        TechMapping.run(
            excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1,
            sc_resolution=2560
        )

        config = {
            "log_directory": td,
            "log_level": "INFO",
            "execution_control": {
                "option": "local",
                "max_workers": 2,
            },
            "excl_fpath": excl_fp,
            "res_fpath": res_fp,
            "tm_dset": TM_DSET,
            "objective_function": OBJECTIVE_FUNCTION,
            "capital_cost_function": CAP_COST_FUN,
            "fixed_operating_cost_function": FOC_FUN,
            "variable_operating_cost_function": VOC_FUN,
            "balance_of_system_cost_function": "0",
            "project_points": [33, 35],
            "sam_files": SAM_CONFIGS,
            "min_spacing": '5x',
            "ga_kwargs": {'max_time': 5},
            "output_request": output_request,
            "ws_bins": (0, 20, 5),
            "wd_bins": (0, 360, 45),
            "excl_dict": EXCL_DICT,
            "area_filter_kernel": "queen",
            "min_area": None,
            "resolution": 64,
            "excl_area": None,
            "data_layers": None,
            "pre_extract_inclusions": False,
            "prior_run": None,
            "gid_map": None,
            "bias_correct": None,
            "pre_load_data": False,
        }
        config_path = os.path.join(td, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        assert not os.path.exists(out_fpath)
        result = runner.invoke(main, ["bespoke", "-c", config_path])
        if result.exit_code != 0:
            msg = "Failed with error {}".format(
                traceback.print_exception(*result.exc_info)
            )
            raise RuntimeError(msg)

        assert os.path.exists(out_fpath)

        with Resource(out_fpath) as f:
            meta = f.meta
            assert len(meta) == 2
            assert SupplyCurveField.SC_POINT_GID in meta
            assert SupplyCurveField.TURBINE_X_COORDS in meta
            assert SupplyCurveField.TURBINE_Y_COORDS in meta
            assert "possible_x_coords" in meta
            assert "possible_y_coords" in meta
            assert SupplyCurveField.RES_GIDS in meta

            dsets_1d = (
                "system_capacity",
                "cf_mean-2012",
                "annual_energy-2012",
                "cf_mean-means",
                "ws_mean",
            )
            for dset in dsets_1d:
                assert dset in list(f)
                assert isinstance(f[dset], np.ndarray)
                assert len(f[dset].shape) == 1
                assert len(f[dset]) == len(meta)
                assert f[dset].any()  # not all zeros

            dsets_2d = (
                "cf_profile-2012",
                "cf_profile-2013",
                "windspeed-2012",
                "windspeed-2013",
            )
            for dset in dsets_2d:
                assert dset in list(f)
                assert isinstance(f[dset], np.ndarray)
                assert len(f[dset].shape) == 2
                assert len(f[dset]) == 8760
                assert f[dset].shape[1] == len(meta)
                assert f[dset].any()  # not all zeros

        clear_loggers()


def test_bespoke_5min_sample():
    """Sample a 5min resource dataset for 60min outputs in bespoke"""
    output_request = (
        "system_capacity",
        "cf_mean",
        "cf_profile",
        "extra_unused_data",
        "winddirection",
        "windspeed",
        "ws_mean",
    )
    tm_dset = "test_wtk_5min"

    with tempfile.TemporaryDirectory() as td:
        out_fpath = os.path.join(td, "wind_bespoke.h5")
        excl_fp = os.path.join(td, "ri_exclusions.h5")
        shutil.copy(EXCL, excl_fp)
        res_fp = os.path.join(TESTDATADIR, "wtk/wtk_2010_*m.h5")

        points = pd.DataFrame(
            {
                SiteDataField.GID: [33, 35],
                SiteDataField.CONFIG: ["default"] * 2,
                "extra_unused_data": [0, 42],
            }
        )
        sam_sys_inputs = copy.deepcopy(SAM_SYS_INPUTS)
        sam_sys_inputs["time_index_step"] = 12
        sam_configs = {"default": sam_sys_inputs}

        # hack techmap because 5min data only has 10 wind resource pixels
        with h5py.File(excl_fp, "a") as excl_file:
            arr = np.random.choice(10, size=excl_file["latitude"].shape)
            excl_file.create_dataset(name=tm_dset, data=arr)

        bsp = BespokeWindPlants(excl_fp, res_fp, tm_dset, OBJECTIVE_FUNCTION,
                                CAP_COST_FUN, FOC_FUN, VOC_FUN, BOS_FUN,
                                points, sam_configs, ga_kwargs={'max_time': 5},
                                excl_dict=EXCL_DICT,
                                output_request=output_request)
        _ = bsp.run(max_workers=1, out_fpath=out_fpath)

        with Resource(out_fpath) as f:
            assert len(f.meta) == 2
            assert len(f) == 8760
            assert len(f["cf_profile-2010"]) == 8760
            assert len(f["time_index-2010"]) == 8760
            assert len(f["windspeed-2010"]) == 8760
            assert len(f["winddirection-2010"]) == 8760
