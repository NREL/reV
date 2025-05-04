# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
PyTest file for reV LCOE economies of scale
"""
import os
import shutil
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest
from rex import Resource

from reV import TESTDATADIR
from reV.econ.economies_of_scale import EconomiesOfScale
from reV.generation.generation import Gen
from reV.supply_curve.sc_aggregation import SupplyCurveAggregation
from reV.utilities import SupplyCurveField

EXCL = os.path.join(TESTDATADIR, "ri_exclusions/ri_exclusions.h5")
GEN = os.path.join(TESTDATADIR, "gen_out/ri_my_pv_gen.h5")
TM_DSET = "techmap_nsrdb"
RES_CLASS_DSET = "ghi_mean-means"
RES_CLASS_BINS = [0, 4, 100]
DATA_LAYERS = {
    "pct_slope": {"dset": "ri_srtm_slope", "method": "mean"},
    "reeds_region": {"dset": "ri_reeds_regions", "method": "mode"},
    "padus": {"dset": "ri_padus", "method": "mode"},
}

EXCL_DICT = {
    "ri_srtm_slope": {"inclusion_range": (None, 5), "exclude_nodata": True},
    "ri_padus": {"exclude_values": [1], "exclude_nodata": True},
}

RTOL = 0.001


def test_pass_through_lcoe_args():
    """Test that the kwarg works to pass through LCOE input args from the SAM
    input to the reV output."""
    year = 2012
    rev2_points = slice(0, 3)
    res_file = os.path.join(TESTDATADIR, "wtk/ri_100_wtk_{}.h5".format(year))
    sam_files = os.path.join(TESTDATADIR, "SAM/i_windpower_lcoe.json")

    output_request = (
        "cf_mean",
        "lcoe_fcr",
        "system_capacity",
        "capital_cost",
        "fixed_charge_rate",
        "variable_operating_cost",
        "fixed_operating_cost",
    )

    # run reV 2.0 generation
    gen = Gen(
        "windpower",
        rev2_points,
        sam_files,
        res_file,
        sites_per_worker=1,
        output_request=output_request,
    )
    gen.run(
        max_workers=1,
    )

    checks = [x in gen.out for x in Gen.LCOE_ARGS]
    assert all(checks)
    assert "lcoe_fcr" in gen.out
    assert "cf_mean" in gen.out


@pytest.mark.parametrize("cap_eqn", [None, 1, 2, '1 / some_sc_output'])
@pytest.mark.parametrize("fixed_eqn", [None, 1, 2, '1 / some_sc_output'])
@pytest.mark.parametrize("var_eqn", [None, 1, 2, '1 / some_sc_output'])
def test_lcoe_calc_simple(cap_eqn, fixed_eqn, var_eqn):
    """Test the EconomiesOfScale LCOE calculator without cap cost scalar"""
    # from pvwattsv7 defaults
    data = {
        "aep": 35188456.00,  # kW
        "capital_cost": 53455000.00,  # $
        "foc": 360000.00,  # $
        "voc": 0.006,  # $/kWh
        "fcr": 0.096,
        "some_sc_output": 2,
    }

    if cap_eqn is None:
        cap_scalar = 1
    elif cap_eqn == '1 / some_sc_output':
        cap_scalar = 0.5
    else:
        cap_scalar = cap_eqn

    if fixed_eqn is None:
        fixed_scalar = 1
    elif fixed_eqn == '1 / some_sc_output':
        fixed_scalar = 0.5
    else:
        fixed_scalar = fixed_eqn

    if var_eqn is None:
        var_scalar = 1
    elif var_eqn == '1 / some_sc_output':
        var_scalar = 0.5
    else:
        var_scalar = var_eqn

    true_lcoe = ((data["fcr"]
                 * data["capital_cost"]
                 + data["foc"])
                 / (data["aep"] / 1000)) + data["voc"] * 1000
    true_scaled = ((data['fcr']
                    * data["capital_cost"] * cap_scalar
                    + data['foc'] * fixed_scalar)
                   / (data['aep'] / 1000)) + data["voc"] * var_scalar * 1000

    data[SupplyCurveField.MEAN_LCOE] = true_lcoe

    eos = EconomiesOfScale(data, cap_eqn=cap_eqn, fixed_eqn=fixed_eqn,
                           var_eqn=var_eqn)
    assert cap_scalar * eos.raw_capital_cost == eos.scaled_capital_cost
    assert eos.raw_capital_cost == data["capital_cost"]
    assert np.allclose(eos.raw_lcoe, true_lcoe, rtol=0.001)
    assert np.allclose(eos.scaled_lcoe, true_scaled, rtol=0.001)


@pytest.mark.parametrize("request_h5", [True, False])
def test_econ_of_scale_baseline(request_h5):
    """Test an economies of scale calculation with scalar = 1 to ensure we can
    reproduce the lcoe values
    """
    data = {
        "capital_cost": 39767200,
        "fixed_operating_cost": 260000,
        "fixed_charge_rate": 0.096,
        "system_capacity": 20000,
        "variable_operating_cost": 0.4,
    }
    h5_dsets = None
    if request_h5:
        h5_dsets = ["capital_cost", "fixed_operating_cost",
                    "fixed_charge_rate", "variable_operating_cost"]

    with tempfile.TemporaryDirectory() as td:
        gen_temp = os.path.join(td, "ri_my_pv_gen.h5")
        shutil.copy(GEN, gen_temp)

        # overwrite the LCOE values since i dont know what econ inputs
        # the original test file was run with
        with Resource(GEN) as res:
            cf = res["cf_mean-means"]

        lcoe = (1000
                * (data['fixed_charge_rate'] * data['capital_cost']
                   + data['fixed_operating_cost'])
                / (cf * data['system_capacity'] * 8760)
                + data['variable_operating_cost'])

        with h5py.File(gen_temp, "a") as res:
            res["lcoe_fcr-means"][...] = lcoe
            for k, v in data.items():
                arr = np.full(res["meta"].shape, v)
                res.create_dataset(k, res["meta"].shape, data=arr)
                res[k].attrs["scale_factor"] = 1.0

            arr = np.full(res["meta"].shape, data["capital_cost"])
            res.create_dataset("base_capital_cost",
                               res["meta"].shape, data=arr)

        out_fp_base = os.path.join(td, "base")
        base = SupplyCurveAggregation(
            EXCL,
            TM_DSET,
            excl_dict=EXCL_DICT,
            res_class_dset=RES_CLASS_DSET,
            res_class_bins=RES_CLASS_BINS,
            data_layers=DATA_LAYERS,
            gids=list(np.arange(10)),
            h5_dsets=h5_dsets,
        )
        base.run(out_fp_base, gen_fpath=gen_temp, max_workers=1)

        out_fp_sc = os.path.join(td, "sc")
        sc = SupplyCurveAggregation(
            EXCL,
            TM_DSET,
            excl_dict=EXCL_DICT,
            res_class_dset=RES_CLASS_DSET,
            res_class_bins=RES_CLASS_BINS,
            data_layers=DATA_LAYERS,
            gids=list(np.arange(10)),
            cap_cost_scale="1",
            h5_dsets=h5_dsets,
        )
        sc.run(out_fp_sc, gen_fpath=gen_temp, max_workers=1)

        base_df = pd.read_csv(out_fp_base + ".csv")
        sc_df = pd.read_csv(out_fp_sc + ".csv")
        assert np.allclose(base_df[SupplyCurveField.MEAN_LCOE],
                           sc_df[SupplyCurveField.MEAN_LCOE])
        assert (sc_df[SupplyCurveField.EOS_MULT] == 1).all()
        if "mean_capital_cost" in sc_df:
            capital_cost_per_mw = SupplyCurveField.COST_SITE_CC_USD_PER_AC_MW
            assert np.allclose(sc_df['mean_capital_cost'],
                               sc_df[capital_cost_per_mw]
                               * data["system_capacity"] / 1000)
        assert np.allclose(sc_df[SupplyCurveField.COST_BASE_CC_USD_PER_AC_MW],
                           sc_df[SupplyCurveField.COST_SITE_CC_USD_PER_AC_MW])


@pytest.mark.parametrize("voc", [0, 0.1])
@pytest.mark.parametrize("fixed_eqn", [None, 0.5])
@pytest.mark.parametrize("voc_eqn", [None, 0.1])
def test_sc_agg_econ_scale(voc, fixed_eqn, voc_eqn):
    """Test supply curve agg with LCOE scaling based on plant capacity."""
    data = {
        "capital_cost": 53455000,
        "fixed_operating_cost": 360000,
        "fixed_charge_rate": 0.096,
        "variable_operating_cost": voc,
    }

    with tempfile.TemporaryDirectory() as td:
        gen_temp = os.path.join(td, "ri_my_pv_gen.h5")
        shutil.copy(GEN, gen_temp)

        with h5py.File(gen_temp, "a") as res:
            for k, v in data.items():
                arr = np.full(res["meta"].shape, v)
                res.create_dataset(k, res["meta"].shape, data=arr)
                res[k].attrs["scale_factor"] = 1.0

        cap_eqn = (
            f"2 * np.multiply(1000, {SupplyCurveField.CAPACITY_AC_MW}) ** -0.3"
            f" * np.where(np.array([2, 5]) > 3)[0][0]"
            f" * np.where(np.array([2, 1]) == 1)[0][0]"
        )
        out_fp_base = os.path.join(td, "base")
        base = SupplyCurveAggregation(
            EXCL,
            TM_DSET,
            excl_dict=EXCL_DICT,
            res_class_dset=RES_CLASS_DSET,
            res_class_bins=RES_CLASS_BINS,
            data_layers=DATA_LAYERS,
            gids=list(np.arange(10)),
            h5_dsets=[
                "capital_cost",
                "fixed_operating_cost",
                "fixed_charge_rate",
                "variable_operating_cost"
            ],
        )
        base.run(out_fp_base, gen_fpath=gen_temp, max_workers=1)

        out_fp_sc = os.path.join(td, "sc")
        sc = SupplyCurveAggregation(
            EXCL,
            TM_DSET,
            excl_dict=EXCL_DICT,
            res_class_dset=RES_CLASS_DSET,
            res_class_bins=RES_CLASS_BINS,
            data_layers=DATA_LAYERS,
            gids=list(np.arange(10)),
            cap_cost_scale=cap_eqn,
            fixed_cost_scale=fixed_eqn,
            var_cost_scale=voc_eqn,
            h5_dsets=[
                "capital_cost",
                "fixed_operating_cost",
                "fixed_charge_rate",
                "variable_operating_cost"
            ],
        )
        sc.run(out_fp_sc, gen_fpath=gen_temp, max_workers=1)

        base_df = pd.read_csv(out_fp_base + ".csv")
        sc_df = pd.read_csv(out_fp_sc + ".csv")

        # check that econ of scale saved the raw lcoe and that it reduced all
        # of the mean lcoe values from baseline
        assert np.allclose(sc_df[SupplyCurveField.RAW_LCOE],
                           base_df[SupplyCurveField.MEAN_LCOE])
        assert all(sc_df[SupplyCurveField.MEAN_LCOE]
                   < base_df[SupplyCurveField.MEAN_LCOE])

        aep = ((sc_df['mean_fixed_charge_rate'] * sc_df['mean_capital_cost']
                + sc_df['mean_fixed_operating_cost'])
               / (sc_df[SupplyCurveField.RAW_LCOE]
                  - data['variable_operating_cost'] * 1000))

        true_raw_lcoe = ((data['fixed_charge_rate'] * data['capital_cost']
                          + data['fixed_operating_cost'])
                         / aep + data['variable_operating_cost'] * 1000)

        eval_inputs = {k: sc_df[k].values.flatten() for k in sc_df.columns}
        # pylint: disable=eval-used
        cc_scalars = eval(str(cap_eqn), globals(), eval_inputs)
        foc_scalar = fixed_eqn or 1
        voc_scalar = voc_eqn or 1
        sc_df["scalars"] = cc_scalars
        true_scaled_lcoe = (
            data["fixed_charge_rate"]
            * data["capital_cost"] * cc_scalars
            + data["fixed_operating_cost"] * foc_scalar
        ) / aep + data["variable_operating_cost"] * voc_scalar * 1000

        assert np.allclose(base_df[SupplyCurveField.EOS_MULT], 1)
        assert SupplyCurveField.FIXED_EOS_MULT not in base_df
        assert SupplyCurveField.VAR_EOS_MULT not in base_df

        assert np.allclose(cc_scalars, sc_df[SupplyCurveField.EOS_MULT])
        if fixed_eqn:
            assert np.allclose(foc_scalar,
                               sc_df[SupplyCurveField.FIXED_EOS_MULT])
        if voc_eqn:
            assert np.allclose(voc_scalar,
                               sc_df[SupplyCurveField.VAR_EOS_MULT])

        assert np.allclose(true_scaled_lcoe, sc_df[SupplyCurveField.MEAN_LCOE])
        assert np.allclose(true_raw_lcoe, sc_df[SupplyCurveField.RAW_LCOE])
        sc_df = sc_df.sort_values(SupplyCurveField.CAPACITY_AC_MW)
        if voc == 0:
            assert all(sc_df[SupplyCurveField.MEAN_LCOE].diff()[1:] < 0)
        for i in sc_df.index.values:
            if sc_df.loc[i, 'scalars'] < 1:
                assert (sc_df.loc[i, SupplyCurveField.MEAN_LCOE]
                        < sc_df.loc[i, SupplyCurveField.RAW_LCOE])
            else:
                assert (sc_df.loc[i, SupplyCurveField.MEAN_LCOE]
                        >= sc_df.loc[i, SupplyCurveField.RAW_LCOE])


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
