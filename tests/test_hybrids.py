# -*- coding: utf-8 -*-
"""reV hybrids tests."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from rex.resource import Resource

from reV import TESTDATADIR, Outputs
from reV.cli import main
from reV.hybrids import HYBRID_METHODS, Hybridization
from reV.hybrids.hybrids import MERGE_COLUMN, OUTPUT_PROFILE_NAMES, HybridsData
from reV.utilities import ModuleName, SupplyCurveField
from reV.utilities.exceptions import FileInputError, InputError, OutputWarning

SOLAR_FPATH = os.path.join(
    TESTDATADIR, "rep_profiles_out", "rep_profiles_solar.h5"
)
WIND_FPATH = os.path.join(
    TESTDATADIR, "rep_profiles_out", "rep_profiles_wind.h5"
)
SOLAR_FPATH_30_MIN = os.path.join(
    TESTDATADIR, "rep_profiles_out", "rep_profiles_solar_30_min.h5"
)
SOLAR_FPATH_MULT = os.path.join(
    TESTDATADIR, "rep_profiles_out", "rep_profiles_solar_multiple.h5"
)
with Resource(SOLAR_FPATH) as res:
    SOLAR_SCPGIDS = set(res.meta["sc_point_gid"])
with Resource(WIND_FPATH) as res:
    WIND_SCPGIDS = set(res.meta["sc_point_gid"])


def _fix_meta(fp):
    with Outputs(fp, mode="a") as out:
        meta = out.meta
        del out._h5['meta']
        out._meta = None
        out.meta = meta.rename(columns=SupplyCurveField.map_from_legacy())


@pytest.fixture(scope="module")
def module_td():
    """Module-level temporaty dirsctory"""
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture(scope="module")
def solar_fpath(module_td):
    """Solar fpath with legacy columns renamed. """
    new_fp = os.path.join(module_td, "solar.h5")
    shutil.copy(SOLAR_FPATH, new_fp)
    _fix_meta(new_fp)
    yield new_fp


@pytest.fixture(scope="module")
def wind_fpath(module_td):
    """Wind fpath with legacy columns renamed. """
    new_fp = os.path.join(module_td, "wind.h5")
    shutil.copy(WIND_FPATH, new_fp)
    _fix_meta(new_fp)
    yield new_fp


@pytest.fixture(scope="module")
def solar_fpath_30_min(module_td):
    """Solar fpath (30 min data) with legacy columns renamed."""
    new_fp = os.path.join(module_td, "solar_30min.h5")
    shutil.copy(SOLAR_FPATH_30_MIN, new_fp)
    _fix_meta(new_fp)
    yield new_fp


@pytest.fixture(scope="module")
def solar_fpath_mult(module_td):
    """Solar fpath (with mult) with legacy columns renamed. """
    new_fp = os.path.join(module_td, "solar_mult.h5")
    shutil.copy(SOLAR_FPATH_MULT, new_fp)
    _fix_meta(new_fp)
    yield new_fp


def test_hybridization_profile_output_single_resource(solar_fpath, wind_fpath):
    """Test that the hybridization calculation is correct (1 resource)."""

    sc_point_gid = 40005

    with Resource(solar_fpath) as res:
        solar_idx = np.where(
            res.meta[SupplyCurveField.SC_POINT_GID] == sc_point_gid
        )[0][0]

        solar_cap = res.meta.loc[solar_idx, SupplyCurveField.CAPACITY_AC_MW]
        solar_test_profile = res["rep_profiles_0", :, solar_idx]

    weighted_solar = solar_cap * solar_test_profile

    h = Hybridization(solar_fpath, wind_fpath, allow_solar_only=True)
    h.run()
    hp, hsp, hwp = h.profiles.values()
    h_meta = h.hybrid_meta
    h_idx = np.where(h_meta[SupplyCurveField.SC_POINT_GID] == sc_point_gid)[0][
        0
    ]

    assert np.allclose(hp[:, h_idx], weighted_solar)
    assert np.allclose(hsp[:, h_idx], weighted_solar)
    assert np.allclose(hp[:, h_idx], hsp[:, h_idx])
    assert np.allclose(hwp[:, h_idx], 0)


def test_hybridization_profile_output_with_ratio_none(solar_fpath, wind_fpath):
    """Test that the hybridization calculation is correct (1 resource)."""

    sc_point_gid = 40005

    with Resource(solar_fpath) as res:

        solar_idx = np.where(
            res.meta[SupplyCurveField.SC_POINT_GID] == sc_point_gid
        )[0][0]

        solar_cap = res.meta.loc[solar_idx, SupplyCurveField.CAPACITY_AC_MW]
        solar_test_profile = res["rep_profiles_0", :, solar_idx]

    weighted_solar = solar_cap * solar_test_profile

    h = Hybridization(
        solar_fpath,
        wind_fpath,
        allow_solar_only=True,
        ratio=None,
        ratio_bounds=None,
    )
    h.run()
    hp, hsp, hwp = h.profiles.values()
    h_meta = h.hybrid_meta
    h_idx = np.where(h_meta[SupplyCurveField.SC_POINT_GID] == sc_point_gid)[0][
        0
    ]

    assert np.allclose(hp[:, h_idx], weighted_solar)
    assert np.allclose(hsp[:, h_idx], weighted_solar)
    assert np.allclose(hp[:, h_idx], hsp[:, h_idx])
    assert np.allclose(hwp[:, h_idx], 0)


def test_hybridization_profile_output(solar_fpath, wind_fpath):
    """Test that the hybridization calculation is correct."""
    common_sc_point_gid = 38883

    with Resource(solar_fpath) as res:
        solar_idx = np.where(
            res.meta[SupplyCurveField.SC_POINT_GID] == common_sc_point_gid
        )[0][0]
        solar_cap = res.meta.loc[solar_idx, SupplyCurveField.CAPACITY_AC_MW]
        solar_test_profile = res["rep_profiles_0", :, solar_idx]

    with Resource(wind_fpath) as res:
        wind_idx = np.where(
            res.meta[SupplyCurveField.SC_POINT_GID] == common_sc_point_gid
        )[0][0]
        wind_cap = res.meta.loc[wind_idx, SupplyCurveField.CAPACITY_AC_MW]
        wind_test_profile = res["rep_profiles_0", :, wind_idx]

    weighted_solar = solar_cap * solar_test_profile
    weighted_wind = wind_cap * wind_test_profile

    h = Hybridization(solar_fpath, wind_fpath)
    h.run()
    (
        hp,
        hsp,
        hwp,
    ) = h.profiles.values()
    h_meta = h.hybrid_meta
    h_idx = np.where(
        h_meta[SupplyCurveField.SC_POINT_GID] == common_sc_point_gid
    )[0][0]

    assert np.allclose(hp[:, h_idx], weighted_solar + weighted_wind)
    assert np.allclose(hsp[:, h_idx], weighted_solar)
    assert np.allclose(hwp[:, h_idx], weighted_wind)


@pytest.mark.parametrize("half_hour", [True, False])
def test_hybridization_output_shapes(half_hour, solar_fpath,
                                     solar_fpath_30_min, wind_fpath):
    """Test that the output shapes are as expected."""
    if half_hour:
        input_files = solar_fpath_30_min, wind_fpath
    else:
        input_files = solar_fpath, wind_fpath

    sfp, wfp = input_files
    h = Hybridization(sfp, wfp)
    h.run()
    out = [*h.profiles.values(), h.hybrid_meta, h.hybrid_time_index]
    expected_shapes = [(8760, 53)] * 3 + [(53, 73), (8760,)]
    for arr, expected_shape in zip(out, expected_shapes):
        assert arr.shape == expected_shape

    h = Hybridization(sfp, wfp, allow_solar_only=True)
    h.run()
    out = [*h.profiles.values(), h.hybrid_meta, h.hybrid_time_index]
    expected_shapes = [(8760, 100)] * 3 + [(100, 73), (8760,)]
    for arr, expected_shape in zip(out, expected_shapes):
        assert arr.shape == expected_shape

    h = Hybridization(sfp, wfp, allow_solar_only=True, allow_wind_only=True)
    h.run()
    out = [*h.profiles.values(), h.hybrid_meta, h.hybrid_time_index]
    expected_shapes = [(8760, 147)] * 3 + [(147, 73), (8760,)]
    for arr, expected_shape in zip(out, expected_shapes):
        assert arr.shape == expected_shape


@pytest.mark.parametrize(
    "input_combination, expected_shape, overlap",
    [
        ((False, False), (53, 73), SOLAR_SCPGIDS & WIND_SCPGIDS),
        ((True, False), (100, 73), SOLAR_SCPGIDS),
        ((False, True), (100, 73), WIND_SCPGIDS),
        ((True, True), (147, 73), SOLAR_SCPGIDS | WIND_SCPGIDS),
    ],
)
def test_meta_hybridization(input_combination, expected_shape, overlap,
                            solar_fpath, wind_fpath):
    """Test that the meta is hybridized properly."""

    allow_solar_only, allow_wind_only = input_combination
    h = Hybridization(
        solar_fpath,
        wind_fpath,
        allow_solar_only=allow_solar_only,
        allow_wind_only=allow_wind_only,
    )
    h.run()
    assert h.hybrid_meta.shape == expected_shape
    assert set(h.hybrid_meta[SupplyCurveField.SC_POINT_GID]) == overlap


def test_limits_and_ratios_output_values(solar_fpath, wind_fpath):
    """Test that limits and ratios are properly applied in succession."""

    limits = {f"solar_{SupplyCurveField.CAPACITY_AC_MW}": 50,
              f"wind_{SupplyCurveField.CAPACITY_AC_MW}": 0.5}
    ratio_numerator = f"solar_{SupplyCurveField.CAPACITY_AC_MW}"
    ratio_denominator = f"wind_{SupplyCurveField.CAPACITY_AC_MW}"
    ratio = "{}/{}".format(ratio_numerator, ratio_denominator)
    ratio_bounds = (0.3, 3.6)
    bounds = (0.3 - 1e6, 3.6 + 1e6)

    h = Hybridization(
        solar_fpath,
        wind_fpath,
        limits=limits,
        ratio=ratio,
        ratio_bounds=ratio_bounds,
    )
    h.run()

    ratios = (
        h.hybrid_meta["hybrid_{}".format(ratio_numerator)]
        / h.hybrid_meta["hybrid_{}".format(ratio_denominator)]
    )
    assert np.all(ratios.between(*bounds))
    assert np.all(
        h.hybrid_meta["hybrid_{}".format(ratio_numerator)]
        <= h.hybrid_meta[ratio_numerator]
    )
    assert np.all(
        h.hybrid_meta["hybrid_{}".format(ratio_denominator)]
        <= h.hybrid_meta[ratio_denominator]
    )
    assert np.all(h.hybrid_meta[f"solar_{SupplyCurveField.CAPACITY_AC_MW}"]
                  <= limits[f"solar_{SupplyCurveField.CAPACITY_AC_MW}"])
    assert np.all(h.hybrid_meta[f"wind_{SupplyCurveField.CAPACITY_AC_MW}"]
                  <= limits[f"wind_{SupplyCurveField.CAPACITY_AC_MW}"])


@pytest.mark.parametrize(
    "ratio_cols",
    [
        (f"solar_{SupplyCurveField.CAPACITY_AC_MW}",
         f"wind_{SupplyCurveField.CAPACITY_AC_MW}"),
        (f"solar_{SupplyCurveField.AREA_SQ_KM}",
         f"wind_{SupplyCurveField.AREA_SQ_KM}"),
    ],
)
@pytest.mark.parametrize(
    "ratio_bounds, bounds",
    [
        ((0.5, 0.5), (0.5 - 1e6, 0.5 + 1e6)),
        ((1, 1), (1 - 1e6, 1 + 1e6)),
        ((0.5, 1.5), (0.5 - 1e6, 1.5 + 1e6)),
        ((0.3, 3.6), (0.3 - 1e6, 3.6 + 1e6)),
    ],
)
def test_ratios_input(ratio_cols, ratio_bounds, bounds, solar_fpath,
                      wind_fpath):
    """Test that the hybrid meta limits the ratio columns correctly."""
    ratio_numerator, ratio_denominator = ratio_cols
    ratio = "{}/{}".format(ratio_numerator, ratio_denominator)
    h = Hybridization(
        solar_fpath, wind_fpath, ratio=ratio, ratio_bounds=ratio_bounds
    )
    h.run()

    ratios = (
        h.hybrid_meta["hybrid_{}".format(ratio_numerator)]
        / h.hybrid_meta["hybrid_{}".format(ratio_denominator)]
    )

    assert np.all(ratios.between(*bounds))
    assert np.all(
        h.hybrid_meta["hybrid_{}".format(ratio_numerator)]
        <= h.hybrid_meta[ratio_numerator]
    )
    assert np.all(
        h.hybrid_meta["hybrid_{}".format(ratio_denominator)]
        <= h.hybrid_meta[ratio_denominator]
    )

    if SupplyCurveField.CAPACITY_AC_MW in ratio:
        col = f"hybrid_solar_{SupplyCurveField.CAPACITY_AC_MW}"
        max_solar_capacities = h.hybrid_meta[col]
        max_solar_capacities = max_solar_capacities.values.reshape(1, -1)
        assert np.all(
            h.profiles["hybrid_solar_profile"] <= max_solar_capacities
        )
        col = f"hybrid_wind_{SupplyCurveField.CAPACITY_AC_MW}"
        max_wind_capacities = h.hybrid_meta[col]
        max_wind_capacities = max_wind_capacities.values.reshape(1, -1)
        assert np.all(h.profiles["hybrid_wind_profile"] <= max_wind_capacities)


def test_rep_profile_idx_map(solar_fpath, wind_fpath):
    """Test that rep profile index mappings are correct shape."""
    h = Hybridization(solar_fpath, wind_fpath, allow_wind_only=True)

    for h_idxs, r_idxs in (
        h.meta_hybridizer.solar_profile_indices_map,
        h.meta_hybridizer.wind_profile_indices_map,
    ):
        assert h_idxs.size == 0
        assert r_idxs.size == 0

    h.meta_hybridizer.hybridize()

    for idxs, shape in zip(
        (
            h.meta_hybridizer.solar_profile_indices_map,
            h.meta_hybridizer.wind_profile_indices_map,
        ),
        (53, 100),
    ):
        h_idxs, r_idxs = idxs
        assert h_idxs.size == shape
        assert r_idxs.size == shape


def test_limits_values(solar_fpath, wind_fpath):
    """Test that column values are properly limited on user input."""

    limits = {f"solar_{SupplyCurveField.CAPACITY_AC_MW}": 100,
              f"wind_{SupplyCurveField.CAPACITY_AC_MW}": 0.5}

    h = Hybridization(solar_fpath, wind_fpath, limits=limits)
    h.run()

    assert np.all(h.hybrid_meta[f"solar_{SupplyCurveField.CAPACITY_AC_MW}"]
                  <= limits[f"solar_{SupplyCurveField.CAPACITY_AC_MW}"])
    assert np.all(h.hybrid_meta[f"wind_{SupplyCurveField.CAPACITY_AC_MW}"]
                  <= limits[f"wind_{SupplyCurveField.CAPACITY_AC_MW}"])


def test_invalid_limits_column_name(solar_fpath, wind_fpath):
    """Test invalid inputs for limits columns."""

    test_limits = {"un_prefixed_col": 0,
                   f"wind_{SupplyCurveField.CAPACITY_AC_MW}": 10}
    with pytest.raises(InputError) as excinfo:
        Hybridization(solar_fpath, wind_fpath, limits=test_limits)

    assert "Input limits column" in str(excinfo.value)
    assert "does not start with a valid prefix" in str(excinfo.value)


def test_fillna_values(solar_fpath, wind_fpath):
    """Test that N/A values are filled properly based on user input."""

    fill_vals = {f"solar_{SupplyCurveField.N_GIDS}": 0,
                 f"wind_{SupplyCurveField.CAPACITY_AC_MW}": -1}

    h = Hybridization(
        solar_fpath,
        wind_fpath,
        allow_solar_only=True,
        allow_wind_only=True,
        fillna=fill_vals,
    )
    h.run()

    assert not np.any(h.hybrid_meta[f"solar_{SupplyCurveField.N_GIDS}"].isna())
    assert not np.any(
        h.hybrid_meta[f"wind_{SupplyCurveField.CAPACITY_AC_MW}"].isna()
    )
    assert np.any(h.hybrid_meta[f"solar_{SupplyCurveField.N_GIDS}"].values
                  == 0)
    assert np.any(
        h.hybrid_meta[f"wind_{SupplyCurveField.CAPACITY_AC_MW}"].values
        == -1)


def test_invalid_fillna_column_name(solar_fpath, wind_fpath):
    """Test invalid inputs for fillna columns."""

    test_fillna = {"un_prefixed_col": 0,
                   f"wind_{SupplyCurveField.CAPACITY_AC_MW}": 10}
    with pytest.raises(InputError) as excinfo:
        Hybridization(solar_fpath, wind_fpath, fillna=test_fillna)

    assert "Input fillna column" in str(excinfo.value)
    assert "does not start with a valid prefix" in str(excinfo.value)


@pytest.mark.parametrize(
    "input_combination, na_vals",
    [
        ((False, False), (False, False)),
        ((True, False), (False, True)),
        ((False, True), (True, False)),
        ((True, True), (True, True)),
    ],
)
def test_all_allow_solar_allow_wind_combinations(input_combination, na_vals,
                                                 solar_fpath, wind_fpath):
    """Test that "allow_x_only" options perform the intended merges."""

    allow_solar_only, allow_wind_only = input_combination
    h = Hybridization(
        solar_fpath,
        wind_fpath,
        allow_solar_only=allow_solar_only,
        allow_wind_only=allow_wind_only,
    )
    h.run()

    cols = [f"solar_{SupplyCurveField.SC_GID}",
            f"wind_{SupplyCurveField.SC_GID}"]
    for col_name, should_have_na_vals in zip(cols, na_vals):
        if should_have_na_vals:
            assert np.any(h.hybrid_meta[col_name].isna())
        else:
            assert not np.any(h.hybrid_meta[col_name].isna())


def test_warning_for_improper_data_output_from_hybrid_method(solar_fpath,
                                                             wind_fpath):
    """Test that hybrid function with incorrect output throws warning."""

    def some_new_hybrid_func(__):
        return [0]

    HYBRID_METHODS["scaled_elevation"] = some_new_hybrid_func

    with pytest.warns(OutputWarning) as records:
        h = Hybridization(solar_fpath, wind_fpath)
        h.run()

    messages = [r.message.args[0] for r in records]
    assert any("Unable to add" in msg for msg in messages)
    assert any("column to hybrid meta" in msg for msg in messages)

    HYBRID_METHODS.pop("scaled_elevation")


def test_hybrid_col_additional_method(solar_fpath, wind_fpath):
    """Test that function decorated with 'hybrid_col' adds to hybrid meta."""

    def some_new_hybrid_func(h):
        return h.hybrid_meta[SupplyCurveField.ELEVATION] * 1000

    HYBRID_METHODS["scaled_elevation"] = some_new_hybrid_func

    h = Hybridization(solar_fpath, wind_fpath)
    h.run()

    assert "scaled_elevation" in HYBRID_METHODS
    assert "scaled_elevation" in h.hybrid_meta.columns
    assert np.allclose(
        h.hybrid_meta[SupplyCurveField.ELEVATION] * 1000,
        h.hybrid_meta["scaled_elevation"],
    )

    HYBRID_METHODS.pop("scaled_elevation")


def test_duplicate_lat_long_values(solar_fpath, wind_fpath, module_td):
    """Test duplicate lat/long values corresponding to unique merge column."""

    fout_solar = os.path.join(module_td, "rep_profiles_solar.h5")
    make_test_file(solar_fpath, fout_solar, duplicate_coord_values=True)

    with pytest.raises(FileInputError) as excinfo:
        h = Hybridization(fout_solar, wind_fpath)
        h.run()

    assert "Detected mismatched coordinate values" in str(excinfo.value)


def test_invalid_ratio_bounds_length_input(solar_fpath, wind_fpath):
    """Test improper ratios input."""

    ratio = (
        f"solar_{SupplyCurveField.CAPACITY_AC_MW}"
        f"/wind_{SupplyCurveField.CAPACITY_AC_MW}"
    )
    with pytest.raises(InputError) as excinfo:
        Hybridization(
            solar_fpath, wind_fpath, ratio=ratio, ratio_bounds=(1, 2, 3)
        )

    msg = (
        "Length of input for ratio_bounds is 3 "
        "- but is required to be of length 2."
    )
    assert msg in str(excinfo.value)


def test_ratio_column_missing(solar_fpath, wind_fpath):
    """Test missing ratio column."""

    ratio = f"solar_col_dne/wind_{SupplyCurveField.CAPACITY_AC_MW}"
    with pytest.raises(FileInputError) as excinfo:
        Hybridization(
            solar_fpath, wind_fpath, ratio=ratio, ratio_bounds=(1, 1)
        )

    assert "Input ratios column" in str(excinfo.value)
    assert "not found" in str(excinfo.value)


@pytest.mark.parametrize("ratio", [None, ("solar_capacity", "wind_capacity")])
def test_ratio_not_string(ratio, solar_fpath, wind_fpath):
    """Test ratio input is not string."""

    with pytest.raises(InputError) as excinfo:
        Hybridization(
            solar_fpath, wind_fpath, ratio=ratio, ratio_bounds=(1, 1)
        )

    assert "Ratio input type " in str(excinfo.value)
    assert "not understood" in str(excinfo.value)


@pytest.mark.parametrize(
    "ratio", ["solar_capacity", "solar_capacity/wind_capacity/solar_capacity"]
)
def test_invalid_ratio_format(ratio, solar_fpath, wind_fpath):
    """Test ratio input is not string."""

    with pytest.raises(InputError) as excinfo:
        Hybridization(
            solar_fpath, wind_fpath, ratio=ratio, ratio_bounds=(1, 1)
        )

    long_msg = (
        "Please make sure the ratio input is a string in the form "
        "'numerator_column_name/denominator_column_name'"
    )
    assert "Ratio input " in str(excinfo.value)
    assert long_msg in str(excinfo.value)


def test_invalid_ratio_column_name(solar_fpath, wind_fpath):
    """Test invalid inputs for ratio columns."""

    ratio = f"un_prefixed_col/wind_{SupplyCurveField.CAPACITY_AC_MW}"
    with pytest.raises(InputError) as excinfo:
        Hybridization(
            solar_fpath, wind_fpath, ratio=ratio, ratio_bounds=(1, 1)
        )

    assert "Input ratios column" in str(excinfo.value)
    assert "does not start with a valid prefix" in str(excinfo.value)


def test_no_overlap_in_merge_column_values(solar_fpath, wind_fpath):
    """Test duplicate values in merge column."""

    with tempfile.TemporaryDirectory() as td:
        fout_solar = os.path.join(td, "rep_profiles_solar.h5")
        fout_wind = os.path.join(td, "rep_profiles_wind.h5")
        make_test_file(solar_fpath, fout_solar, p_slice=slice(0, 3))
        make_test_file(wind_fpath, fout_wind, p_slice=slice(90, 100))

        with pytest.raises(FileInputError) as excinfo:
            Hybridization(fout_solar, fout_wind)

        assert "No overlap detected in the values" in str(excinfo.value)


def test_duplicate_merge_column_values(solar_fpath, wind_fpath):
    """Test duplicate values in merge column."""

    with tempfile.TemporaryDirectory() as td:
        fout_solar = os.path.join(td, "rep_profiles_solar.h5")
        make_test_file(solar_fpath, fout_solar, duplicate_rows=True)

        with pytest.raises(FileInputError) as excinfo:
            Hybridization(fout_solar, wind_fpath)

        assert "Duplicate" in str(excinfo.value)


def test_merge_columns_missing(solar_fpath, wind_fpath):
    """Test missing merge column."""

    with tempfile.TemporaryDirectory() as td:
        fout_solar = os.path.join(td, "rep_profiles_solar.h5")
        make_test_file(solar_fpath, fout_solar, drop_cols=[MERGE_COLUMN])

        with pytest.raises(FileInputError) as excinfo:
            Hybridization(fout_solar, wind_fpath)

        msg = "Cannot hybridize: merge column"
        assert msg in str(excinfo.value)
        assert "missing" in str(excinfo.value)


def test_invalid_num_profiles(solar_fpath_mult, wind_fpath):
    """Test input files with an invalid number of profiles (>1)."""

    with pytest.raises(FileInputError) as excinfo:
        Hybridization(solar_fpath_mult, wind_fpath)

        msg = (
            "This module is not intended for hybridization of "
            "multiple representative profiles. Please re-run "
            "on a single aggregated profile."
        )
        assert msg in str(excinfo.value)


def test_invalid_time_index_overlap(solar_fpath, wind_fpath):
    """Test input files with an invalid time index overlap."""

    with tempfile.TemporaryDirectory() as td:
        fout_solar = os.path.join(td, "rep_profiles_solar.h5")
        fout_wind = os.path.join(td, "rep_profiles_wind.h5")
        make_test_file(solar_fpath, fout_solar, t_slice=slice(0, 1500))
        make_test_file(wind_fpath, fout_wind, t_slice=slice(1000, 3000))

        with pytest.raises(FileInputError) as excinfo:
            Hybridization(fout_solar, fout_wind)

        msg = (
            "Please ensure that the input profiles have a "
            "time index that overlaps >= 8760 times."
        )
        assert msg in str(excinfo.value)


def test_valid_time_index_overlap(solar_fpath_30_min, wind_fpath):
    """Test input files with a valid time index overlap."""

    h = Hybridization(solar_fpath_30_min, wind_fpath)

    with Resource(solar_fpath_30_min) as res:
        assert np.all(res.time_index == h.solar_time_index)

    with Resource(wind_fpath) as res:
        assert np.all(res.time_index == h.wind_time_index)
        assert len(res.time_index) == len(h.hybrid_time_index)


def test_write_to_file(solar_fpath, wind_fpath):
    """Test hybrid rep profiles with file write."""
    with tempfile.TemporaryDirectory() as td:
        fout = os.path.join(td, "temp_hybrid_profiles.h5")
        h = Hybridization(solar_fpath, wind_fpath)
        h.run(fout=fout)

        with Resource(fout) as res:
            for name, p in zip(OUTPUT_PROFILE_NAMES, h.profiles.values()):
                dtype = res.get_dset_properties(name)[1]
                attrs = res.get_attrs(name)
                disk_profiles = res[name]

                assert np.issubdtype(dtype, np.float32)
                assert attrs["units"] == "MW"
                assert np.allclose(p, disk_profiles)

            disk_dsets = res.datasets
            assert "rep_profiles_0" not in disk_dsets


def test_hybrids_data_content(solar_fpath, wind_fpath):
    """Test HybridsData class content."""

    fv = -999
    h_data = HybridsData(solar_fpath, wind_fpath)

    with Resource(solar_fpath) as sr, Resource(wind_fpath) as wr:
        assert np.all(h_data.solar_meta.fillna(fv) == sr.meta.fillna(fv))
        assert np.all(h_data.wind_meta.fillna(fv) == wr.meta.fillna(fv))
        assert np.all(h_data.solar_time_index == sr.time_index)
        assert np.all(h_data.wind_time_index == wr.time_index)
        hyb_idx = sr.time_index.join(wr.time_index, how="inner")
        assert np.all(h_data.hybrid_time_index == hyb_idx)


def test_hybrids_data_contains_col(solar_fpath, wind_fpath):
    """Test the 'contains_col' method of HybridsData for accuracy."""

    h_data = HybridsData(solar_fpath, wind_fpath)
    assert h_data.contains_col(SupplyCurveField.TRANS_CAPACITY)
    assert h_data.contains_col("dist_mi")
    assert h_data.contains_col(SupplyCurveField.DIST_SPUR_KM)
    assert not h_data.contains_col("dne_col_for_test")


@pytest.mark.parametrize("half_hour", [True, False])
@pytest.mark.parametrize(
    "ratio",
    [f"solar_{SupplyCurveField.CAPACITY_AC_MW}"
     f"/wind_{SupplyCurveField.CAPACITY_AC_MW}",
     f"solar_{SupplyCurveField.AREA_SQ_KM}"
     f"/wind_{SupplyCurveField.AREA_SQ_KM}"],
)
@pytest.mark.parametrize("ratio_bounds", [None, [0.5, 1.5], [0.3, 3.6]])
@pytest.mark.parametrize("input_combination", [(False, False), (True, True)])
def test_hybrids_cli_from_config(
    runner, half_hour, ratio, ratio_bounds, input_combination, clear_loggers,
    solar_fpath, solar_fpath_30_min, wind_fpath
):
    """Test hybrids cli from config"""
    fv = -999
    allow_solar_only, allow_wind_only = input_combination
    fill_vals = {f"solar_{SupplyCurveField.N_GIDS}": 0,
                 f"wind_{SupplyCurveField.CAPACITY_AC_MW}": -1}
    limits = {f"solar_{SupplyCurveField.CAPACITY_AC_MW}": 100}

    if half_hour:
        sfp, wfp = solar_fpath_30_min, wind_fpath
    else:
        sfp, wfp = solar_fpath, wind_fpath

    with tempfile.TemporaryDirectory() as td:
        config = {
            "solar_fpath": sfp,
            "wind_fpath": wfp,
            "log_directory": td,
            "execution_control": {
                "nodes": 1,
                "option": "local",
                "sites_per_worker": 10,
            },
            "log_level": "INFO",
            "allow_solar_only": allow_solar_only,
            "allow_wind_only": allow_wind_only,
            "fillna": fill_vals,
            "limits": limits,
            "ratio": ratio,
            "ratio_bounds": ratio_bounds,
        }

        config_path = os.path.join(td, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = runner.invoke(
            main, [str(ModuleName.HYBRIDS), "-c", config_path]
        )

        if result.exit_code != 0:
            import traceback

            msg = "Failed with error {}".format(
                traceback.print_exception(*result.exc_info)
            )
            clear_loggers()
            raise RuntimeError(msg)

        h = Hybridization(
            sfp,
            wfp,
            allow_solar_only=allow_solar_only,
            allow_wind_only=allow_wind_only,
            fillna=fill_vals,
            limits=limits,
            ratio=ratio,
            ratio_bounds=ratio_bounds,
        )
        h.run()
        dirname = os.path.basename(td)
        fn_out = "{}_{}.h5".format(dirname, ModuleName.HYBRIDS)
        out_fpath = os.path.join(td, fn_out)
        with Outputs(out_fpath, "r") as f:
            for dset_name in OUTPUT_PROFILE_NAMES:
                assert dset_name in f.dsets

            meta_from_file = f.meta.fillna(fv).replace("nan", fv)
            assert np.all(meta_from_file == h.hybrid_meta.fillna(fv))
            assert np.all(f.time_index.values == h.hybrid_time_index.values)

            assert "solar_fpath" in f.h5.attrs
            assert "wind_fpath" in f.h5.attrs
            assert "hybrids_config_fp" in f.h5.attrs
            assert "hybrids_config" in f.h5.attrs

            assert Path(f.h5.attrs["solar_fpath"]) == Path(sfp)
            assert Path(f.h5.attrs["wind_fpath"]) == Path(wfp)

            config_fp = Path(config_path).expanduser().resolve()
            assert Path(f.h5.attrs["hybrids_config_fp"]) == config_fp
            assert json.loads(f.h5.attrs["hybrids_config"]) == config

        clear_loggers()


@pytest.mark.parametrize(
    "bad_fpath",
    [
        os.path.join(TESTDATADIR, "rep_profiles_out", "rep_profiles_sol*.h5"),
        os.path.join(TESTDATADIR, "rep_profiles_out", "rep_profiles_dne.h5"),
    ],
)
def test_hybrids_cli_bad_fpath_input(runner, bad_fpath, clear_loggers,
                                     wind_fpath):
    """Test cli when filepath input is ambiguous or invalid."""

    with tempfile.TemporaryDirectory() as td:
        config = {
            "solar_fpath": bad_fpath,
            "wind_fpath": wind_fpath,
            "log_directory": td,
            "execution_control": {
                "nodes": 1,
                "option": "local",
                "sites_per_worker": 10,
            },
            "log_level": "INFO",
        }

        config_path = os.path.join(td, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = runner.invoke(
            main, [str(ModuleName.HYBRIDS), "-c", config_path]
        )

        assert result.exit_code != 0
        assert "No files found" in str(result.exc_info)

        clear_loggers()


# pylint: disable=no-member
def make_test_file(
    in_fp,
    out_fp,
    p_slice=slice(None),
    t_slice=slice(None),
    drop_cols=None,
    duplicate_rows=False,
    duplicate_coord_values=False,
):
    """Generate a test file from existing input file.

    The new test file can have a subset of the data of the original file.

    Parameters
    ----------
    in_fp : str
        Filepath to input file containing a meta, time_index, and at least one
        rep_profile.
    out_fp : str
        Filepath for new output file. This file will contain some subset of
        the data of the input file.
    p_slice : slice, optional
        Point-slice object representing the indices of the points to keep in
        the new file, by default slice(None).
    t_slice : slice, optional
        Time-slice object representing the indices of the time indices to keep
        in the new file, by default slice(None).
    drop_cols : single label or list-like, optional
        Iterable object representing the columns to drop from `meta`, by
        default None.
    duplicate_rows : bool, optional
        Option to duplicate the first half of all rows in meta DataFrame,
        by default False.
    duplicate_coord_values : bool, optional
        Option to randomly duplicate coordinate values (lat and lon) in the
        meta DataFrame, by default False.
    """
    with Resource(in_fp) as res:
        dset_names = [d for d in res.dsets if d not in ("meta", "time_index")]
        shapes = res.shapes
        meta = res.meta.iloc[p_slice]
        if drop_cols is not None:
            meta.drop(columns=drop_cols, inplace=True)
        if duplicate_rows:
            n_rows, __ = meta.shape
            half_n_rows = n_rows // 2
            meta.iloc[-half_n_rows:] = meta.iloc[:half_n_rows].values
        if duplicate_coord_values:
            lat = meta[SupplyCurveField.LATITUDE].iloc[-1]
            meta.loc[0, SupplyCurveField.LATITUDE] = lat
            lon = meta[SupplyCurveField.LONGITUDE].iloc[-1]
            meta.loc[0, SupplyCurveField.LONGITUDE] = lon
        shapes["meta"] = len(meta)
        for d in dset_names:
            shapes[d] = (len(res.time_index[t_slice]), len(meta))

        Outputs.init_h5(
            out_fp,
            dset_names,
            shapes,
            res.attrs,
            res.chunks,
            res.dtypes,
            meta,
            time_index=res.time_index[t_slice],
        )

        with Outputs(out_fp, mode="a") as out:
            for d in dset_names:
                out[d] = res[d, t_slice, p_slice]

            d = "rep_profiles_0"
            assert out._h5[d].shape == (
                len(res.time_index[t_slice]),
                len(meta),
            )
            assert np.all(out[d].sum(axis=0) > 0)


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
    pytest.main(["-q", "--show-capture={}".format(capture), __file__, flags])


if __name__ == "__main__":
    execute_pytest()
