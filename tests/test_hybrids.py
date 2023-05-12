# -*- coding: utf-8 -*-
"""reV hybrids tests.
"""
import os
import pytest
import numpy as np
import tempfile
import json

from reV.hybrids import Hybridization, HYBRID_METHODS
from reV.hybrids.hybrids import HybridsData, MERGE_COLUMN, OUTPUT_PROFILE_NAMES
from reV.hybrids.cli_hybrids import main as hybrids_cli_main
from reV.config.hybrids_config import HybridsConfig
from reV.utilities.exceptions import FileInputError, InputError, OutputWarning
from reV.cli import main
from reV import Outputs, TESTDATADIR

from rex.resource import Resource
from rex.utilities.hpc import SLURM


SOLAR_FPATH = os.path.join(
    TESTDATADIR, 'rep_profiles_out', 'rep_profiles_solar.h5')
WIND_FPATH = os.path.join(
    TESTDATADIR, 'rep_profiles_out', 'rep_profiles_wind.h5')
SOLAR_FPATH_30_MIN = os.path.join(
    TESTDATADIR, 'rep_profiles_out', 'rep_profiles_solar_30_min.h5')
SOLAR_FPATH_MULT = os.path.join(
    TESTDATADIR, 'rep_profiles_out', 'rep_profiles_solar_multiple.h5')
with Resource(SOLAR_FPATH) as res:
    SOLAR_SCPGIDS = set(res.meta['sc_point_gid'])
with Resource(WIND_FPATH) as res:
    WIND_SCPGIDS = set(res.meta['sc_point_gid'])


def test_hybridization_profile_output_single_resource():
    """Test that the hybridization calculation is correct (1 resource). """

    sc_point_gid = 40005

    with Resource(SOLAR_FPATH) as res:
        solar_idx = np.where(
            res.meta['sc_point_gid'] == sc_point_gid
        )[0][0]

        solar_cap = res.meta.loc[solar_idx, 'capacity']
        solar_test_profile = res['rep_profiles_0', :, solar_idx]

    weighted_solar = solar_cap * solar_test_profile

    h = Hybridization(SOLAR_FPATH, WIND_FPATH, allow_solar_only=True).run_all()
    hp, hsp, hwp, = h.profiles.values()
    h_meta = h.hybrid_meta
    h_idx = np.where(h_meta['sc_point_gid'] == sc_point_gid)[0][0]

    assert np.allclose(hp[:, h_idx], weighted_solar)
    assert np.allclose(hsp[:, h_idx], weighted_solar)
    assert np.allclose(hp[:, h_idx], hsp[:, h_idx])
    assert np.allclose(hwp[:, h_idx], 0)


def test_hybridization_profile_output_with_ratio_none():
    """Test that the hybridization calculation is correct (1 resource). """

    sc_point_gid = 40005

    with Resource(SOLAR_FPATH) as res:
        solar_idx = np.where(
            res.meta['sc_point_gid'] == sc_point_gid
        )[0][0]

        solar_cap = res.meta.loc[solar_idx, 'capacity']
        solar_test_profile = res['rep_profiles_0', :, solar_idx]

    weighted_solar = solar_cap * solar_test_profile

    h = Hybridization(
        SOLAR_FPATH, WIND_FPATH, allow_solar_only=True,
        ratio=None, ratio_bounds=None
    ).run_all()
    hp, hsp, hwp, = h.profiles.values()
    h_meta = h.hybrid_meta
    h_idx = np.where(h_meta['sc_point_gid'] == sc_point_gid)[0][0]

    assert np.allclose(hp[:, h_idx], weighted_solar)
    assert np.allclose(hsp[:, h_idx], weighted_solar)
    assert np.allclose(hp[:, h_idx], hsp[:, h_idx])
    assert np.allclose(hwp[:, h_idx], 0)


def test_hybridization_profile_output():
    """Test that the hybridization calculation is correct. """
    common_sc_point_gid = 38883

    with Resource(SOLAR_FPATH) as res:
        solar_idx = np.where(
            res.meta['sc_point_gid'] == common_sc_point_gid
        )[0][0]
        solar_cap = res.meta.loc[solar_idx, 'capacity']
        solar_test_profile = res['rep_profiles_0', :, solar_idx]

    with Resource(WIND_FPATH) as res:
        wind_idx = np.where(
            res.meta['sc_point_gid'] == common_sc_point_gid
        )[0][0]
        wind_cap = res.meta.loc[wind_idx, 'capacity']
        wind_test_profile = res['rep_profiles_0', :, wind_idx]

    weighted_solar = solar_cap * solar_test_profile
    weighted_wind = wind_cap * wind_test_profile

    h = Hybridization(SOLAR_FPATH, WIND_FPATH).run_all()
    hp, hsp, hwp, = h.profiles.values()
    h_meta = h.hybrid_meta
    h_idx = np.where(h_meta['sc_point_gid'] == common_sc_point_gid)[0][0]

    assert np.allclose(hp[:, h_idx], weighted_solar + weighted_wind)
    assert np.allclose(hsp[:, h_idx], weighted_solar)
    assert np.allclose(hwp[:, h_idx], weighted_wind)


@pytest.mark.parametrize("input_files", [(SOLAR_FPATH, WIND_FPATH),
                                         (SOLAR_FPATH_30_MIN, WIND_FPATH)])
def test_hybridization_output_shapes(input_files):
    """Test that the output shapes are as expected. """

    sfp, wfp = input_files
    h = Hybridization(sfp, wfp).run_all()
    out = [*h.profiles.values(), h.hybrid_meta, h.hybrid_time_index]
    expected_shapes = [(8760, 53)] * 3 + [(53, 73), (8760,)]
    for arr, expected_shape in zip(out, expected_shapes):
        assert arr.shape == expected_shape

    h = Hybridization(sfp, wfp, allow_solar_only=True).run_all()
    out = [*h.profiles.values(), h.hybrid_meta, h.hybrid_time_index]
    expected_shapes = [(8760, 100)] * 3 + [(100, 73), (8760,)]
    for arr, expected_shape in zip(out, expected_shapes):
        assert arr.shape == expected_shape

    h = Hybridization(sfp, wfp,
                      allow_solar_only=True, allow_wind_only=True).run_all()
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
        ((True, True), (147, 73), SOLAR_SCPGIDS | WIND_SCPGIDS)
    ]
)
def test_meta_hybridization(input_combination, expected_shape, overlap):
    """Test that the meta is hybridized properly."""

    allow_solar_only, allow_wind_only = input_combination
    h = Hybridization(
        SOLAR_FPATH, WIND_FPATH,
        allow_solar_only=allow_solar_only,
        allow_wind_only=allow_wind_only,
    ).run_all()
    assert h.hybrid_meta.shape == expected_shape
    assert set(h.hybrid_meta['sc_point_gid']) == overlap


def test_limits_and_ratios_output_values():
    """Test that limits and ratios are properly applied in succession. """

    limits = {'solar_capacity': 50, 'wind_capacity': 0.5}
    ratio_numerator = 'solar_capacity'
    ratio_denominator = 'wind_capacity'
    ratio = '{}/{}'.format(ratio_numerator, ratio_denominator)
    ratio_bounds = (0.3, 3.6)
    bounds = (0.3 - 1e6, 3.6 + 1e6)

    h = Hybridization(
        SOLAR_FPATH, WIND_FPATH,
        limits=limits,
        ratio=ratio,
        ratio_bounds=ratio_bounds
    ).run_all()

    ratios = (h.hybrid_meta['hybrid_{}'.format(ratio_numerator)]
              / h.hybrid_meta['hybrid_{}'.format(ratio_denominator)])
    assert np.all(ratios.between(*bounds))
    assert np.all(h.hybrid_meta['hybrid_{}'.format(ratio_numerator)]
                  <= h.hybrid_meta[ratio_numerator])
    assert np.all(h.hybrid_meta['hybrid_{}'.format(ratio_denominator)]
                  <= h.hybrid_meta[ratio_denominator])
    assert np.all(h.hybrid_meta['solar_capacity'] <= limits['solar_capacity'])
    assert np.all(h.hybrid_meta['wind_capacity'] <= limits['wind_capacity'])


@pytest.mark.parametrize("ratio_cols", [
    ('solar_capacity', 'wind_capacity'),
    ('solar_area_sq_km', 'wind_area_sq_km')
])
@pytest.mark.parametrize("ratio_bounds, bounds", [
    ((0.5, 0.5), (0.5 - 1e6, 0.5 + 1e6)),
    ((1, 1), (1 - 1e6, 1 + 1e6)),
    ((0.5, 1.5), (0.5 - 1e6, 1.5 + 1e6)),
    ((0.3, 3.6), (0.3 - 1e6, 3.6 + 1e6))
])
def test_ratios_input(ratio_cols, ratio_bounds, bounds):
    """Test that the hybrid meta limits the ratio columns correctly. """
    ratio_numerator, ratio_denominator = ratio_cols
    ratio = '{}/{}'.format(ratio_numerator, ratio_denominator)
    h = Hybridization(
        SOLAR_FPATH, WIND_FPATH,
        ratio=ratio,
        ratio_bounds=ratio_bounds
    ).run_all()

    ratios = (h.hybrid_meta['hybrid_{}'.format(ratio_numerator)]
              / h.hybrid_meta['hybrid_{}'.format(ratio_denominator)])

    assert np.all(ratios.between(*bounds))
    assert np.all(h.hybrid_meta['hybrid_{}'.format(ratio_numerator)]
                  <= h.hybrid_meta[ratio_numerator])
    assert np.all(h.hybrid_meta['hybrid_{}'.format(ratio_denominator)]
                  <= h.hybrid_meta[ratio_denominator])

    if 'capacity' in ratio:
        max_solar_capacities = h.hybrid_meta['hybrid_solar_capacity']
        max_solar_capacities = max_solar_capacities.values.reshape(1, -1)
        assert np.all(h.profiles['hybrid_solar_profile']
                      <= max_solar_capacities)
        max_wind_capacities = h.hybrid_meta['hybrid_wind_capacity']
        max_wind_capacities = max_wind_capacities.values.reshape(1, -1)
        assert np.all(h.profiles['hybrid_wind_profile']
                      <= max_wind_capacities)


def test_rep_profile_idx_map():
    """Test that rep profile index mappings are correct shape. """
    h = Hybridization(SOLAR_FPATH, WIND_FPATH, allow_wind_only=True)

    for h_idxs, r_idxs in (h.meta_hybridizer.solar_profile_indices_map,
                           h.meta_hybridizer.wind_profile_indices_map):
        assert h_idxs.size == 0
        assert r_idxs.size == 0

    h.meta_hybridizer.hybridize()

    for idxs, shape in zip((h.meta_hybridizer.solar_profile_indices_map,
                            h.meta_hybridizer.wind_profile_indices_map),
                           (53, 100)):
        h_idxs, r_idxs = idxs
        assert h_idxs.size == shape
        assert r_idxs.size == shape


def test_limits_values():
    """Test that column values are properly limited on user input. """

    limits = {'solar_capacity': 100, 'wind_capacity': 0.5}

    h = Hybridization(SOLAR_FPATH, WIND_FPATH, limits=limits).run_all()

    assert np.all(h.hybrid_meta['solar_capacity'] <= limits['solar_capacity'])
    assert np.all(h.hybrid_meta['wind_capacity'] <= limits['wind_capacity'])


def test_invalid_limits_column_name():
    """Test invalid inputs for limits columns. """

    test_limits = {'un_prefixed_col': 0, 'wind_capacity': 10}
    with pytest.raises(InputError) as excinfo:
        Hybridization(SOLAR_FPATH, WIND_FPATH, limits=test_limits)

    assert "Input limits column" in str(excinfo.value)
    assert "does not start with a valid prefix" in str(excinfo.value)


def test_fillna_values():
    """Test that N/A values are filled properly based on user input. """

    fill_vals = {'solar_n_gids': 0, 'wind_capacity': -1}

    h = Hybridization(
        SOLAR_FPATH, WIND_FPATH, allow_solar_only=True,
        allow_wind_only=True, fillna=fill_vals
    ).run_all()

    assert not np.any(h.hybrid_meta['solar_n_gids'].isna())
    assert not np.any(h.hybrid_meta['wind_capacity'].isna())
    assert np.any(h.hybrid_meta['solar_n_gids'].values == 0)
    assert np.any(h.hybrid_meta['wind_capacity'].values == -1)


def test_invalid_fillna_column_name():
    """Test invalid inputs for fillna columns. """

    test_fillna = {'un_prefixed_col': 0, 'wind_capacity': 10}
    with pytest.raises(InputError) as excinfo:
        Hybridization(SOLAR_FPATH, WIND_FPATH, fillna=test_fillna)

    assert "Input fillna column" in str(excinfo.value)
    assert "does not start with a valid prefix" in str(excinfo.value)


@pytest.mark.parametrize("input_combination, na_vals",
                         [((False, False), (False, False)),
                          ((True, False), (False, True)),
                          ((False, True), (True, False)),
                          ((True, True), (True, True))])
def test_all_allow_solar_allow_wind_combinations(input_combination, na_vals):
    """Test that "allow_x_only" options perform the intended merges. """

    allow_solar_only, allow_wind_only = input_combination
    h = Hybridization(
        SOLAR_FPATH, WIND_FPATH,
        allow_solar_only=allow_solar_only,
        allow_wind_only=allow_wind_only,
    ).run_all()

    for col_name, should_have_na_vals in zip(['solar_sc_gid', 'wind_sc_gid'],
                                             na_vals):
        if should_have_na_vals:
            assert np.any(h.hybrid_meta[col_name].isna())
        else:
            assert not np.any(h.hybrid_meta[col_name].isna())


def test_warning_for_improper_data_output_from_hybrid_method():
    """Test that hybrid function with incorrect output throws warning. """

    def some_new_hybrid_func(__):
        return [0]
    HYBRID_METHODS['scaled_elevation'] = some_new_hybrid_func

    with pytest.warns(OutputWarning) as record:
        Hybridization(SOLAR_FPATH, WIND_FPATH).run_all()

    warn_msg = record[0].message.args[0]
    assert "Unable to add" in warn_msg
    assert "column to hybrid meta" in warn_msg


def test_hybrid_col_additional_method():
    """Test that function decorated with 'hybrid_col' adds to hybrid meta. """

    def some_new_hybrid_func(h):
        return h.hybrid_meta['elevation'] * 1000
    HYBRID_METHODS['scaled_elevation'] = some_new_hybrid_func

    h = Hybridization(SOLAR_FPATH, WIND_FPATH).run_all()

    assert 'scaled_elevation' in HYBRID_METHODS
    assert 'scaled_elevation' in h.hybrid_meta.columns
    assert np.allclose(h.hybrid_meta['elevation'] * 1000,
                       h.hybrid_meta['scaled_elevation'])


def test_duplicate_lat_long_values():
    """Test duplicate lat/long values corresponding to unique merge column. """

    with tempfile.TemporaryDirectory() as td:
        fout_solar = os.path.join(td, 'rep_profiles_solar.h5')
        make_test_file(SOLAR_FPATH, fout_solar, duplicate_coord_values=True)

        with pytest.raises(FileInputError) as excinfo:
            Hybridization(fout_solar, WIND_FPATH).run_all()

        assert "Detected mismatched coordinate values" in str(excinfo.value)


def test_invalid_ratio_bounds_length_input():
    """Test improper ratios input. """

    ratio = 'solar_capacity/wind_capacity'
    with pytest.raises(InputError) as excinfo:
        Hybridization(
            SOLAR_FPATH, WIND_FPATH, ratio=ratio, ratio_bounds=(1, 2, 3)
        )

    msg = ("Length of input for ratio_bounds is 3 "
           "- but is required to be of length 2.")
    assert msg in str(excinfo.value)


def test_ratio_column_missing():
    """Test missing ratio column. """

    ratio = 'solar_col_dne/wind_capacity'
    with pytest.raises(FileInputError) as excinfo:
        Hybridization(
            SOLAR_FPATH, WIND_FPATH, ratio=ratio, ratio_bounds=(1, 1)
        )

    assert "Input ratios column" in str(excinfo.value)
    assert "not found" in str(excinfo.value)


@pytest.mark.parametrize("ratio", [None, ('solar_capacity', 'wind_capacity')])
def test_ratio_not_string(ratio):
    """Test ratio input is not string. """

    with pytest.raises(InputError) as excinfo:
        Hybridization(
            SOLAR_FPATH, WIND_FPATH, ratio=ratio, ratio_bounds=(1, 1)
        )

    assert "Ratio input type " in str(excinfo.value)
    assert "not understood" in str(excinfo.value)


@pytest.mark.parametrize(
    "ratio",
    ['solar_capacity',
     'solar_capacity/wind_capacity/solar_capacity']
)
def test_invalid_ratio_format(ratio):
    """Test ratio input is not string. """

    with pytest.raises(InputError) as excinfo:
        Hybridization(
            SOLAR_FPATH, WIND_FPATH, ratio=ratio, ratio_bounds=(1, 1)
        )

    long_msg = ("Please make sure the ratio input is a string in the form "
                "'numerator_column_name/denominator_column_name'")
    assert "Ratio input " in str(excinfo.value)
    assert long_msg in str(excinfo.value)


def test_invalid_ratio_column_name():
    """Test invalid inputs for ratio columns. """

    ratio = 'un_prefixed_col/wind_capacity'
    with pytest.raises(InputError) as excinfo:
        Hybridization(
            SOLAR_FPATH, WIND_FPATH, ratio=ratio, ratio_bounds=(1, 1)
        )

    assert "Input ratios column" in str(excinfo.value)
    assert "does not start with a valid prefix" in str(excinfo.value)


def test_no_overlap_in_merge_column_values():
    """Test duplicate values in merge column. """

    with tempfile.TemporaryDirectory() as td:
        fout_solar = os.path.join(td, 'rep_profiles_solar.h5')
        fout_wind = os.path.join(td, 'rep_profiles_wind.h5')
        make_test_file(SOLAR_FPATH, fout_solar, p_slice=slice(0, 3))
        make_test_file(WIND_FPATH, fout_wind, p_slice=slice(90, 100))

        with pytest.raises(FileInputError) as excinfo:
            Hybridization(fout_solar, fout_wind)

        assert "No overlap detected in the values" in str(excinfo.value)


def test_duplicate_merge_column_values():
    """Test duplicate values in merge column. """

    with tempfile.TemporaryDirectory() as td:
        fout_solar = os.path.join(td, 'rep_profiles_solar.h5')
        make_test_file(SOLAR_FPATH, fout_solar, duplicate_rows=True)

        with pytest.raises(FileInputError) as excinfo:
            Hybridization(fout_solar, WIND_FPATH)

        assert "Duplicate" in str(excinfo.value)


def test_merge_columns_missing():
    """Test missing merge column. """

    with tempfile.TemporaryDirectory() as td:
        fout_solar = os.path.join(td, 'rep_profiles_solar.h5')
        make_test_file(SOLAR_FPATH, fout_solar, drop_cols=[MERGE_COLUMN])

        with pytest.raises(FileInputError) as excinfo:
            Hybridization(fout_solar, WIND_FPATH)

        msg = "Cannot hybridize: merge column"
        assert msg in str(excinfo.value)
        assert "missing" in str(excinfo.value)


def test_invalid_num_profiles():
    """Test input files with an invalid number of profiles (>1). """

    with pytest.raises(FileInputError) as excinfo:
        Hybridization(SOLAR_FPATH_MULT, WIND_FPATH)

        msg = ("This module is not intended for hybridization of "
               "multiple representative profiles. Please re-run "
               "on a single aggregated profile.")
        assert msg in str(excinfo.value)


def test_invalid_time_index_overlap():
    """Test input files with an invalid time index overlap. """

    with tempfile.TemporaryDirectory() as td:
        fout_solar = os.path.join(td, 'rep_profiles_solar.h5')
        fout_wind = os.path.join(td, 'rep_profiles_wind.h5')
        make_test_file(SOLAR_FPATH, fout_solar, t_slice=slice(0, 1500))
        make_test_file(WIND_FPATH, fout_wind, t_slice=slice(1000, 3000))

        with pytest.raises(FileInputError) as excinfo:
            Hybridization(fout_solar, fout_wind)

        msg = ("Please ensure that the input profiles have a "
               "time index that overlaps >= 8760 times.")
        assert msg in str(excinfo.value)


def test_valid_time_index_overlap():
    """Test input files with a valid time index overlap. """

    h = Hybridization(SOLAR_FPATH_30_MIN, WIND_FPATH)

    with Resource(SOLAR_FPATH_30_MIN) as res:
        assert np.all(res.time_index == h.solar_time_index)

    with Resource(WIND_FPATH) as res:
        assert np.all(res.time_index == h.wind_time_index)
        assert len(res.time_index) == len(h.hybrid_time_index)


def test_write_to_file():
    """Test hybrid rep profiles with file write."""
    with tempfile.TemporaryDirectory() as td:
        fout = os.path.join(td, 'temp_hybrid_profiles.h5')
        h = Hybridization(SOLAR_FPATH, WIND_FPATH).run_all(fout=fout)

        with Resource(fout) as res:
            for name, p in zip(OUTPUT_PROFILE_NAMES, h.profiles.values()):
                dtype = res.get_dset_properties(name)[1]
                attrs = res.get_attrs(name)
                disk_profiles = res[name]

                assert np.issubdtype(dtype, np.float32)
                assert attrs['units'] == 'MW'
                assert np.allclose(p, disk_profiles)

            disk_dsets = res.datasets
            assert 'rep_profiles_0' not in disk_dsets


def test_hybrids_data_content():
    """Test HybridsData class content. """

    fv = -999
    h_data = HybridsData(SOLAR_FPATH, WIND_FPATH)

    with Resource(SOLAR_FPATH) as sr, Resource(WIND_FPATH) as wr:
        assert np.all(h_data.solar_meta.fillna(fv) == sr.meta.fillna(fv))
        assert np.all(h_data.wind_meta.fillna(fv) == wr.meta.fillna(fv))
        assert np.all(h_data.solar_time_index == sr.time_index)
        assert np.all(h_data.wind_time_index == wr.time_index)
        hyb_idx = sr.time_index.join(wr.time_index, how='inner')
        assert np.all(h_data.hybrid_time_index == hyb_idx)


def test_hybrids_data_contains_col():
    """Test the 'contains_col' method of HybridsData for accuracy."""

    h_data = HybridsData(SOLAR_FPATH, WIND_FPATH)
    assert h_data.contains_col('trans_capacity')
    assert h_data.contains_col('dist_mi')
    assert h_data.contains_col('dist_km')
    assert not h_data.contains_col('dne_col_for_test')


@pytest.mark.parametrize("input_files", [
    (SOLAR_FPATH, WIND_FPATH),
    (SOLAR_FPATH_30_MIN, WIND_FPATH)
])
@pytest.mark.parametrize("ratio", [
    'solar_capacity/wind_capacity',
    'solar_area_sq_km/wind_area_sq_km'
])
@pytest.mark.parametrize("ratio_bounds", [None, (0.5, 1.5), (0.3, 3.6)])
@pytest.mark.parametrize("input_combination", [(False, False), (True, True)])
def test_hybrids_cli_from_config(runner, input_files, ratio, ratio_bounds,
                                 input_combination, clear_loggers):
    """Test hybrids cli from config"""
    fv = -999
    sfp, wfp = input_files
    allow_solar_only, allow_wind_only = input_combination
    fill_vals = {'solar_n_gids': 0, 'wind_capacity': -1}
    limits = {'solar_capacity': 100}

    with tempfile.TemporaryDirectory() as td:
        config = {
            "analysis_years": 2012,
            "solar_fpath": sfp,
            "wind_fpath": wfp,
            "log_directory": td,
            "execution_control": {
                "nodes": 1,
                "option": "local",
                "sites_per_worker": 10
            },
            "log_level": "INFO",
            "allow_solar_only": allow_solar_only,
            "allow_wind_only": allow_wind_only,
            "fillna": fill_vals,
            'limits': limits,
            'ratio': ratio,
            'ratio_bounds': ratio_bounds
        }

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['-c', config_path, 'hybrids'])
        clear_loggers()

        if result.exit_code != 0:
            import traceback
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            clear_loggers()
            raise RuntimeError(msg)

        h = Hybridization(
            sfp, wfp,
            allow_solar_only=allow_solar_only,
            allow_wind_only=allow_wind_only,
            fillna=fill_vals, limits=limits,
            ratio=ratio,
            ratio_bounds=ratio_bounds
        ).run_all()
        dirname = os.path.basename(td)
        fn_out = "{}_{}.h5".format(dirname, HybridsConfig.NAME)
        out_fpath = os.path.join(td, fn_out)
        with Outputs(out_fpath, 'r') as f:
            for dset_name in OUTPUT_PROFILE_NAMES:
                assert dset_name in f.dsets

            meta_from_file = f.meta.fillna(fv).replace('nan', fv)
            assert np.all(meta_from_file == h.hybrid_meta.fillna(fv))
            assert np.all(f.time_index == h.hybrid_time_index)

        clear_loggers()


@pytest.mark.parametrize("bad_fpath", [
    os.path.join(TESTDATADIR, 'rep_profiles_out', 'rep_profiles_sol*.h5'),
    os.path.join(TESTDATADIR, 'rep_profiles_out', 'rep_profiles_dne.h5'),
])
def test_hybrids_cli_bad_fpath_input(runner, bad_fpath, clear_loggers):
    """Test cli when filepath input is ambiguous or invalid. """

    with tempfile.TemporaryDirectory() as td:
        config = {
            "solar_fpath": bad_fpath,
            "wind_fpath": WIND_FPATH,
            "log_directory": td,
            "execution_control": {
                "nodes": 1,
                "option": "local",
                "sites_per_worker": 10
            },
            "log_level": "INFO",
        }

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['-c', config_path, 'hybrids'])
        clear_loggers()

        if result.exit_code != 0:
            import traceback
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            clear_loggers()
            raise RuntimeError(msg)

        dirname = os.path.basename(td)
        fn_out = "{}_{}.h5".format(dirname, HybridsConfig.NAME)
        assert "WARNING" in result.stdout
        assert fn_out not in os.listdir(td)

        clear_loggers()


@pytest.mark.parametrize("input_files", [
    (SOLAR_FPATH, WIND_FPATH),
    (SOLAR_FPATH_30_MIN, WIND_FPATH)
])
@pytest.mark.parametrize("ratio", [
    'solar_capacity/wind_capacity',
    'solar_area_sq_km/wind_area_sq_km'
])
@pytest.mark.parametrize("ratio_bounds", [None, (0.5, 1.5), (0.3, 3.6)])
@pytest.mark.parametrize("input_combination", [(False, False), (True, True)])
def test_hybrids_cli_direct(runner, input_files, ratio, ratio_bounds,
                            input_combination, clear_loggers):
    """Test hybrids cli 'direct' command. """

    fv = -999
    sfp, wfp = input_files
    allow_solar_only, allow_wind_only = input_combination
    fill_vals = {'solar_n_gids': 0, 'wind_capacity': -1}
    limits = {'solar_capacity': 100}

    with tempfile.TemporaryDirectory() as td:

        args = ['-s {}'.format(SLURM.s(sfp)),
                '-w {}'.format(SLURM.s(wfp)),
                '-fna {}'.format(SLURM.s(fill_vals)),
                '-l {}'.format(SLURM.s(limits)),
                '-od {}'.format(SLURM.s(td)),
                '-ld {}'.format(SLURM.s(td)),
                '-r {}'.format(SLURM.s(ratio))
                ]

        if ratio_bounds is not None:
            args.append('-rb {}'.format(SLURM.s(ratio_bounds)))

        if allow_solar_only:
            args.append('-so')

        if allow_wind_only:
            args.append('-wo')

        cmd = '-n {} direct {}'.format(SLURM.s("hybrids-test"), ' '.join(args))
        result = runner.invoke(hybrids_cli_main, cmd)
        clear_loggers()

        if result.exit_code != 0:
            import traceback
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            clear_loggers()
            raise RuntimeError(msg)

        h = Hybridization(
            sfp, wfp,
            allow_solar_only=allow_solar_only,
            allow_wind_only=allow_wind_only,
            fillna=fill_vals, limits=limits,
            ratio_bounds=ratio_bounds,
            ratio=ratio
        ).run_all()

        out_fpath = os.path.join(td, 'hybrids-test.h5')
        with Outputs(out_fpath, 'r') as f:
            for dset_name in OUTPUT_PROFILE_NAMES:
                assert dset_name in f.dsets

            meta_from_file = f.meta.fillna(fv).replace('nan', fv)
            assert np.all(meta_from_file == h.hybrid_meta.fillna(fv))
            assert np.all(f.time_index == h.hybrid_time_index)

        clear_loggers()


def make_test_file(in_fp, out_fp, p_slice=slice(None), t_slice=slice(None),
                   drop_cols=None, duplicate_rows=False,
                   duplicate_coord_values=False):
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
        dset_names = [d for d in res.dsets if d not in ('meta', 'time_index')]
        shapes = res.shapes
        meta = res.meta.iloc[p_slice]
        if drop_cols is not None:
            meta.drop(columns=drop_cols, inplace=True)
        if duplicate_rows:
            n_rows, __ = meta.shape
            half_n_rows = n_rows // 2
            meta.iloc[-half_n_rows:] = meta.iloc[:half_n_rows].values
        if duplicate_coord_values:
            meta.loc[0, 'latitude'] = meta['latitude'].iloc[-1]
            meta.loc[0, 'latitude'] = meta['latitude'].iloc[-1]
        shapes['meta'] = len(meta)
        for d in dset_names:
            shapes[d] = (len(res.time_index[t_slice]), len(meta))

        Outputs.init_h5(out_fp, dset_names, shapes, res.attrs, res.chunks,
                        res.dtypes, meta,
                        time_index=res.time_index[t_slice])

        with Outputs(out_fp, mode='a') as out:
            for d in dset_names:
                out[d] = res[d, t_slice, p_slice]

            d = 'rep_profiles_0'
            assert out._h5[d].shape == (len(res.time_index[t_slice]),
                                        len(meta))
            assert np.all(out[d].sum(axis=0) > 0)


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
    pytest.main(['-q', '--show-capture={}'.format(capture), __file__, flags])


if __name__ == '__main__':
    execute_pytest()
