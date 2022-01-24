# -*- coding: utf-8 -*-
"""reV hybrids tests.
"""
import os
import pytest
import numpy as np
import tempfile

from reV.hybrids import Hybridization, hybrid_col, HYBRID_METHODS
from reV.utilities.exceptions import FileInputError, InputError, OutputWarning
from reV import Outputs, TESTDATADIR

from rex.resource import Resource


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


def test_parallel_run():
    """Test that serial and parallel execution match. """

    out_serial = Hybridization.run(SOLAR_FPATH, WIND_FPATH, max_workers=1)
    out_parallel = Hybridization.run(SOLAR_FPATH, WIND_FPATH, max_workers=10)

    *out_serial, serial_h_meta, serial_h_time_index = out_serial
    *out_parallel, parallel_h_meta, parallel_h_time_index = out_parallel

    for out_s, out_p in zip(out_serial, out_parallel):
        assert (out_s == out_p).all()

    serial_h_meta.fillna(-999, inplace=True)
    parallel_h_meta.fillna(-999, inplace=True)
    assert (serial_h_meta == parallel_h_meta).all().all()

    assert (serial_h_time_index == parallel_h_time_index).all()


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

    hp, hsp, hwp, h_meta, __ = Hybridization.run(SOLAR_FPATH, WIND_FPATH,
                                                 allow_solar_only=True)
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

    hp, hsp, hwp, h_meta, __ = Hybridization.run(SOLAR_FPATH, WIND_FPATH)
    h_idx = np.where(h_meta['sc_point_gid'] == common_sc_point_gid)[0][0]

    assert np.allclose(hp[:, h_idx], weighted_solar + weighted_wind)
    assert np.allclose(hsp[:, h_idx], weighted_solar)
    assert np.allclose(hwp[:, h_idx], weighted_wind)


@pytest.mark.parametrize("input_files", [(SOLAR_FPATH, WIND_FPATH),
                                         (SOLAR_FPATH_30_MIN, WIND_FPATH)])
def test_hybridization_output_shapes(input_files):
    """Test that the output shapes are as expected. """

    sfp, wfp = input_files
    out = Hybridization.run(sfp, wfp)
    expected_shapes = [(8760, 53)] * 3 + [(53, 73), (8760,)]
    for arr, expected_shape in zip(out, expected_shapes):
        assert arr.shape == expected_shape

    out = Hybridization.run(sfp, wfp, allow_solar_only=True)
    expected_shapes = [(8760, 100)] * 3 + [(100, 73), (8760,)]
    for arr, expected_shape in zip(out, expected_shapes):
        assert arr.shape == expected_shape

    out = Hybridization.run(
        sfp, wfp, allow_solar_only=True, allow_wind_only=True
    )
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
    *__, hybrid_meta, __ = Hybridization.run(
        SOLAR_FPATH, WIND_FPATH,
        allow_solar_only=allow_solar_only,
        allow_wind_only=allow_wind_only
    )
    assert hybrid_meta.shape == expected_shape
    assert set(hybrid_meta['sc_point_gid']) == overlap


@pytest.mark.parametrize("ratio_cols", [
    ('solar_capacity', 'wind_capacity'),
    ('solar_area_sq_km', 'wind_area_sq_km')
])
@pytest.mark.parametrize("ratio, bounds", [
    ((0.5, 1.5), (500, 1500)),
    ((0.3, 3.6), (300, 3600))
])
def test_allowed_ratio(ratio_cols, ratio, bounds):
    """Test that the hybrid meta limits the ratio columns correctly. """
    *__, hybrid_meta, __ = Hybridization.run(
        SOLAR_FPATH, WIND_FPATH, allowed_ratio=ratio,
        ratio_cols=ratio_cols
    )

    c1, c2 = ratio_cols
    ratios = (hybrid_meta['hybrid_{}'.format(c1)]
              / hybrid_meta['hybrid_{}'.format(c2)])

    assert ((ratios * 1000).astype(int).between(*bounds)).all()
    assert (hybrid_meta['hybrid_{}'.format(c1)] <= hybrid_meta[c1]).all()
    assert (hybrid_meta['hybrid_{}'.format(c2)] <= hybrid_meta[c2]).all()


def test_fillna_values():
    """Test that N/A values are filled properly based on user input. """

    fill_vals = {'solar_n_gids': 0, 'wind_capacity': -1}

    *__, hybrid_meta, __ = Hybridization.run(
        SOLAR_FPATH, WIND_FPATH, allow_solar_only=True,
        allow_wind_only=True, fillna=fill_vals
    )

    assert not hybrid_meta['solar_n_gids'].isna().values.any()
    assert not hybrid_meta['wind_capacity'].isna().values.any()
    assert (hybrid_meta['solar_n_gids'].values == 0).any()
    assert (hybrid_meta['wind_capacity'].values == -1).any()


@pytest.mark.parametrize("input_combination, na_vals",
                         [((False, False), (False, False)),
                          ((True, False), (False, True)),
                          ((False, True), (True, False)),
                          ((True, True), (True, True))])
def test_all_allow_solar_allow_wind_combinations(input_combination, na_vals):
    """Test that "allow_x_only" options perform the intended merges. """

    allow_solar_only, allow_wind_only = input_combination
    *__, hybrid_meta, __ = Hybridization.run(
        SOLAR_FPATH, WIND_FPATH,
        allow_solar_only=allow_solar_only,
        allow_wind_only=allow_wind_only
    )
    for col_name, should_have_na_vals in zip(['solar_sc_gid', 'wind_sc_gid'],
                                             na_vals):
        if should_have_na_vals:
            assert hybrid_meta[col_name].isna().values.any()
        else:
            assert not hybrid_meta[col_name].isna().values.any()


def test_warning_for_improper_data_output_from_hybrid_method():
    """Test that hybrid function with incorrect output throws warning. """

    @hybrid_col('scaled_elevation')
    def some_new_hybrid_func(__):
        return [0]

    with pytest.warns(OutputWarning) as record:
        Hybridization.run(SOLAR_FPATH, WIND_FPATH)

    warn_msg = record[0].message.args[0]
    assert "Unable to add" in warn_msg
    assert "column to hybrid meta" in warn_msg


def test_hybrid_col_decorator():
    """Test that function decorated with 'hybrid_col' adds to hybrid meta. """

    @hybrid_col('scaled_elevation')
    def some_new_hybrid_func(h):
        return h.hybrid_meta['elevation'] * 1000

    *__, hybrid_meta, __ = Hybridization.run(SOLAR_FPATH, WIND_FPATH)

    assert 'scaled_elevation' in HYBRID_METHODS
    assert 'scaled_elevation' in hybrid_meta.columns
    assert (hybrid_meta['elevation'] * 1000
            == hybrid_meta['scaled_elevation']).all()


def test_duplicate_lat_long_values():
    """Test duplicate lat/long values corresponding to unique merge column. """

    with tempfile.TemporaryDirectory() as td:
        fout_solar = os.path.join(td, 'rep_profiles_solar.h5')
        make_test_file(SOLAR_FPATH, fout_solar, duplicate_coord_values=True)

        with pytest.raises(FileInputError) as excinfo:
            Hybridization.run(fout_solar, WIND_FPATH)

        assert "Detected mismatched coordinate values" in str(excinfo.value)


def test_invalid_ratio_input():
    """Test improper ratio input. """

    with pytest.raises(InputError) as excinfo:
        Hybridization(SOLAR_FPATH, WIND_FPATH, allowed_ratio=(1, 2, 3))

    assert "Input for 'allowed_ratio' not understood" in str(excinfo.value)


def test_ratio_column_missing():
    """Test missing ratio column. """

    cols = ('solar_col_dne', 'wind_capacity')
    with pytest.raises(FileInputError) as excinfo:
        Hybridization(SOLAR_FPATH, WIND_FPATH,
                      allowed_ratio=1, ratio_cols=cols)

    assert "Input ratio column" in str(excinfo.value)
    assert "not found" in str(excinfo.value)


def test_invalid_ratio_column_name():
    """Test invalid inputs for ratio columns. """

    cols = ('unprefixed_col', 'wind_capacity')
    with pytest.raises(InputError) as excinfo:
        Hybridization(SOLAR_FPATH, WIND_FPATH,
                      allowed_ratio=1, ratio_cols=cols)

    assert "Input ratio column" in str(excinfo.value)
    assert "does not start with a valid prefix" in str(excinfo.value)


def test_invalid_ratio_column_len():
    """Test invalid number of input ratio columns. """

    cols = ('solar_capacity', 'wind_capacity', 'a_third_col')
    with pytest.raises(InputError) as excinfo:
        Hybridization(SOLAR_FPATH, WIND_FPATH,
                      allowed_ratio=1, ratio_cols=cols)

    assert "Input for 'allowed_ratio' not understood" in str(excinfo.value)
    assert "Please make sure this value is a two-tuple" in str(excinfo.value)


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


def test_merge_columns_missings():
    """Test missing merge column. """

    with tempfile.TemporaryDirectory() as td:
        fout_solar = os.path.join(td, 'rep_profiles_solar.h5')
        make_test_file(SOLAR_FPATH, fout_solar,
                       drop_cols=[Hybridization.MERGE_COLUMN])

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
        assert (res.time_index == h.solar_time_index).all()

    with Resource(WIND_FPATH) as res:
        assert (res.time_index == h.wind_time_index).all()
        assert len(res.time_index) == len(h.hybrid_time_index)


def test_write_to_file():
    """Test hybrid rep profiles with file write."""
    with tempfile.TemporaryDirectory() as td:
        fout = os.path.join(td, 'temp_hybrid_profiles.h5')
        *p_out, __, __ = Hybridization.run(
            SOLAR_FPATH, WIND_FPATH, fout=fout
        )
        with Resource(fout) as res:
            for name, p in zip(Hybridization.OUTPUT_PROFILE_NAMES, p_out):
                dtype = res.get_dset_properties(name)[1]
                attrs = res.get_attrs(name)
                disk_profiles = res[name]

                assert np.issubdtype(dtype, np.float32)
                assert attrs['units'] == 'MW'
                assert np.allclose(p, disk_profiles)

            disk_dsets = res.datasets
            assert 'rep_profiles_0' not in disk_dsets


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
        the data of the imput file.
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
            assert (out[d].sum(axis=0) > 0).all()


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
