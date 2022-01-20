# -*- coding: utf-8 -*-
"""reV hybrids tests.
"""
import os
import pytest
import pandas as pd
import numpy as np
import json
import tempfile

from reV.hybrids.hybrids import Hybridization, hybrid_col, HYBRID_METHODS
from reV.utilities.exceptions import FileInputError
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


def test_hybrid_col_decorator():
    """Test that function decorated with 'hybrid_col' adds to hybrid meta. """

    @hybrid_col('scaled_elevation')
    def some_new_hybrid_func(h):
        return h.hybrid_meta['elevation'] * 1000

    h = Hybridization(SOLAR_FPATH, WIND_FPATH)
    h._run()

    assert 'scaled_elevation' in HYBRID_METHODS
    assert 'scaled_elevation' in h.hybrid_meta.columns
    assert (h.hybrid_meta['elevation'] * 1000
            == h.hybrid_meta['scaled_elevation']).all()


def test_duplicate_lat_long_values():
    """Test duplicate lat/long values corresponding to unique merge column. """
    with tempfile.TemporaryDirectory() as td:
        fout_solar = os.path.join(td, 'rep_profiles_solar.h5')
        make_test_file(SOLAR_FPATH, fout_solar, duplicate_coord_values=True)

        with pytest.raises(FileInputError) as excinfo:
            Hybridization.run(fout_solar, WIND_FPATH)

        assert "Detected mismatched coordinate values" in str(excinfo.value)


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

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
