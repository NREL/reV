# -*- coding: utf-8 -*-
"""reVX hybrids tests.
"""
import os
import pytest
import pandas as pd
import numpy as np
import json
import tempfile

from reV.hybrids.hybrids import Hybridization
from reV import Outputs, TESTDATADIR

from rex.resource import Resource


SOLAR_FPATH = os.path.join(
    TESTDATADIR, 'rep_profiles_out', 'rep_profiles_solar.h5')
WIND_FPATH = os.path.join(
    TESTDATADIR, 'rep_profiles_out', 'rep_profiles_wind.h5')
SOLAR_FPATH_30_MIN = os.path.join(
    TESTDATADIR, 'rep_profiles_out', 'rep_profiles_solar_30_min.h5')
WIND_FPATH_30_MIN = os.path.join(
    TESTDATADIR, 'rep_profiles_out', 'rep_profiles_wind_30_min.h5')


def test_invalid_time_index_overlap():
    with tempfile.TemporaryDirectory() as td:
        fout_solar = os.path.join(td, 'rep_profiles_solar.h5')
        fout_wind = os.path.join(td, 'rep_profiles_wind.h5')
        make_test_file(SOLAR_FPATH, fout_solar, t_slice=slice(0, 1500))
        make_test_file(WIND_FPATH, fout_wind, t_slice=slice(1000, 3000))

        with pytest.raises(ValueError) as excinfo:
            Hybridization(fout_solar, fout_wind)

        msg = ("Please ensure that the input profiles have a "
               "time index that overlaps >= 8760 times.")
        assert msg in str(excinfo.value)


def make_test_file(in_fp, out_fp, p_slice=slice(None), t_slice=slice(None)):
    with Resource(in_fp) as res:
        dset_names = [d for d in res.dsets if d not in ('meta', 'time_index')]
        shapes = res.shapes
        meta = res.meta.iloc[p_slice]
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
