# -*- coding: utf-8 -*-
# pylint: skip-file
"""reV SAM unit test module
"""
import os
import pytest
import numpy as np
import pandas as pd
import json
import warnings

from reV.SAM.SAM import SAM, ParametersManager, SiteOutput
from reV.SAM.generation import PV
from reV.handlers.resource import NSRDB
from reV import TESTDATADIR
from reV.config.project_points import ProjectPoints


@pytest.fixture
def res():
    """Initialize a SAM resource object to test SAM functions on."""
    res_file = TESTDATADIR + '/nsrdb/ri_100_nsrdb_2012.h5'
    rev2_points = TESTDATADIR + '/project_points/ri.csv'

    sam_files = [TESTDATADIR + '/SAM/naris_pv_1axis_inv13.json',
                 TESTDATADIR + '/SAM/naris_pv_1axis_inv13.json']
    sam_files = {'sam_param_{}'.format(i): k for i, k in
                 enumerate(sam_files)}

    pp = ProjectPoints(rev2_points, sam_files, 'pv')
    res = NSRDB.preload_SAM(res_file, pp)
    return res


@pytest.mark.parametrize('module', ('pvwattsv5', 'tcsmolten_salt', 'lcoefcr',
                                    'windpower', 'singleowner'))
def test_param_manager(module):
    """Test the parameters manager default setting."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = ParametersManager({}, module)
    jf = os.path.join(SAM.DIR, 'defaults', module + '.json')
    with open(jf, 'r') as f:
        defaults = json.load(f)
    for key in pm.keys():
        assert pm[key] == defaults[key]


def test_site_output():
    """Test the getter and setter methods of the custom site output class."""
    so = SiteOutput()
    try:
        so['test']
    except KeyError as e:
        if '"test" has not been saved' not in str(e):
            assert False
    try:
        so['test'] = 1
    except KeyError as e:
        if 'Could not save "test"' not in str(e):
            assert False
    so['cf_mean'] = 100.0
    assert so['cf_mean'] == 100.0


def test_res_length(res):
    """Test the method to ensure resource array length with truncation."""
    for res_df, meta in res:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_dropped = PV.ensure_res_len(res_df.values)
        break
    compare = np.allclose(res_dropped[:9000, :], res_df.values[:9000, :])
    return compare


def test_leap_year(res):
    """Test the method to ensure resource array length with dropping leap day.
    """
    for res_df, meta in res:
        res_dropped = PV.drop_leap(res_df)
        break
    compare = np.allclose(res_dropped.iloc[-9000:, :].values,
                          res_df.iloc[-9000:, :].values)
    assert compare


@pytest.mark.parametrize('site_index', range(5))
def test_PV_lat_tilt(res, site_index):
    """Test the method to set tilt based on latitude."""
    rev2_points = TESTDATADIR + '/project_points/ri.csv'
    sam_files = [TESTDATADIR + '/SAM/naris_pv_1axis_inv13.json',
                 TESTDATADIR + '/SAM/naris_pv_1axis_inv13.json']
    sam_files = {'sam_param_{}'.format(i): k for i, k in
                 enumerate(sam_files)}
    pp = ProjectPoints(rev2_points, sam_files, 'pv')

    for i, [res_df, meta] in enumerate(res):
        if i == site_index:
            # get SAM inputs from project_points based on the current site
            site = res_df.name
            config, inputs = pp[site]
            inputs['tilt'] = 'latitude'
            # iterate through requested sites.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sim = PV(resource=res_df, meta=meta, parameters=inputs,
                         output_request=('cf_mean',))
            break
        else:
            pass

    assert sim.parameters['tilt'] == meta['latitude']


@pytest.mark.parametrize('dt', ('1h', '30min', '5min'))
def test_time_interval(dt):
    """Test the method to get the 'time interval' from the time index obj."""
    baseline = {'1h': 1, '30min': 2, '5min': 12}
    ti = pd.date_range('1-1-{y}'.format(y=2012), '1-1-{y}'.format(y=2013),
                       freq=dt)[:-1]
    interval = PV.get_time_interval(ti)
    assert interval == baseline[dt]


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
