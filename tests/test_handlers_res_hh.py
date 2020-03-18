# -*- coding: utf-8 -*-
"""
pytests for resource handlers with a single hub height
"""
import numpy as np
import os
import pytest
from reV.handlers.rev_resource import WindResource
from reV import TESTDATADIR
from reV.config.project_points import ProjectPoints


def test_single_hh():
    """Test that resource with data at a single hub height will always return
    the data at that hub height (and also return a warning)"""
    h5 = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012_incomplete_1.h5')
    with WindResource(h5) as wind:
        # Existing datasets are P0m and T80m
        assert np.array_equal(wind['pressure_80m'], wind['pressure_0m'])
        assert np.array_equal(wind['temperature_10m'], wind['temperature_80m'])


def test_check_hh():
    """Test that check hub height method will return the hh at the single
    windspeed"""
    h5 = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012_incomplete_2.h5')
    msg = ('Wind resource method _check_hub_height() failed! Should have '
           'returned 100 because theres only windspeed at 100m')
    with WindResource(h5) as wind:
        assert (wind._check_hub_height(120) == 100), msg


def test_sam_df_hh():
    """Test that if there's only windspeed at one HH, all data is returned
    from that hh
    """

    h5 = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012_incomplete_2.h5')
    with WindResource(h5) as wind:
        sam_df = wind._get_SAM_df('pressure_80m', 0)

        arr1 = wind['pressure_100m', :, 0] * 9.86923e-6
        arr2 = sam_df['pressure_100m'].values

        msg1 = ('Error: pressure should have been loaded at 100m '
                'b/c there is only windspeed at 100m.')

        assert np.array_equal(arr1, arr2), msg1


def test_preload_sam_hh():
    """Test the preload_SAM method with a single hub height windspeed in res.

    In this case, all variables should be loaded at the single windspeed hh
    """

    h5 = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012_incomplete_2.h5')
    sam_configs = {'wind': os.path.join(TESTDATADIR, 'SAM/i_windpower.json')}
    project_points = ProjectPoints(slice(0, 200), sam_configs, 'wind')
    msg = ('Test SAM config {} changed! Expected HH at 80m but received: {}'
           .format(list(sam_configs.values())[0], project_points.h[0]))
    assert all([h == 80 for h in project_points.h]), msg

    SAM_res = WindResource.preload_SAM(h5, project_points)

    with WindResource(h5) as wind:
        p = wind['pressure_100m'] * 9.86923e-6
        t = wind['temperature_100m']
        msg1 = ('Error: pressure should have been loaded at 100m '
                'b/c there is only windspeed at 100m.')
        msg2 = ('Error: temperature should have been loaded at 100m '
                'b/c there is only windspeed at 100m.')
        assert np.allclose(SAM_res['pressure', :, :].values, p), msg1
        assert np.allclose(SAM_res['temperature', :, :].values, t), msg2


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
