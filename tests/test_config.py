# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import pytest

from reV.utilities.exceptions import HandlerKeyError
from reV.SAM.SAM import SAM
from reV.config.project_points import ProjectPoints, PointsControl
from reV import TESTDATADIR


def test_clearsky():
    res_file = os.path.join(TESTDATADIR, 'nsrdb/', 'ri_100_nsrdb_2012.h5')
    sam_config_dict = {0: os.path.join(TESTDATADIR, 'SAM/'
                                       'naris_pv_1axis_inv13_cs.json')}
    pp = ProjectPoints(slice(0, 10), sam_config_dict, 'pv', res_file=res_file)
    try:
        # Get the SAM resource object
        SAM.get_sam_res(res_file, pp, pp.tech)
        assert False
    except HandlerKeyError as e:
        # Should look for clearsky_dni and not find it in RI data
        assert True


@pytest.mark.parametrize('start, interval',
                         [[0, 1], [13, 1], [10, 2], [13, 3]])
def test_proj_control_iter(start, interval):
    """Test the iteration of the points control."""
    n = 3
    res_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    sam_files = os.path.join(TESTDATADIR,
                             'SAM/wind_gen_standard_losses_0.json')
    pp = ProjectPoints(slice(start, 100, interval), sam_files, 'wind',
                       res_file=res_file)
    pc = PointsControl(pp, sites_per_split=n)

    for i, pp_split in enumerate(pc):
        i0_nom = i * n
        i1_nom = i * n + n
        split = pp_split.project_points.df
        target = pp.df.iloc[i0_nom:i1_nom, :]
        msg = 'PointsControl iterator split did not function correctly!'
        assert all(split == target), msg


@pytest.mark.parametrize('start, interval',
                         [[0, 1], [13, 1], [10, 2], [13, 3]])
def test_proj_points_split(start, interval):
    """Test the split operation of project points."""
    res_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    sam_files = os.path.join(TESTDATADIR,
                             'SAM/wind_gen_standard_losses_0.json')
    pp = ProjectPoints(slice(start, 100, interval), sam_files, 'wind',
                       res_file=res_file)

    iter_interval = 5
    for i0 in range(start, 100, iter_interval):
        i1 = i0 + iter_interval

        pp_0 = ProjectPoints.split(i0, i1, pp)

        if not pp_0.sites:
            break

        msg = 'ProjectPoints split did not function correctly!'
        assert pp_0.sites == pp.sites[i0:i1], msg
        assert all(pp_0.df == pp.df.iloc[i0:i1, :]), msg


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
