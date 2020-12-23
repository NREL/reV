# -*- coding: utf-8 -*-
# pylint: skip-file
"""
PyTest file for reV LCOE economies of scale
"""
import pytest
import os

from reV.generation.generation import Gen
from reV.config.project_points import ProjectPoints
from reV import TESTDATADIR


def test_pass_through_lcoe_args():
    """Test that the kwarg works to pass through LCOE input args from the SAM
    input to the reV output."""
    year = 2012
    rev2_points = slice(0, 3)
    res_file = TESTDATADIR + '/wtk/ri_100_wtk_{}.h5'.format(year)
    sam_files = TESTDATADIR + '/SAM/i_windpower_lcoe.json'

    output_request = ('cf_mean', 'lcoe_fcr')

    # run reV 2.0 generation
    pp = ProjectPoints(rev2_points, sam_files, 'windpower', res_file=res_file)
    gen = Gen.reV_run(tech='windpower', points=rev2_points,
                      sam_files=sam_files, res_file=res_file, max_workers=1,
                      sites_per_worker=1, fout=None,
                      pass_through_lcoe_args=True,
                      output_request=output_request)

    checks = [x in gen.out for x in Gen.LCOE_ARGS]
    assert all(checks)
    assert 'lcoe_fcr' in gen.out
    assert 'cf_mean' in gen.out


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
