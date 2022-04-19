# -*- coding: utf-8 -*-
"""
PyTest file for scheduled losses.

Created on Mon Apr 18 12:52:16 2021

@author: ppinchuk
"""

import os
import pytest

from reV.SAM.losses import format_month_name, full_month_name_from_abbr


def test_full_month_name_from_abbr():
    """Test that month names are retrieved from abbreviations. """

    assert full_month_name_from_abbr('Jun') == 'June'
    assert full_month_name_from_abbr('') is None
    assert full_month_name_from_abbr('June') is None
    assert full_month_name_from_abbr('Abcdef') is None


def test_format_month_name():
    """Test that month names are formatter appropriately. """

    assert format_month_name(' aprIl  ') == 'April'
    assert format_month_name('Jun') == 'Jun'


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
