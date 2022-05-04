# -*- coding: utf-8 -*-
"""
PyTest file for reV loss utilities.

Created on Mon Apr 18 12:52:16 2021

@author: ppinchuk
"""

import os
import pytest

from reV.losses.utils import (format_month_name, full_month_name_from_abbr,
                              month_index, convert_to_full_month_names,
                              filter_unknown_month_names, month_indices,
                              hourly_indices_for_months)


def test_hourly_indices_for_months():
    """Test that the correct indices are returned for the input months. """

    assert not hourly_indices_for_months([])
    assert not hourly_indices_for_months(['Abc'])

    indices = hourly_indices_for_months(['January', 'Abc'])
    assert indices[0] == 0
    assert indices[-1] == len(indices) - 1
    assert len(indices) == 31 * 24  # 31 days in Jan
    assert all(i < 31 * 24 for i in indices)

    indices = hourly_indices_for_months(['March', 'January'])
    assert indices[0] == 0
    assert len(indices) == (31 + 31) * 24  # 31 days in Jan and Mar
    assert 744 not in indices
    assert indices[744] - indices[743] - 1 == 28 * 24  # we skip Feb

    all_months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November',
                  'December']
    indices = hourly_indices_for_months(all_months)
    assert indices[0] == 0
    assert indices[-1] == len(indices) - 1
    assert len(indices) == 8760


def test_month_indices():
    """Test that month indices are generated correctly. """

    assert not month_indices(['Abc'])
    assert month_indices(['March', 'April', 'June', 'July']) == {2, 3, 5, 6}
    assert -1 not in month_indices(['March', 'April', 'June', 'July', 'Abc'])
    assert month_indices(['March', 'April', 'March']) == {2, 3}


def test_filter_unknown_month_names():
    """Test that month names are filtered correctly. """

    input_names = ['March', 'April', 'June', 'July', 'Abc', ' unformaTTed']
    expected_known_names = ['March', 'April', 'June', 'July']
    expected_unknown_names = ['Abc', ' unformaTTed']

    known_months, unknown_months = filter_unknown_month_names(input_names)

    assert known_months == expected_known_names
    assert unknown_months == expected_unknown_names


def test_convert_to_full_month_names():
    """Test that an iterable of names is formatted correctly. """

    input_names = ['March', ' aprIl  ', 'Jun', 'jul', '  abc ']
    expected_output_names = ['March', 'April', 'June', 'July', 'Abc']
    assert convert_to_full_month_names(input_names) == expected_output_names


def test_month_index():
    """Test that the correct month index is returned for input. """

    assert month_index("June") == 5
    assert month_index("July") == 6
    assert month_index("Jun") == -1
    assert month_index("jul") == -1
    assert month_index('') == -1
    assert month_index('Abcdef') == -1
    assert month_index(' aprIl  ') == -1


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
