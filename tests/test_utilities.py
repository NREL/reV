# -*- coding: utf-8 -*-
"""
PyTest file reV utilities module

@author: ppinchuk
"""
import os
import pytest

from reV.utilities import ModuleName, DocEnum
from reV.cli import main


class _EnumForTests(str, DocEnum):
    """Test Enum with a description"""

    VALUE_ONE = 'value_one'
    """First Value to test"""
    VALUE_TWO = 'value_two'


def test_module_names_enum():
    """Verify that all enum values match the commands in the main reV cli."""

    all_commands = set(main.commands.keys())
    msg = ("Enum value {!r} (ModuleName.{}) is not a valid command in the "
           "reV main cli. Main cli commands: {}")

    for enum in ModuleName.__members__.values():
        err_msg = msg.format(enum.value, enum.name, all_commands)
        assert enum.value in all_commands, err_msg


def test_doc_enum():
    """Test the DocEnum class and its description property"""

    assert _EnumForTests.VALUE_ONE == 'value_one'
    assert _EnumForTests.VALUE_TWO == 'value_two'
    assert _EnumForTests.VALUE_ONE.description == "First Value to test"
    assert _EnumForTests.VALUE_TWO.description is None


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
