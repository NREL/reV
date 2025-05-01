# -*- coding: utf-8 -*-
# pylint: skip-file
"""
A collection of useful fixtures for tests
"""
import pytest
from click.testing import CliRunner

from rex.utilities.loggers import LOGGERS


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


@pytest.fixture
def clear_loggers():
    pass
    """Fixture to clear loggers when called.

    This is mostly helpful for tests that initialize loggers, since
    windows doesn't release log file handlers unless they are cleared.
    """
    def _clear():
        LOGGERS["gaps"] = {}  # clear gaps loggers as well
        LOGGERS.clear()
    return _clear
