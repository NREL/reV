# -*- coding: utf-8 -*-
# pylint: skip-file
"""
A collection of useful fixtures for tests
"""
import pytest
from click.testing import CliRunner


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()
