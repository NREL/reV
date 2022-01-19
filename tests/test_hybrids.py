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


SOLAR_FPATH = os.path.join(TESTDATADIR,
                           'rep_profiles_out/rep_profiles_solar.h5')
WIND_FPATH = os.path.join(TESTDATADIR, 'rep_profiles_out/rep_profiles_wind.h5')


def test_main():
    Hybridization(SOLAR_FPATH, WIND_FPATH)


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
    # execute_pytest()
    test_main()
