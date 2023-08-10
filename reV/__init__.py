# -*- coding: utf-8 -*-
"""
The Renewable Energy Potential Model
"""
from __future__ import print_function, division, absolute_import
import logging
if not logging.getLogger().handlers:
    logging.getLogger().addHandler(logging.NullHandler())
import os

from reV.econ import Econ
from reV.generation import Gen
from reV.handlers import Outputs, ExclusionLayers
from reV.qa_qc import QaQc
from reV.rep_profiles import RepProfiles
from reV.supply_curve import (Aggregation, ExclusionMask,
                              ExclusionMaskFromDict, SupplyCurveAggregation,
                              SupplyCurve, TechMapping)
from reV.version import __version__

__author__ = """Galen Maclaurin"""
__email__ = "galen.maclaruin@nrel.gov"


REVDIR = os.path.dirname(os.path.realpath(__file__))
TESTDATADIR = os.path.join(os.path.dirname(REVDIR), 'tests', 'data')
