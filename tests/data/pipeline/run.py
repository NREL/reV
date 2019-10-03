# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:24:16 2019

@author: gbuster
"""

import os

from reV import TESTDATADIR
from reV.pipeline.pipeline import Pipeline
fpipeline = os.path.join(TESTDATADIR, 'pipeline/config_pipeline.json')
Pipeline.run(fpipeline, monitor=True)
