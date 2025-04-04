# -*- coding: utf-8 -*-
"""
PyTest file wind generation directly from s3 file

Note that this directly tests the example here:
    https://nrel.github.io/reV/misc/examples.running_locally.html
"""

import os
import numpy as np
from reV import TESTDATADIR
from reV.config.project_points import ProjectPoints
from reV.generation.generation import Gen


def test_windpower_s3():
    lat_lons = np.array([[41.25, -71.66]])

    res_file = 's3://nrel-pds-wtk/conus/v1.0.0/wtk_conus_2007.h5'
    sam_file = os.path.join(TESTDATADIR, 'SAM/wind_gen_standard_losses_0.json')

    pp = ProjectPoints.lat_lon_coords(lat_lons, res_file, sam_file)
    gen = Gen('windpower', pp, sam_file, res_file,
              output_request=('cf_mean', 'cf_profile'))
    gen.run(max_workers=1)

    assert isinstance(gen.out['cf_profile'], np.ndarray)
    assert gen.out['cf_profile'].sum() > 0
