# -*- coding: utf-8 -*-
# pylint: skip-file
"""reV SAM unit test module
"""
import json
import os
import shutil
import tempfile
import numpy as np
import pytest

from reV import TESTDATADIR
from reV.bespoke.bespoke import BespokeWindFarms
from reV.supply_curve.tech_mapping import TechMapping

from rex import init_logger

pytest.importorskip("shapely")
pytest.importorskip("rasterio")


SAM = os.path.join(TESTDATADIR, 'SAM/i_windpower.json')
EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
RES = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_{}.h5')
TM_DSET = 'techmap_wtk_ri_100'
AGG_DSET = ('cf_mean', 'cf_profile')

# note that this differs from the
EXCL_DICT = {'ri_srtm_slope': {'inclusion_range': (None, 5),
                               'exclude_nodata': False},
             'ri_padus': {'exclude_values': [1],
                          'exclude_nodata': False},
             'ri_reeds_regions': {'inclusion_range': (None, 400),
                                  'exclude_nodata': False}}

with open(SAM, 'r') as f:
    SAM_SYS_INPUTS = json.load(f)

SAM_SYS_INPUTS['wind_farm_wake_model'] = 2
SAM_CONFIGS = {'default': SAM_SYS_INPUTS}


def cost_function(x):
    """dummy cost function"""
    R = 0.1
    return 200 * x * np.exp(-x / 1E5 * R + (1 - R))


def objective_function(aep, cost):
    """dummy objective function"""
    return cost / aep


def test_single_serial(gid=33):
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')
    with tempfile.TemporaryDirectory() as td:
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')
        points = [gid]

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        out = BespokeWindFarms.run(excl_fp, res_fp, TM_DSET,
                                   objective_function, cost_function,
                                   points, SAM_CONFIGS,
                                   ga_time=5,
                                   excl_dict=EXCL_DICT,
                                   output_request=output_request,
                                   max_workers=1, sites_per_worker=1)

        assert gid in out
        assert 'cf_profile-2012' in out[gid]
        assert 'cf_profile-2013' in out[gid]
        assert 'cf_mean-2012' in out[gid]
        assert 'cf_mean-2013' in out[gid]
        assert 'cf_mean-means' in out[gid]
        assert 'annual_energy-2012' in out[gid]
        assert 'annual_energy-2013' in out[gid]
        assert 'annual_energy-means' in out[gid]
        assert len(out[gid]['cf_profile-2012']) == 8760
        assert len(out[gid]['cf_profile-2013']) == 8760


if __name__ == '__main__':
    init_logger('reV', log_level='DEBUG')
    points = np.arange(33, 40)
    points = [33]  # 39% included
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')

    with open(SAM, 'r') as f:
        sam_sys_inputs = json.load(f)

    sam_sys_inputs['wind_farm_wake_model'] = 2
    sam_configs = {'default': sam_sys_inputs}

    rotor_diameter = sam_sys_inputs["wind_turbine_rotor_diameter"]

    with tempfile.TemporaryDirectory() as td:
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        out = BespokeWindFarms.run(excl_fp, res_fp, TM_DSET,
                                   objective_function, cost_function,
                                   points, sam_configs,
                                   min_spacing='5x', ga_time=20,
                                   excl_dict=EXCL_DICT,
                                   output_request=output_request,
                                   max_workers=1, sites_per_worker=2)
        print(out)
        print(list(out.keys()))
