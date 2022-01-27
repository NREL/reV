# -*- coding: utf-8 -*-
# pylint: skip-file
"""reV SAM unit test module
"""
import copy
import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from reV import TESTDATADIR
from reV.bespoke.bespoke import BespokeSinglePlant, BespokeWindPlants
from reV.SAM.generation import WindPower
from reV.supply_curve.tech_mapping import TechMapping

from rex import init_logger, Resource

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
SAM_SYS_INPUTS['wind_farm_losses_percent'] = 0
del SAM_SYS_INPUTS['wind_resource_filename']
TURB_RATING = np.max(SAM_SYS_INPUTS['wind_turbine_powercurve_powerout'])
SAM_CONFIGS = {'default': SAM_SYS_INPUTS}


def cost_function(x):
    """dummy cost function"""
    R = 0.1
    return 200 * x * np.exp(-x / 1E5 * R + (1 - R))


def objective_function(aep, cost):
    """dummy objective function"""
    return cost / aep


def test_single(gid=33):
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')
    with tempfile.TemporaryDirectory() as td:
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant.run(gid, excl_fp, res_fp, TM_DSET,
                                     SAM_SYS_INPUTS,
                                     objective_function, cost_function,
                                     ga_time=5,
                                     excl_dict=EXCL_DICT,
                                     output_request=output_request,
                                     )
        out = bsp.outputs

        assert 'cf_profile-2012' in out
        assert 'cf_profile-2013' in out
        assert 'cf_mean-2012' in out
        assert 'cf_mean-2013' in out
        assert 'cf_mean-means' in out
        assert 'annual_energy-2012' in out
        assert 'annual_energy-2013' in out
        assert 'annual_energy-means' in out

        assert TURB_RATING * out['n_turbines'] == out['system_capacity']
        x_coords = json.loads(bsp.meta['turbine_x_coords'])
        y_coords = json.loads(bsp.meta['turbine_y_coords'])
        assert out['n_turbines'] == len(x_coords)
        assert out['n_turbines'] == len(y_coords)

        for y in (2012, 2013):
            cf = out[f'cf_profile-{y}']
            assert cf.min() == 0
            assert cf.max() == 1
            assert np.allclose(cf.mean(), out[f'cf_mean-{y}'])

        # simple windpower obj for comparison
        wp_sam_config = bsp.sam_sys_inputs
        wp_sam_config['wind_farm_wake_model'] = 0
        wp_sam_config['wake_int_loss'] = 0
        wp_sam_config['wind_farm_xCoordinates'] = [0]
        wp_sam_config['wind_farm_yCoordinates'] = [0]
        wp_sam_config['system_capacity'] = TURB_RATING
        res_df = bsp.res_df[(bsp.res_df.index.year == 2012)].copy()
        wp = WindPower(res_df, bsp.meta, wp_sam_config,
                       output_request=bsp._out_req)
        wp.run()

        # make sure the wind resource was loaded correctly
        res_ideal = np.array(wp['wind_resource_data']['data'])
        bsp_2012 = bsp.wind_plant_ts[2012]
        res_bsp = np.array(bsp_2012['wind_resource_data']['data'])
        ws_ideal = res_ideal[:, 2]
        ws_bsp = res_bsp[:, 2]
        assert np.allclose(ws_ideal, ws_bsp)

        # make sure that the zero-losses analysis has greater CF
        cf_bespoke = out['cf_profile-2012']
        cf_ideal = wp.outputs['cf_profile']
        diff = cf_ideal - cf_bespoke
        assert all(diff > -0.00001)
        assert diff.mean() > 0.02


def test_bespoke():
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')
    with tempfile.TemporaryDirectory() as td:
        out_fpath = os.path.join(td, 'bespoke_out.h5')
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')
        points = [33, 35]

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        _ = BespokeWindPlants.run(excl_fp, res_fp, TM_DSET,
                                  objective_function, cost_function,
                                  points, SAM_CONFIGS,
                                  ga_time=5,
                                  excl_dict=EXCL_DICT,
                                  output_request=output_request,
                                  max_workers=1,
                                  out_fpath=out_fpath)

        with Resource(out_fpath) as f:
            print(list(f))


def test_bespoke_points():
    """Test the bespoke points input options"""

    with tempfile.TemporaryDirectory() as td:
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        shutil.copy(EXCL, excl_fp)
        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)

        points = None
        points_range = None
        pc = BespokeWindPlants._parse_points(excl_fp, TM_DSET, 64, points,
                                             points_range, SAM)
        pp = pc.project_points

        assert len(pp) == 100
        for gid in pp.gids:
            assert pp[gid][0] == SAM

        points = None
        points_range = (0, 10)
        pc = BespokeWindPlants._parse_points(excl_fp, TM_DSET, 64, points,
                                             points_range, {'default': SAM})
        pp = pc.project_points
        assert len(pp) == 10
        for gid in pp.gids:
            assert pp[gid][0] == 'default'

        points = pd.DataFrame({'gid': [33, 34, 35], 'config': ['default'] * 3})
        points_range = None
        pc = BespokeWindPlants._parse_points(excl_fp, TM_DSET, 64, points,
                                             points_range, {'default': SAM})
        pp = pc.project_points
        assert len(pp) == 3
        for gid in pp.gids:
            assert pp[gid][0] == 'default'


if __name__ == '__main__':
    init_logger('reV', log_level='DEBUG')
    test_single()
