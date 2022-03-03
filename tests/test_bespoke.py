# -*- coding: utf-8 -*-
"""reV bespoke wind plant optimization tests
"""
from glob import glob
import json
import os
import shutil
import tempfile
from tkinter import Place
import numpy as np
import pandas as pd
import pytest

from reV import TESTDATADIR
from reV.bespoke.bespoke import BespokeSinglePlant, BespokeWindPlants
from reV.handlers.collection import Collector
from reV.SAM.generation import WindPower
from reV.supply_curve.tech_mapping import TechMapping

from rex import Resource

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


def test_turbine_placement(gid=33):
    """Test turbine placement with zero available area. """
    np.random.seed(0)
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')
    cost_function = """200 * system_capacity * np.exp(-system_capacity /
        1E5 * 0.1 + (1 - 0.1))"""
    objective_function = "cost / aep"
    with tempfile.TemporaryDirectory() as td:
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function, cost_function,
                                 ga_time=5,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        place_optimizer = bsp.plant_optimizer
        place_optimizer.place_turbines()

        assert place_optimizer.nturbs == 95
        assert place_optimizer.capacity == 142500.0
        assert place_optimizer.area == 13421700.0
        assert place_optimizer.capacity_density == 10.617134938197099
        assert place_optimizer.objective == 0.15975631472465107
        assert place_optimizer.annual_cost == 60788710.38507378


def test_zero_area(gid=33):
    """Test turbine placement with zero available area. """
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')
    cost_function = """200 * system_capacity * np.exp(-system_capacity /
        1E5 * 0.1 + (1 - 0.1))"""
    objective_function = "cost / aep"
    with tempfile.TemporaryDirectory() as td:
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function, cost_function,
                                 ga_time=5,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        optimizer = bsp.plant_optimizer
        optimizer.include_mask = np.zeros_like(optimizer.include_mask)
        optimizer.place_turbines()

        assert len(optimizer.turbine_x) == 0
        assert len(optimizer.turbine_y) == 0
        assert optimizer.nturbs == 0
        assert optimizer.capacity == 0
        assert optimizer.area == 0
        assert optimizer.capacity_density == 0
        assert optimizer.objective == 0
        assert optimizer.annual_cost == 0


def test_packing_algorithm(gid=33):
    """Test turbine placement with zero available area. """
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')
    cost_function = """200 * system_capacity * np.exp(-system_capacity /
        1E5 * 0.1 + (1 - 0.1))"""
    objective_function = "cost / aep"
    with tempfile.TemporaryDirectory() as td:
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function, cost_function,
                                 ga_time=5,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        optimizer = bsp.plant_optimizer
        optimizer.define_exclusions()
        optimizer.initialize_packing()

        test_x = optimizer.x_locations
        test_y = optimizer.y_locations

        truth_x = np.load(os.path.join(TESTDATADIR,
                                       'bespoke/packing_data_x.npy'))
        truth_y = np.load(os.path.join(TESTDATADIR,
                                       'bespoke/packing_data_y.npy'))

        assert np.allclose(test_x, truth_x)
        assert np.allclose(test_y, truth_y)


def test_bespoke_points():
    """Test the bespoke points input options"""
    with tempfile.TemporaryDirectory() as td:
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        shutil.copy(EXCL, excl_fp)
        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)

        points = None
        points_range = None
        pc = BespokeWindPlants._parse_points(excl_fp, RES.format(2012),
                                             TM_DSET, 64, points,
                                             points_range, SAM)
        pp = pc.project_points

        assert len(pp) == 100
        for gid in pp.gids:
            assert pp[gid][0] == SAM

        points = None
        points_range = (0, 10)
        pc = BespokeWindPlants._parse_points(excl_fp, RES.format(2012),
                                             TM_DSET, 64, points,
                                             points_range, {'default': SAM})
        pp = pc.project_points
        assert len(pp) == 10
        for gid in pp.gids:
            assert pp[gid][0] == 'default'

        points = pd.DataFrame({'gid': [33, 34, 35], 'config': ['default'] * 3})
        points_range = None
        pc = BespokeWindPlants._parse_points(excl_fp, RES.format(2012),
                                             TM_DSET, 64, points,
                                             points_range, {'default': SAM})
        pp = pc.project_points
        assert len(pp) == 3
        for gid in pp.gids:
            assert pp[gid][0] == 'default'


def test_single(gid=33):
    """Test a single wind plant bespoke optimization run"""
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')
    cost_function = """200 * system_capacity * np.exp(-system_capacity /
        1E5 * 0.1 + (1 - 0.1))"""
    objective_function = "cost / aep"
    with tempfile.TemporaryDirectory() as td:
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function, cost_function,
                                 ga_time=5,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )
        out = bsp.run_plant_optimization()
        out = bsp.run_wind_plant_ts()

        assert 'cf_profile-2012' in out
        assert 'cf_profile-2013' in out
        assert 'cf_mean-2012' in out
        assert 'cf_mean-2013' in out
        assert 'cf_mean-means' in out
        assert 'annual_energy-2012' in out
        assert 'annual_energy-2013' in out
        assert 'annual_energy-means' in out

        assert TURB_RATING * out['n_turbines'] == out['system_capacity']
        x_coords = json.loads(bsp.meta['turbine_x_coords'].values[0])
        y_coords = json.loads(bsp.meta['turbine_y_coords'].values[0])
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
    """Test bespoke optimization with multiple plants, parallel processing, and
    file output. """
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')
    cost_function = """200 * system_capacity * np.exp(-system_capacity /
        1E5 * 0.1 + (1 - 0.1))"""
    objective_function = "cost / aep"
    with tempfile.TemporaryDirectory() as td:
        out_fpath = os.path.join(td, 'bespoke_out.h5')
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')
        points = [33, 35]  # both 33 and 35 are included
#        points = [36, 37]  # 37 is fully excluded

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        _ = BespokeWindPlants.run(excl_fp, res_fp, TM_DSET,
                                  objective_function, cost_function,
                                  points, SAM_CONFIGS,
                                  ga_time=5,
                                  excl_dict=EXCL_DICT,
                                  output_request=output_request,
                                  max_workers=2,
                                  out_fpath=out_fpath)

        with Resource(out_fpath) as f:
            meta = f.meta
            assert len(meta) <= len(points)
            assert 'sc_point_gid' in meta
            assert 'turbine_x_coords' in meta
            assert 'turbine_y_coords' in meta
            assert 'possible_x_coords' in meta
            assert 'possible_y_coords' in meta

            dsets_1d = ('n_turbines', 'system_capacity', 'cf_mean-2012',
                        'annual_energy-2012', 'cf_mean-means')
            for dset in dsets_1d:
                assert dset in list(f)
                assert isinstance(f[dset], np.ndarray)
                assert len(f[dset].shape) == 1
                assert len(f[dset]) == len(meta)

            dsets_2d = ('cf_profile-2012', 'cf_profile-2013')
            for dset in dsets_2d:
                assert dset in list(f)
                assert isinstance(f[dset], np.ndarray)
                assert len(f[dset].shape) == 2
                assert len(f[dset]) == 8760
                assert f[dset].shape[1] == len(meta)

#        shutil.copy(out_fpath, './data/bespoke/test_bespoke_node00.h5')


def test_collect_bespoke():
    """Test the collection of multiple chunked bespoke files. """
    with tempfile.TemporaryDirectory() as td:
        source_dir = os.path.join(TESTDATADIR, 'bespoke/')
        source_fps = sorted(glob(source_dir + '/test_bespoke*.h5'))
        assert len(source_fps) > 1

        h5_file = os.path.join(td, 'collection.h5')

        Collector.collect(h5_file, source_dir, None, 'cf_profile-2012',
                          dset_out=None, file_prefix='test_bespoke')

        with Resource(h5_file) as fout:
            meta = fout.meta
            assert all(meta['gid'].values == sorted(meta['gid'].values))
            ti = fout.time_index
            assert len(ti) == 8760
            assert 'time_index-2012' in fout
            assert 'time_index-2013' in fout
            data = fout['cf_profile-2012']

        for fp in source_fps:
            with Resource(fp) as source:
                assert all(np.isin(source.meta['gid'].values,
                                   meta['gid'].values))
                for isource, gid in enumerate(source.meta['gid'].values):
                    iout = np.where(meta['gid'].values == gid)[0]
                    truth = source['cf_profile-2012', :, isource].flatten()
                    test = data[:, iout].flatten()
                    assert np.allclose(truth, test)
