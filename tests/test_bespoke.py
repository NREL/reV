# -*- coding: utf-8 -*-
"""reV bespoke wind plant optimization tests
"""
import copy
from glob import glob
import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
import h5py

from reV import TESTDATADIR
from reV.bespoke.bespoke import BespokeSinglePlant, BespokeWindPlants
from reV.handlers.collection import Collector
from reV.handlers.outputs import Outputs
from reV.supply_curve.tech_mapping import TechMapping
from reV.supply_curve.supply_curve import SupplyCurve
from reV.SAM.generation import WindPower
from reV.losses.power_curve import PowerCurveLossesMixin
from reV.losses.scheduled import ScheduledLossesMixin

from rex import Resource

pytest.importorskip("shapely")

SAM = os.path.join(TESTDATADIR, 'SAM/i_windpower.json')
EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
RES = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_{}.h5')
TM_DSET = 'techmap_wtk_ri_100'
AGG_DSET = ('cf_mean', 'cf_profile')

DATA_LAYERS = {'pct_slope': {'dset': 'ri_srtm_slope',
                             'method': 'mean',
                             'fpath': EXCL},
               'reeds_region': {'dset': 'ri_reeds_regions',
                                'method': 'mode',
                                'fpath': EXCL},
               'padus': {'dset': 'ri_padus',
                         'method': 'mode',
                         'fpath': EXCL}}

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
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')

    cap_cost_fun = ('140 * system_capacity '
                    '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    foc_fun = ('60 * system_capacity '
               '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    voc_fun = '3'
    objective_function = (
        '(0.0975 * capital_cost + fixed_operating_cost) '
        '/ aep + variable_operating_cost')

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function,
                                 cap_cost_fun,
                                 foc_fun,
                                 voc_fun,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        place_optimizer = bsp.plant_optimizer
        place_optimizer.place_turbines(max_time=5)

        assert place_optimizer.nturbs == len(place_optimizer.turbine_x)
        assert place_optimizer.capacity == place_optimizer.nturbs *\
            place_optimizer.turbine_capacity
        assert place_optimizer.area == place_optimizer.full_polygons.area
        assert place_optimizer.capacity_density == place_optimizer.capacity\
            / place_optimizer.area * 1E3

        place_optimizer.wind_plant["wind_farm_xCoordinates"] = \
            place_optimizer.turbine_x
        place_optimizer.wind_plant["wind_farm_yCoordinates"] = \
            place_optimizer.turbine_y
        place_optimizer.wind_plant["system_capacity"] =\
            place_optimizer.capacity
        place_optimizer.wind_plant.assign_inputs()
        place_optimizer.wind_plant.execute()

        assert place_optimizer.aep == \
            place_optimizer.wind_plant.annual_energy()

        # pylint: disable=W0641
        system_capacity = place_optimizer.capacity
        # pylint: disable=W0641
        aep = place_optimizer.aep
        # pylint: disable=W0123
        capital_cost = eval(cap_cost_fun, globals(), locals())
        fixed_operating_cost = eval(foc_fun, globals(), locals())
        variable_operating_cost = eval(voc_fun, globals(), locals())
        # pylint: disable=W0123
        assert place_optimizer.objective ==\
            eval(objective_function, globals(), locals())
        assert place_optimizer.capital_cost == capital_cost
        assert place_optimizer.fixed_operating_cost == fixed_operating_cost
        assert (place_optimizer.variable_operating_cost
                == variable_operating_cost)

        bsp.close()


def test_zero_area(gid=33):
    """Test turbine placement with zero available area. """
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')

    cap_cost_fun = ('140 * system_capacity '
                    '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    foc_fun = ('60 * system_capacity '
               '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    voc_fun = '3'
    objective_function = (
        '(0.0975 * capital_cost + fixed_operating_cost) '
        '/ (aep + 1E-6) + variable_operating_cost')

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function, cap_cost_fun,
                                 foc_fun, voc_fun,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        optimizer = bsp.plant_optimizer
        optimizer.include_mask = np.zeros_like(optimizer.include_mask)
        optimizer.place_turbines(max_time=5)

        # pylint: disable=W0123
        assert len(optimizer.turbine_x) == 0
        assert len(optimizer.turbine_y) == 0
        assert optimizer.nturbs == 0
        assert optimizer.capacity == 0
        assert optimizer.area == 0
        assert optimizer.capacity_density == 0
        assert optimizer.objective == eval(voc_fun)
        assert optimizer.capital_cost == 0
        assert optimizer.fixed_operating_cost == 0

        bsp.close()


def test_packing_algorithm(gid=33):
    """Test turbine placement with zero available area. """
    output_request = ()
    cap_cost_fun = ""
    foc_fun = ""
    voc_fun = ""
    objective_function = ""
    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function, cap_cost_fun,
                                 foc_fun, voc_fun,
                                 ga_kwargs={'max_time': 5},
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        optimizer = bsp.plant_optimizer
        optimizer.define_exclusions()
        optimizer.initialize_packing()

        assert len(optimizer.x_locations) < 165
        assert len(optimizer.x_locations) > 145
        assert np.sum(optimizer.include_mask) ==\
            optimizer.safe_polygons.area / (optimizer.pixel_side_length**2)

        bsp.close()


def test_bespoke_points():
    """Test the bespoke points input options"""
    # pylint: disable=W0612
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

    cap_cost_fun = ('140 * system_capacity '
                    '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    foc_fun = ('60 * system_capacity '
               '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    voc_fun = '3'
    objective_function = (
        '(0.0975 * capital_cost + fixed_operating_cost) '
        '/ aep + variable_operating_cost')

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function, cap_cost_fun,
                                 foc_fun, voc_fun,
                                 ga_kwargs={'max_time': 5},
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

        bsp.close()


def test_extra_outputs(gid=33):
    """Test running bespoke single farm optimization with lcoe requests"""
    output_request = ('system_capacity', 'cf_mean', 'cf_profile', 'lcoe_fcr')

    cap_cost_fun = ('140 * system_capacity '
                    '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    foc_fun = ('60 * system_capacity '
               '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    voc_fun = '3'
    objective_function = (
        '(fixed_charge_rate * capital_cost + fixed_operating_cost) '
        '/ aep + variable_operating_cost')

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)

        with pytest.raises(KeyError):
            bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                     SAM_SYS_INPUTS,
                                     objective_function, cap_cost_fun,
                                     foc_fun, voc_fun,
                                     ga_kwargs={'max_time': 5},
                                     excl_dict=EXCL_DICT,
                                     output_request=output_request,
                                     )

        sam_sys_inputs = copy.deepcopy(SAM_SYS_INPUTS)
        sam_sys_inputs['fixed_charge_rate'] = 0.0975

        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 sam_sys_inputs,
                                 objective_function, cap_cost_fun,
                                 foc_fun, voc_fun,
                                 ga_kwargs={'max_time': 5},
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 data_layers=DATA_LAYERS,
                                 )

        out = bsp.run_plant_optimization()
        out = bsp.run_wind_plant_ts()
        bsp.agg_data_layers()

        assert 'lcoe_fcr-2012' in out
        assert 'lcoe_fcr-2013' in out
        assert 'lcoe_fcr-means' in out

        assert 'capacity' in bsp.meta
        assert 'mean_cf' in bsp.meta
        assert 'mean_lcoe' in bsp.meta

        assert 'pct_slope' in bsp.meta
        assert 'reeds_region' in bsp.meta
        assert 'padus' in bsp.meta

        bsp.close()


def test_bespoke():
    """Test bespoke optimization with multiple plants, parallel processing, and
    file output. """
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')

    cap_cost_fun = ('140 * system_capacity '
                    '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    foc_fun = ('60 * system_capacity '
               '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    voc_fun = '3'
    objective_function = (
        '(0.0975 * capital_cost + fixed_operating_cost) '
        '/ aep + variable_operating_cost')

    with tempfile.TemporaryDirectory() as td:
        out_fpath = os.path.join(td, 'bespoke_out.h5')
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')
        # both 33 and 35 are included, 37 is fully excluded
        points = [33, 35]

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        _ = BespokeWindPlants.run(excl_fp, res_fp, TM_DSET,
                                  objective_function, cap_cost_fun,
                                  foc_fun, voc_fun,
                                  points, SAM_CONFIGS,
                                  ga_kwargs={'max_time': 5},
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
                assert f[dset].any()  # not all zeros

            dsets_2d = ('cf_profile-2012', 'cf_profile-2013')
            for dset in dsets_2d:
                assert dset in list(f)
                assert isinstance(f[dset], np.ndarray)
                assert len(f[dset].shape) == 2
                assert len(f[dset]) == 8760
                assert f[dset].shape[1] == len(meta)
                assert f[dset].any()  # not all zeros


def test_collect_bespoke():
    """Test the collection of multiple chunked bespoke files. """
    with tempfile.TemporaryDirectory() as td:
        source_dir = os.path.join(TESTDATADIR, 'bespoke/')
        source_pattern = source_dir + '/test_bespoke*.h5'
        source_fps = sorted(glob(source_pattern))
        assert len(source_fps) > 1

        h5_file = os.path.join(td, 'collection.h5')

        Collector.collect(h5_file, source_pattern, None, 'cf_profile-2012',
                          dset_out=None)

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


def test_consistent_eval_namespace(gid=33):
    """Test that all the same variables are available for every eval."""
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')
    cap_cost_fun = "2000"
    foc_fun = "0"
    voc_fun = "0"
    objective_function = ("n_turbines + id(self.wind_plant) "
                          "+ system_capacity + capital_cost + aep")
    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function, cap_cost_fun,
                                 foc_fun, voc_fun,
                                 ga_kwargs={'max_time': 5},
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )
        out = bsp.run_plant_optimization()

        assert out["bespoke_aep"] == bsp.plant_optimizer.aep
        assert out["bespoke_objective"] == bsp.plant_optimizer.objective

        bsp.close()


def test_bespoke_supply_curve():
    """Test supply curve compute from a bespoke output that acts as the
    traditional reV-sc-aggregation output table."""

    bespoke_sample_fout = os.path.join(TESTDATADIR,
                                       'bespoke/test_bespoke_node00.h5')

    normal_path = os.path.join(TESTDATADIR, 'sc_out/baseline_agg_summary.csv')
    normal_sc_points = pd.read_csv(normal_path)

    with tempfile.TemporaryDirectory() as td:
        bespoke_sc_fp = os.path.join(td, 'bespoke_out.h5')
        shutil.copy(bespoke_sample_fout, bespoke_sc_fp)
        with h5py.File(bespoke_sc_fp, 'a') as f:
            del f['meta']
        with Outputs(bespoke_sc_fp, mode='a') as f:
            bespoke_meta = normal_sc_points.copy()
            bespoke_meta = bespoke_meta.drop('sc_gid', axis=1)
            f.meta = bespoke_meta

        # this is basically copied from test_supply_curve_compute.py
        trans_tables = [os.path.join(TESTDATADIR, 'trans_tables',
                                     f'costs_RI_{cap}MW.csv')
                        for cap in [100, 200, 400, 1000]]

        sc_full = SupplyCurve.full(bespoke_sc_fp, trans_tables, fcr=0.1,
                                   avail_cap_frac=0.1)

        assert all(gid in sc_full['sc_gid']
                   for gid in normal_sc_points['sc_gid'])
        for _, inp_row in normal_sc_points.iterrows():
            sc_gid = inp_row['sc_gid']
            assert sc_gid in sc_full['sc_gid']
            test_ind = np.where(sc_full['sc_gid'] == sc_gid)[0]
            assert len(test_ind) == 1
            test_row = sc_full.iloc[test_ind]
            assert test_row['total_lcoe'].values[0] > inp_row['mean_lcoe']

    fpath_baseline = os.path.join(TESTDATADIR, 'sc_out/sc_full_lc.csv')
    sc_baseline = pd.read_csv(fpath_baseline)
    assert np.allclose(sc_baseline['total_lcoe'], sc_full['total_lcoe'])


@pytest.mark.parametrize('wlm', [2, 100])
def test_wake_loss_multiplier(wlm):
    """Test wake loss multiplier. """
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')

    cap_cost_fun = ('140 * system_capacity '
                    '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    foc_fun = ('60 * system_capacity '
               '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    voc_fun = '3'
    objective_function = (
        '(0.0975 * capital_cost + fixed_operating_cost) '
        '/ aep + variable_operating_cost')

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant(33, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function,
                                 cap_cost_fun,
                                 foc_fun,
                                 voc_fun,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        optimizer = bsp.plant_optimizer
        optimizer.define_exclusions()
        optimizer.initialize_packing()

        optimizer.wind_plant["wind_farm_xCoordinates"] = optimizer.x_locations
        optimizer.wind_plant["wind_farm_yCoordinates"] = optimizer.y_locations

        system_capacity = (len(optimizer.x_locations)
                           * optimizer.turbine_capacity)
        optimizer.wind_plant["system_capacity"] = system_capacity

        optimizer.wind_plant.assign_inputs()
        optimizer.wind_plant.execute()
        aep = optimizer._aep_after_scaled_wake_losses()
        bsp.close()

        bsp = BespokeSinglePlant(33, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function,
                                 cap_cost_fun,
                                 foc_fun,
                                 voc_fun,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 wake_loss_multiplier=wlm)

        optimizer2 = bsp.plant_optimizer
        optimizer2.wind_plant["wind_farm_xCoordinates"] = optimizer.x_locations
        optimizer2.wind_plant["wind_farm_yCoordinates"] = optimizer.y_locations

        system_capacity = (len(optimizer.x_locations)
                           * optimizer.turbine_capacity)
        optimizer2.wind_plant["system_capacity"] = system_capacity

        optimizer2.wind_plant.assign_inputs()
        optimizer2.wind_plant.execute()
        aep_wlm = optimizer2._aep_after_scaled_wake_losses()
        bsp.close()

    assert aep > aep_wlm
    assert aep_wlm >= 0


def test_bespoke_wind_plant_with_power_curve_losses():
    """Test bespoke ``wind_plant`` with power curve losses. """
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')

    cap_cost_fun = ('140 * system_capacity '
                    '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    foc_fun = ('60 * system_capacity '
               '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    voc_fun = '3'
    objective_function = (
        '(0.0975 * capital_cost + fixed_operating_cost) '
        '/ aep + variable_operating_cost')

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant(33, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function,
                                 cap_cost_fun,
                                 foc_fun,
                                 voc_fun,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        optimizer = bsp.plant_optimizer
        optimizer.wind_plant["wind_farm_xCoordinates"] = [1000, -1000]
        optimizer.wind_plant["wind_farm_yCoordinates"] = [1000, -1000]
        cap = 2 * optimizer.turbine_capacity
        optimizer.wind_plant["system_capacity"] = cap

        optimizer.wind_plant.assign_inputs()
        optimizer.wind_plant.execute()
        aep = optimizer._aep_after_scaled_wake_losses()
        bsp.close()

        sam_inputs = copy.deepcopy(SAM_SYS_INPUTS)
        sam_inputs[PowerCurveLossesMixin.POWER_CURVE_CONFIG_KEY] = {
            'target_losses_percent': 10,
            'transformation': 'exponential_stretching'
        }
        bsp = BespokeSinglePlant(33, excl_fp, res_fp, TM_DSET,
                                 sam_inputs,
                                 objective_function,
                                 cap_cost_fun,
                                 foc_fun,
                                 voc_fun,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request)

        optimizer2 = bsp.plant_optimizer
        optimizer2.wind_plant["wind_farm_xCoordinates"] = [1000, -1000]
        optimizer2.wind_plant["wind_farm_yCoordinates"] = [1000, -1000]
        cap = 2 * optimizer2.turbine_capacity
        optimizer2.wind_plant["system_capacity"] = cap

        optimizer2.wind_plant.assign_inputs()
        optimizer2.wind_plant.execute()
        aep_losses = optimizer2._aep_after_scaled_wake_losses()
        bsp.close()

    assert aep > aep_losses, f"{aep}, {aep_losses}"

    err_msg = "{:0.3f} != 0.9".format(aep_losses / aep)
    assert np.isclose(aep_losses / aep, 0.9), err_msg


def test_bespoke_run_with_power_curve_losses():
    """Test bespoke run with power curve losses. """
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')

    cap_cost_fun = ('140 * system_capacity '
                    '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    foc_fun = ('60 * system_capacity '
               '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    voc_fun = '3'
    objective_function = (
        '(0.0975 * capital_cost + fixed_operating_cost) '
        '/ aep + variable_operating_cost')

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant(33, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function, cap_cost_fun,
                                 foc_fun, voc_fun,
                                 ga_kwargs={'max_time': 5},
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request)

        out = bsp.run_plant_optimization()
        out = bsp.run_wind_plant_ts()
        bsp.close()

        sam_inputs = copy.deepcopy(SAM_SYS_INPUTS)
        sam_inputs[PowerCurveLossesMixin.POWER_CURVE_CONFIG_KEY] = {
            'target_losses_percent': 10,
            'transformation': 'exponential_stretching'
        }
        bsp = BespokeSinglePlant(33, excl_fp, res_fp, TM_DSET,
                                 sam_inputs,
                                 objective_function,
                                 cap_cost_fun,
                                 foc_fun,
                                 voc_fun,
                                 ga_kwargs={'max_time': 5},
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request)

        out_losses = bsp.run_plant_optimization()
        out_losses = bsp.run_wind_plant_ts()
        bsp.close()

    ae_dsets = ['annual_energy-2012',
                'annual_energy-2013',
                'annual_energy-means']
    for dset in ae_dsets:
        assert not np.isclose(out[dset], out_losses[dset])
        assert out[dset] > out_losses[dset]


def test_bespoke_run_with_scheduled_losses():
    """Test bespoke run with scheduled losses. """
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')

    cap_cost_fun = ('140 * system_capacity '
                    '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    foc_fun = ('60 * system_capacity '
               '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    voc_fun = '3'
    objective_function = (
        '(0.0975 * capital_cost + fixed_operating_cost) '
        '/ aep + variable_operating_cost')

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant(33, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function, cap_cost_fun,
                                 foc_fun, voc_fun,
                                 ga_kwargs={'max_time': 5},
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request)

        out = bsp.run_plant_optimization()
        out = bsp.run_wind_plant_ts()
        bsp.close()

        sam_inputs = copy.deepcopy(SAM_SYS_INPUTS)
        sam_inputs[ScheduledLossesMixin.OUTAGE_CONFIG_KEY] = [{
            'name': 'Environmental',
            'count': 115,
            'duration': 2,
            'percentage_of_capacity_lost': 100,
            'allowed_months': ['April', 'May', 'June', 'July', 'August',
                               'September', 'October']}]
        sam_inputs['hourly'] = [0] * 8760  # only needed for testing
        output_request = ('system_capacity', 'cf_mean', 'cf_profile', 'hourly')

        bsp = BespokeSinglePlant(33, excl_fp, res_fp, TM_DSET,
                                 sam_inputs,
                                 objective_function,
                                 cap_cost_fun,
                                 foc_fun,
                                 voc_fun,
                                 ga_kwargs={'max_time': 5},
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request)

        out_losses = bsp.run_plant_optimization()
        out_losses = bsp.run_wind_plant_ts()
        bsp.close()

    ae_dsets = ['annual_energy-2012',
                'annual_energy-2013',
                'annual_energy-means']
    for dset in ae_dsets:
        assert not np.isclose(out[dset], out_losses[dset])
        assert out[dset] > out_losses[dset]

    assert not np.allclose(out_losses['hourly-2012'],
                           out_losses['hourly-2013'])


def test_bespoke_wind_plant_with_power_curve_losses():
    """Test bespoke ``wind_plant`` with power curve losses. """
    output_request = ('system_capacity', 'cf_mean', 'cf_profile')

    cap_cost_fun = ('140 * system_capacity '
                    '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    foc_fun = ('60 * system_capacity '
               '* np.exp(-system_capacity / 1E5 * 0.1 + (1 - 0.1))')
    voc_fun = '3'
    objective_function = 'aep'

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant(33, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function,
                                 cap_cost_fun,
                                 foc_fun,
                                 voc_fun,
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )

        optimizer = bsp.plant_optimizer
        optimizer.define_exclusions()
        optimizer.initialize_packing()
        optimizer.wind_plant["wind_farm_xCoordinates"] = []
        optimizer.wind_plant["wind_farm_yCoordinates"] = []
        optimizer.wind_plant["system_capacity"] = 0

        aep = optimizer.optimization_objective(x=[])
        bsp.close()

    assert aep == 0
