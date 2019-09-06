# -*- coding: utf-8 -*-
"""
PyTest file for SAM/reV econ Wind Balance of System cost model.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""
import json
import os
import pytest
import numpy as np

from reV.generation.generation import Gen
from reV.econ.econ import Econ
from reV.SAM.windbos import WindBos
from reV import TESTDATADIR


RTOL = 0.000001
ATOL = 0.001
PURGE_OUT = True
OUT_DIR = os.path.join(TESTDATADIR, 'ri_wind_reV2/')


DEFAULTS = {'tech_model': 'windbos',
            'financial_model': 'none',
            'machine_rating': 1500.0,
            'rotor_diameter': 77.0,
            'hub_height': 80.0,
            'number_of_turbines': 32,
            'interconnect_voltage': 137.0,
            'distance_to_interconnect': 5.0,
            'site_terrain': 0,
            'turbine_layout': 1,
            'soil_condition': 0,
            'construction_time': 6,
            'om_building_size': 3000.0,
            'quantity_test_met_towers': 1,
            'quantity_permanent_met_towers': 1,
            'weather_delay_days': 6,
            'crane_breakdowns': 2,
            'access_road_entrances': 2,
            'turbine_capital_cost': 0.0,
            'turbine_cost_per_kw': 1094.0,  # not in SDK tool, but important
            'tower_top_mass': 88.0,  # tonnes, tuned to default foundation cost
            'delivery_assist_required': 0,
            'pad_mount_transformer_required': 1,
            'new_switchyard_required': 1,
            'rock_trenching_required': 10,
            'mv_thermal_backfill': 0.0,
            'mv_overhead_collector': 0.0,
            'performance_bond': 0,
            'contingency': 3.0,
            'warranty_management': 0.02,
            'sales_and_use_tax': 5.0,
            'overhead': 5.0,
            'profit_margin': 5.0,
            'development_fee': 5.0,  # Millions of dollars
            'turbine_transportation': 0.0,
            }


# baseline single owner + windbos results using 2012 RI WTK data.
BASELINE = {'project_return_aftertax_npv': np.array([7876459.5, 7875551.5,
                                                     7874505., 7875270.,
                                                     7875349.5, 7872819.5,
                                                     7871078.5, 7871352.5,
                                                     7871153.5, 7869134.5]),
            'lcoe_real': np.array([71.00705, 69.21617, 67.24464, 68.67666,
                                   68.82837, 64.25754, 61.394363, 61.831406,
                                   61.513832, 58.435482]),
            'lcoe_nom': np.array([89.432816, 87.17721, 84.69409, 86.49771,
                                  86.6888, 80.93186, 77.325714, 77.87617,
                                  77.47619, 73.59903]),
            'flip_actual_irr': np.array([10.999977, 10.999978, 10.999978,
                                         10.999978, 10.999978, 10.999978,
                                         10.999979, 10.999979, 10.999979,
                                         10.99998])}


def test_sam_windbos():
    """Test SAM SSC from dict with windbos"""
    from PySAM.PySSC import ssc_sim_from_dict

    out = ssc_sim_from_dict(DEFAULTS)

    tcost = ((DEFAULTS['turbine_cost_per_kw']
              + DEFAULTS['turbine_capital_cost'])
             * DEFAULTS['machine_rating'] * DEFAULTS['number_of_turbines'])
    total_installed_cost = tcost + out['project_total_budgeted_cost']

    assert np.allclose(total_installed_cost, 88892240.00,
                       atol=ATOL, rtol=RTOL)


def test_rev_windbos():
    """Test baseline windbos calc with single owner defaults"""
    fpath = TESTDATADIR + '/SAM/i_windbos.json'
    with open(fpath, 'r') as f:
        inputs = json.load(f)
    wb = WindBos(inputs)
    assert np.allclose(wb.turbine_cost, 52512000.00, atol=ATOL, rtol=RTOL)
    assert np.allclose(wb.bos_cost, 36380236.00, atol=ATOL, rtol=RTOL)
    assert np.allclose(wb.total_installed_cost, 88892240.00, atol=ATOL,
                       rtol=RTOL)


def test_rev_windbos_perf_bond():
    """Test windbos calc with performance bonds"""
    fpath = TESTDATADIR + '/SAM/i_windbos.json'
    with open(fpath, 'r') as f:
        inputs = json.load(f)
    inputs['performance_bond'] = 10.0
    wb = WindBos(inputs)
    assert np.allclose(wb.turbine_cost, 52512000.00, atol=ATOL, rtol=RTOL)
    assert np.allclose(wb.bos_cost, 36686280.00, atol=ATOL, rtol=RTOL)
    assert np.allclose(wb.total_installed_cost, 89198280.00, atol=ATOL,
                       rtol=RTOL)


def test_rev_windbos_transport():
    """Test windbos calc with turbine transport costs"""
    fpath = TESTDATADIR + '/SAM/i_windbos.json'
    with open(fpath, 'r') as f:
        inputs = json.load(f)
    inputs['turbine_transportation'] = 100.0
    wb = WindBos(inputs)
    assert np.allclose(wb.turbine_cost, 52512000.00, atol=ATOL, rtol=RTOL)
    assert np.allclose(wb.bos_cost, 37720412.00, atol=ATOL, rtol=RTOL)
    assert np.allclose(wb.total_installed_cost, 90232416.00, atol=ATOL,
                       rtol=RTOL)


def test_rev_run(points=slice(0, 10), year=2012, n_workers=1):
    """Test full reV2 gen->econ pipeline with windbos inputs and benchmark
    against baseline results."""

    # get full file paths.
    sam_files = TESTDATADIR + '/SAM/i_windbos.json'
    res_file = TESTDATADIR + '/wtk/ri_100_wtk_{}.h5'.format(year)
    fgen = os.path.join(OUT_DIR, 'windbos_gen_{}.h5'.format(year))

    # run reV 2.0 generation
    Gen.run_direct('wind', points, sam_files, res_file,
                   output_request=('cf_mean', 'cf_profile'),
                   n_workers=n_workers, sites_per_split=3, fout=fgen,
                   return_obj=False)

    econ_outs = ('lcoe_nom', 'lcoe_real', 'flip_actual_irr',
                 'project_return_aftertax_npv')
    e = Econ.run_direct(points=points, sam_files=sam_files, cf_file=fgen,
                        cf_year=year, site_data=None, output_request=econ_outs,
                        n_workers=1, sites_per_split=3, fout=None,
                        return_obj=True)

    for k in econ_outs:
        msg = 'Failed for {}'.format(k)
        assert np.allclose(e.out[k], BASELINE[k], atol=ATOL, rtol=RTOL), msg

    if PURGE_OUT:
        for fn in os.listdir(OUT_DIR):
            os.remove(os.path.join(OUT_DIR, fn))
    return e


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
    execute_pytest()
