# -*- coding: utf-8 -*-
"""
PyTest file for reV power curve losses.

Created on Mon Apr 18 12:52:16 2021

@author: ppinchuk
"""

import os
import pytest
import tempfile
import json
import copy

import numpy as np

from reV import TESTDATADIR
from reV.generation.generation import Gen
from reV.utilities.exceptions import reVLossesValueError
from reV.losses.power_curve import (PowerCurve, PowerCurveLosses,
                                    PowerCurveLossesMixin,
                                    HorizontalPowerCurveTranslation)
from reV.losses.scheduled import ScheduledLossesMixin


REV_POINTS = slice(0, 5)
RES_FILE = TESTDATADIR + '/wtk/ri_100_wtk_2012.h5'
SAM_FILES = [
    TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json',
    TESTDATADIR + '/SAM/wind_gen_non_standard_0.json',
    TESTDATADIR + '/SAM/wind_gen_non_standard_1.json'
]
BASIC_WIND_RES = [10, 20, 20]


@pytest.fixture
def simple_power_curve():
    """Return a simple synthetic power curve."""
    wind_speed = [0, 10, 20, 30]
    generation = [0, 20, 15, 10]
    return PowerCurve(wind_speed, generation)


@pytest.fixture
def real_power_curve():
    """Return a basic power curve."""
    with open(SAM_FILES[0], 'r') as fh:
        sam_config = json.load(fh)

    wind_speed = sam_config['wind_turbine_powercurve_windspeeds']
    generation = sam_config['wind_turbine_powercurve_powerout']
    return PowerCurve(wind_speed, generation)


@pytest.mark.parametrize('generic_losses', [0, 0.2])
@pytest.mark.parametrize('target_losses', [0, 10, 50])
def test_power_curve_losses(generic_losses, target_losses):
    """Test full gen run with scheduled losses. """
    gen_profiles, gen_profiles_with_losses = _run_gen_with_and_without_losses(
        generic_losses, target_losses=target_losses
    )

    assert (gen_profiles - gen_profiles_with_losses > 0).any()
    assert np.isclose(gen_profiles, gen_profiles_with_losses).any()
    assert gen_profiles.max() == gen_profiles_with_losses.max()

    annual_gen_ratio = (gen_profiles_with_losses.sum() / gen_profiles.sum())
    assert ((1 - annual_gen_ratio) * 100 - target_losses) < 1


def _run_gen_with_and_without_losses(
    generic_losses, target_losses, include_outages=False
):
    """Run generaion with and without losses for testing. """

    sam_file = SAM_FILES[0]

    with open(sam_file, 'r', encoding='utf-8') as fh:
        sam_config = json.load(fh)

    with tempfile.TemporaryDirectory() as td:
        del sam_config['wind_farm_losses_percent']
        sam_config['turb_generic_loss'] = generic_losses

        sam_config[PowerCurveLossesMixin.POWERCURVE_CONFIG_KEY] = {
            'target_losses_percent': target_losses
        }
        if include_outages:
            sam_config[ScheduledLossesMixin.OUTAGE_CONFIG_KEY] = [
                {
                    'count': 5,
                    'duration': 24,
                    'percentage_of_farm_down': 100,
                    'allowed_months': ['January'],
                    'allow_outage_overlap': True
                }
            ]
        sam_fp = os.path.join(td, 'gen.json')
        with open(sam_fp, 'w+') as fh:
            fh.write(json.dumps(sam_config))

        gen = Gen.reV_run('windpower', REV_POINTS, sam_fp, RES_FILE,
                          output_request=('gen_profile'),
                          max_workers=1, sites_per_worker=3, out_fpath=None)
    gen_profiles_with_losses = gen.out['gen_profile']

    # undo UTC array rolling
    for ind, row in gen.meta.iterrows():
        time_shift = row['timezone']
        gen_profiles_with_losses[:, ind] = np.roll(
            gen_profiles_with_losses[:, ind], time_shift
        )

    pc = Gen.get_pc(REV_POINTS, None, sam_file, 'windpower',
                    sites_per_worker=3, res_file=RES_FILE)

    del pc.project_points.sam_inputs[sam_file]['wind_farm_losses_percent']
    pc.project_points.sam_inputs[sam_file]['turb_generic_loss'] = (
        generic_losses
    )

    gen = Gen.reV_run('windpower', pc, sam_file, RES_FILE,
                      output_request=('gen_profile'),
                      max_workers=1, sites_per_worker=3, out_fpath=None)
    gen_profiles = gen.out['gen_profile']

    for ind, row in gen.meta.iterrows():
        time_shift = row['timezone']
        gen_profiles[:, ind] = np.roll(gen_profiles[:, ind], time_shift)

    return gen_profiles, gen_profiles_with_losses


def test_power_curve_losses_witch_scheduled_outages():
    """Test full gen run with scheduled losses. """
    gen_profiles, gen_profiles_with_losses = _run_gen_with_and_without_losses(
        generic_losses=0.2, target_losses=20, include_outages=True
    )

    annual_gen_ratio = (gen_profiles_with_losses.sum() / gen_profiles.sum())
    assert (1 - annual_gen_ratio) * 100 > 21  # 1% tolerance


@pytest.mark.parametrize('config', SAM_FILES)
def test_power_curve_losses_mixin_class_add_power_curve_losses(config):
    """Test mixin class behavior when adding losses. """

    with open(config, 'r') as fh:
        sam_config = json.load(fh)

    og_power_curve = np.array(sam_config["wind_turbine_powercurve_powerout"])

    # patch required for 'wind_resource_data' access below
    def get_item_patch(self, key):
        return self.sam_sys_inputs.get(key)
    PowerCurveLossesMixin.__getitem__ = get_item_patch

    mixin = PowerCurveLossesMixin()
    mixin.sam_sys_inputs = copy.deepcopy(sam_config)
    mixin.sam_sys_inputs[PowerCurveLossesMixin.POWERCURVE_CONFIG_KEY] = {
        'target_losses_percent': 10
    }
    mixin.sam_sys_inputs['wind_resource_data'] = {
        'data': [(0, 0, val) for val in BASIC_WIND_RES]
    }
    mixin.add_power_curve_losses()
    new_power_curve = np.array(
        mixin.sam_sys_inputs["wind_turbine_powercurve_powerout"]
    )

    assert mixin.POWERCURVE_CONFIG_KEY not in mixin.sam_sys_inputs
    assert any(og_power_curve != new_power_curve)


@pytest.mark.parametrize('config', SAM_FILES)
def test_power_curve_losses_mixin_class_no_losses_input(config):
    """Test mixin class behavior when no losses should be added. """

    with open(config, 'r') as fh:
        sam_config = json.load(fh)

    og_power_curve = np.array(sam_config["wind_turbine_powercurve_powerout"])

    mixin = PowerCurveLossesMixin()
    mixin.sam_sys_inputs = copy.deepcopy(sam_config)
    mixin.add_power_curve_losses()
    new_power_curve = np.array(
        mixin.sam_sys_inputs["wind_turbine_powercurve_powerout"]
    )

    assert mixin.POWERCURVE_CONFIG_KEY not in mixin.sam_sys_inputs
    assert (og_power_curve == new_power_curve).all()


@pytest.mark.parametrize('bad_wind_speed', ([], [-10, 10]))
def test_power_curve_class_bad_wind_speed_input(bad_wind_speed):
    """Test that error is raised for bad wind speed inputs. """
    power_curve = [10, 100]

    with pytest.raises(reVLossesValueError) as excinfo:
        PowerCurve(bad_wind_speed, power_curve)
    assert "Invalid wind speed input" in str(excinfo.value)


@pytest.mark.parametrize('bad_generation', ([], [0, 0, 0, 0], [0, 20, 0, 10]))
def test_power_curve_class_bad_generation_input(bad_generation):
    """Test that error is raised for bad generation inputs. """
    wind_speed = [0, 10, 20, 30]

    with pytest.raises(reVLossesValueError) as excinfo:
        PowerCurve(wind_speed, bad_generation)
    assert "Invalid generation input" in str(excinfo.value)


@pytest.mark.parametrize('bad_wind_res', ([], [-10, 10]))
def test_power_curve_losses_class_bad_wind_res_input(bad_wind_res):
    """Test that error is raised for bad wind resource inputs. """
    wind_speed = [0, 10]
    generation = [10, 100]
    power_curve = PowerCurve(wind_speed, generation)
    with pytest.raises(reVLossesValueError) as excinfo:
        PowerCurveLosses(power_curve, bad_wind_res)
    assert "Invalid wind resource input" in str(excinfo.value)


def test_horizontal_transformation_class_apply(real_power_curve):
    """Test that the power curve shift is applied correctly. """

    real_power_curve.generation[-1] = real_power_curve.generation[-2]
    curve_shift = (
        real_power_curve.wind_speed[1] - real_power_curve.wind_speed[0]
    )
    transformation = HorizontalPowerCurveTranslation(real_power_curve)
    new_power_curve = transformation.apply(curve_shift)

    assert new_power_curve != real_power_curve
    assert np.isclose(real_power_curve[:-2], new_power_curve[1:-1]).all()

    new_co_ws = real_power_curve.wind_speed[15]
    transformation.power_curve._cutoff_wind_speed = new_co_ws
    new_power_curve = transformation.apply(curve_shift)
    mask = new_power_curve.wind_speed >= real_power_curve.wind_speed[15]
    assert (new_power_curve[mask] == 0).all()


def test_power_curve_losses_class_annual_losses_with_transformed_power_curve():
    """Test that the average difference is calculated correctly. """

    windspeed = [0, 10, 20, 30, 40]
    generation = [0, 10, 15, 20, 0]
    power_curve = PowerCurve(windspeed, generation)
    transformation = HorizontalPowerCurveTranslation(power_curve)
    pc_losses = PowerCurveLosses(power_curve, BASIC_WIND_RES)

    new_pc = transformation.apply(10)
    avg_diff = pc_losses.annual_losses_with_transformed_power_curve(new_pc)

    # original power curve: [0, 10, 15, 20, 0]
    # expected power curve: [0,  0, 10, 15, 0]
    # powers from wind resource with original curve: 10 + 15 + 15 = 40
    # powers from wind resource with expected curve: 0 + 10 + 10 = 20
    # expected % difference: (40 - 20) / 40 = 50%

    assert abs(avg_diff - 50) < 1


def test_horizontal_transformation_class_bounds(real_power_curve):
    """Test that shift_bounds are set correctly. """

    transformation = HorizontalPowerCurveTranslation(real_power_curve)
    bounds_min, bounds_max = transformation.bounds
    assert bounds_min == 0
    assert bounds_max <= real_power_curve.cutoff_wind_speed
    assert bounds_max <= max(real_power_curve.wind_speed)


def test_power_curve_losses_class_power_gen_no_losses(simple_power_curve):
    """Test that power_gen_no_losses is calculated correctly. """

    pc_losses = PowerCurveLosses(simple_power_curve, BASIC_WIND_RES)

    # powers from wind resource: 20 + 15 + 15 = 50
    assert abs(pc_losses.power_gen_no_losses - 50) < 1E-6


def test_power_curve_class_cutoff_wind_speed(
    simple_power_curve, real_power_curve
):
    """Test that cutoff_wind_speed is calculated correctly. """

    assert simple_power_curve.cutoff_wind_speed == np.inf
    assert (
        real_power_curve.cutoff_wind_speed == real_power_curve.wind_speed[-1]
    )

    power_curve = PowerCurve(
        real_power_curve.wind_speed[:-1], real_power_curve.generation[:-1],
    )
    assert power_curve.cutoff_wind_speed == np.inf


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
