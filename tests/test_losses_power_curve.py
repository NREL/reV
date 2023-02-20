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
import pandas as pd

from reV import TESTDATADIR
from reV.generation.generation import Gen
from reV.utilities.exceptions import reVLossesValueError, reVLossesWarning
from reV.losses.power_curve import (PowerCurve, PowerCurveLosses,
                                    PowerCurveLossesMixin,
                                    PowerCurveLossesInput,
                                    TRANSFORMATIONS,
                                    HorizontalTranslation,
                                    AbstractPowerCurveTransformation
                                    )
from reV.losses.scheduled import ScheduledLossesMixin


REV_POINTS = list(range(3))
RES_FILE = TESTDATADIR + '/wtk/ri_100_wtk_2012.h5'
SAM_FILES = [
    TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json',
    TESTDATADIR + '/SAM/wind_gen_non_standard_0.json',
    TESTDATADIR + '/SAM/wind_gen_non_standard_1.json',
    TESTDATADIR + '/SAM/wind_gen_non_standard_2.json'
]
BASIC_WIND_RES = [10, 20, 20]
SINGLE_SITE_PC_LOSSES = {
    'target_losses_percent': 16,
    'transformation': 'horizontal_translation'
}


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
@pytest.mark.parametrize('transformation', TRANSFORMATIONS)
def test_power_curve_losses(generic_losses, target_losses, transformation):
    """Test full gen run with scheduled losses. """
    gen_profiles, gen_profiles_with_losses = _run_gen_with_and_without_losses(
        generic_losses,
        target_losses=target_losses,
        transformation=transformation
    )

    assert np.isclose(gen_profiles, gen_profiles_with_losses).any()
    assert gen_profiles.max() == gen_profiles_with_losses.max()

    if target_losses > 0:
        assert (gen_profiles - gen_profiles_with_losses > 0).any()
    else:
        assert np.allclose(gen_profiles, gen_profiles_with_losses)

    annual_gen_ratio = (gen_profiles_with_losses.sum() / gen_profiles.sum())
    assert ((1 - annual_gen_ratio) * 100 - target_losses) < 1


@pytest.mark.parametrize('generic_losses', [0, 0.2])
def test_power_curve_losses_site_specific(generic_losses):
    """Test full gen run with scheduled losses. """
    gen_profiles, gen_profiles_with_losses = _run_gen_with_and_without_losses(
        generic_losses,
        target_losses=10,
        site_losses=SINGLE_SITE_PC_LOSSES,
        transformation='exponential_stretching'
    )

    target_losses = SINGLE_SITE_PC_LOSSES['target_losses_percent']

    assert np.isclose(gen_profiles, gen_profiles_with_losses).any()
    assert gen_profiles.max() == gen_profiles_with_losses.max()
    assert (gen_profiles - gen_profiles_with_losses > 0).any()

    annual_gen_ratio = (gen_profiles_with_losses.sum() / gen_profiles.sum())
    assert ((1 - annual_gen_ratio) * 100 - target_losses) < 1


def _run_gen_with_and_without_losses(
    generic_losses, target_losses, transformation, include_outages=False,
    site_losses=None
):
    """Run generation with and without losses for testing. """

    sam_file = SAM_FILES[0]

    with open(sam_file, 'r', encoding='utf-8') as fh:
        sam_config = json.load(fh)

    with tempfile.TemporaryDirectory() as td:
        del sam_config['wind_farm_losses_percent']
        sam_config['turb_generic_loss'] = generic_losses

        sam_config[PowerCurveLossesMixin.POWER_CURVE_CONFIG_KEY] = {
            'target_losses_percent': target_losses,
            'transformation': transformation
        }
        if include_outages:
            sam_config[ScheduledLossesMixin.OUTAGE_CONFIG_KEY] = [
                {
                    'count': 5,
                    'duration': 24,
                    'percentage_of_capacity_lost': 100,
                    'allowed_months': ['January'],
                    'allow_outage_overlap': True
                }
            ]
        sam_fp = os.path.join(td, 'gen.json')
        with open(sam_fp, 'w+') as fh:
            fh.write(json.dumps(sam_config))

        site_data = _make_site_data_df(site_losses)
        gen = Gen.reV_run('windpower', REV_POINTS, sam_fp, RES_FILE,
                          output_request=('gen_profile'), site_data=site_data,
                          max_workers=None, sites_per_worker=3, out_fpath=None)
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
                      max_workers=None, sites_per_worker=3, out_fpath=None)
    gen_profiles = gen.out['gen_profile']

    for ind, row in gen.meta.iterrows():
        time_shift = row['timezone']
        gen_profiles[:, ind] = np.roll(gen_profiles[:, ind], time_shift)

    return gen_profiles, gen_profiles_with_losses


def _make_site_data_df(site_data):
    """Make site data DataFrame for a specific power curve loss input. """
    if site_data is not None:
        site_specific_losses = [json.dumps(site_data)] * len(REV_POINTS)
        site_data_dict = {
            'gid': REV_POINTS,
            PowerCurveLossesMixin.POWER_CURVE_CONFIG_KEY: site_specific_losses
        }
        site_data = pd.DataFrame(site_data_dict)
    return site_data


def test_power_curve_losses_witch_scheduled_outages():
    """Test full gen run with scheduled losses. """
    gen_profiles, gen_profiles_with_losses = _run_gen_with_and_without_losses(
        generic_losses=0.2,
        target_losses=20, transformation='exponential_stretching',
        include_outages=True
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
    mixin.sam_sys_inputs[PowerCurveLossesMixin.POWER_CURVE_CONFIG_KEY] = {
        'target_losses_percent': 10,
        'transformation': 'horizontal_translation'
    }
    # order is [(temp_C, pressure_ATM, windspeed_m/s, windir)]
    mixin.sam_sys_inputs['wind_resource_data'] = {
        'data': [(20, 1, val, 0) for val in BASIC_WIND_RES]
    }
    mixin.add_power_curve_losses()
    new_power_curve = np.array(
        mixin.sam_sys_inputs["wind_turbine_powercurve_powerout"]
    )

    assert mixin.POWER_CURVE_CONFIG_KEY not in mixin.sam_sys_inputs
    assert any(og_power_curve != new_power_curve)


@pytest.mark.parametrize('config', SAM_FILES[0:2])
def test_power_curve_losses_mixin_class_wind_resource_too_high(config):
    """Test mixin class behavior when wind resource is too high. """

    with open(config, 'r') as fh:
        sam_config = json.load(fh)

    og_power_curve = np.array(sam_config["wind_turbine_powercurve_powerout"])

    # patch required for 'wind_resource_data' access below
    def get_item_patch(self, key):
        return self.sam_sys_inputs.get(key)
    PowerCurveLossesMixin.__getitem__ = get_item_patch

    mixin = PowerCurveLossesMixin()
    mixin.sam_sys_inputs = copy.deepcopy(sam_config)
    mixin.sam_sys_inputs[PowerCurveLossesMixin.POWER_CURVE_CONFIG_KEY] = {
        'target_losses_percent': 10,
        'transformation': 'horizontal_translation'
    }
    mixin.sam_sys_inputs['wind_resource_data'] = {
        'data': [(1, 10, 1_000, 0) for __ in BASIC_WIND_RES]
    }
    with pytest.warns(reVLossesWarning):
        mixin.add_power_curve_losses()
    new_power_curve = np.array(
        mixin.sam_sys_inputs["wind_turbine_powercurve_powerout"]
    )

    assert mixin.POWER_CURVE_CONFIG_KEY not in mixin.sam_sys_inputs
    assert (og_power_curve == new_power_curve).all()


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

    assert mixin.POWER_CURVE_CONFIG_KEY not in mixin.sam_sys_inputs
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


@pytest.mark.parametrize('bad_weights', ([], [1]))
def test_power_curve_losses_class_bad_weights_input(bad_weights):
    """Test that error is raised for bad weights input. """
    wind_speed = [0, 10]
    generation = [10, 100]
    wind_res = [0, 0, 5]
    power_curve = PowerCurve(wind_speed, generation)
    with pytest.raises(reVLossesValueError) as excinfo:
        PowerCurveLosses(power_curve, wind_res, weights=bad_weights)
    assert "Invalid weights input" in str(excinfo.value)


@pytest.mark.parametrize('pc_transformation', TRANSFORMATIONS.values())
def test_transformation_classes_apply(pc_transformation, real_power_curve):
    """Test that the power curve transformations are applied correctly. """

    real_power_curve.generation[-1] = real_power_curve.generation[-2]
    transformation = pc_transformation(real_power_curve)
    min_b, max_b = transformation.bounds
    strength = min_b + (max_b - min_b) / 4
    new_power_curve = transformation.apply(strength)

    assert new_power_curve != real_power_curve

    new_co_ws = real_power_curve.wind_speed[-10]
    transformation.power_curve._cutoff_wind_speed = new_co_ws
    new_power_curve = transformation.apply(strength)
    mask = new_power_curve.wind_speed >= real_power_curve.wind_speed[-10]
    assert (new_power_curve[mask] == 0).all()


def test_horizontal_transformation_class_apply(real_power_curve):
    """Test apply method for power curve shift in particular."""
    real_power_curve.generation[-1] = real_power_curve.generation[-2]
    curve_shift = (
        real_power_curve.wind_speed[1] - real_power_curve.wind_speed[0]
    )
    transformation = HorizontalTranslation(real_power_curve)
    new_power_curve = transformation.apply(curve_shift)

    assert np.allclose(real_power_curve[:-2], new_power_curve[1:-1])


def test_power_curve_losses_class_annual_losses_with_transformed_power_curve():
    """Test that the average difference is calculated correctly. """

    wind_speed = [0, 10, 20, 30, 40]
    generation = [0, 10, 15, 20, 0]
    power_curve = PowerCurve(wind_speed, generation)
    transformation = HorizontalTranslation(power_curve)
    pc_losses = PowerCurveLosses(power_curve, BASIC_WIND_RES)

    new_pc = transformation.apply(10)
    avg_diff = pc_losses.annual_losses_with_transformed_power_curve(new_pc)

    # original power curve: [0, 10, 15, 20, 0]
    # expected power curve: [0,  0, 10, 15, 0]
    # powers from wind resource with original curve: 10 + 15 + 15 = 40
    # powers from wind resource with expected curve: 0 + 10 + 10 = 20
    # expected % difference: (40 - 20) / 40 = 50%

    assert abs(avg_diff - 50) < 1


@pytest.mark.parametrize('sam_file', SAM_FILES)
@pytest.mark.parametrize('pc_transformation', TRANSFORMATIONS.values())
def test_transformation_classes_bounds(sam_file, pc_transformation):
    """Test that shift_bounds are set correctly. """

    with open(sam_file, 'r') as fh:
        sam_config = json.load(fh)

    wind_speed = sam_config['wind_turbine_powercurve_windspeeds']
    generation = sam_config['wind_turbine_powercurve_powerout']
    power_curve = PowerCurve(wind_speed, generation)

    transformation = pc_transformation(power_curve)
    bounds_min, bounds_max = transformation.bounds

    assert bounds_max > bounds_min
    assert bounds_max <= power_curve.cutoff_wind_speed
    assert bounds_max <= max(power_curve.wind_speed)


def test_transformation_invalid_result(real_power_curve):
    """Test a transformation with invalid result. """

    transformation = HorizontalTranslation(real_power_curve)
    with pytest.raises(reVLossesValueError) as excinfo:
        transformation.apply(transformation.bounds[-1] + 0.2)

    err_msg = str(excinfo.value)
    assert "Calculated power curve is invalid" in err_msg
    assert "No power generation below the cutoff wind speed" in err_msg


def test_power_curve_loss_input_class_valid_inputs():
    """Test PowerCurveLossesInput class with valid input. """

    specs = {'target_losses_percent': 50}
    pc_input = PowerCurveLossesInput(specs)

    assert abs(pc_input.target - 50) < 1E-6
    assert pc_input.transformation in TRANSFORMATIONS.values()

    assert '50' in str(pc_input)
    assert any(t in str(pc_input) for t in TRANSFORMATIONS)


@pytest.mark.parametrize('bad_percent', [-10, 105])
def test_power_curve_loss_input_class_bad_percent_input(bad_percent):
    """Test PowerCurveLossesInput class with bad percent input. """

    bad_specs = {'target_losses_percent': bad_percent}

    with pytest.raises(reVLossesValueError) as excinfo:
        PowerCurveLossesInput(bad_specs)
    assert "Percentage" in str(excinfo.value)
    assert "must be in the range [0, 100]" in str(excinfo.value)


def test_power_curve_loss_input_class_bad_transformation_input():
    """Test PowerCurveLossesInput class with bad transformation input. """

    bad_specs = {'target_losses_percent': 50, 'transformation': 'DNE'}

    with pytest.raises(reVLossesValueError) as excinfo:
        PowerCurveLossesInput(bad_specs)
    assert "Transformation" in str(excinfo.value)
    assert "not understood!" in str(excinfo.value)


def test_power_curve_loss_input_class_missing_required_keys():
    """Test PowerCurveLossesInput class with missing keys input. """

    with pytest.raises(reVLossesValueError) as excinfo:
        PowerCurveLossesInput({})
    assert "The following required keys are missing" in str(excinfo.value)


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


def test_power_curve_class_comparisons(simple_power_curve):
    """Test power curve class comparison and call operators. """

    assert simple_power_curve == [0, 20, 15, 10]
    assert simple_power_curve != [0, 20, 15, 0]
    assert sum(simple_power_curve < 15) == 2
    assert sum(simple_power_curve <= 15) == 3
    assert sum(simple_power_curve > 15) == 1
    assert sum(simple_power_curve >= 15) == 2

    assert simple_power_curve(5) == 10


def test_bad_transformation_implementation(real_power_curve):
    """Test an invalid transformation implementation. """

    class NewTransformation(AbstractPowerCurveTransformation):
        """Test class"""
        # pylint: disable=useless-super-delegation
        def apply(self, *args, **kwargs):
            """Test apply method."""
            return super().apply(*args, **kwargs)

        @property
        def bounds(self):
            """Test bounds."""
            return (0, 1)

    transformation = NewTransformation(real_power_curve)
    with pytest.raises(NotImplementedError) as excinfo:
        transformation.apply(0.5)

    err_msg = str(excinfo.value)
    assert "Transformation implementation" in err_msg
    assert "did not set the `_transformed_generation` attribute" in err_msg


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
