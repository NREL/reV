# -*- coding: utf-8 -*-
"""
PyTest file for reV powercurve losses.

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
from reV.losses.powercurve import PowercurveLosses, PowercurveLossesMixin
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
def basic_powercurve():
    """Return a basic powercurve."""
    with open(SAM_FILES[0], 'r') as fh:
        sam_config = json.load(fh)

    wind_speed = sam_config['wind_turbine_powercurve_windspeeds']
    powercurve = sam_config['wind_turbine_powercurve_powerout']
    return wind_speed, powercurve


@pytest.mark.parametrize('generic_losses', [0, 0.2])
@pytest.mark.parametrize('target_losses', [0, 10, 50])
def test_powercurve_losses(generic_losses, target_losses):
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

        sam_config[PowercurveLossesMixin.POWERCURVE_CONFIG_KEY] = target_losses
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


def test_powercurve_losses_witch_scheduled_outages():
    """Test full gen run with scheduled losses. """
    gen_profiles, gen_profiles_with_losses = _run_gen_with_and_without_losses(
        generic_losses=0.2, target_losses=20, include_outages=True
    )

    annual_gen_ratio = (gen_profiles_with_losses.sum() / gen_profiles.sum())
    assert (1 - annual_gen_ratio) * 100 > 21  # 1% tolerance


@pytest.mark.parametrize('config', SAM_FILES)
def test_powercurve_losses_mixin_class_add_powercurve_losses(config):
    """Test mixin class behavior when adding losses. """

    with open(config, 'r') as fh:
        sam_config = json.load(fh)

    og_powercurve = np.array(sam_config["wind_turbine_powercurve_powerout"])

    # patch required for 'wind_resource_data' access below
    def get_item_patch(self, key):
        return self.sam_sys_inputs.get(key)
    PowercurveLossesMixin.__getitem__ = get_item_patch

    mixin = PowercurveLossesMixin()
    mixin.sam_sys_inputs = copy.deepcopy(sam_config)
    mixin.sam_sys_inputs[PowercurveLossesMixin.POWERCURVE_CONFIG_KEY] = 10
    mixin.sam_sys_inputs['wind_resource_data'] = {
        'data': [(0, 0, val) for val in BASIC_WIND_RES]
    }
    mixin.add_powercurve_losses()
    new_powercurve = np.array(
        mixin.sam_sys_inputs["wind_turbine_powercurve_powerout"]
    )

    assert mixin.POWERCURVE_CONFIG_KEY not in mixin.sam_sys_inputs
    assert any(og_powercurve != new_powercurve)


@pytest.mark.parametrize('config', SAM_FILES)
def test_powercurve_losses_mixin_class_no_losses_input(config):
    """Test mixin class behavior when no losses should be added. """

    with open(config, 'r') as fh:
        sam_config = json.load(fh)

    og_powercurve = np.array(sam_config["wind_turbine_powercurve_powerout"])

    mixin = PowercurveLossesMixin()
    mixin.sam_sys_inputs = copy.deepcopy(sam_config)
    mixin.add_powercurve_losses()
    new_powercurve = np.array(
        mixin.sam_sys_inputs["wind_turbine_powercurve_powerout"]
    )

    assert mixin.POWERCURVE_CONFIG_KEY not in mixin.sam_sys_inputs
    assert (og_powercurve == new_powercurve).all()


@pytest.mark.parametrize('bad_wind_speed', ([], [-10, 10]))
def test_powercurve_losses_class_bad_windspeed_input(bad_wind_speed):
    """Test that error is raised for bad windspeed inputs. """
    powercurve = [10, 100]

    with pytest.raises(reVLossesValueError) as excinfo:
        PowercurveLosses(bad_wind_speed, powercurve, BASIC_WIND_RES)
    assert "Invalid wind speed input" in str(excinfo.value)


@pytest.mark.parametrize('bad_powercurve', ([], [0, 0, 0, 0], [0, 20, 0, 10]))
def test_powercurve_losses_class_bad_powercurve_input(bad_powercurve):
    """Test that error is raised for bad powercurve inputs. """
    wind_speed = [0, 10, 20, 30]

    with pytest.raises(reVLossesValueError) as excinfo:
        PowercurveLosses(wind_speed, bad_powercurve, BASIC_WIND_RES)
    assert "Invalid powercurve input" in str(excinfo.value)


@pytest.mark.parametrize('bad_wind_res', ([], [-10, 10]))
def test_powercurve_losses_class_bad_wind_res_input(bad_wind_res):
    """Test that error is raised for bad wind resource inputs. """
    bad_wind_speed = [0, 10]
    powercurve = [10, 100]

    with pytest.raises(reVLossesValueError) as excinfo:
        PowercurveLosses(bad_wind_speed, powercurve, bad_wind_res)
    assert "Invalid wind resource input" in str(excinfo.value)


def test_powercurve_losses_class_apply_shift(basic_powercurve):
    """Test that the powercurve shift is applied correctly. """

    windspeed, powercurve = basic_powercurve
    powercurve[-1] = powercurve[-2]
    pc_losses = PowercurveLosses(windspeed, powercurve, BASIC_WIND_RES)
    curve_shift = windspeed[1] - windspeed[0]

    new_powercurve = pc_losses.apply_shift(shift=curve_shift)
    assert new_powercurve.shape == pc_losses.powercurve.shape
    assert not np.isclose(new_powercurve, powercurve).all()
    assert not (new_powercurve == pc_losses.powercurve).all()
    assert np.isclose(powercurve[:-1], new_powercurve[1:]).all()

    pc_losses._cutoff_wind_speed = windspeed[10]
    new_powercurve = pc_losses.apply_shift(shift=curve_shift)
    assert (new_powercurve[10:] == 0).all()


def test_powercurve_losses_class_annual_losses_with_shifted_powercurve():
    """Test that the average difference is calculated correctly. """

    windspeed = [0, 10, 20, 30, 40]
    powercurve = [0, 10, 15, 20, 0]
    pc_losses = PowercurveLosses(windspeed, powercurve, BASIC_WIND_RES)
    avg_diff = pc_losses.annual_losses_with_shifted_powercurve(shift=10)

    # original powercurve: [0, 10, 15, 20, 0]
    # expected powercurve: [0,  0, 10, 15, 0]
    # powers from wind resource with original curve: 10 + 15 + 15 = 40
    # powers from wind resource with expected curve: 0 + 10 + 10 = 20
    # expected % difference: (40 - 20) / 40 = 50%

    assert abs(avg_diff - 50) < 1


def test_powercurve_losses_class_cutoff_wind_speed(basic_powercurve):
    """Test that cutoff_wind_speed is calculated correctly. """

    windspeed, powercurve = basic_powercurve
    pc_losses = PowercurveLosses(windspeed, powercurve, BASIC_WIND_RES)
    assert pc_losses.cutoff_wind_speed == windspeed[-1]

    pc_losses = PowercurveLosses(
        windspeed[:-1], powercurve[:-1], BASIC_WIND_RES
    )
    assert pc_losses.cutoff_wind_speed == np.inf

    windspeed = [0, 10, 20, 30]
    powercurve = [0, 20, 15, 10]

    pc_losses = PowercurveLosses(
        windspeed[:-1], powercurve[:-1], BASIC_WIND_RES
    )
    assert pc_losses.cutoff_wind_speed == np.inf


def test_powercurve_losses_class_shift_bounds(basic_powercurve):
    """Test that shift_bounds are set correctly. """

    windspeed, powercurve = basic_powercurve
    pc_losses = PowercurveLosses(windspeed, powercurve, BASIC_WIND_RES)
    bounds_min, bounds_max = pc_losses.shift_bounds
    assert bounds_min == 0
    assert bounds_max <= pc_losses.cutoff_wind_speed
    assert bounds_max <= max(windspeed)


def test_powercurve_losses_class_power_gen_no_losses():
    """Test that power_gen_no_losses is calculated correctly. """

    windspeed = [0, 10, 20, 30]
    powercurve = [0, 20, 15, 10]

    pc_losses = PowercurveLosses(windspeed, powercurve, BASIC_WIND_RES)

    # powers from wind resource: 20 + 15 + 15 = 50
    assert abs(pc_losses.power_gen_no_losses - 50) < 1E-6


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
