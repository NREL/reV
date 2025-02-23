# -*- coding: utf-8 -*-
"""
PyTest file for reV scheduled loss.

Created on Mon Apr 18 12:52:16 2021

@author: ppinchuk
"""

import copy
import glob
import json
import os
import random
import tempfile
import traceback

import numpy as np
import pandas as pd
import pytest
from rex.utilities.utilities import safe_json_load

from reV import TESTDATADIR
from reV.cli import main
from reV.generation.generation import Gen
from reV.handlers.outputs import Outputs
from reV.losses.scheduled import (
    Outage,
    OutageScheduler,
    ScheduledLossesMixin,
    SingleOutageScheduler,
)
from reV.losses.utils import hourly_indices_for_months
from reV.utilities.exceptions import reVLossesValueError, reVLossesWarning
from reV.utilities import ResourceMetaField

REV_POINTS = list(range(3))
RTOL = 0
ATOL = 0.001
WIND_SAM_FILE = TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json'
WIND_RES_FILE = TESTDATADIR + '/wtk/ri_100_wtk_2012.h5'
PV_SAM_FILE = TESTDATADIR + '/SAM/naris_pv_1axis_inv13.json'
PV_RES_FILE = TESTDATADIR + '/nsrdb/ri_100_nsrdb_2012.h5'
NOMINAL_OUTAGES = [
    [
        {
            'count': 5,
            'duration': 24,
            'percentage_of_capacity_lost': 100,
            'allowed_months': ['January'],
            'allow_outage_overlap': True
        },
        {
            'count': 5,
            'duration': 10,
            'percentage_of_capacity_lost': 60,
            'allowed_months': ['January'],
            'allow_outage_overlap': True
        },
        {
            'count': 5,
            'duration': 5,
            'percentage_of_capacity_lost': 53,
            'allowed_months': ['January'],
            'allow_outage_overlap': False
        },
        {
            'count': 100,
            'duration': 1,
            'percentage_of_capacity_lost': 17,
            'allowed_months': ['January'],
            'allow_outage_overlap': False
        },
        {
            'count': 100,
            'duration': 2,
            'percentage_of_capacity_lost': 7,
            'allowed_months': ['January'],
            'allow_outage_overlap': True
        }
    ],
    [
        {
            'count': 1,
            'duration': 744,
            'percentage_of_capacity_lost': 10,
            'allowed_months': ['January'],
            'allow_outage_overlap': True
        },
        {
            'count': 5,
            'duration': 10,
            'percentage_of_capacity_lost': 17,
            'allowed_months': ['January'],
            'allow_outage_overlap': True
        }
    ]
]
SINGLE_SITE_OUTAGE = [{
    'count': 100,
    'duration': 2,
    'percentage_of_capacity_lost': 42,
    'allowed_months': ['February'],
}]


@pytest.fixture
def basic_outage_dict():
    """Return a basic outage dictionary."""
    outage_info = {
        'count': 5,
        'duration': 24,
        'percentage_of_capacity_lost': 100,
        'allowed_months': ['Jan']
    }
    return outage_info


@pytest.fixture
def so_scheduler(basic_outage_dict):
    """Return a basic initialized `SingleOutageScheduler` object."""
    outage = Outage(basic_outage_dict)
    scheduler = OutageScheduler([])
    return SingleOutageScheduler(outage, scheduler)


@pytest.mark.parametrize('generic_losses', [0, 0.2])
@pytest.mark.parametrize('outages', NOMINAL_OUTAGES)
@pytest.mark.parametrize('haf', [None, np.random.randint(0, 100, 8760)])
@pytest.mark.parametrize('files', [
    (WIND_SAM_FILE, WIND_RES_FILE, 'windpower'),
    (PV_SAM_FILE, PV_RES_FILE, 'pvwattsv5'),
    (PV_SAM_FILE, PV_RES_FILE, 'pvwattsv7')
])
def test_scheduled_losses(generic_losses, outages, haf, files):
    """Test full gen run with scheduled losses."""

    gen_profiles, gen_profiles_with_losses = _run_gen_with_and_without_losses(
        generic_losses, outages, None, haf, files
    )

    outages = [Outage(outage) for outage in outages]
    min_loss = min(outage.percentage_of_capacity_lost / 100
                   for outage in outages)
    assert (gen_profiles - gen_profiles_with_losses >= min_loss).any()

    losses = (1 - (gen_profiles_with_losses / gen_profiles)) * 100
    site_loss_inds = []
    zero_gen_inds_all_sites = set()
    for site_losses, site_gen in zip(losses.T, gen_profiles.T):
        non_zero_gen = site_gen > 0
        zero_gen_inds = set(np.where(~non_zero_gen)[0])
        zero_gen_inds_all_sites |= zero_gen_inds
        site_loss_inds += [set(np.where(site_losses > 0 & non_zero_gen)[0])]
        for outage in outages:
            outage_percentage = outage.percentage_of_capacity_lost
            outage_allowed_hourly_inds = hourly_indices_for_months(
                outage.allowed_months
            )
            zero_gen_in_comparison_count = sum(
                ind in outage_allowed_hourly_inds for ind in zero_gen_inds
            )
            comparison_inds = list(
                set(outage_allowed_hourly_inds) - zero_gen_inds
            )

            if not outage.allow_outage_overlap or outage_percentage == 100:
                min_num_expected_outage_hours = (
                    outage.count * outage.duration
                    - zero_gen_in_comparison_count
                )
                max_num_expected_outage_hours = (
                    outage.count * outage.duration
                )
                observed_outages = np.isclose(
                    site_losses[comparison_inds], outage_percentage,
                    atol=ATOL, rtol=RTOL
                )
            else:
                num_outages_possible_per_day = np.floor(
                    100 / outage.percentage_of_capacity_lost
                )
                min_num_expected_outage_hours = (
                    outage.count * outage.duration
                    / num_outages_possible_per_day
                )
                min_num_expected_outage_hours = max(
                    0,
                    min_num_expected_outage_hours
                    - zero_gen_in_comparison_count
                )
                max_num_expected_outage_hours = len(comparison_inds)
                observed_outages = (
                    site_losses[comparison_inds] >= outage_percentage - ATOL
                )

            num_outage_hours = observed_outages.sum()

            num_outage_hours_meet_expectations = (
                min_num_expected_outage_hours
                <= num_outage_hours
                <= max_num_expected_outage_hours
            )
            err_msg = (f"{min_num_expected_outage_hours=}, "
                       f"{num_outage_hours=}, "
                       f"{max_num_expected_outage_hours=}")
            assert num_outage_hours_meet_expectations, err_msg

        total_expected_outage = sum(
            outage.count * outage.duration * outage.percentage_of_capacity_lost
            for outage in outages
        )
        assert 0 < site_losses[non_zero_gen].sum() <= total_expected_outage

    outages_allow_different_scheduled_losses = []
    for outage in outages:
        if not outage.allow_outage_overlap or outage_percentage == 100:
            outages_allow_different_scheduled_losses.append(
                outage.total_available_hours <= outage.duration * outage.count
            )
        else:
            outages_allow_different_scheduled_losses.append(
                outage.total_available_hours <= outage.duration
            )

    if all(outages_allow_different_scheduled_losses):
        site_loss_inds = [
            inds - zero_gen_inds_all_sites for inds in site_loss_inds
        ]
        common_inds = set.intersection(*site_loss_inds)

        error_msg = "Scheduled losses do not vary between sites!"
        assert any(inds - common_inds for inds in site_loss_inds), error_msg


@pytest.mark.parametrize('generic_losses', [0, 0.2])
@pytest.mark.parametrize('haf', [None, np.random.randint(0, 100, 8760)])
@pytest.mark.parametrize('files', [
    (WIND_SAM_FILE, WIND_RES_FILE, 'windpower'),
    (PV_SAM_FILE, PV_RES_FILE, 'pvwattsv5'),
    (PV_SAM_FILE, PV_RES_FILE, 'pvwattsv7')
])
def test_scheduled_losses_site_specific(generic_losses, haf, files):
    """Test full gen run with scheduled losses."""

    gen_profiles, gen_profiles_with_losses = _run_gen_with_and_without_losses(
        generic_losses, NOMINAL_OUTAGES[0], SINGLE_SITE_OUTAGE, haf, files
    )

    outages = [Outage(outage) for outage in SINGLE_SITE_OUTAGE]
    min_loss = min(outage.percentage_of_capacity_lost / 100
                   for outage in outages)
    assert (gen_profiles - gen_profiles_with_losses >= min_loss).any()

    losses = (1 - (gen_profiles_with_losses / gen_profiles)) * 100
    site_loss_inds = []
    zero_gen_inds_all_sites = set()
    for site_losses, site_gen in zip(losses.T, gen_profiles.T):
        non_zero_gen = site_gen > 0
        zero_gen_inds = set(np.where(~non_zero_gen)[0])
        zero_gen_inds_all_sites |= zero_gen_inds
        site_loss_inds += [set(np.where(site_losses > 0 & non_zero_gen)[0])]
        for outage in outages:
            outage_percentage = outage.percentage_of_capacity_lost
            outage_allowed_hourly_inds = hourly_indices_for_months(
                outage.allowed_months
            )
            zero_gen_in_comparison_count = sum(
                ind in outage_allowed_hourly_inds for ind in zero_gen_inds
            )
            comparison_inds = list(
                set(outage_allowed_hourly_inds) - zero_gen_inds
            )
            num_outages_possible_per_day = np.floor(
                100 / outage.percentage_of_capacity_lost
            )
            min_num_expected_outage_hours = (
                outage.count * outage.duration
                / num_outages_possible_per_day
            )
            min_num_expected_outage_hours = max(
                0,
                min_num_expected_outage_hours
                - zero_gen_in_comparison_count
            )
            max_num_expected_outage_hours = len(comparison_inds)
            observed_outages = (
                site_losses[comparison_inds] >= outage_percentage - ATOL
            )

            num_outage_hours = observed_outages.sum()

            num_outage_hours_meet_expectations = (
                min_num_expected_outage_hours
                <= num_outage_hours
                <= max_num_expected_outage_hours
            )
            assert num_outage_hours_meet_expectations

        total_expected_outage = sum(
            outage.count * outage.duration * outage.percentage_of_capacity_lost
            for outage in outages
        )
        assert 0 < site_losses[non_zero_gen].sum() <= total_expected_outage

    outages_allow_different_scheduled_losses = []
    for outage in outages:
        outages_allow_different_scheduled_losses.append(
            outage.total_available_hours <= outage.duration
        )

    if all(outages_allow_different_scheduled_losses):
        site_loss_inds = [
            inds - zero_gen_inds_all_sites for inds in site_loss_inds
        ]
        common_inds = set.intersection(*site_loss_inds)

        error_msg = "Scheduled losses do not vary between sites!"
        assert any(inds - common_inds for inds in site_loss_inds), error_msg


def _run_gen_with_and_without_losses(
    generic_losses, outages, site_outages, haf, files
):
    """Run generation with and without losses for testing."""
    sam_file, res_file, tech = files
    with open(sam_file) as fh:
        sam_config = json.load(fh)

    with tempfile.TemporaryDirectory() as td:
        if tech == 'windpower':
            del sam_config['wind_farm_losses_percent']
            sam_config['turb_generic_loss'] = generic_losses
        else:
            sam_config['losses'] = generic_losses

        if haf is not None:
            sam_config['hourly'] = haf.tolist()

        sam_config[ScheduledLossesMixin.OUTAGE_CONFIG_KEY] = outages
        sam_fp = os.path.join(td, 'gen.json')
        with open(sam_fp, 'w+') as fh:
            fh.write(json.dumps(sam_config))

        site_data = _make_site_data_df(site_outages)
        gen = Gen(tech, REV_POINTS, sam_fp, res_file,
                  output_request=('gen_profile'), site_data=site_data,
                  sites_per_worker=3)
        gen.run(max_workers=None)
    gen_profiles_with_losses = gen.out['gen_profile']
    # subsample to hourly generation
    time_steps_in_hour = int(round(gen_profiles_with_losses.shape[0] / 8760))
    gen_profiles_with_losses = gen_profiles_with_losses[::time_steps_in_hour]
    # undo UTC array rolling
    for ind, row in gen.meta.iterrows():
        time_shift = row[ResourceMetaField.TIMEZONE]
        gen_profiles_with_losses[:, ind] = np.roll(
            gen_profiles_with_losses[:, ind], time_shift
        )

    pc = Gen.get_pc(REV_POINTS, None, sam_file, tech,
                    sites_per_worker=3, res_file=res_file)
    if tech == 'windpower':
        del pc.project_points.sam_inputs[sam_file]['wind_farm_losses_percent']
        pc.project_points.sam_inputs[sam_file]['turb_generic_loss'] = (
            generic_losses
        )
    else:
        pc.project_points.sam_inputs[sam_file]['losses'] = generic_losses

    if haf is not None:
        pc.project_points.sam_inputs[sam_file]['hourly'] = haf.tolist()

    gen = Gen(tech, pc, sam_file, res_file, output_request=('gen_profile'),
              sites_per_worker=3)
    gen.run(max_workers=None)
    gen_profiles = gen.out['gen_profile']
    time_steps_in_hour = int(round(gen_profiles.shape[0] / 8760))
    gen_profiles = gen_profiles[::time_steps_in_hour]
    for ind, row in gen.meta.iterrows():
        time_shift = row[ResourceMetaField.TIMEZONE]
        gen_profiles[:, ind] = np.roll(gen_profiles[:, ind], time_shift)

    return gen_profiles, gen_profiles_with_losses


def _make_site_data_df(site_data):
    """Make site data DataFrame for a specific outage input."""
    if site_data is not None:
        site_specific_outages = [json.dumps(site_data)] * len(REV_POINTS)
        site_data_dict = {
            ResourceMetaField.GID: REV_POINTS,
            ScheduledLossesMixin.OUTAGE_CONFIG_KEY: site_specific_outages
        }
        site_data = pd.DataFrame(site_data_dict)
    return site_data


@pytest.mark.parametrize('generic_losses', [0, 0.2])
@pytest.mark.parametrize('outages', NOMINAL_OUTAGES)
@pytest.mark.parametrize('site_outages', [None, SINGLE_SITE_OUTAGE])
@pytest.mark.parametrize('files', [
    (WIND_SAM_FILE, WIND_RES_FILE, 'windpower'),
    (PV_SAM_FILE, PV_RES_FILE, 'pvwattsv5'),
    (PV_SAM_FILE, PV_RES_FILE, 'pvwattsv7')
])
def test_scheduled_losses_repeatability(
    generic_losses, outages, site_outages, files
):
    """Test that losses are reproducible between runs."""
    sam_file, res_file, tech = files
    with open(sam_file) as fh:
        sam_config = json.load(fh)

    with tempfile.TemporaryDirectory() as td:
        if tech == 'windpower':
            del sam_config['wind_farm_losses_percent']
            sam_config['turb_generic_loss'] = generic_losses
        else:
            sam_config['losses'] = generic_losses

        sam_config[ScheduledLossesMixin.OUTAGE_CONFIG_KEY] = outages
        sam_fp = os.path.join(td, 'gen.json')
        with open(sam_fp, 'w+') as fh:
            fh.write(json.dumps(sam_config))

        site_data = _make_site_data_df(site_outages)
        gen = Gen(tech, REV_POINTS, sam_fp, res_file,
                  output_request=('gen_profile'), site_data=site_data,
                  sites_per_worker=3)
        gen.run(max_workers=None)
        gen_profiles_first_run = gen.out['gen_profile']

        outages = copy.deepcopy(outages)
        random.shuffle(outages)
        sam_config[ScheduledLossesMixin.OUTAGE_CONFIG_KEY] = outages
        with open(sam_fp, 'w+') as fh:
            fh.write(json.dumps(sam_config))

        site_data = _make_site_data_df(site_outages)
        gen = Gen(tech, REV_POINTS, sam_fp, res_file,
                  output_request=('gen_profile'), site_data=site_data,
                  sites_per_worker=3)
        gen.run(max_workers=None)
        gen_profiles_second_run = gen.out['gen_profile']

    assert np.allclose(gen_profiles_first_run, gen_profiles_second_run)


@pytest.mark.parametrize('files', [
    (WIND_SAM_FILE, WIND_RES_FILE, 'windpower'),
    (PV_SAM_FILE, PV_RES_FILE, 'pvwattsv5'),
    (PV_SAM_FILE, PV_RES_FILE, 'pvwattsv7')
])
def test_scheduled_losses_repeatability_with_seed(files):
    """Test that losses are reproducible between runs."""
    sam_file, res_file, tech = files
    outages = copy.deepcopy(NOMINAL_OUTAGES[0])
    with open(sam_file) as fh:
        sam_config = json.load(fh)

    with tempfile.TemporaryDirectory() as td:
        if tech == 'windpower':
            del sam_config['wind_farm_losses_percent']
            sam_config['turb_generic_loss'] = 0.2
        else:
            sam_config['losses'] = 0.2

        sam_config[ScheduledLossesMixin.OUTAGE_CONFIG_KEY] = outages
        sam_config[ScheduledLossesMixin.OUTAGE_SEED_CONFIG_KEY] = 42
        sam_fp = os.path.join(td, 'gen.json')
        with open(sam_fp, 'w+') as fh:
            fh.write(json.dumps(sam_config))

        site_data = _make_site_data_df(SINGLE_SITE_OUTAGE)
        gen = Gen(tech, REV_POINTS, sam_fp, res_file,
                  output_request=('gen_profile'), site_data=site_data,
                  sites_per_worker=3)
        gen.run(max_workers=None)
        gen_profiles_first_run = gen.out['gen_profile']

        random.shuffle(outages)
        sam_config[ScheduledLossesMixin.OUTAGE_CONFIG_KEY] = outages
        sam_config[ScheduledLossesMixin.OUTAGE_SEED_CONFIG_KEY] = 42
        with open(sam_fp, 'w+') as fh:
            fh.write(json.dumps(sam_config))

        site_data = _make_site_data_df(SINGLE_SITE_OUTAGE)
        gen = Gen(tech, REV_POINTS, sam_fp, res_file,
                  output_request=('gen_profile'), site_data=site_data,
                  sites_per_worker=3)
        gen.run(max_workers=None)
        gen_profiles_second_run = gen.out['gen_profile']

        random.shuffle(outages)
        sam_config[ScheduledLossesMixin.OUTAGE_CONFIG_KEY] = outages
        sam_config[ScheduledLossesMixin.OUTAGE_SEED_CONFIG_KEY] = 1234
        with open(sam_fp, 'w+') as fh:
            fh.write(json.dumps(sam_config))

        site_data = _make_site_data_df(SINGLE_SITE_OUTAGE)
        gen = Gen(tech, REV_POINTS, sam_fp, res_file,
                  output_request=('gen_profile'), site_data=site_data,
                  sites_per_worker=3)
        gen.run(max_workers=None)
        gen_profiles_third_run = gen.out['gen_profile']

    assert np.allclose(gen_profiles_first_run, gen_profiles_second_run)
    assert not np.allclose(gen_profiles_first_run, gen_profiles_third_run)
    assert not np.allclose(gen_profiles_second_run, gen_profiles_third_run)


@pytest.mark.parametrize('outages', NOMINAL_OUTAGES)
def test_scheduled_losses_mixin_class_add_scheduled_losses(outages):
    """Test mixin class behavior when adding losses."""

    mixin = ScheduledLossesMixin()
    mixin.sam_sys_inputs = {mixin.OUTAGE_CONFIG_KEY: outages}
    sample_df_with_dt = pd.DataFrame(index=pd.to_datetime(["2020-01-01"]))
    mixin.add_scheduled_losses(sample_df_with_dt)

    assert mixin.OUTAGE_CONFIG_KEY not in mixin.sam_sys_inputs
    assert 'hourly' in mixin.sam_sys_inputs


def test_scheduled_losses_mixin_class_no_losses_input():
    """Test mixin class behavior when adding losses."""

    mixin = ScheduledLossesMixin()
    mixin.sam_sys_inputs = {}
    sample_df_with_dt = pd.DataFrame(index=pd.to_datetime(["2020-01-01"]))
    mixin.add_scheduled_losses(sample_df_with_dt)

    assert mixin.OUTAGE_CONFIG_KEY not in mixin.sam_sys_inputs
    assert 'hourly' not in mixin.sam_sys_inputs


@pytest.mark.parametrize('allow_outage_overlap', [True, False])
def test_single_outage_scheduler_normal_run(
    allow_outage_overlap, so_scheduler
):
    """Test that single outage is scheduled correctly."""

    so_scheduler.outage._specs['allow_outage_overlap'] = allow_outage_overlap
    outage = so_scheduler.outage
    scheduler = so_scheduler.scheduler
    so_scheduler.calculate()

    assert scheduler.total_losses[:744].any()
    assert not scheduler.total_losses[744:].any()

    outage_percentage = outage.percentage_of_capacity_lost
    num_expected_outage_hours = outage.count * outage.duration

    if not outage.allow_outage_overlap or outage_percentage == 100:
        num_outage_hours = (scheduler.total_losses == outage_percentage).sum()
        assert num_outage_hours == num_expected_outage_hours
    else:
        num_outage_hours = (scheduler.total_losses >= outage_percentage).sum()
        assert num_outage_hours >= num_expected_outage_hours

    total_expected_outage = (outage.count * outage.duration
                             * outage.percentage_of_capacity_lost)

    assert scheduler.total_losses.sum() == total_expected_outage


def test_single_outage_scheduler_update_when_can_schedule_from_months(
    so_scheduler
):
    """Test that single outage is scheduled correctly."""

    so_scheduler.update_when_can_schedule_from_months()

    assert so_scheduler.can_schedule_more[:744].all()
    assert not so_scheduler.can_schedule_more[744:].any()


def test_single_outage_scheduler_update_when_can_schedule(so_scheduler):
    """Test that single outage is scheduled correctly."""

    so_scheduler.update_when_can_schedule_from_months()

    so_scheduler.scheduler.can_schedule_more[:10] = False
    so_scheduler.scheduler.total_losses[740:744] = 10
    so_scheduler.update_when_can_schedule()

    assert so_scheduler.can_schedule_more[10:740].all()
    assert not so_scheduler.can_schedule_more[0:10].any()
    assert not so_scheduler.can_schedule_more[740:].any()


def test_single_outage_scheduler_find_random_outage_slice(so_scheduler):
    """Test single outage class method."""

    so_scheduler.update_when_can_schedule_from_months()
    random_slice = so_scheduler.find_random_outage_slice()
    assert 0 <= random_slice.start < 744
    assert 0 < random_slice.stop <= 744

    slice_len = random_slice.stop - random_slice.start
    assert slice_len == so_scheduler.outage.duration


@pytest.mark.parametrize('allow_outage_overlap', [True, False])
def test_single_outage_scheduler_schedule_losses(
    allow_outage_overlap, so_scheduler
):
    """Test single outage class method."""

    so_scheduler.outage._specs['allow_outage_overlap'] = allow_outage_overlap
    so_scheduler.update_when_can_schedule_from_months()

    so_scheduler.schedule_losses(slice(0, 25))

    assert (so_scheduler.scheduler.total_losses[0:25] == 100).all()

    if not so_scheduler.outage.allow_outage_overlap:
        assert not (so_scheduler.scheduler.can_schedule_more[0:25]).any()


@pytest.mark.parametrize('outages_info', NOMINAL_OUTAGES)
def test_outage_scheduler_normal_run(outages_info):
    """Test hourly outage losses for a reasonable outage info input."""

    outages = [Outage(spec) for spec in outages_info]
    losses = OutageScheduler(outages).calculate()

    assert len(losses) == 8760
    assert losses[:744].any()
    assert not losses[744:].any()

    for outage in outages:
        outage_percentage = outage.percentage_of_capacity_lost
        num_expected_outage_hours = outage.count * outage.duration
        if not outage.allow_outage_overlap or outage_percentage == 100:
            num_outage_hours = (losses == outage_percentage).sum()
            assert num_outage_hours == num_expected_outage_hours
        else:
            num_outage_hours = (losses >= outage_percentage).sum()
            assert num_outage_hours >= num_expected_outage_hours

    total_expected_outage = sum(outage.count * outage.duration
                                * outage.percentage_of_capacity_lost
                                for outage in outages)
    assert losses.sum() == total_expected_outage


def test_outage_scheduler_no_outages():
    """Test hourly outage losses for no outage input."""

    losses = OutageScheduler([]).calculate()

    assert len(losses) == 8760
    assert not losses.any()


def test_outage_scheduler_cannot_schedule_any_more():
    """Test scheduler when little or no outages are allowed."""

    outage_info = {
        'count': 5,
        'duration': 10,
        'percentage_of_capacity_lost': 17,
        'allowed_months': ['January'],
        'allow_outage_overlap': False
    }
    losses = OutageScheduler([Outage(outage_info)])
    losses.can_schedule_more[:31 * 24] = False

    with pytest.warns(reVLossesWarning) as record:
        losses.calculate()
    warn_msg = record[0].message.args[0]
    assert "Could not schedule any requested outages" in warn_msg

    losses.can_schedule_more[100:130] = True
    with pytest.warns(reVLossesWarning) as record:
        losses.calculate()
    warn_msg = record[0].message.args[0]
    assert "Could only schedule" in warn_msg


def test_outage_class_missing_keys(basic_outage_dict):
    """Test Outage class behavior for inputs with missing keys."""

    for key in basic_outage_dict:
        bad_input = basic_outage_dict.copy()
        bad_input.pop(key)
        with pytest.raises(reVLossesValueError) as excinfo:
            Outage(bad_input)
        assert "The following required keys are missing" in str(excinfo.value)


def test_outage_class_count(basic_outage_dict):
    """Test Outage class behavior for different count inputs."""

    basic_outage_dict['count'] = 0
    with pytest.raises(reVLossesValueError) as excinfo:
        Outage(basic_outage_dict)
    assert "Number of outages must be greater than 0" in str(excinfo.value)

    basic_outage_dict['count'] = 5.5
    with pytest.raises(reVLossesValueError) as excinfo:
        Outage(basic_outage_dict)
    assert "Number of outages must be an integer" in str(excinfo.value)


def test_outage_class_allowed_months(basic_outage_dict):
    """Test Outage class behavior for different allowed_month inputs."""

    basic_outage_dict['allowed_months'] = []
    with pytest.raises(reVLossesValueError) as excinfo:
        Outage(basic_outage_dict)
    assert "No known month names were provided!" in str(excinfo.value)

    basic_outage_dict['allowed_months'] = ['Jan', 'unknown_month']
    with pytest.warns(reVLossesWarning) as record:
        outage = Outage(basic_outage_dict)
    warn_msg = record[0].message.args[0]
    assert "The following month names were not understood" in warn_msg
    assert outage.allowed_months == ['January']
    assert outage.total_available_hours == 31 * 24  # 31 days in Jan

    basic_outage_dict['allowed_months'] = ['mArcH', 'jan']
    outage = Outage(basic_outage_dict)
    assert 'January' in outage.allowed_months
    assert 'March' in outage.allowed_months
    assert outage.total_available_hours == (31 + 31) * 24

    basic_outage_dict['allowed_months'] = [
        'Jan', 'March', 'April  ', 'mAy', 'jun', 'July', 'October',
        'November', 'September', 'feb', 'December', 'August', 'May'
    ]
    outage = Outage(basic_outage_dict)
    assert len(outage.allowed_months) == 12
    assert outage.total_available_hours == 8760


def test_outage_class_duration(basic_outage_dict):
    """Test Outage class behavior for different duration inputs."""

    err_msg = "Duration of outage must be between 1 and the total available"

    basic_outage_dict['duration'] = 0
    with pytest.raises(reVLossesValueError) as excinfo:
        Outage(basic_outage_dict)
    assert err_msg in str(excinfo.value)

    basic_outage_dict['duration'] = 745
    with pytest.raises(reVLossesValueError) as excinfo:
        Outage(basic_outage_dict)
    assert err_msg in str(excinfo.value)

    basic_outage_dict['duration'] = 10.5
    with pytest.raises(reVLossesValueError) as excinfo:
        Outage(basic_outage_dict)
    assert "Duration must be an integer number of hours" in str(excinfo.value)

    basic_outage_dict['duration'] = 5
    assert Outage(basic_outage_dict).duration == 5


def test_outage_class_percentage(basic_outage_dict):
    """Test Outage class behavior for different percentage inputs."""

    err_msg = "Percentage of farm down during outage must be in the range"

    basic_outage_dict['percentage_of_capacity_lost'] = 0
    with pytest.raises(reVLossesValueError) as excinfo:
        Outage(basic_outage_dict)
    assert err_msg in str(excinfo.value)

    basic_outage_dict['percentage_of_capacity_lost'] = 100.1
    with pytest.raises(reVLossesValueError) as excinfo:
        Outage(basic_outage_dict)
    assert err_msg in str(excinfo.value)

    basic_outage_dict['percentage_of_capacity_lost'] = 100.0
    assert Outage(basic_outage_dict).percentage_of_capacity_lost == 100


def test_outage_class_allow_outage_overlap(basic_outage_dict):
    """
    Test Outage class behavior for different allow_outage_overlap inputs.
    """

    assert Outage(basic_outage_dict).allow_outage_overlap
    basic_outage_dict['allow_outage_overlap'] = True
    assert Outage(basic_outage_dict).allow_outage_overlap
    basic_outage_dict['allow_outage_overlap'] = False
    assert not Outage(basic_outage_dict).allow_outage_overlap


@pytest.mark.parametrize('files', [
    (WIND_SAM_FILE, TESTDATADIR + '/wtk/ri_100_wtk_{}.h5', 'windpower'),
    (PV_SAM_FILE, TESTDATADIR + '/nsrdb/ri_100_nsrdb_{}.h5', 'pvwattsv5'),
    (PV_SAM_FILE, TESTDATADIR + '/nsrdb/ri_100_nsrdb_{}.h5', 'pvwattsv7')
])
def test_scheduled_outages_multi_year(runner, files, clear_loggers):
    """Test that scheduled outages are different year to year."""
    sam_file, res_file, tech = files
    with open(sam_file) as fh:
        sam_config = json.load(fh)

    outages = NOMINAL_OUTAGES[0]
    sam_config['hourly'] = [0] * 8760
    sam_config[ScheduledLossesMixin.OUTAGE_CONFIG_KEY] = outages
    sam_config.pop('wind_farm_losses_percent', None)

    with tempfile.TemporaryDirectory() as td:
        sam_fp = os.path.join(td, 'gen.json')
        if 'pv' in tech:
            config_file_path = 'local_pv.json'
            project_points = os.path.join(TESTDATADIR, 'config',
                                          "project_points_10.csv")

            sam_config['losses'] = 0
            with open(sam_fp, 'w+') as fh:
                fh.write(json.dumps(sam_config))
            sam_files = {"sam gen pv_1": sam_fp}
        else:
            config_file_path = 'local_wind.json'
            project_points = os.path.join(TESTDATADIR, 'config',
                                          "wtk_pp_2012_10.csv")
            sam_config['turb_generic_loss'] = 0

            with open(sam_fp, 'w+') as fh:
                fh.write(json.dumps(sam_config))
            sam_files = {"wind0": sam_fp}

        config_file_path = 'config/{}'.format(config_file_path)
        config = os.path.join(TESTDATADIR, config_file_path).replace('\\', '/')
        config = safe_json_load(config)
        config['project_points'] = project_points
        config['resource_file'] = res_file
        config['sam_files'] = sam_files
        config['log_directory'] = td
        config['output_request'] = config['output_request'] + ['hourly']
        config['analysis_years'] = ['2012', '2013']

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['generation', '-c', config_path])
        clear_loggers()

        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        scheduled_outages = []
        for file in glob.glob(os.path.join(td, '*.h5')):
            with Outputs(file, 'r') as out:
                scheduled_outages.append(out['hourly'])

        # pylint: disable=unbalanced-tuple-unpacking
        outages_2012, outages_2013 = scheduled_outages
        for o1, o2 in zip(outages_2012.T, outages_2013.T):
            assert len(o1) == 8760
            assert len(o2) == 8760
            assert not np.allclose(o1, o2)


def test_outage_class_name(basic_outage_dict):
    """Test Outage class behavior for different name inputs."""

    expected_name = (
        "Outage(count=5, duration=24, percentage_of_capacity_lost=100, "
        "allowed_months=['January'], allow_outage_overlap=True)"
    )
    assert Outage(basic_outage_dict).name == expected_name
    basic_outage_dict['name'] = "My Outage"
    assert Outage(basic_outage_dict).name == "My Outage"


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
