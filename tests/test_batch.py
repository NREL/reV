# -*- coding: utf-8 -*-
"""
PyTest file for batch jobs.

Created on Sep 8, 2020

@author: gbuster
"""
import pandas as pd
from pandas.testing import assert_frame_equal
import os
import pytest

from reV.config.batch import BatchCsv
from reV.batch.batch import BatchJob
from reV import TESTDATADIR

from rex.utilities import safe_json_load

BATCH_DIR_0 = os.path.join(TESTDATADIR, 'batch_project_0/')
FP_CONFIG_0 = os.path.join(BATCH_DIR_0, 'config_batch.json')

BATCH_DIR_1 = os.path.join(TESTDATADIR, 'batch_project_1/')
FP_CONFIG_1 = os.path.join(BATCH_DIR_1, 'config_batch.csv')


def test_batch_job_setup():
    """Test the creation and deletion of a batch job directory.
    Does not test batch execution which will require slurm."""

    # persisting the batch dir change can mess up downstream pytests.
    previous_dir = os.getcwd()

    config = safe_json_load(FP_CONFIG_0)

    count_0 = len(os.listdir(BATCH_DIR_0))

    assert count_0 == 7, 'Unknown starting files detected!'

    BatchJob.run(FP_CONFIG_0, dry_run=True)

    dir_list = os.listdir(BATCH_DIR_0)
    set1_count = len([fn for fn in dir_list if fn.startswith('set1_')])
    set2_count = len([fn for fn in dir_list if fn.startswith('set2_')])
    assert set1_count == 6
    assert set2_count == 3

    assert 'set1_wthh80_wtpp0' in dir_list
    assert 'set1_wthh110_wtpp1' in dir_list
    assert 'set1_wthh140_wtpp1' in dir_list
    assert 'set2_wthh80' in dir_list
    assert 'set2_wthh110' in dir_list
    assert 'batch_jobs.csv' in dir_list

    args = config['sets'][0]['args']
    job_dir = os.path.join(BATCH_DIR_0, 'set1_wthh140_wtpp1/')
    config_gen = safe_json_load(os.path.join(job_dir, 'config_gen.json'))
    config_col = safe_json_load(os.path.join(job_dir, 'config_collect.json'))
    turbine_base = safe_json_load(os.path.join(
        BATCH_DIR_0, 'sam_configs/turbine.json'))
    turbine = safe_json_load(os.path.join(
        job_dir, 'sam_configs/turbine.json'))
    assert config_gen['project_points'] == args['project_points'][0]
    assert config_col['project_points'] == args['project_points'][0]
    assert turbine['wind_turbine_hub_ht'] == args['wind_turbine_hub_ht'][2]
    assert (turbine['wind_turbine_powercurve_powerout']
            == args['wind_turbine_powercurve_powerout'][1])
    assert (turbine['wind_resource_shear']
            == turbine_base['wind_resource_shear'])
    assert (turbine['wind_resource_turbulence_coeff']
            == turbine_base['wind_resource_turbulence_coeff'])
    assert (turbine['wind_turbine_rotor_diameter']
            == turbine_base['wind_turbine_rotor_diameter'])

    args = config['sets'][1]['args']
    job_dir = os.path.join(BATCH_DIR_0, 'set2_wthh140/')
    config_gen = safe_json_load(os.path.join(job_dir, 'config_gen.json'))
    config_col = safe_json_load(os.path.join(job_dir, 'config_collect.json'))
    turbine = safe_json_load(os.path.join(job_dir, 'sam_configs/turbine.json'))
    assert config_gen['project_points'] == args['project_points'][0]
    assert config_col['project_points'] == args['project_points'][0]
    assert turbine['wind_turbine_hub_ht'] == args['wind_turbine_hub_ht'][2]
    assert (turbine['wind_turbine_powercurve_powerout']
            == turbine_base['wind_turbine_powercurve_powerout'])
    assert (turbine['wind_resource_shear']
            == turbine_base['wind_resource_shear'])
    assert (turbine['wind_resource_turbulence_coeff']
            == turbine_base['wind_resource_turbulence_coeff'])
    assert (turbine['wind_turbine_rotor_diameter']
            == turbine_base['wind_turbine_rotor_diameter'])

    count_1 = len(os.listdir(BATCH_DIR_0))
    assert count_1 == 17, 'Batch generated unexpected files or directories!'

    BatchJob.run(FP_CONFIG_0, delete=True)
    count_2 = len(os.listdir(BATCH_DIR_0))
    assert count_2 == count_0, 'Batch did not clear all job files!'

    os.chdir(previous_dir)


def test_batch_csv_config():
    """Test the batch job csv parser."""
    table = pd.read_csv(FP_CONFIG_1, index_col=0)
    c = BatchCsv(FP_CONFIG_1)
    assert 'logging' in c
    assert 'pipeline_config' in c
    assert 'sets' in c
    sets = c['sets']
    assert len(sets) == len(table)
    for _, row in table.iterrows():
        row = row.to_dict()
        set_tag = row['set_tag']
        found = False
        for job_set in sets:
            if job_set['set_tag'] == set_tag:
                found = True
                for k, v in row.items():
                    if k not in ('set_tag', 'files'):
                        assert [v] == job_set['args'][k]
                break

        assert found


def test_batch_csv_setup():
    """Test a batch project setup from csv config"""

    previous_dir = os.getcwd()
    config_table = pd.read_csv(FP_CONFIG_1, index_col=0)
    count_0 = len(os.listdir(BATCH_DIR_1))
    assert count_0 == 5, 'Unknown starting files detected!'

    BatchJob.run(FP_CONFIG_1, dry_run=True)

    dirs = os.listdir(BATCH_DIR_1)
    count_1 = len(dirs)
    assert (count_1 - count_0) == len(config_table) + 1
    for job in config_table.index.values:
        assert job in dirs

    job_table = pd.read_csv(os.path.join(BATCH_DIR_1, 'batch_jobs.csv'),
                            index_col=0)
    assert_frame_equal(config_table, job_table)

    BatchJob.run(FP_CONFIG_1, delete=True)
    count_2 = len(os.listdir(BATCH_DIR_1))
    assert count_2 == count_0, 'Batch did not clear all job files!'

    os.chdir(previous_dir)


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
