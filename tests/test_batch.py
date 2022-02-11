# -*- coding: utf-8 -*-
"""
PyTest file for batch jobs.

Created on Sep 8, 2020

@author: gbuster
"""
import json
import pandas as pd
from pandas.testing import assert_frame_equal
import os
import pytest
import warnings

from reV.config.batch import BatchCsv
from reV.batch.batch import BatchJob
from reV import TESTDATADIR
from reV.utilities.exceptions import InputError

from rex.utilities import safe_json_load, safe_yaml_load

BATCH_DIR_0 = os.path.join(TESTDATADIR, 'batch_project_0/')
FP_CONFIG_0 = os.path.join(BATCH_DIR_0, 'config_batch.json')

BATCH_DIR_1 = os.path.join(TESTDATADIR, 'batch_project_1/')
FP_CONFIG_1 = os.path.join(BATCH_DIR_1, 'config_batch.csv')

BATCH_DIR_2 = os.path.join(TESTDATADIR, 'batch_project_2/')
FP_CONFIG_2 = os.path.join(BATCH_DIR_2, 'config_batch.json')


@pytest.fixture()
def save_test_dir():
    """Return to the starting dir after running a test.

    In particular, persisting the batch dir change that happens during
    a BatchJob run can mess up downstream pytests.
    """
    # Startup
    previous_dir = os.getcwd()

    # test happens here
    yield

    # teardown (i.e. return to original dir)
    os.chdir(previous_dir)


# pylint: disableW0613
def test_batch_job_setup_with_yaml_files_no_sort(save_test_dir):
    """Test the creation and deletion of a batch job directory with yaml files,
    and ensure that the output yaml files are NOT sorted."""

    count_0 = len(os.listdir(BATCH_DIR_2))
    assert count_0 == 7, 'Unknown starting files detected!'

    BatchJob.run(FP_CONFIG_2, dry_run=True)
    job_dir = os.path.join(BATCH_DIR_2, 'set1_ic10_ic31/')
    with open(os.path.join(job_dir, 'test.yaml'), 'r') as fh:
        key_order = [line.split(':')[0] for line in fh]

    correct_key_order = ['input_constant_1', 'input_constant_2',
                         'another_input_constant', 'some_equation']
    e_msg = "Output YAML file does not have correct key order!"
    assert key_order == correct_key_order, e_msg

    BatchJob.run(FP_CONFIG_2, delete=True)
    count_1 = len(os.listdir(BATCH_DIR_2))
    assert count_1 == count_0, 'Batch did not clear all job files!'


# pylint: disableW0613
def test_batch_job_setup_with_yaml_files(save_test_dir):
    """Test the creation and deletion of a batch job directory with yaml files.
    Does not test batch execution which will require slurm."""

    config = safe_json_load(FP_CONFIG_2)

    count_0 = len(os.listdir(BATCH_DIR_2))
    assert count_0 == 7, 'Unknown starting files detected!'

    BatchJob.run(FP_CONFIG_2, dry_run=True)

    dir_list = os.listdir(BATCH_DIR_2)
    set1_count = len([fn for fn in dir_list if fn.startswith('set1_')])
    set2_count = len([fn for fn in dir_list if fn.startswith('set2_')])
    assert set1_count == 6
    assert set2_count == 18

    assert 'set1_ic10_ic30' in dir_list
    assert 'set1_ic11_ic31' in dir_list
    assert 'set1_ic12_ic31' in dir_list
    assert 'set2_ic218020_se0_se20' in dir_list
    assert 'set2_ic218020_se1_se21' in dir_list
    assert 'set2_ic219040_se2_se20' in dir_list
    assert 'set2_ic219040_se2_se22' in dir_list
    assert 'batch_jobs.csv' in dir_list

    args = config['sets'][0]['args']
    job_dir = os.path.join(BATCH_DIR_2, 'set1_ic10_ic31/')
    test_yaml = safe_yaml_load(os.path.join(job_dir, 'test.yaml'))
    test_yml = safe_yaml_load(os.path.join(job_dir, 'test.yml'))
    assert test_yaml['input_constant_1'] == args['input_constant_1'][0]
    assert test_yaml['input_constant_2'] == args['input_constant_2'][0]
    assert test_yml['input_constant_3'] == args['input_constant_3'][1]

    args = config['sets'][1]['args']
    job_dir = os.path.join(BATCH_DIR_2, 'set2_ic219040_se1_se21/')
    test_yaml = safe_yaml_load(os.path.join(job_dir, 'test.yaml'))
    test_yml = safe_yaml_load(os.path.join(job_dir, 'test.yml'))
    assert test_yaml['input_constant_2'] == args['input_constant_2'][1]
    assert test_yaml['some_equation'] == args['some_equation'][1]
    assert test_yml['some_equation_2'] == args['some_equation_2'][1]

    count_1 = len(os.listdir(BATCH_DIR_2))
    assert count_1 == 32, 'Batch generated unexpected files or directories!'

    BatchJob.run(FP_CONFIG_2, delete=True)
    count_2 = len(os.listdir(BATCH_DIR_2))
    assert count_2 == count_0, 'Batch did not clear all job files!'


# pylint: disableW0613
def test_invalid_mod_file_input(save_test_dir):
    """Test that error is raised for unknown file input type. """

    bad_config_file = os.path.join(BATCH_DIR_2, 'config_batch_bad_fpath.json')
    with pytest.raises(InputError) as excinfo:
        BatchJob.run(bad_config_file, dry_run=True)

    assert "Unknown" in str(excinfo.value)
    assert "type: 'test.yamlet'" in str(excinfo.value)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        BatchJob.run(bad_config_file, delete=True)


# pylint: disableW0613
def test_batch_job_setup(save_test_dir):
    """Test the creation and deletion of a batch job directory.
    Does not test batch execution which will require slurm."""

    config = safe_json_load(FP_CONFIG_0)

    count_0 = len(os.listdir(BATCH_DIR_0))
    assert count_0 == 8, 'Unknown starting files detected!'

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
    config_agg = safe_json_load(
        os.path.join(job_dir, 'config_aggregation.json'))
    turbine = safe_json_load(os.path.join(job_dir, 'sam_configs/turbine.json'))
    assert config_gen['project_points'] == args['project_points'][0]
    assert config_gen['resource_file'] == args['resource_file'][0]
    assert config_col['project_points'] == args['project_points'][0]
    assert isinstance(config_agg['data_layers']['big_brown_bat'], dict)
    assert (config_agg['data_layers']['big_brown_bat']
            == json.loads(args['big_brown_bat'][0].replace("'", '"')))
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
    assert count_1 == 18, 'Batch generated unexpected files or directories!'

    BatchJob.run(FP_CONFIG_0, delete=True)
    count_2 = len(os.listdir(BATCH_DIR_0))
    assert count_2 == count_0, 'Batch did not clear all job files!'


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


# pylint: disableW0613
def test_batch_csv_setup(save_test_dir):
    """Test a batch project setup from csv config"""

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

    # test that the dict was input properly
    fp_agg = os.path.join(BATCH_DIR_1, 'blanket_cf0_sd0/',
                          'config_aggregation.json')
    with open(fp_agg, 'r') as f:
        config_agg = json.load(f)
    arg = config_agg['data_layers']['big_brown_bat']
    assert isinstance(arg, dict)
    assert arg['dset'] == 'big_brown_bat'
    assert arg['method'] == 'sum'

    BatchJob.run(FP_CONFIG_1, delete=True)
    count_2 = len(os.listdir(BATCH_DIR_1))
    assert count_2 == count_0, 'Batch did not clear all job files!'


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
