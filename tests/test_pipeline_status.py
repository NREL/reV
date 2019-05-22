# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import pytest
import json

from reV.pipeline.status import Status
from reV import TESTDATADIR

RTOL = 0.0
ATOL = 0.04
PURGE_OUT = True

STATUS_DIR = os.path.join(TESTDATADIR, 'status/')
if not os.path.exists(STATUS_DIR):
    os.makedirs(STATUS_DIR)

TEST_1_ATTRS_1 = {'job_name': 'test1', 'job_status': 'R', 'run_id': 1234}
TEST_1_ATTRS_2 = {'job_name': 'test1', 'job_status': 'successful'}
TEST_2_ATTRS_1 = {'job_name': 'test2', 'job_status': 'R'}


def purge():
    """Purge files in the status dir"""
    if PURGE_OUT:
        for fname in os.listdir(STATUS_DIR):
            os.remove(os.path.join(STATUS_DIR, fname))


def test_recursive_update():
    """Test a recursive merge of two status dictionaries"""

    test = Status.update_dict({'generation': TEST_1_ATTRS_1},
                              {'generation': TEST_1_ATTRS_2})

    assert test['generation']['run_id'] == TEST_1_ATTRS_1['run_id']
    assert test['generation']['job_status'] == TEST_1_ATTRS_2['job_status']


def test_file_collection():
    """Test file creation and collection"""
    purge()

    Status.make_job_file(STATUS_DIR, 'generation', 'test1', TEST_1_ATTRS_1)
    Status.make_job_file(STATUS_DIR, 'generation', 'test2', TEST_2_ATTRS_1)

    Status.update(STATUS_DIR)
    with open(os.path.join(STATUS_DIR, 'rev_status.json'), 'r') as f:
        data = json.load(f)
    assert str(TEST_1_ATTRS_1) in str(data)
    assert str(TEST_2_ATTRS_1) in str(data)
    purge()


def test_make_file():
    """Test file creation and reading"""
    purge()
    Status.make_job_file(STATUS_DIR, 'generation', 'test1', TEST_1_ATTRS_1)
    status = Status.retrieve_job_status(STATUS_DIR, 'generation', 'test1')
    msg = 'Failed, status is "{}"'.format(status)
    assert status == 'R', msg
    purge()


def test_job_exists():
    """Test job addition and exist check"""
    purge()
    Status.add_job(STATUS_DIR, 'generation', 'test1',
                   job_attrs={'job_status': 'submitted'})
    exists = Status.job_exists(STATUS_DIR, 'test1')
    assert exists
    purge()


def test_job_addition():
    """Test job addition and exist check"""
    purge()
    Status.add_job(STATUS_DIR, 'generation', 'test1')
    status1 = Status(STATUS_DIR).data['generation']['test1']['job_status']

    Status.add_job(STATUS_DIR, 'generation', 'test1',
                   job_attrs={'job_status': 'finished', 'additional': 'test'})
    status2 = Status(STATUS_DIR).data['generation']['test1']['job_status']

    assert status2 == status1
    purge()


def test_job_replacement():
    """Test job addition and replacement"""
    purge()
    Status.add_job(STATUS_DIR, 'generation', 'test1',
                   job_attrs={'job_status': 'submitted'})

    Status.add_job(STATUS_DIR, 'generation', 'test1',
                   job_attrs={'addition': 'test', 'job_status': 'finished'},
                   replace=True)

    status = Status(STATUS_DIR).data['generation']['test1']['job_status']
    addition = Status(STATUS_DIR).data['generation']['test1']['addition']
    assert status == 'finished'
    assert addition == 'test'
    purge()


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
