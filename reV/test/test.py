#!/usr/bin/env python
"""reV unit test module
"""
from copy import deepcopy
import json
import logging
import numpy as np
import os
import pytest

from reV.SAM import SAM
from reV.rev_logger import setup_logger


def jsonify(outputs):
    """Convert outputs dictionary to JSON compatitble format."""
    for key in outputs.keys():
        if isinstance(outputs[key], np.ndarray):
            outputs[key] = outputs[key].tolist()
    return outputs


def dicts_match(x, y):
    """Check whether two dictionaries match."""
    if len(list(x.keys())) == len(list(y.keys())):
        shared_items = {k: x[k] for k in x if k in y and x[k] == y[k]}
        if len(shared_items) == len(list(x.keys())):
            # everything matches
            return True
        else:
            # values in keys do not match
            bad_items = {k: x[k] for k in x if k in y and x[k] != y[k]}
            return list(bad_items.keys())

    else:
        # keys are missing
        x = set(x.keys())
        y = set(y.keys())
        return list(x.symmetric_difference(y))


class SAM_Test_Manager():
    """SAM unit test manager."""
    def __init__(self):
        """Initialize a SAM unit test manager."""
        logger = logging.getLogger("reV.SAM")
        self.logger = logging.getLogger(self.__class__.__name__)

        _, handler = setup_logger(__name__)

        if not logger.handlers:
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def execute_defaults(self, io_dir='./data/SAM', module='pvwatts',
                         i_fname='i_pvwatts.json', o_fname='o_pvwatts.json'):
        """Execute a test case with SAM default inputs."""
        i_fname = os.path.join(io_dir, str(i_fname))
        o_fname = os.path.join(io_dir, o_fname)

        if os.path.exists(i_fname):
            with open(i_fname, 'r') as f:
                # get unit test inputs
                inputs = json.load(f)
        else:
            self.logger.warning('Inputs file does not exist: {}'
                                ''.format(i_fname))

        if module == 'pvwatts':
            # test SAM pvwatts module
            sim = SAM.PV(resource=None, meta=None, parameters=inputs,
                         output_request=['cf_mean', 'cf_profile',
                                         'annual_energy', 'energy_yield',
                                         'gen_profile'])
        if module == 'pvwatts_def':
            # test SAM pvwatts module with all defaults
            sim = SAM.PV(resource=None, meta=None, parameters=None,
                         output_request=['cf_mean', 'cf_profile',
                                         'annual_energy', 'energy_yield',
                                         'gen_profile'])
        elif module == 'pvwatts_lcoe':
            # test SAM pvwatts module with LCOE
            sim = SAM.PV(resource=None, meta=None, parameters=inputs,
                         output_request=['cf_mean', 'cf_profile',
                                         'annual_energy', 'energy_yield',
                                         'gen_profile', 'lcoe_fcr'])
        elif module == 'tcsmolten_salt':
            # test SAM tcs molten salt module with single owner
            sim = SAM.CSP(resource=None, meta=None, parameters=inputs,
                          output_request=['cf_mean', 'cf_profile',
                                          'annual_energy', 'energy_yield',
                                          'gen_profile', 'ppa_price'])

        sim.execute()
        test = self.check_test_results(sim.outputs, o_fname,
                                       module)

        self.logger.debug('{} results: {}'.format(module, sim.outputs))
        return test

    def execute_nsrdb(self, module='pvwatts', site=0, res_dir='./data/nsrdb',
                      res='ri_100_nsrdb_2012.h5', io_dir='./data/SAM',
                      i_fname='i_pvwatts_res.json',
                      o_fname='o_pvwatts_res.json'):
        """Execute a test case with SAM using NSRDB resource inputs."""

        res_f = os.path.join(res_dir, res)
        i_fname = os.path.join(io_dir, str(i_fname))
        o_fname = os.path.join(io_dir, o_fname)

        if os.path.exists(i_fname):
            with open(i_fname, 'r') as f:
                # get unit test inputs
                inputs = json.load(f)
        else:
            self.logger.warning('Inputs file does not exist: {}'
                                ''.format(i_fname))

        res, meta = SAM.SAM.setup_resource_df(res_f, site, ['dni', 'dhi',
                                                            'wind_speed',
                                                            'air_temperature'])
        if module == 'pvwatts':
            sim = SAM.PV(resource=res, meta=meta,
                         parameters=inputs,
                         output_request=['cf_mean', 'cf_profile',
                                         'annual_energy', 'energy_yield',
                                         'gen_profile'])
        if module == 'pvwatts_lcoe':
            sim = SAM.PV(resource=res, meta=meta,
                         parameters=inputs,
                         output_request=['cf_mean', 'cf_profile',
                                         'annual_energy', 'energy_yield',
                                         'gen_profile', 'lcoe_fcr'])
        elif module == 'tcsmolten_salt':
            sim = SAM.CSP(resource=res, meta=meta,
                          parameters=inputs,
                          output_request=['cf_mean', 'cf_profile',
                                          'annual_energy', 'energy_yield',
                                          'gen_profile', 'ppa_price'])

        sim.execute()
        test = self.check_test_results(sim.outputs, o_fname, module)

        self.logger.debug('{} results: {}'.format(module, sim.outputs))
        return test

    def execute_reV(self, module='pvwatts', site=0, res_dir='./data/nsrdb',
                    res='ri_100_nsrdb_2012.h5', io_dir='./data/SAM',
                    i_fname='i_pvwatts_reV.json',
                    o_fname='o_pvwatts_reV.json'):
        """Execute a test case with SAM using reV defaults."""

        res_f = os.path.join(res_dir, res)
        i_fname = os.path.join(io_dir, str(i_fname))
        o_fname = os.path.join(io_dir, o_fname)

        if os.path.exists(i_fname):
            with open(i_fname, 'r') as f:
                # get unit test inputs
                inputs = json.load(f)
        else:
            self.logger.warning('Inputs file does not exist: {}'
                                ''.format(i_fname))

        if module == 'pvwatts':
            outputs = SAM.PV.reV_run(site, res_f, inputs)
        elif module == 'tcsmolten_salt':
            outputs = SAM.CSP.reV_run(site, res_f, inputs)

        test = self.check_test_results(outputs, o_fname, module)

        self.logger.debug('{} results: {}'.format(module, outputs))

        return test

    def check_test_results(self, new_outputs, baseline_fname, module):
        """Check the test results against a baseline set of results.

        Returns
        -------
        match : bool
            Whether or not the new outputs match the json in the baseline
            outputs file.

            Possible scenarios:
                - Dictionaries match perfectly (returns True)
                - Output dictionary does not exist (returns True,
                  prints warnings, and creates a new baseline outputs file)
                - Output dictionary does not match (returns False, notes which
                  dictionary keys have issues)
        """
        new_o_json = jsonify(deepcopy(new_outputs))
        if os.path.exists(baseline_fname):
            with open(baseline_fname) as f:
                # get previous baseline outputs for checking
                baseline = json.load(f)

            # check new outputs against the baseline
            match = dicts_match(new_o_json, baseline)
            if match is True:
                self.logger.info('Unit test for {} was successful.'
                                 ''.format(module))
            else:
                self.logger.error('Unit test for {} failed with errors in '
                                  'the following variables: '
                                  '"{}"'.format(module, match))
            return match

        else:
            self.logger.warning('(STRONG!!!) '
                                'Previous baseline outputs for {} do not'
                                ' exist. Looked for outputs in the following '
                                'file: {}'.format(module, baseline_fname))
            self.logger.warning('Writing new baseline output file: {}'
                                ''.format(baseline_fname))
            with open(baseline_fname, 'w+') as f:
                json.dump(new_o_json, f, sort_keys=True,
                          indent=4, separators=(',', ': '))
            return True


@pytest.fixture
def init_SAM():
    """Return a SAM test manager instance."""
    return SAM_Test_Manager()


@pytest.mark.parametrize('module, i_fname, o_fname', [
    ('pvwatts', 'i_pvwatts.json', 'o_pvwatts.json'),
    ('pvwatts_lcoe', 'i_pvwatts_lcoe.json', 'o_pvwatts_lcoe.json'),
    ('pvwatts_def', None, 'o_pvwatts_def.json'),
    ('tcsmolten_salt', 'i_csp_tcsmolten_salt.json',
     'o_csp_tcsmolten_salt.json')])
def test_SAM_defaults(init_SAM, module, i_fname, o_fname):
    """Test the SAM simulation module."""
    result = init_SAM.execute_defaults(module=module, i_fname=i_fname,
                                       o_fname=o_fname)
    assert result is True


@pytest.mark.parametrize('module, res_dir, res, io_dir, i_fname, o_fname', [
    ('pvwatts', './data/nsrdb', 'ri_100_nsrdb_2012.h5', './data/SAM',
     'i_pvwatts_res.json', 'o_pvwatts_res.json'),
    ('pvwatts_lcoe', './data/nsrdb', 'ri_100_nsrdb_2012.h5', './data/SAM',
     'i_pvwatts_lcoe_res.json', 'o_pvwatts_lcoe_res.json'),
    ('tcsmolten_salt', './data/nsrdb', 'ri_100_nsrdb_2012.h5', './data/SAM',
     'i_csp_tcsmolten_salt_res.json', 'o_csp_tcsmolten_salt_res.json')])
def test_SAM_NSRDB(init_SAM, module, res_dir, res, io_dir, i_fname, o_fname):
    """Simple SAM pytest for NSRDB."""
    result = init_SAM.execute_nsrdb(module=module, res_dir=res_dir, res=res,
                                    io_dir=io_dir, i_fname=i_fname,
                                    o_fname=o_fname)
    assert result is True


@pytest.mark.parametrize('module, res_dir, res, io_dir, i_fname, o_fname', [
    ('pvwatts', './data/nsrdb', 'ri_100_nsrdb_2012.h5', './data/SAM',
     'i_pvwatts_reV.json', 'o_pvwatts_reV.json'),
    ('tcsmolten_salt', './data/nsrdb', 'ri_100_nsrdb_2012.h5', './data/SAM',
     'i_csp_tcsmolten_salt_reV.json', 'o_csp_tcsmolten_salt_reV.json')])
def test_SAM_reV(init_SAM, module, res_dir, res, io_dir, i_fname, o_fname):
    """Simple SAM pytest for reV default runs."""
    result = init_SAM.execute_reV(module=module, res_dir=res_dir, res=res,
                                  io_dir=io_dir, i_fname=i_fname,
                                  o_fname=o_fname)
    assert result is True


if __name__ == '__main__':
    """Execute module as pytest with detailed summary report.

    Options
    -------
    --show-capture=
        log (only logger)
        all (includes stdout/stderr)
    """
    pytest.main(['-q', '--show-capture=log', 'test.py', '-rapP'])
