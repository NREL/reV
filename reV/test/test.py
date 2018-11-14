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
    orig_key_list = list(outputs.keys())
    for key in orig_key_list:
        if isinstance(outputs[key], np.ndarray):
            outputs[key] = outputs[key].tolist()
        if isinstance(key, (int, float)):
            outputs[str(key)] = outputs[key]
            del outputs[key]
    return outputs


def get_shared_items(x, y):
    """Get a dict of shared values between the two input dicts."""
    shared_items = {}
    for k, v in x.items():
        if k in y:
            if isinstance(v, dict) and isinstance(y[k], dict):
                # recursion! go one level deeper.
                shared_items_2 = get_shared_items(v, y[k])
                if shared_items_2:
                    shared_items[k] = v
            elif x[k] == y[k]:
                shared_items[k] = v
    return shared_items


def dicts_match(x, y):
    """Check whether two dictionaries match."""
    if len(list(x.keys())) == len(list(y.keys())):
        # dicts have the same number of keys (good sign)
        shared_items = get_shared_items(x, y)
        if len(shared_items) == len(list(x.keys())):
            # everything matches
            return True, list(shared_items.keys())
        else:
            # values in keys do not match
            bad_items = {k: x[k] for k in x if k in y and x[k] != y[k]}
            return False, list(bad_items.keys())

    else:
        # keys are missing
        x = set(x.keys())
        y = set(y.keys())
        return False, list(x.symmetric_difference(y))


class SAM_Test_Manager:
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
        elif module == 'landbasedwind':
            # test SAM windpower module
            sim = SAM.LandBasedWind(resource=None, meta=None,
                                    parameters=inputs,
                                    output_request=['cf_mean', 'cf_profile',
                                                    'annual_energy',
                                                    'energy_yield',
                                                    'gen_profile'])
        elif module == 'landbasedwind_lcoe':
            # test SAM windpower module
            sim = SAM.LandBasedWind(resource=None, meta=None,
                                    parameters=inputs,
                                    output_request=['cf_mean', 'cf_profile',
                                                    'annual_energy',
                                                    'energy_yield',
                                                    'gen_profile',
                                                    'lcoe_fcr'])

        sim.execute(sim.MODULE)
        test = self.check_test_results(sim.outputs, o_fname,
                                       module)

        self.logger.debug('{} results: {}'.format(module, sim.outputs))
        return test

    def execute_reV(self, module='pvwatts',
                    sites=range(0, 2),
                    res_dir='./data/nsrdb',
                    res='ri_100_nsrdb_2012.h5',
                    io_dir='./data/SAM',
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
            outputs = SAM.PV.reV_run(res_f, sites, inputs)
        elif module == 'tcsmolten_salt':
            outputs = SAM.CSP.reV_run(res_f, sites, inputs)
        elif module == 'windpower':
            outputs = SAM.LandBasedWind.reV_run(res_f, sites, inputs)
        elif module == 'offshore':
            outputs = SAM.OffshoreWind.reV_run(res_f, sites, inputs)

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
            match, items = dicts_match(new_o_json, baseline)
            if match is True:
                self.logger.info('Unit test for {} was successful.'
                                 ''.format(module))
            else:
                self.logger.error('Unit test for {} failed with errors in '
                                  'the following variables: '
                                  '"{}"'.format(module, items))
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
     'o_csp_tcsmolten_salt.json'),
    ('landbasedwind', 'i_windpower.json', 'o_windpower.json'),
    ('landbasedwind_lcoe', 'i_windpower_lcoe.json', 'o_windpower_lcoe.json')])
def test_SAM_defaults(init_SAM, module, i_fname, o_fname):
    """Test the SAM simulation module."""
    result = init_SAM.execute_defaults(module=module, i_fname=i_fname,
                                       o_fname=o_fname)
    assert result is True


@pytest.mark.parametrize(('module, sites, res_dir, res, io_dir, i_fname, '
                         'o_fname'), [
    ('pvwatts', slice(0, 10), './data/nsrdb', 'ri_100_nsrdb_2012.h5',
     './data/SAM', 'i_pvwatts_reV.json', 'o_pvwatts_reV.json'),
    ('tcsmolten_salt', 0, './data/nsrdb', 'ri_100_nsrdb_2012.h5', './data/SAM',
     'i_csp_tcsmolten_salt_reV.json', 'o_csp_tcsmolten_salt_reV.json'),
    ('windpower', range(0, 10, 2), './data/wtk', 'ri_100_wtk_2012.h5',
     './data/SAM', 'i_windpower_reV.json', 'o_windpower_reV.json'),
    ('offshore', [150, 170, 192], './data/wtk', 'ri_100_wtk_2012.h5',
     './data/SAM', 'i_offshore_reV.json', 'o_offshore_reV.json')])
def test_SAM_reV(init_SAM, module, sites, res_dir, res, io_dir, i_fname,
                 o_fname):
    """Simple SAM pytest for reV default runs."""
    result = init_SAM.execute_reV(module=module, sites=sites, res_dir=res_dir,
                                  res=res, io_dir=io_dir, i_fname=i_fname,
                                  o_fname=o_fname)
    assert result is True


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

    pytest.main(['-q', '--show-capture={}'.format(capture), 'test.py', flags])


if __name__ == '__main__':
    execute_pytest()
