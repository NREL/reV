#!/usr/bin/env python
# pylint: skip-file
"""reV unit test modul
"""
import json
import os
import pytest

import utilities as ut
from reV.SAM import SAM
from reV.utilities.loggers import init_logger


class SAMTestManager:
    """SAM unit test manager."""
    def __init__(self):
        """Initialize a SAM unit test manager."""
        self._logger = init_logger("reV.SAM")

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
            self._logger.warning('Inputs file does not exist: {}'
                                 .format(i_fname))

        if module == 'pvwatts':
            # test SAM pvwatts module
            sim = SAM.PV(resource=None, meta=None, parameters=inputs,
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

        self._logger.debug('{} results: {}'.format(module, sim.outputs))
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
            self._logger.warning('Inputs file does not exist: {}'
                                 .format(i_fname))

        if module == 'pvwatts':
            outputs = SAM.PV.reV_run(res_f, sites, inputs)
        elif module == 'tcsmolten_salt':
            outputs = SAM.CSP.reV_run(res_f, sites, inputs)
        elif module == 'windpower':
            outputs = SAM.LandBasedWind.reV_run(res_f, sites, inputs)
        elif module == 'offshore':
            outputs = SAM.OffshoreWind.reV_run(res_f, sites, inputs)

        test = self.check_test_results(outputs, o_fname, module)

        self._logger.debug('{} results: {}'.format(module, outputs))

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
        new_o_json = ut.jsonify(new_outputs)
        if os.path.exists(baseline_fname):
            with open(baseline_fname) as f:
                # get previous baseline outputs for checking
                baseline = json.load(f)

            # check new outputs against the baseline
            match, items = ut.dicts_match(new_o_json, baseline)
            if match is True:
                self._logger.info('Unit test for {} was successful.'
                                  .format(module))
            else:
                self._logger.error('Unit test for {} failed with errors in '
                                   'the following variables: "{}"'
                                   .format(module, items))
            return match

        else:
            self._logger.warning('(STRONG!!!) '
                                 'Previous baseline outputs for {} do not'
                                 ' exist. Looked for outputs in the following '
                                 'file: {}'.format(module, baseline_fname))
            self._logger.warning('Writing new baseline output file: {}'
                                 .format(baseline_fname))
            with open(baseline_fname, 'w+') as f:
                json.dump(new_o_json, f, sort_keys=True,
                          indent=4, separators=(',', ': '))
            return True


@pytest.fixture
def init_SAM():
    """Return a SAM test manager instance."""
    return SAMTestManager()


@pytest.mark.parametrize('module, i_fname, o_fname', [
    ('pvwatts', 'i_pvwatts.json', 'o_pvwatts.json'),
    ('pvwatts_lcoe', 'i_pvwatts_lcoe.json', 'o_pvwatts_lcoe.json'),
    ('pvwatts', 'i_pvwatts_def.json', 'o_pvwatts_def.json'),
    ('tcsmolten_salt', 'i_csp_tcsmolten_salt.json',
     'o_csp_tcsmolten_salt.json'),
    ('landbasedwind', 'i_windpower.json', 'o_windpower.json'),
    ('landbasedwind_lcoe', 'i_windpower_lcoe.json', 'o_windpower_lcoe.json')])
def test_SAM_defaults(init_SAM, module, i_fname, o_fname):
    """Test the SAM simulation module."""
    result = init_SAM.execute_defaults(module=module, i_fname=i_fname,
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

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
