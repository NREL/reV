# -*- coding: utf-8 -*-
"""
PyTest file for Wind generic and icing losses.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import json
import tempfile

import numpy as np
import pytest

from reV import TESTDATADIR
from reV.generation.generation import Gen

YEAR = 2012
REV2_POINTS = slice(0, 5)
SAM_FILE = TESTDATADIR + '/SAM/wind_gen_standard_losses_0.json'
RES_FILE = TESTDATADIR + '/wtk/ri_100_wtk_{}.h5'.format(YEAR)
LAYOUT_SAM_FILE = TESTDATADIR + '/nrwal/offshore.json'
RTOL = 0
ATOL = 0.001

LOSS_BASELINE = {0.0: [0.422, 0.429, 0.437, 0.431, 0.431],
                 16.7: [0.352, 0.357, 0.364, 0.359, 0.359],
                 30.0: [0.296, 0.3, 0.306, 0.302, 0.302],
                 }


ICING_BASELINE = {0: {'temp': 0.0,
                      'rh': 95.0,
                      'output': [0.348, 0.354, 0.361, 0.356, 0.356]},
                  1: {'temp': 0.0,
                      'rh': 90.0,
                      'output': [0.345, 0.351, 0.358, 0.352, 0.353]},
                  2: {'temp': -5.0,
                      'rh': 90.0,
                      'output': [0.35, 0.355, 0.362, 0.357, 0.357]},
                  }


LOW_TEMP_BASELINE = {0: {'temp': -5.0,
                         'output': [0.334, 0.339, 0.345, 0.342, 0.342]},
                     1: {'temp': -12.0,
                         'output': [0.35, 0.355, 0.362, 0.357, 0.357]},
                     }


@pytest.mark.parametrize('loss', [0.0, 16.7, 30.0])
def test_wind_generic_losses(loss):
    """Test varying wind turbine losses"""
    pc = Gen.get_pc(REV2_POINTS, None, SAM_FILE, 'windpower',
                    sites_per_worker=3, res_file=RES_FILE)

    del pc.project_points.sam_inputs[SAM_FILE]['wind_farm_losses_percent']
    pc.project_points.sam_inputs[SAM_FILE]['turb_generic_loss'] = loss

    gen = Gen('windpower', pc, SAM_FILE, RES_FILE, sites_per_worker=3)
    gen.run(max_workers=1)
    gen_outs = list(gen.out['cf_mean'])

    assert np.allclose(gen_outs, LOSS_BASELINE[loss], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('i', range(3))
def test_wind_icing_losses(i):
    """Test wind icing losses."""
    pc = Gen.get_pc(REV2_POINTS, None, SAM_FILE, 'windpower',
                    sites_per_worker=3, res_file=RES_FILE)

    pc.project_points.sam_inputs[SAM_FILE]['en_icing_cutoff'] = 1
    pc.project_points.sam_inputs[SAM_FILE]['icing_cutoff_temp'] = \
        ICING_BASELINE[i]['temp']
    pc.project_points.sam_inputs[SAM_FILE]['icing_cutoff_rh'] = \
        ICING_BASELINE[i]['rh']

    gen = Gen('windpower', pc, SAM_FILE, RES_FILE, sites_per_worker=3)
    gen.run(max_workers=1)
    gen_outs = list(gen.out['cf_mean'])

    assert np.allclose(gen_outs, ICING_BASELINE[i]['output'],
                       rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('i', range(2))
def test_wind_low_temp_cutoff(i):
    """Test wind low temperature cutoff."""
    pc = Gen.get_pc(REV2_POINTS, None, SAM_FILE, 'windpower',
                    sites_per_worker=3, res_file=RES_FILE)

    pc.project_points.sam_inputs[SAM_FILE]['en_low_temp_cutoff'] = 1
    pc.project_points.sam_inputs[SAM_FILE]['low_temp_cutoff'] = \
        LOW_TEMP_BASELINE[i]['temp']

    gen = Gen('windpower', pc, SAM_FILE, RES_FILE, sites_per_worker=3)
    gen.run(max_workers=1)
    gen_outs = list(gen.out['cf_mean'])

    assert np.allclose(gen_outs, LOW_TEMP_BASELINE[i]['output'],
                       rtol=RTOL, atol=ATOL)


def test_wind_wake_loss_multiplier():
    """Test internal SAM wake loss multiplier"""
    pc = Gen.get_pc(REV2_POINTS, None, LAYOUT_SAM_FILE, 'windpower',
                    sites_per_worker=3, res_file=RES_FILE)
    output_request = ('cf_mean', 'annual_wake_loss_total_percent')

    gen_baseline = Gen('windpower', pc, LAYOUT_SAM_FILE, RES_FILE,
                       output_request=output_request, sites_per_worker=3)
    gen_baseline.run(max_workers=1)
    cf_baseline = gen_baseline.out['cf_mean']
    wl_baseline = gen_baseline.out['annual_wake_loss_total_percent']

    pc.project_points.sam_inputs[LAYOUT_SAM_FILE]['wake_loss_multiplier'] = 1.5
    gen_test = Gen('windpower', pc, LAYOUT_SAM_FILE, RES_FILE,
                   output_request=output_request, sites_per_worker=3)
    gen_test.run(max_workers=1)
    cf_test = gen_test.out['cf_mean']
    wl_test = gen_test.out['annual_wake_loss_total_percent']

    assert (cf_baseline > cf_test).all()
    assert np.allclose(wl_baseline * 1.5, wl_test)


def test_wind_gen_with_ct_curve():
    """Test generation with CT curve"""

    output_request = ("cf_mean", "cf_profile")
    gen = Gen("windpower", (0,), LAYOUT_SAM_FILE, RES_FILE, sites_per_worker=3,
              output_request=output_request)
    gen.run(max_workers=1)
    cf_baseline = gen.out["cf_mean"]

    with open(LAYOUT_SAM_FILE, encoding='utf-8') as fh:
        sam_config = json.load(fh)

    with tempfile.TemporaryDirectory() as td:
        ws_len = len(sam_config['wind_turbine_powercurve_windspeeds'])
        sam_config['wind_turbine_ct_curve'] = [0.1] * ws_len

        sam_fp = os.path.join(td, 'gen.json')
        with open(sam_fp, 'w+') as fh:
            fh.write(json.dumps(sam_config))

        gen_ct = Gen("windpower", (0,), sam_fp, RES_FILE, sites_per_worker=3,
                     output_request=output_request)
        gen_ct.run(max_workers=1)
        cf_with_ct = gen_ct.out["cf_mean"]

    assert cf_with_ct > cf_baseline


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
