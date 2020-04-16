# -*- coding: utf-8 -*-
# pylint: skip-file
"""
Created on Fri Mar  1 15:24:13 2019

@author: gbuster
"""
from copy import deepcopy
import os
import numpy as np
import pandas as pd
import pytest
from reV.SAM.SAM import SAM
from reV.config.project_points import ProjectPoints
from reV import TESTDATADIR
from reV.utilities.curtailment import curtail
from reV.generation.generation import Gen

from rex.utilities.solar_position import SolarPosition


def get_curtailment(year):
    """Get the curtailed and non-curtailed resource objects, and project points
    """
    res_file = os.path.join(TESTDATADIR, 'wtk/',
                            'ri_100_wtk_{}.h5'.format(year))
    sam_files = {0: os.path.join(TESTDATADIR, 'SAM/',
                                 'wind_gen_standard_losses_0.json')}
    curtailment = os.path.join(TESTDATADIR, 'config/', 'curtailment.json')
    pp = ProjectPoints(slice(0, 100), sam_files, 'windpower',
                       curtailment=curtailment)

    resource = SAM.get_sam_res(res_file, pp, 'windpower')
    non_curtailed_res = deepcopy(resource)

    out = curtail(resource, pp.curtailment, random_seed=0)

    return out, non_curtailed_res, pp


@pytest.mark.parametrize(('year', 'site'),
                         [('2012', 0),
                          ('2012', 10),
                          ('2013', 0),
                          ('2013', 10)])
def test_cf_curtailment(year, site):
    """Run Wind generation and ensure that the cf_profile is zero when
    curtailment is expected.

    Note that the probability of curtailment must be 1 for this to succeed.
    """

    res_file = os.path.join(TESTDATADIR,
                            'wtk/ri_100_wtk_{}.h5'.format(year))
    sam_files = os.path.join(TESTDATADIR,
                             'SAM/wind_gen_standard_losses_0.json')

    curtailment = os.path.join(TESTDATADIR, 'config/', 'curtailment.json')
    points = slice(site, site + 1)

    # run reV 2.0 generation and write to disk
    gen = Gen.reV_run('windpower', points, sam_files, res_file, fout=None,
                      output_request=('cf_profile',),
                      curtailment=curtailment,
                      max_workers=1, sites_per_worker=50,
                      scale_outputs=True)
    results, check_curtailment = test_res_curtailment(year, site=site)
    results['cf_profile'] = gen.out['cf_profile'].flatten()

    # was capacity factor NOT curtailed?
    check_cf = (gen.out['cf_profile'].flatten() != 0)

    # Were all thresholds met and windspeed NOT curtailed?
    check = check_curtailment & check_cf

    msg = ('All curtailment thresholds were met and cf_profile '
           'was not curtailed!')
    assert np.sum(check) == 0, msg

    return results


@pytest.mark.parametrize(('year', 'site'),
                         [('2012', 10),
                          ('2013', 10)])
def test_random(year, site):
    """Run wind generation and ensure that no curtailment, 100% probability
    curtailment, and 50% probability curtailment result in expected decreases
    in the annual cf_mean.
    """
    res_file = os.path.join(TESTDATADIR,
                            'wtk/ri_100_wtk_{}.h5'.format(year))
    sam_files = os.path.join(TESTDATADIR,
                             'SAM/wind_gen_standard_losses_0.json')
    results = []
    no_curtail = None
    curtailment = {"dawn_dusk": "nautical", "months": [4, 5, 6, 7],
                   "precipitation": None, "probability": 1,
                   "temperature": None, "wind_speed": 10.0}
    prob_curtail = {"dawn_dusk": "nautical", "months": [4, 5, 6, 7],
                    "precipitation": None, "probability": 0.5,
                    "temperature": None, "wind_speed": 10.0}

    for c in [no_curtail, curtailment, prob_curtail]:

        points = slice(site, site + 1)

        # run reV 2.0 generation and write to disk
        gen = Gen.reV_run('windpower', points, sam_files, res_file, fout=None,
                          output_request=('cf_profile',),
                          curtailment=c,
                          max_workers=1, sites_per_worker=50,
                          scale_outputs=True)

        results.append(gen.out['cf_mean'])

    assert results[0] > results[1], 'Curtailment did not decrease cf_mean!'

    expected = (results[0] + results[1]) / 2
    diff = expected - results[2]
    msg = ('Curtailment with 50% probability did not result in 50% less '
           'curtailment! No curtailment, curtailment, and 50% curtailment '
           'have the following cf_means: {}'.format(results))
    assert diff <= 2, msg


@pytest.mark.parametrize(('year', 'site'),
                         [('2012', 50),
                          ('2013', 50)])
def test_res_curtailment(year, site):
    """Test wind resource curtailment."""
    out, non_curtailed_res, pp = get_curtailment(year)

    sza = SolarPosition(
        non_curtailed_res.time_index,
        non_curtailed_res.meta[['latitude', 'longitude']].values).zenith

    ti = non_curtailed_res.time_index

    # was it in a curtailment month?
    check1 = np.isin(non_curtailed_res.time_index.month, pp.curtailment.months)
    check1 = np.tile(np.expand_dims(check1, axis=1),
                     non_curtailed_res.shape[1])

    # was the non-curtailed wind speed threshold met?
    check2 = (non_curtailed_res._res_arrays['windspeed']
              < pp.curtailment.wind_speed)

    # was it nighttime?
    check3 = (sza > pp.curtailment.dawn_dusk)

    # was the temperature threshold met?
    check4 = (out._res_arrays['temperature'] > pp.curtailment.temperature)

    # thresholds for curtailment
    check_curtailment = check1 & check2 & check3 & check4

    # was windspeed NOT curtailed?
    check5 = (out._res_arrays['windspeed'] != 0)

    # Were all thresholds met and windspeed NOT curtailed?
    check = check_curtailment & check5

    msg = ('All curtailment thresholds were met and windspeed '
           'was not curtailed!')

    assert np.sum(check) == 0, msg

    # optional output df to help check results
    i = site
    df = pd.DataFrame({'i': range(len(sza)),
                       'curtailed_wind': out._res_arrays['windspeed'][:, i],
                       'original_wind':
                           non_curtailed_res._res_arrays['windspeed'][:, i],
                       'temperature': out._res_arrays['temperature'][:, i],
                       'sza': sza[:, i],
                       'wind_curtail': check2[:, i],
                       'month_curtail': check1[:, i],
                       'sza_curtail': check3[:, i],
                       'temp_curtail': check4[:, i],
                       },
                      index=ti)

    if str(year) == '2012':
        drop_day = ((ti.month == 12) & (ti.day == 31))
        df = df.drop(df.index[drop_day])
        check_curtailment = check_curtailment[~drop_day, :]

    return df, check_curtailment[:, site]


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
#    result = test_cf_curtailment(2012, 10)
