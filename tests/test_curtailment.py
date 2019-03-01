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
from reV.SAM.SAM import SAM
from reV.config.project_points import ProjectPoints
from reV import TESTDATADIR
from reV.utilities.curtailment import curtail
from reV.utilities.solar_position import SolarPosition
from reV.generation.generation import Gen


def get_curtailment(year):
    res_file = os.path.join(TESTDATADIR, 'wtk/',
                            'ri_100_wtk_{}.h5'.format(year))
    sam_files = {0: os.path.join(TESTDATADIR, 'SAM/', 'i_windpower.json')}

    pp = ProjectPoints(slice(0, 100), sam_files, 'wind')

    pp.curtailment = os.path.join(TESTDATADIR, 'config/', 'curtailment.json')

    resource = SAM.get_sam_res(res_file, pp, 'wind')
    non_curtailed_res = deepcopy(resource)

    out = curtail(resource, pp.curtailment)

    return out, non_curtailed_res, pp


def test_cf_curtailment(year):

    res_file = os.path.join(TESTDATADIR,
                            'wtk/ri_100_wtk_{}.h5'.format(year))
    sam_files = os.path.join(TESTDATADIR, 'SAM/i_windpower.json')

    curtailment = os.path.join(TESTDATADIR, 'config/', 'curtailment.json')
    points = slice(0, 1)

    # run reV 2.0 generation and write to disk
    gen = Gen.run_direct('wind', points, sam_files, res_file, fout=None,
                         output_request=('cf_profile',),
                         curtailment=curtailment,
                         n_workers=1, sites_per_split=50,
                         return_obj=True, scale_outputs=True)
    results, check = test_res_curtailment(year)
    results['cf_profile'] = gen.out[0]['cf_profile']
    return results


def test_res_curtailment(year):
    """Test wind curtailment."""
    out, non_curtailed_res, pp = get_curtailment(year)

    sza = SolarPosition(
        non_curtailed_res.time_index,
        non_curtailed_res.meta[['latitude', 'longitude']].values).zenith

    ti = non_curtailed_res.time_index

    # optional output df to help check results
    i = 0
    df = pd.DataFrame({'curtailed_wind': out._res_arrays['windspeed'][:, i],
                       'original_wind':
                           non_curtailed_res._res_arrays['windspeed'][:, i],
                       'temperature': out._res_arrays['temperature'][:, i],
                       'sza': sza[:, i],
                       'i': range(len(sza))},
                      index=ti)

    drop_day = ((ti.month == 12) & (ti.day == 31))
    df = df.drop(df.index[drop_day])

    # was it in a curtailment month?
    check1 = np.isin(non_curtailed_res.time_index.month, pp.curtailment.months)
    check1 = np.tile(np.expand_dims(check1, axis=1),
                     non_curtailed_res.shape[1])

    # was the non-curtailed wind speed threshold met?
    check2 = (non_curtailed_res._res_arrays['windspeed'] <
              pp.curtailment.wind_speed)

    # was it nighttime?
    check3 = (sza > pp.curtailment.dawn_dusk)

    # was the temperature threshold met?
    check4 = (out._res_arrays['temperature'] > pp.curtailment.temperature)

    # was windspeed NOT curtailed?
    check5 = (out._res_arrays['windspeed'] != 0)

    # Were all thresholds met and windspeed NOT curtailed?
    check = check1 & check2 & check3 & check4 & check5

    if np.sum(check):
        raise ValueError('All curtailment thresholds were met and windspeed '
                         'was not curtailed!')

    return df, check


if __name__ == '__main__':
    results, check = test_res_curtailment(2012)
    gen = test_cf_curtailment(2012)
