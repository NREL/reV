# -*- coding: utf-8 -*-
"""
PyTest file for offshore aggregation of reV generation results and ORCA econ.

Created on Dec 16 2019

@author: gbuster
"""

import os
import pytest
import numpy as np
import json

from reV.handlers.outputs import Outputs
from reV import TESTDATADIR
from reV.offshore.offshore import Offshore


CF_FILE = os.path.join(TESTDATADIR, 'gen_out/ri_wind_gen_profiles_2010.h5')
OFFSHORE_FILE = os.path.join(
    TESTDATADIR, 'offshore/preliminary_orca_results_09042019_JN.csv')
POINTS = os.path.join(TESTDATADIR, 'offshore/project_points.csv')
SAM_FILES = {'default': os.path.join(TESTDATADIR,
                                     'offshore/6MW_offshore.json')}


@pytest.fixture
def offshore():
    """Offshore aggregation object for tests and plotting."""
    offshore = Offshore.run(CF_FILE, OFFSHORE_FILE, POINTS, SAM_FILES)
    return offshore


def test_offshore_agg(offshore):
    """Run an offshore aggregation test and validate a few outputs against
    the raw gen output."""
    assert len(offshore.out['cf_mean']) == len(offshore.meta_out_offshore)

    for i in range(0, 20):

        agg_gids = offshore.meta_out_offshore.iloc[i]['aggregated_gids']
        agg_gids = json.loads(agg_gids)

        assert offshore.meta_out_offshore.iloc[i]['gid'] - 1e7 in agg_gids

        with Outputs(CF_FILE) as out:
            mask = np.isin(out.meta['gid'], agg_gids)
            gen_gids = np.where(mask)[0]
            ws_mean = out['ws_mean', gen_gids]
            lcoe_land = out['lcoe_fcr', gen_gids]
            cf_mean = out['cf_mean', gen_gids]
            cf_profile = out['cf_profile', :, gen_gids]

        assert offshore.out['lcoe_fcr'][i] != lcoe_land.mean()
        assert offshore.out['cf_mean'][i] == cf_mean.mean()
        assert offshore.out['ws_mean'][i] == ws_mean.mean()
        check_profiles = np.allclose(offshore.out['cf_profile'][:, i],
                                     cf_profile.mean(axis=1))
        assert check_profiles


def plot_map(offshore):
    """Plot a map of offshore farm aggregation."""
    import matplotlib.pyplot as plt
    plt.scatter(offshore.meta_source_onshore['longitude'],
                offshore.meta_source_onshore['latitude'],
                c=(0.5, 0.5, 0.5), marker='s')

    cs = ['r', 'g', 'c', 'm', 'y', 'b'] * 100
    for ic, i in enumerate(np.unique(offshore._i)):
        if i != -1:
            ilocs = np.where(offshore._i == i)[0]
            plt.scatter(offshore.meta_source_offshore.iloc[ilocs]['longitude'],
                        offshore.meta_source_offshore.iloc[ilocs]['latitude'],
                        c=cs[ic], marker='s')
    plt.scatter(offshore.meta_out_offshore['longitude'],
                offshore.meta_out_offshore['latitude'],
                c='k', marker='x')
    plt.axis('equal')
    plt.show()
    plt.close()


def plot_timeseries(offshore, i=0):
    """Plot a timeseries of aggregated cf profile for offshore"""
    import matplotlib.pyplot as plt

    agg_gids = offshore.meta_out_offshore.iloc[i]['aggregated_gids']
    agg_gids = json.loads(agg_gids)

    with Outputs(CF_FILE) as out:
        mask = np.isin(out.meta['gid'], agg_gids)
        gen_gids = np.where(mask)[0]
        cf_profile = out['cf_profile', :, gen_gids]

    tslice = slice(100, 120)
    a = plt.plot(cf_profile[tslice, :], c=(0.8, 0.8, 0.8))
    b = plt.plot(offshore.out['cf_profile'][tslice, i], c='b')
    plt.legend([a[0], b[0]], ['Resource Pixels', 'Offshore Aggregate'])
    plt.ylabel('Capacity Factor Profile')
    plt.show()
    plt.close()


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
