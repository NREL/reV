# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:37:05 2019

@author: gbuster
"""
import h5py
import numpy as np
import pandas as pd
import pytest
import os

from reV import TESTDATADIR
from reV.handlers.outputs import Outputs
from reV.supply_curve.tech_mapping import TechMapping
from reV.handlers.exclusions import ExclusionLayers


F_EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
F_RES = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_2012.h5')
F_GEN = os.path.join(TESTDATADIR, 'gen_out/gen_ri_pv_2012_x000.h5')
DSET_TM = 'techmap_nsrdb_ri_truth'


def test_resource_tech_mapping():
    """Run the supply curve technology mapping and compare to baseline file"""

    lats, lons, ind = TechMapping.run(F_EXCL, F_RES, DSET_TM, n_cores=2,
                                      save_flag=False, return_flag=True)

    with ExclusionLayers(F_EXCL) as ex:
        lat_truth = ex.latitude
        lon_truth = ex.longitude
        ind_truth = ex[DSET_TM]

    msg = 'Tech mapping failed for {} vs. baseline results.'
    assert np.allclose(lats, lat_truth), msg.format('latitudes')
    assert np.allclose(lons, lon_truth), msg.format('longitudes')
    assert np.allclose(ind, ind_truth), msg.format('index mappings')

    msg = 'Tech mapping didnt find all 100 generation points!'
    assert len(set(ind.flatten())) == 101, msg


def plot_tech_mapping():
    """Run the supply curve technology mapping and plot the resulting mapped
    points."""

    import matplotlib.pyplot as plt

    with h5py.File(F_EXCL, 'r') as f:
        lats = f['latitude'][...].flatten()
        lons = f['longitude'][...].flatten()
        ind = f[DSET_TM][...].flatten()

    with Outputs(F_GEN) as fgen:
        gen_meta = fgen.meta

    df = pd.DataFrame({'latitude': lats,
                       'longitude': lons,
                       DSET_TM: ind})

    _, axs = plt.subplots(1, 1)
    colors = ['b', 'g', 'c', 'm', 'k', 'y']
    colors *= 100

    for i, ind in enumerate(df[DSET_TM].unique()):
        if ind != -1:
            mask = df[DSET_TM] == ind
            axs.scatter(df.loc[mask, 'longitude'],
                        df.loc[mask, 'latitude'],
                        c=colors[i], s=0.001)

        elif ind == -1:
            mask = df[DSET_TM] == ind
            axs.scatter(df.loc[mask, 'longitude'],
                        df.loc[mask, 'latitude'],
                        c='r', s=0.001)

    for ind in df[DSET_TM].unique():
        if ind != -1:
            axs.scatter(gen_meta.loc[ind, 'longitude'],
                        gen_meta.loc[ind, 'latitude'],
                        c='w', s=1)

    axs.axis('equal')
    plt.show()
    return df


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
