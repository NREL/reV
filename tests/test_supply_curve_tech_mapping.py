# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:37:05 2019

@author: gbuster
"""
import os

import h5py
import numpy as np
import pandas as pd
import pytest

from reV import TESTDATADIR
from reV.handlers.exclusions import ExclusionLayers, LATITUDE, LONGITUDE
from reV.handlers.outputs import Outputs
from reV.supply_curve.tech_mapping import TechMapping
from reV.utilities import MetaKeyName

EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
RES = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_2012.h5')
GEN = os.path.join(TESTDATADIR, 'gen_out/gen_ri_pv_2012_x000.h5')
TM_DSET = 'techmap_nsrdb_ri_truth'


def test_resource_tech_mapping():
    """Run the supply curve technology mapping and compare to baseline file"""
    ind = TechMapping.run(EXCL, RES, dset=None, max_workers=2)

    with ExclusionLayers(EXCL) as ex:
        ind_truth = ex[TM_DSET]

    msg = 'Tech mapping failed for {} vs. baseline results.'
    assert np.allclose(ind, ind_truth), msg.format('index mappings')

    msg = 'Tech mapping didnt find all 100 generation points!'
    assert len(set(ind.flatten())) == 101, msg


# pylint: disable=no-member
def plot_tech_mapping(dist_margin=1.05):
    """Run the supply curve technology mapping and plot the resulting mapped
    points."""

    import matplotlib.pyplot as plt

    with h5py.File(EXCL, 'r') as f:
        lats = f[LATITUDE][...].flatten()
        lons = f[LONGITUDE][...].flatten()
        ind_truth = f[TM_DSET][...].flatten()

    with Outputs(GEN) as fgen:
        gen_meta = fgen.meta

    ind_test = TechMapping.run(EXCL, RES, dset=None, max_workers=2,
                               dist_margin=dist_margin)

    df = pd.DataFrame({LATITUDE: lats,
                       LONGITUDE: lons,
                       TM_DSET: ind_truth,
                       'test': ind_test.flatten()})

    _, axs = plt.subplots(1, 1)
    colors = ['b', 'g', 'c', 'm', 'y']
    colors *= 100

    for i, ind in enumerate(df[TM_DSET].unique()):
        if ind != -1:
            mask = df[TM_DSET] == ind
            axs.scatter(df.loc[mask, LONGITUDE],
                        df.loc[mask, LATITUDE],
                        c=colors[i], s=0.001)

        elif ind == -1:
            mask = df[TM_DSET] == ind
            axs.scatter(df.loc[mask, LONGITUDE],
                        df.loc[mask, LATITUDE],
                        c='r', s=0.001)

    for ind in df[TM_DSET].unique():
        if ind != -1:
            axs.scatter(gen_meta.loc[ind, LONGITUDE],
                        gen_meta.loc[ind, LATITUDE],
                        c='w', s=1)

    for ind in df['test'].unique():
        if ind != -1:
            axs.scatter(gen_meta.loc[ind, LONGITUDE],
                        gen_meta.loc[ind, LATITUDE],
                        c='r', s=1)

    mask = df[TM_DSET].values != df['test']
    axs.scatter(df.loc[mask, LONGITUDE],
                df.loc[mask, LATITUDE],
                c='k', s=1)

    axs.axis('equal')
    plt.show()
    return df, df.loc[mask]


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
