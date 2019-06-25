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


F_EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/exclusions.tif')
F_GEN = os.path.join(TESTDATADIR, 'gen_out/gen_ri_pv_2012_x000.h5')
F_OUT = os.path.join(TESTDATADIR, 'sc_out/tech_map.h5')
F_BASELINE = os.path.join(TESTDATADIR, 'sc_out/baseline_ri_tech_map.h5')

PURGE_OUT = True


def test_tech_mapping():
    """Run the supply curve technology mapping and compare to baseline file"""

    TechMapping.run_map(F_EXCL, F_GEN, F_OUT, n_cores=2)

    with h5py.File(F_BASELINE, 'r') as f_baseline:
        with h5py.File(F_OUT, 'r') as f_test:

            for d in list(f_test):

                msg1 = ('Dset "{}" not in the baseline file! '
                        'Please check the test file: {}'
                        .format(d, F_OUT))
                msg2 = ('Data from "{}" does not match the baseline file! '
                        'Please check the test file: {}'
                        .format(d, F_OUT))

                assert d in list(f_baseline), msg1
                assert np.array_equal(f_baseline[d][...], f_test[d][...]), msg2

                if d == 'gen_ind':
                    inds = f_test[d][...].flatten()
                    msg = 'Tech mapping didnt find all 100 generation points!'
                    assert len(set(inds)) == 101, msg

    if PURGE_OUT:
        os.remove(F_OUT)


def plot_tech_mapping():
    """Run the supply curve technology mapping and plot the resulting mapped
    points."""

    import matplotlib.pyplot as plt

    TechMapping.run_map(F_EXCL, F_GEN, F_OUT, n_cores=2)

    with h5py.File(F_OUT, 'r') as f:
        ind = f['gen_ind'][...].flatten()
        lats = f['latitude'][...].flatten()
        lons = f['longitude'][...].flatten()
    os.remove(F_OUT)

    with Outputs(F_GEN) as fgen:
        gen_meta = fgen.meta

    df = pd.DataFrame({'latitude': lats,
                       'longitude': lons,
                       'gen_ind': ind})

    _, axs = plt.subplots(1, 1)
    colors = ['b', 'g', 'c', 'm', 'k', 'y']
    colors *= 100

    for i, ind in enumerate(df['gen_ind'].unique()):
        if ind != -1:
            mask = df['gen_ind'] == ind
            axs.scatter(df.loc[mask, 'longitude'],
                        df.loc[mask, 'latitude'],
                        c=colors[i], s=0.001)

        elif ind == -1:
            mask = df['gen_ind'] == ind
            axs.scatter(df.loc[mask, 'longitude'],
                        df.loc[mask, 'latitude'],
                        c='r', s=0.001)

    for ind in df['gen_ind'].unique():
        if ind != -1:
            axs.scatter(gen_meta.loc[ind, 'longitude'],
                        gen_meta.loc[ind, 'latitude'],
                        c='w', s=1)

    axs.axis('equal')
    plt.show()


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
