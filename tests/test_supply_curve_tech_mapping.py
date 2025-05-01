# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:37:05 2019

@author: gbuster
"""
import os
import shutil
import json
import traceback

import h5py
import numpy as np
import pandas as pd
import pytest

from reV import TESTDATADIR
from reV.cli import main
from reV.utilities import ModuleName
from reV.handlers.exclusions import ExclusionLayers, LATITUDE, LONGITUDE
from reV.handlers.outputs import Outputs
from reV.supply_curve.tech_mapping import TechMapping


EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
RES = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_2012.h5')
GEN = os.path.join(TESTDATADIR, 'gen_out/gen_ri_pv_2012_x000.h5')
TM_DSET = 'techmap_nsrdb_ri_truth'


@pytest.mark.parametrize("batch_size", [100, 50])
def test_resource_tech_mapping(tmp_path, batch_size):
    """Run the supply curve technology mapping and compare to baseline file"""

    excl_fpath = EXCL
    excl_fpath = tmp_path.joinpath("excl.h5").as_posix()
    shutil.copy(EXCL, excl_fpath)

    dset = "tm"
    TechMapping.run(
        excl_fpath, RES, dset=dset, max_workers=2, sc_resolution=2560,
        batch_size=batch_size
    )

    with ExclusionLayers(EXCL) as ex:
        ind_truth = ex[TM_DSET]

    with ExclusionLayers(excl_fpath) as out:
        assert dset in out, "Techmap dataset was not written to H5"
        ind = out[dset]

    msg = 'Tech mapping failed for {} vs. baseline results.'
    assert np.allclose(ind, ind_truth), msg.format('index mappings')

    msg = 'Tech mapping didnt find all 100 generation points!'
    assert len(set(ind.flatten())) == 101, msg


def test_tech_mapping_cli(runner, clear_loggers, tmp_path):
    """Test tech-mapping CLI command"""

    excl_fpath = EXCL
    excl_fpath = tmp_path.joinpath("excl.h5").as_posix()
    shutil.copy(EXCL, excl_fpath)

    dset = "tm"
    config = {
        "log_directory": tmp_path.as_posix(),
        "execution_control": {
            "option": "local",
            "max_workers": 2,
        },
        "log_level": "INFO",
        "excl_fpath": excl_fpath,
        "dset": "tm",
        "res_fpath": RES,
        "sc_resolution": 2560,
        "batch_size": 50,
    }

    config_path = tmp_path.joinpath("config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    result = runner.invoke(
        main, [ModuleName.TECH_MAPPING, "-c", config_path.as_posix()]
    )
    clear_loggers()

    if result.exit_code != 0:
        msg = "Failed with error {}".format(
            traceback.print_exception(*result.exc_info)
        )
        raise RuntimeError(msg)

    with ExclusionLayers(EXCL) as ex:
        ind_truth = ex[TM_DSET]

    with ExclusionLayers(excl_fpath) as out:
        assert dset in out, "Techmap dataset was not written to H5"
        ind = out[dset]

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
                               dist_margin=dist_margin, sc_resolution=2560)

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

    fname = __file__
    pytest.main(["-q", "--show-capture={}".format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
