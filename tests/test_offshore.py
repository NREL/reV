# -*- coding: utf-8 -*-
"""
PyTest file for offshore aggregation of reV generation results and ORCA econ.

Created on Dec 16 2019

@author: gbuster
"""

import os
import pandas as pd
import numpy as np
import shutil
import pytest
import tempfile

from reV.handlers.outputs import Outputs
from reV.config.project_points import ProjectPoints
from reV.offshore.offshore import Offshore
from reV import TESTDATADIR


SOURCE_DIR = os.path.join(TESTDATADIR, 'offshore/')


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
    #    execute_pytest()
    from rex import init_logger
    init_logger('NRWAL')
    init_logger('reV.offshore')

    with tempfile.TemporaryDirectory() as td:
        for fn in os.listdir(SOURCE_DIR):
            shutil.copy(os.path.join(SOURCE_DIR, fn), os.path.join(td, fn))

        gen_fpath = os.path.join(td, 'gen_2010_node00.h5')
        offshore_fpath = os.path.join(td, 'example_offshore_data.csv')
        offshore_config = os.path.join(td, 'offshore.json')
        onshore_config = os.path.join(td, 'onshore.json')
        sam_configs = {'onshore': onshore_config,
                       'offshore': offshore_config}
        nrwal_configs = {'offshore': os.path.join(td, 'nrwal_offshore.yaml')}
        project_points = os.path.join(td, 'ri_offshore_proj_points.csv')
        project_points = ProjectPoints(project_points, sam_configs)

        with Outputs(gen_fpath, 'a') as f:
            f.time_index = pd.date_range('20100101', '20110101',
                                         closed='right', freq='1h')
            f._add_dset('cf_profile', np.random.random(f.shape),
                        np.uint32, attrs={'scale_factor': 1000},
                        chunks=(None, 10))

        with Outputs(gen_fpath, 'r') as f:
            meta_raw = f.meta
            lcoe_raw = f['lcoe_fcr']
            cf_mean_raw = f['cf_mean']
            cf_profile_raw = f['cf_profile']
            mask = meta_raw.offshore == 1

        off = Offshore(gen_fpath, offshore_fpath, nrwal_configs,
                       project_points, max_workers=1)

        off.run()
        off.write_to_gen_fpath()

        with Outputs(gen_fpath, 'r') as f:
            meta_new = f.meta
            lcoe_new = f['lcoe_fcr']
            cf_mean_new = f['cf_mean']
            cf_profile_new = f['cf_profile']

            for col in off._offshore_meta_cols:
                assert col in meta_new

            for key in off._offshore_nrwal_keys:
                assert key in f.dsets
                assert np.isnan(f[key][mask]).sum() == 0
                assert np.isnan(f[key][~mask]).all()

        assert (lcoe_new[mask] != lcoe_raw[mask]).all()
        assert (lcoe_new[~mask] == lcoe_raw[~mask]).all()
        assert (cf_mean_new[mask] < cf_mean_raw[mask]).all()
        assert (cf_mean_new[~mask] == cf_mean_raw[~mask]).all()
        assert (cf_profile_new[:, mask] <= cf_profile_raw[:, mask]).all()
        assert (cf_profile_new[:, ~mask] == cf_profile_raw[:, ~mask]).all()
