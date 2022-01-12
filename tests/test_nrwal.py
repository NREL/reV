# -*- coding: utf-8 -*-
"""
PyTest file for NRWAL analysis module

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
from reV.nrwal.nrwal import RevNrwal
from reV import TESTDATADIR


SOURCE_DIR = os.path.join(TESTDATADIR, 'nrwal/')


def test_nrwal():
    """Test the reV nrwal module, calculating offshore wind lcoe and losses
    from reV generation outputs and using the NRWAL library"""
    with tempfile.TemporaryDirectory() as td:
        for fn in os.listdir(SOURCE_DIR):
            shutil.copy(os.path.join(SOURCE_DIR, fn), os.path.join(td, fn))

        gen_fpath = os.path.join(td, 'gen_2010_node00.h5')
        site_data = os.path.join(td, 'example_offshore_data.csv')
        offshore_config = os.path.join(td, 'offshore.json')
        onshore_config = os.path.join(td, 'onshore.json')
        sam_configs = {'onshore': onshore_config,
                       'offshore': offshore_config}
        nrwal_configs = {'offshore': os.path.join(td, 'nrwal_offshore.yaml')}

        with Outputs(gen_fpath, 'a') as f:
            f.time_index = pd.date_range('20100101', '20110101',
                                         closed='right', freq='1h')
            f._add_dset('cf_profile', np.random.random(f.shape),
                        np.uint32, attrs={'scale_factor': 1000},
                        chunks=(None, 10))
            f._add_dset('fixed_charge_rate',
                        0.09 * np.ones(f.shape[1], dtype=np.float32),
                        np.float32, attrs={'scale_factor': 1},
                        chunks=None)

        with Outputs(gen_fpath, 'r') as f:
            original_dsets = [d for d in f.dsets
                              if d not in ('meta', 'time_index')]
            meta_raw = f.meta
            lcoe_raw = f['lcoe_fcr']
            cf_mean_raw = f['cf_mean']
            cf_profile_raw = f['cf_profile']
            mask = meta_raw.offshore == 1

        output_request = ['fixed_charge_rate', 'depth', 'total_losses',
                          'array', 'export', 'gcf_adjustment',
                          'lcoe_fcr', 'cf_mean', 'cf_profile']

        obj = RevNrwal.run(gen_fpath, site_data, sam_configs, nrwal_configs,
                           output_request, site_meta_cols=['depth'])

        with Outputs(gen_fpath, 'r') as f:
            meta_new = f.meta
            lcoe_new = f['lcoe_fcr']
            losses = f['total_losses']
            gcf_adjustment = f['gcf_adjustment']
            assert np.allclose(cf_mean_raw, f['cf_mean_raw'])
            assert np.allclose(cf_profile_raw, f['cf_profile_raw'])
            cf_mean_new = f['cf_mean']
            cf_profile_new = f['cf_profile']
            fcr = f['fixed_charge_rate']
            depth = f['depth']

            for key in [d for d in original_dsets if d in f]:
                assert key in f
                assert np.isnan(f[key]).sum() == 0

            # check nrwal keys requested as h5 dsets
            for key in obj._output_request:
                assert key in f
                if 'profile' not in key:
                    assert np.isnan(f[key][mask]).sum() == 0
                else:
                    assert np.isnan(f[key][:, mask]).sum() == 0

                if key in ('total_losses', 'array', 'export'):
                    assert np.isnan(f[key][~mask]).all()

        # run offshore twice and make sure losses don't get doubled
        _ = RevNrwal.run(gen_fpath, site_data, sam_configs, nrwal_configs,
                         output_request, site_meta_cols=['depth'])

        # make sure the second offshore compute gives same results as first
        with Outputs(gen_fpath, 'r') as f:
            assert np.allclose(lcoe_new, f['lcoe_fcr'])
            assert np.allclose(cf_mean_new, f['cf_mean'])
            assert np.allclose(cf_profile_new, f['cf_profile'])
            assert np.allclose(cf_mean_raw, f['cf_mean_raw'])
            assert np.allclose(cf_profile_raw, f['cf_profile_raw'])

        # check offshore depth data
        assert 'depth' in meta_new
        assert all(meta_new.loc[(meta_new.offshore == 1), 'depth'] >= 0)
        assert all(np.isnan(
            meta_new.loc[(meta_new.offshore == 0), 'depth']))
        assert all(depth[(meta_new.offshore == 1)] >= 0)
        assert all(np.isnan(depth[(meta_new.offshore == 0)]))

        # check difference fcr values for onshore/offshore
        assert all(fcr[(meta_new.offshore == 1)] == 0.071)
        assert all(fcr[(meta_new.offshore == 0)] == 0.09)
        assert not any(np.isnan(fcr))

        # make sure all of the requested offshore meta columns got
        # sent to the new meta data
        for col in obj._site_meta_cols:
            assert col in meta_new

        # sanity check lcoe and cf values
        assert (lcoe_new[mask] != lcoe_raw[mask]).all()
        assert (lcoe_new[~mask] == lcoe_raw[~mask]).all()
        assert (cf_mean_new[mask] < cf_mean_raw[mask]).all()

        cf_net = (gcf_adjustment[mask] * (1 - losses[mask])
                  * cf_profile_raw[:, mask])
        assert np.allclose(cf_profile_new[:, mask], cf_net, rtol=0.005,
                           atol=0.001)
        assert np.allclose(cf_profile_new[:, ~mask], cf_profile_raw[:, ~mask])

        cf_net = gcf_adjustment[mask] * (1 - losses[mask]) * cf_mean_raw[mask]
        assert np.allclose(cf_mean_new[mask], cf_net, rtol=0.005)
        assert np.allclose(cf_mean_new[~mask], cf_mean_raw[~mask])


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
