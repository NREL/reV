# -*- coding: utf-8 -*-
"""
PyTest file for NRWAL analysis module

Created on Dec 16 2019

@author: gbuster
"""

import json
import os
import shutil
import tempfile
import traceback

import numpy as np
import pandas as pd
import pytest
from rex.utilities.utilities import pd_date_range

from reV import TESTDATADIR
from reV.cli import main
from reV.handlers.outputs import Outputs
from reV.nrwal.nrwal import RevNrwal
from reV.utilities import MetaKeyName, ModuleName

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
                       MetaKeyName.OFFSHORE: offshore_config}
        nrwal_configs = {MetaKeyName.OFFSHORE: os.path.join(td, 'nrwal_offshore.yaml')}

        with Outputs(gen_fpath, 'a') as f:
            f.time_index = pd_date_range('20100101', '20110101',
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

        obj = RevNrwal(gen_fpath, site_data, sam_configs, nrwal_configs,
                       output_request, site_meta_cols=['depth'])
        obj.run()

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
        _ = RevNrwal(gen_fpath, site_data, sam_configs, nrwal_configs,
                     output_request, site_meta_cols=['depth']).run()

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


@pytest.mark.parametrize("out_fn", ["nrwal_meta.csv", None])
def test_nrwal_csv(out_fn):
    """Test the reV nrwal class with csv output."""
    with tempfile.TemporaryDirectory() as td:
        for fn in os.listdir(SOURCE_DIR):
            shutil.copy(os.path.join(SOURCE_DIR, fn), os.path.join(td, fn))

        gen_fpath = os.path.join(td, 'gen_2010_node00.h5')
        site_data = os.path.join(td, 'example_offshore_data.csv')
        offshore_config = os.path.join(td, 'offshore.json')
        onshore_config = os.path.join(td, 'onshore.json')
        sam_configs = {'onshore': onshore_config,
                       MetaKeyName.OFFSHORE: offshore_config}
        nrwal_configs = {MetaKeyName.OFFSHORE: os.path.join(td, 'nrwal_offshore.yaml')}

        with Outputs(gen_fpath, 'a') as f:
            f.time_index = pd_date_range('20100101', '20110101',
                                         closed='right', freq='1h')
            f._add_dset('cf_profile_raw', np.random.random(f.shape),
                        np.uint32, attrs={'scale_factor': 1000},
                        chunks=(None, 10))
            f._add_dset('cf_mean_raw', np.random.random(f.shape[1]),
                        np.uint32, attrs={'scale_factor': 1000},
                        chunks=None)
            f._add_dset('fixed_charge_rate',
                        0.09 * np.ones(f.shape[1], dtype=np.float32),
                        np.float32, attrs={'scale_factor': 1},
                        chunks=None)

        compatible = ['depth', 'total_losses', 'array', 'export',
                      'gcf_adjustment', 'fixed_charge_rate', 'lcoe_fcr',
                      'cf_mean']
        incompatible = ['cf_profile']
        output_request = compatible + incompatible

        with pytest.warns(Warning) as record:
            rev_nrwal = RevNrwal(gen_fpath, site_data, sam_configs,
                                 nrwal_configs, output_request,
                                 site_meta_cols=['depth'])
            out_fpath = os.path.join(td, out_fn) if out_fn else None
            out_fpath = rev_nrwal.run(csv_output=True, out_fpath=out_fpath)

        expected_message_out = ["`save_raw` option not allowed with "
                                "`csv_output`"]
        for r, m in zip(record, expected_message_out):
            warn_msg = r.message.args[0]
            assert m in warn_msg

        out_fn = out_fn or "gen_2010_node00.csv"
        expected_out_fpath = os.path.join(td, out_fn)
        assert expected_out_fpath == out_fpath
        assert out_fn in os.listdir(td)

        new_data = pd.read_csv(out_fpath)
        for col in compatible:
            assert col in new_data
        for col in incompatible:
            assert col not in new_data


def test_nrwal_constant_eq_output_request():
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
                       MetaKeyName.OFFSHORE: offshore_config}
        nrwal_configs = {MetaKeyName.OFFSHORE: os.path.join(td, 'nrwal_offshore.yaml')}

        with Outputs(gen_fpath, 'a') as f:
            f.time_index = pd_date_range('20100101', '20110101',
                                         closed='right', freq='1h')
            f._add_dset('cf_profile', np.random.random(f.shape),
                        np.uint32, attrs={'scale_factor': 1000},
                        chunks=(None, 10))
            f._add_dset('fixed_charge_rate',
                        0.09 * np.ones(f.shape[1], dtype=np.float32),
                        np.float32, attrs={'scale_factor': 1},
                        chunks=None)

        with Outputs(gen_fpath, 'r') as f:
            meta_raw = f.meta
            mask = meta_raw.offshore == 1

        output_request = ['cf_mean', 'cf_profile',
                          'lease_price', 'lease_price_mil']

        RevNrwal(gen_fpath, site_data, sam_configs, nrwal_configs,
                 output_request, site_meta_cols=['depth']).run()

        with Outputs(gen_fpath, 'r') as f:
            lease_price = f['lease_price']
            scaled_lease_price = f['lease_price_mil']

        assert np.allclose(lease_price[mask] / 1e6, scaled_lease_price[mask])


def test_nrwal_cli(runner, clear_loggers):
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
                       MetaKeyName.OFFSHORE: offshore_config}
        nrwal_configs = {MetaKeyName.OFFSHORE: os.path.join(td, 'nrwal_offshore.yaml')}

        with Outputs(gen_fpath, 'a') as f:
            f.time_index = pd_date_range('20100101', '20110101',
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

        config = {
            "execution_control": {
                "nodes": 1,
                "option": "local",
            },
            "log_level": "INFO",
            "log_directory": td,
            "gen_fpath": gen_fpath,
            "site_data": site_data,
            "sam_files": sam_configs,
            "nrwal_configs": nrwal_configs,
            "output_request": output_request,
            'site_meta_cols': ['depth'],
        }

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, [str(ModuleName.NRWAL),
                                      '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

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
            for key in output_request:
                assert key in f
                if 'profile' not in key:
                    assert np.isnan(f[key][mask]).sum() == 0
                else:
                    assert np.isnan(f[key][:, mask]).sum() == 0

                if key in ('total_losses', 'array', 'export'):
                    assert np.isnan(f[key][~mask]).all()

        clear_loggers()

        result = runner.invoke(main, [str(ModuleName.NRWAL),
                                      '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

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
        assert 'depth' in meta_new

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

        clear_loggers()


def test_nrwal_cli_csv(runner, clear_loggers):
    """Test the reV nrwal module, calculating offshore wind lcoe and losses
    from reV generation outputs and using the NRWAL library and saving outputs
    to CSV file"""
    with tempfile.TemporaryDirectory() as td:
        for fn in os.listdir(SOURCE_DIR):
            shutil.copy(os.path.join(SOURCE_DIR, fn), os.path.join(td, fn))

        gen_fpath = os.path.join(td, 'gen_2010_node00.h5')
        site_data = os.path.join(td, 'example_offshore_data.csv')
        offshore_config = os.path.join(td, 'offshore.json')
        onshore_config = os.path.join(td, 'onshore.json')
        sam_configs = {'onshore': onshore_config,
                       MetaKeyName.OFFSHORE: offshore_config}
        nrwal_configs = {MetaKeyName.OFFSHORE: os.path.join(td, 'nrwal_offshore.yaml')}

        with Outputs(gen_fpath, 'a') as f:
            f.time_index = pd_date_range('20100101', '20110101',
                                         closed='right', freq='1h')
            f._add_dset('cf_profile_raw', np.random.random(f.shape),
                        np.uint32, attrs={'scale_factor': 1000},
                        chunks=(None, 10))
            f._add_dset('cf_mean_raw', np.random.random(f.shape[1]),
                        np.uint32, attrs={'scale_factor': 1000},
                        chunks=None)
            f._add_dset('fixed_charge_rate',
                        0.09 * np.ones(f.shape[1], dtype=np.float32),
                        np.float32, attrs={'scale_factor': 1},
                        chunks=None)

        output_request = ['fixed_charge_rate', 'depth', 'total_losses',
                          'array', 'export', 'gcf_adjustment',
                          'lcoe_fcr', 'cf_mean', 'cf_profile']

        config = {
            "execution_control": {
                "nodes": 1,
                "option": "local",
            },
            "log_level": "INFO",
            "log_directory": td,
            "gen_fpath": gen_fpath,
            "site_data": site_data,
            "sam_files": sam_configs,
            "nrwal_configs": nrwal_configs,
            "output_request": output_request,
            'site_meta_cols': ['depth'],
            "csv_output": True
        }

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, [str(ModuleName.NRWAL),
                                      '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        out_fn = "{}_{}.csv".format(os.path.basename(td), ModuleName.NRWAL)
        assert out_fn in os.listdir(td)
        new_data = pd.read_csv(os.path.join(td, out_fn))
        for col in output_request[:-1]:
            assert col in new_data
        assert 'cf_profile' not in new_data

        clear_loggers()


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
