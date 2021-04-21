# -*- coding: utf-8 -*-
"""
reV offshore wind analysis module. This module uses the NRWAL library to
assess offshore losses and LCOE to complement the simple SAM windpower module.

Everything in this module operates on the native wind resource resolution.
"""
import numpy as np
import pandas as pd
import logging
from warnings import warn

from reV.generation.generation import Gen
from reV.handlers.outputs import Outputs
from reV.utilities.exceptions import (OffshoreWindInputWarning,
                                      OffshoreWindInputError)
from reV.utilities import log_versions


logger = logging.getLogger(__name__)


class Offshore:
    """Framework to handle offshore wind analysis."""

    # Default columns from the offshore wind data table to join to the
    # offshore meta data
    DEFAULT_META_COLS = ('config', )

    # Default keys from the NRWAL config to export as new datasets
    # in the reV output h5
    DEFAULT_NRWAL_KEYS = ('total_losses', 'array', 'export')

    def __init__(self, gen_fpath, offshore_fpath, nrwal_configs,
                 project_points, offshore_meta_cols=None,
                 offshore_nrwal_keys=None, nrwal_lcoe_key='lcoe',
                 nrwal_loss_key='total_losses'):
        """
        Parameters
        ----------
        gen_fpath : str
            Full filepath to reV gen h5 output file.
        offshore_fpath : str
            Full filepath to offshore wind farm data file. Needs "gid" and
            "config" columns matching the project points input.
        nrwal_configs : dict
            Dictionary lookup of config_id values mapped to config filepaths.
            The same config_id values will be used from the sam_files lookup
            in project_points
        project_points : reV.config.project_points.ProjectPoints
            Instantiated project points instance.
        offshore_meta_cols : list | tuple | None
            Column labels from offshore_fpath to pass through to the output
            meta data. None (default) will use class variable
            DEFAULT_META_COLS, and any additional cols requested here will be
            added to DEFAULT_META_COLS.
        offshore_nrwal_keys : list | tuple | None
            Equation labels from the NRWAL configs to pass through to the
            output h5 file. None will use class variable DEFAULT_NRWAL_KEYS,
            and any additional cols requested here will be added to
            DEFAULT_NRWAL_KEYS.
        nrwal_lcoe_key : str
            Key in the NRWAL config for final LCOE output value. Can be
            changed and runtime for different NRWAL configs using this kwarg.
        nrwal_loss_key : str
            Key in the NRWAL config for final capacity factor losses output
            value. Can be changed and runtime for different NRWAL configs
            using this kwarg.
        """

        log_versions(logger)

        # delayed NRWAL import to cause less errors with old reV installs
        # if not running offshore.
        from NRWAL import NrwalConfig

        self._gen_fpath = gen_fpath
        self._offshore_fpath = offshore_fpath
        self._project_points = project_points
        self._meta_out = None
        self._time_index = None
        self._lcoe_key = nrwal_lcoe_key
        self._loss_key = nrwal_loss_key

        self._nrwal_configs = {k: NrwalConfig(v) for k, v in
                               nrwal_configs.items()}

        self._offshore_meta_cols = offshore_meta_cols
        if self._offshore_meta_cols is None:
            self._offshore_meta_cols = list(self.DEFAULT_META_COLS)
        else:
            self._offshore_meta_cols = list(self._offshore_meta_cols)
            self._offshore_meta_cols += list(self.DEFAULT_META_COLS)
            self._offshore_meta_cols = list(set(self._offshore_meta_cols))

        self._offshore_nrwal_keys = offshore_nrwal_keys
        if self._offshore_nrwal_keys is None:
            self._offshore_nrwal_keys = list(self.DEFAULT_NRWAL_KEYS)
        else:
            self._offshore_nrwal_keys = list(self._offshore_nrwal_keys)
            self._offshore_nrwal_keys += list(self.DEFAULT_NRWAL_KEYS)
            self._offshore_nrwal_keys = list(set(self._offshore_nrwal_keys))

        out = self._parse_gen_data(self._gen_fpath)
        self._meta_source, self._onshore_mask = out[:2]
        self._offshore_mask, self._cf_mean = out[2:]

        self._offshore_data = self._parse_offshore_data(self._offshore_fpath)
        self._system_inputs = self._parse_system_inputs()
        self._preflight_checks()

        logger.info('Initialized offshore wind farm aggregation module with '
                    '{} onshore resource points, {} offshore resource points.'
                    .format(len(self.meta_source_onshore),
                            len(self.meta_source_offshore)))

        self._out = {self._lcoe_key: np.full(len(self._offshore_data), np.nan),
                     self._loss_key: np.full(len(self._offshore_data), np.nan)}
        for key in self._offshore_nrwal_keys:
            if key in self._offshore_data:
                self._out[key] = self._offshore_data[key].values
            else:
                self._out[key] = np.full(len(self._offshore_data), np.nan)

    @staticmethod
    def _parse_gen_data(gen_fpath):
        """Parse cf meta dataframe and get masks for onshore/offshore points.

        Parameters
        ----------
        gen_fpath : str
            Full filepath to reV gen h5 output file.

        Returns
        -------
        meta : pd.DataFrame
            Full meta data from gen_fpath with "offshore" column.
        onshore_mask : pd.Series
            Boolean series indicating where onshore sites are.
        offshore_mask : pd.Series
            Boolean series indicating where offshore sites are.
        cf_mean : np.ndarray
            1D array of mean capacity factor values corresponding to the
            un-masked meta data
        """

        with Outputs(gen_fpath, mode='r') as out:
            if 'cf_mean' not in out.dsets:
                msg = ('Could not find cf_mean (required) in file: {}'
                       .format(gen_fpath))
                logger.error(msg)
                raise OffshoreWindInputError(msg)

            meta = out.meta
            cf_mean = out['cf_mean']

        msg = ('Could not find "gid" column in source '
               'capacity factor meta data!')
        assert 'gid' in meta, msg

        # currently an assumption of sorted gids in the reV gen output
        msg = ('Source capacity factor meta data is not ordered!')
        assert list(meta['gid']) == sorted(list(meta['gid'])), msg

        if 'offshore' not in meta:
            e = ('Offshore module cannot run without "offshore" flag in meta '
                 'data of gen_fpath: {}'.format(gen_fpath))
            logger.error(e)
            raise KeyError(e)

        onshore_mask = meta['offshore'] == 0
        offshore_mask = meta['offshore'] == 1

        logger.info('Finished parsing reV gen output for resource gid '
                    '{} through {} with {} offshore points.'
                    .format(meta['gid'].values.min(),
                            meta['gid'].values.max(), offshore_mask.sum()))
        logger.info('Offshore capacity factor has min / median / mean / max: '
                    '{:.3f} / {:.3f} / {:.3f} / {:.3f}'
                    .format(cf_mean.min(), np.median(cf_mean),
                            np.mean(cf_mean), cf_mean.max()))

        return meta, onshore_mask, offshore_mask, cf_mean

    def _parse_offshore_data(self, offshore_fpath,
                             required_columns=('gid', 'config')):
        """Parse the offshore data file for offshore farm site data and coords.

        Parameters
        ----------
        offshore_fpath : str
            Full filepath to offshore wind farm data file.
        required_columns : tuple | list
            List of column names that must be in the offshore data in
            order to run the reV offshore module.

        Returns
        -------
        offshore_data : pd.DataFrame
            Dataframe of extracted offshore farm data. Each row is a farm and
            columns are farm data attributes.
        """

        offshore_data = pd.read_csv(offshore_fpath)

        if 'dist_l_to_ts' in offshore_data:
            if offshore_data['dist_l_to_ts'].sum() > 0:
                w = ('Possible incorrect Offshore data input! "dist_l_to_ts" '
                     '(distance land to transmission) input is non-zero. '
                     'Most reV runs set this to zero and input the cost '
                     'of transmission from landfall tie-in to '
                     'transmission feature in the supply curve module.')
                logger.warning(w)
                warn(w, OffshoreWindInputWarning)

        for c in required_columns:
            if c not in offshore_data:
                msg = ('Did not find required "{}" column in offshore_data!'
                       .format(c))
                logger.error(msg)
                raise KeyError(msg)

        available_gids = list(offshore_data['gid'].values)
        missing = set(self.offshore_res_gids) - set(available_gids)
        if any(missing):
            msg = ('The following gids were requested in the reV project '
                   'points input but were not available in the offshore data '
                   'input: {}'.format(missing))
            logger.error(msg)
            raise OffshoreWindInputError(msg)

        # only keep the offshore data corresponding to relevant project points
        mask = offshore_data['gid'].isin(self.offshore_res_gids)
        offshore_data = offshore_data[mask]
        offshore_data = offshore_data.sort_values('gid')
        offshore_data = offshore_data.reset_index(drop=True)

        return offshore_data

    def _parse_system_inputs(self):
        """Get the system inputs dict (SAM tech inputs) from project points.

        Returns
        -------
        system_inputs : pd.DataFrame
            DataFrame of SAM config inputs (columns) for every offshore
            resource gid (row). Index is resource gids and there is also
            a column "gid" with the copied gids.
        """

        system_inputs = {}

        for gid in self.offshore_res_gids:
            system_inputs[gid] = self._project_points[gid][1]

            if 'turbine_capacity' not in system_inputs[gid]:
                # convert from SAM kw powercurve to MW.
                arr = system_inputs[gid]['wind_turbine_powercurve_powerout']
                cap_kw = np.max(arr)
                cap_mw = cap_kw / 1000
                system_inputs[gid]['turbine_capacity'] = cap_mw

        system_inputs = pd.DataFrame(system_inputs).T
        system_inputs = system_inputs.sort_index()
        system_inputs['gid'] = system_inputs.index.values
        system_inputs.index.name = 'gid'

        return system_inputs

    def _preflight_checks(self):
        """Run some preflight checks on the offshore inputs"""
        sam_configs = {k: v for k, v in
                       self._project_points.sam_inputs.items()
                       if k in self._nrwal_configs}
        for cid, sys_in in sam_configs.items():
            if 'turbine_capacity' not in sys_in:
                msg = ('System input key "turbine_capacity" not found in '
                       'SAM system inputs for "{}". Calculating from turbine '
                       'power curves.'.format(cid))
                logger.warning(msg)
                warn(msg, OffshoreWindInputWarning)

            loss1 = sys_in.get('wind_farm_losses_percent', 0)
            loss2 = sys_in.get('turb_generic_loss', 0)
            if loss1 != 0 or loss2 != 0:
                msg = ('Wind farm loss for config "{}" is not 0. The offshore '
                       'module uses gross capacity factors from reV '
                       'generation and applies losses from the NRWAL equations'
                       .format(cid))
                logger.warning(msg)
                warn(msg, OffshoreWindInputWarning)

        available_ids = list(self._nrwal_configs.keys())
        requested_ids = list(self._offshore_data['config'].values)
        missing = set(requested_ids) - set(available_ids)
        if any(missing):
            msg = ('The following config ids were requested in the offshore '
                   'data input but were not available in the NRWAL config '
                   'input dict: {}'.format(missing))
            logger.error(msg)
            raise OffshoreWindInputError(msg)

        check_gid_order = (self._offshore_data['gid'].values
                           == self._system_inputs['gid'].values)
        msg = 'Offshore and system input dataframes had bad order'
        assert (check_gid_order).all(), msg

        if 'gcf' in self._offshore_data:
            msg = 'Offshore data input already had gross capacity factor!'
            logger.error(msg)
            raise OffshoreWindInputError(msg)
        self._offshore_data['gcf'] = self._cf_mean[self._offshore_mask]

        for config_id, nrwal_config in self._nrwal_configs.items():
            system_vars = [var for var in nrwal_config.required_inputs
                           if var not in self._offshore_data]
            missing_vars = [var for var in nrwal_config.required_inputs
                            if var not in self._offshore_data
                            and var not in self._system_inputs]

            if any(missing_vars):
                msg = ('Could not find required input variables {} '
                       'for NRWAL config "{}" in either the offshore '
                       'data or the SAM system data!'
                       .format(missing_vars, config_id))
                logger.error(msg)
                raise OffshoreWindInputError(msg)

            for var in system_vars:
                sys_data_arr = self._system_inputs[var].values
                self._offshore_data[var] = sys_data_arr

        missing = [c for c in self._offshore_meta_cols
                   if c not in self._offshore_data]
        if any(missing):
            msg = ('Could not find requested offshore pass through columns '
                   'in offshore input data: {}'.format(missing))
            logger.error(msg)
            raise OffshoreWindInputError(msg)

    @property
    def time_index(self):
        """Get the source time index."""
        if self._time_index is None:
            with Outputs(self._gen_fpath, mode='r') as out:
                self._time_index = out.time_index

        return self._time_index

    @property
    def meta_source_full(self):
        """Get the full meta data (onshore + offshore)"""
        return self._meta_source

    @property
    def meta_source_onshore(self):
        """Get the onshore only meta data."""
        return self._meta_source[self._onshore_mask]

    @property
    def meta_source_offshore(self):
        """Get the offshore only meta data."""
        return self._meta_source[self._offshore_mask]

    @property
    def meta_out(self):
        """Get the combined onshore and offshore meta data."""
        if self._meta_out is None:
            self._meta_out = self.meta_source_full.copy()
            for col in self._offshore_meta_cols:
                if col not in self._meta_out:
                    self._meta_out[col] = np.nan

                # note that this assumes that offshore data has been reduced
                # to only those rows with gids in meta_out and is sorted by gid
                data = self._offshore_data[col]
                self._meta_out.loc[self._offshore_mask, col] = data.values

        return self._meta_out

    @property
    def meta_out_onshore(self):
        """Get the onshore only meta data."""
        return self.meta_out[self._onshore_mask]

    @property
    def meta_out_offshore(self):
        """Get the output offshore meta data."""
        return self.meta_out[self._offshore_mask]

    @property
    def onshore_res_gids(self):
        """Get a list of resource gids for the onshore sites."""
        return self.meta_source_onshore['gid'].values.tolist()

    @property
    def offshore_res_gids(self):
        """Get a list of resource gids for the offshore sites."""
        return self.meta_source_offshore['gid'].values.tolist()

    @property
    def outputs(self):
        """Get a dict of offshore outputs"""
        return self._out

    def run_nrwal(self):
        """Run offshore analysis via the NRWAL analysis library"""
        from NRWAL import Equation
        for i, (cid, nrwal_config) in enumerate(self._nrwal_configs.items()):
            mask = self._offshore_data['config'].values == cid
            logger.info('Running offshore config {} of {}: "{}" and applying '
                        'to {} out of {} offshore gids'
                        .format(i + 1, len(self._nrwal_configs), cid,
                                mask.sum(), len(mask)))

            outs = nrwal_config.eval(inputs=self._offshore_data)

            # pylint: disable=C0201
            for name in self._out.keys():
                if name in outs:
                    value = outs[name]
                    self._out[name][mask] = value[mask]

                elif name in nrwal_config.keys():
                    value = nrwal_config[name]
                    if isinstance(value, Equation):
                        msg = ('Cannot retrieve Equation "{}" from NRWAL. '
                               'Must be a number!'.format(name))
                        assert not any(value.variables), msg
                        value = value.eval()
                    if np.issubdtype(type(value), np.number):
                        value *= np.ones(len(self._offshore_data))
                    if not isinstance(value, np.ndarray):
                        msg = ('NRWAL key "{}" returned bad type of "{}", '
                               'needs to be numeric or an output array.'
                               .format(name, type(value)))
                        logger.error(msg)
                        raise TypeError(msg)
                    self._out[name][mask] = value[mask]

                elif name not in self._offshore_data:
                    msg = ('Could not find "{}" in the output dict of NRWAL '
                           'config {}'.format(name, cid))
                    logger.error(msg)
                    raise KeyError(msg)

                logger.debug('NRWAL output "{}": {}'.format(name, value))

    def check_outputs(self):
        """Check the nrwal outputs for nan values and raise errors if found."""
        for name, arr in self._out.items():
            if np.isnan(arr).any():
                mask = np.isnan(arr)
                nan_meta = self.meta_out_offshore[mask]
                nan_gids = nan_meta['gid'].values
                msg = ('NaN values ({} out of {}) persist in offshore '
                       'output "{}"!'
                       .format(np.isnan(arr).sum(), len(arr), name))
                logger.error(msg)
                logger.error('This is the offshore meta that is causing NaN '
                             'outputs: {}'.format(nan_meta))
                logger.error('These are the resource gids causing NaN '
                             'outputs: {}'.format(nan_gids))
                raise ValueError(msg)

    def write_to_gen_fpath(self):
        """Save offshore outputs to input generation fpath file. This will
        overwrite data!"""

        loss_mult = 1 - self._out[self._loss_key]

        with Outputs(self._gen_fpath, 'a') as f:
            meta_attrs = f.get_attrs('meta')
            del f._h5['meta']
            f._set_meta('meta', self.meta_out, attrs=meta_attrs)

            lcoe = f['lcoe_fcr']
            lcoe[self._offshore_mask] = self._out[self._lcoe_key]
            f['lcoe_fcr'] = lcoe

            cf_mean = f['cf_mean']
            cf_mean[self._offshore_mask] *= loss_mult
            f['cf_mean'] = cf_mean

            if 'cf_profile' in f.dsets:
                profiles = f['cf_profile']
                profiles[:, self._offshore_mask] *= loss_mult
                f['cf_profile'] = profiles

            for key, arr in self._out.items():
                if key not in (self._lcoe_key, ):
                    if key not in f.dsets:
                        data = np.full(len(f.meta), np.nan).astype(np.float32)
                    else:
                        data = f[key]

                    data[self._offshore_mask] = arr
                    f._add_dset(key, data, np.float32,
                                attrs={'scale_factor': 1})

    @classmethod
    def run(cls, gen_fpath, offshore_fpath, sam_files, nrwal_configs,
            points, offshore_meta_cols=None, offshore_nrwal_keys=None,
            nrwal_lcoe_key='lcoe', nrwal_loss_key='total_losses'):
        """
        Parameters
        ----------
        gen_fpath : str
            Full filepath to reV gen h5 output file.
        offshore_fpath : str
            Full filepath to offshore wind farm data file. Needs "gid" and
            "config" columns matching the project points input.
        sam_files : dict
            Dictionary lookup of config_id values mapped to config filepaths.
            The same config_id values will be used from the nrwal_configs
            lookup input.
        nrwal_configs : dict
            Dictionary lookup of config_id values mapped to config filepaths.
            The same config_id values will be used from the sam_files lookup
            in project_points
        points : str
            reV project points to analyze. Has to be a string file path to a
            project points csv with "gid" and "config" columns. The config
            column maps to the sam_files and nrwal_configs inputs.
        offshore_meta_cols : list | tuple | None
            Column labels from offshore_fpath to pass through to the output
            meta data. None (default) will use class variable
            DEFAULT_META_COLS, and any additional cols requested here will be
            added to DEFAULT_META_COLS.
        offshore_nrwal_keys : list | tuple | None
            Equation labels from the NRWAL configs to pass through to the
            output h5 file. None will use class variable DEFAULT_NRWAL_KEYS,
            and any additional cols requested here will be added to
            DEFAULT_NRWAL_KEYS.
        nrwal_lcoe_key : str
            Key in the NRWAL config for final LCOE output value. Can be
            changed and runtime for different NRWAL configs using this kwarg.
        nrwal_loss_key : str
            Key in the NRWAL config for final capacity factor losses output
            value. Can be changed and runtime for different NRWAL configs
            using this kwarg.

        Returns
        -------
        offshore : Offshore
            Instantiated Offshore analysis object.
        """

        points_range = None
        pc = Gen.get_pc(points, points_range, sam_files, 'windpower')

        offshore = cls(gen_fpath, offshore_fpath, nrwal_configs,
                       pc.project_points,
                       offshore_meta_cols=offshore_meta_cols,
                       offshore_nrwal_keys=offshore_nrwal_keys,
                       nrwal_lcoe_key=nrwal_lcoe_key,
                       nrwal_loss_key=nrwal_loss_key)

        if any(offshore.offshore_res_gids):
            offshore.run_nrwal()
            offshore.check_outputs()
            offshore.write_to_gen_fpath()

        logger.info('Offshore wind gen/econ module complete!')

        return offshore
