# -*- coding: utf-8 -*-
"""
reV offshore wind farm aggregation  module. This module aggregates offshore
generation data from high res wind resource data to coarse wind farm sites
and then calculates the ORCA econ data.

Offshore resource / generation data refers to WTK 2km (fine resolution)
Offshore farms refer to ORCA data on 600MW wind farms (coarse resolution)
"""
import numpy as np
import pandas as pd
import logging
from warnings import warn

from NRWAL import NrwalConfig

from reV.handlers.outputs import Outputs
from reV.utilities.exceptions import (OffshoreWindInputWarning,
                                      OffshoreWindInputError)

from rex.utilities.utilities import get_lat_lon_cols

logger = logging.getLogger(__name__)


class Offshore:
    """Framework to handle offshore wind analysis."""

    # Default columns from the offshore wind farm data to join to the
    # offshore meta data
    DEFAULT_META_COLS = ('min_sub_tech', 'sub_type', 'array_cable_CAPEX',
                         'export_cable_CAPEX')

    def __init__(self, gen_fpath, offshore_fpath, nrwal_configs,
                 project_points, max_workers=None, offshore_meta_cols=None):
        """
        Parameters
        ----------
        gen_fpath : str
            Full filepath to reV gen h5 output file.
        offshore_fpath : str
            Full filepath to offshore wind farm data file.
        nrwal_configs : dict
            Dictionary lookup of config_id values mapped to config filepaths.
            The same config_id values will be used from the sam_files lookup
            in project_points
        project_points : reV.config.project_points.ProjectPoints
            Instantiated project points instance.
        max_workers : int | None
            Number of workers for process pool executor. 1 will run in serial.
        offshore_meta_cols : list | tuple | None
            Column labels from offshore_fpath to preserve in the output
            meta data. None will use class variable DEFAULT_META_COLS, and any
            additional requested cols will be added to DEFAULT_META_COLS.
        """
        log_versions(logger)
        self._gen_fpath = gen_fpath
        self._offshore_fpath = offshore_fpath
        self._project_points = project_points
        self._meta_out = None
        self._time_index = None
        self._max_workers = max_workers

        self._nrwal_configs = {k: NrwalConfig(v) for k, v in
                               nrwal_configs.items()}

        if offshore_meta_cols is None:
            offshore_meta_cols = list(self.DEFAULT_META_COLS)
        else:
            offshore_meta_cols = list(offshore_meta_cols)
            offshore_meta_cols += list(self.DEFAULT_META_COLS)
            offshore_meta_cols = list(set(offshore_meta_cols))

        self._offshore_meta_cols = offshore_meta_cols

        self._meta_source, self._onshore_mask, self._offshore_mask = \
            self._parse_cf_meta(self._gen_fpath)

        self._offshore_data, self._farm_coords = self._parse_offshore_data(
            self._offshore_fpath, self._project_points)

        self._system_inputs = {gid: self._get_system_inputs(gid)
                               for gid in self.offshore_res_gids}
        self._system_inputs = pd.DataFrame(self._system_inputs)
        print(self._system_inputs)

        logger.info('Initialized offshore wind farm aggregation module with '
                    '{} onshore resource points, {} offshore resource points.'
                    .format(len(self.meta_source_onshore),
                            len(self.meta_source_offshore)))

        self._out = {'lcoe_fcr': np.full(len(self._offshore_data), np.nan),
                     'losses': np.full(len(self._offshore_data), np.nan)}

    @staticmethod
    def _parse_cf_meta(gen_fpath):
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
        """

        with Outputs(gen_fpath, mode='r') as out:
            meta = out.meta

        msg = ('Could not find "gid" column in source '
               'capacity factor meta data!')
        assert 'gid' in meta, msg

        msg = ('Source capacity factor meta data is not ordered!')
        assert list(meta['gid']) == sorted(list(meta['gid'])), msg

        if 'offshore' not in meta:
            e = ('Offshore module cannot run without "offshore" flag in meta '
                 'data of gen_fpath: {}'.format(gen_fpath))
            logger.error(e)
            raise KeyError(e)

        onshore_mask = meta['offshore'] == 0
        offshore_mask = meta['offshore'] == 1

        return meta, onshore_mask, offshore_mask

    @staticmethod
    def _parse_offshore_data(offshore_fpath, project_points,
                              required_columns=('gid', 'config')):
        """Parse the offshore data file for offshore farm site data and coords.

        Parameters
        ----------
        offshore_fpath : str
            Full filepath to offshore wind farm data file.
        project_points : reV.config.project_points.ProjectPoints
            Instantiated project points instance.
        required_columns : tuple | list
            List of column names that must be in the offshore data in
            order to run the reV offshore module.

        Returns
        -------
        offshore_data : pd.DataFrame
            Dataframe of extracted offshore farm data. Each row is a farm and
            columns are farm data attributes.
        farm_coords : pd.DataFrame
            Latitude/longitude coordinates for each offshore farm.
        """

        offshore_data = pd.read_csv(offshore_fpath)
        lat_label, lon_label = get_lat_lon_cols(offshore_data)

        if len(lat_label) > 1 or len(lon_label) > 1:
            e = ('Found multiple lat/lon columns: {} {}'
                 .format(lat_label, lon_label))
            logger.error(e)
            raise KeyError(e)
        else:
            c_labels = [lat_label[0], lon_label[0]]

        if 'dist_l_to_ts' in offshore_data:
            if offshore_data['dist_l_to_ts'].sum() > 0:
                w = ('Possible incorrect ORCA input! "dist_l_to_ts" '
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

        requested_gids = list(project_points.gids)
        available_gids = list(offshore_data['gid'].values)
        missing = set(requested_gids) - set(available_gids)
        if any(missing):
            msg = ('The following gids were requested in the reV project '
                   'points input but were not available in the offshore data '
                   'input: {}'.format(missing))
            logger.error(msg)
            raise OffshoreWindInputError(msg)

        # only keep the offshore data corresponding to relevant project points
        mask = offshore_data['gid'].isin(project_points.gids)
        offshore_data = offshore_data[mask]

        return offshore_data, offshore_data[c_labels]

    def _get_system_inputs(self, res_gid):
        """Get the system inputs dict (SAM tech inputs) from project points.

        Parameters
        ----------
        res_gid : int
            WTK resource gid for wind farm (nearest neighbor).

        Returns
        -------
        system_inputs : dict
            Dictionary of SAM system inputs for wtk resource gid input.
        """
        system_inputs = self._project_points[res_gid][1]

        if 'turbine_capacity' not in system_inputs:
            # convert from SAM kw powercurve to MW.
            cap = np.max(system_inputs['wind_turbine_powercurve_powerout'])
            cap_mw = cap / 1000
            system_inputs['turbine_capacity'] = cap_mw
            m = ('Offshore wind farm system input key "turbine_capacity" not '
                 'specified for res_gid {}. Setting to 1/1000 the max of the '
                 'SAM power curve: {} MW'.format(res_gid, cap_mw))
            logger.warning(m)
            warn(m, OffshoreWindInputWarning)

        return system_inputs

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
            pass
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
        return self.meta_out_onshore['gid'].values.tolist()

    @property
    def offshore_res_gids(self):
        """Get a list of resource gids for the offshore sites."""
        return self.meta_out_offshore['gid'].values.tolist()

    def run(self):
        """Run offshore analysis"""
        available_ids = list(self._nrwal_configs.keys())
        requested_ids = list(self._offshore_data['config'].values)
        missing = set(requested_ids) - set(available_ids)
        if any(missing):
            msg = ('The following config ids were requested in the offshore '
                   'data input but were not available in the NRWAL config '
                   'input dict: {}'.format(missing))
            logger.error(msg)
            raise OffshoreWindInputError(msg)

        check_gid_order = (self._offshore_data['gid']
                           == self._system_inputs['gid'])
        msg = 'Offshore and system input dataframes had bad order'
        assert (check_gid_order).all(), msg

        for config_id in requested_ids:
            nrwal_config = self._nrwal_configs[config_id]
            for var in nrwal_config.required_inputs:
                if var not in self._offshore_data:
                    if var in self._system_inputs:
                        self._offshore_data[var] = self._system_inputs[var]
                    else:
                        msg = ('Could not find required input variable "{}" '
                               'for NRWAL config "{}" in either the offshore '
                               'data or the SAM system data!'
                               .format(var, config_id))
                        logger.error(msg)
                        raise OffshoreWindInputError(msg)

            outs = nrwal_config.eval(inputs=self._offshore_data)
            mask = self._offshore_data['config'].values == config_id

            for name in self._out.keys():
                msg = ('Could not find "{}" in the output dict of NRWAL '
                       'config {}'.format(name, config_id))
                assert name in outs, msg

                self._out[name][mask] = outs[name][mask]
