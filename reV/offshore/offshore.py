# -*- coding: utf-8 -*-
"""
reV offshore wind farm aggregation  module. This module aggregates offshore
generation data from high res wind resource data to coarse wind farm sites
and then calculates the ORCA econ data.

Offshore resource / generation data refers to WTK 2km (fine resolution)
Offshore farms refer to ORCA data on 600MW wind farms (coarse resolution)
"""
from concurrent.futures import as_completed
import numpy as np
import os
import shutil
import pandas as pd
from scipy.spatial import cKDTree
import logging
from warnings import warn

from reV.generation.generation import Gen
from reV.handlers.collection import DatasetCollector
from reV.handlers.outputs import Outputs
from reV.offshore.orca import ORCA_LCOE
from reV.utilities.exceptions import (OffshoreWindInputWarning,
                                      NearestNeighborError)

from rex.utilities.execution import SpawnProcessPool

logger = logging.getLogger(__name__)


class Offshore:
    """Framework to handle offshore wind analysis."""

    # Default columns from the offshore wind farm data to join to the
    # offshore meta data
    DEFAULT_META_COLS = ('min_sub_tech', 'sub_type', 'array_cable_CAPEX',
                         'export_cable_CAPEX')

    def __init__(self, gen_fpath, offshore_fpath, project_points,
                 max_workers=None, offshore_gid_adder=1e7,
                 farm_gid_label='wfarm_id', small_farm_limit=7,
                 offshore_meta_cols=None):
        """
        Parameters
        ----------
        gen_fpath : str
            Full filepath to reV gen h5 output file.
        offshore_fpath : str
            Full filepath to offshore wind farm data file.
        project_points : reV.config.project_points.ProjectPoints
            Instantiated project points instance.
        max_workers : int | None
            Number of workers for process pool executor. 1 will run in serial.
        offshore_gid_adder : int | float
            The offshore Supply Curve gids will be set equal to the respective
            resource gids plus this number.
        farm_gid_label : str
            Label in offshore_fpath for the wind farm gid unique identifier.
        small_farm_limit : int
            Wind farms with less than this number of neighboring resource
            pixels will not be included in the output. Default is 7 based on
            median number of farm resource neighbors in a small test case.
        offshore_meta_cols : list | tuple | None
            Column labels from offshore_fpath to preserve in the output
            meta data. None will use class variable DEFAULT_META_COLS, and any
            additional requested cols will be added to DEFAULT_META_COLS.
        """

        self._gen_fpath = gen_fpath
        self._offshore_fpath = offshore_fpath
        self._project_points = project_points
        self._offshore_gid_adder = offshore_gid_adder
        self._meta_out_offshore = None
        self._time_index = None
        self._warned = False
        self._max_workers = max_workers
        self._farm_gid_label = farm_gid_label
        self._small_farm_limit = small_farm_limit

        if offshore_meta_cols is None:
            offshore_meta_cols = list(self.DEFAULT_META_COLS)
        else:
            offshore_meta_cols = list(offshore_meta_cols)
            offshore_meta_cols += list(self.DEFAULT_META_COLS)
            offshore_meta_cols = list(set(offshore_meta_cols))

        self._offshore_meta_cols = offshore_meta_cols

        self._meta_source, self._onshore_mask, self._offshore_mask = \
            self._parse_cf_meta(self._gen_fpath)

        self._offshore_data, self._farm_coords = \
            self._parse_offshore_fpath(self._offshore_fpath)

        self._d, self._i, self._d_lim = self._run_nn()

        self._out = self._init_offshore_out_arrays()

        logger.info('Initialized offshore wind farm aggregation module with '
                    '{} onshore resource points, {} offshore resource points, '
                    'and {} output wind farms.'
                    .format(len(self.meta_source_onshore),
                            len(self.meta_source_offshore),
                            len(self.meta_out_offshore)))

    def _init_offshore_out_arrays(self):
        """Get a dictionary of initialized output arrays for offshore outputs.

        Returns
        -------
        out_arrays : dict
            Dictionary of output arrays filled with zeros for offshore data.
            Has keys for all datasets present in gen_fpath.
        """

        out_arrays = {}

        with Outputs(self._gen_fpath, mode='r') as out:
            dsets = [d for d in out.datasets
                     if d not in ('time_index', 'meta')]

            for dset in dsets:
                shape = out.get_dset_properties(dset)[0]
                if len(shape) == 1:
                    dset_shape = (len(self.meta_out_offshore), )
                else:
                    dset_shape = (shape[0], len(self.meta_out_offshore))

                logger.debug('Initializing offshore output data array for '
                             '"{}" with shape {}.'.format(dset, dset_shape))
                out_arrays[dset] = np.zeros(dset_shape, dtype=np.float32)

        return out_arrays

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

        if 'offshore' not in meta:
            e = ('Offshore module cannot run without "offshore" flag in meta '
                 'data of gen_fpath: {}'.format(gen_fpath))
            logger.error(e)
            raise KeyError(e)

        onshore_mask = meta['offshore'] == 0
        offshore_mask = meta['offshore'] == 1

        return meta, onshore_mask, offshore_mask

    @staticmethod
    def _parse_offshore_fpath(offshore_fpath):
        """Parse the offshore data file for offshore farm site data and coords.

        Parameters
        ----------
        offshore_fpath : str
            Full filepath to offshore wind farm data file.

        Returns
        -------
        offshore_data : pd.DataFrame
            Dataframe of extracted offshore farm data. Each row is a farm and
            columns are farm data attributes.
        farm_coords : pd.DataFrame
            Latitude/longitude coordinates for each offshore farm.
        """

        offshore_data = pd.read_csv(offshore_fpath)

        lat_label = [c for c in offshore_data.columns
                     if c.lower().startswith('latitude')]
        lon_label = [c for c in offshore_data.columns
                     if c.lower().startswith('longitude')]

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

        return offshore_data, offshore_data[c_labels]

    def _run_nn(self):
        """Run a spatial NN on the offshore resource points and the offshore
        wind farm data.

        Returns
        -------
        d : np.ndarray
            Distance between offshore resource pixel and offshore wind farm.
        i : np.ndarray
            Offshore farm row numbers corresponding to resource pixels
            (length is number of offshore resource pixels in gen_fpath).
        d_lim : float
            Maximum distance limit between wind farm points and resouce pixels.
        """

        tree = cKDTree(self._farm_coords)  # pylint: disable=not-callable
        d, i = tree.query(self.meta_source_offshore[['latitude', 'longitude']])

        d_lim = 0
        if len(self._farm_coords) > 1:
            d_lim, _ = tree.query(self._farm_coords, k=2)
            d_lim = 0.5 * np.median(d_lim[:, 1])
            i[(d > d_lim)] = -1

        return d, i, d_lim

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
        if any(self.offshore_gids) and any(self.onshore_gids):
            meta = self.meta_out_onshore.append(self.meta_out_offshore,
                                                sort=False)
        elif any(self.offshore_gids):
            meta = self.meta_out_offshore
        elif any(self.onshore_gids):
            meta = self.meta_out_onshore

        return meta

    @property
    def meta_out_onshore(self):
        """Get the onshore only meta data."""
        meta_out_onshore = self._meta_source[self._onshore_mask].copy()
        meta_out_onshore['offshore_res_gids'] = '[-1]'

        return meta_out_onshore

    @property
    def meta_out_offshore(self):
        """Get the output offshore meta data.

        Returns
        -------
        meta_out_offshore : pd.DataFrame
            Offshore farm meta data. Offshore farms without resource
            neighbors are dropped.
        """

        if self._meta_out_offshore is None:
            self._meta_out_offshore = self._farm_coords.copy()

            new_offshore_gids = []
            new_agg_gids = []

            misc_cols = ['country', 'state', 'county', 'timezone']
            new_misc = {k: [] for k in misc_cols if k in self.meta_source_full}

            for i in self._offshore_data.index:
                farm_gid, res_gid = self._get_farm_gid(i)

                agg_gids = None
                misc = {k: None for k in new_misc.keys()}

                if res_gid is not None:
                    ilocs = np.where(self._i == i)[0]

                    if len(ilocs) > self._small_farm_limit:
                        meta_sub = self.meta_source_offshore.iloc[ilocs]
                        agg_gids = str(meta_sub['gid'].values.tolist())

                        mask = self.meta_source_offshore['gid'] == res_gid
                        for k in misc.keys():
                            misc[k] = self.meta_source_offshore.loc[mask, k]\
                                .values[0]

                new_offshore_gids.append(farm_gid)
                new_agg_gids.append(agg_gids)

                for k, v in misc.items():
                    new_misc[k].append(v)

            for k, v in new_misc.items():
                self._meta_out_offshore[k] = v

            self._meta_out_offshore['elevation'] = 0.0
            self._meta_out_offshore['offshore'] = 1
            self._meta_out_offshore['gid'] = new_offshore_gids
            self._meta_out_offshore['offshore_res_gids'] = new_agg_gids
            self._meta_out_offshore['reV_tech'] = 'offshore_wind'

            self._meta_out_offshore = self._meta_out_offshore.dropna(
                subset=['gid', 'offshore_res_gids'])

            # Index must not be re-ordered because it corresponds to index in
            # self._offshore_data
            self._meta_out_offshore = \
                self._meta_out_offshore.sort_values('gid')

            # add additional columns from the offshore input data to the meta
            if self._offshore_meta_cols is not None:
                missing = [c not in self._offshore_data
                           for c in self._offshore_meta_cols]
                if any(missing):
                    e = ('Could not find the requested columns {} in the '
                         'offshore data input. The following are available: '
                         '{}'.format(self._offshore_meta_cols,
                                     self._offshore_data.columns.values))
                    logger.error(e)
                    raise KeyError(e)
                meta = self._offshore_data[self._offshore_meta_cols]
                self._meta_out_offshore = self._meta_out_offshore.join(
                    meta, how='left')

        return self._meta_out_offshore

    @property
    def onshore_gids(self):
        """Get a list of gids for the onshore sites."""
        return self.meta_out_onshore['gid'].values.tolist()

    @property
    def offshore_gids(self):
        """Get a list of gids for the offshore sites."""
        return self.meta_out_offshore['gid'].values.tolist()

    @property
    def out(self):
        """Output data.

        Returns
        -------
        out : dict
            Output data keyed by reV dataset names. Each dataset will have a
            spatial dimension (for all the offshore wind farms) and maybe a
            time dimension if the dataset is profiles.
        """
        return self._out

    def save_output(self, fpath_out):
        """
        Save all onshore and offshore data to offshore output file

        Parameters
        ----------
        fpath_out : str
            Output filepath.
        """
        logger.info('Writing offshore output data to: {}'
                    .format(fpath_out))

        self._init_fout(fpath_out)
        self._collect_onshore(fpath_out)
        self._collect_offshore(fpath_out)

    def _init_fout(self, fpath_out):
        """
        Initialize the offshore aggregated output file and collect
        non-aggregated onshore data.

        Parameters
        ----------
        fpath_out : str
            Output filepath.
        """

        logger.debug('Initializing offshore output file: {}'
                     .format(fpath_out))
        with Outputs(self._gen_fpath, mode='r') as source:
            meta_attrs = source.get_attrs(dset='meta')
            ti_attrs = source.get_attrs(dset='time_index')

        with Outputs(fpath_out, mode='w') as out:
            out._set_meta('meta', self.meta_out, attrs=meta_attrs)
            out._set_time_index('time_index', self.time_index,
                                attrs=ti_attrs)

    def _collect_onshore(self, fpath_out):
        """Collect non-aggregated onshore data to initialized file.

        Parameters
        ----------
        fpath_out : str
            Output filepath.
        """

        with Outputs(self._gen_fpath, mode='r') as source:
            dsets = [d for d in source.datasets
                     if d not in ('meta', 'time_index')]

        if any(self.onshore_gids):
            for dset in dsets:
                logger.debug('Collecting onshore data for "{}"'
                             .format(dset))
                DatasetCollector.collect_dset(fpath_out, [self._gen_fpath],
                                              self.onshore_gids, dset)
        else:
            logger.debug('No onshore data in source file to collect.')
            for dset in dsets:
                logger.debug('Initializing offshore dataset "{}".'
                             .format(dset))
                DatasetCollector(fpath_out, [self._gen_fpath],
                                 self.offshore_gids, dset)

    def _collect_offshore(self, fpath_out):
        """Collect aggregated offshore data to initialized file.

        Parameters
        ----------
        fpath_out : str
            Output filepath.
        """

        if any(self.offshore_gids):
            offshore_bool = np.isin(self.meta_out['gid'].values,
                                    self.offshore_gids)
            offshore_locs = np.where(offshore_bool)[0]
            offshore_slice = slice(offshore_locs.min(),
                                   offshore_locs.max() + 1)

            with Outputs(self._gen_fpath, mode='r') as source:
                dsets = [d for d in source.datasets
                         if d not in ('meta', 'time_index')]

            with Outputs(fpath_out, mode='a') as out:
                shapes = {d: out.get_dset_properties(d)[0] for d in dsets}
                for dset in dsets:
                    logger.info('Writing offshore output data for "{}".'
                                .format(dset))
                    if len(shapes[dset]) == 1:
                        out[dset, offshore_slice] = self.out[dset]
                    else:
                        out[dset, :, offshore_slice] = self.out[dset]

    def move_input_file(self, sub_dir):
        """
        Move the generation input file to a sub dir (after offshore agg).

        Parameters
        ----------
        sub_dir : str | None
            Sub directory name to move chunks to. None to not move files.
        """
        if sub_dir is not None:
            base_dir, fn = os.path.split(self._gen_fpath)
            new_dir = os.path.join(base_dir, sub_dir)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            new_fpath = os.path.join(new_dir, fn)
            shutil.move(self._gen_fpath, new_fpath)

    @staticmethod
    def _get_farm_data(gen_fpath, meta, system_inputs, site_data, site_gid=0):
        """Get the offshore farm aggregated cf data and calculate LCOE.

        Parameters
        ----------
        gen_fpath : str
            Full filepath to reV gen h5 output file.
        meta : pd.DataFrame
            Offshore resource meta data for resource pixels belonging to the
            single wind farm. The meta index should correspond to the gids in
            the gen_fpath.
        system_inputs : dict
            Wind farm system inputs.
        site_data : dict
            Wind-farm site-specific data inputs.
        site_gid : int
            Optional site gid (farm index) for logging and debugging.

        Returns
        -------
        gen_data : dict
            Dictionary of all available generation datasets. Keys are reV gen
            output dataset names, values are spatial averages - scalar resource
            data (cf_mean) gets averaged to one offshore farm value (float),
            profiles (cf_profile) gets averaged to one offshore farm profile
            (1D arrays). Added ORCA lcoe as "lcoe_fcr" with wind farm site
            LCOE value with units: $/MWh.
        """

        gen_data = Offshore._get_farm_gen_data(gen_fpath, meta)
        cf = gen_data['cf_mean'].mean()

        if cf > 1:
            m = ('Offshore wind aggregated mean capacity factor ({}) for '
                 'wind farm gid {} is greater than 1, maybe the data is '
                 'still integer scaled.'.format(cf, site_gid))
            logger.warning(m)
            warn(m, OffshoreWindInputWarning)

        lcoe = Offshore._run_orca(cf, system_inputs, site_data,
                                  site_gid=site_gid)
        gen_data['lcoe_fcr'] = lcoe

        return gen_data

    @staticmethod
    def _get_farm_gen_data(gen_fpath, meta, ignore=('meta', 'time_index',
                                                    'lcoe_fcr')):
        """Get the aggregated generation data for a single wind farm.

        Parameters
        ----------
        gen_fpath : str
            Full filepath to reV gen h5 output file.
        meta : pd.DataFrame
            Offshore resource meta data for resource pixels belonging to the
            single wind farm. The meta index should correspond to the gids in
            the gen_fpath.
        ignore : list | tuple
            List of datasets to ignore and not retrieve.

        Returns
        -------
        gen_data : dict
            Dictionary of all available generation datasets. Keys are reV gen
            output dataset names, values are spatial averages - scalar resource
            data (cf_mean) gets averaged to one offshore farm value (float),
            profiles (cf_profile) gets averaged to one offshore farm profile
            (1D arrays).
        """

        gen_data = {}
        with Outputs(gen_fpath, mode='r', unscale=True) as out:

            dsets = [d for d in out.datasets if d not in ignore]

            if 'cf_mean' not in dsets:
                m = ('Offshore wind data aggregation needs cf_mean but reV '
                     'gen output file only had: {}'.format(out.datasets))
                logger.error(m)
                raise KeyError(m)

            for dset in dsets:
                shape = out.get_dset_properties(dset)[0]
                if len(shape) == 1:
                    gen_data[dset] = out[dset, meta.index.values].mean()
                else:
                    arr = out[dset, :, meta.index.values]
                    gen_data[dset] = arr.mean(axis=1)

        return gen_data

    @staticmethod
    def _run_orca(cf_mean, system_inputs, site_data, site_gid=0):
        """Run an ORCA LCOE compute for a wind farm.

        Parameters
        ----------
        cf_mean : float
            Annual mean capacity factor for wind farm site.
        system_inputs : dict
            Wind farm system inputs.
        site_data : dict
            Wind-farm site-specific data inputs.
        site_gid : int
            Optional site gid for logging and debugging.

        Results
        -------
        orca.lcoe : float
            Site LCOE value with units: $/MWh.
        """
        site_data['gcf'] = cf_mean
        orca = ORCA_LCOE(system_inputs, site_data, site_gid=site_gid)

        return orca.lcoe

    def _get_farm_gid(self, ifarm):
        """Get a unique resource gid for a wind farm.

        Parameters
        ----------
        ifarm : int
            Row number in offshore_data DataFrame for farm of interest.

        Returns
        -------
        farm_gid : int | None
            Unique GID for the offshore farm. This is the offshore
            gid adder plus the farm gid (from self._offshore_data).
            None will be returned if the farm is not close to any
            resource sites in gen_fpath.
        res_gid : int | None
            Resource gid of the closest resource pixel to ifarm. None if farm
            is not close to any resource sites in gen_fpath.
        """
        res_gid = None
        farm_gid = None

        if ifarm in self._i:
            inds = np.where(self._i == ifarm)[0]
            dists = self._d[inds]
            ind_min = inds[np.argmin(dists)]
            res_site = self.meta_source_offshore.iloc[ind_min]
            res_gid = res_site['gid']

            farm_gid = self._offshore_data.iloc[ifarm][self._farm_gid_label]
            farm_gid = int(self._offshore_gid_adder + farm_gid)

        return farm_gid, res_gid

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

    @staticmethod
    def _check_dist(meta_out_row, farm_data_row):
        """Check that the offshore meta data and farm input data match.

        Parameters
        ----------
        meta_out_row : pd.Series
            Output meta data for farm.
        farm_data_row : pd.Series
            Farm input data
        """

        lat_label = [c for c in farm_data_row.index
                     if c.lower().startswith('latitude')][0]
        lon_label = [c for c in farm_data_row.index
                     if c.lower().startswith('longitude')][0]

        dist = (meta_out_row[['latitude', 'longitude']]
                - farm_data_row[[lat_label, lon_label]]).sum()
        if dist > 0:
            m = ('Offshore farm NN failed, output meta:\n{}\nfarm data '
                 'input:\n{}'.format(meta_out_row, farm_data_row))
            logger.error(m)
            raise NearestNeighborError(m)

    def _check_sys_inputs(self, system_inputs, site_data):
        """Check system inputs and site data for duplicates and print warning.

        system_inputs : dict
            System (non-site-specific) inputs.
        site_data : dict
            Site specific inputs (will overwrite system_inputs)
        """
        overlap = [k for k in site_data.keys() if k in system_inputs]
        if any(overlap) and not self._warned:
            w = ('Offshore site inputs (from {}) will overwrite system '
                 'json inputs for the following columns: {}'
                 .format(os.path.basename(self._offshore_fpath), overlap))
            logger.warning(w)
            warn(w, OffshoreWindInputWarning)
            self._warned = True

    def _run_serial(self):
        """Run offshore gen aggregation and ORCA econ compute in serial."""

        for i, (ifarm, meta) in enumerate(self.meta_out_offshore.iterrows()):

            row = self._offshore_data.loc[ifarm, :]
            farm_gid, res_gid = self._get_farm_gid(ifarm)

            self._check_dist(meta, row)

            if farm_gid is not None:
                cf_ilocs = np.where(self._i == ifarm)[0]
                meta = self.meta_source_offshore.iloc[cf_ilocs]
                system_inputs = self._get_system_inputs(res_gid)
                site_data = row.to_dict()

                logger.debug('Running offshore gen aggregation and ORCA econ '
                             'compute for ifarm {}, farm gid {}, res gid {}'
                             .format(ifarm, farm_gid, res_gid))

                self._check_sys_inputs(system_inputs, site_data)

                gen_data = self._get_farm_data(self._gen_fpath, meta,
                                               system_inputs, site_data,
                                               site_gid=farm_gid)

                for k, v in gen_data.items():
                    if isinstance(v, (np.ndarray, list, tuple)):
                        self._out[k][:, i] = v
                    else:
                        self._out[k][i] = v

    def _run_parallel(self):
        """Run offshore gen aggregation and ORCA econ compute in parallel."""

        futures = {}
        loggers = [__name__, 'reV']
        with SpawnProcessPool(max_workers=self._max_workers,
                              loggers=loggers) as exe:

            iterator = self.meta_out_offshore.iterrows()
            for i, (ifarm, meta) in enumerate(iterator):

                row = self._offshore_data.loc[ifarm, :]
                farm_gid, res_gid = self._get_farm_gid(ifarm)

                self._check_dist(meta, row)

                if farm_gid is not None:
                    cf_ilocs = np.where(self._i == ifarm)[0]
                    meta = self.meta_source_offshore.iloc[cf_ilocs]
                    system_inputs = self._get_system_inputs(res_gid)
                    site_data = row.to_dict()

                    self._check_sys_inputs(system_inputs, site_data)

                    future = exe.submit(self._get_farm_data, self._gen_fpath,
                                        meta, system_inputs, site_data,
                                        site_gid=farm_gid)

                    futures[future] = i

            for fi, future in enumerate(as_completed(futures)):
                logger.info('Completed {} out of {} offshore compute futures.'
                            .format(fi + 1, len(futures)))
                i = futures[future]
                gen_data = future.result()
                for k, v in gen_data.items():
                    if isinstance(v, (np.ndarray, list, tuple)):
                        self._out[k][:, i] = v
                    else:
                        self._out[k][i] = v

    def _run(self):
        """Run offshore gen aggregation and ORCA econ compute"""
        if self._max_workers == 1:
            self._run_serial()
        else:
            self._run_parallel()

    @classmethod
    def run(cls, gen_fpath, offshore_fpath, points, sam_files, fpath_out=None,
            max_workers=None, offshore_gid_adder=1e7, small_farm_limit=7,
            farm_gid_label='wfarm_id', sub_dir='chunk_files'):
        """Run the offshore aggregation methods.

        Parameters
        ----------
        gen_fpath : str
            Full filepath to reV gen h5 output file.
        offshore_fpath : str
            Full filepath to offshore wind farm data file.
        points : slice | list | str | reV.config.project_points.PointsControl
            Slice specifying project points, or string pointing to a project
            points csv, or a fully instantiated PointsControl object.
        sam_files : dict | str | list
            Dict contains SAM input configuration ID(s) and file path(s).
            Keys are the SAM config ID(s), top level value is the SAM path.
            Can also be a single config file str. If it's a list, it is mapped
            to the sorted list of unique configs requested by points csv.
        fpath_out : str | NoneType
            Optional output filepath.
        max_workers : int | None
            Number of workers for process pool executor. 1 will run in serial.
        offshore_gid_adder : int | float
            The offshore Supply Curve gids will be set equal to the respective
            resource gids plus this number.
        small_farm_limit : int
            Wind farms with less than this number of neighboring resource
            pixels will not be included in the output. Default is 7 based on
            median number of farm resource neighbors in a small test case.
        farm_gid_label : str
            Label in offshore_fpath for the wind farm gid unique identifier.
        sub_dir : str | None
            Sub directory name to move chunks to. None to not move files.

        Returns
        -------
        offshore : Offshore
            Offshore aggregation object.
        """
        points_range = None
        pc = Gen.get_pc(points, points_range, sam_files, 'windpower',
                        sites_per_worker=100)
        offshore = cls(gen_fpath, offshore_fpath, pc.project_points,
                       offshore_gid_adder=offshore_gid_adder,
                       small_farm_limit=small_farm_limit,
                       farm_gid_label=farm_gid_label,
                       max_workers=max_workers)

        if any(offshore.offshore_gids):
            offshore._run()

        if fpath_out is not None:
            offshore.save_output(fpath_out)

        offshore.move_input_file(sub_dir)
        logger.info('Offshore wind gen/econ module complete!')

        return offshore
