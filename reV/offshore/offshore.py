# -*- coding: utf-8 -*-
"""
reV offshore wind module. This module aggregates offshore generation data
from high res wind resource data to coarse wind farm sites and then
calculates ORCA econ data.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import logging
from warnings import warn

from reV.generation.generation import Gen
from reV.handlers.outputs import Outputs
from reV.offshore.orca import ORCA_LCOE
from reV.utilities.exceptions import OffshoreWindInputWarning


logger = logging.getLogger(__name__)


class Offshore:
    """Framework to handle offshore wind analysis."""

    def __init__(self, cf_file, offshore_file, project_points,
                 offshore_gid_adder=1e7):
        """
        Parameters
        ----------
        cf_file : str
            Full filepath to reV gen h5 output file.
        offshore_file : str
            Full filepath to offshore wind farm data file.
        project_points : reV.config.project_points.ProjectPoints
            Instantiated project points instance.
        offshore_gid_adder : int | float
            The offshore Supply Curve gids will be set equal to the respective
            resource gids plus this number.
        """

        self._cf_file = cf_file
        self._offshore_file = offshore_file
        self._project_points = project_points
        self._offshore_gid_adder = offshore_gid_adder
        self._meta_out_offshore = None

        self._meta_source, self._onshore_mask, self._offshore_mask = \
            self._parse_cf_meta(self._cf_file)

        self._offshore_data, self._farm_coords = \
            self._parse_offshore_file(self._offshore_file)

        self._d, self._i = self._run_nn()

        self._out = self._init_out_arrays()

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
            new_timezones = []

            for i in self._offshore_data.index:
                res_gid, farm_gid = self._get_farm_gid(i)

                timezone = None
                if (res_gid is not None
                        and 'timezone' in self.meta_source_offshore):
                    mask = self.meta_source_offshore['gid'] == res_gid
                    timezone = self.meta_source_offshore.loc[mask, 'timezone']

                new_timezones.append(timezone)
                new_offshore_gids.append(farm_gid)

            self._meta_out_offshore['elevation'] = 0.0
            self._meta_out_offshore['timezone'] = new_timezones
            self._meta_out_offshore['gid'] = new_offshore_gids
            self._meta_out_offshore['reV_tech'] = 'offshore_wind'

            self._meta_out_offshore = self._meta_out_offshore.dropna(
                subset=['gid'])

        return self._meta_out_offshore

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

    def _init_out_arrays(self):
        """Get a dictionary of initialized output arrays for offshore outputs.

        Returns
        -------
        out_arrays : dict
            Dictionary of output arrays filled with zeros for offshore data.
            Has keys for all datasets present in cf_file.
        """

        out_arrays = {}

        with Outputs(self._cf_file, mode='r') as out:
            dsets = [d for d in out.dsets if d not in ('time_index', 'meta')]

            for dset in dsets:
                shape = out.get_dset_properties(dset)[0]
                if len(shape) == 1:
                    dset_shape = (len(self.meta_out_offshore), )
                else:
                    dset_shape = (shape[0], len(self.meta_out_offshore))

                out_arrays[dset] = np.zeros(dset_shape, dtype=np.float32)

        return out_arrays

    @staticmethod
    def _parse_cf_meta(cf_file):
        """Parse cf meta dataframe and get masks for onshore/offshore points.

        Parameters
        ----------
        cf_file : str
            Full filepath to reV gen h5 output file.

        Returns
        -------
        meta : pd.DataFrame
            Full meta data from cf_file with "offshore" column.
        onshore_mask : pd.Series
            Boolean series indicating where onshore sites are.
        offshore_mask : pd.Series
            Boolean series indicating where offshore sites are.
        """

        with Outputs(cf_file, mode='r') as out:
            meta = out.meta
        if 'offshore' not in meta:
            e = ('Offshore module cannot run without "offshore" flag in meta '
                 'data of cf_file: {}'.format(cf_file))
            logger.error(e)
            raise KeyError(e)

        onshore_mask = (meta['offshore'] == 0)
        offshore_mask = (meta['offshore'] == 1)

        return meta, onshore_mask, offshore_mask

    @staticmethod
    def _parse_offshore_file(offshore_file):
        """Parse the offshore data file for offshore farm site data and coords.

        Parameters
        ----------
        offshore_file : str
            Full filepath to offshore wind farm data file.

        Returns
        -------
        offshore_data : pd.DataFrame
            Dataframe of extracted offshore farm data. Each row is a farm and
            columns are farm data attributes.
        farm_coords : pd.DataFrame
            Latitude/longitude coordinates for each offshore farm.
        """

        offshore_data = pd.read_csv(offshore_file)

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
            (length is number of offshore resource pixels in cf_file).
        """

        tree = cKDTree(self._farm_coords)
        d, i = tree.query(self.meta_source_offshore[['latitude', 'longitude']])

        if len(self._farm_coords) > 1:
            d_lim, _ = tree.query(self._farm_coords, k=2)
            d_lim = np.max(d_lim[:, 1])
            i[(d > d_lim)] = -1

        return d, i

    @staticmethod
    def _get_farm_data(cf_file, meta, system_inputs, site_data):
        """Get the offshore farm aggregated cf data and calculate LCOE.

        Parameters
        ----------
        cf_file : str
            Full filepath to reV gen h5 output file.
        meta : pd.DataFrame
            Offshore resource meta data for resource pixels belonging to the
            single wind farm. The meta index should correspond to the gids in
            the cf_file.
        system_inputs : dict
            Wind farm system inputs.
        site_data : dict
            Wind-farm site-specific data inputs.

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

        gen_data = Offshore._get_farm_gen_data(cf_file, meta)
        cf = gen_data['cf_mean'].mean()

        if cf > 1:
            m = ('Offshore wind aggregated mean capacity factor ({}) is '
                 'greater than 1, maybe the data is still integer scaled.'
                 .format(cf))
            logger.warning(m)
            warn(m, OffshoreWindInputWarning)

        lcoe = Offshore._run_orca(cf, system_inputs, site_data)
        gen_data['lcoe_fcr'] = lcoe
        return gen_data

    @staticmethod
    def _get_farm_gen_data(cf_file, meta, ignore=('meta', 'time_index',
                                                  'lcoe_fcr')):
        """Get the aggregated generation data for a single wind farm.

        Parameters
        ----------
        cf_file : str
            Full filepath to reV gen h5 output file.
        meta : pd.DataFrame
            Offshore resource meta data for resource pixels belonging to the
            single wind farm. The meta index should correspond to the gids in
            the cf_file.
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
        with Outputs(cf_file, mode='r', unscale=True) as out:

            dsets = [d for d in out.dsets if d not in ignore]

            if 'cf_mean' not in dsets:
                m = ('Offshore wind data aggregation needs cf_mean but reV '
                     'gen output file only had: {}'.format(out.dsets))
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
    def _run_orca(cf_mean, system_inputs, site_data):
        """Run an ORCA LCOE compute for a wind farm.

        Parameters
        ----------
        cf_mean : float
            Annual mean capacity factor for wind farm site.
        system_inputs : dict
            Wind farm system inputs.
        site_data : dict
            Wind-farm site-specific data inputs.

        Results
        -------
        orca.lcoe : float
            Site LCOE value with units: $/MWh.
        """

        site_data['gcf'] = cf_mean
        orca = ORCA_LCOE(system_inputs, site_data)
        return orca.lcoe

    def _get_farm_gid(self, ifarm):
        """Get a unique resource gid for a wind farm.

        Parameters
        ----------
        ifarm : int
            Row number in offshore_data DataFrame for farm of interest.

        Returns
        -------
        res_gid : int | None
            Resource gid of the closest resource pixel to ifarm. None if farm
            is not close to any resource sites in cf_file.
        farm_gid : int | None
            Unique resource GID for the offshore farm. This is the offshore
            gid adder plus the closest resource gid. None will be returned if
            the farm is not close to any resource sites in cf_file.
        """
        res_gid = None
        farm_gid = None

        if ifarm in self._i:
            inds = np.where(self._i == ifarm)[0]
            dists = self._d[inds]
            ind_min = inds[np.argmin(dists)]
            res_site = self.meta_source_offshore.iloc[ind_min]
            res_gid = res_site['gid']
            farm_gid = int(self._offshore_gid_adder + res_gid)

        return res_gid, farm_gid

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
        return system_inputs

    def _run_serial(self):
        """Run offshore compute in serial.
        """

        for ifarm, row in self._offshore_data.iterrows():
            farm_gid, res_gid = self._get_farm_gid(ifarm)
            if farm_gid is not None:
                cf_ilocs = np.where(self._i == ifarm)[0]
                meta = self.meta_source_offshore.iloc[cf_ilocs]
                system_inputs = self._get_system_inputs(res_gid)
                site_data = row.to_dict()

                gen_data = self._get_farm_data(self._cf_file, meta,
                                               system_inputs, site_data)

                for k, v in gen_data.items():
                    if isinstance(v, (float, int)):
                        self._out[k][ifarm] = v
                    else:
                        self._out[k][:, ifarm] = v

    @classmethod
    def run(cls, cf_file, offshore_file, points, sam_files,
            offshore_gid_adder=1e7):
        """Run the offshore aggregation methods.

        Parameters
        ----------
        cf_file : str
            Full filepath to reV gen h5 output file.
        offshore_file : str
            Full filepath to offshore wind farm data file.
        points : slice | list | str | reV.config.project_points.PointsControl
            Slice specifying project points, or string pointing to a project
            points csv, or a fully instantiated PointsControl object.
        sam_files : dict | str | list
            Dict contains SAM input configuration ID(s) and file path(s).
            Keys are the SAM config ID(s), top level value is the SAM path.
            Can also be a single config file str. If it's a list, it is mapped
            to the sorted list of unique configs requested by points csv.
        offshore_gid_adder : int | float
            The offshore Supply Curve gids will be set equal to the respective
            resource gids plus this number.

        Returns
        -------
        offshore : Offshore
            Offshore aggregation object.
        """
        points_range = None
        pc = Gen.get_pc(points, points_range, sam_files, 'wind',
                        sites_per_worker=100)
        offshore = cls(cf_file, offshore_file, pc.project_points,
                       offshore_gid_adder)
        return offshore
