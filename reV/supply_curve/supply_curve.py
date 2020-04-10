# -*- coding: utf-8 -*-
"""
reV supply curve module
- Calculation of LCOT
- Supply Curve creation
"""
from copy import deepcopy
import os
import logging
import numpy as np
import pandas as pd
from warnings import warn

from reV.handlers.transmission import TransmissionCosts as TC
from reV.handlers.transmission import TransmissionFeatures as TF
from reV.utilities.execution import SpawnProcessPool
from reV.utilities.exceptions import SupplyCurveInputError, SupplyCurveError

logger = logging.getLogger(__name__)


class CompetitiveWindFarms:
    """
    Handle competitive wind farm exclusion during supply curve sorting
    """
    def __init__(self, wind_dirs, sc_points, n_dirs=2):
        """
        Parameters
        ----------
        wind_dirs : pandas.DataFrame | str
            path to .csv or reVX.wind_dirs.wind_dirs.WindDirs output with
            the neighboring supply curve point gids and power-rose value at
            each cardinal direction
        sc_points : pandas.DataFrame | str
            Supply curve point summary table
        n_dirs : int, optional
            Number of prominent directions to use, by default 2
        """
        self._wind_dirs = self._parse_wind_dirs(wind_dirs)

        self._sc_gids, self._sc_point_gids, self._mask = \
            self._parse_sc_points(sc_points)

        self._wind_dirs = self._wind_dirs.loc[self._sc_point_gids.index.values]
        self._upwind, self._downwind = self._get_neighbors(self._wind_dirs,
                                                           n_dirs=n_dirs)

    def __repr__(self):
        shape = self._upwind.shape
        msg = ("{} with {} sc_point_gids and {} prominent directions"
               .format(self.__class__.__name__, shape[0], shape[1]))

        return msg

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            msg = ("{} must be a tuple of form (source, gid) where source is: "
                   "'sc_gid', 'sc_point_gid',  or 'upwind', 'downwind'"
                   .format(keys))
            logger.error(msg)
            raise ValueError(msg)

        source, gid = keys
        if source == 'sc_gid':
            out = self._sc_gids.loc[gid].values
        elif source == 'sc_point_gid':
            out = self._sc_point_gids.loc[gid].values[0]
        elif source == 'upwind':
            out = self._upwind.loc[gid].values
        elif source == 'downwind':
            out = self._downwind.loc[gid].values
        else:
            msg = ("{} must be: 'sc_gid', 'sc_point_gid',  or 'upwind', "
                   "'downwind'".format(keys))
            logger.error(msg)
            raise ValueError(msg)

        return out

    @property
    def sc_point_gids(self):
        """
        Un-masked sc_point_gids

        Returns
        -------
        ndarray
        """
        pos = np.where(self._mask)[0]

        return self._sc_point_gids.index.values[pos]

    @property
    def sc_gids(self):
        """
        Un-masked sc_gids

        Returns
        -------
        ndarray
        """
        pos = np.where(self._mask)[0]
        sc_gids = np.concatenate(self._sc_point_gids.loc[pos].values.flatten())

        return sc_gids

    @staticmethod
    def _parse_wind_dirs(wind_dirs):
        """
        Parse prominent direction neighbors

        Parameters
        ----------
        wind_dirs : pandas.DataFrame | str
            Neighboring supply curve point gids and power-rose value at each
            cardinal direction

        Returns
        -------
        wind_dirs : pandas.DataFrame
            Neighboring supply curve point gids and power-rose value at each
            cardinal direction for each sc point gid
        """
        wind_dirs = SupplyCurve._load_table(wind_dirs)

        wind_dirs = wind_dirs.set_index('sc_point_gid')
        columns = [c for c in wind_dirs if c.endswith('_gid', '_pr')]
        wind_dirs = wind_dirs[columns]

        return wind_dirs

    @staticmethod
    def _parse_sc_points(sc_points):
        """
        Parse supply curve point summary table into sc_gid to sc_point_gid
        mapping and vis-versa.

        Parameters
        ----------
        sc_points : pandas.DataFrame | str
            Supply curve point summary table

        Returns
        -------
        sc_gids : pandas.DataFrame
            sc_gid to sc_point_gid mapping
        sc_point_gids : pandas.DataFrame
            sc_point_gid to sc_gid mapping
        mask : ndarray
            Mask array to mask excluded sc_point_gids
        """
        sc_points = SupplyCurve._load_table(sc_points)
        sc_points = sc_points[['sc_gid', 'sc_point_gid']]
        sc_gids = sc_points.set_index('sc_gid')

        sc_point_gids = \
            sc_points.groupby('sc_point_gid')['sc_gid'].unique().to_frame()

        mask = np.ones(int(1 + sc_points['sc_point_gid'].max()), dtype=bool)

        return sc_gids, sc_point_gids, mask

    @staticmethod
    def _get_neighbors(wind_dirs, n_dirs=2):
        """
        Parse prominent direction neighbors

        Parameters
        ----------
        wind_dirs : pandas.DataFrame | str
            Neighboring supply curve point gids and power-rose value at each
            cardinal direction for each available sc point gid
        n_dirs : int, optional
            Number of prominent directions to use, by default 2

        Returns
        -------
        upwind_gids : pandas.DataFrame
            Upwind neighbor gids for n prominent wind directions
        downwind_gids : pandas.DataFrame
            Downwind neighbor gids for n prominent wind directions
        """
        cols = [c for c in wind_dirs if c.endswith('_gid')]
        directions = [c.split('_')[0] for c in cols]
        upwind_gids = wind_dirs[cols].values

        cols = ['{}_pr'.format(d) for d in directions]
        neighbor_pr = wind_dirs[cols].values

        neighbors = np.argsort(neighbor_pr)[:, :n_dirs]
        upwind_gids = np.take_along_axis(upwind_gids, neighbors, axis=1)
        upwind_gids = pd.DataFrame(upwind_gids, index=wind_dirs.index.values,
                                   columns=[i for i in range(n_dirs)])

        downwind_map = {'N': 'S', 'NE': 'SW', 'E': 'W', 'SE': 'NW', 'S': 'N',
                        'SW': 'NE', 'W': 'E', 'NW': 'SE'}
        cols = ["{}_gid".format(downwind_map[d]) for d in directions]
        downwind_gids = wind_dirs[cols].values
        downwind_gids = np.take_along_axis(downwind_gids, neighbors, axis=1)
        downwind_gids = pd.DataFrame(downwind_gids,
                                     index=wind_dirs.index.values,
                                     columns=[i for i in range(n_dirs)])

        return upwind_gids, downwind_gids

    def remove_noncompetitive_farm(self, sc_points, sort_on='total_lcoe',
                                   downwind=False):
        """
        Remove neighboring sc points for given number of prominent wind
        directions

        Parameters
        ----------
        sc_points : pandas.DataFrame | str
            Supply curve point summary table
        sort_on : str, optional
            column to sort on before excluding neighbors,
            by default 'total_lcoe'
        downwind : bool, optional
            Flag to remove downwind neighbors as well as upwind neighbors,
            by default False

        Returns
        -------
        sc_points : pandas.DataFrame
            Updated supply curve points after removing non-competative
            wind farms
        """
        sc_points = SupplyCurve._load_table(sc_points)
        sc_points = sc_points.sort_values(sort_on)

        sc_point_gids = sc_points['sc_point_gid'].values.astype(int)

        for i in range(len(sc_points)):
            gid = sc_point_gids[i]
            if self._mask[gid]:
                upwind_gids = self['upwind', gid]
                for n in upwind_gids:
                    self._mask[n] = False

                if downwind:
                    downwind_gids = self['downwind', gid]
                    for n in downwind_gids:
                        self._mask[n] = False

        sc_gids = self.sc_gids
        mask = sc_points['sc_gid'].isin(sc_gids)

        return sc_points.loc[mask]

    @classmethod
    def run(cls, wind_dirs, sc_points, n_dirs=2, sort_on='total_lcoe',
            downwind=False, out_fpath=None):
        """
        Exclude given number of neighboring Supply Point gids based on most
        prominent wind directions

        Parameters
        ----------
        wind_dirs : pandas.DataFrame | str
            path to .csv or reVX.wind_dirs.wind_dirs.WindDirs output with
            the neighboring supply curve point gids and power-rose value at
            each cardinal direction
        sc_points : pandas.DataFrame | str
            Supply curve point summary table
        n_dirs : int, optional
            Number of prominent directions to use, by default 2
        sort_on : str, optional
            column to sort on before excluding neighbors,
            by default 'total_lcoe'
        downwind : bool, optional
            Flag to remove downwind neighbors as well as upwind neighbors,
            by default False
        out_fpath : str, optional
            Path to .csv file to save updated sc_points to,
            by default None

        Returns
        -------
        sc_points : pandas.DataFrame
            Updated supply curve points after removing non-competative
            wind farms
        """
        cwf = cls(wind_dirs, sc_points, n_dirs=n_dirs)
        sc_points = cwf.remove_noncompetitive_farm(sc_points, sort_on=sort_on,
                                                   downwind=downwind)

        if out_fpath is not None:
            sc_points.to_csv(out_fpath, index=False)

        return sc_points


class SupplyCurve:
    """
    Class to handle LCOT calcuation and SupplyCurve sorting
    """
    def __init__(self, sc_points, trans_table, fcr, sc_features=None,
                 transmission_costs=None, line_limited=False,
                 connectable=True, max_workers=None, consider_friction=True):
        """
        Parameters
        ----------
        sc_points : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supply curve
            point summary
        trans_table : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supply curve
            transmission mapping
        fcr : float
            Fixed charge rate, used to compute LCOT
        sc_features : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing additional supply
            curve features, e.g. transmission multipliers, regions
        transmission_costs : str | dict
            Transmission feature costs to use with TransmissionFeatures
            handler: line_tie_in_cost, line_cost, station_tie_in_cost,
            center_tie_in_cost, sink_tie_in_cost
        line_limited : bool
            Substation connection is limited by maximum capacity of the
            attached lines, legacy method
        connectable : bool
            Determine if connection is possible
        max_workers : int | NoneType
            Number of workers to use to compute lcot, if > 1 run in parallel.
            None uses all available cpu's.
        consider_friction : bool
            Flag to consider friction layer on LCOE.
        """

        logger.info('Supply curve points input: {}'.format(sc_points))
        logger.info('Transmission table input: {}'.format(trans_table))

        trans_costs = transmission_costs
        self._sc_points = self._parse_sc_points(sc_points,
                                                sc_features=sc_features)
        self._trans_table = self._parse_trans_table(trans_table)
        self._trans_table = self._merge_sc_trans_tables(self._sc_points,
                                                        self._trans_table)
        self._check_sc_trans_table(self._sc_points, self._trans_table)
        self._trans_table = self._add_trans_lcot(self._trans_table, fcr,
                                                 trans_costs=trans_costs,
                                                 line_limited=line_limited,
                                                 connectable=connectable,
                                                 max_workers=max_workers)
        self._trans_features = self._create_handler(self._trans_table,
                                                    trans_costs=trans_costs)

        self._consider_friction = consider_friction
        self._calculate_total_lcoe_friction()

        out = self._parse_sc_gids(self._trans_table)
        self._trans_table, self._sc_gids, self._mask = out

    def __repr__(self):
        msg = "{} with {} points".format(self.__class__.__name__, len(self))

        return msg

    def __len__(self):
        return len(self._sc_gids)

    def __getitem__(self, gid):
        if gid not in self._sc_gids:
            msg = "Invalid supply curve gid {}".format(gid)
            logger.error(msg)
            raise KeyError(msg)

        i = self._sc_gids.index(gid)

        return self._sc_points.iloc[i]

    @staticmethod
    def _load_table(table):
        """
        Extract features and their capacity from supply curve transmission
        mapping table

        Parameters
        ----------
        table : str | pd.DataFrame
            Path to .csv or .json or DataFrame to parse

        Returns
        -------
        table : pandas.DataFrame
            DataFrame extracted from file path
        """
        if isinstance(table, str):
            if table.endswith('.csv'):
                table = pd.read_csv(table)
            elif table.endswith('.json'):
                table = pd.read_json(table)
            else:
                raise ValueError('Cannot parse {}'.format(table))

        elif not isinstance(table, pd.DataFrame):
            raise ValueError("Table must be a .csv, .json, or "
                             "a pandas DataFrame")

        return table

    @staticmethod
    def _parse_sc_points(sc_points, sc_features=None):
        """
        Import supply curve point summary and add any additional features

        Parameters
        ----------
        sc_points : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supply curve
        sc_features : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing additional supply
            curve features, e.g. transmission multipliers, regions

        Returns
        -------
        sc_points : pandas.DataFrame
            DataFrame of supply curve point summary with additional features
            added if supplied
        """
        sc_points = SupplyCurve._load_table(sc_points)
        logger.debug('Supply curve points table imported with columns: {}'
                     .format(sc_points.columns.values.tolist()))

        if sc_features is not None:
            sc_features = SupplyCurve._load_table(sc_features)
            merge_cols = [c for c in sc_features
                          if c in sc_points]
            sc_points = sc_points.merge(sc_features, on=merge_cols, how='left')
            logger.debug('Adding Supply Curve Features table with columns: {}'
                         .format(sc_features.columns.values.tolist()))

        logger.debug('Final supply curve points table has columns: {}'
                     .format(sc_points.columns.values.tolist()))

        return sc_points

    @staticmethod
    def _parse_sc_gids(trans_table, gid_key='sc_gid'):
        """Filter the trans table, extract unique sc gids, make bool mask.

        Parameters
        ----------
        trans_table : pd.DataFrame
            reV Supply Curve table joined with transmission features table.
        gid_key : str
            Column label in trans_table containing the supply curve points
            primary key.

        Returns
        -------
        trans_table : pd.DataFrame
            Same as input but filtered to only include non-nan
            supply curve gids.
        sc_gids : list
            List of unique integer supply curve gids (non-nan)
        mask : np.ndarray
            Boolean array initialized as true. Length is equal to the maximum
            SC gid so that the SC gids can be used to index the mask directly.
        """

        filter_mask = ~pd.isna(trans_table[gid_key])
        trans_table = trans_table[filter_mask]
        sc_gids = list(np.sort(trans_table[gid_key].unique()))
        sc_gids = [int(gid) for gid in sc_gids]
        mask = np.ones((int(1 + max(sc_gids)), ), dtype=bool)

        return trans_table, sc_gids, mask

    @staticmethod
    def _create_handler(trans_table, trans_costs=None):
        """
        Create TransmissionFeatures handler from supply curve transmission
        mapping table.  Update connection costs if given.

        Parameters
        ----------
        trans_table : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supply curve
            transmission mapping
        trans_costs : str | dict
            Transmission feature costs to use with TransmissionFeatures
            handler: line_tie_in_cost, line_cost, station_tie_in_cost,
            center_tie_in_cost, sink_tie_in_cost

        Returns
        -------
        trans_features : TransmissionFeatures
            TransmissionFeatures or TransmissionCosts instance initilized
            with specified transmission costs
        """
        if trans_costs is not None:
            kwargs = TF._parse_dictionary(trans_costs)
        else:
            kwargs = {}

        trans_features = TF(trans_table, **kwargs)

        return trans_features

    @staticmethod
    def _get_merge_cols(columns):
        """
        Get columns with 'row' or 'col' in them to use for merging

        Parameters
        ----------
        columns : list
            Columns to search

        Returns
        -------
        merge_cols : list
            Columns to merge on
        """
        merge_cols = [c for c in columns if 'row' in c or 'col' in c]

        return sorted(merge_cols)

    @staticmethod
    def _compute_lcot(trans_table, fcr, trans_costs=None, max_workers=None,
                      connectable=True, line_limited=False):
        """
        Compute levelized cost of transmission for all combinations of
        supply curve points and tranmission features in trans_table

        Parameters
        ----------
        trans_table : pd.DataFrame
            Table mapping supply curve points to transmission features
            MUST contain supply curve point capacity
        fcr : float
            Fixed charge rate needed to compute LCOT
        trans_costs : str | dict
            Transmission feature costs to use with TransmissionFeatures
            handler: line_tie_in_cost, line_cost, station_tie_in_cost,
            center_tie_in_cost, sink_tie_in_cost
        max_workers : int | NoneType
            Number of workers to use to compute lcot, if > 1 run in parallel.
            None uses all available cpu's.
        connectable : bool
            Determine if connection is possible
        line_limited : bool
            Substation connection is limited by maximum capacity of the
            attached lines, legacy method

        Returns
        -------
        lcot : list
            Levelized cost of transmission for all supply curve -
            tranmission feature connections
        cost : list
            Capital cost of tramsmission for all supply curve - transmission
            feature connections
        """
        if 'capacity' not in trans_table:
            raise SupplyCurveInputError('Supply curve table must have '
                                        'supply curve point capacity '
                                        'to compute lcot')

        if trans_costs is not None:
            trans_costs = TF._parse_dictionary(trans_costs)
        else:
            trans_costs = {}

        if max_workers is None:
            max_workers = os.cpu_count()

        gid_mask = ~pd.isna(trans_table['sc_gid'])

        logger.info('Computing LCOT costs for all possible connections...')
        if max_workers > 1:
            groups = trans_table[gid_mask].groupby('sc_gid')
            with SpawnProcessPool(max_workers=max_workers) as exe:
                futures = []
                for sc_gid, sc_table in groups:
                    if connectable:
                        capacity = sc_table['capacity'].unique()
                        if len(capacity) == 1:
                            capacity = capacity[0]
                        else:
                            msg = ('Each supply curve point should only have '
                                   'a single capacity, but {} has {}'
                                   .format(sc_gid, capacity))
                            logger.error(msg)
                            raise RuntimeError(msg)
                    else:
                        capacity = None

                    futures.append(exe.submit(TC.feature_costs, sc_table,
                                              capacity=capacity,
                                              line_limited=line_limited,
                                              **trans_costs))

                cost = [future.result() for future in futures]
                cost = np.hstack(cost)
        else:
            feature = TC(trans_table, line_limited=line_limited,
                         **trans_costs)
            cost = []
            for _, row in trans_table[gid_mask].iterrows():
                if connectable:
                    capacity = row['capacity']
                else:
                    capacity = None

                tm = row.get('transmission_multiplier', 1)
                cost.append(feature.cost(row['trans_line_gid'], row['dist_mi'],
                                         capacity=capacity,
                                         transmission_multiplier=tm))

            cost = np.array(cost, dtype='float32')

        cf_mean_arr = trans_table.loc[gid_mask, 'mean_cf'].values
        lcot = (cost * fcr) / (cf_mean_arr * 8760)

        logger.info('LCOT cost calculation is complete.')

        return lcot, cost

    @staticmethod
    def _feature_capacity(trans_table, trans_costs=None):
        """
        Add the transmission connection feature capacity to the trans table.

        Parameters
        ----------
        trans_table : pd.DataFrame
            Table mapping supply curve points to transmission features
        trans_costs : str | dict
            Transmission feature costs to use with TransmissionFeatures
            handler: line_tie_in_cost, line_cost, station_tie_in_cost,
            center_tie_in_cost, sink_tie_in_cost

        Returns
        -------
        trans_table : pd.DataFrame
            Table mapping supply curve points to transmission features with
            'avail_cap' column.
        """
        avc = 0.1
        if trans_costs is not None:
            if 'available_capacity' in trans_costs:
                avc = trans_costs['available_capacity']

        feature_cap = TF.feature_capacity(trans_table, available_capacity=avc)
        dtype = trans_table['trans_line_gid'].dtype
        feature_cap['trans_line_gid'] = \
            feature_cap['trans_line_gid'].astype(dtype)
        trans_table = trans_table.merge(feature_cap, on='trans_line_gid')

        return trans_table

    @staticmethod
    def _parse_trans_table(trans_table):
        """
        Import transmission features table

        Parameters
        ----------
        trans_table : pd.DataFrame | str
            Table mapping supply curve points to transmission features
            (either str filepath to table file or pre-loaded dataframe).

        Returns
        -------
        trans_table : pd.DataFrame
            Loaded transmission feature table.
        """

        trans_table = SupplyCurve._load_table(trans_table)

        drop_cols = ['sc_point_gid', 'sc_gid', 'cap_left']
        for col in drop_cols:
            if col in trans_table:
                trans_table = trans_table.drop(col, axis=1)

        return trans_table

    @staticmethod
    def _check_sc_trans_table(sc_points, trans_table):
        """Run self checks on sc_points table and the merged trans_table

        Parameters
        ----------
        sc_points : pd.DataFrame
            Table of supply curve point summary
        trans_table : pd.DataFrame
            Table mapping supply curve points to transmission features
            (should already be merged with SC points).
        """
        sc_gids = set(sc_points['sc_gid'].unique())
        trans_sc_gids = set(trans_table['sc_gid'].unique())
        missing = sorted(list(sc_gids - trans_sc_gids))
        if any(missing):
            msg = ("There are {} Supply Curve points with missing "
                   "transmission mappings. Supply curve points with no "
                   "transmission features will not be connected! "
                   "Missing sc_gid's: {}"
                   .format(len(missing), missing))
            logger.warning(msg)
            warn(msg)

        if not any(trans_sc_gids) or not any(sc_gids):
            msg = ('Merging of sc points table and transmission features '
                   'table failed with {} original sc gids and {} transmission '
                   'sc gids after table merge.'
                   .format(len(sc_gids), len(trans_sc_gids)))
            logger.error(msg)
            raise SupplyCurveError(msg)

        logger.debug('There are {} original SC gids and {} sc gids in the '
                     'merged transmission table.'
                     .format(len(sc_gids), len(trans_sc_gids)))
        logger.debug('Transmission Table created with columns: {}'
                     .format(trans_table.columns.values.tolist()))

    @staticmethod
    def _merge_sc_trans_tables(sc_points, trans_table,
                               merge_cols=('capacity', 'sc_gid', 'mean_cf',
                                           'mean_lcoe')):
        """Merge the supply curve table with the transmission features table.

        Parameters
        ----------
        sc_points : pd.DataFrame
            Table of supply curve point summary
        trans_table : pd.DataFrame
            Table mapping supply curve points to transmission features
        merge_cols : tuple | list
            List of column from sc_points to merge into the trans table.

        Returns
        -------
        trans_table : pd.DataFrame
            Updated table mapping supply curve points to transmission features.
            This is performed by merging left with trans_table, so there may be
            rows with nan sc_gid.
        """

        if isinstance(merge_cols, tuple):
            merge_cols = list(merge_cols)

        if 'mean_lcoe_friction' in sc_points:
            merge_cols.append('mean_lcoe_friction')

        point_merge_cols = SupplyCurve._get_merge_cols(sc_points.columns)
        table_merge_cols = SupplyCurve._get_merge_cols(trans_table.columns)

        merge_cols = (point_merge_cols + merge_cols)
        if 'transmission_multiplier' in sc_points:
            merge_cols.append('transmission_multiplier')
            col = 'transmission_multiplier'
            sc_points.loc[:, col] = sc_points.loc[:, col].fillna(1)

        sc_cap = sc_points[merge_cols].copy()
        rename = {p: t for p, t in zip(point_merge_cols, table_merge_cols)}
        sc_cap = sc_cap.rename(columns=rename)

        logger.debug('Merging SC table and Trans Table on columns: {}'
                     .format(table_merge_cols))
        trans_table = trans_table.merge(sc_cap, on=table_merge_cols,
                                        how='left')

        return trans_table

    @staticmethod
    def _add_trans_lcot(trans_table, fcr, trans_costs=None,
                        line_limited=False, connectable=True,
                        max_workers=None):
        """Compute LCOT for possible connections and add to the trans_table

        Parameters
        ----------
        trans_table : pd.DataFrame
            Table mapping supply curve points to transmission features.
        fcr : float
            Fixed charge rate, used to compute LCOT
        trans_costs : str | dict | None
            Transmission feature costs to use with TransmissionFeatures
            handler: line_tie_in_cost, line_cost, station_tie_in_cost,
            center_tie_in_cost, sink_tie_in_cost
        line_limited : bool
            Substation connection is limited by maximum capacity of the
            attached lines, legacy method
        connectable : bool
            Determine if connection is possible
        max_workers : int | NoneType
            Number of workers to use to compute lcot, if > 1 run in parallel.
            None uses all available cpu's.

        Returns
        -------
        trans_table : pd.DataFrame
            Same as input table but with new columns for trans_cap_cost, lcot,
            and total_lcoe.
        """

        trans_table = SupplyCurve._feature_capacity(trans_table,
                                                    trans_costs=trans_costs)
        trans_table = trans_table.sort_values('sc_gid')
        lcot, cost = SupplyCurve._compute_lcot(trans_table, fcr,
                                               trans_costs=trans_costs,
                                               line_limited=line_limited,
                                               connectable=connectable,
                                               max_workers=max_workers)

        gid_mask = ~pd.isna(trans_table['sc_gid'])
        trans_table.loc[gid_mask, 'trans_cap_cost'] = cost
        trans_table.loc[gid_mask, 'lcot'] = lcot
        trans_table['total_lcoe'] = (trans_table['lcot']
                                     + trans_table['mean_lcoe'])

        return trans_table

    def _calculate_total_lcoe_friction(self):
        """Look for site mean LCOE with friction in the trans table and if
        found make a total LCOE column with friction."""

        if ('mean_lcoe_friction' in self._trans_table
                and self._consider_friction):
            lcoe_friction = (self._trans_table['lcot']
                             + self._trans_table['mean_lcoe_friction'])
            self._trans_table['total_lcoe_friction'] = lcoe_friction
            logger.info('Found mean LCOE with friction. Adding key '
                        '"total_lcoe_friction" to trans table.')

    def _competitive_wind_farms(self, wind_dirs, sc_gid, downwind=False):
        """
        Exclude non-competitive wind farms for given sc_gid

        Parameters
        ----------
        wind_dirs : CompetitiveWindFarms
            Pre-initilized CompetitiveWindFarms instance
        sc_gid : int
            Supply curve gid to exclude non-competitive wind farms around
        downwind : bool, optional
            Flag to remove downwind neighbors as well as upwind neighbors,
            by default False

        Returns
        -------
        wind_dirs : CompetitiveWindFarms
            updated CompetitiveWindFarms instance
        """
        gid = wind_dirs['sc_gid', sc_gid]
        if wind_dirs._mask[gid]:
            upwind_gids = wind_dirs['upwind', gid]
            for n in upwind_gids:
                wind_dirs._mask[n] = False
                sc_gids = wind_dirs['sc_point_gids', n]
                for sc_id in sc_gids:
                    self._mask[sc_id] = False

        if downwind:
            downwind_gids = wind_dirs['downwind', gid]
            for n in downwind_gids:
                wind_dirs._mask[n] = False
                sc_gids = wind_dirs['sc_point_gids', n]
                for sc_id in sc_gids:
                    self._mask[sc_id] = False

        return wind_dirs

    def _full_sort(self, trans_table, wind_dirs, total_lcoe_fric,
                   sort_on='total_lcoe',
                   columns=('trans_gid', 'trans_capacity', 'trans_type',
                            'trans_cap_cost', 'dist_mi', 'lcot', 'total_lcoe'),
                   downwind=False):
        """
        [summary]

        Parameters
        ----------
        trans_table : pandas.DataFrame
            Supply Curve Tranmission table to sort on
        wind_dirs : CompetitiveWindFarms
            Pre-initilized CompetitiveWindFarms instance
        total_lcoe_fric : ndarray
            Vector of lcoe friction values
        sort_on : str, optional
            Column label to sort the Supply Curve table on. This affects the
            build priority - connections with the lowest value in this column
            will be built first, by default 'total_lcoe'
        columns : tuple, optional
            Columns to preserve in output connections dataframe,
            by default ('trans_gid', 'trans_capacity', 'trans_type',
                        'trans_cap_cost', 'dist_mi', 'lcot', 'total_lcoe')
        downwind : bool, optional
            Flag to remove downwind neighbors as well as upwind neighbors,
            by default False

        Returns
        -------
        supply_curve : pandas.DataFrame
            Updated sc_points table with transmission connections, LCOT
            and LCOE+LCOT based on full supply curve connections
        """
        init_list = [np.nan] * int(1 + np.max(self._sc_gids))
        conn_lists = {k: deepcopy(init_list) for k in columns}

        trans_sc_gids = trans_table['sc_gid'].values.astype(int)
        trans_gids = trans_table['trans_line_gid'].values
        trans_cap = trans_table['avail_cap'].values
        capacities = trans_table['capacity'].values
        categories = trans_table['category'].values
        dists = trans_table['dist_mi'].values
        trans_cap_costs = trans_table['trans_cap_cost'].values
        lcots = trans_table['lcot'].values
        total_lcoes = trans_table['total_lcoe'].values

        connected = 0
        progress = 0
        for i in range(len(trans_table)):
            sc_gid = trans_sc_gids[i]
            if self._mask[sc_gid]:
                trans_gid = trans_gids[i]
                connect = self._trans_features.connect(trans_gid,
                                                       capacities[i])
                if connect:
                    connected += 1
                    self._mask[sc_gid] = False

                    conn_lists['trans_gid'][sc_gid] = trans_gid
                    conn_lists['trans_capacity'][sc_gid] = trans_cap[i]
                    conn_lists['trans_type'][sc_gid] = categories[i]
                    conn_lists['trans_cap_cost'][sc_gid] = trans_cap_costs[i]
                    conn_lists['dist_mi'][sc_gid] = dists[i]
                    conn_lists['lcot'][sc_gid] = lcots[i]
                    conn_lists['total_lcoe'][sc_gid] = total_lcoes[i]

                    if total_lcoe_fric is not None:
                        conn_lists['total_lcoe_friction'][sc_gid] = \
                            total_lcoe_fric[i]

                    current_prog = connected // (len(self) / 100)
                    if current_prog > progress:
                        progress = current_prog
                        logger.info('{} % of supply curve points connected'
                                    .format(progress))

                    if wind_dirs is not None:
                        wind_dirs = \
                            self._competitive_wind_farms(wind_dirs, sc_gid,
                                                         downwind=downwind)

        index = range(0, int(1 + np.max(self._sc_gids)))
        connections = pd.DataFrame(conn_lists, index=index)
        connections.index.name = 'sc_gid'
        connections = connections.dropna(subset=[sort_on])
        connections = connections[columns]
        connections = connections.reset_index()

        if connected != len(self):
            msg = ("{} supply curve points were not connected to tranmission! "
                   "Unconnected sc_gid's: {}"
                   .format(len(self) - connected,
                           np.where(self._mask[self._sc_gids])[0].tolist()))
            logger.warning(msg)
            warn(msg)

        supply_curve = self._sc_points.merge(connections, on='sc_gid')
        if wind_dirs is not None:
            sc_gids = wind_dirs.sc_gids
            mask = supply_curve['sc_gid'].isin(sc_gids)
            supply_curve = supply_curve.loc[mask]

        return supply_curve

    def full_sort(self, trans_table=None, sort_on='total_lcoe',
                  columns=('trans_gid', 'trans_capacity', 'trans_type',
                           'trans_cap_cost', 'dist_mi', 'lcot', 'total_lcoe'),
                  wind_dirs=None, n_dirs=2, downwind=False):
        """
        run supply curve sorting in serial

        Parameters
        ----------
        trans_table : pandas.DataFrame | NoneType
            Supply Curve Tranmission table to sort on
            If none use self._trans_table
        sort_on : str
            Column label to sort the Supply Curve table on. This affects the
            build priority - connections with the lowest value in this column
            will be built first.
        columns : list | tuple
            Columns to preserve in output connections dataframe.
        wind_dirs : pandas.DataFrame | str
            path to .csv or reVX.wind_dirs.wind_dirs.WindDirs output with
            the neighboring supply curve point gids and power-rose value at
            each cardinal direction
        n_dirs : int, optional
            Number of prominent directions to use, by default 2
        downwind : bool, optional
            Flag to remove downwind neighbors as well as upwind neighbors,
            by default False

        Returns
        -------
        supply_curve : pandas.DataFrame
            Updated sc_points table with transmission connections, LCOT
            and LCOE+LCOT based on full supply curve connections
        """
        if trans_table is None:
            trans_table = self._trans_table

        if isinstance(columns, tuple):
            columns = list(columns)

        pos = trans_table['lcot'].isnull()
        trans_table = trans_table.loc[~pos].sort_values(sort_on)

        total_lcoe_fric = None
        if self._consider_friction and 'mean_lcoe_friction' in trans_table:
            columns.append('total_lcoe_friction')
            total_lcoe_fric = trans_table['total_lcoe_friction'].values

        if wind_dirs is not None:
            wind_dirs = CompetitiveWindFarms(wind_dirs, self._sc_points,
                                             n_dirs=n_dirs)

        supply_curve = self._full_sort(trans_table, wind_dirs, total_lcoe_fric,
                                       sort_on=sort_on, columns=columns,
                                       downwind=downwind)

        return supply_curve

    def simple_sort(self, trans_table=None, sort_on='total_lcoe',
                    columns=('trans_gid', 'trans_type', 'lcot', 'total_lcoe',
                             'trans_cap_cost'),
                    wind_dirs=None, n_dirs=2, downwind=False):
        """
        Run simple supply curve sorting that does not take into account
        available capacity

        Parameters
        ----------
        trans_table : pandas.DataFrame | NoneType
            Supply Curve Tranmission table to sort on
            If none use self._trans_table
        sort_on : str
            Column label to sort the Supply Curve table on. This affects the
            build priority - connections with the lowest value in this column
            will be built first.
        columns : list | tuple
            Columns to preserve in output connections dataframe.
        wind_dirs : pandas.DataFrame | str
            path to .csv or reVX.wind_dirs.wind_dirs.WindDirs output with
            the neighboring supply curve point gids and power-rose value at
            each cardinal direction
        n_dirs : int, optional
            Number of prominent directions to use, by default 2
        downwind : bool, optional
            Flag to remove downwind neighbors as well as upwind neighbors

        Returns
        -------
        supply_curve : pandas.DataFrame
            Updated sc_points table with transmission connections, LCOT
            and LCOE+LCOT based on simple supply curve connections
        """
        if trans_table is None:
            trans_table = self._trans_table

        if isinstance(columns, tuple):
            columns = list(columns)

        if self._consider_friction and 'total_lcoe_friction' in trans_table:
            columns.append('total_lcoe_friction')

        connections = trans_table.sort_values(sort_on).groupby('sc_gid')
        connections = connections.first()
        rename = {'trans_line_gid': 'trans_gid',
                  'category': 'trans_type'}
        connections = connections.rename(columns=rename)
        connections = connections[columns].reset_index()

        supply_curve = self._sc_points.merge(connections, on='sc_gid')
        if wind_dirs is not None:
            supply_curve = CompetitiveWindFarms.run(wind_dirs,
                                                    supply_curve,
                                                    n_dirs=n_dirs,
                                                    sort_on=sort_on,
                                                    downwind=downwind)

        return supply_curve

    @classmethod
    def full(cls, sc_points, trans_table, fcr, sc_features=None,
             transmission_costs=None, line_limited=False, sort_on='total_lcoe',
             columns=('trans_gid', 'trans_capacity', 'trans_type',
                      'trans_cap_cost', 'dist_mi', 'lcot', 'total_lcoe'),
             max_workers=None, wind_dirs=None, n_dirs=2, downwind=False):
        """
        Run full supply curve taking into account available capacity of
        tranmission features when making connections.

        Parameters
        ----------
        sc_points : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supplcy curve
            point summary
        trans_table : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supply curve
            transmission mapping
        fcr : float
            Fixed charge rate, used to compute LCOT
        sc_features : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing additional supply
            curve features, e.g. transmission multipliers, regions
        transmission_costs : str | dict
            Transmission feature costs to use with TransmissionFeatures
            handler: line_tie_in_cost, line_cost, station_tie_in_cost,
            center_tie_in_cost, sink_tie_in_cost
        line_limited : bool
            Substation connection is limited by maximum capacity of the
            attached lines, legacy method
        sort_on : str
            Column label to sort the Supply Curve table on. This affects the
            build priority - connections with the lowest value in this column
            will be built first.
        columns : list | tuple
            Columns to preserve in output supply curve dataframe.
        max_workers : int | NoneType
            Number of workers to use to compute lcot, if > 1 run in parallel.
            None uses all available cpu's.
        wind_dirs : pandas.DataFrame | str
            path to .csv or reVX.wind_dirs.wind_dirs.WindDirs output with
            the neighboring supply curve point gids and power-rose value at
            each cardinal direction
        n_dirs : int, optional
            Number of prominent directions to use, by default 2
        downwind : bool, optional
            Flag to remove downwind neighbors as well as upwind neighbors

        Returns
        -------
        supply_curve : pandas.DataFrame
            Updated sc_points table with transmission connections, LCOT
            and LCOE+LCOT
        """
        sc = cls(sc_points, trans_table, fcr, sc_features=sc_features,
                 transmission_costs=transmission_costs,
                 line_limited=line_limited, max_workers=max_workers)
        supply_curve = sc.full_sort(sort_on=sort_on, columns=columns,
                                    wind_dirs=wind_dirs, n_dirs=n_dirs,
                                    downwind=downwind)

        return supply_curve

    @classmethod
    def simple(cls, sc_points, trans_table, fcr, sc_features=None,
               transmission_costs=None, sort_on='total_lcoe',
               columns=('trans_gid', 'trans_type', 'lcot', 'total_lcoe',
                        'trans_cap_cost'),
               max_workers=None, wind_dirs=None, n_dirs=2, downwind=False):
        """
        Run simple supply curve by connecting to the cheapest tranmission
        feature.

        Parameters
        ----------
        sc_points : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supplcy curve
            point summary
        trans_table : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supply curve
            transmission mapping
        fcr : float
            Fixed charge rate, used to compute LCOT
        sc_features : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing additional supply
            curve features, e.g. transmission multipliers, regions
        transmission_costs : str | dict
            Transmission feature costs to use with TransmissionFeatures
            handler: line_tie_in_cost, line_cost, station_tie_in_cost,
            center_tie_in_cost, sink_tie_in_cost
        sort_on : str
            Column label to sort the Supply Curve table on. This affects the
            build priority - connections with the lowest value in this column
            will be built first.
        columns : list | tuple
            Columns to preserve in output supply curve dataframe.
        max_workers : int | NoneType
            Number of workers to use to compute lcot, if > 1 run in parallel.
            None uses all available cpu's.
        wind_dirs : pandas.DataFrame | str
            path to .csv or reVX.wind_dirs.wind_dirs.WindDirs output with
            the neighboring supply curve point gids and power-rose value at
            each cardinal direction
        n_dirs : int, optional
            Number of prominent directions to use, by default 2
        downwind : bool, optional
            Flag to remove downwind neighbors as well as upwind neighbors

        Returns
        -------
        supply_curve : pandas.DataFrame
            Updated sc_points table with transmission connections, LCOT
            and LCOE+LCOT
        """
        sc = cls(sc_points, trans_table, fcr, sc_features=sc_features,
                 transmission_costs=transmission_costs, connectable=False,
                 max_workers=max_workers)
        supply_curve = sc.simple_sort(sort_on=sort_on, columns=columns,
                                      wind_dirs=wind_dirs, n_dirs=n_dirs,
                                      downwind=downwind)

        return supply_curve
