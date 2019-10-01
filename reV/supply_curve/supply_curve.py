# -*- coding: utf-8 -*-
"""
reV supply curve module
- Calculation of LCOT
- Supply Curve creation
"""
import concurrent.futures as cf
import logging
import numpy as np
import pandas as pd
from warnings import warn

from reV.handlers.transmission import TransmissionCosts as TC
from reV.handlers.transmission import TransmissionFeatures as TF
from reV.utilities.exceptions import SupplyCurveInputError

logger = logging.getLogger(__name__)


class SupplyCurve:
    """
    Class to handle LCOT calcuation and SupplyCurve sorting
    """
    def __init__(self, sc_points, trans_table, fcr, sc_features=None,
                 transmission_costs=None, **kwargs):
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
            handler
        kwargs : dict
            Internal kwargs for _parse_trans_table to compute LCOT
        """
        trans_costs = transmission_costs
        self._sc_points = self._parse_sc_points(sc_points,
                                                sc_features=sc_features)
        self._trans_table = self._parse_trans_table(self._sc_points,
                                                    trans_table, fcr,
                                                    trans_costs=trans_costs,
                                                    **kwargs)
        self._trans_features = self._create_handler(self._trans_table,
                                                    trans_costs=trans_costs)

        self._sc_gids = list(np.sort(self._trans_table['sc_gid'].unique()))
        self._mask = np.ones((len(self._sc_gids), ), dtype=bool)

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
        table : str
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
        if sc_features is not None:
            sc_features = SupplyCurve._load_table(sc_features)
            merge_cols = [c for c in sc_features
                          if c in sc_points]
            sc_points = sc_points.merge(sc_features, on=merge_cols, how='left')

        return sc_points

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
            handler

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
    def _compute_lcot(trans_table, fcr, trans_costs=None, max_workers=1,
                      connectable=True, **kwargs):
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
            handler
        max_workers : int | NoneType
            Number of workers to use to compute lcot, if > 1 run in parallel
        connectable : bool
            Determine if connection is possible
        kwargs : dict
            kwargs for feature.cost

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
        logger.info('Computing LCOT costs for all possible connections...')
        if max_workers > 1:
            if trans_costs is not None:
                kwargs.update(trans_costs)

            feature_cap = TF.feature_capacity(trans_table, **kwargs)
            dtype = trans_table['trans_line_gid'].dtype
            feature_cap['trans_line_gid'] = \
                feature_cap['trans_line_gid'].astype(dtype)
            trans_table = trans_table.merge(feature_cap, on='trans_line_gid')
            groups = trans_table.sort_values('sc_gid').groupby('sc_gid')
            with cf.ProcessPoolExecutor(max_workers=max_workers) as exe:
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
                                              capacity=capacity, **kwargs))

                cost = [future.result() for future in futures]
                cost = np.hstack(cost)
        else:
            feature = SupplyCurve._create_handler(trans_table,
                                                  costs=trans_costs)
            cost = []
            for _, row in trans_table.iterrows():
                if connectable:
                    capacity = row['capacity']
                else:
                    capacity = None

                tm = row.get('transmission_multiplier', 1)
                cost.append(feature.cost(row['trans_line_gid'], row['dist_mi'],
                                         capacity=capacity,
                                         transmission_multiplier=tm, **kwargs))

            cost = np.array(cost, dtype='float32')

        lcot = (cost * fcr) / (trans_table['mean_cf'].values * 8760)

        logger.info('LCOT cost calculation is complete.')

        return lcot, cost

    @staticmethod
    def _parse_trans_table(sc_points, trans_table, fcr, **kwargs):
        """
        Import supply curve table, add in supply curve point capacity

        Parameters
        ----------
        sc_points : pd.DataFrame
            Table of supply curve point summary
        trans_table : pd.DataFrame
            Table mapping supply curve points to transmission features
        fcr : float
            Fixed charge rate, used to compute LCOT
        kwargs : dict
            Internal kwargs for _parse_trans_table to compute LCOT

        Returns
        -------
        trans_table : pd.DataFrame
            Updated table mapping supply curve points to transmission features
        """
        trans_table = SupplyCurve._load_table(trans_table)

        drop_cols = ['sc_point_gid', 'sc_gid']
        for col in drop_cols:
            if col in trans_table:
                trans_table = trans_table.drop(col, axis=1)

        point_merge_cols = SupplyCurve._get_merge_cols(sc_points.columns)
        table_merge_cols = SupplyCurve._get_merge_cols(trans_table.columns)

        merge_cols = (point_merge_cols
                      + ['capacity', 'sc_gid', 'mean_cf', 'mean_lcoe'])
        if 'transmission_multiplier' in sc_points:
            merge_cols.append('transmission_multiplier')
            col = 'transmission_multiplier'
            sc_points.loc[:, col] = sc_points.loc[:, col].fillna(1)

        sc_cap = sc_points[merge_cols].copy()
        rename = {p: t for p, t in zip(point_merge_cols, table_merge_cols)}
        sc_cap = sc_cap.rename(columns=rename)

        sc_gids = len(sc_cap)
        trans_sc_gids = len(trans_table[table_merge_cols].drop_duplicates())
        if sc_gids != trans_sc_gids:
            msg = ("The number of supply Curve points ({}) and transmission "
                   "to supply curve point mappings ({}) do not match!"
                   .format(sc_gids, trans_sc_gids))
            logger.warning(msg)
            warn(msg)

        trans_table = trans_table.merge(sc_cap, on=table_merge_cols,
                                        how='inner').sort_values('sc_gid')
        lcot, cost = SupplyCurve._compute_lcot(trans_table, fcr, **kwargs)
        trans_table['trans_cap_cost'] = cost
        trans_table['lcot'] = lcot
        trans_table['total_lcoe'] = (trans_table['lcot']
                                     + trans_table['mean_lcoe'])

        return trans_table

    def full_sort(self, trans_table=None):
        """
        run supply curve sorting in serial

        Parameters
        ----------
        trans_table : pandas.DataFrame | NoneType
            Supply Curve Tranmission table to sort on
            If none use self._trans_table
        kwargs : dict
            Kwargs to compute lcot

        Returns
        -------
        connections : pandas.DataFrame
            DataFrame with Supply Curve connections
        """
        if trans_table is None:
            trans_table = self._trans_table

        columns = ['trans_gid', 'trans_type', 'lcot', 'total_lcoe']
        connections = pd.DataFrame(columns=columns, index=self._sc_gids)
        connections.index.name = 'sc_gid'

        pos = trans_table['lcot'].isnull()
        trans_table = trans_table.loc[~pos].sort_values('total_lcoe')

        sc_gids = trans_table['sc_gid'].values
        trans_gids = trans_table['trans_line_gid'].values
        capacities = trans_table['capacity'].values
        categories = trans_table['category'].values
        dists = trans_table['dist_mi'].values
        trans_cap_costs = trans_table['trans_cap_cost'].values
        lcots = trans_table['lcot'].values
        total_lcoes = trans_table['total_lcoe'].values

        progress = 0
        for i in range(len(trans_table)):
            sc_gid = sc_gids[i]
            i_mask = self._sc_gids.index(sc_gid)
            if self._mask[i_mask]:
                trans_gid = trans_gids[i]
                connect = self._trans_features.connect(trans_gid,
                                                       capacities[i])
                if connect:
                    self._mask[i_mask] = False
                    connections.at[sc_gid, 'trans_gid'] = trans_gid
                    connections.at[sc_gid, 'trans_type'] = categories[i]
                    connections.at[sc_gid, 'dist_mi'] = dists[i]
                    connections.at[sc_gid, 'trans_cap_cost'] = \
                        trans_cap_costs[i]
                    connections.at[sc_gid, 'lcot'] = lcots[i]
                    connections.at[sc_gid, 'total_lcoe'] = total_lcoes[i]

                    current_prog = np.sum(~self._mask) // (len(self) / 100)
                    if current_prog > progress:
                        progress = current_prog
                        logger.info('{} % of supply curve points connected'
                                    .format(progress))

        if np.any(self._mask):
            msg = ("{} supply curve points were not connected to tranmission!"
                   .format(np.sum(self._mask)))
            logger.warning(msg)
            warn(msg)

        return connections.reset_index()

    def simple_sort(self, trans_table=None):
        """
        Run simple supply curve sorting that does not take into account
        available capacity

        Parameters
        ----------
        trans_table : pandas.DataFrame | NoneType
            Supply Curve Tranmission table to sort on
            If none use self._trans_table
        kwargs : dict
            Kwargs to compute lcot

        Returns
        -------
        connections : pandas.DataFrame
            DataFrame with simple Supply Curve connections
        """
        if trans_table is None:
            trans_table = self._trans_table

        connections = trans_table.sort_values('total_lcoe').groupby('sc_gid')
        columns = ['trans_line_gid', 'category', 'lcot', 'total_lcoe',
                   'trans_cap_cost']
        connections = connections.first()[columns]
        rename = {'trans_line_gid': 'trans_gid',
                  'category': 'trans_type'}
        connections = connections.rename(columns=rename)

        return connections.reset_index()

    @classmethod
    def full(cls, sc_points, trans_table, fcr, sc_features=None,
             transmission_costs=None, **kwargs):
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
            handler
        kwargs : dict
            Internal kwargs for computing LCOT

        Returns
        -------
        supply_curve : pandas.DataFrame
            Updated sc_points table with transmission connections, LCOT
            and LCOE+LCOT
        """
        sc = cls(sc_points, trans_table, fcr, sc_features=sc_features,
                 transmission_costs=transmission_costs, **kwargs)
        connections = sc.full_sort()
        supply_curve = sc._sc_points.merge(connections, on='sc_gid')
        return supply_curve

    @classmethod
    def simple(cls, sc_points, trans_table, fcr, sc_features=None,
               transmission_costs=None, **kwargs):
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
            handler
        kwargs : dict
            Internal kwargs for computing LCOT

        Returns
        -------
        supply_curve : pandas.DataFrame
            Updated sc_points table with transmission connections, LCOT
            and LCOE+LCOT
        """
        sc = cls(sc_points, trans_table, fcr, sc_features=sc_features,
                 transmission_costs=transmission_costs, connectable=False,
                 **kwargs)
        connections = sc.simple_sort()
        supply_curve = sc._sc_points.merge(connections, on='sc_gid')
        return supply_curve
