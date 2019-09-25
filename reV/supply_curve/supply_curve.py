# -*- coding: utf-8 -*-
"""
reV supply curve module
- Calculation of LCOT
- Supply Curve creation
"""
import concurrent.futures as cf
import json
import numpy as np
import os
import pandas as pd

from reV.handlers.transmission import TransmissionFeatures
from reV.utilities.exceptions import SupplyCurveInputError


class SupplyCurve:
    """
    Class to handle LCOT calcuation and SupplyCurve sorting
    """
    def __init__(self, sc_points, sc_table, fcr, sc_features=None,
                 transmission_costs=None, **kwargs):
        """
        Parameters
        ----------
        sc_points : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supply curve
            point summary
        sc_table : str | pandas.DataFrame
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
            Internal kwargs for _parse_sc_table to compute LCOT
        """
        self._sc_points = self._parse_sc_points(sc_points,
                                                sc_features=sc_features)
        self._sc_table = self._parse_sc_table(self._sc_points, sc_table, fcr,
                                              **kwargs)
        self._trans_features = self._create_handler(self._sc_table,
                                                    costs=transmission_costs)

        self._sc_gids = list(np.sort(self._sc_table['sc_gid'].unique()))
        self._mask = np.ones((len(self._sc_gids), ), dtype=bool)

    @staticmethod
    def _parse_table(table):
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
        sc_points = SupplyCurve._parse_table(sc_points)
        if sc_features is not None:
            sc_features = SupplyCurve._parse_table(sc_features)
            merge_cols = [c for c in sc_features
                          if c in sc_points]
            sc_points = sc_points.merge(sc_features, on=merge_cols, how='left')

        return sc_points

    @staticmethod
    def _create_handler(sc_table, costs=None):
        """
        Create TransmissionFeatures handler from supply curve transmission
        mapping table.  Update connection costs if given.

        Parameters
        ----------
        sc_table : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supply curve
            transmission mapping
        costs : str | dict
            Transmission feature costs to use with TransmissionFeatures
            handler
        """
        if costs is not None:
            if os.path.isfile(costs):
                with open(costs, 'r') as f:
                    kwargs = json.load(f)
            else:
                kwargs = json.loads(costs)
        else:
            kwargs = {}

        trans_features = TransmissionFeatures(sc_table, **kwargs)

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
    def _compute_lcot(sc_table, fcr, max_workers=1, connectable=True,
                      **kwargs):
        """
        Compute levelized cost of transmission for all combinations of
        supply curve points and tranmission features in sc_table

        Parameters
        ----------
        sc_table : pd.DataFrame
            Table mapping supply curve points to transmission features
            MUST contain supply curve point capacity
        fcr : float
            Fixed charge rate needed to compute LCOT
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
        if 'capacity' not in sc_table:
            raise SupplyCurveInputError('Supply curve table must have '
                                        'supply curve point capacity '
                                        'to compute lcot')

        feature = TransmissionFeatures(sc_table)
        if max_workers > 1:
            with cf.ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures = []
                for _, row in sc_table.iterrows():
                    if connectable:
                        capacity = row['capacity']
                    else:
                        capacity = None

                    tm = row.get('transmission_multiplier', 1)
                    futures.append(exe.submit(feature.cost, row['trans_gid'],
                                              row['dist_mi'],
                                              capacity=capacity,
                                              transmission_multiplier=tm,
                                              **kwargs))

                lcot = [future.result() for future in futures]
        else:
            cost = []
            for _, row in sc_table.iterrows():
                if connectable:
                    capacity = row['capacity']
                else:
                    capacity = None

                tm = row.get('transmission_multiplier', 1)
                cost.append(feature.cost(row['trans_line_gid'], row['dist_mi'],
                                         capacity=capacity,
                                         transmission_multiplier=tm, **kwargs))

        lcot = ((np.array(cost, dtype=np.float) * fcr)
                / (sc_table['mean_cf'].values * 8760))

        return lcot, cost

    @staticmethod
    def _parse_sc_table(sc_points, sc_table, fcr, **kwargs):
        """
        Import supply curve table, add in supply curve point capacity

        Parameters
        ----------
        sc_points : pd.DataFrame
            Table of supply curve point summary
        sc_table : pd.DataFrame
            Table mapping supply curve points to transmission features
        fcr : float
            Fixed charge rate, used to compute LCOT
        kwargs : dict
            Internal kwargs for _parse_sc_table to compute LCOT

        Returns
        -------
        sc_table : pd.DataFrame
            Updated table mapping supply curve points to transmission features
        """
        sc_table = SupplyCurve._parse_table(sc_table)
        point_merge_cols = SupplyCurve._get_merge_cols(sc_points.columns)
        table_merge_cols = SupplyCurve._get_merge_cols(sc_table.columns)

        merge_cols = (point_merge_cols
                      + ['capacity', 'sc_gid', 'mean_cf', 'mean_lcoe'])
        if 'transmission_multiplier' in sc_points:
            merge_cols.append('transmission_multiplier')

        sc_cap = sc_points[merge_cols].copy()
        rename = {p: t for p, t in zip(point_merge_cols, table_merge_cols)}
        sc_cap = sc_cap.rename(columns=rename)

        sc_table = sc_table.merge(sc_cap, on=table_merge_cols, how='inner')
        lcot, cost = SupplyCurve._compute_lcot(sc_table, fcr, **kwargs)
        sc_table['trans_cap_cost'] = cost
        sc_table['lcot'] = lcot
        sc_table['total_lcoe'] = sc_table['lcot'] + sc_table['mean_lcoe']

        return sc_table

    def full_sort(self, sc_table=None):
        """
        run supply curve sorting in serial

        Parameters
        ----------
        sc_table : pandas.DataFrame | NoneType
            Supply Curve Tranmission table to sort on
            If none use self._sc_table
        kwargs : dict
            Kwargs to compute lcot

        Returns
        -------
        connections : pandas.DataFrame
            DataFrame with Supply Curve connections
        """
        if sc_table is None:
            sc_table = self._sc_table

        columns = ['trans_gid', 'trans_type', 'lcot', 'total_lcoe']
        connections = pd.DataFrame(columns=columns, index=self._sc_gids)
        connections.index.name = 'sc_gid'

        pos = sc_table['lcot'].isnull()
        sc_table = sc_table.loc[~pos].sort_values('total_lcoe')

        sc_gids = sc_table['sc_gid'].values
        trans_gids = sc_table['trans_line_gid'].values
        capacities = sc_table['capacity'].values
        categories = sc_table['category'].values
        dists = sc_table['dist_mi'].values
        trans_cap_costs = sc_table['trans_cap_cost'].values
        lcots = sc_table['lcot'].values
        total_lcoes = sc_table['total_lcoe'].values

        for i in range(len(sc_table)):
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

        return connections.reset_index()

    def simple_sort(self, sc_table=None):
        """
        Run simple supply curve sorting that does not take into account
        available capacity

        Parameters
        ----------
        sc_table : pandas.DataFrame | NoneType
            Supply Curve Tranmission table to sort on
            If none use self._sc_table
        kwargs : dict
            Kwargs to compute lcot

        Returns
        -------
        connections : pandas.DataFrame
            DataFrame with simple Supply Curve connections
        """
        if sc_table is None:
            sc_table = self._sc_table

        connections = sc_table.sort_values('total_lcoe').groupby('sc_gid')
        columns = ['trans_line_gid', 'category', 'lcot', 'total_lcoe',
                   'trans_cap_cost']
        connections = connections.first()[columns]
        rename = {'trans_line_gid': 'trans_gid',
                  'category': 'trans_type'}
        connections = connections.rename(columns=rename)

        return connections.reset_index()

    @classmethod
    def full(cls, sc_points, sc_table, fcr, sc_features=None,
             transmission_costs=None, **kwargs):
        """
        Run full supply curve taking into account available capacity of
        tranmission features when making connections.

        Parameters
        ----------
        sc_points : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supplcy curve
            point summary
        sc_table : str | pandas.DataFrame
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
        sc = cls(sc_points, sc_table, fcr, sc_features=sc_features,
                 transmission_costs=transmission_costs, **kwargs)
        connections = sc.full_sort()
        supply_curve = sc._sc_points.merge(connections, on='sc_gid')
        return supply_curve

    @classmethod
    def simple(cls, sc_points, sc_table, fcr, sc_features=None,
               transmission_costs=None, **kwargs):
        """
        Run simple supply curve by connecting to the cheapest tranmission
        feature.

        Parameters
        ----------
        sc_points : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supplcy curve
            point summary
        sc_table : str | pandas.DataFrame
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
        sc = cls(sc_points, sc_table, fcr, sc_features=sc_features,
                 transmission_costs=transmission_costs, connectable=False,
                 **kwargs)
        connections = sc.simple_sort()
        supply_curve = sc._sc_points.merge(connections, on='sc_gid')
        return supply_curve
