# -*- coding: utf-8 -*-
"""
reV supply curve module
- Calculation of LCOT
- Supply Curve creation
"""
import concurrent.futures as cf
import pandas as pd

from reV.handlers.transmission import TransmissionFeatures
from reV.utilities.exceptions import SupplyCurveInputError


class SupplyCurve:
    """
    Class to handle LCOT calcuation and SupplyCurve sorting
    """
    def __init__(self, sc_points, sc_table):
        """
        Parameters
        ----------
        sc_points : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supplcy curve
            point summary
        sc_table : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supply curve
            transmission mapping
        max_workers : int | NoneType
            Number of workers to use to compute lcot, if > 1 run in parallel
        """
        sc_points = self._parse_table(sc_points)
        self._sc_points = sc_points.set_index('sc_gid')
        self._sc_table = self._parse_sc_table(sc_points, sc_table)
        self._trans_features = TransmissionFeatures(self._sc_table)
        self._mask = pd.DataFrame(index=self._sc_table['sc_gid'].unique())
        self._mask['empty'] = True

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
    def _parse_sc_table(sc_points, sc_table):
        """
        Import supply curve table, add in supply curve point capacity

        Parameters
        ----------
        sc_points : pd.DataFrame
            Table of supply curve point summary
        sc_table : pd.DataFrame
            Table mapping supply curve points to transmission features

        Returns
        -------
        sc_table : pd.DataFrame
            Updated table mapping supply curve points to transmission features
        """
        sc_table = SupplyCurve._parse_table(sc_table)
        point_merge_cols = SupplyCurve._get_merge_cols(sc_points.columns)
        table_merge_cols = SupplyCurve._get_merge_cols(sc_table.columns)

        sc_cap = sc_points[point_merge_cols + ['capacity', 'sc_gid']]
        rename = {p: t for p, t in zip(point_merge_cols, table_merge_cols)}
        sc_cap = sc_cap.rename(columns=rename)

        sc_table = sc_table.merge(sc_cap, on=table_merge_cols, how='inner')

        return sc_table

    @staticmethod
    def _compute_lcot(sc_table, max_workers=1, **kwargs):
        """
        Compute costs for all combinations of supply curve points and
        tranmission features in _sc_table

        Parameters
        ----------
        sc_table : pd.DataFrame
            Table mapping supply curve points to transmission features
            MUST contain supply curve point capacity
        max_workers : int | NoneType
            Number of workers to use to compute lcot, if > 1 run in parallel
        kwargs : dict
            kwargs for TransmissionFeatures.lcot

        Returns
        -------
        lcot : list
            Levelized cost of transmission for all supply curve -
            tranmission feature connections
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
                    futures.append(exe.submit(feature.lcot, row['trans_gid'],
                                              row['dist_mi'], row['capacity'],
                                              **kwargs))

                lcot = [future.result() for future in futures]
        else:
            lcot = []
            for _, row in sc_table.iterrows():
                lcot.append(feature.lcot(row['trans_line_gid'], row['dist_mi'],
                                         row['capacity'], **kwargs))

        return lcot

    def _serial_sort(self, sc_table=None, **kwargs):
        """
        run supply curve sorting in serial

        Parameters
        ----------
        sc_table : pandas.DataFrame | NoneType
            Supply Curve Tranmission table to sort on
            If none use self._sc_table
        kwargs : dict
            Kwargs to compute lcot
        """
        if sc_table is None:
            sc_table = self._sc_table

        if 'lcot' not in sc_table:
            sc_table['lcot'] = SupplyCurve._compute_lcot(sc_table, **kwargs)

        pos = sc_table['lcot'].isnull()
        for _, row in sc_table.loc[~pos].sort_values('lcot').iterrows():
            sc_gid = row['sc_gid']
            if self._mask.loc[sc_gid, 'empty']:
                trans_gid = row['trans_line_gid']
                connect = self._trans_features.connect(trans_gid,
                                                       row['capacity'])
                if connect:
                    self._mask.loc[sc_gid, 'empty'] = False
                    self._sc_points.at[sc_gid, 'trans_gid'] = trans_gid
                    self._sc_points.at[sc_gid, 'trans_type'] = row['category']
                    self._sc_points.at[sc_gid, 'lcot'] = row['lcot']
