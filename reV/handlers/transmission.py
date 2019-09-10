# -*- coding: utf-8 -*-
"""
Module to handle Supply Curve Transmission features
"""
import pandas as pd
from warnings import warn

from reV.utilities.exceptions import HandlerWarning


class TransmissionFeatures:
    """
    Class to handle Supply Curve Transmission features
    """
    LINE_TIE_IN_COST = 14000  # $/MW
    LINE_COST = 3667  # $/MW-mile
    STATION_TIE_IN_COST = 0  # $/MW
    CENTER_TIE_IN_COST = 0  # $/MW
    AVAILABLE_CAPACITY = 0.1

    def __init__(self, sc_table):
        """
        Parameters
        ----------
        sc_table : str | pandas.DataFrame
            Path to .csv or .json containing supply curve transmission mapping
        """
        self._features = self._parse_table(sc_table)

    @staticmethod
    def _parse_table(sc_table):
        """
        Extract features and their capacity from supply curve transmission
        mapping table

        Parameters
        ----------
        sc_table : str
            Path to .csv or .json containing supply curve transmission mapping

        Returns
        -------
        features : dict
            Nested dictionary of features (lines, substations, loadcenters)
            lines : {capacity}
            substations : {lines}
            loadcenters : {capacity}
        """
        if isinstance(sc_table, str):
            if sc_table.endswith('.csv'):
                sc_table = pd.read_csv(sc_table)
            elif sc_table.endswith('.json'):
                sc_table = pd.read_json(sc_table)
            else:
                raise ValueError('Cannot parse {}'.format(sc_table))
        elif not isinstance(sc_table, pd.DataFrame):
            raise ValueError("Supply Curve table must be a .csv, .json, or "
                             "a pandas DataFrame")

        features = {}
        trans_features = sc_table.groupby('trans_line_gid').first()
        for gid, feature in trans_features.iterrows():
            name = feature['category']
            feature_dict = {'type': name}
            if name == "TransLine":
                feature_dict['capacity'] = feature['ac_cap']
            elif name == "Substation":
                feature_dict['lines'] = feature['trans_gids']
            elif name == "LoadCen":
                feature_dict['capacity'] = feature['ac_cap']
            elif name == "PCALoadCen":
                feature_dict['capacity'] = None

            features[gid] = feature_dict

        return features

    def _get_station_capacity(self, line_gids):
        """
        Get capacity of a substation from its tranmission lines

        Parameters
        ----------
        line_gids : list
            List of transmission line gids connected to the substation

        Returns
        -------
        capacity : float
            Substation capacity = sum(line capacities) / 2
        """
        capacity = 0
        for gid in line_gids:
            line = self._features[gid]
            if line['type'] == 'TransLine':
                capacity += line['capacity']
            else:
                warn("Feature type is {} but should be 'TransLine'"
                     .format(line['type']), HandlerWarning)

        return capacity / 2

    def available_capacity(self, gid):
        """
        Get available capacity for given line

        Parameters
        ----------
        gid : int
            Unique id of feature of interest

        Returns
        -------
        capacity : float
            Available capacity = capacity * available fraction
            default = 10%
        feature_type : str
            Feature type (TransLine, Substation, LoadCen, PCALoadCen)
        """
        feature = self._features[gid]
        feature_type = feature['type']
        if feature_type == 'Substation':
            capacity = self._get_station_capacity(feature['lines'])
        else:
            capacity = feature['capacity']

        if capacity is not None:
            capacity *= self.AVAILABLE_CAPACITY

        return capacity, feature_type

    def cost(self, gid, distance, capacity):
        """
        Get available capacity for given line

        Parameters
        ----------
        gid : int
            Unique id of feature of intereset
        distance : float
            Distance to feature in miles
        capacity : float
            Capacity needed in MW

        Returns
        -------
        cost : float
            Cost to connect, if None means there is not enough
            available capacity
        """
        avail_cap, f_type = self.available_capacity(gid)

        if f_type == 'TransLine':
            tie_in_cost = self.LINE_TIE_IN_COST
        elif f_type == 'Substation':
            tie_in_cost = self.STATION_TIE_IN_COST
        else:
            tie_in_cost = self.CENTER_TIE_IN_COST

        cost = capacity * (distance * self.LINE_COST + tie_in_cost)
        if avail_cap is not None and capacity > avail_cap:
            cost = None

        return cost

    def connect(self, gid, capacity):
        """
        Connect to given feature and update internal dictionary

        Parameters
        ----------
        gid : int
            Unique id of feature of intereset
        capacity : float
            Capacity needed in MW
        """
        avail_cap, f_type = self.available_capacity(gid)
        if avail_cap is not None and capacity > avail_cap:
            msg = ("Cannot connect to {} {}:"
                   "\n\tCapacity({}) > Available Capacity({})"
                   .format(f_type, gid, capacity, avail_cap))
            warn(msg)
