# -*- coding: utf-8 -*-
"""
Module to handle Supply Curve Transmission features
"""
import numpy as np
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
    SINK_TIE_IN_COST = 0  # $/MW
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

    @staticmethod
    def _calc_lcot(distance, capacity,
                   line_cost=TransmissionFeatures.LINE_COST,
                   tie_in_cost=0):
        """
        Compute levelized cost of transmission (LCOT)

        Parameters
        ----------
        distance : float
            Distance to feature in miles
        capacity : float
            Capacity needed in MW
        line_cost : float
            Cost of tranmission lines in $/MW-mile
        tie_in_cost : float
            Cost to connect to feature in $/MW

        Returns
        -------
        lcot : float
            Levelized cost of transmission
        """
        lcot = capacity * (distance * line_cost + tie_in_cost)

        return lcot

    def _substation_capacity(self, line_gids, line_limited=False):
        """
        Get capacity of a substation from its tranmission lines

        Parameters
        ----------
        line_gids : list
            List of transmission line gids connected to the substation
        line_lmited : bool
            Substation connection is limited by maximum capacity of the
            attached lines

        Returns
        -------
        capacity : float
            Substation capacity = sum(line capacities) / 2
        """
        capacity = 0
        line_max = 0
        for gid in line_gids:
            line = self._features[gid]
            if line['type'] == 'TransLine':
                line_cap = line['capacity']
                capacity += line_cap
                if line_cap > line_max:
                    line_max = line_cap
            else:
                warn("Feature type is {} but should be 'TransLine'"
                     .format(line['type']), HandlerWarning)

        capacity /= 2
        if line_limited:
            if line_max < capacity:
                capacity = line_max

        return capacity

    def available_capacity(self, gid, **kwargs):
        """
        Get available capacity for given line

        Parameters
        ----------
        gid : int
            Unique id of feature of interest
        kwargs : dict
            Internal kwargs for _substation_capacity

        Returns
        -------
        capacity : float
            Available capacity = capacity * available fraction
            default = 10%
        """
        feature = self._features[gid]
        if feature['type'] == 'Substation':
            capacity = self._substation_capacity(feature['lines'], **kwargs)
        else:
            capacity = feature['capacity']

        if capacity is not None:
            capacity *= self.AVAILABLE_CAPACITY

        return capacity

    def _connect(self, gid, capacity):
        """
        Get capacity of a substation from its tranmission lines

        Parameters
        ----------
        gid : list
            Feature gid to connect to
        capacity : float
            Capacity needed in MW
        """
        avail_cap = self._features[gid]['capacity'] * self.AVAILABLE_CAPACITY
        if avail_cap < capacity:
            raise RuntimeError("Cannot connect to {}: "
                               "needed capacity({} MW) > "
                               "available capacity({} MW)"
                               .format(gid, capacity, avail_cap))

        self._features[gid]['capacity'] -= capacity

    def _connect_to_substation(self, line_gids, capacity,
                               line_limited=False):
        """
        Connect to substation and update internal dictionary accordingly

        Parameters
        ----------
        line_gids : list
            List of transmission line gids connected to the substation
        capacity : float
            Capacity needed in MW
        line_lmited : bool
            Substation connection is limited by maximum capacity of the
            attached lines
        """
        if line_limited:
            line_caps = [self._features[gid]['capacity'] for gid in line_gids]
            line_gids = [line_gids[np.argmax(line_caps)], ]
        else:
            capacity /= len(line_gids)

        for gid in line_gids:
            self._connect(gid, capacity)

    def connect(self, gid, capacity, apply=True, **kwargs):
        """
        Check if you can connect to given feature
        If apply, update internal dictionary accordingly

        Parameters
        ----------
        gid : int
            Unique id of feature of intereset
        capacity : float
            Capacity needed in MW
        apply : bool
            Apply capacity to feature with given gid and update
            internal dictionary

        Returns
        -------
        connected : bool
            Flag as to whether connection is possible or not
        """
        avail_cap = self.available_capacity(gid, **kwargs)
        if avail_cap is not None and capacity > avail_cap:
            msg = ("Cannot connect to {}: "
                   "needed capacity({} MW) > available capacity({} MW)"
                   .format(gid, capacity, avail_cap))
            warn(msg, HandlerWarning)
            connected = False
        else:
            connected = True
            if apply:
                feature_type = self._features[gid]['type']
                if feature_type == 'TransLine':
                    self._connect(gid, capacity)
                elif feature_type == 'Substation':
                    self._connect_to_substation(self._features[gid]['lines'],
                                                capacity, **kwargs)
                elif feature_type == 'LoadCen':
                    self._connect(gid, capacity)

        return connected

    def lcot(self, gid, distance, capacity):
        """
        Compute levelized cost of transmission (LCOT) for connecting to give
        feature

        Parameters
        ----------
        gid : int
            Feature gid to connect to
        distance : float
            Distance to feature in miles
        capacity : float
            Capacity needed in MW

        Returns
        -------
        lcot : float
            Levelized cost of transmission
        """
        feature_type = self._features[gid]['type']
        line_cost = self.LINE_COST
        if feature_type == 'TransLine':
            tie_in_cost = self.LINE_TIE_IN_COST
        elif feature_type == 'Substation':
            tie_in_cost = self.STATION_TIE_IN_COST
        elif feature_type == 'LoadCen':
            tie_in_cost = self.CENTER_TIE_IN_COST
        elif feature_type == 'PCALoadCen':
            tie_in_cost = self.SINK_TIE_IN_COST
        else:
            tie_in_cost = 0
            msg = ("Do not recognize feature type {}, tie_in_cost set to 0"
                   .format(feature_type))
            warn(msg, HandlerWarning)

        lcot = self._calc_lcot(distance, capacity, line_cost=line_cost,
                               tie_in_cost=tie_in_cost)
        return lcot
