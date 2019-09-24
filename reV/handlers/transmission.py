# -*- coding: utf-8 -*-
"""
Module to handle Supply Curve Transmission features
"""
import json
import numpy as np
import pandas as pd
from warnings import warn

from reV.utilities.exceptions import HandlerWarning


class TransmissionFeatures:
    """
    Class to handle Supply Curve Transmission features
    """
    def __init__(self, sc_table, line_tie_in_cost=14000, line_cost=3667,
                 station_tine_in_cost=0, center_tie_in_cost=0,
                 sink_tie_in_cost=0, available_capacity=0.1):
        """
        Parameters
        ----------
        sc_table : str | pandas.DataFrame
            Path to .csv or .json containing supply curve transmission mapping
        line_tie_in_cost : float
            Cost of connecting to atransmission line in $/MW
        line_cost : float
            Cost of building transmission line during connection in $/MW-mile
        station_tine_in_cost : float
            Cost of connecting to a substation in $/MW
        center_tie_in_cost : float
            Cost of connecting to a load center in $/MW
         : float
            Fraction of capacity that is available for connection
        """
        self._line_tie_in_cost = line_tie_in_cost
        self._line_cost = line_cost
        self._station_tie_in_cost = station_tine_in_cost
        self._center_tie_in_cost = center_tie_in_cost
        self._sink_tie_in_cost = sink_tie_in_cost
        self._available_capacity = available_capacity

        self._features = self._parse_table(sc_table)
        self._mask = pd.DataFrame(index=self._features)
        self._mask['available'] = True

    def _parse_table(self, sc_table):
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
        cap_perc = self._available_capacity
        trans_features = sc_table.groupby('trans_line_gid').first()
        for gid, feature in trans_features.iterrows():
            name = feature['category']
            feature_dict = {'type': name}
            if name == "TransLine":
                feature_dict['avail_cap'] = feature['ac_cap'] * cap_perc
            elif name == "Substation":
                feature_dict['lines'] = json.loads(feature['trans_gids'])
            elif name == "LoadCen":
                feature_dict['avail_cap'] = feature['ac_cap'] * cap_perc
            elif name == "PCALoadCen":
                feature_dict['avail_cap'] = None

            features[gid] = feature_dict

        return features

    @staticmethod
    def _calc_cost(distance, line_cost=3667, tie_in_cost=0,
                   transmission_multiplier=1):
        """
        Compute transmission cost in $/MW

        Parameters
        ----------
        distance : float
            Distance to feature in miles
        line_cost : float
            Cost of tranmission lines in $/MW-mile
        tie_in_cost : float
            Cost to connect to feature in $/MW
        line_multiplier : float
            Multiplier for region specific line cost increases

        Returns
        -------
        cost : float
            Cost of transmission in $/MW
        """
        cost = (distance * line_cost * transmission_multiplier + tie_in_cost)

        return cost

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
        avail_cap : float
            Substation available capacity
        """
        avail_cap = 0
        line_max = 0
        for gid in line_gids:
            line = self._features[gid]
            if line['type'] == 'TransLine':
                line_cap = line['avail_cap']
                avail_cap += line_cap
                if line_cap > line_max:
                    line_max = line_cap
            else:
                warn("Feature type is {} but should be 'TransLine'"
                     .format(line['type']), HandlerWarning)

        avail_cap /= 2
        if line_limited:
            if line_max < avail_cap:
                avail_cap = line_max

        return avail_cap

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
        avail_cap : float
            Available capacity = capacity * available fraction
            default = 10%
        """
        feature = self._features[gid]
        if feature['type'] == 'Substation':
            avail_cap = self._substation_capacity(feature['lines'], **kwargs)
        else:
            avail_cap = feature['avail_cap']

        return avail_cap

    def _update_availability(self, gid, **kwargs):
        """
        Check features available capacity, if its 0 update _mask

        Parameters
        ----------
        gid : list
            Feature gid to check
        kwargs : dict
            Internal kwargs for substations
        """
        avail_cap = self.available_capacity(gid, **kwargs)
        if avail_cap == 0:
            self._mask.loc[gid, 'available'] = False

    def check_availability(self, gid):
        """
        Check availablity of feature with given gid

        Parameters
        ----------
        gid : list
            Feature gid to check

        Returns
        -------
        bool
            Whether the gid is available or not
        """
        return self._mask.loc[gid, 'available']

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
        avail_cap = self._features[gid]['avail_cap']
        if avail_cap < capacity:
            raise RuntimeError("Cannot connect to {}: "
                               "needed capacity({} MW) > "
                               "available capacity({} MW)"
                               .format(gid, capacity, avail_cap))

        self._features[gid]['avail_cap'] -= capacity

    def _fill_lines(self, line_gids, line_caps, capacity):
        """
        Fill any lines that cannot handle equal portion of capacity and
        remove from lines to be filled and capacity needed

        Parameters
        ----------
        line_gids : ndarray
            Vector of transmission line gids connected to the substation
        line_caps : ndarray
            Vector of available capacity of the transmission lines
        capacity : float
            Capacity needed in MW

        Returns
        ----------
        line_gids : ndarray
            Transmission lines with available capacity
        line_caps : ndarray
            Capacity of lines with available capacity
        capacity : float
            Updated capacity needed to be applied to substation in MW
        """
        apply_cap = capacity / len(line_gids)
        mask = line_caps < apply_cap
        for pos in np.where(line_caps < apply_cap)[0]:
            gid = line_gids[pos]
            apply_cap = line_caps[pos]
            self._connect(gid, apply_cap)
            capacity -= apply_cap

        return line_gids[~mask], line_caps[~mask], capacity

    def _spread_substation_load(self, line_gids, line_caps, capacity):
        """
        Spread needed capacity over all lines connected to substation

        Parameters
        ----------
        line_gids : ndarray
            Vector of transmission line gids connected to the substation
        line_caps : ndarray
            Vector of available capacity of the transmission lines
        capacity : float
            Capacity needed to be applied to substation in MW
        """
        while True:
            lines, line_caps, capacity = self._fill_lines(line_gids, line_caps,
                                                          capacity)
            if len(lines) < len(line_gids):
                line_gids = lines
            else:
                break

        apply_cap = capacity / len(lines)
        for gid in lines:
            self._connect(gid, apply_cap)

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
        line_caps = np.array([self._features[gid]['avail_cap']
                              for gid in line_gids])
        if line_limited:
            gid = line_gids[np.argmax(line_caps)]
            self._connect(gid, capacity)
        else:
            non_zero = np.nonzero(line_caps)[0]
            line_gids = np.array([line_gids[i] for i in non_zero])
            line_caps = line_caps[non_zero]
            self._spread_substation_load(line_gids, line_caps, capacity)

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
        kwargs : dict
            Internal kwargs for substations

        Returns
        -------
        connected : bool
            Flag as to whether connection is possible or not
        """
        if self.check_availability(gid):
            avail_cap = self.available_capacity(gid, **kwargs)
            if avail_cap is not None and capacity > avail_cap:
                connected = False
            else:
                connected = True
                if apply:
                    feature_type = self._features[gid]['type']
                    if feature_type == 'TransLine':
                        self._connect(gid, capacity)
                    elif feature_type == 'Substation':
                        lines = self._features[gid]['lines']
                        self._connect_to_substation(lines, capacity,
                                                    **kwargs)
                    elif feature_type == 'LoadCen':
                        self._connect(gid, capacity)

                    self._update_availability(gid)
        else:
            connected = False

        return connected

    def cost(self, gid, distance, transmission_multiplier=1,
             capacity=None, **kwargs):
        """
        Compute levelized cost of transmission (LCOT) for connecting to give
        feature

        Parameters
        ----------
        gid : int
            Feature gid to connect to
        distance : float
            Distance to feature in miles
        line_multiplier : float
            Multiplier for region specific line cost increases
        capacity : float
            Capacity needed in MW, if None DO NOT check if connection is
            possible
        kwargs : dict
            Internal kwargs for connect

        Returns
        -------
        cost : float
            Cost of transmission in $/MW, if None indicates connection is
            NOT possible
        """
        feature_type = self._features[gid]['type']
        line_cost = self._line_cost
        if feature_type == 'TransLine':
            tie_in_cost = self._line_tie_in_cost
        elif feature_type == 'Substation':
            tie_in_cost = self._station_tie_in_cost
        elif feature_type == 'LoadCen':
            tie_in_cost = self._center_tie_in_cost
        elif feature_type == 'PCALoadCen':
            tie_in_cost = self._sink_tie_in_cost
        else:
            tie_in_cost = 0
            msg = ("Do not recognize feature type {}, tie_in_cost set to 0"
                   .format(feature_type))
            warn(msg, HandlerWarning)

        cost = self._calc_cost(distance, line_cost=line_cost,
                               tie_in_cost=tie_in_cost,
                               transmission_multiplier=transmission_multiplier)
        if capacity is not None:
            if not self.connect(gid, capacity, apply=False, **kwargs):
                cost = None

        return cost
