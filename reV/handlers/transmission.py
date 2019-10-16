# -*- coding: utf-8 -*-
"""
Module to handle Supply Curve Transmission features
"""
import json
import logging
import numpy as np
import os
import pandas as pd
from warnings import warn

from reV.utilities import safe_json_load
from reV.utilities.exceptions import HandlerWarning

logger = logging.getLogger(__name__)


class TransmissionFeatures:
    """
    Class to handle Supply Curve Transmission features
    """
    def __init__(self, trans_table, line_tie_in_cost=14000, line_cost=3667,
                 station_tie_in_cost=0, center_tie_in_cost=0,
                 sink_tie_in_cost=14000, available_capacity=0.1,
                 line_limited=False):
        """
        Parameters
        ----------
        trans_table : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supply curve
            transmission mapping
        line_tie_in_cost : float
            Cost of connecting to a transmission line in $/MW
        line_cost : float
            Cost of building transmission line during connection in $/MW-mile
        station_tie_in_cost : float
            Cost of connecting to a substation in $/MW
        center_tie_in_cost : float
            Cost of connecting to a load center in $/MW
        sink_tie_in_cost : float
            Cost of connecting to a synthetic load center (infinite sink)
            in $/MW
        available_capacity : float
            Fraction of capacity that is available for connection
        line_lmited : bool
            Substation connection is limited by maximum capacity of the
            attached lines
        """

        logger.debug('Line tie in cost: {} $/MW'.format(line_tie_in_cost))
        logger.debug('Line cost: {} $/MW-mile'.format(line_cost))
        logger.debug('Station tie in cost: {} $/MW'
                     .format(station_tie_in_cost))
        logger.debug('Center tie in cost: {} $/MW'.format(center_tie_in_cost))
        logger.debug('Synthetic load center tie in cost: {} $/MW'
                     .format(sink_tie_in_cost))
        logger.debug('Available capacity fraction: {}'
                     .format(available_capacity))
        logger.debug('Line limited substation connections: {}'
                     .format(line_limited))

        self._line_tie_in_cost = line_tie_in_cost
        self._line_cost = line_cost
        self._station_tie_in_cost = station_tie_in_cost
        self._center_tie_in_cost = center_tie_in_cost
        self._sink_tie_in_cost = sink_tie_in_cost
        self._available_capacity_frac = available_capacity

        self._feature_types, self._avail_capacities, self._substation_lines = \
            self._get_features(trans_table)

        self._feature_gid_list = list(self._feature_types.keys())
        self._available_mask = np.ones((len(self._feature_types), ),
                                       dtype=bool)

        self._line_limited = line_limited

    def __repr__(self):
        msg = "{} with {} features".format(self.__class__.__name__, len(self))
        return msg

    def __len__(self):
        return len(self._feature_types)

    @staticmethod
    def _parse_dictionary(features):
        """
        Parse features dict from .json or json object

        Parameters
        ----------
        features : dict | str
            Dictionary of transmission features or path to .json containing
            dictionary of transmission features

        Returns
        -------
        features : dict
            Nested dictionary of features (lines, substations, loadcenters)
            lines : {capacity}
            substations : {lines}
            loadcenters : {capacity}
        """
        if isinstance(features, str):
            if os.path.isfile(features):
                features = safe_json_load(features)
            else:
                features = json.loads(features)

        elif not isinstance(features, dict):
            msg = ("Transmission featurse must be a .json file, object, "
                   "or a dictionary")
            logger.error(msg)
            raise ValueError(msg)

        return features

    @staticmethod
    def _parse_table(trans_table):
        """
        Extract features and their capacity from supply curve transmission
        mapping table

        Parameters
        ----------
        trans_table : str | pandas.DataFrame
            Path to .csv or .json containing supply curve transmission mapping

        Returns
        -------
        trans_table : pandas.DataFrame
            DataFrame of transmission features
        """
        if isinstance(trans_table, str):
            if trans_table.endswith('.csv'):
                trans_table = pd.read_csv(trans_table)
            elif trans_table.endswith('.json'):
                trans_table = pd.read_json(trans_table)
            else:
                raise ValueError('Cannot parse {}'.format(trans_table))
        elif not isinstance(trans_table, pd.DataFrame):
            msg = ("Supply Curve table must be a .csv, .json, or "
                   "a pandas DataFrame")
            logger.error(msg)
            raise ValueError(msg)

        return trans_table

    def _features_from_table(self, trans_table):
        """
        Extract features and their capacity from supply curve transmission
        mapping table

        Parameters
        ----------
        trans_table : pandas.DataFrame
            DataFrame of transmission features

        Returns
        -------
        features : dict
            Nested dictionary of features (lines, substations, loadcenters)
            lines : {capacity}
            substations : {lines}
            loadcenters : {capacity}
        """

        feature_types = {}
        avail_capacities = {}
        substation_lines = {}

        cap_frac = self._available_capacity_frac
        trans_features = trans_table.groupby('trans_line_gid').first()

        for gid, feature in trans_features.iterrows():
            name = feature['category'].lower()
            feature_types[gid] = name

            if name == 'transline':
                avail_capacities[gid] = feature['ac_cap'] * cap_frac

            elif name == 'substation':
                substation_lines[gid] = json.loads(feature['trans_gids'])

            elif name == 'loadcen':
                avail_capacities[gid] = feature['ac_cap'] * cap_frac

            elif name == 'pcaloadcen':
                avail_capacities[gid] = None

        return feature_types, avail_capacities, substation_lines

    def _get_features(self, trans_table):
        """
        Create transmission features dictionary either from supply curve
        transmission mapping or from pre-created dictionary

        Parameters
        ----------
        trans_table : str
            Path to .csv or .json containing supply curve transmission mapping

        Returns
        -------
        features : dict
            Nested dictionary of features (lines, substations, loadcenters)
            lines : {capacity}
            substations : {lines}
            loadcenters : {capacity}
        """

        trans_table = self._parse_table(trans_table)
        feature_types, avail_capacities, substation_lines = \
            self._features_from_table(trans_table)

        return feature_types, avail_capacities, substation_lines

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

    def _substation_capacity(self, line_gids):
        """
        Get capacity of a substation from its tranmission lines

        Parameters
        ----------
        line_gids : list
            List of transmission line gids connected to the substation


        Returns
        -------
        avail_cap : float
            Substation available capacity
        """

        line_caps = [self._avail_capacities[gid] for gid in line_gids]
        avail_cap = np.sum(line_caps) / 2

        if self._line_limited:
            if np.max(line_caps) < avail_cap:
                avail_cap = np.max(line_caps)

        return avail_cap

    def available_capacity(self, gid):
        """
        Get available capacity for given line

        Parameters
        ----------
        gid : int
            Unique id of feature of interest

        Returns
        -------
        avail_cap : float
            Available capacity = capacity * available fraction
            default = 10%
        """

        if gid in self._avail_capacities:
            avail_cap = self._avail_capacities[gid]

        else:
            avail_cap = self._substation_capacity(self._substation_lines[gid])

        return avail_cap

    def _update_availability(self, gid):
        """
        Check features available capacity, if its 0 update _available_mask

        Parameters
        ----------
        gid : list
            Feature gid to check
        """
        avail_cap = self.available_capacity(gid)
        if avail_cap == 0:
            i = self._feature_gid_list.index(gid)
            self._available_mask[i] = False

    def check_availability(self, gid):
        """
        Check availablity of feature with given gid

        Parameters
        ----------
        gid : int
            Feature gid to check

        Returns
        -------
        bool
            Whether the gid is available or not
        """
        i = self._feature_gid_list.index(gid)
        return self._available_mask[i]

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
        avail_cap = self._avail_capacities[gid]
        if avail_cap < capacity:
            raise RuntimeError("Cannot connect to {}: "
                               "needed capacity({} MW) > "
                               "available capacity({} MW)"
                               .format(gid, capacity, avail_cap))

        self._avail_capacities[gid] -= capacity

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

    def _connect_to_substation(self, line_gids, capacity):
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
        line_caps = np.array([self._avail_capacities[gid]
                              for gid in line_gids])
        if self._line_limited:
            gid = line_gids[np.argmax(line_caps)]
            self._connect(gid, capacity)
        else:
            non_zero = np.nonzero(line_caps)[0]
            line_gids = np.array([line_gids[i] for i in non_zero])
            line_caps = line_caps[non_zero]
            self._spread_substation_load(line_gids, line_caps, capacity)

    def connect(self, gid, capacity, apply=True):
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
        if self.check_availability(gid):
            avail_cap = self.available_capacity(gid)
            if avail_cap is not None and capacity > avail_cap:
                connected = False
            else:
                connected = True
                if apply:
                    feature_type = self._feature_types[gid]
                    if feature_type == 'transline':
                        self._connect(gid, capacity)
                    elif feature_type == 'substation':
                        lines = self._substation_lines[gid]
                        self._connect_to_substation(lines, capacity)
                    elif feature_type == 'loadcen':
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
        feature_type = self._feature_types[gid]
        line_cost = self._line_cost
        if feature_type == 'transline':
            tie_in_cost = self._line_tie_in_cost
        elif feature_type == 'substation':
            tie_in_cost = self._station_tie_in_cost
        elif feature_type == 'loadcen':
            tie_in_cost = self._center_tie_in_cost
        elif feature_type == 'pcaloadcen':
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

    @classmethod
    def feature_capacity(cls, trans_table, available_capacity=0.1):
        """
        Compute available capacity for all features

        Parameters
        ----------
        trans_table : str | pandas.DataFrame
            Path to .csv or .json containing supply curve transmission mapping
        line_tie_in_cost : float
            Cost of connecting to a transmission line in $/MW
        line_cost : float
            Cost of building transmission line during connection in $/MW-mile
        station_tine_in_cost : float
            Cost of connecting to a substation in $/MW
        center_tie_in_cost : float
            Cost of connecting to a load center in $/MW
        center_tie_in_cost : float
            Cost of connecting to a synthetic load center (infinite sink)
            in $/MW
        available_capacity : float
            Fraction of capacity that is available for connection

        Returns
        -------
        feature_cap : pandas.DataFrame
            Available Capacity for each transmission feature
        """
        try:
            feature = cls(trans_table, available_capacity=available_capacity)

            feature_cap = {}
            for gid in feature._feature_types:
                feature_cap[gid] = feature.available_capacity(gid)
        except Exception:
            logger.exception("Error computing available capacity for all "
                             "features in {}".format(cls))
            raise

        feature_cap = pd.Series(feature_cap)
        feature_cap.name = 'avail_cap'
        feature_cap.index.name = 'trans_line_gid'
        feature_cap = feature_cap.to_frame().reset_index()

        return feature_cap


class TransmissionCosts(TransmissionFeatures):
    """
    Class to compute supply curve -> transmission feature costs
    """
    def _features_from_table(self, trans_table):
        """
        Extract features and their capacity from supply curve transmission
        mapping table

        Parameters
        ----------
        trans_table : pandas.DataFrame
            DataFrame of transmission features

        Returns
        -------
        features : dict
            Nested dictionary of features (lines, substations, loadcenters)
            lines : {capacity}
            substations : {lines}
            loadcenters : {capacity}
        """

        feature_types = {}
        avail_capacities = {}
        substation_lines = {}

        if 'avail_cap' not in trans_table:
            kwargs = {'available_capacity': self._available_capacity_frac}
            fc = TransmissionFeatures.feature_capacity(trans_table,
                                                       **kwargs)
            trans_table = trans_table.merge(fc, on='trans_line_gid')

        trans_features = trans_table.groupby('trans_line_gid').first()
        for gid, feature in trans_features.iterrows():
            feature_types[gid] = feature['category'].lower()
            avail_capacities[gid] = feature['avail_cap']

        return feature_types, avail_capacities, substation_lines

    def available_capacity(self, gid):
        """
        Get available capacity for given line

        Parameters
        ----------
        gid : int
            Unique id of feature of interest

        Returns
        -------
        avail_cap : float
            Available capacity = capacity * available fraction
            default = 10%
        """

        return self._avail_capacities[gid]

    @classmethod
    def feature_costs(cls, trans_table, capacity=None, line_tie_in_cost=14000,
                      line_cost=3667, station_tie_in_cost=0,
                      center_tie_in_cost=0, sink_tie_in_cost=14000,
                      available_capacity=0.1, **kwargs):
        """
        Compute costs for all connections in given transmission table

        Parameters
        ----------
        trans_table : str | pandas.DataFrame
            Path to .csv or .json containing supply curve transmission mapping
        capacity : float
            Capacity needed in MW, if None DO NOT check if connection is
            possible
        line_tie_in_cost : float
            Cost of connecting to a transmission line in $/MW
        line_cost : float
            Cost of building transmission line during connection in $/MW-mile
        station_tine_in_cost : float
            Cost of connecting to a substation in $/MW
        center_tie_in_cost : float
            Cost of connecting to a load center in $/MW
        center_tie_in_cost : float
            Cost of connecting to a synthetic load center (infinite sink)
            in $/MW
        available_capacity : float
            Fraction of capacity that is available for connection
        kwargs : dict
            Internal kwargs for connect

        Returns
        -------
        cost : ndarray
            Cost of transmission in $/MW, if None indicates connection is
            NOT possible
        """
        try:
            feature = cls(trans_table,
                          line_tie_in_cost=line_tie_in_cost,
                          line_cost=line_cost,
                          station_tie_in_cost=station_tie_in_cost,
                          center_tie_in_cost=center_tie_in_cost,
                          sink_tie_in_cost=sink_tie_in_cost,
                          available_capacity=available_capacity)

            costs = []
            for _, row in trans_table.iterrows():
                tm = row.get('transmission_multiplier', 1)
                costs.append(feature.cost(row['trans_line_gid'],
                                          row['dist_mi'], capacity=capacity,
                                          transmission_multiplier=tm,
                                          **kwargs))
        except Exception:
            logger.exception("Error computing costs for all connections in {}"
                             .format(cls))
            raise

        return np.array(costs, dtype='float32')
