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

from reV.utilities import SupplyCurveField
from reV.utilities.exceptions import (HandlerWarning, HandlerKeyError,
                                      HandlerRuntimeError)

from rex.utilities.utilities import parse_table
from gaps.config import load_config

logger = logging.getLogger(__name__)


class TransmissionFeatures:
    """
    Class to handle Supply Curve Transmission features
    """
    def __init__(self, trans_table, line_tie_in_cost=14000, line_cost=2279,
                 station_tie_in_cost=0, center_tie_in_cost=0,
                 sink_tie_in_cost=1e9, avail_cap_frac=1,
                 line_limited=False):
        """
        Parameters
        ----------
        trans_table : str | pandas.DataFrame
            Path to .csv or config file or DataFrame containing supply curve
            transmission mapping
        line_tie_in_cost : float, optional
            Cost of connecting to a transmission line in $/MW,
            by default 14000
        line_cost : float, optional
            Cost of building transmission line during connection in $/MW-km,
            by default 2279
        station_tie_in_cost : float, optional
            Cost of connecting to a substation in $/MW,
            by default 0
        center_tie_in_cost : float, optional
            Cost of connecting to a load center in $/MW,
            by default 0
        sink_tie_in_cost : float, optional
            Cost of connecting to a synthetic load center (infinite sink)
            in $/MW, by default 1e9
        avail_cap_frac : float, optional
            Fraction of capacity that is available for connection, by default 1
        line_limited : bool, optional
            Substation connection is limited by maximum capacity of the
            attached lines, legacy method, by default False
        """

        logger.debug('Trans table input: {}'.format(trans_table))
        logger.debug('Line tie in cost: {} $/MW'.format(line_tie_in_cost))
        logger.debug('Line cost: {} $/MW-km'.format(line_cost))
        logger.debug('Station tie in cost: {} $/MW'
                     .format(station_tie_in_cost))
        logger.debug('Center tie in cost: {} $/MW'.format(center_tie_in_cost))
        logger.debug('Synthetic load center tie in cost: {} $/MW'
                     .format(sink_tie_in_cost))
        logger.debug('Available capacity fraction: {}'
                     .format(avail_cap_frac))
        logger.debug('Line limited substation connections: {}'
                     .format(line_limited))

        self._line_tie_in_cost = line_tie_in_cost
        self._line_cost = line_cost
        self._station_tie_in_cost = station_tie_in_cost
        self._center_tie_in_cost = center_tie_in_cost
        self._sink_tie_in_cost = sink_tie_in_cost
        self._avail_cap_frac = avail_cap_frac

        self._features = self._get_features(trans_table)

        self._feature_gid_list = list(self._features.keys())
        self._available_mask = np.ones(
            (int(1 + max(list(self._features.keys()))), ), dtype=bool)

        self._line_limited = line_limited

    def __repr__(self):
        msg = "{} with {} features".format(self.__class__.__name__, len(self))
        return msg

    def __len__(self):
        return len(self._features)

    def __getitem__(self, gid):
        if gid not in self._features:
            msg = "Invalid feature gid {}".format(gid)
            logger.error(msg)
            raise HandlerKeyError(msg)

        return self._features[gid]

    @staticmethod
    def _parse_dictionary(features):
        """
        Parse features dict object or config file

        Parameters
        ----------
        features : dict | str
            Dictionary of transmission features or path to config containing
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
                features = load_config(features)
            else:
                features = json.loads(features)

        elif not isinstance(features, dict):
            msg = ("Transmission features must be a config file, object, "
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
        try:
            trans_table = parse_table(trans_table)
        except ValueError as ex:
            logger.error(ex)
            raise

        trans_table = \
            trans_table.rename(
                columns={'trans_line_gid': SupplyCurveField.TRANS_GID,
                         'trans_gids': 'trans_line_gids'})

        contains_dist_in_miles = "dist_mi" in trans_table
        missing_km_dist = SupplyCurveField.DIST_SPUR_KM not in trans_table
        if contains_dist_in_miles and missing_km_dist:
            trans_table = trans_table.rename(
                columns={"dist_mi": SupplyCurveField.DIST_SPUR_KM}
            )
            trans_table[SupplyCurveField.DIST_SPUR_KM] *= 1.60934

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

        features = {}

        cap_frac = self._avail_cap_frac
        trans_features = trans_table.groupby(SupplyCurveField.TRANS_GID)
        trans_features = trans_features.first()

        for gid, feature in trans_features.iterrows():
            name = feature[SupplyCurveField.TRANS_TYPE].lower()
            feature_dict = {'type': name}

            if name == 'transline':
                feature_dict[SupplyCurveField.TRANS_CAPACITY] = (
                    feature['ac_cap'] * cap_frac
                )

            elif name == 'substation':
                feature_dict['lines'] = json.loads(feature['trans_line_gids'])

            elif name == 'loadcen':
                feature_dict[SupplyCurveField.TRANS_CAPACITY] = (
                    feature['ac_cap'] * cap_frac
                )

            elif name == 'pcaloadcen':
                feature_dict[SupplyCurveField.TRANS_CAPACITY] = None

            else:
                msg = ('Cannot not recognize feature type "{}" '
                       'for trans gid {}!'.format(name, gid))
                logger.error(msg)
                raise HandlerKeyError(msg)

            features[gid] = feature_dict

        return features

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
        features = self._features_from_table(trans_table)

        return features

    def check_feature_dependencies(self):
        """Check features for dependencies that are missing and raise error."""
        missing = {}
        for gid, feature_dict in self._features.items():
            for line_gid in feature_dict.get('lines', []):
                if line_gid not in self._features:
                    if gid not in missing:
                        missing[gid] = []
                    missing[gid].append(line_gid)

        if any(missing):
            emsg = ('Transmission feature table has {} parent features that '
                    'depend on missing lines. Missing dependencies: {}'
                    .format(len(missing), missing))
            logger.error(emsg)
            raise RuntimeError(emsg)

    @staticmethod
    def _calc_cost(distance, line_cost=2279, tie_in_cost=0,
                   transmission_multiplier=1):
        """
        Compute transmission cost in $/MW

        Parameters
        ----------
        distance : float
            Distance to feature in kms
        line_tie_in_cost : float, optional
            Cost of connecting to a transmission line in $/MW,
            by default 14000
        tie_in_cost : float, optional
            Cost to tie in line to feature in $/MW, by default 0
        tranmission_multiplier : float, optional
            Multiplier for region specific line cost increases, by default 1

        Returns
        -------
        cost : float
            Cost of transmission in $/MW
        """
        cost = (distance * line_cost * transmission_multiplier + tie_in_cost)

        return cost

    def _substation_capacity(self, gid, line_gids):
        """
        Get capacity of a substation from its tranmission lines

        Parameters
        ----------
        gid : int
            Substation gid
        line_gids : list
            List of transmission line gids connected to the substation

        Returns
        -------
        avail_cap : float
            Substation available capacity
        """
        try:
            line_caps = [self[l_gid][SupplyCurveField.TRANS_CAPACITY]
                         for l_gid in line_gids]
        except HandlerKeyError as e:
            msg = ('Could not find capacities for substation gid {} and '
                   'connected lines: {}'.format(gid, line_gids))
            logger.error(msg)
            raise HandlerKeyError(msg) from e

        avail_cap = sum(line_caps) / 2

        if self._line_limited:
            max_cap = max(line_caps) / 2
            if max_cap < avail_cap:
                avail_cap = max_cap

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
            default = 100%
        """

        feature = self[gid]

        if SupplyCurveField.TRANS_CAPACITY in feature:
            avail_cap = feature[SupplyCurveField.TRANS_CAPACITY]

        elif 'lines' in feature:
            avail_cap = self._substation_capacity(gid, feature['lines'])

        else:
            msg = ('Could not parse available capacity from feature: {}'
                   .format(feature))
            logger.error(msg)
            raise HandlerRuntimeError(msg)

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
            self._available_mask[gid] = False

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
        return self._available_mask[gid]

    def _connect(self, gid, capacity):
        """
        Connect to a standalone transmission feature (not a substation)
        and decrement the feature's available capacity.
        Raise exception if not able to connect.

        Parameters
        ----------
        gid : int
            Feature gid to connect to
        capacity : float
            Capacity needed in MW
        """
        avail_cap = self[gid][SupplyCurveField.TRANS_CAPACITY]

        if avail_cap < capacity:
            msg = ("Cannot connect to {}: "
                   "needed capacity({} MW) > "
                   "available capacity({} MW)"
                   .format(gid, capacity, avail_cap))
            logger.error(msg)
            raise RuntimeError(msg)

        self[gid][SupplyCurveField.TRANS_CAPACITY] -= capacity

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
        line_caps = np.array([self[gid][SupplyCurveField.TRANS_CAPACITY]
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
                    feature_type = self[gid]['type']
                    if feature_type == 'transline':
                        self._connect(gid, capacity)
                    elif feature_type == 'substation':
                        lines = self[gid]['lines']
                        self._connect_to_substation(lines, capacity)
                    elif feature_type == 'loadcen':
                        self._connect(gid, capacity)

                    self._update_availability(gid)
        else:
            connected = False

        return connected

    def cost(self, gid, distance, transmission_multiplier=1,
             capacity=None):
        """
        Compute levelized cost of transmission (LCOT) for connecting to give
        feature

        Parameters
        ----------
        gid : int
            Feature gid to connect to
        distance : float
            Distance to feature in kms
        line_multiplier : float
            Multiplier for region specific line cost increases
        capacity : float
            Capacity needed in MW, if None DO NOT check if connection is
            possible

        Returns
        -------
        cost : float
            Cost of transmission in $/MW, if None indicates connection is
            NOT possible
        """
        feature_type = self[gid]['type']
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
            logger.warning(msg)
            warn(msg, HandlerWarning)

        cost = self._calc_cost(distance, line_cost=line_cost,
                               tie_in_cost=tie_in_cost,
                               transmission_multiplier=transmission_multiplier)
        if capacity is not None:
            if not self.connect(gid, capacity, apply=False):
                cost = None

        return cost

    @classmethod
    def feature_capacity(cls, trans_table, avail_cap_frac=1):
        """
        Compute available capacity for all features

        Parameters
        ----------
        trans_table : str | pandas.DataFrame
            Path to .csv or .json containing supply curve transmission mapping
        avail_cap_frac : float, optional
            Fraction of capacity that is available for connection, by default 1

        Returns
        -------
        feature_cap : pandas.DataFrame
            Available Capacity for each transmission feature
        """
        try:
            feature = cls(trans_table, avail_cap_frac=avail_cap_frac)

            feature_cap = {}
            for gid, _ in feature._features.items():
                feature_cap[gid] = feature.available_capacity(gid)
        except Exception:
            logger.exception("Error computing available capacity for all "
                             "features in {}".format(cls))
            raise

        feature_cap = pd.Series(feature_cap)
        feature_cap.name = SupplyCurveField.TRANS_CAPACITY
        feature_cap.index.name = SupplyCurveField.TRANS_GID
        feature_cap = feature_cap.to_frame().reset_index()

        return feature_cap


class TransmissionCosts(TransmissionFeatures):
    """
    Class to compute supply curve -> transmission feature costs
    """
    def _features_from_table(self, trans_table):
        """
        Extract features and their capacity from supply curve transmission
        mapping table and pre-compute the available capacity of each feature

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

        features = {}

        if SupplyCurveField.TRANS_CAPACITY not in trans_table:
            kwargs = {'avail_cap_frac': self._avail_cap_frac}
            fc = TransmissionFeatures.feature_capacity(trans_table,
                                                       **kwargs)
            trans_table = trans_table.merge(fc, on=SupplyCurveField.TRANS_GID)

        trans_features = trans_table.groupby(SupplyCurveField.TRANS_GID)
        trans_features = trans_features.first()
        for gid, feature in trans_features.iterrows():
            name = feature[SupplyCurveField.TRANS_TYPE].lower()
            feature_dict = {'type': name,
                            SupplyCurveField.TRANS_CAPACITY: (
                                feature[SupplyCurveField.TRANS_CAPACITY]
                            )}
            features[gid] = feature_dict

        return features

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
            default = 100%
        """

        return self[gid][SupplyCurveField.TRANS_CAPACITY]

    @classmethod
    def feature_costs(cls, trans_table, capacity=None, line_tie_in_cost=14000,
                      line_cost=2279, station_tie_in_cost=0,
                      center_tie_in_cost=0, sink_tie_in_cost=1e9,
                      avail_cap_frac=1, line_limited=False):
        """
        Compute costs for all connections in given transmission table

        Parameters
        ----------
        trans_table : str | pandas.DataFrame
            Path to .csv or .json containing supply curve transmission mapping
        capacity : float
            Capacity needed in MW, if None DO NOT check if connection is
            possible
        line_tie_in_cost : float, optional
            Cost of connecting to a transmission line in $/MW,
            by default 14000
        line_cost : float, optional
            Cost of building transmission line during connection in $/MW-km,
            by default 2279
        station_tie_in_cost : float, optional
            Cost of connecting to a substation in $/MW,
            by default 0
        center_tie_in_cost : float, optional
            Cost of connecting to a load center in $/MW,
            by default 0
        sink_tie_in_cost : float, optional
            Cost of connecting to a synthetic load center (infinite sink)
            in $/MW, by default 1e9
        avail_cap_frac : float, optional
            Fraction of capacity that is available for connection, by default 1
        line_limited : bool, optional
            Substation connection is limited by maximum capacity of the
            attached lines, legacy method, by default False

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
                          avail_cap_frac=avail_cap_frac,
                          line_limited=line_limited)

            costs = []
            for _, row in trans_table.iterrows():
                tm = row.get('transmission_multiplier', 1)
                costs.append(feature.cost(row[SupplyCurveField.TRANS_GID],
                                          row[SupplyCurveField.DIST_SPUR_KM],
                                          capacity=capacity,
                                          transmission_multiplier=tm))
        except Exception:
            logger.exception("Error computing costs for all connections in {}"
                             .format(cls))
            raise

        return np.array(costs, dtype='float32')
