# -*- coding: utf-8 -*-
"""
reV supply curve module
- Calculation of LCOT
- Supply Curve creation
"""
import json
import logging
import os
from itertools import chain
from copy import deepcopy
from warnings import warn

import numpy as np
import pandas as pd
from rex import Resource
from rex.utilities import SpawnProcessPool, parse_table

from reV.handlers.transmission import TransmissionCosts as TC
from reV.handlers.transmission import TransmissionFeatures as TF
from reV.supply_curve.competitive_wind_farms import CompetitiveWindFarms
from reV.utilities import SupplyCurveField, log_versions
from reV.utilities.exceptions import SupplyCurveError, SupplyCurveInputError

logger = logging.getLogger(__name__)


# map is column name to relative order in which it should appear in output file
_REQUIRED_COMPUTE_AND_OUTPUT_COLS = {
    SupplyCurveField.TRANS_GID: 0,
    SupplyCurveField.TRANS_TYPE: 1,
    SupplyCurveField.N_PARALLEL_TRANS: 2,
    SupplyCurveField.DIST_SPUR_KM: 3,
    SupplyCurveField.TOTAL_TRANS_CAP_COST_PER_MW: 10,
    SupplyCurveField.LCOT: 11,
    SupplyCurveField.TOTAL_LCOE: 12,
}
_REQUIRED_OUTPUT_COLS = {SupplyCurveField.DIST_EXPORT_KM: 4,
                         SupplyCurveField.REINFORCEMENT_DIST_KM: 5,
                         SupplyCurveField.TIE_LINE_COST_PER_MW: 6,
                         SupplyCurveField.CONNECTION_COST_PER_MW: 7,
                         SupplyCurveField.EXPORT_COST_PER_MW: 8,
                         SupplyCurveField.REINFORCEMENT_COST_PER_MW: 9,
                         SupplyCurveField.POI_LAT: 13,
                         SupplyCurveField.POI_LON: 14,
                         SupplyCurveField.REINFORCEMENT_POI_LAT: 15,
                         SupplyCurveField.REINFORCEMENT_POI_LON: 16}
DEFAULT_COLUMNS = tuple(str(field)
                        for field in chain(_REQUIRED_COMPUTE_AND_OUTPUT_COLS,
                                           _REQUIRED_OUTPUT_COLS))
"""Default output columns from supply chain computation (not ordered)"""


class SupplyCurve:
    """SupplyCurve"""

    def __init__(self, sc_points, trans_table, sc_features=None,
                 # str() to fix docs
                 sc_capacity_col=str(SupplyCurveField.CAPACITY_AC_MW)):
        """ReV LCOT calculation and SupplyCurve sorting class.

        ``reV`` supply curve computes the transmission costs associated
        with each supply curve point output by ``reV`` supply curve
        aggregation. Transmission costs can either be computed
        competitively (where total capacity remaining on the
        transmission grid is tracked and updated after each new
        connection) or non-competitively (where the cheapest connections
        for each supply curve point are allowed regardless of the
        remaining transmission grid capacity). In both cases, the
        permutation of transmission costs between supply curve points
        and transmission grid features should be computed using the
        `reVX Least Cost Transmission Paths
        <https://github.com/NREL/reVX/tree/main/reVX/least_cost_xmission>`_
        utility.

        Parameters
        ----------
        sc_points : str | pandas.DataFrame
            Path to CSV or JSON or DataFrame containing supply curve
            point summary. Can also be a filepath to a ``reV`` bespoke
            HDF5 output file where the ``meta`` dataset has the same
            format as the supply curve aggregation output.

            .. Note:: If executing ``reV`` from the command line, this
              input can also be ``"PIPELINE"`` to parse the output of
              the previous pipeline step and use it as input to this
              call. However, note that duplicate executions of any
              preceding commands within the pipeline may invalidate this
              parsing, meaning the `sc_points` input will have to be
              specified manually.

        trans_table : str | pandas.DataFrame | list
            Path to CSV or JSON or DataFrame containing supply curve
            transmission mapping. This can also be a list of
            transmission tables with different line voltage (capacity)
            ratings. See the `reVX Least Cost Transmission Paths
            <https://github.com/NREL/reVX/tree/main/reVX/least_cost_xmission>`_
            utility to generate these input tables.
        sc_features : str | pandas.DataFrame, optional
            Path to CSV or JSON or DataFrame containing additional
            supply curve features (e.g. transmission multipliers,
            regions, etc.). These features will be merged to the
            `sc_points` input table on ALL columns that both have in
            common. If ``None``, no extra supply curve features are
            added. By default, ``None``.
        sc_capacity_col : str, optional
            Name of capacity column in `trans_sc_table`. The values in
            this column determine the size of transmission lines built.
            The transmission capital costs per MW and the reinforcement
            costs per MW will be returned in terms of these capacity
            values. Note that if this column != "capacity", then
            "capacity" must also be included in `trans_sc_table` since
            those values match the "mean_cf" data (which is used to
            calculate LCOT and Total LCOE). This input can be used to,
            e.g., size transmission lines based on solar AC capacity (
            ``sc_capacity_col="capacity_ac"``). By default,
            ``"capacity"``.

        Examples
        --------
        Standard outputs in addition to the values provided in
        `sc_points`, produced by
        :class:`reV.supply_curve.sc_aggregation.SupplyCurveAggregation`:

            - transmission_multiplier : int | float
                Transmission cost multiplier that scales the line cost
                but not the tie-in cost in the calculation of LCOT.
            - trans_gid : int
                Unique transmission feature identifier that each supply
                curve point was connected to.
            - trans_capacity : float
                Total capacity (not available capacity) of the
                transmission feature that each supply curve point was
                connected to. Default units are MW.
            - trans_type : str
                Tranmission feature type that each supply curve point
                was connected to (e.g. Transline, Substation).
            - trans_cap_cost_per_mw : float
                Capital cost of connecting each supply curve point to
                their respective transmission feature. This value
                includes line cost with transmission_multiplier and the
                tie-in cost. Default units are $/MW.
            - dist_km : float
                Distance in km from supply curve point to transmission
                connection.
            - lcot : float
                Levelized cost of connecting to transmission ($/MWh).
            - total_lcoe : float
                Total LCOE of each supply curve point (mean_lcoe + lcot)
                ($/MWh).
            - total_lcoe_friction : float
                Total LCOE of each supply curve point considering the
                LCOE friction scalar from the aggregation step
                (mean_lcoe_friction + lcot) ($/MWh).
        """
        log_versions(logger)
        logger.info("Supply curve points input: {}".format(sc_points))
        logger.info("Transmission table input: {}".format(trans_table))
        logger.info("Supply curve capacity column: {}".format(sc_capacity_col))

        self._sc_capacity_col = sc_capacity_col
        self._sc_points = self._parse_sc_points(
            sc_points, sc_features=sc_features
        )
        self._trans_table = self._map_tables(
            self._sc_points, trans_table, sc_capacity_col=sc_capacity_col
        )
        self._sc_gids, self._mask = self._parse_sc_gids(self._trans_table)

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
    def _parse_sc_points(sc_points, sc_features=None):
        """
        Import supply curve point summary and add any additional features

        Parameters
        ----------
        sc_points : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing supply curve point
            summary. Can also now be a filepath to a bespoke h5 where the
            "meta" dataset has the same format as the sc aggregation output.
        sc_features : str | pandas.DataFrame
            Path to .csv or .json or DataFrame containing additional supply
            curve features, e.g. transmission multipliers, regions

        Returns
        -------
        sc_points : pandas.DataFrame
            DataFrame of supply curve point summary with additional features
            added if supplied
        """
        if isinstance(sc_points, str) and sc_points.endswith(".h5"):
            with Resource(sc_points) as res:
                sc_points = res.meta
                sc_points.index.name = SupplyCurveField.SC_GID
                sc_points = sc_points.reset_index()
        else:
            sc_points = parse_table(sc_points)
            sc_points = sc_points.rename(
                columns=SupplyCurveField.map_from_legacy())

        logger.debug(
            "Supply curve points table imported with columns: {}".format(
                sc_points.columns.values.tolist()
            )
        )

        if sc_features is not None:
            sc_features = parse_table(sc_features)
            sc_features = sc_features.rename(
                columns=SupplyCurveField.map_from_legacy())
            merge_cols = [c for c in sc_features if c in sc_points]
            sc_points = sc_points.merge(sc_features, on=merge_cols, how="left")
            logger.debug(
                "Adding Supply Curve Features table with columns: {}".format(
                    sc_features.columns.values.tolist()
                )
            )

        if "transmission_multiplier" in sc_points:
            col = "transmission_multiplier"
            sc_points.loc[:, col] = sc_points.loc[:, col].fillna(1)

        logger.debug(
            "Final supply curve points table has columns: {}".format(
                sc_points.columns.values.tolist()
            )
        )

        return sc_points

    @staticmethod
    def _get_merge_cols(sc_columns, trans_columns):
        """
        Get columns with 'row' or 'col' in them to use for merging

        Parameters
        ----------
        sc_columns : list
            Columns to search
        trans_cols

        Returns
        -------
        merge_cols : dict
            Columns to merge on which maps the sc columns (keys) to the
            corresponding trans table columns (values)
        """
        sc_columns = [c for c in sc_columns if c.startswith("sc_")]
        trans_columns = [c for c in trans_columns if c.startswith("sc_")]
        merge_cols = {}
        for c_val in ["row", "col"]:
            trans_col = [c for c in trans_columns if c_val in c]
            sc_col = [c for c in sc_columns if c_val in c]
            if trans_col and sc_col:
                merge_cols[sc_col[0]] = trans_col[0]

        if len(merge_cols) != 2:
            msg = (
                "Did not find a unique set of sc row and column ids to "
                "merge on: {}".format(merge_cols)
            )
            logger.error(msg)
            raise RuntimeError(msg)

        return merge_cols

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

        trans_table = parse_table(trans_table)

        # Update legacy transmission table columns to match new less ambiguous
        # column names:
        # trans_gid -> the transmission feature id, legacy name: trans_line_gid
        # trans_line_gids -> gids of transmission lines connected to the given
        # transmission feature (only used for Substations),
        # legacy name: trans_gids
        # also xformer_cost_p_mw -> xformer_cost_per_mw (not sure why there
        # would be a *_p_mw but here we are...)
        rename_map = {
            "trans_line_gid": SupplyCurveField.TRANS_GID,
            "trans_gids": "trans_line_gids",
            "xformer_cost_p_mw": "xformer_cost_per_mw",
        }
        trans_table = trans_table.rename(columns=rename_map)

        contains_dist_in_miles = "dist_mi" in trans_table
        missing_km_dist = SupplyCurveField.DIST_SPUR_KM not in trans_table
        if contains_dist_in_miles and missing_km_dist:
            trans_table = trans_table.rename(
                columns={"dist_mi": SupplyCurveField.DIST_SPUR_KM}
            )
            trans_table[SupplyCurveField.DIST_SPUR_KM] *= 1.60934

        drop_cols = [SupplyCurveField.SC_GID, 'cap_left',
                     SupplyCurveField.SC_POINT_GID]
        drop_cols = [c for c in drop_cols if c in trans_table]
        if drop_cols:
            trans_table = trans_table.drop(columns=drop_cols)

        return trans_table.rename(columns=SupplyCurveField.map_from_legacy())

    @staticmethod
    def _map_trans_capacity(trans_sc_table,
                            sc_capacity_col=SupplyCurveField.CAPACITY_AC_MW):
        """
        Map SC gids to transmission features based on capacity. For any SC
        gids with capacity > the maximum transmission feature capacity, map
        SC gids to the feature with the largest capacity

        Parameters
        ----------
        trans_sc_table : pandas.DataFrame
            Table mapping supply curve points to transmission features.
        sc_capacity_col : str, optional
            Name of capacity column in `trans_sc_table`. The values in
            this column determine the size of transmission lines built.
            The transmission capital costs per MW and the reinforcement
            costs per MW will be returned in terms of these capacity
            values. Note that if this column != "capacity", then
            "capacity" must also be included in `trans_sc_table` since
            those values match the "mean_cf" data (which is used to
            calculate LCOT and Total LCOE). By default, ``"capacity"``.

        Returns
        -------
        trans_sc_table : pandas.DataFrame
            Updated table mapping supply curve points to transmission features
            based on maximum capacity
        """

        nx = trans_sc_table[sc_capacity_col] / trans_sc_table["max_cap"]
        nx = np.ceil(nx).astype(int)
        trans_sc_table[SupplyCurveField.N_PARALLEL_TRANS] = nx

        if (nx > 1).any():
            mask = nx > 1
            tie_line_cost = (
                trans_sc_table.loc[mask, "tie_line_cost"] * nx[mask]
            )

            xformer_cost = (
                trans_sc_table.loc[mask, "xformer_cost_per_mw"]
                * trans_sc_table.loc[mask, "max_cap"]
                * nx[mask]
            )

            conn_cost = (
                xformer_cost
                + trans_sc_table.loc[mask, "sub_upgrade_cost"]
                + trans_sc_table.loc[mask, "new_sub_cost"]
            )

            trans_cap_cost = tie_line_cost + conn_cost

            trans_sc_table.loc[mask, "tie_line_cost"] = tie_line_cost
            trans_sc_table.loc[mask, "xformer_cost"] = xformer_cost
            trans_sc_table.loc[mask, "connection_cost"] = conn_cost
            trans_sc_table.loc[mask, "trans_cap_cost"] = trans_cap_cost

            msg = (
                "{} SC points have a capacity that exceeds the maximum "
                "transmission feature capacity and will be connected with "
                "multiple parallel transmission features.".format(
                    (nx > 1).sum()
                )
            )
            logger.info(msg)

        return trans_sc_table

    @staticmethod
    def _parse_trans_line_gids(trans_line_gids):
        """
        Parse json string of trans_line_gids if needed

        Parameters
        ----------
        trans_line_gids : str | list
            list of transmission line 'trans_gid's, if a json string, convert
            to list

        Returns
        -------
        trans_line_gids : list
            list of transmission line 'trans_gid's
        """
        if isinstance(trans_line_gids, str):
            trans_line_gids = json.loads(trans_line_gids)

        return trans_line_gids

    @classmethod
    def _check_sub_trans_lines(cls, features):
        """
        Check to make sure all trans-lines are available for all sub-stations

        Parameters
        ----------
        features : pandas.DataFrame
            Table of transmission feature to check substation to transmission
            line gid connections

        Returns
        -------
        line_gids : list
            List of missing transmission line 'trans_gid's for all substations
            in features table
        """
        features = features.rename(
            columns={
                "trans_line_gid": SupplyCurveField.TRANS_GID,
                "trans_gids": "trans_line_gids",
            }
        )
        mask = (features[SupplyCurveField.TRANS_TYPE].str.casefold()
                == "substation")

        if not any(mask):
            return []

        line_gids = features.loc[mask, "trans_line_gids"].apply(
            cls._parse_trans_line_gids
        )

        line_gids = np.unique(np.concatenate(line_gids.values))

        test = np.isin(line_gids, features[SupplyCurveField.TRANS_GID].values)

        return line_gids[~test].tolist()

    @classmethod
    def _check_substation_conns(cls, trans_table,
                                sc_cols=SupplyCurveField.SC_GID):
        """
        Run checks on substation transmission features to make sure that
        every sc point connecting to a substation can also connect to its
        respective transmission lines

        Parameters
        ----------
        trans_table : pd.DataFrame
            Table mapping supply curve points to transmission features
            (should already be merged with SC points).
        sc_cols : str | list, optional
            Column(s) in trans_table with unique supply curve id,
            by default SupplyCurveField.SC_GID
        """
        missing = {}
        for sc_point, sc_table in trans_table.groupby(sc_cols):
            tl_gids = cls._check_sub_trans_lines(sc_table)
            if tl_gids:
                missing[sc_point] = tl_gids

        if any(missing):
            msg = (
                "The following sc_gid (keys) were connected to substations "
                "but were not connected to the respective transmission line"
                " gids (values) which is required for full SC sort: {}".format(
                    missing
                )
            )
            logger.error(msg)
            raise SupplyCurveInputError(msg)

    @classmethod
    def _check_sc_trans_table(cls, sc_points, trans_table):
        """Run self checks on sc_points table and the merged trans_table

        Parameters
        ----------
        sc_points : pd.DataFrame
            Table of supply curve point summary
        trans_table : pd.DataFrame
            Table mapping supply curve points to transmission features
            (should already be merged with SC points).
        """
        sc_gids = set(sc_points[SupplyCurveField.SC_GID].unique())
        trans_sc_gids = set(trans_table[SupplyCurveField.SC_GID].unique())
        missing = sorted(list(sc_gids - trans_sc_gids))
        if any(missing):
            msg = (
                "There are {} Supply Curve points with missing "
                "transmission mappings. Supply curve points with no "
                "transmission features will not be connected! "
                "Missing sc_gid's: {}".format(len(missing), missing)
            )
            logger.warning(msg)
            warn(msg)

        if not any(trans_sc_gids) or not any(sc_gids):
            msg = (
                "Merging of sc points table and transmission features "
                "table failed with {} original sc gids and {} transmission "
                "sc gids after table merge.".format(
                    len(sc_gids), len(trans_sc_gids)
                )
            )
            logger.error(msg)
            raise SupplyCurveError(msg)

        logger.debug(
            "There are {} original SC gids and {} sc gids in the "
            "merged transmission table.".format(
                len(sc_gids), len(trans_sc_gids)
            )
        )
        logger.debug(
            "Transmission Table created with columns: {}".format(
                trans_table.columns.values.tolist()
            )
        )

    @classmethod
    def _merge_sc_trans_tables(cls, sc_points, trans_table,
                               sc_cols=(SupplyCurveField.SC_GID,
                                        SupplyCurveField.CAPACITY_AC_MW,
                                        SupplyCurveField.MEAN_CF_AC,
                                        SupplyCurveField.MEAN_LCOE),
                               sc_capacity_col=SupplyCurveField.CAPACITY_AC_MW
                               ):
        """
        Merge the supply curve table with the transmission features table.

        Parameters
        ----------
        sc_points : pd.DataFrame
            Table of supply curve point summary
        trans_table : pd.DataFrame | str
            Table mapping supply curve points to transmission features
            (either str filepath to table file, list of filepaths to tables by
             line voltage (capacity) or pre-loaded dataframe).
        sc_cols : tuple | list, optional
            List of column from sc_points to transfer into the trans table,
            If the `sc_capacity_col` is not included, it will get added.
            by default (SupplyCurveField.SC_GID, 'capacity', 'mean_cf',
            'mean_lcoe')
        sc_capacity_col : str, optional
            Name of capacity column in `trans_sc_table`. The values in
            this column determine the size of transmission lines built.
            The transmission capital costs per MW and the reinforcement
            costs per MW will be returned in terms of these capacity
            values. Note that if this column != "capacity", then
            "capacity" must also be included in `trans_sc_table` since
            those values match the "mean_cf" data (which is used to
            calculate LCOT and Total LCOE). By default, ``"capacity"``.

        Returns
        -------
        trans_sc_table : pd.DataFrame
            Updated table mapping supply curve points to transmission features.
            This is performed by an inner merging with trans_table
        """
        if sc_capacity_col not in sc_cols:
            sc_cols = tuple([sc_capacity_col] + list(sc_cols))

        if isinstance(trans_table, (list, tuple)):
            trans_sc_table = []
            for table in trans_table:
                trans_sc_table.append(
                    cls._merge_sc_trans_tables(
                        sc_points,
                        table,
                        sc_cols=sc_cols,
                        sc_capacity_col=sc_capacity_col,
                    )
                )

            trans_sc_table = pd.concat(trans_sc_table)
        else:
            trans_table = cls._parse_trans_table(trans_table)

            merge_cols = cls._get_merge_cols(
                sc_points.columns, trans_table.columns
            )
            logger.info(
                "Merging SC table and Trans Table with "
                "{} mapping: {}".format(
                    "sc_table_col: trans_table_col", merge_cols
                )
            )
            sc_points = sc_points.rename(columns=merge_cols)
            merge_cols = list(merge_cols.values())

            if isinstance(sc_cols, tuple):
                sc_cols = list(sc_cols)

            extra_cols = [SupplyCurveField.CAPACITY_DC_MW,
                          SupplyCurveField.MEAN_CF_DC,
                          SupplyCurveField.MEAN_LCOE_FRICTION,
                          "transmission_multiplier"]
            for col in extra_cols:
                if col in sc_points:
                    sc_cols.append(col)

            sc_cols += merge_cols
            sc_points = sc_points[sc_cols].copy()
            trans_sc_table = trans_table.merge(
                sc_points, on=merge_cols, how="inner"
            )

        return trans_sc_table

    @classmethod
    def _map_tables(cls, sc_points, trans_table,
                    sc_cols=(SupplyCurveField.SC_GID,
                             SupplyCurveField.CAPACITY_AC_MW,
                             SupplyCurveField.MEAN_CF_AC,
                             SupplyCurveField.MEAN_LCOE),
                    sc_capacity_col=SupplyCurveField.CAPACITY_AC_MW):
        """
        Map supply curve points to transmission features

        Parameters
        ----------
        sc_points : pd.DataFrame
            Table of supply curve point summary
        trans_table : pd.DataFrame | str
            Table mapping supply curve points to transmission features
            (either str filepath to table file, list of filepaths to tables by
             line voltage (capacity) or pre-loaded DataFrame).
        sc_cols : tuple | list, optional
            List of column from sc_points to transfer into the trans table,
            If the `sc_capacity_col` is not included, it will get added.
            by default (SupplyCurveField.SC_GID,
            SupplyCurveField.CAPACITY_AC_MW, SupplyCurveField.MEAN_CF_AC,
            SupplyCurveField.MEAN_LCOE)
        sc_capacity_col : str, optional
            Name of capacity column in `trans_sc_table`. The values in
            this column determine the size of transmission lines built.
            The transmission capital costs per MW and the reinforcement
            costs per MW will be returned in terms of these capacity
            values. Note that if this column != "capacity", then
            "capacity" must also be included in `trans_sc_table` since
            those values match the "mean_cf" data (which is used to
            calculate LCOT and Total LCOE). By default, ``"capacity"``.

        Returns
        -------
        trans_sc_table : pd.DataFrame
            Updated table mapping supply curve points to transmission features.
            This is performed by an inner merging with trans_table
        """
        scc = sc_capacity_col
        trans_sc_table = cls._merge_sc_trans_tables(
            sc_points, trans_table, sc_cols=sc_cols, sc_capacity_col=scc
        )

        if "max_cap" in trans_sc_table:
            trans_sc_table = cls._map_trans_capacity(
                trans_sc_table, sc_capacity_col=scc
            )

        sort_cols = [SupplyCurveField.SC_GID, SupplyCurveField.TRANS_GID]
        trans_sc_table = trans_sc_table.sort_values(sort_cols)
        trans_sc_table = trans_sc_table.reset_index(drop=True)

        cls._check_sc_trans_table(sc_points, trans_sc_table)

        return trans_sc_table

    @staticmethod
    def _create_handler(trans_table, trans_costs=None, avail_cap_frac=1):
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
        avail_cap_frac: int, optional
            Fraction of transmissions features capacity 'ac_cap' to make
            available for connection to supply curve points, by default 1

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

        trans_features = TF(
            trans_table, avail_cap_frac=avail_cap_frac, **kwargs
        )

        return trans_features

    @staticmethod
    def _parse_sc_gids(trans_table, gid_key=SupplyCurveField.SC_GID):
        """Extract unique sc gids, make bool mask from tranmission table

        Parameters
        ----------
        trans_table : pd.DataFrame
            reV Supply Curve table joined with transmission features table.
        gid_key : str
            Column label in trans_table containing the supply curve points
            primary key.

        Returns
        -------
        sc_gids : list
            List of unique integer supply curve gids (non-nan)
        mask : np.ndarray
            Boolean array initialized as true. Length is equal to the maximum
            SC gid so that the SC gids can be used to index the mask directly.
        """
        sc_gids = list(np.sort(trans_table[gid_key].unique()))
        sc_gids = [int(gid) for gid in sc_gids]
        mask = np.ones(int(1 + max(sc_gids)), dtype=bool)

        return sc_gids, mask

    @staticmethod
    def _get_capacity(sc_gid, sc_table, connectable=True,
                      sc_capacity_col=SupplyCurveField.CAPACITY_AC_MW):
        """
        Get capacity of supply curve point

        Parameters
        ----------
        sc_gid : int
            Supply curve gid
        sc_table : pandas.DataFrame
            DataFrame of sc point to transmission features mapping for given
            sc_gid
        connectable : bool, optional
            Flag to ensure SC point can connect to transmission features,
            by default True
        sc_capacity_col : str, optional
            Name of capacity column in `trans_sc_table`. The values in
            this column determine the size of transmission lines built.
            The transmission capital costs per MW and the reinforcement
            costs per MW will be returned in terms of these capacity
            values. Note that if this column != "capacity", then
            "capacity" must also be included in `trans_sc_table` since
            those values match the "mean_cf" data (which is used to
            calculate LCOT and Total LCOE). By default, ``"capacity"``.

        Returns
        -------
        capacity : float
            Capacity of supply curve point
        """
        if connectable:
            capacity = sc_table[sc_capacity_col].unique()
            if len(capacity) == 1:
                capacity = capacity[0]
            else:
                msg = (
                    "Each supply curve point should only have "
                    "a single capacity, but {} has {}".format(sc_gid, capacity)
                )
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            capacity = None

        return capacity

    @classmethod
    def _compute_trans_cap_cost(cls, trans_table, trans_costs=None,
                                avail_cap_frac=1, max_workers=None,
                                connectable=True, line_limited=False,
                                sc_capacity_col=(
                                    SupplyCurveField.CAPACITY_AC_MW)):
        """
        Compute levelized cost of transmission for all combinations of
        supply curve points and tranmission features in trans_table

        Parameters
        ----------
        trans_table : pd.DataFrame
            Table mapping supply curve points to transmission features
            MUST contain `sc_capacity_col` column.
        fcr : float
            Fixed charge rate needed to compute LCOT
        trans_costs : str | dict
            Transmission feature costs to use with TransmissionFeatures
            handler: line_tie_in_cost, line_cost, station_tie_in_cost,
            center_tie_in_cost, sink_tie_in_cost
        avail_cap_frac: int, optional
            Fraction of transmissions features capacity 'ac_cap' to make
            available for connection to supply curve points, by default 1
        max_workers : int | NoneType
            Number of workers to use to compute lcot, if > 1 run in parallel.
            None uses all available cpu's.
        connectable : bool, optional
            Flag to only compute tranmission capital cost if transmission
            feature has enough available capacity, by default True
        line_limited : bool
            Substation connection is limited by maximum capacity of the
            attached lines, legacy method
        sc_capacity_col : str, optional
            Name of capacity column in `trans_sc_table`. The values in
            this column determine the size of transmission lines built.
            The transmission capital costs per MW and the reinforcement
            costs per MW will be returned in terms of these capacity
            values. Note that if this column != "capacity", then
            "capacity" must also be included in `trans_sc_table` since
            those values match the "mean_cf" data (which is used to
            calculate LCOT and Total LCOE). By default, ``"capacity"``.

        Returns
        -------
        lcot : list
            Levelized cost of transmission for all supply curve -
            tranmission feature connections
        cost : list
            Capital cost of tramsmission for all supply curve - transmission
            feature connections
        """
        scc = sc_capacity_col
        if scc not in trans_table:
            raise SupplyCurveInputError(
                "Supply curve table must have "
                "supply curve point capacity column"
                "({}) to compute lcot".format(scc)
            )

        if trans_costs is not None:
            trans_costs = TF._parse_dictionary(trans_costs)
        else:
            trans_costs = {}

        if max_workers is None:
            max_workers = os.cpu_count()

        logger.info('Computing LCOT costs for all possible connections...')
        groups = trans_table.groupby(SupplyCurveField.SC_GID)
        if max_workers > 1:
            loggers = [__name__, "reV.handlers.transmission", "reV"]
            with SpawnProcessPool(
                max_workers=max_workers, loggers=loggers
            ) as exe:
                futures = []
                for sc_gid, sc_table in groups:
                    capacity = cls._get_capacity(
                        sc_gid,
                        sc_table,
                        connectable=connectable,
                        sc_capacity_col=scc,
                    )
                    futures.append(
                        exe.submit(
                            TC.feature_costs,
                            sc_table,
                            capacity=capacity,
                            avail_cap_frac=avail_cap_frac,
                            line_limited=line_limited,
                            **trans_costs,
                        )
                    )

                cost = [future.result() for future in futures]
        else:
            cost = []
            for sc_gid, sc_table in groups:
                capacity = cls._get_capacity(
                    sc_gid,
                    sc_table,
                    connectable=connectable,
                    sc_capacity_col=scc,
                )
                cost.append(
                    TC.feature_costs(
                        sc_table,
                        capacity=capacity,
                        avail_cap_frac=avail_cap_frac,
                        line_limited=line_limited,
                        **trans_costs,
                    )
                )

        cost = np.hstack(cost).astype("float32")
        logger.info("LCOT cost calculation is complete.")

        return cost

    def compute_total_lcoe(
        self,
        fcr,
        transmission_costs=None,
        avail_cap_frac=1,
        line_limited=False,
        connectable=True,
        max_workers=None,
        consider_friction=True,
    ):
        """
        Compute LCOT and total LCOE for all sc point to transmission feature
        connections

        Parameters
        ----------
        fcr : float
            Fixed charge rate, used to compute LCOT
        transmission_costs : str | dict, optional
            Transmission feature costs to use with TransmissionFeatures
            handler: line_tie_in_cost, line_cost, station_tie_in_cost,
            center_tie_in_cost, sink_tie_in_cost, by default None
        avail_cap_frac : int, optional
            Fraction of transmissions features capacity 'ac_cap' to make
            available for connection to supply curve points, by default 1
        line_limited : bool, optional
            Flag to have substation connection is limited by maximum capacity
            of the attached lines, legacy method, by default False
        connectable : bool, optional
            Flag to only compute tranmission capital cost if transmission
            feature has enough available capacity, by default True
        max_workers : int | NoneType, optional
            Number of workers to use to compute lcot, if > 1 run in parallel.
            None uses all available cpu's. by default None
        consider_friction : bool, optional
            Flag to consider friction layer on LCOE when "mean_lcoe_friction"
            is in the sc points input, by default True
        """
        tcc_per_mw_col = SupplyCurveField.TOTAL_TRANS_CAP_COST_PER_MW
        if tcc_per_mw_col in self._trans_table:
            cost = self._trans_table[tcc_per_mw_col].values.copy()
        elif "trans_cap_cost" not in self._trans_table:
            scc = self._sc_capacity_col
            cost = self._compute_trans_cap_cost(
                self._trans_table,
                trans_costs=transmission_costs,
                avail_cap_frac=avail_cap_frac,
                line_limited=line_limited,
                connectable=connectable,
                max_workers=max_workers,
                sc_capacity_col=scc,
            )
            self._trans_table[tcc_per_mw_col] = cost  # $/MW
        else:
            cost = self._trans_table["trans_cap_cost"].values.copy()  # $
            cost /= self._trans_table[SupplyCurveField.CAPACITY_AC_MW]  # $/MW
            self._trans_table[tcc_per_mw_col] = cost

        self._trans_table[tcc_per_mw_col] = (
            self._trans_table[tcc_per_mw_col].astype("float32")
        )
        cost = cost.astype("float32")
        cf_mean_arr = self._trans_table[SupplyCurveField.MEAN_CF_AC]
        cf_mean_arr = cf_mean_arr.values.astype("float32")
        resource_lcoe = self._trans_table[SupplyCurveField.MEAN_LCOE]
        resource_lcoe = resource_lcoe.values.astype("float32")

        if 'reinforcement_cost_floored_per_mw' in self._trans_table:
            logger.info("'reinforcement_cost_floored_per_mw' column found in "
                        "transmission table. Adding floored reinforcement "
                        "cost LCOE as sorting option.")
            fr_cost = (self._trans_table['reinforcement_cost_floored_per_mw']
                       .values.copy())

            lcot_fr = ((cost + fr_cost) * fcr) / (cf_mean_arr * 8760)
            lcoe_fr = lcot_fr + resource_lcoe
            self._trans_table['lcot_floored_reinforcement'] = lcot_fr
            self._trans_table['lcoe_floored_reinforcement'] = lcoe_fr

        if SupplyCurveField.REINFORCEMENT_COST_PER_MW in self._trans_table:
            logger.info("%s column found in transmission table. Adding "
                        "reinforcement costs to total LCOE.",
                        SupplyCurveField.REINFORCEMENT_COST_PER_MW)
            lcot_nr = (cost * fcr) / (cf_mean_arr * 8760)
            lcoe_nr = lcot_nr + resource_lcoe
            self._trans_table['lcot_no_reinforcement'] = lcot_nr
            self._trans_table['lcoe_no_reinforcement'] = lcoe_nr

            col_name = SupplyCurveField.REINFORCEMENT_COST_PER_MW
            r_cost = self._trans_table[col_name].astype("float32")
            r_cost = r_cost.values.copy()
            self._trans_table[tcc_per_mw_col] += r_cost
            cost += r_cost  # $/MW

        lcot = (cost * fcr) / (cf_mean_arr * 8760)
        self._trans_table[SupplyCurveField.LCOT] = lcot
        self._trans_table[SupplyCurveField.TOTAL_LCOE] = lcot + resource_lcoe

        if consider_friction:
            self._calculate_total_lcoe_friction()

    def _calculate_total_lcoe_friction(self):
        """Look for site mean LCOE with friction in the trans table and if
        found make a total LCOE column with friction."""

        if SupplyCurveField.MEAN_LCOE_FRICTION in self._trans_table:
            lcoe_friction = (
                self._trans_table[SupplyCurveField.LCOT]
                + self._trans_table[SupplyCurveField.MEAN_LCOE_FRICTION])
            self._trans_table[SupplyCurveField.TOTAL_LCOE_FRICTION] = (
                lcoe_friction
            )
            logger.info('Found mean LCOE with friction. Adding key '
                        '"total_lcoe_friction" to trans table.')

    def _exclude_noncompetitive_wind_farms(
        self, comp_wind_dirs, sc_gid, downwind=False
    ):
        """
        Exclude non-competitive wind farms for given sc_gid

        Parameters
        ----------
        comp_wind_dirs : CompetitiveWindFarms
            Pre-initilized CompetitiveWindFarms instance
        sc_gid : int
            Supply curve gid to exclude non-competitive wind farms around
        downwind : bool, optional
            Flag to remove downwind neighbors as well as upwind neighbors,
            by default False

        Returns
        -------
        comp_wind_dirs : CompetitiveWindFarms
            updated CompetitiveWindFarms instance
        """
        gid = comp_wind_dirs.check_sc_gid(sc_gid)
        if gid is not None:
            if comp_wind_dirs.mask[gid]:
                exclude_gids = comp_wind_dirs["upwind", gid]
                if downwind:
                    exclude_gids = np.append(
                        exclude_gids, comp_wind_dirs["downwind", gid]
                    )
                for n in exclude_gids:
                    check = comp_wind_dirs.exclude_sc_point_gid(n)
                    if check:
                        sc_gids = comp_wind_dirs[SupplyCurveField.SC_GID, n]
                        for sc_id in sc_gids:
                            if self._mask[sc_id]:
                                logger.debug(
                                    "Excluding sc_gid {}".format(sc_id)
                                )
                                self._mask[sc_id] = False

        return comp_wind_dirs

    @staticmethod
    def add_sum_cols(table, sum_cols):
        """Add a summation column to table.

        Parameters
        ----------
        table : pd.DataFrame
            Supply curve table.
        sum_cols : dict
            Mapping of new column label(s) to multiple column labels to sum.
            Example: sum_col={'total_cap_cost': ['cap_cost1', 'cap_cost2']}
            Which would add a new 'total_cap_cost' column which would be the
            sum of 'cap_cost1' and 'cap_cost2' if they are present in table.

        Returns
        -------
        table : pd.DataFrame
            Supply curve table with additional summation columns.
        """

        for new_label, sum_labels in sum_cols.items():
            missing = [s for s in sum_labels if s not in table]

            if any(missing):
                logger.info(
                    'Could not make sum column "{}", missing: {}'.format(
                        new_label, missing
                    )
                )
            else:
                sum_arr = np.zeros(len(table))
                for s in sum_labels:
                    temp = table[s].values
                    temp[np.isnan(temp)] = 0
                    sum_arr += temp

                table[new_label] = sum_arr

        return table

    def _full_sort(  # noqa: C901
        self,
        trans_table,
        trans_costs=None,
        avail_cap_frac=1,
        comp_wind_dirs=None,
        total_lcoe_fric=None,
        sort_on=SupplyCurveField.TOTAL_LCOE,
        columns=(
            SupplyCurveField.TRANS_GID,
            SupplyCurveField.TRANS_CAPACITY,
            SupplyCurveField.TRANS_TYPE,
            SupplyCurveField.TOTAL_TRANS_CAP_COST_PER_MW,
            SupplyCurveField.DIST_SPUR_KM,
            SupplyCurveField.LCOT,
            SupplyCurveField.TOTAL_LCOE,
        ),
        downwind=False,
    ):
        """
        Internal method to handle full supply curve sorting

        Parameters
        ----------
        trans_table : pandas.DataFrame
            Supply Curve Tranmission table to sort on
        trans_costs : str | dict, optional
            Transmission feature costs to use with TransmissionFeatures
            handler: line_tie_in_cost, line_cost, station_tie_in_cost,
            center_tie_in_cost, sink_tie_in_cost, by default None
        avail_cap_frac : int, optional
            Fraction of transmissions features capacity 'ac_cap' to make
            available for connection to supply curve points, by default 1
        comp_wind_dirs : CompetitiveWindFarms, optional
            Pre-initilized CompetitiveWindFarms instance, by default None
        total_lcoe_fric : ndarray, optional
            Vector of lcoe friction values, by default None
        sort_on : str, optional
            Column label to sort the Supply Curve table on. This affects the
            build priority - connections with the lowest value in this column
            will be built first, by default 'total_lcoe'
        columns : tuple, optional
            Columns to preserve in output connections dataframe,
            by default ('trans_gid', 'trans_capacity', 'trans_type',
                        'trans_cap_cost_per_mw', 'dist_km', 'lcot',
                        'total_lcoe')
        downwind : bool, optional
            Flag to remove downwind neighbors as well as upwind neighbors,
            by default False

        Returns
        -------
        supply_curve : pandas.DataFrame
            Updated sc_points table with transmission connections, LCOT
            and LCOE+LCOT based on full supply curve connections
        """
        trans_features = self._create_handler(
            self._trans_table,
            trans_costs=trans_costs,
            avail_cap_frac=avail_cap_frac,
        )
        init_list = [np.nan] * int(1 + np.max(self._sc_gids))
        columns = list(columns)
        if sort_on not in columns:
            columns.append(sort_on)

        conn_lists = {k: deepcopy(init_list) for k in columns}

        trans_sc_gids = trans_table[SupplyCurveField.SC_GID].values.astype(int)

        # syntax is final_key: source_key (source from trans_table)
        all_cols = list(columns)
        essentials = [SupplyCurveField.TRANS_GID,
                      SupplyCurveField.TRANS_CAPACITY,
                      SupplyCurveField.TRANS_TYPE,
                      SupplyCurveField.DIST_SPUR_KM,
                      SupplyCurveField.TOTAL_TRANS_CAP_COST_PER_MW,
                      SupplyCurveField.LCOT,
                      SupplyCurveField.TOTAL_LCOE]

        for col in essentials:
            if col not in all_cols:
                all_cols.append(col)

        arrays = {col: trans_table[col].values for col in all_cols}

        sc_capacities = trans_table[self._sc_capacity_col].values

        connected = 0
        progress = 0
        for i in range(len(trans_table)):
            sc_gid = trans_sc_gids[i]
            if self._mask[sc_gid]:
                connect = trans_features.connect(
                    arrays[SupplyCurveField.TRANS_GID][i], sc_capacities[i]
                )
                if connect:
                    connected += 1
                    logger.debug("Connecting sc gid {}".format(sc_gid))
                    self._mask[sc_gid] = False

                    for col_name, data_arr in arrays.items():
                        conn_lists[col_name][sc_gid] = data_arr[i]

                    if total_lcoe_fric is not None:
                        col_name = SupplyCurveField.TOTAL_LCOE_FRICTION
                        conn_lists[col_name][sc_gid] = total_lcoe_fric[i]

                    current_prog = connected // (len(self) / 100)
                    if current_prog > progress:
                        progress = current_prog
                        logger.info(
                            "{} % of supply curve points connected".format(
                                progress
                            )
                        )

                    if comp_wind_dirs is not None:
                        comp_wind_dirs = (
                            self._exclude_noncompetitive_wind_farms(
                                comp_wind_dirs, sc_gid, downwind=downwind
                            )
                        )

        index = range(0, int(1 + np.max(self._sc_gids)))
        connections = pd.DataFrame(conn_lists, index=index)
        connections.index.name = SupplyCurveField.SC_GID
        connections = connections.dropna(subset=[sort_on])
        connections = connections[columns].reset_index()

        sc_gids = self._sc_points[SupplyCurveField.SC_GID].values
        connected = connections[SupplyCurveField.SC_GID].values
        logger.debug('Connected gids {} out of total supply curve gids {}'
                     .format(len(connected), len(sc_gids)))
        unconnected = ~np.isin(sc_gids, connected)
        unconnected = sc_gids[unconnected].tolist()

        if unconnected:
            msg = (
                "{} supply curve points were not connected to tranmission! "
                "Unconnected sc_gid's: {}".format(
                    len(unconnected), unconnected
                )
            )
            logger.warning(msg)
            warn(msg)

        supply_curve = self._sc_points.merge(
            connections, on=SupplyCurveField.SC_GID)

        return supply_curve.reset_index(drop=True)

    def _check_feature_capacity(self, avail_cap_frac=1):
        """
        Add the transmission connection feature capacity to the trans table if
        needed
        """
        if SupplyCurveField.TRANS_CAPACITY not in self._trans_table:
            kwargs = {"avail_cap_frac": avail_cap_frac}
            fc = TF.feature_capacity(self._trans_table, **kwargs)
            self._trans_table = self._trans_table.merge(
                fc, on=SupplyCurveField.TRANS_GID)

    def _adjust_output_columns(self, columns, consider_friction):
        """Add extra output columns, if needed."""

        for col in _REQUIRED_COMPUTE_AND_OUTPUT_COLS:
            if col not in columns:
                columns.append(col)

        for col in _REQUIRED_OUTPUT_COLS:
            if col not in self._trans_table:
                self._trans_table[col] = np.nan
            if col not in columns:
                columns.append(col)

        missing_cols = [col for col in columns if col not in self._trans_table]
        if missing_cols:
            msg = (f"The following requested columns are not found in "
                   f"transmission table: {missing_cols}.\nSkipping...")
            logger.warning(msg)
            warn(msg)

        columns = [col for col in columns if col in self._trans_table]

        fric_col = SupplyCurveField.TOTAL_LCOE_FRICTION
        if consider_friction and fric_col in self._trans_table:
            columns.append(fric_col)

        return sorted(columns, key=_column_sort_key)

    def _determine_sort_on(self, sort_on):
        """Determine the `sort_on` column from user input and trans table"""
        r_cost_col = SupplyCurveField.REINFORCEMENT_COST_PER_MW
        found_reinforcement_costs = (
            r_cost_col in self._trans_table
            and not self._trans_table[r_cost_col].isna().all()
        )
        if found_reinforcement_costs:
            sort_on = sort_on or "lcoe_no_reinforcement"
        return sort_on or SupplyCurveField.TOTAL_LCOE

    def full_sort(
        self,
        fcr,
        transmission_costs=None,
        avail_cap_frac=1,
        line_limited=False,
        connectable=True,
        max_workers=None,
        consider_friction=True,
        sort_on=None,
        columns=(
            SupplyCurveField.TRANS_GID,
            SupplyCurveField.TRANS_CAPACITY,
            SupplyCurveField.TRANS_TYPE,
            SupplyCurveField.TOTAL_TRANS_CAP_COST_PER_MW,
            SupplyCurveField.DIST_SPUR_KM,
            SupplyCurveField.LCOT,
            SupplyCurveField.TOTAL_LCOE,
        ),
        wind_dirs=None,
        n_dirs=2,
        downwind=False,
        offshore_compete=False,
    ):
        """
        run full supply curve sorting

        Parameters
        ----------
        fcr : float
            Fixed charge rate, used to compute LCOT
        transmission_costs : str | dict, optional
            Transmission feature costs to use with TransmissionFeatures
            handler: line_tie_in_cost, line_cost, station_tie_in_cost,
            center_tie_in_cost, sink_tie_in_cost, by default None
        avail_cap_frac : int, optional
            Fraction of transmissions features capacity 'ac_cap' to make
            available for connection to supply curve points, by default 1
        line_limited : bool, optional
            Flag to have substation connection is limited by maximum capacity
            of the attached lines, legacy method, by default False
        connectable : bool, optional
            Flag to only compute tranmission capital cost if transmission
            feature has enough available capacity, by default True
        max_workers : int | NoneType, optional
            Number of workers to use to compute lcot, if > 1 run in parallel.
            None uses all available cpu's. by default None
        consider_friction : bool, optional
            Flag to consider friction layer on LCOE when "mean_lcoe_friction"
            is in the sc points input, by default True
        sort_on : str, optional
            Column label to sort the Supply Curve table on. This affects the
            build priority - connections with the lowest value in this column
            will be built first, by default `None`, which will use
            total LCOE without any reinforcement costs as the sort value.
        columns : list | tuple, optional
            Columns to preserve in output connections dataframe,
            by default ('trans_gid', 'trans_capacity', 'trans_type',
            'trans_cap_cost_per_mw', 'dist_km', 'lcot', 'total_lcoe')
        wind_dirs : pandas.DataFrame | str, optional
            path to .csv or reVX.wind_dirs.wind_dirs.WindDirs output with
            the neighboring supply curve point gids and power-rose value at
            each cardinal direction, by default None
        n_dirs : int, optional
            Number of prominent directions to use, by default 2
        downwind : bool, optional
            Flag to remove downwind neighbors as well as upwind neighbors,
            by default False
        offshore_compete : bool, default
            Flag as to whether offshore farms should be included during
            CompetitiveWindFarms, by default False

        Returns
        -------
        supply_curve : pandas.DataFrame
            Updated sc_points table with transmission connections, LCOT
            and LCOE+LCOT based on full supply curve connections
        """
        logger.info("Starting full competitive supply curve sort.")
        self._check_substation_conns(self._trans_table)
        self.compute_total_lcoe(
            fcr,
            transmission_costs=transmission_costs,
            avail_cap_frac=avail_cap_frac,
            line_limited=line_limited,
            connectable=connectable,
            max_workers=max_workers,
            consider_friction=consider_friction,
        )
        self._check_feature_capacity(avail_cap_frac=avail_cap_frac)

        if isinstance(columns, tuple):
            columns = list(columns)

        columns = self._adjust_output_columns(columns, consider_friction)
        sort_on = self._determine_sort_on(sort_on)

        trans_table = self._trans_table.copy()
        pos = trans_table[SupplyCurveField.LCOT].isnull()
        trans_table = trans_table.loc[~pos].sort_values(
            [sort_on, SupplyCurveField.TRANS_GID]
        )

        total_lcoe_fric = None
        col_in_table = SupplyCurveField.MEAN_LCOE_FRICTION in trans_table
        if consider_friction and col_in_table:
            total_lcoe_fric = \
                trans_table[SupplyCurveField.TOTAL_LCOE_FRICTION].values

        comp_wind_dirs = None
        if wind_dirs is not None:
            msg = "Excluding {} upwind".format(n_dirs)
            if downwind:
                msg += " and downwind"

            msg += " onshore"
            if offshore_compete:
                msg += " and offshore"

            msg += " windfarms"
            logger.info(msg)
            comp_wind_dirs = CompetitiveWindFarms(
                wind_dirs,
                self._sc_points,
                n_dirs=n_dirs,
                offshore=offshore_compete,
            )

        supply_curve = self._full_sort(
            trans_table,
            trans_costs=transmission_costs,
            avail_cap_frac=avail_cap_frac,
            comp_wind_dirs=comp_wind_dirs,
            total_lcoe_fric=total_lcoe_fric,
            sort_on=sort_on,
            columns=columns,
            downwind=downwind,
        )

        return supply_curve

    def simple_sort(
        self,
        fcr,
        transmission_costs=None,
        avail_cap_frac=1,
        max_workers=None,
        consider_friction=True,
        sort_on=None,
        columns=DEFAULT_COLUMNS,
        wind_dirs=None,
        n_dirs=2,
        downwind=False,
        offshore_compete=False,
    ):
        """
        Run simple supply curve sorting that does not take into account
        available capacity

        Parameters
        ----------
        fcr : float
            Fixed charge rate, used to compute LCOT
        transmission_costs : str | dict, optional
            Transmission feature costs to use with TransmissionFeatures
            handler: line_tie_in_cost, line_cost, station_tie_in_cost,
            center_tie_in_cost, sink_tie_in_cost, by default None
        avail_cap_frac : int, optional
            Fraction of transmissions features capacity 'ac_cap' to make
            available for connection to supply curve points, by default 1
        line_limited : bool, optional
            Flag to have substation connection is limited by maximum capacity
            of the attached lines, legacy method, by default False
        connectable : bool, optional
            Flag to only compute tranmission capital cost if transmission
            feature has enough available capacity, by default True
        max_workers : int | NoneType, optional
            Number of workers to use to compute lcot, if > 1 run in parallel.
            None uses all available cpu's. by default None
        consider_friction : bool, optional
            Flag to consider friction layer on LCOE when "mean_lcoe_friction"
            is in the sc points input, by default True
        sort_on : str, optional
            Column label to sort the Supply Curve table on. This affects the
            build priority - connections with the lowest value in this column
            will be built first, by default `None`, which will use
            total LCOE without any reinforcement costs as the sort value.
        columns : list | tuple, optional
            Columns to preserve in output connections dataframe.
            By default, :obj:`DEFAULT_COLUMNS`.
        wind_dirs : pandas.DataFrame | str, optional
            path to .csv or reVX.wind_dirs.wind_dirs.WindDirs output with
            the neighboring supply curve point gids and power-rose value at
            each cardinal direction, by default None
        n_dirs : int, optional
            Number of prominent directions to use, by default 2
        downwind : bool, optional
            Flag to remove downwind neighbors as well as upwind neighbors
        offshore_compete : bool, default
            Flag as to whether offshore farms should be included during
            CompetitiveWindFarms, by default False

        Returns
        -------
        supply_curve : pandas.DataFrame
            Updated sc_points table with transmission connections, LCOT
            and LCOE+LCOT based on simple supply curve connections
        """
        logger.info("Starting simple supply curve sort (no capacity limits).")
        self.compute_total_lcoe(
            fcr,
            transmission_costs=transmission_costs,
            avail_cap_frac=avail_cap_frac,
            connectable=False,
            max_workers=max_workers,
            consider_friction=consider_friction,
        )
        sort_on = self._determine_sort_on(sort_on)

        if isinstance(columns, tuple):
            columns = list(columns)
        columns = self._adjust_output_columns(columns, consider_friction)

        trans_table = self._trans_table.copy()
        connections = trans_table.sort_values(
            [sort_on, SupplyCurveField.TRANS_GID])
        connections = connections.groupby(SupplyCurveField.SC_GID).first()
        connections = connections[columns].reset_index()

        supply_curve = self._sc_points.merge(connections,
                                             on=SupplyCurveField.SC_GID)
        if wind_dirs is not None:
            supply_curve = CompetitiveWindFarms.run(
                wind_dirs,
                supply_curve,
                n_dirs=n_dirs,
                offshore=offshore_compete,
                sort_on=sort_on,
                downwind=downwind,
            )

        supply_curve = supply_curve.reset_index(drop=True)

        return supply_curve

    def run(
        self,
        out_fpath,
        fixed_charge_rate,
        simple=True,
        avail_cap_frac=1,
        line_limited=False,
        transmission_costs=None,
        consider_friction=True,
        sort_on=None,
        columns=DEFAULT_COLUMNS,
        max_workers=None,
        competition=None,
    ):
        """Run Supply Curve Transmission calculations.

        Run full supply curve taking into account available capacity of
        tranmission features when making connections.

        Parameters
        ----------
        out_fpath : str
            Full path to output CSV file. Does not need to include file
            ending - it will be added automatically if missing.
        fixed_charge_rate : float
            Fixed charge rate, (in decimal form: 5% = 0.05). This value
            is used to compute LCOT.
        simple : bool, optional
            Option to run the simple sort (does not keep track of
            capacity available on the existing transmission grid). If
            ``False``, a full transmission sort (where connections are
            limited based on available transmission capacity) is run.
            Note that the full transmission sort requires the
            `avail_cap_frac` and `line_limited` inputs.
            By default, ``True``.
        avail_cap_frac : int, optional
            This input has no effect if ``simple=True``. Fraction of
            transmissions features capacity ``ac_cap`` to make available
            for connection to supply curve points. By default, ``1``.
        line_limited : bool, optional
            This input has no effect if ``simple=True``. Flag to have
            substation connection limited by maximum capacity
            of the attached lines. This is a legacy method.
            By default, ``False``.
        transmission_costs : str | dict, optional
            Dictionary of transmission feature costs or path to JSON
            file containing a dictionary of transmission feature costs.
            These costs are used to compute transmission capital cost
            if the input transmission tables do not have a
            ``"trans_cap_cost"`` column (this input is ignored
            otherwise). The dictionary must include:

                - line_tie_in_cost
                - line_cost
                - station_tie_in_cost
                - center_tie_in_cost
                - sink_tie_in_cost

            By default, ``None``.
        consider_friction : bool, optional
            Flag to add a new ``"total_lcoe_friction"`` column to the
            supply curve output that contains the sum of the computed
            ``"total_lcoe"`` value and the input
            ``"mean_lcoe_friction"`` values. If ``"mean_lcoe_friction"``
            is not in the `sc_points` input, this option is ignored.
            By default, ``True``.
        sort_on : str, optional
            Column label to sort the supply curve table on. This affects
            the build priority when doing a "full" sort - connections
            with the lowest value in this column will be built first.
            For a "simple" sort, only connections with the lowest value
            in this column will be considered. If ``None``, the sort is
            performed on the total LCOE *without* any reinforcement
            costs added (this is typically what you want - it avoids
            unrealistically long spur-line connections).
            By default ``None``.
        columns : list | tuple, optional
            Columns to preserve in output supply curve dataframe.
            By default, :obj:`DEFAULT_COLUMNS`.
        max_workers : int, optional
            Number of workers to use to compute LCOT. If > 1,
            computation is run in parallel. If ``None``, computation
            uses all available CPU's. By default, ``None``.
        competition : dict, optional
            Optional dictionary of arguments for competitive wind farm
            exclusions, which removes supply curve points upwind (and
            optionally downwind) of the lowest LCOE supply curves.
            If ``None``, no competition is applied. Otherwise, this
            dictionary can have up to four keys:

                - ``wind_dirs`` (required) : A path to a CSV file or
                  :py:class:`reVX ProminentWindDirections
                  <reVX.wind_dirs.prominent_wind_dirs.ProminentWindDirections>`
                  output with the neighboring supply curve point gids
                  and power-rose values at each cardinal direction.
                - ``n_dirs`` (optional) : An integer representing the
                  number of prominent directions to use during wind farm
                  competition. By default, ``2``.
                - ``downwind`` (optional) : A flag indicating that
                  downwind neighbors should be removed in addition to
                  upwind neighbors during wind farm competition.
                  By default, ``False``.
                - ``offshore_compete`` (optional) : A flag indicating
                  that offshore farms should be included during wind
                  farm competition. By default, ``False``.

            By default ``None``.

        Returns
        -------
        str
            Path to output supply curve.
        """
        kwargs = {
            "fcr": fixed_charge_rate,
            "transmission_costs": transmission_costs,
            "consider_friction": consider_friction,
            "sort_on": sort_on,
            "columns": columns,
            "max_workers": max_workers,
        }
        kwargs.update(competition or {})

        if simple:
            supply_curve = self.simple_sort(**kwargs)
        else:
            kwargs["avail_cap_frac"] = avail_cap_frac
            kwargs["line_limited"] = line_limited
            supply_curve = self.full_sort(**kwargs)

        out_fpath = _format_sc_out_fpath(out_fpath)
        supply_curve.to_csv(out_fpath, index=False)

        return out_fpath


def _format_sc_out_fpath(out_fpath):
    """Add CSV file ending and replace underscore, if necessary."""
    if not out_fpath.endswith(".csv"):
        out_fpath = "{}.csv".format(out_fpath)

    project_dir, out_fn = os.path.split(out_fpath)
    out_fn = out_fn.replace("supply_curve", "supply-curve")
    return os.path.join(project_dir, out_fn)


def _column_sort_key(col):
    """Determine the sort order of the input column. """
    col_value = _REQUIRED_COMPUTE_AND_OUTPUT_COLS.get(col)
    if col_value is None:
        col_value = _REQUIRED_OUTPUT_COLS.get(col)
    if col_value is None:
        col_value = 1e6

    return col_value, str(col)
