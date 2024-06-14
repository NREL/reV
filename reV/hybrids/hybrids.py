# -*- coding: utf-8 -*-
"""reV Hybridization module.

@author: ppinchuk
"""

import logging
import re
from collections import namedtuple
from string import ascii_letters
from warnings import warn

import numpy as np
import pandas as pd
from rex.resource import Resource
from rex.utilities.utilities import to_records_array

from reV.handlers.outputs import Outputs
from reV.hybrids.hybrid_methods import HYBRID_METHODS
from reV.utilities import SupplyCurveField
from reV.utilities.exceptions import (
    FileInputError,
    InputError,
    InputWarning,
    OutputWarning,
)

logger = logging.getLogger(__name__)

MERGE_COLUMN = SupplyCurveField.SC_POINT_GID
PROFILE_DSET_REGEX = 'rep_profiles_[0-9]+$'
SOLAR_PREFIX = 'solar_'
WIND_PREFIX = 'wind_'
NON_DUPLICATE_COLS = {
    SupplyCurveField.LATITUDE, SupplyCurveField.LONGITUDE,
    SupplyCurveField.COUNTRY, SupplyCurveField.STATE, SupplyCurveField.COUNTY,
    SupplyCurveField.ELEVATION, SupplyCurveField.TIMEZONE,
    SupplyCurveField.SC_POINT_GID, SupplyCurveField.SC_ROW_IND,
    SupplyCurveField.SC_COL_IND
}
HYBRIDS_GID_COL = "gid"
DEFAULT_FILL_VALUES = {f'solar_{SupplyCurveField.CAPACITY_AC_MW}': 0,
                       f'wind_{SupplyCurveField.CAPACITY_AC_MW}': 0,
                       f'solar_{SupplyCurveField.MEAN_CF_AC}': 0,
                       f'wind_{SupplyCurveField.MEAN_CF_AC}': 0}
OUTPUT_PROFILE_NAMES = ['hybrid_profile',
                        'hybrid_solar_profile',
                        'hybrid_wind_profile']
RatioColumns = namedtuple('RatioColumns', ['num', 'denom', 'fixed'],
                          defaults=(None, None, None))


class ColNameFormatter:
    """Column name formatting helper class."""

    ALLOWED = set(ascii_letters)

    @classmethod
    def fmt(cls, n):
        """Format an input column name to remove excess chars and whitespace.

        This method should help facilitate the merging of column names
        between two DataFrames.

        Parameters
        ----------
        n : str
            Input column name.

        Returns
        -------
        str
            The column name with all characters except ascii stripped
            and all lowercase.
        """
        return "".join(c for c in n if c in cls.ALLOWED).lower()


class HybridsData:
    """Hybrids input data container."""

    def __init__(self, solar_fpath, wind_fpath):
        """
        Parameters
        ----------
        solar_fpath : str
            Filepath to rep profile output file to extract solar profiles and
            summaries from.
        wind_fpath : str
            Filepath to rep profile output file to extract wind profiles and
            summaries from.
        """
        self.solar_fpath = solar_fpath
        self.wind_fpath = wind_fpath
        self.profile_dset_names = []
        self.merge_col_overlap_values = set()
        self._solar_meta = None
        self._wind_meta = None
        self._solar_time_index = None
        self._wind_time_index = None
        self._hybrid_time_index = None
        self.__profile_reg_check = re.compile(PROFILE_DSET_REGEX)
        self.__solar_cols = self.solar_meta.columns.map(ColNameFormatter.fmt)
        self.__wind_cols = self.wind_meta.columns.map(ColNameFormatter.fmt)

    @property
    def solar_meta(self):
        """Summary for the solar representative profiles.

        Returns
        -------
        solar_meta : pd.DataFrame
            Summary for the solar representative profiles.
        """
        if self._solar_meta is None:
            with Resource(self.solar_fpath) as res:
                self._solar_meta = res.meta
        return self._solar_meta

    @property
    def wind_meta(self):
        """Summary for the wind representative profiles.

        Returns
        -------
        wind_meta : pd.DataFrame
            Summary for the wind representative profiles.
        """
        if self._wind_meta is None:
            with Resource(self.wind_fpath) as res:
                self._wind_meta = res.meta
        return self._wind_meta

    @property
    def solar_time_index(self):
        """Get the time index for the solar rep profiles.

        Returns
        -------
        solar_time_index : pd.datetimeindex
            Time index sourced from the solar reV gen file.
        """
        if self._solar_time_index is None:
            with Resource(self.solar_fpath) as res:
                self._solar_time_index = res.time_index
        return self._solar_time_index

    @property
    def wind_time_index(self):
        """Get the time index for the wind rep profiles.

        Returns
        -------
        wind_time_index : pd.datetimeindex
            Time index sourced from the wind reV gen file.
        """
        if self._wind_time_index is None:
            with Resource(self.wind_fpath) as res:
                self._wind_time_index = res.time_index
        return self._wind_time_index

    @property
    def hybrid_time_index(self):
        """Get the time index for the hybrid rep profiles.

        Returns
        -------
        hybrid_time_index : pd.datetimeindex
            Time index for the hybrid rep profiles.
        """
        if self._hybrid_time_index is None:
            self._hybrid_time_index = self.solar_time_index.join(
                self.wind_time_index, how="inner"
            )
        return self._hybrid_time_index

    def contains_col(self, col_name):
        """Check if input column name exists in either meta data set.

        Parameters
        ----------
        col_name : str
            Name of column to check for.

        Returns
        -------
        bool
            Whether or not the column is found in either meta data set.
        """
        fmt_name = ColNameFormatter.fmt(col_name)
        col_in_solar = fmt_name in self.__solar_cols
        col_in_wind = fmt_name in self.__wind_cols
        return col_in_solar or col_in_wind

    def validate(self):
        """Validate the input data.

        This method checks for a minimum time index length, a unique
        profile, and unique merge column that overlaps between both data
        sets.

        """
        self._validate_time_index()
        self._validate_num_profiles()
        self._validate_merge_col_exists()
        self._validate_unique_merge_col()
        self._validate_merge_col_overlaps()

    def _validate_time_index(self):
        """Validate the hybrid time index to be of len >= 8760.

        Raises
        ------
        FileInputError
            If len(time_index) < 8760 for the hybrid profile.
        """
        if len(self.hybrid_time_index) < 8760:
            msg = (
                "The length of the merged time index ({}) is less than "
                "8760. Please ensure that the input profiles have a "
                "time index that overlaps >= 8760 times."
            )
            e = msg.format(len(self.hybrid_time_index))
            logger.error(e)
            raise FileInputError(e)

    def _validate_num_profiles(self):
        """Validate the number of input profiles.

        Raises
        ------
        FileInputError
            If # of rep_profiles > 1.
        """
        for fp in [self.solar_fpath, self.wind_fpath]:
            with Resource(fp) as res:
                profile_dset_names = [
                    n for n in res.dsets if self.__profile_reg_check.match(n)
                ]
                if not profile_dset_names:
                    msg = (
                        "Did not find any data sets matching the regex: "
                        "{!r} in {!r}. Please ensure that the profile data "
                        "exists and that the data set is named correctly."
                    )
                    e = msg.format(PROFILE_DSET_REGEX, fp)
                    logger.error(e)
                    raise FileInputError(e)
                if len(profile_dset_names) > 1:
                    msg = ("Found more than one profile in {!r}: {}. "
                           "This module is not intended for hybridization of "
                           "multiple representative profiles. Please re-run "
                           "on a single aggregated profile.")
                    e = msg.format(fp, profile_dset_names)
                    logger.error(e)
                    raise FileInputError(e)
                self.profile_dset_names += profile_dset_names

    def _validate_merge_col_exists(self):
        """Validate the existence of the merge column.

        Raises
        ------
        FileInputError
            If merge column is missing from either the solar or
            the wind meta data.
        """
        msg = (
            "Cannot hybridize: merge column {!r} missing from the "
            "{} meta data! ({!r})"
        )

        mc = ColNameFormatter.fmt(MERGE_COLUMN)
        for cols, fp, res in zip(
            [self.__solar_cols, self.__wind_cols],
            [self.solar_fpath, self.wind_fpath],
            ["solar", "wind"],
        ):
            if mc not in cols:
                e = msg.format(MERGE_COLUMN, res, fp)
                logger.error(e)
                raise FileInputError(e)

    def _validate_unique_merge_col(self):
        """Validate the existence of unique values in the merge column.

        Raises
        ------
        FileInputError
            If merge column contains duplicate values  in either the solar or
            the wind meta data.
        """
        msg = (
            "Duplicate {}s were found. This is likely due to resource "
            "class binning, which is not supported at this time. "
            "Please re-run supply curve aggregation without "
            "resource class binning and ensure there are no duplicate "
            "values in {!r}. File: {!r}"
        )

        mc = ColNameFormatter.fmt(MERGE_COLUMN)
        for ds, cols, fp in zip(
            [self.solar_meta, self.wind_meta],
            [self.__solar_cols, self.__wind_cols],
            [self.solar_fpath, self.wind_fpath],
        ):
            merge_col = ds.columns[cols == mc].item()
            if not ds[merge_col].is_unique:
                e = msg.format(merge_col, merge_col, fp)
                logger.error(e)
                raise FileInputError(e)

    def _validate_merge_col_overlaps(self):
        """Validate the existence of overlap in the merge column values.

        Raises
        ------
        FileInputError
            If merge column values do not overlap between the tow input files.
        """
        mc = ColNameFormatter.fmt(MERGE_COLUMN)
        merge_col = self.solar_meta.columns[self.__solar_cols == mc].item()
        solar_vals = set(self.solar_meta[merge_col].values)
        merge_col = self.wind_meta.columns[self.__wind_cols == mc].item()
        wind_vals = set(self.wind_meta[merge_col].values)
        self.merge_col_overlap_values = solar_vals & wind_vals

        if not self.merge_col_overlap_values:
            msg = (
                "No overlap detected in the values of {!r} across the "
                "input files. Please ensure that at least one of the "
                "{!r} values is the same for input files {!r} and {!r}"
            )
            e = msg.format(
                merge_col, merge_col, self.solar_fpath, self.wind_fpath
            )
            logger.error(e)
            raise FileInputError(e)


class MetaHybridizer:
    """Framework to handle hybridization of meta data."""

    _INTERNAL_COL_PREFIX = "_h_internal"

    def __init__(
        self,
        data,
        allow_solar_only=False,
        allow_wind_only=False,
        fillna=None,
        limits=None,
        ratio_bounds=None,
        ratio="solar_capacity/wind_capacity",
    ):
        """
        Parameters
        ----------
        data : `HybridsData`
            Instance of `HybridsData` containing input data to
            hybridize.
        allow_solar_only : bool, optional
            Option to allow SC points with only solar capacity
            (no wind). By default, ``False``.
        allow_wind_only : bool, optional
            Option to allow SC points with only wind capacity
            (no solar), By default, ``False``.
        fillna : dict, optional
            Dictionary containing column_name, fill_value pairs
            representing any fill values that should be applied after
            merging the wind and solar meta. Note that column names will
            likely have to be prefixed with ``solar`` or ``wind``.
            By default, ``None``.
        limits : dict, optional
            Option to specify mapping (in the form of a dictionary) of
            {colum_name: max_value} representing the upper limit
            (maximum value) for the values of a column in the merged
            meta. For example, `limits={'solar_capacity': 100}` would
            limit all the values of the solar capacity in the merged
            meta to a maximum value of 100. This limit is applied
            *BEFORE* ratio calculations. The names of the columns should
            match the column names in the merged meta, so they are
            likely prefixed with ``solar`` or ``wind`. By default,
            ``None`` (no limits applied).
        ratio_bounds : tuple, optional
            Option to set ratio bounds (in two-tuple form) on the
            columns of the `ratio` input. For example,
            `ratio_bounds=(0.5, 1.5)` would adjust the values of both of
            the `ratio` columns such that their ratio is always between
            half and double (e.g., no value would be more than double
            the other). To specify a single ratio value, use the same
            value as the upper and lower bound. For example,
            `ratio_bounds=(1, 1)` would adjust the values of both of the
            `ratio` columns such that their ratio is always equal.
            By default, ``None`` (no limit on the ratio).
        ratio : str, optional
            Option to specify the columns used to calculate the ratio
            that is limited by the `ratio_bounds` input. This input is a
            string in the form
            "numerator_column_name/denominator_column_name".
            For example, `ratio='solar_capacity/wind_capacity'` would
            limit the ratio of the solar to wind capacities as specified
            by the `ratio_bounds` input. If `ratio_bounds` is ``None``,
            this input does nothing. The names of the columns should be
            prefixed with one of the prefixes defined as class
            variables. By default ``'solar_capacity/wind_capacity'``.
        """
        self.data = data
        self._allow_solar_only = allow_solar_only
        self._allow_wind_only = allow_wind_only
        self._fillna = {**DEFAULT_FILL_VALUES, **(fillna or {})}
        self._limits = limits or {}
        self._ratio_bounds = ratio_bounds
        self._ratio = ratio
        self._hybrid_meta = None
        self.__hybrid_meta_cols = None
        self.__col_name_map = None
        self.__solar_rpi_n = "{}_solar_rpidx".format(self._INTERNAL_COL_PREFIX)
        self.__wind_rpi_n = "{}_wind_rpidx".format(self._INTERNAL_COL_PREFIX)

    @property
    def hybrid_meta(self):
        """Hybridized summary for the representative profiles.

        Returns
        -------
        hybrid_meta : pd.DataFrame
            Summary for the hybridized representative profiles.
            At the very least, this has a column that the data was merged on.
        """
        if self._hybrid_meta is None or self.__hybrid_meta_cols is None:
            return self._hybrid_meta
        return self._hybrid_meta[self.__hybrid_meta_cols]

    def validate_input(self):
        """Validate the input parameters.

        This method validates that the input limit, fill, and ratio columns
        are formatted correctly.
        """
        self._validate_limits_cols_prefixed()
        self._validate_fillna_cols_prefixed()
        self._validate_ratio_input()

    def _validate_limits_cols_prefixed(self):
        """Ensure the limits columns are formatted correctly.

        This check is important because the limiting happens
        after the meta has been merged (so columns are already prefixed),
        but before the hybrid columns are computed. As a result, the limits
        columns _must_ have a valid prefix.

        Raises
        ------
        InputError
            If limits columns are not prefixed correctly.
        """
        for col in self._limits:
            self.__validate_col_prefix(
                col, (SOLAR_PREFIX, WIND_PREFIX), input_name="limits"
            )

    @staticmethod
    def __validate_col_prefix(col, prefixes, input_name):
        """Validate the the col starts with the correct prefix."""

        missing = [not col.startswith(p) for p in prefixes]
        if all(missing):
            msg = (
                "Input {0} column {1!r} does not start with a valid "
                "prefix: {2!r}. Please ensure that the {0} column "
                "names specify the correct resource prefix."
            )
            e = msg.format(input_name, col, prefixes)
            logger.error(e)
            raise InputError(e)

    def _validate_fillna_cols_prefixed(self):
        """Ensure the fillna columns are formatted correctly.

        This check is important because the fillna step happens
        after the meta has been merged (so columns are already prefixed),
        but before the hybrid columns are computed. As a result, the fillna
        columns _must_ have a valid prefix.

        Raises
        ------
        InputError
            If fillna columns are not prefixed correctly.
        """
        for col in self._fillna:
            self.__validate_col_prefix(
                col, (SOLAR_PREFIX, WIND_PREFIX), input_name="fillna"
            )

    def _validate_ratio_input(self):
        """Validate the ratio input parameters.

        This method validates that the input ratio columns are formatted
        correctly and exist in the input data. It also verifies that
        the `ratio_bounds` is correctly formatted.
        """
        if self._ratio_bounds is None:
            return

        self._validate_ratio_bounds()
        self._validate_ratio_type()
        self._validate_ratio_format()
        self._validate_ratio_cols_prefixed()
        self._validate_ratio_cols_exist()

    def _validate_ratio_bounds(self):
        """Ensure the ratio value is input correctly.

        Raises
        ------
        InputError
            If ratio is not a len 2 container of floats.
        """

        try:
            if len(self._ratio_bounds) != 2:
                msg = (
                    "Length of input for ratio_bounds is {} - but is "
                    "required to be of length 2. Please make sure this "
                    "input is a len 2 container of floats. If you would "
                    "like to specify a single ratio value, use the same "
                    "float for both limits (i.e. ratio_bounds=(1, 1))."
                )
                e = msg.format(len(self._ratio_bounds))
                logger.error(e)
                raise InputError(e)
        except TypeError:
            msg = (
                "Input for ratio_bounds not understood: {!r}. "
                "Please make sure this value is a len 2 container "
                "of floats."
            )
            e = msg.format(self._ratio_bounds)
            logger.error(e)
            raise InputError(e) from None

    def _validate_ratio_type(self):
        """Ensure that the ratio input is a string.

        Raises
        ------
        InputError
            If `ratio` is not a string.
        """
        if not isinstance(self._ratio, str):
            msg = (
                "Ratio input type {} not understood. Please make sure "
                "the ratio input is a string in the form "
                "'numerator_column_name/denominator_column_name'. Ratio "
                "input: {!r}"
            )
            e = msg.format(type(self._ratio), self._ratio)
            logger.error(e)
            raise InputError(e)

    def _validate_ratio_format(self):
        """Validate that the ratio input format is correct and can be parsed.

        Raises
        ------
        InputError
            If the '/' character is missing or of there are too many
            '/' characters.
        """
        if "/" not in self._ratio:
            msg = (
                "Ratio input {} does not contain the '/' character. "
                "Please make sure the ratio input is a string in the form "
                "'numerator_column_name/denominator_column_name'"
            )
            e = msg.format(self._ratio)
            logger.error(e)
            raise InputError(e)

        if len(self._ratio_cols) != 2:
            msg = (
                "Ratio input {} contains too many '/' characters. Please "
                "make sure the ratio input is a string in the form "
                "'numerator_column_name/denominator_column_name'."
            )
            e = msg.format(self._ratio)
            logger.error(e)
            raise InputError(e)

    def _validate_ratio_cols_prefixed(self):
        """Ensure the ratio columns are formatted correctly.

        This check is important because the ratio limit step happens
        after the meta has been merged (so columns are already prefixed),
        but before the hybrid columns are computed. As a result, the ratio
        columns _must_ have a valid prefix.

        Raises
        ------
        InputError
            If ratio columns are not prefixed correctly.
        """

        for col in self._ratio_cols:
            self.__validate_col_prefix(
                col, (SOLAR_PREFIX, WIND_PREFIX), input_name="ratios"
            )

    def _validate_ratio_cols_exist(self):
        """Ensure the ratio columns exist if a ratio is specified.

        Raises
        ------
        FileInputError
            If ratio columns are not found in the meta data.
        """

        for col in self._ratio_cols:
            no_prefix_name = "_".join(col.split("_")[1:])
            if not self.data.contains_col(no_prefix_name):
                msg = (
                    "Input ratios column {!r} not found in either meta "
                    "data! Please check the input files {!r} and {!r}"
                )
                e = msg.format(
                    no_prefix_name, self.data.solar_fpath, self.data.wind_fpath
                )
                logger.error(e)
                raise FileInputError(e)

    @property
    def _ratio_cols(self):
        """Get the ratio columns from the ratio input."""
        if self._ratio is None:
            return []
        return self._ratio.strip().split("/")

    def hybridize(self):
        """Combine the solar and wind metas and run hybridize methods."""
        self._format_meta_pre_merge()
        self._merge_solar_wind_meta()
        self._verify_lat_lon_match_post_merge()
        self._format_meta_post_merge()
        self._fillna_meta_cols()
        self._apply_limits()
        self._limit_by_ratio()
        self._add_hybrid_cols()
        self._sort_hybrid_meta_cols()

    def _format_meta_pre_merge(self):
        """Prepare solar and wind meta for merging."""
        self.__col_name_map = {
            ColNameFormatter.fmt(c): c
            for c in self.data.solar_meta.columns.values
        }

        self._rename_cols(self.data.solar_meta, prefix=SOLAR_PREFIX)
        self._rename_cols(self.data.wind_meta, prefix=WIND_PREFIX)

        self._save_rep_prof_index_internally()

    @staticmethod
    def _rename_cols(df, prefix):
        """Replace column names with the ColNameFormatter.fmt is needed."""
        df.columns = [
            ColNameFormatter.fmt(col_name)
            if col_name in NON_DUPLICATE_COLS
            else "{}{}".format(prefix, col_name)
            for col_name in df.columns.values
        ]

    def _save_rep_prof_index_internally(self):
        """Save rep profiles index in hybrid meta for access later."""

        self.data.solar_meta[self.__solar_rpi_n] = self.data.solar_meta.index
        self.data.wind_meta[self.__wind_rpi_n] = self.data.wind_meta.index

    def _merge_solar_wind_meta(self):
        """Merge the wind and solar meta DataFrames."""
        self._hybrid_meta = self.data.solar_meta.merge(
            self.data.wind_meta,
            on=ColNameFormatter.fmt(MERGE_COLUMN),
            suffixes=[None, "_x"],
            how=self._merge_type(),
        )

    def _merge_type(self):
        """Determine the type of merge to use for meta based on user input."""
        if self._allow_solar_only and self._allow_wind_only:
            return 'outer'
        if self._allow_solar_only and not self._allow_wind_only:
            return 'left'
        if not self._allow_solar_only and self._allow_wind_only:
            return 'right'
        return 'inner'

    def _format_meta_post_merge(self):
        """Format hybrid meta after merging."""

        duplicate_cols = [n for n in self._hybrid_meta.columns if "_x" in n]
        self._propagate_duplicate_cols(duplicate_cols)
        self._drop_cols(duplicate_cols)
        self._hybrid_meta.rename(self.__col_name_map, inplace=True, axis=1)
        self._hybrid_meta.index.name = HYBRIDS_GID_COL

    def _propagate_duplicate_cols(self, duplicate_cols):
        """Fill missing column values from outer merge."""
        for duplicate in duplicate_cols:
            no_suffix = "_".join(duplicate.split("_")[:-1])
            null_idx = self._hybrid_meta[no_suffix].isnull()
            non_null_vals = self._hybrid_meta.loc[null_idx, duplicate].values
            self._hybrid_meta.loc[null_idx, no_suffix] = non_null_vals

    def _drop_cols(self, duplicate_cols):
        """Drop any remaining duplicate and 'HYBRIDS_GID_COL' columns."""
        self._hybrid_meta.drop(
            duplicate_cols + [HYBRIDS_GID_COL],
            axis=1,
            inplace=True,
            errors="ignore",
        )

    def _sort_hybrid_meta_cols(self):
        """Sort the columns of the hybrid meta."""
        self.__hybrid_meta_cols = sorted(
            [
                c
                for c in self._hybrid_meta.columns
                if not c.startswith(self._INTERNAL_COL_PREFIX)
            ],
            key=self._column_sorting_key,
        )

    def _column_sorting_key(self, c):
        """Helper function to sort hybrid meta columns."""
        first_index = 0
        if c.startswith("hybrid"):
            first_index = 1
        elif c.startswith("solar"):
            first_index = 2
        elif c.startswith("wind"):
            first_index = 3
        elif c == MERGE_COLUMN:
            first_index = -1
        return first_index, self._hybrid_meta.columns.get_loc(c)

    def _verify_lat_lon_match_post_merge(self):
        """Verify that all the lat/lon values match post merge."""
        lat = self._verify_col_match_post_merge(
            col_name=ColNameFormatter.fmt(SupplyCurveField.LATITUDE)
        )
        lon = self._verify_col_match_post_merge(
            col_name=ColNameFormatter.fmt(SupplyCurveField.LONGITUDE)
        )
        if not lat or not lon:
            msg = (
                "Detected mismatched coordinate values (latitude or "
                "longitude) post merge. Please ensure that all matching "
                "values of {!r} correspond to the same values of latitude "
                "and longitude across the input files {!r} and {!r}"
            )
            e = msg.format(
                MERGE_COLUMN, self.data.solar_fpath, self.data.wind_fpath
            )
            logger.error(e)
            raise FileInputError(e)

    def _verify_col_match_post_merge(self, col_name):
        """Verify that all (non-null) values in a column match post merge."""
        c1, c2 = col_name, '{}_x'.format(col_name)
        if c1 in self._hybrid_meta.columns and c2 in self._hybrid_meta.columns:
            compare_df = self._hybrid_meta[
                (self._hybrid_meta[c1].notnull())
                & (self._hybrid_meta[c2].notnull())
            ]
            return np.allclose(compare_df[c1], compare_df[c2])
        return True

    def _fillna_meta_cols(self):
        """Fill N/A values as specified by user (and internals)."""
        for col_name, fill_value in self._fillna.items():
            if col_name in self._hybrid_meta.columns:
                self._hybrid_meta[col_name].fillna(fill_value, inplace=True)
            else:
                self.__warn_missing_col(col_name, action="fill")

        self._hybrid_meta[self.__solar_rpi_n].fillna(-1, inplace=True)
        self._hybrid_meta[self.__wind_rpi_n].fillna(-1, inplace=True)

    @staticmethod
    def __warn_missing_col(col_name, action):
        """Warn that a column the user request an action for is missing."""
        msg = ("Skipping {} values for {!r}: Unable to find column "
               "in hybrid meta. Did you forget to prefix with "
               "{!r} or {!r}? ")
        w = msg.format(action, col_name, SOLAR_PREFIX, WIND_PREFIX)
        logger.warning(w)
        warn(w, InputWarning)

    def _apply_limits(self):
        """Clip column values as specified by user."""
        for col_name, max_value in self._limits.items():
            if col_name in self._hybrid_meta.columns:
                self._hybrid_meta[col_name].clip(upper=max_value, inplace=True)
            else:
                self.__warn_missing_col(col_name, action="limit")

    def _limit_by_ratio(self):
        """Limit the given pair of ratio columns based on input ratio."""

        if self._ratio_bounds is None:
            return

        numerator_col, denominator_col = self._ratio_cols
        min_ratio, max_ratio = sorted(self._ratio_bounds)

        overlap_idx = self._hybrid_meta[MERGE_COLUMN].isin(
            self.data.merge_col_overlap_values
        )

        numerator_vals = self._hybrid_meta[numerator_col].copy()
        denominator_vals = self._hybrid_meta[denominator_col].copy()

        ratios = (
            numerator_vals.loc[overlap_idx] / denominator_vals.loc[overlap_idx]
        )
        ratio_too_low = (ratios < min_ratio) & overlap_idx
        ratio_too_high = (ratios > max_ratio) & overlap_idx

        numerator_vals.loc[ratio_too_high] = (
            denominator_vals.loc[ratio_too_high].values * max_ratio
        )
        denominator_vals.loc[ratio_too_low] = (
            numerator_vals.loc[ratio_too_low].values / min_ratio
        )

        h_num_name = "hybrid_{}".format(numerator_col)
        h_denom_name = "hybrid_{}".format(denominator_col)
        self._hybrid_meta[h_num_name] = numerator_vals.values
        self._hybrid_meta[h_denom_name] = denominator_vals.values

    def _add_hybrid_cols(self):
        """Add new hybrid columns using registered hybrid methods."""
        for new_col_name, method in HYBRID_METHODS.items():
            out = method(self)
            if out is not None:
                try:
                    self._hybrid_meta[new_col_name] = out
                except ValueError as e:
                    msg = (
                        "Unable to add {!r} column to hybrid meta. The "
                        "following exception was raised when adding "
                        "the data output by '{}': {!r}."
                    )
                    w = msg.format(new_col_name, method.__name__, e)
                    logger.warning(w)
                    warn(w, OutputWarning)

    @property
    def solar_profile_indices_map(self):
        """Map hybrid to solar rep indices.

        Returns
        -------
        hybrid_indices : np.ndarray
            Index values corresponding to hybrid rep profiles.
        solar_indices : np.ndarray
            Index values of the solar rep profiles corresponding
            to the hybrid rep profile indices.
        """

        if self._hybrid_meta is None:
            return np.array([]), np.array([])

        idxs = self._hybrid_meta[self.__solar_rpi_n].astype(int)
        idxs = idxs[idxs >= 0]

        return idxs.index.values, idxs.values

    @property
    def wind_profile_indices_map(self):
        """Map hybrid to wind rep indices.

        Returns
        -------
        hybrid_indices : np.ndarray
            Index values corresponding to hybrid rep profiles.
        wind_indices : np.ndarray
            Index values of the wind rep profiles corresponding
            to the hybrid rep profile indices.
        """
        if self._hybrid_meta is None:
            return np.array([]), np.array([])

        idxs = self._hybrid_meta[self.__wind_rpi_n].astype(int)
        idxs = idxs[idxs >= 0]

        return idxs.index.values, idxs.values


class Hybridization:
    """Hybridization"""

    def __init__(
        self,
        solar_fpath,
        wind_fpath,
        allow_solar_only=False,
        allow_wind_only=False,
        fillna=None,
        limits=None,
        ratio_bounds=None,
        ratio="solar_capacity/wind_capacity",
    ):
        """Framework to handle hybridization of SC and corresponding profiles.

        ``reV`` hybrids computes a "hybrid" wind and solar supply curve,
        where each supply curve point contains some wind and some solar
        capacity. Various ratio limits on wind-to-solar farm properties
        (e.g. wind-to-solar capacity) can be applied during the
        hybridization process. Hybrid generation profiles are also
        computed during this process.

        Parameters
        ----------
        solar_fpath : str
            Filepath to rep profile output file to extract solar
            profiles and summaries from.
        wind_fpath : str
            Filepath to rep profile output file to extract wind profiles
            and summaries from.
        allow_solar_only : bool, optional
            Option to allow SC points with only solar capacity
            (no wind). By default, ``False``.
        allow_wind_only : bool, optional
            Option to allow SC points with only wind capacity
            (no solar). By default, ``False``.
        fillna : dict, optional
            Dictionary containing column_name, fill_value pairs
            representing any fill values that should be applied after
            merging the wind and solar meta. Note that column names will
            likely have to be prefixed with ``solar`` or ``wind``.
            By default ``None``.
        limits : dict, optional
            Option to specify mapping (in the form of a dictionary) of
            {colum_name: max_value} representing the upper limit
            (maximum value) for the values of a column in the merged
            meta. For example, ``limits={'solar_capacity': 100}`` would
            limit all the values of the solar capacity in the merged
            meta to a maximum value of 100. This limit is applied
            *BEFORE* ratio calculations. The names of the columns should
            match the column names in the merged meta, so they are
            likely prefixed with ``solar`` or ``wind``.
            By default, ``None`` (no limits applied).
        ratio_bounds : tuple, optional
            Option to set ratio bounds (in two-tuple form) on the
            columns of the ``ratio`` input. For example,
            ``ratio_bounds=(0.5, 1.5)`` would adjust the values of both
            of the ``ratio`` columns such that their ratio is always
            between half and double (e.g., no value would be more than
            double the other). To specify a single ratio value, use the
            same value as the upper and lower bound. For example,
            ``ratio_bounds=(1, 1)`` would adjust the values of both of
            the ``ratio`` columns such that their ratio is always equal.
            By default, ``None`` (no limit on the ratio).
        ratio : str, optional
            Option to specify the columns used to calculate the ratio
            that is limited by the `ratio_bounds` input. This input is a
            string in the form "{numerator_column}/{denominator_column}".
            For example, ``ratio='solar_capacity/wind_capacity'``
            would limit the ratio of the solar to wind capacities as
            specified by the ``ratio_bounds`` input. If ``ratio_bounds``
            is None, this input does nothing. The names of the columns
            should be prefixed with one of the prefixes defined as class
            variables. By default ``'solar_capacity/wind_capacity'``.
        """

        logger.info(
            "Running hybridization of rep profiles with solar_fpath: "
            '"{}"'.format(solar_fpath)
        )
        logger.info(
            "Running hybridization of rep profiles with solar_fpath: "
            '"{}"'.format(wind_fpath)
        )
        logger.info(
            "Running hybridization of rep profiles with "
            'allow_solar_only: "{}"'.format(allow_solar_only)
        )
        logger.info(
            "Running hybridization of rep profiles with "
            'allow_wind_only: "{}"'.format(allow_wind_only)
        )
        logger.info(
            'Running hybridization of rep profiles with fillna: "{}"'.format(
                fillna
            )
        )
        logger.info(
            'Running hybridization of rep profiles with limits: "{}"'.format(
                limits
            )
        )
        logger.info(
            "Running hybridization of rep profiles with ratio_bounds: "
            '"{}"'.format(ratio_bounds)
        )
        logger.info(
            'Running hybridization of rep profiles with ratio: "{}"'.format(
                ratio
            )
        )

        self.data = HybridsData(solar_fpath, wind_fpath)
        self.meta_hybridizer = MetaHybridizer(
            data=self.data,
            allow_solar_only=allow_solar_only,
            allow_wind_only=allow_wind_only,
            fillna=fillna,
            limits=limits,
            ratio_bounds=ratio_bounds,
            ratio=ratio,
        )
        self._profiles = None
        self._validate_input()

    def _validate_input(self):
        """Validate the user input and input files."""
        self.data.validate()
        self.meta_hybridizer.validate_input()

    @property
    def solar_meta(self):
        """Summary for the solar representative profiles.

        Returns
        -------
        solar_meta : pd.DataFrame
            Summary for the solar representative profiles.
        """
        return self.data.solar_meta

    @property
    def wind_meta(self):
        """Summary for the wind representative profiles.

        Returns
        -------
        wind_meta : pd.DataFrame
            Summary for the wind representative profiles.
        """
        return self.data.wind_meta

    @property
    def hybrid_meta(self):
        """Hybridized summary for the representative profiles.

        Returns
        -------
        hybrid_meta : pd.DataFrame
            Summary for the hybridized representative profiles.
            At the very least, this has a column that the data was merged on.
        """
        return self.meta_hybridizer.hybrid_meta

    @property
    def solar_time_index(self):
        """Get the time index for the solar rep profiles.

        Returns
        -------
        solar_time_index : pd.Datetimeindex
            Time index sourced from the solar rep profile file.
        """
        return self.data.solar_time_index

    @property
    def wind_time_index(self):
        """Get the time index for the wind rep profiles.

        Returns
        -------
        wind_time_index : pd.Datetimeindex
            Time index sourced from the wind rep profile file.
        """
        return self.data.wind_time_index

    @property
    def hybrid_time_index(self):
        """Get the time index for the hybrid rep profiles.

        Returns
        -------
        hybrid_time_index : pd.Datetimeindex
            Time index for the hybrid rep profiles.
        """
        return self.data.hybrid_time_index

    @property
    def profiles(self):
        """Get the arrays of the hybridized representative profiles.

        Returns
        -------
        profiles : dict
            Dict of hybridized representative profiles.
        """
        return self._profiles

    def run(self, fout=None, save_hybrid_meta=True):
        """Run hybridization of profiles and save to disc.

        Parameters
        ----------
        fout : str, optional
            Filepath to output HDF5 file. If ``None``, output data are
            not written to a file. By default, ``None``.
        save_hybrid_meta : bool, optional
            Flag to save hybrid SC table to hybrid rep profile output.
            By default, ``True``.

        Returns
        -------
        str
            Filepath to output h5 file.
        """

        self.run_meta()
        self.run_profiles()

        if fout is not None:
            self.save_profiles(fout, save_hybrid_meta=save_hybrid_meta)

        logger.info("Hybridization of representative profiles complete!")
        return fout

    def run_meta(self):
        """Compute the hybridized profiles.

        Returns
        -------
        `Hybridization`
            Instance of Hybridization object (itself) containing the
            hybridized meta as an attribute.
        """
        self.meta_hybridizer.hybridize()
        return self

    def run_profiles(self):
        """Compute all hybridized profiles.

        Returns
        -------
        `Hybridization`
            Instance of Hybridization object (itself) containing the
            hybridized profiles as attributes.
        """

        logger.info("Running hybrid profile calculations.")

        self._init_profiles()
        self._compute_hybridized_profile_components()
        self._compute_hybridized_profiles_from_components()

        logger.info("Profile hybridization complete.")

        return self

    def _init_profiles(self):
        """Initialize the output rep profiles attribute."""
        self._profiles = {
            k: np.zeros(
                (len(self.hybrid_time_index), len(self.hybrid_meta)),
                dtype=np.float32,
            )
            for k in OUTPUT_PROFILE_NAMES
        }

    def _compute_hybridized_profile_components(self):
        """Compute the resource components of the hybridized profiles."""

        for params in self.__rep_profile_hybridization_params:
            col, (hybrid_idxs, solar_idxs), fpath, p_name, dset_name = params
            capacity = self.hybrid_meta.loc[hybrid_idxs, col].values

            with Resource(fpath) as res:
                data = res[
                    dset_name, res.time_index.isin(self.hybrid_time_index)
                ]
                self._profiles[p_name][:, hybrid_idxs] = (
                    data[:, solar_idxs] * capacity
                )

    @property
    def __rep_profile_hybridization_params(self):
        """Zip the rep profile hybridization parameters."""

        cap_col_names = [f"hybrid_solar_{SupplyCurveField.CAPACITY_AC_MW}",
                         f"hybrid_wind_{SupplyCurveField.CAPACITY_AC_MW}"]
        idx_maps = [
            self.meta_hybridizer.solar_profile_indices_map,
            self.meta_hybridizer.wind_profile_indices_map,
        ]
        fpaths = [self.data.solar_fpath, self.data.wind_fpath]
        zipped = zip(
            cap_col_names,
            idx_maps,
            fpaths,
            OUTPUT_PROFILE_NAMES[1:],
            self.data.profile_dset_names,
        )
        return zipped

    def _compute_hybridized_profiles_from_components(self):
        """Compute the hybridized profiles from the resource components."""

        hp_name, sp_name, wp_name = OUTPUT_PROFILE_NAMES
        self._profiles[hp_name] = (
            self._profiles[sp_name] + self._profiles[wp_name]
        )

    def _init_h5_out(self, fout, save_hybrid_meta=True):
        """Initialize an output h5 file for hybrid profiles.

        Parameters
        ----------
        fout : str
            Filepath to output h5 file.
        save_hybrid_meta : bool
            Flag to save hybrid SC table to hybrid rep profile output.
        """
        dsets = []
        shapes = {}
        attrs = {}
        chunks = {}
        dtypes = {}

        for dset, data in self.profiles.items():
            dsets.append(dset)
            shapes[dset] = data.shape
            chunks[dset] = None
            attrs[dset] = {Outputs.UNIT_ATTR: "MW"}
            dtypes[dset] = data.dtype

        meta = self.hybrid_meta.copy()
        for c in meta.columns:
            try:
                meta[c] = pd.to_numeric(meta[c])
            except ValueError:
                pass

        Outputs.init_h5(
            fout,
            dsets,
            shapes,
            attrs,
            chunks,
            dtypes,
            meta,
            time_index=self.hybrid_time_index,
        )

        if save_hybrid_meta:
            with Outputs(fout, mode="a") as out:
                hybrid_meta = to_records_array(self.hybrid_meta)
                out._create_dset(
                    "meta",
                    hybrid_meta.shape,
                    hybrid_meta.dtype,
                    data=hybrid_meta,
                )

    def _write_h5_out(self, fout, save_hybrid_meta=True):
        """Write hybrid profiles and meta to an output file.

        Parameters
        ----------
        fout : str
            Filepath to output h5 file.
        save_hybrid_meta : bool
            Flag to save hybrid SC table to hybrid rep profile output.
        """

        with Outputs(fout, mode="a") as out:
            if "meta" in out.datasets and save_hybrid_meta:
                hybrid_meta = to_records_array(self.hybrid_meta)
                out["meta"] = hybrid_meta

            for dset, data in self.profiles.items():
                out[dset] = data

    def save_profiles(self, fout, save_hybrid_meta=True):
        """Initialize fout and save profiles.

        Parameters
        ----------
        fout : str
            Filepath to output h5 file.
        save_hybrid_meta : bool
            Flag to save hybrid SC table to hybrid rep profile output.
        """

        self._init_h5_out(fout, save_hybrid_meta=save_hybrid_meta)
        self._write_h5_out(fout, save_hybrid_meta=save_hybrid_meta)
