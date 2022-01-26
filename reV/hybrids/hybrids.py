# -*- coding: utf-8 -*-
"""reV Hybridization module.

@author: ppinchuk
"""

from concurrent.futures import as_completed
import logging
import numpy as np
import re
import pandas as pd
from string import ascii_letters
from warnings import warn


from reV.handlers.outputs import Outputs
from reV.utilities.exceptions import (FileInputError, InputError,
                                      InputWarning, OutputWarning)
from reV.hybrids.hybrid_methods import HYBRID_METHODS

from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem
from rex.utilities.utilities import to_records_array

logger = logging.getLogger(__name__)

MERGE_COLUMN = 'sc_point_gid'
PROFILE_DSET_REGEX = 'rep_profiles_[0-9]+$'
SOLAR_PREFIX = 'solar_'
WIND_PREFIX = 'wind_'
NON_DUPLICATE_COLS = {
    'latitude', 'longitude', 'country', 'state', 'county', 'elevation',
    'timezone', 'sc_point_gid', 'sc_row_ind', 'sc_col_ind'
}
DROPPED_COLUMNS = ['gid']
DEFAULT_FILL_VALUES = {'solar_capacity': 0, 'wind_capacity': 0,
                       'solar_mean_cf': 0, 'wind_mean_cf': 0}
OUTPUT_PROFILE_NAMES = ['hybrid_profile', 'solar_profile', 'wind_profile']


class ColNameFormatter:
    """Column name formatting helper class. """
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
            The column name with all charactes except ascii stripped
            and all lowercase.
        """
        return ''.join(c for c in n if c in cls.ALLOWED).lower()


class HybridsData:
    """Hybrids input data container. """

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
                self.wind_time_index, how='inner')
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
            msg = ("The length of the merged time index ({}) is less than "
                   "8760. Please ensure that the input profiles have a "
                   "time index that overlaps >= 8760 times.")
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
                    n for n in res.dsets
                    if self.__profile_reg_check.match(n)
                ]
                if not profile_dset_names:
                    msg = ("Did not find any data sets matching the regex: "
                           "{!r} in {!r}. Please ensure that the profile data "
                           "exists and that the data set is named correctly.")
                    e = msg.format(PROFILE_DSET_REGEX, fp)
                    logger.error(e)
                    raise FileInputError(e)
                elif len(profile_dset_names) > 1:
                    msg = ("Found more than one profile in {!r}: {}. "
                           "This module is not intended for hybridization of "
                           "multiple representative profiles. Please re-run "
                           "on a single aggregated profile.")
                    e = msg.format(fp, profile_dset_names)
                    logger.error(e)
                    raise FileInputError(e)
                else:
                    self.profile_dset_names += profile_dset_names

    def _validate_merge_col_exists(self):
        """Validate the existence of the merge column.

        Raises
        ------
        FileInputError
            If merge column is missing from either the solar or
            the wind meta data.
        """
        msg = ("Cannot hybridize: merge column {!r} missing from the "
               "{} meta data! ({!r})")

        mc = ColNameFormatter.fmt(MERGE_COLUMN)
        for cols, fp, res in zip([self.__solar_cols, self.__wind_cols],
                                 [self.solar_fpath, self.wind_fpath],
                                 ['solar', 'wind']):
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
        msg = ("Duplicate {}s were found. This is likely due to resource "
               "class binning, which is not supported at this time. "
               "Please re-run supply curve aggregation without "
               "resource class binning and ensure there are no duplicate "
               "values in {!r}. File: {!r}")

        mc = ColNameFormatter.fmt(MERGE_COLUMN)
        for ds, cols, fp in zip([self.solar_meta, self.wind_meta],
                                [self.__solar_cols, self.__wind_cols],
                                [self.solar_fpath, self.wind_fpath]):
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
            msg = ("No overlap detected in the values of {!r} across the "
                   "input files. Please ensure that at least one of the "
                   "{!r} values is the same for input files {!r} and {!r}")
            e = msg.format(merge_col, merge_col, self.solar_fpath,
                           self.wind_fpath)
            logger.error(e)
            raise FileInputError(e)


class MetaHybridizer:
    """Framework to handle hybridization of meta data."""

    _INTERNAL_COL_PREFIX = '_h_internal'

    def __init__(self, data, allow_solar_only=False,
                 allow_wind_only=False, fillna=None, allowed_ratio=None,
                 ratio_cols=('solar_capacity', 'wind_capacity')):
        """
        Parameters
        ----------
        data : `HybridsData`
            Instance of `HybridsData` containing input data to hybridize.
        allow_solar_only : bool, optional
            Option to allow SC points with only solar capcity (no wind), by
            default False.
        allow_wind_only : bool, optional
            Option to allow SC points with only wind capcity (no solar), by
            default False.
        fillna : dict, optional
            Dictionary containing column_name, fill_value pairs reprenting any
            fill values that should be applied after merging the wind and solar
            meta. Note that column names will likely have to be prefixed
            with "solar_" or "wind_". By default None.
        allowed_ratio : float | tuple, optional
            Option to set a single ratio or ratio bounds (in two-tuple form)
            on the `ratio_cols`. A single value is treated as both an upper
            and a lower bound. This input limits the ratio of the values
            of the input `ratio_cols` (ratio is always computed with the
            first column as the numerator and the second column as the
            denomiator). For example, `allowed_ratio=1` would limit
            the values of the `ratio_cols` to always be equal. On the other
            hand, `allowed_ratio=(0.5, 1.5)` would limit the ratio to be
            between half and double (e.g., if `ratio_cols` are the capacity
            columns, no capacity value would be more than double the other).
            By default, None (no limit).
        ratio_cols : tuple, optional
            Option to specify the columns used to calculate the ratio that is
            limited by the `allowed_ratio` input. If `allowed_ratio` is None,
            this input does nothing. The names of the columns should be
            prefixed with one of the prefixes defined as class variables.
            The order of the colum names specifies the way the ratio is
            calculated: the first column is always treated as the ratio
            numerator and the second column is the ratio denominator.
            By default ('solar_capacity', 'wind_capacity').
        """
        self.data = data
        self._allow_solar_only = allow_solar_only
        self._allow_wind_only = allow_wind_only
        self._fillna = {**DEFAULT_FILL_VALUES, **(fillna or {})}
        self._allowed_ratio = allowed_ratio
        self._ratio_cols = ratio_cols
        self._hybrid_meta = None
        self.__hybrid_meta_cols = None
        self.__col_name_map = None
        self.__solar_rpi_n = '{}_solar_rpidx'.format(self._INTERNAL_COL_PREFIX)
        self.__wind_rpi_n = '{}_wind_rpidx'.format(self._INTERNAL_COL_PREFIX)

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
        else:
            return self._hybrid_meta[self.__hybrid_meta_cols]

    def validate_input(self):
        """Validate the input parameters.

        This method validates that the input ratio columns are formatted
        correctly and exist in the input data. It also verifies that
        the `allowed_ratio` is correctly formatted.

        """
        self._validate_ratio_cols_length()
        self._validate_ratio_cols_prefixed()
        self._validate_ratio_cols_exist()
        self._validate_ratio()

    def _validate_ratio_cols_length(self):
        """Ensure exactly two ratio column names are provided.

        Raises
        ------
        InputError
            If len(ratio_cols) != 2, or ratio_cols does not have a "len".
        """
        try:
            if len(self._ratio_cols) != 2:
                raise Exception
        except Exception:
            msg = ("Input for 'allowed_ratio' not understood: {!r}. "
                   "Please make sure this value is a two-tuple containing "
                   "prefixed column names.")
            e = msg.format(self._ratio_cols)
            logger.error(e)
            raise InputError(e) from None

    def _validate_ratio_cols_prefixed(self):
        """Ensure the ratio columns are formatted correctly.

        Raises
        ------
        InputError
            If ratio columns are not prefixed correctly.
        """

        if self._allowed_ratio is None:
            return

        for col in self._ratio_cols:
            missing_solar_prefix = not col.startswith(SOLAR_PREFIX)
            missing_wind_prefix = not col.startswith(WIND_PREFIX)
            if missing_solar_prefix and missing_wind_prefix:
                msg = ("Input ratio column {!r} does not start with a valid "
                       "prefix: {!r}. Please ensure that the ratio column "
                       "names specify the correct resource prefix.")
                e = msg.format(col, (SOLAR_PREFIX, WIND_PREFIX))
                logger.error(e)
                raise InputError(e)

    def _validate_ratio_cols_exist(self):
        """Ensure the ratio columns exist if a ratio is specified.

        Raises
        ------
        FileInputError
            If ratio columns are not found in the meta data.
        """

        if self._allowed_ratio is None:
            return

        for col in self._ratio_cols:
            no_prefix_name = "_".join(col.split('_')[1:])
            if not self.data.contains_col(no_prefix_name):
                msg = ("Input ratio column {!r} not found in either meta "
                       "data! Please check the input files {!r} and {!r}")
                e = msg.format(no_prefix_name, self.data.solar_fpath,
                               self.data.wind_fpath)
                logger.error(e)
                raise FileInputError(e)

    def _validate_ratio(self):
        """Ensure the ratio value is input correctly.

        Raises
        ------
        InputError
            If ratio is not one of: `None`, float, or len 2 container.
        """
        if self._allowed_ratio is None:
            self._allowed_ratio = (0, float('inf'))
            return

        try:
            if len(self._allowed_ratio) != 2:
                msg = ("Input for 'allowed_ratio' not understood: {!r}. "
                       "Please make sure this value is one of: `None`, "
                       "float, or len 2 container.")
                e = msg.format(self._allowed_ratio)
                logger.error(e)
                raise InputError(e)
        except TypeError:
            self._allowed_ratio = (self._allowed_ratio, self._allowed_ratio)

    def hybridize(self):
        """Combine the solar and wind metas and run hybridize methods."""
        self._format_meta_pre_merge()
        self._merge_solar_wind_meta()
        self._verify_lat_long_match_post_merge()
        self._format_meta_post_merge()
        self._fillna_meta_cols()
        self._limit_by_ratio()
        self._add_hybrid_cols()
        self._sort_hybrid_meta_cols()

    def _format_meta_pre_merge(self):
        """Prepare solar and wind meta for merging. """
        self.__col_name_map = {
            ColNameFormatter.fmt(c): c
            for c in self.data.solar_meta.columns.values
        }

        self._rename_cols(self.data.solar_meta, prefix=SOLAR_PREFIX)
        self._rename_cols(self.data.wind_meta, prefix=WIND_PREFIX)

        self._save_rep_prof_index_internally()

    @staticmethod
    def _rename_cols(df, prefix):
        """Replace column names with the ColNameFormatter.fmt is needed. """
        df.columns = [
            ColNameFormatter.fmt(col_name)
            if col_name in NON_DUPLICATE_COLS
            else '{}{}'.format(prefix, col_name)
            for col_name in df.columns.values
        ]

    def _save_rep_prof_index_internally(self):
        """Save rep profiles index in hybrid meta for access later. """

        self.data.solar_meta[self.__solar_rpi_n] = self.data.solar_meta.index
        self.data.wind_meta[self.__wind_rpi_n] = self.data.wind_meta.index

    def _merge_solar_wind_meta(self):
        """Merge the wind and solar meta DetaFrames. """
        self._hybrid_meta = self.data.solar_meta.merge(
            self.data.wind_meta,
            on=ColNameFormatter.fmt(MERGE_COLUMN),
            suffixes=[None, '_x'], how=self._merge_type()
        )

    def _merge_type(self):
        """Determine the type of merge to use for meta based on user input. """
        if self._allow_solar_only and self._allow_wind_only:
            return 'outer'
        elif self._allow_solar_only and not self._allow_wind_only:
            return 'left'
        elif not self._allow_solar_only and self._allow_wind_only:
            return 'right'
        return 'inner'

    def _format_meta_post_merge(self):
        """Format hybrid meta after merging. """

        duplicate_cols = [n for n in self._hybrid_meta.columns if "_x" in n]
        self._propagate_duplicate_cols(duplicate_cols)
        self._drop_cols(duplicate_cols)
        self._hybrid_meta.rename(self.__col_name_map, inplace=True, axis=1)
        self._hybrid_meta.index.name = 'gid'

    def _propagate_duplicate_cols(self, duplicate_cols):
        """Fill missing column values from outer merge. """
        for duplicate in duplicate_cols:
            no_sufflix = "_".join(duplicate.split("_")[:-1])
            null_idx = self._hybrid_meta[no_sufflix].isnull()
            non_null_vals = self._hybrid_meta.loc[null_idx, duplicate].values
            self._hybrid_meta.loc[null_idx, no_sufflix] = non_null_vals

    def _drop_cols(self, duplicate_cols):
        """Drop any remaning duplicate and 'DROPPED_COLUMNS' columns. """
        self._hybrid_meta.drop(
            duplicate_cols + DROPPED_COLUMNS,
            axis=1, inplace=True, errors='ignore'
        )

    def _sort_hybrid_meta_cols(self):
        """Sort the columns of the hybrid meta. """
        self.__hybrid_meta_cols = sorted(
            [c for c in self._hybrid_meta.columns
             if not c.startswith(self._INTERNAL_COL_PREFIX)],
            key=self._column_sorting_key
        )

    def _column_sorting_key(self, c):
        """Helper function to sort hybrid meta columns. """
        first_index = 0
        if c.startswith('hybrid'):
            first_index = 1
        elif c.startswith('solar'):
            first_index = 2
        elif c.startswith('wind'):
            first_index = 3
        elif c == MERGE_COLUMN:
            first_index = -1
        return first_index, self._hybrid_meta.columns.get_loc(c)

    def _verify_lat_long_match_post_merge(self):
        """Verify that all the lat/lon values match post merge."""
        lat = self._verify_col_match_post_merge(col_name='latitude')
        lon = self._verify_col_match_post_merge(col_name='longitude')
        if not lat or not lon:
            msg = ("Detected mismatched coordinate values (latitude or "
                   "longitude) post merge. Please ensure that all matching "
                   "values of {!r} correspond to the same values of latitude "
                   "and longitude across the input files {!r} and {!r}")
            e = msg.format(MERGE_COLUMN, self.data.solar_fpath,
                           self.data.wind_fpath)
            logger.error(e)
            raise FileInputError(e)

    def _verify_col_match_post_merge(self, col_name):
        """Verify that all (non-null) values in a column match post merge. """
        c1, c2 = col_name, '{}_x'.format(col_name)
        if c1 in self._hybrid_meta.columns and c2 in self._hybrid_meta.columns:
            compare_df = self._hybrid_meta[
                (self._hybrid_meta[c1].notnull())
                & (self._hybrid_meta[c2].notnull())
            ]
            return np.allclose(compare_df[c1], compare_df[c2])
        else:
            return True

    def _fillna_meta_cols(self):
        """Fill N/A values as specified by user. """
        for col_name, fill_value in self._fillna.items():
            if col_name in self._hybrid_meta.columns:
                self._hybrid_meta[col_name].fillna(fill_value, inplace=True)
            else:
                msg = ("Skipping fill values for {!r}: Unable to find column "
                       "in hybrid meta. Did you forget to prefilx with "
                       "{!r} or {!r}? ")
                w = msg.format(col_name, SOLAR_PREFIX, WIND_PREFIX)
                logger.warning(w)
                warn(w, InputWarning)

        self._hybrid_meta[self.__solar_rpi_n].fillna(-1, inplace=True)
        self._hybrid_meta[self.__wind_rpi_n].fillna(-1, inplace=True)

    def _limit_by_ratio(self):
        """ Limit the ratio columns based on input ratio. """
        numerator_col, denominator_col = self._ratio_cols
        min_ratio, max_ratio = sorted(self._allowed_ratio)
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
        """Add new hybrid columns using registered hybrid methods. """
        for new_col_name, method in HYBRID_METHODS.items():
            out = method(self)
            if out is not None:
                try:
                    self._hybrid_meta[new_col_name] = out
                except ValueError as e:
                    msg = ("Unable to add {!r} column to hybrid meta. The "
                           "following exception was raised when adding "
                           "the data output by '{}': {!r}.")
                    w = msg.format(new_col_name, method.__name__, e)
                    logger.warning(w)
                    warn(w, OutputWarning)

    @property
    def solar_profile_indices_map(self):
        """Map hybrid to solar rep indices.
        Returns
        -------
        hybrid_indicies : np.ndarray
            Index values corresponding to hybrid rep profiles.
        solar_indicies : np.ndarray
            Index values of the solar rep profiles corresponding
            to the hybrid rep profile indicies.
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
        hybrid_indicies : np.ndarray
            Index values corresponding to hybrid rep profiles.
        wind_indicies : np.ndarray
            Index values of the wind rep profiles corresponding
            to the hybrid rep profile indicies.
        """
        if self._hybrid_meta is None:
            return np.array([]), np.array([])

        idxs = self._hybrid_meta[self.__wind_rpi_n].astype(int)
        idxs = idxs[idxs >= 0]

        return idxs.index.values, idxs.values


class Hybridization:
    """Framework to handle hybridization of SC and corresponding profiles."""

    def __init__(self, solar_fpath, wind_fpath, allow_solar_only=False,
                 allow_wind_only=False, fillna=None, allowed_ratio=None,
                 ratio_cols=('solar_capacity', 'wind_capacity')):
        """
        Parameters
        ----------
        solar_fpath : str
            Filepath to rep profile output file to extract solar profiles and
            summaries from.
        wind_fpath : str
            Filepath to rep profile output file to extract wind profiles and
            summaries from.
        allow_solar_only : bool, optional
            Option to allow SC points with only solar capcity (no wind), by
            default False.
        allow_wind_only : bool, optional
            Option to allow SC points with only wind capcity (no solar), by
            default False.
        fillna : dict, optional
            Dictionary containing column_name, fill_value pairs reprenting any
            fill values that should be applied after merging the wind and solar
            meta. Note that column names will likely have to be prefixed
            with "solar_" or "wind_". By default None.
        allowed_ratio : float | tuple, optional
            Option to set a single ratio or ratio bounds (in two-tuple form)
            on the `ratio_cols`. A single value is treated as both an upper
            and a lower bound. This input limits the ratio of the values
            of the input `ratio_cols` (ratio is always computed with the
            first column as the numerator and the second column as the
            denomiator). For example, `allowed_ratio=1` would limit
            the values of the `ratio_cols` to always be equal. On the other
            hand, `allowed_ratio=(0.5, 1.5)` would limit the ratio to be
            between half and double (e.g., if `ratio_cols` are the capacity
            columns, no capacity value would be more than double the other).
            By default, None (no limit).
        ratio_cols : tuple, optional
            Option to specify the columns used to calculate the ratio that is
            limited by the `allowed_ratio` input. If `allowed_ratio` is None,
            this input does nothing. The names of the columns should be
            prefixed with one of the prefixes defined as class variables.
            The order of the colum names specifies the way the ratio is
            calculated: the first column is always treated as the ratio
            numerator and the second column is the ratio denominator.
            By default ('solar_capacity', 'wind_capacity').
        """

        logger.info('Running hybridization rep profiles with solar_fpath: "{}"'
                    .format(solar_fpath))
        logger.info('Running hybridization rep profiles with solar_fpath: "{}"'
                    .format(wind_fpath))
        logger.info('Running hybridization rep profiles with '
                    'allow_solar_only: "{}"'.format(allow_solar_only))
        logger.info('Running hybridization rep profiles with '
                    'allow_wind_only: "{}"'.format(allow_wind_only))
        logger.info('Running hybridization rep profiles with fillna: "{}"'
                    .format(fillna))
        logger.info('Running hybridization rep profiles with '
                    'allowed_ratio: "{}"'.format(allowed_ratio))
        logger.info('Running hybridization rep profiles with ratio_cols: "{}"'
                    .format(ratio_cols))

        self.data = HybridsData(solar_fpath, wind_fpath)
        self.meta_hybridizer = MetaHybridizer(
            data=self.data, allow_solar_only=allow_solar_only,
            allow_wind_only=allow_wind_only, fillna=fillna,
            allowed_ratio=allowed_ratio, ratio_cols=ratio_cols
        )
        self._profiles = None
        self._validate_input()

    def _validate_input(self):
        """Validate the user input and input files. """
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
            Filepath to output h5 file, by default None.
        save_hybrid_meta : bool, optional
            Flag to save hybrid SC table to hybrid rep profile output,
            by default True.

        Returns
        -------
        `Hybridization`
            Instance of Hybridization object (itself) containing the
            hybridized meta and profiles as attributes.
        """

        self.meta_hybridizer.hybridize()
        self._init_profiles()
        out = self._run()

        if fout is not None:
            self.save_profiles(fout, save_hybrid_meta=save_hybrid_meta)

        logger.info('Hybridization of representative profiles complete!')
        return out

    def _init_profiles(self):
        """Initialize the output rep profiles attribute."""
        self._profiles = {
            k: np.zeros((len(self.hybrid_time_index), len(self.hybrid_meta)),
                        dtype=np.float32)
            for k in OUTPUT_PROFILE_NAMES}

    def _run(self):
        """Compute all hybridized profiles."""

        logger.info('Running hybrid profile calculations.')

        self._compute_hybridized_resource_profiles()
        self._compute_hybridized_profiles_from_components()

        logger.info('Profile hybridization complete.')

        return self

    def _compute_hybridized_resource_profiles(self):
        """Compute the resource components of the hybridized profiles. """

        for params in self.__rep_profile_hybridization_params:
            col, (hybrid_idxs, solar_idxs), fpath, p_name, dset_name = params
            capacity = self.hybrid_meta.loc[hybrid_idxs, col].values

            with Resource(fpath) as res:
                data = res[dset_name,
                           res.time_index.isin(self.hybrid_time_index)]
                self._profiles[p_name][:, hybrid_idxs] = (data[:, solar_idxs]
                                                          * capacity)

    @property
    def __rep_profile_hybridization_params(self):
        """Zip the rep profile hybridization parameters. """

        cap_col_names = ['hybrid_solar_capacity', 'hybrid_wind_capacity']
        idx_maps = [self.meta_hybridizer.solar_profile_indices_map,
                    self.meta_hybridizer.wind_profile_indices_map]
        fpaths = [self.data.solar_fpath, self.data.wind_fpath]
        zipped = zip(cap_col_names, idx_maps, fpaths, OUTPUT_PROFILE_NAMES[1:],
                     self.data.profile_dset_names)
        return zipped

    def _compute_hybridized_profiles_from_components(self):
        """Compute the hybridized profiles from the resource components. """

        hp_name, sp_name, wp_name = OUTPUT_PROFILE_NAMES
        self._profiles[hp_name] = (self._profiles[sp_name]
                                   + self._profiles[wp_name])

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

        Outputs.init_h5(fout, dsets, shapes, attrs, chunks, dtypes,
                        meta, time_index=self.hybrid_time_index)

        if save_hybrid_meta:
            with Outputs(fout, mode='a') as out:
                hybrid_meta = to_records_array(self.hybrid_meta)
                out._create_dset('meta', hybrid_meta.shape,
                                 hybrid_meta.dtype, data=hybrid_meta)

    def _write_h5_out(self, fout, save_hybrid_meta=True):
        """Write hybrid profiles and meta to an output file.

        Parameters
        ----------
        fout : str
            Filepath to output h5 file.
        save_hybrid_meta : bool
            Flag to save hybrid SC table to hybrid rep profile output.
        """

        with Outputs(fout, mode='a') as out:
            if 'meta' in out.datasets and save_hybrid_meta:
                hybrid_meta = to_records_array(self.hybrid_meta)
                out['meta'] = hybrid_meta

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
