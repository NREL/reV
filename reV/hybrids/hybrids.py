# -*- coding: utf-8 -*-
"""Hybridization utilities.

@author: ppinchuk
"""

from concurrent.futures import as_completed
from copy import deepcopy
import json
import logging
import numpy as np
import os
import re
import pandas as pd
from scipy import stats
from string import ascii_letters
from warnings import warn


from reV.handlers.outputs import Outputs
from reV.utilities.exceptions import (FileInputError, InputError,
                                      InputWarning, OutputWarning)
from reV.utilities import log_versions

from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem
from rex.utilities.utilities import parse_year, to_records_array

logger = logging.getLogger(__name__)
HYBRID_METHODS = {}


def hybrid_col(col_name):
    """A decorator factory that facitilitates the registry of new hybrids.

    This decorator takes a column name as input and registers the decorated
    function as a method that computes a hybrid variable. During the
    hybridization step, the registered function (which takes an instance
    of the hybridization object as input) will be run and its output
    will be stored in the new hybrid meta DataFrame under the registered column
    name.

    Parameters
    ----------
    col_name : str
        Name of the new hybrid column. This should typically start with
        "hybrid_".

    Examples
    --------
    Writing and registering a new hybridization:

    >>> from reV.hybrids import hybrid_col, Hybridization
    >>> SOLAR_FPATH = '/path/to/input/solar/file.h5
    >>> WIND_FPATH = '/path/to/input/wind/file.h5
    >>>
    >>> @hybrid_col('scaled_elevation')
    >>> def some_new_hybrid_func(h):
    >>>     return h.hybrid_meta['elevation'] * 1000
    >>>
    >>> __, hybrid_meta, __ = Hybridization.run(SOLAR_FPATH, WIND_FPATH)
    >>> assert 'scaled_elevation' in hybrid_meta.columns

    """
    def _register(func):
        HYBRID_METHODS[col_name] = func
        return func
    return _register


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


class RepProfileHybridizer:
    """Framework to handle hybridization of representative profiles"""

    def __init__(self, solar_fpath, wind_fpath, hybrid_solar_capacity,
                 hybrid_wind_capacity, solar_rep_idx, wind_rep_idx,
                 hybrid_time_idx, rep_profile_dset_names):
        """
        Parameters
        ----------
        solar_fpath : str
            Filepath to rep profile output file to extract solar profiles and
            summaries from.
        wind_fpath : str
            Filepath to rep profile output file to extract wind profiles and
            summaries from.
        hybrid_solar_capacity : float
            Built-out hybrid solar capacity. Used as a weight for profile
            aggregation (meanoid).
        hybrid_wind_capacity : float
            Built-out hybrid wind capacity. Used as a weight for profile
            aggregation (meanoid).
        solar_rep_idx : int
            Index of the solar representative profile.
        wind_rep_idx : int
            Index of the wind representative profile.
        hybrid_time_idx : pd.DatetimeIndex
            DatetimeIndex for the hybrid profile.
        rep_profile_dset_names : iterable of length 2
            Iterable containing the rep profile names for solar and wind,
            in that order.
        """

        self.hybrid_solar_capacity = hybrid_solar_capacity
        self.hybrid_wind_capacity = hybrid_wind_capacity
        self._hti = hybrid_time_idx
        self._fp = {'solar': solar_fpath, 'wind': wind_fpath}
        self._ri = {'solar': solar_rep_idx, 'wind': wind_rep_idx}
        self._rpdn = {r: dn for r, dn in
                      zip(['solar', 'wind'], rep_profile_dset_names)}
        self._source_cf_profiles = {}

    def _set_source_cf_profile(self, resource):
        """Extract source profile data, if needeed. """
        if self._source_cf_profiles.get(resource) is None:
            self._source_cf_profiles[resource] = np.zeros(len(self._hti),
                                                          dtype=np.float32)
            if self._ri[resource] >= 0:
                with Resource(self._fp[resource]) as res:
                    self._source_cf_profiles[resource] = res[
                        self._rpdn[resource],
                        res.time_index.isin(self._hti),
                        self._ri[resource]
                    ]

    @property
    def solar_cf_profile(self):
        """Retrieve the solar cf profile array from the input h5 file.

        Returns
        -------
        profile : np.array
            Timeseries array of solar cf profile data.
        """
        self._set_source_cf_profile('solar')
        return self._source_cf_profiles['solar']

    @property
    def wind_cf_profile(self):
        """Retrieve the wind cf profile array from the input h5 file.

        Returns
        -------
        profile : np.array
            Timeseries array of wind cf profile data.
        """
        self._set_source_cf_profile('wind')
        return self._source_cf_profiles['wind']

    @property
    def solar_built_capacity(self):
        """Calculate the built-out solar representative profile array.

        Returns
        -------
        profile : np.array
            Timeseries array of built-out solar capacity.
        """
        return self.solar_cf_profile * self.hybrid_solar_capacity

    @property
    def wind_built_capacity(self):
        """Calculate the built-out wind representative profile array.

        Returns
        -------
        profile : np.array
            Timeseries array of built-out wind capacity.
        """
        return self.wind_cf_profile * self.hybrid_wind_capacity

    @property
    def hybrid_built_capacity(self):
        """Calculate the built-out hybrid representative profile array.

        Returns
        -------
        profile : np.array
            Timeseries array of built-out bybrid capacity.
        """
        return self.solar_built_capacity + self.wind_built_capacity

    @classmethod
    def get_hybrid_rep_profiles(cls, solar_fpath, wind_fpath,
                                hybrid_solar_capacity, hybrid_wind_capacity,
                                solar_rep_idx, wind_rep_idx, hybrid_time_idx,
                                rep_profile_dset_names):
        """Class method for parallelization of rep profile calc.

        Parameters
        ----------
        solar_fpath : str
            Filepath to rep profile output file to extract solar profiles and
            summaries from.
        wind_fpath : str
            Filepath to rep profile output file to extract wind profiles and
            summaries from.
        hybrid_solar_capacity : float
            Built-out hybrid solar capacity. Used as a weight for profile
            aggregation (meanoid).
        hybrid_wind_capacity : float
            Built-out hybrid wind capacity. Used as a weight for profile
            aggregation (meanoid).
        solar_rep_idx : int
            Index of the solar representative profile.
        wind_rep_idx : int
            Index of the wind representative profile.
        hybrid_time_idx : pd.DatetimeIndex
            DatetimeIndex for the hybrid profile.
        rep_profile_dset_names : iterable of length 2
            Iterable containing the rep profile names for solar and wind,
            in that order.

        Returns
        -------
        hybrid_built_capacity : np.array
            Array with data for the built-out hybrid capacity over time.
        solar_built_capacity : np.array
            Array with data for the built-out solar capacity over time.
        wind_built_capacity : np.array
            Array with data for the built-out wind capacity over time.
        """
        r = cls(solar_fpath, wind_fpath, hybrid_solar_capacity,
                hybrid_wind_capacity, solar_rep_idx, wind_rep_idx,
                hybrid_time_idx, rep_profile_dset_names)

        return (r.hybrid_built_capacity,
                r.solar_built_capacity,
                r.wind_built_capacity)


class Hybridization:
    """Framework to handle hybridization of SC and corresponding profiles."""

    NON_DUPLICATE_COLS = {
        'latitude', 'longitude', 'country', 'state', 'county', 'elevation',
        'timezone', 'sc_point_gid', 'sc_row_ind', 'sc_col_ind'
    }
    DROPPED_COLUMNS = ['gid']
    MERGE_COLUMN = 'sc_point_gid'
    SOLAR_PREFIX = 'solar_'
    WIND_PREFIX = 'wind_'
    DEFAULT_FILL_VALUES = {'solar_capacity': 0, 'wind_capacity': 0,
                           'solar_mean_cf': 0, 'wind_mean_cf': 0}
    PROFILE_DSET_REGEX = 'rep_profiles_[0-9]+$'
    OUTPUT_PROFILE_NAMES = ['hybrid', 'solar_time_built', 'wind_time_built']
    _INTERNAL_COL_PREFIX = '_h_internal'

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
            meta, by default None.
        allowed_ratio : float | tuple, optional
            Option to set a ratio or ratio bounds (in two-tuple form) on the
            `ratio_cols`. This number would limit the hybridization values to
            the ratio value. By default, None (no limit).
        ratio_cols : tuple, optional
            Option to specify the columns used to calculate the ratio that is
            limited by the `allowed_ratio` input. If `allowed_ratio` is None,
            this input does nothing. The names of the columns should be
            prefixed with one of the prefixes defined as class variables.
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

        self._solar_fpath = solar_fpath
        self._wind_fpath = wind_fpath
        self._allow_solar_only = allow_solar_only
        self._allow_wind_only = allow_wind_only
        self.DEFAULT_FILL_VALUES.update(fillna or {})
        self._allowed_ratio = allowed_ratio
        self._ratio_cols = ratio_cols
        self._profiles = None
        self._solar_meta = None
        self._wind_meta = None
        self._hybrid_meta = None
        self._solar_time_index = None
        self._wind_time_index = None
        self._hybrid_time_index = None
        self.__merge_col_overlap = None
        self.__profile_reg_check = re.compile(self.PROFILE_DSET_REGEX)
        self.__profile_dset_names = []
        self.__solar_cols = self.solar_meta.columns.map(ColNameFormatter.fmt)
        self.__wind_cols = self.wind_meta.columns.map(ColNameFormatter.fmt)
        self.__hybrid_meta_cols = None
        self.__col_name_map = None
        self.__solar_rpi_n = '{}_solar_rpidx'.format(self._INTERNAL_COL_PREFIX)
        self.__wind_rpi_n = '{}_wind_rpidx'.format(self._INTERNAL_COL_PREFIX)

        self._validate_input()

    def _validate_input(self):
        """Validate the user input and input files. """

        self._validate_time_index()
        self._validate_num_profiles()
        self._validate_merge_col_exists()
        self._validate_unique_merge_col()
        self._validate_merge_col_overlaps()
        self._validate_ratio_cols_length()
        self._validate_ratio_cols_prefixed()
        self._validate_ratio_cols_exist()
        self._validate_ratio()

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
        for fp in [self._solar_fpath, self._wind_fpath]:
            with Resource(fp) as res:
                profile_dset_names = [
                    n for n in res.dsets
                    if self.__profile_reg_check.match(n)
                ]
                if not profile_dset_names:
                    msg = ("Did not find any data sets matching the regex: "
                           "{!r} in {!r}. Please ensure that the profile data "
                           "exists and that the data set is named correctly.")
                    e = msg.format(self.PROFILE_DSET_REGEX, fp)
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
                    self.__profile_dset_names += profile_dset_names

    def _validate_merge_col_exists(self):
        """Validate the existence of the merge column.

        Raises
        ------
        FileInputError
            If merge column is missing from either the solar or
            the wind meta data.
        """
        if ColNameFormatter.fmt(self.MERGE_COLUMN) not in self.__solar_cols:
            msg = ("Cannot hybridize: merge column {!r} missing from the "
                   "solar meta data! ({!r})")
            e = msg.format(self.MERGE_COLUMN, self._solar_fpath)
            logger.error(e)
            raise FileInputError(e)

        if ColNameFormatter.fmt(self.MERGE_COLUMN) not in self.__wind_cols:
            msg = ("Cannot hybridize: merge column {!r} missing from the "
                   "wind meta data! ({!r})")
            e = msg.format(self.MERGE_COLUMN, self._wind_fpath)
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

        merge_col = self.solar_meta.columns[
            self.__solar_cols == ColNameFormatter.fmt(self.MERGE_COLUMN)
        ].item()
        if not self.solar_meta[merge_col].is_unique:
            e = msg.format(merge_col, merge_col, self._solar_fpath)
            logger.error(e)
            raise FileInputError(e)

        merge_col = self.wind_meta.columns[
            self.__wind_cols == ColNameFormatter.fmt(self.MERGE_COLUMN)
        ].item()
        if not self.wind_meta[merge_col].is_unique:
            e = msg.format(merge_col, merge_col, self._wind_fpath)
            logger.error(e)
            raise FileInputError(e)

    def _validate_merge_col_overlaps(self):
        """Validate the existence of overlap in the merge column values.

        Raises
        ------
        FileInputError
            If merge column values do not overlap between the tow input files.
        """
        merge_col = self.solar_meta.columns[
            self.__solar_cols == ColNameFormatter.fmt(self.MERGE_COLUMN)
        ].item()
        solar_vals = set(self.solar_meta[merge_col].values)
        merge_col = self.wind_meta.columns[
            self.__wind_cols == ColNameFormatter.fmt(self.MERGE_COLUMN)
        ].item()
        wind_vals = set(self.wind_meta[merge_col].values)
        self.__merge_col_overlap = solar_vals & wind_vals

        if not self.__merge_col_overlap:
            msg = ("No overlap detected in the values of {!r} across the "
                   "input files. Please ensure that at least one of the "
                   "{!r} values is the same for input files {!r} and {!r}")
            e = msg.format(merge_col, merge_col, self._solar_fpath,
                           self._wind_fpath)
            logger.error(e)
            raise FileInputError(e)

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
                   "Please make sure this value is a two-tuple containg"
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
            missing_solar_prefix = not col.startswith(self.SOLAR_PREFIX)
            missing_wind_prefix = not col.startswith(self.WIND_PREFIX)
            if missing_solar_prefix and missing_wind_prefix:
                msg = ("Input ratio column {!r} does not start with a valid "
                       "prefix: {!r}. Please ensure that the ratio column "
                       "names specify the correct resource prefix.")
                e = msg.format(col, (self.SOLAR_PREFIX, self.WIND_PREFIX))
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
            fmt_name = ColNameFormatter.fmt(no_prefix_name)
            col_in_solar = fmt_name in self.__solar_cols
            col_in_wind = fmt_name in self.__wind_cols
            if not col_in_solar and not col_in_wind:
                msg = ("Input ratio column {!r} not found in either meta "
                       "data! Please check the input files {!r} and {!r}")
                e = msg.format(no_prefix_name, self._solar_fpath,
                               self._wind_fpath)
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

    @property
    def solar_meta(self):
        """Summary for the solar representative profiles.

        Returns
        -------
        solar_meta : pd.DataFrame
            Summary for the solar representative profiles.
        """
        if self._solar_meta is None:
            with Resource(self._solar_fpath) as res:
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
            with Resource(self._wind_fpath) as res:
                self._wind_meta = res.meta
        return self._wind_meta

    @property
    def hybrid_meta(self):
        """Hybridized summary for the representative profiles.

        Returns
        -------
        hybrid_meta : pd.DataFrame
            Summary for the hybridized representative profiles.
            At the very least, this has a column that the data was merged on.
        """
        if self.__hybrid_meta_cols is None:
            return self._hybrid_meta
        else:
            return self._hybrid_meta[self.__hybrid_meta_cols]

    @property
    def solar_time_index(self):
        """Get the time index for the solar rep profiles.

        Returns
        -------
        solar_time_index : pd.datetimeindex
            Time index sourced from the solar reV gen file.
        """
        if self._solar_time_index is None:
            with Resource(self._solar_fpath) as res:
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
            with Resource(self._wind_fpath) as res:
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

    @property
    def profiles(self):
        """Get the arrays of the hybridized representative profiles.

        Returns
        -------
        profiles : dict
            dict of n_profile-keyed arrays with shape (time, n) for the
            hybridized representative profiles for each region.
        """
        return self._profiles

    @hybrid_col('hybrid_solar_capacity')
    def aggregate_solar_capacity(self):
        """Compute the total solar capcity allowed in hybridization.

        Note
        ----
        No limiting is done on the ratio of wind to solar. This method
        checks for an existing 'hybrid_solar_capacity'. If one does not exist,
        it is assumed that there is no limit on the solar to wind capacity
        ratio and the solar capacity is copied into this new column.

        Returns
        -------
        data : Series | None
            A series of data containing the capacity allowed in the hybrid
            capacity sum, or `None` if 'hybrid_solar_capacity' already exists.

        Notes
        -----

        """
        if 'hybrid_solar_capacity' in self.hybrid_meta:
            return None
        return self.hybrid_meta['solar_capacity']

    @hybrid_col('hybrid_wind_capacity')
    def aggregate_wind_capacity(self):
        """Compute the total wind capcity allowed in hybridization.

        Note
        ----
        No limiting is done on the ratio of wind to solar. This method
        checks for an existing 'hybrid_wind_capacity'. If one does not exist,
        it is assumed that there is no limit on the solar to wind capacity
        ratio and the wind capacity is copied into this new column.

        Returns
        -------
        data : Series | None
            A series of data containing the capacity allowed in the hybrid
            capacity sum, or `None` if 'hybrid_solar_capacity' already exists.

        Notes
        -----

        """
        if 'hybrid_wind_capacity' in self.hybrid_meta:
            return None
        return self.hybrid_meta['wind_capacity']

    @hybrid_col('hybrid_capacity')
    def aggregate_capacity(self):
        """Compute the total capcity by summing the individual capacities.

        Returns
        -------
        data : Series | None
            A series of data containing the aggregated capacity, or `None`
            if the capacity columns are missing.
        """

        sc, wc = 'hybrid_solar_capacity', 'hybrid_wind_capacity'
        missing_solar_cap = sc not in self.hybrid_meta.columns
        missing_wind_cap = wc not in self.hybrid_meta.columns
        if missing_solar_cap or missing_wind_cap:
            return None

        total_cap = self.hybrid_meta[sc] + self.hybrid_meta[wc]
        return total_cap

    @hybrid_col('hybrid_mean_cf')
    def aggregate_capacity_factor(self):
        """Compute the capacity-weighted mean capcity factor.

        Returns
        -------
        data : Series | None
            A series of data containing the aggregated capacity, or `None`
            if the capacity and/or mean_cf columns are missing.
        """

        sc, wc = 'hybrid_solar_capacity', 'hybrid_wind_capacity'
        scf, wcf = 'solar_mean_cf', 'wind_mean_cf'
        missing_solar_cap = sc not in self.hybrid_meta.columns
        missing_wind_cap = wc not in self.hybrid_meta.columns
        missing_solar_mean_cf = scf not in self.hybrid_meta.columns
        missing_wind_mean_cf = wcf not in self.hybrid_meta.columns
        missing_any = (missing_solar_cap or missing_wind_cap
                       or missing_solar_mean_cf or missing_wind_mean_cf)
        if missing_any:
            return None

        solar_cf_weighted = self.hybrid_meta[sc] * self.hybrid_meta[scf]
        wind_cf_weighted = self.hybrid_meta[wc] * self.hybrid_meta[wcf]
        total_capacity = self.aggregate_capacity()
        hybrid_cf = (solar_cf_weighted + wind_cf_weighted) / total_capacity
        return hybrid_cf

    def _run(self, fout=None, save_hybrid_meta=True, scaled_precision=False,
             max_workers=None):
        """
        Run hybridization of profiles in serial or parallel and save to disc

        Parameters
        ----------
        fout : str, optional
            filepath to output h5 file, by default None
        save_hybrid_meta : bool, optional
            Flag to save hybrid reV SC table to rep profile output.,
            by default True
        scaled_precision : bool, optional
            Flag to scale cf_profiles by 1000 and save as uint16.,
            by default False
        max_workers : int, optional
            Number of parallel workers. 1 will run serial, None will use all
            available., by default None
        """

        self._hybridize_meta()
        self._init_profiles()

        if max_workers == 1:
            self._run_serial()
        else:
            self._run_parallel(max_workers=max_workers)

        if fout is not None:
            self.save_profiles(fout, save_hybrid_meta=save_hybrid_meta,
                               scaled_precision=scaled_precision)

        logger.info('Hybridization of representative profiles complete!')

    def _hybridize_meta(self):
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
            for c in self.solar_meta.columns.values
        }

        self._rename_cols(self.solar_meta, prefix=self.SOLAR_PREFIX)
        self._rename_cols(self.wind_meta, prefix=self.WIND_PREFIX)

        self._save_rep_prof_index_internally()

    def _rename_cols(self, df, prefix):
        """Replace column names with the ColNameFormatter.fmt is needed. """
        df.columns = [
            ColNameFormatter.fmt(col_name)
            if col_name in self.NON_DUPLICATE_COLS
            else '{}{}'.format(prefix, col_name)
            for col_name in df.columns.values
        ]

    def _save_rep_prof_index_internally(self):
        """Save rep profiles index in hybrid meta for access later. """

        self.solar_meta[self.__solar_rpi_n] = self.solar_meta.index
        self.wind_meta[self.__wind_rpi_n] = self.wind_meta.index

    def _merge_solar_wind_meta(self):
        """Merge the wind and solar meta DetaFrames. """
        self._hybrid_meta = self.solar_meta.merge(
            self.wind_meta, on=ColNameFormatter.fmt(self.MERGE_COLUMN),
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
            duplicate_cols + self.DROPPED_COLUMNS,
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
        elif c == self.MERGE_COLUMN:
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
            e = msg.format(self.MERGE_COLUMN, self._solar_fpath,
                           self._wind_fpath)
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
            return (compare_df[c1] == compare_df[c2]).all()
        else:
            return True

    def _fillna_meta_cols(self):
        """Fill N/A values as specified by user. """
        for col_name, fill_value in self.DEFAULT_FILL_VALUES.items():
            if col_name in self._hybrid_meta.columns:
                self._hybrid_meta[col_name].fillna(fill_value, inplace=True)
            else:
                msg = ("Skipping fill values for {!r}: Unable to find column "
                       "in hybrid meta. Did you forget to prefilx with "
                       "{!r} or {!r}? ")
                w = msg.format(col_name, self.SOLAR_PREFIX, self.WIND_PREFIX)
                logger.warning(w)
                warn(w, InputWarning)

        self._hybrid_meta[self.__solar_rpi_n].fillna(-1, inplace=True)
        self._hybrid_meta[self.__wind_rpi_n].fillna(-1, inplace=True)

    def _limit_by_ratio(self):
        """ Limit the ratio columns based on input ratio. """
        c1, c2 = self._ratio_cols
        min_r, max_r = self._allowed_ratio
        overlap_idx = self._hybrid_meta[self.MERGE_COLUMN].isin(
            self.__merge_col_overlap
        )
        hc1 = self._hybrid_meta[c1].copy()
        hc2 = self._hybrid_meta[c2].copy()
        ratios = (hc1.loc[overlap_idx] / hc2.loc[overlap_idx])
        ratio_too_low = (ratios < min_r) & overlap_idx
        ratio_too_high = (ratios > max_r) & overlap_idx
        hc1.loc[ratio_too_high] = hc2.loc[ratio_too_high].values * max_r
        hc2.loc[ratio_too_low] = hc1.loc[ratio_too_low].values / min_r
        self._hybrid_meta["hybrid_{}".format(c1)] = hc1.values
        self._hybrid_meta["hybrid_{}".format(c2)] = hc2.values

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

    def _init_profiles(self):
        """Initialize the output rep profiles attribute."""
        self._profiles = {
            k: np.zeros((len(self.hybrid_time_index), len(self._hybrid_meta)),
                        dtype=np.float32)
            for k in self.OUTPUT_PROFILE_NAMES}

    def _init_h5_out(self, fout, save_hybrid_meta=True,
                     scaled_precision=False):
        """Initialize an output h5 file for n_profiles

        Parameters
        ----------
        fout : str
            None or filepath to output h5 file.
        save_hybrid_meta : bool
            Flag to save hybridized reV SC table to rep profile output.
        scaled_precision : bool
            Flag to scale cf_profiles by 1000 and save as uint16.
        """
        dsets = []
        shapes = {}
        attrs = {}
        chunks = {}
        dtypes = {}

        for i in range(self._n_profiles):
            dset = 'rep_profiles_{}'.format(i)
            dsets.append(dset)
            shapes[dset] = self.profiles[0].shape
            chunks[dset] = None

            if scaled_precision:
                attrs[dset] = {'scale_factor': 1000}
                dtypes[dset] = np.uint16
            else:
                attrs[dset] = None
                dtypes[dset] = self.profiles[0].dtype

        meta = self.hybrid_meta.copy()
        for c in meta.columns:
            try:
                meta[c] = pd.to_numeric(meta[c])
            except ValueError:
                pass

        Outputs.init_h5(fout, dsets, shapes, attrs, chunks, dtypes,
                        meta, time_index=self.time_index)

        if save_hybrid_meta:
            with Outputs(fout, mode='a') as out:
                hybrid_meta = to_records_array(self.hybrid_meta)
                out._create_dset('meta', hybrid_meta.shape,
                                 hybrid_meta.dtype, data=hybrid_meta)

    def _write_h5_out(self, fout, save_hybrid_meta=True):
        """Write profiles and meta to an output file.

        Parameters
        ----------
        fout : str
            None or filepath to output h5 file.
        save_hybrid_meta : bool
            Flag to save hybridized reV SC table to rep profile output.
        scaled_precision : bool
            Flag to scale cf_profiles by 1000 and save as uint16.
        """
        with Outputs(fout, mode='a') as out:

            if 'hybrid_meta' in out.datasets and \
                    save_hybrid_meta:
                hybrid_meta = to_records_array(self._hybrid_meta)
                out['meta'] = hybrid_meta

            for i in range(self._n_profiles):
                dset = 'rep_profiles_{}'.format(i)
                out[dset] = self.profiles[i]

    def save_profiles(self, fout, save_hybrid_meta=True,
                      scaled_precision=False):
        """Initialize fout and save profiles.

        Parameters
        ----------
        fout : str
            None or filepath to output h5 file.
        save_hybrid_meta : bool
            Flag to save full reV SC table to rep profile output.
        scaled_precision : bool
            Flag to scale cf_profiles by 1000 and save as uint16.
        """

        self.hybrid_meta.to_csv(fout)
        # self._init_h5_out(fout, save_hybrid_meta=save_hybrid_meta,
        #                   scaled_precision=scaled_precision)
        # self._write_h5_out(fout, save_hybrid_meta=save_hybrid_meta)

    def _run_serial(self):
        """Compute all representative profiles in serial."""

        logger.info('Running {} rep profile calculations in serial.'
                    .format(len(self.hybrid_meta)))

        for i, row in self._hybrid_meta.iterrows():
            logger.debug('Working on profile {} out of {}'
                         .format(i + 1, len(self.hybrid_meta)))

            out = RepProfileHybridizer.get_hybrid_rep_profiles(
                self._solar_fpath, self._wind_fpath,
                row['hybrid_solar_capacity'], row['hybrid_wind_capacity'],
                int(row[self.__solar_rpi_n]), int(row[self.__wind_rpi_n]),
                self.hybrid_time_index, self.__profile_dset_names)

            logger.info('Profile {} out of {} complete '
                        .format(i + 1, len(self.hybrid_meta)))

            for k, p in zip(self.OUTPUT_PROFILE_NAMES, out):
                self._profiles[k][:, i] = p

    def _run_parallel(self, max_workers=None, pool_size=72):
        """Compute all representative profiles in parallel.

        Parameters
        ----------
        max_workers : int | None
            Number of parallel workers. 1 will run serial, None will use all
            available.
        pool_size : int
            Number of futures to submit to a single process pool for
            parallel futures.
        """

        logger.info('Kicking off {} rep profile futures.'
                    .format(len(self.hybrid_meta)))

        iter_chunks = np.array_split(
            self._hybrid_meta.index.values,
            np.ceil(len(self._hybrid_meta) / pool_size)
        )
        n_complete = 0
        for iter_chunk in iter_chunks:
            logger.debug('Starting process pool...')
            futures = {}
            loggers = [__name__, 'reV']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                for i in iter_chunk:
                    row = self._hybrid_meta.loc[i, :]

                    future = exe.submit(
                        RepProfileHybridizer.get_hybrid_rep_profiles,
                        self._solar_fpath, self._wind_fpath,
                        row['hybrid_solar_capacity'],
                        row['hybrid_wind_capacity'],
                        int(row[self.__solar_rpi_n]),
                        int(row[self.__wind_rpi_n]),
                        self.hybrid_time_index,
                        self.__profile_dset_names
                    )

                    futures[future] = i

                for future in as_completed(futures):
                    i = futures[future]
                    out = future.result()
                    n_complete += 1
                    logger.info('Future {} out of {} complete '
                                .format(n_complete, len(self.hybrid_meta)))
                    log_mem(logger, log_level='DEBUG')

                    for k, p in zip(self.OUTPUT_PROFILE_NAMES, out):
                        self._profiles[k][:, i] = p

    @classmethod
    def run(cls, solar_fpath, wind_fpath, allow_solar_only=False,
            allow_wind_only=False, fillna=None, allowed_ratio=None,
            ratio_cols=('solar_capacity', 'wind_capacity'), fout=None,
            save_hybrid_meta=True, scaled_precision=False, max_workers=None):
        """Run hybridization by merging the profiles of each SC region.

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
            meta, by default None.
        allowed_ratio : float | tuple, optional
            Option to set a ratio or ratio bounds (in two-tuple form) on the
            `ratio_cols`. This number would limit the hybridization values to
            the ratio value. By default, None (no limit).
        ratio_cols : tuple, optional
            Option to specify the columns used to calculate the ratio that is
            limited by the `allowed_ratio` input. If `allowed_ratio` is None,
            this input does nothing.
            By default ('solar_capacity', 'wind_capacity').
        fout : str, optional
            filepath to output h5 file, by default None.
        save_hybrid_meta : bool, optional
            Flag to save full reV SC table to rep profile output.,
            by default True.
        scaled_precision : bool, optional
            Flag to scale cf_profiles by 1000 and save as uint16.,
            by default False
        max_workers : int, optional
            Number of parallel workers. 1 will run serial, None will use all
            available., by default None.

        Returns
        -------
        hybrid_profiles : dict
            dict of n_profile-keyed arrays with shape (time, n) for the
            hybridized profiles for each region.
        hybrid_meta : pd.DataFrame
            Meta dataframes recording the regions and the selected rep profile
            gid.
        hybrid_time_index : pd.DatatimeIndex
            Datetime Index for hybridized profiles
        """

        rp = cls(solar_fpath, wind_fpath, allow_solar_only=allow_solar_only,
                 allow_wind_only=allow_wind_only, fillna=fillna,
                 allowed_ratio=allowed_ratio, ratio_cols=ratio_cols)

        rp._run(fout=fout, save_hybrid_meta=save_hybrid_meta,
                scaled_precision=scaled_precision, max_workers=max_workers)

        return (*rp.profiles.values(), rp.hybrid_meta, rp.hybrid_time_index)
