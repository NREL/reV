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
from reV.utilities.exceptions import FileInputError, OutputWarning
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

    >>> from reV.hybrids import hybrid_col
    >>>
    >>> @hybrid_col('scaled_elevation')
    >>> def some_new_hybrid_func(h):
    >>>     return h.hybrid_meta['elevation'] * 1000
    >>>
    >>> h = Hybridization(SOLAR_FPATH, WIND_FPATH)
    >>> h._run()
    >>> assert 'scaled_elevation' in h.hybrid_meta.columns

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


class Hybridization:
    """Framework to handle hybridization for one resource region"""

    NON_DUPLICATE_COLS = {
        'latitude', 'longitude', 'country', 'state', 'county', 'elevation',
        'timezone', 'sc_point_gid', 'sc_row_ind', 'sc_col_ind'
    }
    DROPPED_COLUMNS = ['gid']
    MERGE_COLUMN = 'sc_point_gid'
    PROFILE_DSET_REGEX = 'rep_profiles_[0-9]+$'
    OUTPUT_PROFILE_NAMES = ['hybrid', 'solar_time_built', 'wind_time_built']

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

        logger.info('Running hybridization rep profiles with solar_fpath: "{}"'
                    .format(solar_fpath))
        logger.info('Running hybridization rep profiles with solar_fpath: "{}"'
                    .format(wind_fpath))

        self._solar_fpath = solar_fpath
        self._wind_fpath = wind_fpath
        self._profiles = None
        self._solar_meta = None
        self._wind_meta = None
        self._hybrid_meta = None
        self._solar_time_index = None
        self._wind_time_index = None
        self._hybrid_time_index = None
        self.__profile_reg_check = re.compile(self.PROFILE_DSET_REGEX)
        self.__solar_cols = self.solar_meta.columns.map(ColNameFormatter.fmt)
        self.__wind_cols = self.wind_meta.columns.map(ColNameFormatter.fmt)
        self.__col_name_map = None

        self._validate_input_files()

    def _validate_input_files(self):
        """Validate the input files.
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
        for fp in [self._solar_fpath, self._wind_fpath]:
            with Resource(fp) as res:
                profile_dset_names = [
                    n for n in res.dsets
                    if self.__profile_reg_check.match(n)
                ]
                if len(profile_dset_names) > 1:
                    msg = ("Found more than one profile in {!r}: {}. "
                           "This module is not intended for hybridization of "
                           "multiple representative profiles. Please re-run "
                           "on a single aggregated profile.")
                    e = msg.format(fp, profile_dset_names)
                    logger.error(e)
                    raise FileInputError(e)

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

        if not (solar_vals & wind_vals):
            msg = ("No overlap detected in the values of {!r} across the "
                   "input files. Please ensure that at least one of the "
                   "{!r} values is the same for input files {!r} and {!r}")
            e = msg.format(merge_col, merge_col, self._solar_fpath,
                           self._wind_fpath)
            logger.error(e)
            raise FileInputError(e)

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
        return self._hybrid_meta

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

    @hybrid_col('hybrid_capacity')
    def aggregate_capacity(self):
        """Compute the total capcity by summing the individual capacities.

        Returns
        -------
        data : Series
            A series of data containing the aggregated capacity.
        """
        total_cap = (self.hybrid_meta['solar_capacity']
                     + self.hybrid_meta['wind_capacity'])
        return total_cap

    @hybrid_col('hybrid_cf')
    def aggregate_capacity_factor(self):
        """Compute the capacity-weighted mean capcity factor.

        Returns
        -------
        data : Series
            A series of data containing the aggregated capacity.
        """
        solar_cf_weighted = (self.hybrid_meta['solar_capacity']
                             * self.hybrid_meta['solar_mean_cf'])
        wind_cf_weighted = (self.hybrid_meta['wind_capacity']
                            * self.hybrid_meta['wind_mean_cf'])
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
        # self._init_profiles()

        # if max_workers == 1:
        #     self._run_serial()
        # else:
        #     self._run_parallel(max_workers=max_workers)

        # if fout is not None:
        #     self.save_profiles(fout, save_hybrid_meta=save_hybrid_meta,
        #                        scaled_precision=scaled_precision)

        # logger.info('Hybridization of representative profiles complete!')

    def _hybridize_meta(self):
        """Combine the solar and wind metas and run hybridize methods."""
        self._format_meta_pre_merge()
        self._hybrid_meta = self.solar_meta.merge(
            self.wind_meta, on=ColNameFormatter.fmt(self.MERGE_COLUMN),
            suffixes=[None, '_x'],  # how='outer'
        )
        self._verify_lat_long_match_post_merge()
        self._format_meta_post_merge()

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

        self._sort_hybrid_meta_cols()

    def _format_meta_pre_merge(self):
        """Prepare solar and wind meta for merging. """
        self.__col_name_map = {
            ColNameFormatter.fmt(c): c
            for c in self.solar_meta.columns.values
        }

        self._rename_cols(self.solar_meta, prefix='solar')
        self._rename_cols(self.wind_meta, prefix='wind')

    def _format_meta_post_merge(self):
        """Format hybrid meta after merging. """
        self._hybrid_meta.drop(
            [n for n in self._hybrid_meta.columns if "_x" in n]
            + self.DROPPED_COLUMNS,
            axis=1, inplace=True, errors='ignore'
        )
        self._hybrid_meta.rename(self.__col_name_map, inplace=True, axis=1)

    def _sort_hybrid_meta_cols(self):
        """Sort the columns of the hybrid meta. """
        self._hybrid_meta = self._hybrid_meta[
            sorted(self._hybrid_meta.columns, key=self._column_sorting_key)
        ]

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

    def _rename_cols(self, df, prefix):
        """Replace column names with the ColNameFormatter.fmt is needed. """
        df.columns = [
            ColNameFormatter.fmt(col_name)
            if col_name in self.NON_DUPLICATE_COLS
            else '{}_{}'.format(prefix, col_name)
            for col_name in df.columns.values
        ]

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
        """Verify that all the values in a column match post merge. """
        c1, c2 = col_name, '{}_x'.format(col_name)
        if c1 in self._hybrid_meta.columns and c2 in self._hybrid_meta.columns:
            return (self._hybrid_meta[c1] == self._hybrid_meta[c2]).all()
        else:
            return True

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

        meta = self.meta.copy()
        for c in meta.columns:
            try:
                meta[c] = pd.to_numeric(meta[c])
            except ValueError:
                pass

        Outputs.init_h5(fout, dsets, shapes, attrs, chunks, dtypes,
                        meta, time_index=self.time_index)

        if save_hybrid_meta:
            with Outputs(fout, mode='a') as out:
                hybrid_meta = to_records_array(self._hybrid_meta)
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

        self._init_h5_out(fout, save_hybrid_meta=save_hybrid_meta,
                          scaled_precision=scaled_precision)
        self._write_h5_out(fout, save_hybrid_meta=save_hybrid_meta)

    def _run_serial(self):
        """Compute all representative profiles in serial."""

        logger.info('Running {} rep profile calculations in serial.'
                    .format(len(self.meta)))
        for i, row in self.hybrid_meta.iterrows():
            logger.debug('Working on profile {} out of {}'
                         .format(i + 1, len(self.hybrid_meta)))
            # out = self._hybridize_profile(row)
            logger.info('Profile {} out of {} complete '
                        .format(i + 1, len(self.hybrid_meta)))

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
                    .format(len(self.meta)))

        iter_chunks = np.array_split(self.hybrid_meta.index.values,
                                     np.ceil(len(self.hybrid_meta)
                                             / pool_size))
        n_complete = 0
        for iter_chunk in iter_chunks:
            logger.debug('Starting process pool...')
            futures = {}
            loggers = [__name__, 'reV']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                for i in iter_chunk:
                    row = self.meta.loc[i, :]

                    future = exe.submit(
                        # self._hybridize_profile,
                        row
                    )

                    futures[future] = [i, region_dict]

                for future in as_completed(futures):
                    i, region_dict = futures[future]
                    out = future.result()
                    n_complete += 1
                    logger.info('Future {} out of {} complete '
                                .format(n_complete, len(self.meta)))
                    log_mem(logger, log_level='DEBUG')

    @classmethod
    def run(cls, solar_fpath, wind_fpath, fout=None, save_hybrid_meta=True,
            scaled_precision=False, max_workers=None):
        """Run hybridization by merging the profiles of each SC region.

        Parameters
        ----------
        solar_fpath : str
            Filepath to rep profile output file to extract solar profiles and
            summaries from.
        wind_fpath : str
            Filepath to rep profile output file to extract wind profiles and
            summaries from.
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

        rp = cls(solar_fpath, wind_fpath)

        rp._run(fout=fout, save_hybrid_meta=save_hybrid_meta,
                scaled_precision=scaled_precision, max_workers=max_workers)

        # return rp._profiles, rp.hybrid_meta, rp.hybrid_time_index
