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
from warnings import warn


from reV.handlers.outputs import Outputs
from reV.utilities.exceptions import FileInputError, DataShapeError
from reV.utilities import log_versions

from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem
from rex.utilities.utilities import parse_year, to_records_array

logger = logging.getLogger(__name__)


class Hybridization:
    """Framework to handle hybridization for one resource region"""

    NON_DUPLICATE_COLS = set(
        ['latitude', 'longitude', 'country', 'state', 'county', 'elevation',
         'timezone', 'sc_point_gid', 'sc_row_ind', 'sc_col_ind']
    )
    DROPPED_COLUMNS = set(['gid'])
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

        self._validate_input_files()
        # self._hybridize_summary()
        # self._init_profiles()

    def _hybridize_summary(self):
        with Resource(self._solar_fpath) as res:
            solar_meta = res.meta

        with Resource(self._wind_fpath) as res:
            wind_meta = res.meta

        col_name_map = {
            c.lower().replace(" ", "_"): c
            for c in solar_meta.columns.values
        }

        solar_meta.columns = [
            col_name.lower().replace(" ", "_")
            if col_name.lower().replace(" ", "_") in self.NON_DUPLICATE_COLS
            else '{}_solar'.format(col_name)
            for col_name in solar_meta.columns.values
        ]

        wind_meta.columns = [
            col_name.lower().replace(" ", "_")
            if col_name.lower().replace(" ", "_") in self.NON_DUPLICATE_COLS
            else '{}_wind'.format(col_name)
            for col_name in wind_meta.columns.values
        ]

        sc = set(solar_meta.columns.values)
        wc = set(wind_meta.columns.values)
        duplicate_cols = sc & wc

        if self.MERGE_COLUMN.lower().replace(" ", "_") not in duplicate_cols:
            msg = "Merge column {} missing from one or both summaries!"
            raise ValueError(msg.format(self.MERGE_COLUMN))

        self._hybrid_meta = solar_meta.merge(
            wind_meta, on=self.MERGE_COLUMN.lower().replace(" ", "_"),
            suffixes=[None, '_x']
        )
        self._hybrid_meta.drop(
            [n for n in self._hybrid_meta.columns if "_x" in n],
            axis=1, inplace=True
        )
        self._hybrid_meta.rename(col_name_map, inplace=True, axis=1)
        self._hybrid_meta.to_csv("combined.csv")

    def _init_profiles(self):
        """Initialize the output rep profiles attribute."""
        self._profiles = {
            k: np.zeros((len(self.hybrid_time_index), len(self._hybrid_meta)),
                        dtype=np.float32)
            for k in self.OUTPUT_PROFILE_NAMES}

    def _validate_input_files(self):
        """Validate the input files.
        """
        self._validate_time_index()
        self._validate_num_profiles()

    def _validate_time_index(self):
        """Validate the hybrid time index to be of len >= 8760.

        Raises
        ------
        ValueError
            If len(time_index) < 8760 for the hybrid profile.
        """
        if len(self.hybrid_time_index) < 8760:
            msg = ("The length of the merged time index ({}) is less than "
                   "8760. Please ensure that the input profiles have a "
                   "time index that overlaps >= 8760 times.")
            raise ValueError(msg.format(len(self.hybrid_time_index)))

    def _validate_num_profiles(self):
        """Validate the number of input profiles.

        Raises
        ------
        ValueError
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
                    raise ValueError(msg.format(fp, profile_dset_names))

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
        """Get the arrays of representative CF profiles corresponding to meta.

        Returns
        -------
        profiles : dict
            dict of n_profile-keyed arrays with shape (time, n) for the
            representative profiles for each region.
        """
        return self._profiles

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
        if max_workers == 1:
            self._run_serial()
        else:
            self._run_parallel(max_workers=max_workers)

        if fout is not None:
            self.save_profiles(fout, save_hybrid_meta=save_hybrid_meta,
                               scaled_precision=scaled_precision)

        logger.info('Hybridization of representative profiles complete!')

    @classmethod
    def run(cls, gen_fpath, fout=None, save_hybrid_meta=True,
            scaled_precision=False, max_workers=None):
        """Run hybridization by merging the profiles of each SC region.

        Parameters
        ----------
        gen_fpath : str
            Filepath to reV gen output file to extract "cf_profile" from.
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
        profiles : dict
            dict of n_profile-keyed arrays with shape (time, n) for the
            hybridized profiles for each region.
        hybrid_meta : pd.DataFrame
            Meta dataframes recording the regions and the selected rep profile
            gid.
        hybrid_time_index : pd.DatatimeIndex
            Datetime Index for hybridized profiles
        """

        rp = cls(gen_fpath)

        rp._run(fout=fout, save_hybrid_meta=save_hybrid_meta,
                scaled_precision=scaled_precision, max_workers=max_workers)

        return rp._profiles, rp.hybrid_meta, rp.hybrid_time_index
