# -*- coding: utf-8 -*-
"""Representative profile extraction utilities.

Created on Thu Oct 31 12:49:23 2019

@author: gbuster
"""
from abc import ABC, abstractmethod
from concurrent.futures import as_completed
from copy import deepcopy
import json
import logging
import numpy as np
import os
import pandas as pd
from scipy import stats
from warnings import warn


from reV.handlers.outputs import Outputs
from reV.utilities.exceptions import FileInputError, DataShapeError

from rex.rechunk_h5 import to_records_array
from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.utilities import parse_year
from rex.utilities.loggers import log_mem

logger = logging.getLogger(__name__)


class RepresentativeMethods:
    """Class for organizing the methods to determine representative-ness"""

    def __init__(self, profiles, weights=None, rep_method='meanoid',
                 err_method='rmse'):
        """
        Parameters
        ----------
        profiles : np.ndarray
            (time, sites) timeseries array of cf profile data.
        weights : np.ndarray | list
            1D array of weighting factors (multiplicative) for profiles.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        """
        self._rep_method = self.rep_methods[rep_method]
        self._err_method = self.err_methods[err_method]
        self._profiles = profiles
        self._weights = weights
        self._parse_weights()

    def _parse_weights(self):
        """Parse the weights attribute. Check shape and make np.array."""
        if isinstance(self._weights, (list, tuple)):
            self._weights = np.array(self._weights)

        if self._weights is not None:
            emsg = ('Weighting factors array of length {} does not match '
                    'profiles of shape {}'
                    .format(len(self._weights), self._profiles.shape[1]))
            assert len(self._weights) == self._profiles.shape[1], emsg

    @property
    def rep_methods(self):
        """Lookup table of representative methods"""
        methods = {'mean': self.meanoid,
                   'meanoid': self.meanoid,
                   'median': self.medianoid,
                   'medianoid': self.medianoid,
                   }

        return methods

    @property
    def err_methods(self):
        """Lookup table of error methods"""
        methods = {'mbe': self.mbe,
                   'mae': self.mae,
                   'rmse': self.rmse,
                   None: None,
                   }

        return methods

    @staticmethod
    def nargmin(arr, n):
        """Get the index of the Nth min value in arr.

        Parameters
        ----------
        arr : np.ndarray
            1D array.
        n : int
            If n is 0, this returns the location of the min value in arr.
            If n is 1, this returns the location of the 2nd min value in arr.

        Returns
        -------
        i : int
            Location of the Nth min value in arr.
        """
        return arr.argsort()[:(n + 1)][-1]

    @staticmethod
    def meanoid(profiles, weights=None):
        """Find the mean profile across all sites.

        Parameters
        ----------
        profiles : np.ndarray
            (time, sites) timeseries array of cf profile data.
        weights : np.ndarray | list
            1D array of weighting factors (multiplicative) for profiles.

        Returns
        -------
        arr : np.ndarray
            (time, 1) timeseries of the mean of all cf profiles across sites.
        """
        if weights is None:
            arr = profiles.mean(axis=1).reshape((len(profiles), 1))
        else:
            if not isinstance(weights, np.ndarray):
                weights = np.array(weights)

            arr = (profiles * weights).sum(axis=1) / weights.sum()
            if len(arr.shape) == 1:
                arr = np.expand_dims(arr, axis=1)

        return arr

    @staticmethod
    def medianoid(profiles):
        """Find the median profile across all sites.

        Parameters
        ----------
        profiles : np.ndarray
            (time, sites) timeseries array of cf profile data.

        Returns
        -------
        arr : np.ndarray
            (time, 1) timeseries of the median at every timestep of all
            cf profiles across sites.
        """
        arr = np.median(profiles, axis=1)
        arr = arr.reshape((len(profiles), 1))

        return arr

    @classmethod
    def mbe(cls, profiles, baseline, i_profile=0):
        """Calculate the mean bias error of profiles vs. a baseline profile.

        Parameters
        ----------
        profiles : np.ndarray
            (time, sites) timeseries array of cf profile data.
        baseline : np.ndarray
            (time, 1) timeseries of the meanoid or medianoid to which
            cf profiles should be compared.
        i_profile : int
            The index of the represntative profile being saved
            (for n_profiles). 0 is the most representative profile.

        Returns
        -------
        profile : np.ndarray
            (time, 1) array for the most representative profile
        i_rep : int
            Column Index in profiles of the representative profile.
        """
        diff = profiles - baseline.reshape((len(baseline), 1))
        mbe = diff.mean(axis=0)
        i_rep = cls.nargmin(mbe, i_profile)

        return profiles[:, i_rep], i_rep

    @classmethod
    def mae(cls, profiles, baseline, i_profile=0):
        """Calculate the mean absolute error of profiles vs. a baseline profile

        Parameters
        ----------
        profiles : np.ndarray
            (time, sites) timeseries array of cf profile data.
        baseline : np.ndarray
            (time, 1) timeseries of the meanoid or medianoid to which
            cf profiles should be compared.
        i_profile : int
            The index of the represntative profile being saved
            (for n_profiles). 0 is the most representative profile.

        Returns
        -------
        profile : np.ndarray
            (time, 1) array for the most representative profile
        i_rep : int
            Column Index in profiles of the representative profile.
        """
        diff = profiles - baseline.reshape((len(baseline), 1))
        mae = np.abs(diff).mean(axis=0)
        i_rep = cls.nargmin(mae, i_profile)

        return profiles[:, i_rep], i_rep

    @classmethod
    def rmse(cls, profiles, baseline, i_profile=0):
        """Calculate the RMSE of profiles vs. a baseline profile

        Parameters
        ----------
        profiles : np.ndarray
            (time, sites) timeseries array of cf profile data.
        baseline : np.ndarray
            (time, 1) timeseries of the meanoid or medianoid to which
            cf profiles should be compared.
        i_profile : int
            The index of the represntative profile being saved
            (for n_profiles). 0 is the most representative profile.

        Returns
        -------
        profile : np.ndarray
            (time, 1) array for the most representative profile
        i_rep : int
            Column Index in profiles of the representative profile.
        """
        rmse = profiles - baseline.reshape((len(baseline), 1))
        rmse **= 2
        rmse = np.sqrt(np.mean(rmse, axis=0))
        i_rep = cls.nargmin(rmse, i_profile)

        return profiles[:, i_rep], i_rep

    @classmethod
    def run(cls, profiles, weights=None, rep_method='meanoid',
            err_method='rmse', n_profiles=1):
        """Run representative profile methods.

        Parameters
        ----------
        profiles : np.ndarray
            (time, sites) timeseries array of cf profile data.
        weights : np.ndarray | list
            1D array of weighting factors (multiplicative) for profiles.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        n_profiles : int
            Number of representative profiles to save to fout.

        Returns
        -------
        profiles : np.ndarray
            (time, n_profiles) array for the most representative profile(s)
        i_reps : list
            List (length of n_profiles) with column Index in profiles of the
            representative profile(s).
        """
        inst = cls(profiles, weights=weights, rep_method=rep_method,
                   err_method=err_method)

        if inst._weights is not None:
            baseline = inst._rep_method(inst._profiles, weights=inst._weights)
        else:
            baseline = inst._rep_method(inst._profiles)

        if err_method is None:
            profiles = baseline
            i_reps = [None]

        else:
            profiles = None
            i_reps = []
            for i in range(n_profiles):
                p, ir = inst._err_method(inst._profiles, baseline, i_profile=i)
                if profiles is None:
                    profiles = np.zeros((len(p), n_profiles), dtype=p.dtype)

                profiles[:, i] = p
                i_reps.append(ir)

        return profiles, i_reps


class RegionRepProfile:
    """Framework to handle rep profile for one resource region"""

    def __init__(self, gen_fpath, rev_summary, gid_col='gen_gids',
                 cf_dset='cf_profile', rep_method='meanoid', err_method='rmse',
                 weight='gid_counts', n_profiles=1):
        """
        Parameters
        ----------
        gen_fpath : str
            Filepath to reV gen output file to extract "cf_profile" from.
        rev_summary : pd.DataFrame
            Aggregated rev supply curve summary file trimmed to just one
            region to get a rep profile for.
        gid_col : str
            Column label in rev_summary that contains the generation gids
            (data index in gen_fpath).
        cf_dset : str
            Dataset name to pull generation profiles from.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        weight : str | None
            Column in rev_summary used to apply weighted mean to profiles.
            The supply curve table data in the weight column should have
            weight values corresponding to the gid_col in the same row.
        n_profiles : int
            Number of representative profiles to retrieve.
        """

        self._gen_fpath = gen_fpath
        self._rev_summary = rev_summary
        self._gid_col = gid_col
        self._cf_dset = cf_dset
        self._profiles = None
        self._source_profiles = None
        self._weights = None
        self._i_reps = None
        self._rep_method = rep_method
        self._err_method = err_method
        self._weight = weight
        self._n_profiles = n_profiles

    @property
    def source_profiles(self):
        """Retrieve the cf profile array from the source generation h5 file.

        Returns
        -------
        profiles : np.ndarray
            Timeseries array of cf profile data.
        """
        if self._source_profiles is None:
            gen_gids = self._get_region_attr(self._rev_summary, self._gid_col)
            with Resource(self._gen_fpath) as res:
                self._source_profiles = res[self._cf_dset, :, gen_gids]

        return self._source_profiles

    @property
    def weights(self):
        """Get the weights array

        Returns
        -------
        weights : np.ndarray | None
            Flat array of weight values from the weight column. The supply
            curve table data in the weight column should have a list of weight
            values corresponding to the gen_gids list in the same row.
        """

        if self._weights is None:
            if self._weight is None:
                self._weights = None
            else:
                self._weights = self._get_region_attr(self._rev_summary,
                                                      self._weight)
                self._weights = np.array(self._weights)

        return self._weights

    @staticmethod
    def _get_region_attr(rev_summary, attr_name):
        """Retrieve a flat list of attribute data from a col in rev summary.

        Parameters
        ----------
        rev_summary : pd.DataFrame
            Aggregated rev supply curve summary file trimmed to just one
            region to get a rep profile for.
        attr_name : str
            Column label to extract flattened data from (gen_gids,
            gid_counts, etc...)

        Returns
        -------
        data : list
            Flat list of data from the column with label "attr_name".
            Either a list of numbers or strings. Lists of jsonified lists
            will be unpacked.
        """
        data = rev_summary[attr_name].values.tolist()

        if any(data):
            if isinstance(data[0], str):
                if ('[' and ']' in data[0]) or ('(' and ')' in data[0]):
                    data = [json.loads(s) for s in data]

            if isinstance(data[0], (list, tuple)):
                data = [a for b in data for a in b]

        return data

    def _run_rep_methods(self):
        """Run the representative profile methods to find the meanoid/medianoid
        profile and find the profiles most similar."""

        if self.weights is not None:
            if len(self.weights) != self.source_profiles.shape[1]:
                e = ('Weights column "{}" resulted in {} weight scalars '
                     'which doesnt match gid column "{}" which yields '
                     'profiles with shape {}.'
                     .format(self._weight, len(self.weights),
                             self._gid_col, self.source_profiles.shape))
                gen_gids = self._get_region_attr(self._rev_summary,
                                                 self._gid_col)
                logger.debug('Gids from column "{}" with len {}: {}'
                             .format(self._gid_col, len(gen_gids), gen_gids))
                logger.debug('Weights from column "{}" with len {}: {}'
                             .format(self._weight, len(self.weights),
                                     self.weights))
                logger.error(e)
                raise DataShapeError(e)

        self._profiles, self._i_reps = RepresentativeMethods.run(
            self.source_profiles, weights=self.weights,
            rep_method=self._rep_method, err_method=self._err_method,
            n_profiles=self._n_profiles)

    @property
    def rep_profiles(self):
        """Get the representative profiles of this region."""
        if self._profiles is None:
            self._run_rep_methods()

        return self._profiles

    @property
    def i_reps(self):
        """Get the representative profile index(es) of this region."""
        if self._i_reps is None:
            self._run_rep_methods()

        return self._i_reps

    @property
    def rep_gen_gids(self):
        """Get the representative profile gen gids of this region."""
        gids = self._get_region_attr(self._rev_summary, self._gid_col)
        if self.i_reps[0] is None:
            rep_gids = None
        else:
            rep_gids = [gids[i] for i in self.i_reps]

        return rep_gids

    @property
    def rep_res_gids(self):
        """Get the representative profile resource gids of this region."""
        if self._gid_col == 'gen_gids':
            gids = self._get_region_attr(self._rev_summary, 'res_gids')
        else:
            gids = None

        if self.i_reps[0] is None or gids is None:
            rep_gids = [None]
        else:
            rep_gids = [gids[i] for i in self.i_reps]

        return rep_gids

    @classmethod
    def get_region_rep_profile(cls, gen_fpath, rev_summary, gid_col='gen_gids',
                               cf_dset='cf_profile', rep_method='meanoid',
                               err_method='rmse', weight='gid_counts',
                               n_profiles=1):
        """Class method for parallelization of rep profile calc.

        Parameters
        ----------
        gen_fpath : str
            Filepath to reV gen output file to extract "cf_profile" from.
        rev_summary : pd.DataFrame
            Aggregated rev supply curve summary file trimmed to just one
            region to get a rep profile for.
        gid_col : str
            Column label in rev_summary that contains the generation gids
            (data index in gen_fpath).
        cf_dset : str
            Dataset name to pull generation profiles from.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        weight : str | None
            Column in rev_summary used to apply weighted mean to profiles.
            The supply curve table data in the weight column should have
            weight values corresponding to the gid_col in the same row.
        n_profiles : int
            Number of representative profiles to retrieve.

        Returns
        -------
        rep_profile : np.ndarray
            (time, n_profiles) array for the most representative profile(s)
        i_rep : list
            Column Index in profiles of the representative profile(s).
        gen_gid_reps : list
            Generation gid(s) of the representative profile(s).
        res_gid_reps : list
            Resource gid(s) of the representative profile(s).
        """
        r = cls(gen_fpath, rev_summary, gid_col=gid_col, cf_dset=cf_dset,
                rep_method=rep_method, err_method=err_method, weight=weight,
                n_profiles=n_profiles)

        return r.rep_profiles, r.i_reps, r.rep_gen_gids, r.rep_res_gids


class RepProfilesBase(ABC):
    """Abstract utility framework for representative profile run classes."""

    def __init__(self, gen_fpath, rev_summary, reg_cols=None,
                 gid_col='gen_gids', cf_dset='cf_profile',
                 rep_method='meanoid', err_method='rmse',
                 weight='gid_counts', n_profiles=1):
        """
        Parameters
        ----------
        gen_fpath : str
            Filepath to reV gen output file to extract "cf_profile" from.
        rev_summary : str | pd.DataFrame
            Aggregated rev supply curve summary file. Str filepath or full df.
        reg_cols : str | list | None
            Label(s) for a categorical region column(s) to extract profiles
            for. e.g. "state" will extract a rep profile for each unique entry
            in the "state" column in rev_summary.
        gid_col : str
            Column label in rev_summary that contains the generation gids
            (data index in gen_fpath).
        cf_dset : str
            Dataset name to pull generation profiles from.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        weight : str | None
            Column in rev_summary used to apply weighted mean to profiles.
            The supply curve table data in the weight column should have
            weight values corresponding to the gid_col in the same row.
        n_profiles : int
            Number of representative profiles to save to fout.
        """

        logger.info('Running rep profiles with gen_fpath: "{}"'
                    .format(gen_fpath))
        logger.info('Running rep profiles with rev_summary: "{}"'
                    .format(rev_summary))
        logger.info('Running rep profiles with gid column: "{}"'
                    .format(gid_col))
        logger.info('Running rep profiles with region columns: "{}"'
                    .format(reg_cols))
        logger.info('Running rep profiles with representative method: "{}"'
                    .format(rep_method))
        logger.info('Running rep profiles with error method: "{}"'
                    .format(err_method))
        logger.info('Running rep profiles with weight factor: "{}"'
                    .format(weight))

        self._weight = weight
        self._gid_col = gid_col
        self._n_profiles = n_profiles
        self._cf_dset = cf_dset
        self._gen_fpath = gen_fpath
        self._reg_cols = reg_cols
        self._regions = None

        self._rev_summary = self._parse_rev_summary(rev_summary)

        self._check_req_cols(self._rev_summary, self._reg_cols)
        self._check_req_cols(self._rev_summary, self._weight)
        self._check_req_cols(self._rev_summary, self._gid_col)

        self._check_rev_gen(gen_fpath, cf_dset, self._rev_summary)

        if self._reg_cols is not None:
            self._regions = {k: self._rev_summary[k].unique().tolist()
                             for k in self._reg_cols}
        self._time_index = None
        self._meta = None
        self._profiles = None
        self._rep_method = rep_method
        self._err_method = err_method

    @staticmethod
    def _parse_rev_summary(rev_summary):
        """Extract, parse, and check the rev summary table.

        Parameters
        ----------
        rev_summary : str | pd.DataFrame
            Aggregated rev supply curve summary file. Str filepath or full df.

        Returns
        -------
        rev_summary : pd.DataFrame
            Aggregated rev supply curve summary file.
        """

        if isinstance(rev_summary, str):
            if os.path.exists(rev_summary) and rev_summary.endswith('.csv'):
                rev_summary = pd.read_csv(rev_summary)
            elif os.path.exists(rev_summary) and rev_summary.endswith('.json'):
                rev_summary = pd.read_json(rev_summary)
            else:
                e = 'Could not parse reV summary file: {}'.format(rev_summary)
                logger.error(e)
                raise FileInputError(e)
        elif not isinstance(rev_summary, pd.DataFrame):
            e = ('Bad input dtype for rev_summary input: {}'
                 .format(type(rev_summary)))
            logger.error(e)
            raise TypeError(e)

        return rev_summary

    @staticmethod
    def _check_req_cols(df, cols):
        """Check a dataframe for required columns.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to check columns.
        cols : str | list | tuple
            Required columns in df.
        """
        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]

            missing = []
            for c in cols:
                if c not in df:
                    missing.append(c)

            if any(missing):
                e = ('Column labels not found in rev_summary table: {}'
                     .format(missing))
                logger.error(e)
                raise KeyError(e)

    @staticmethod
    def _check_rev_gen(gen_fpath, cf_dset, rev_summary):
        """Check rev gen file for requisite datasets.

        Parameters
        ----------
        gen_fpath : str
            Filepath to reV gen output file to extract "cf_profile" from.
        cf_dset : str
            Dataset name to pull generation profiles from.
        rev_summary : pd.DataFrame
            Aggregated rev supply curve summary file.
        """
        with Resource(gen_fpath) as res:
            dsets = res.datasets
            if cf_dset not in dsets:
                raise KeyError('reV gen file needs to have "{}" '
                               'dataset to calculate representative profiles!'
                               .format(cf_dset))

            if 'time_index' not in str(dsets):
                raise KeyError('reV gen file needs to have "time_index" '
                               'dataset to calculate representative profiles!')

            shape = res.get_dset_properties(cf_dset)[0]

        if len(rev_summary) > shape[1]:
            msg = ('WARNING: reV SC summary table has {} sc points and CF '
                   'dataset "{}" has {} profiles. There should never be more '
                   'SC points than CF profiles.'
                   .format(len(rev_summary), cf_dset, shape[1]))
            logger.warning(msg)
            warn(msg)

    def _init_profiles(self):
        """Initialize the output rep profiles attribute."""
        self._profiles = {k: np.zeros((len(self.time_index),
                                       len(self.meta)),
                                      dtype=np.float32)
                          for k in range(self._n_profiles)}

    @property
    def time_index(self):
        """Get the time index for the rep profiles.

        Returns
        -------
        time_index : pd.datetimeindex
            Time index sourced from the reV gen file.
        """
        if self._time_index is None:
            with Resource(self._gen_fpath) as res:
                ds = 'time_index'
                if parse_year(self._cf_dset, option='bool'):
                    year = parse_year(self._cf_dset, option='raise')
                    ds += '-{}'.format(year)

                self._time_index = res._get_time_index(ds, slice(None))

        return self._time_index

    @property
    def meta(self):
        """Meta data for the representative profiles.

        Returns
        -------
        meta : pd.DataFrame
            Meta data for the representative profiles. At the very least,
            this has columns for the region and res class.
        """
        return self._meta

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

    def _init_h5_out(self, fout, save_rev_summary=True,
                     scaled_precision=False):
        """Initialize an output h5 file for n_profiles

        Parameters
        ----------
        fout : str
            None or filepath to output h5 file.
        save_rev_summary : bool
            Flag to save full reV SC table to rep profile output.
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

        if save_rev_summary:
            with Outputs(fout, mode='a') as out:
                rev_sum = to_records_array(self._rev_summary)
                out._create_dset('rev_summary', rev_sum.shape,
                                 rev_sum.dtype, data=rev_sum)

    def _write_h5_out(self, fout, save_rev_summary=True):
        """Write profiles and meta to an output file.

        Parameters
        ----------
        fout : str
            None or filepath to output h5 file.
        save_rev_summary : bool
            Flag to save full reV SC table to rep profile output.
        scaled_precision : bool
            Flag to scale cf_profiles by 1000 and save as uint16.
        """
        with Outputs(fout, mode='a') as out:

            if 'rev_summary' in out.datasets and save_rev_summary:
                rev_sum = to_records_array(self._rev_summary)
                out['rev_summary'] = rev_sum

            for i in range(self._n_profiles):
                dset = 'rep_profiles_{}'.format(i)
                out[dset] = self.profiles[i]

    def save_profiles(self, fout, save_rev_summary=True,
                      scaled_precision=False):
        """Initialize fout and save profiles.

        Parameters
        ----------
        fout : str
            None or filepath to output h5 file.
        save_rev_summary : bool
            Flag to save full reV SC table to rep profile output.
        scaled_precision : bool
            Flag to scale cf_profiles by 1000 and save as uint16.
        """

        self._init_h5_out(fout, save_rev_summary=save_rev_summary,
                          scaled_precision=scaled_precision)
        self._write_h5_out(fout, save_rev_summary=save_rev_summary)

    @abstractmethod
    def _run_serial(self):
        """Abstract method for serial run method."""

    @abstractmethod
    def _run_parallel(self):
        """Abstract method for parallel run method."""

    @abstractmethod
    def run(self):
        """Abstract method for generic run method."""


class RepProfiles(RepProfilesBase):
    """Framework for calculating the representative profiles based on an
    error metric vs. a mean or median profile.
    """

    def __init__(self, gen_fpath, rev_summary, reg_cols, gid_col='gen_gids',
                 cf_dset='cf_profile', rep_method='meanoid',
                 err_method='rmse', weight='gid_counts',
                 n_profiles=1):
        """
        Parameters
        ----------
        gen_fpath : str
            Filepath to reV gen output file to extract "cf_profile" from.
        rev_summary : str | pd.DataFrame
            Aggregated rev supply curve summary file. Str filepath or full df.
        reg_cols : str | list | None
            Label(s) for a categorical region column(s) to extract profiles
            for. e.g. "state" will extract a rep profile for each unique entry
            in the "state" column in rev_summary.
        gid_col : str
            Column label in rev_summary that contains the generation gids
            (data index in gen_fpath).
        cf_dset : str
            Dataset name to pull generation profiles from.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        weight : str | None
            Column in rev_summary used to apply weighted mean to profiles.
            The supply curve table data in the weight column should have
            weight values corresponding to the gid_col in the same row.
        n_profiles : int
            Number of representative profiles to save to fout.
        """

        logger.info('Finding representative profiles that are most similar '
                    'to the weighted meanoid for each supply curve region.')

        if reg_cols is None:
            e = ('Need to define "reg_cols"! If you want a profile for each '
                 'supply curve point, try setting "reg_cols" to a primary '
                 'key such as "sc_gid".')
            logger.error(e)
            raise ValueError(e)
        elif isinstance(reg_cols, str):
            reg_cols = [reg_cols]
        elif not isinstance(reg_cols, list):
            reg_cols = list(reg_cols)

        super().__init__(gen_fpath, rev_summary, reg_cols=reg_cols,
                         gid_col=gid_col, cf_dset=cf_dset,
                         rep_method=rep_method, err_method=err_method,
                         weight=weight, n_profiles=n_profiles)

        self._set_meta()
        self._init_profiles()

    def _set_meta(self):
        """Set the rep profile meta data with each row being a unique
        combination of the region columns."""
        self._meta = self._rev_summary.groupby(self._reg_cols)
        self._meta = \
            self._meta['timezone'].apply(lambda x: stats.mode(x).mode[0])
        self._meta = self._meta.reset_index()

        self._meta['rep_gen_gid'] = None
        self._meta['rep_res_gid'] = None

    def _get_mask(self, region_dict):
        """Get the mask for a given region and res class.

        Parameters
        ----------
        region_dict : dict
            Column-value pairs to filter the rev summary on.

        Returns
        -------
        mask : np.ndarray
            Boolean mask to filter rev_summary to the appropriate
            region_dict values.
        """
        mask = None
        for k, v in region_dict.items():
            temp = (self._rev_summary[k] == v)
            if mask is None:
                mask = temp
            else:
                mask = (mask & temp)

        return mask

    def _run_serial(self):
        """Compute all representative profiles in serial."""

        logger.info('Running {} rep profile calculations in serial.'
                    .format(len(self.meta)))
        meta_static = deepcopy(self.meta)
        for i, row in meta_static.iterrows():
            region_dict = {k: v for (k, v) in row.to_dict().items()
                           if k in self._reg_cols}
            mask = self._get_mask(region_dict)

            if not any(mask):
                logger.warning('Skipping profile {} out of {} '
                               'for region: {} with no valid mask.'
                               .format(i + 1, len(meta_static), region_dict))
            else:
                logger.debug('Working on profile {} out of {} for region: {}'
                             .format(i + 1, len(meta_static), region_dict))
                out = RegionRepProfile.get_region_rep_profile(
                    self._gen_fpath, self._rev_summary[mask],
                    gid_col=self._gid_col, cf_dset=self._cf_dset,
                    rep_method=self._rep_method, err_method=self._err_method,
                    weight=self._weight, n_profiles=self._n_profiles)
                profiles, _, ggids, rgids = out
                logger.info('Profile {} out of {} complete '
                            'for region: {}'
                            .format(i + 1, len(meta_static), region_dict))

                for n in range(profiles.shape[1]):
                    self._profiles[n][:, i] = profiles[:, n]

                    if len(ggids) == 1:
                        self._meta.at[i, 'rep_gen_gid'] = ggids[0]
                        self._meta.at[i, 'rep_res_gid'] = rgids[0]
                    else:
                        self._meta.at[i, 'rep_gen_gid'] = str(ggids)
                        self._meta.at[i, 'rep_res_gid'] = str(rgids)

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

        iter_chunks = np.array_split(self.meta.index.values,
                                     np.ceil(len(self.meta) / pool_size))
        n_complete = 0
        for iter_chunk in iter_chunks:
            logger.debug('Starting process pool...')
            futures = {}
            loggers = [__name__, 'reV']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                for i in iter_chunk:
                    row = self.meta.loc[i, :]
                    region_dict = {k: v for (k, v) in row.to_dict().items()
                                   if k in self._reg_cols}

                    mask = self._get_mask(region_dict)

                    if not any(mask):
                        logger.info('Skipping profile {} out of {} '
                                    'for region: {} with no valid mask.'
                                    .format(i + 1, len(self.meta),
                                            region_dict))
                    else:
                        future = exe.submit(
                            RegionRepProfile.get_region_rep_profile,
                            self._gen_fpath, self._rev_summary[mask],
                            gid_col=self._gid_col,
                            cf_dset=self._cf_dset,
                            rep_method=self._rep_method,
                            err_method=self._err_method,
                            weight=self._weight,
                            n_profiles=self._n_profiles)

                        futures[future] = [i, region_dict]

                for future in as_completed(futures):
                    i, region_dict = futures[future]
                    profiles, _, ggids, rgids = future.result()
                    n_complete += 1
                    logger.info('Future {} out of {} complete '
                                'for region: {}'
                                .format(n_complete, len(self.meta),
                                        region_dict))
                    log_mem(logger, log_level='DEBUG')

                    for n in range(profiles.shape[1]):
                        self._profiles[n][:, i] = profiles[:, n]

                    if len(ggids) == 1:
                        self._meta.at[i, 'rep_gen_gid'] = ggids[0]
                        self._meta.at[i, 'rep_res_gid'] = rgids[0]
                    else:
                        self._meta.at[i, 'rep_gen_gid'] = str(ggids)
                        self._meta.at[i, 'rep_res_gid'] = str(rgids)

    def _run(self, fout=None, save_rev_summary=True, scaled_precision=False,
             max_workers=None):
        """
        Run representative profiles in serial or parallel and save to disc

        Parameters
        ----------
        fout : str, optional
            filepath to output h5 file, by default None
        save_rev_summary : bool, optional
            Flag to save full reV SC table to rep profile output.,
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
            self.save_profiles(fout, save_rev_summary=save_rev_summary,
                               scaled_precision=scaled_precision)

        logger.info('Representative profiles complete!')

    @classmethod
    def run(cls, gen_fpath, rev_summary, reg_cols, gid_col='gen_gids',
            cf_dset='cf_profile', rep_method='meanoid', err_method='rmse',
            weight='gid_counts', n_profiles=1, fout=None,
            save_rev_summary=True, scaled_precision=False, max_workers=None):
        """Run representative profiles by finding the closest single profile
        to the weighted meanoid for each SC region.

        Parameters
        ----------
        gen_fpath : str
            Filepath to reV gen output file to extract "cf_profile" from.
        rev_summary : str | pd.DataFrame
            Aggregated rev supply curve summary file. Str filepath or full df.
        reg_cols : str | list | None
            Label(s) for a categorical region column(s) to extract profiles
            for. e.g. "state" will extract a rep profile for each unique entry
            in the "state" column in rev_summary.
        gid_col : str
            Column label in rev_summary that contains the generation gids
            (data index in gen_fpath).
        cf_dset : str
            Dataset name to pull generation profiles from.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        weight : str | None
            Column in rev_summary used to apply weighted mean to profiles.
            The supply curve table data in the weight column should have
            weight values corresponding to the gid_col in the same row.
        n_profiles : int
            Number of representative profiles to save to fout.
        fout : str, optional
            filepath to output h5 file, by default None
        save_rev_summary : bool, optional
            Flag to save full reV SC table to rep profile output.,
            by default True
        scaled_precision : bool, optional
            Flag to scale cf_profiles by 1000 and save as uint16.,
            by default False
        max_workers : int, optional
            Number of parallel workers. 1 will run serial, None will use all
            available., by default None

        Returns
        -------
        profiles : dict
            dict of n_profile-keyed arrays with shape (time, n) for the
            representative profiles for each region.
        meta : pd.DataFrame
            Meta dataframes recording the regions and the selected rep profile
            gid.
        time_index : pd.DatatimeIndex
            Datetime Index for represntative profiles
        """

        rp = cls(gen_fpath, rev_summary, reg_cols, gid_col=gid_col,
                 cf_dset=cf_dset, rep_method=rep_method, err_method=err_method,
                 n_profiles=n_profiles, weight=weight)

        rp._run(fout=fout, save_rev_summary=save_rev_summary,
                scaled_precision=scaled_precision, max_workers=max_workers)

        return rp._profiles, rp._meta, rp._time_index


class AggregatedRepProfiles(RepProfilesBase):
    """Framework for calculating the aggregate representative supply curve
    cf profiles based on an area-weighted aggregation of contributing profiles.
    """

    def __init__(self, gen_fpath, rev_summary, gid_col='gen_gids',
                 cf_dset='cf_profile', weight='gid_counts'):
        """
        Parameters
        ----------
        gen_fpath : str
            Filepath to reV gen output file to extract "cf_profile" from.
        rev_summary : str | pd.DataFrame
            Aggregated rev supply curve summary file. Str filepath or full df.
        gid_col : str
            Column label in rev_summary that contains the generation gids
            (data index in gen_fpath).
        cf_dset : str
            Dataset name to pull generation profiles from.
        weight : str | None
            Column in rev_summary used to apply weighted mean to profiles.
            The supply curve table data in the weight column should have
            weight values corresponding to the gid_col in the same row.
        """

        logger.info('Calculating the weighted aggregate (meanoid) '
                    'representative profiles for each supply curve point.')

        super().__init__(gen_fpath, rev_summary, reg_cols=None,
                         gid_col=gid_col, cf_dset=cf_dset,
                         rep_method='meanoid', err_method=None,
                         weight=weight, n_profiles=1)

        self._meta = self._rev_summary
        self._init_profiles()

    def _run_serial(self):
        """Compute all representative profiles in serial."""

        logger.info('Running {} aggregate rep profile calculations in serial.'
                    .format(len(self.meta)))
        meta_static = deepcopy(self.meta)
        for i, row in meta_static.iterrows():
            row = pd.DataFrame(row).T

            profile = RegionRepProfile.get_region_rep_profile(
                self._gen_fpath, row, gid_col=self._gid_col,
                rep_method=self._rep_method, err_method=self._err_method,
                weight=self._weight, n_profiles=self._n_profiles)[0]

            logger.info('Profile {} out of {} complete.'
                        .format(i + 1, len(meta_static)))

            self._profiles[0][:, i] = profile.flatten()

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

        logger.info('Kicking off {} aggregate rep profile futures.'
                    .format(len(self.meta)))

        iter_chunks = np.array_split(self.meta.index.values,
                                     np.ceil(len(self.meta) / pool_size))
        n_complete = 0
        for iter_chunk in iter_chunks:
            logger.debug('Starting process pool...')
            futures = {}
            loggers = [__name__, 'reV']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                for i in iter_chunk:
                    row = self.meta.loc[i, :]
                    row = pd.DataFrame(row).T
                    future = exe.submit(
                        RegionRepProfile.get_region_rep_profile,
                        self._gen_fpath, row,
                        gid_col=self._gid_col,
                        cf_dset=self._cf_dset,
                        rep_method=self._rep_method,
                        err_method=self._err_method,
                        weight=self._weight,
                        n_profiles=self._n_profiles)

                    futures[future] = i

                for future in as_completed(futures):
                    i = futures[future]
                    profile = future.result()[0]
                    n_complete += 1
                    logger.info('Future {} out of {} complete.'
                                .format(n_complete, len(self.meta)))
                    log_mem(logger, log_level='DEBUG')

                    self._profiles[0][:, i] = profile.flatten()

    def _run(self, fout=None, scaled_precision=False,
             max_workers=None):
        """
        Run representative profiles in serial or parallel and save to disc

        Parameters
        ----------
        fout : str, optional
            filepath to output h5 file, by default None
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
            self.save_profiles(fout, scaled_precision=scaled_precision,
                               save_rev_summary=False)

        logger.info('Representative aggregate profiles complete!')

    @classmethod
    def run(cls, gen_fpath, rev_summary, gid_col='gen_gids',
            cf_dset='cf_profile', weight='gid_counts', fout=None,
            scaled_precision=False, max_workers=None):
        """Run representative profiles by calculating the weighted aggregate
        (meanoid) profile for each SC point.

        Parameters
        ----------
        gen_fpath : str
            Filepath to reV gen output file to extract "cf_profile" from.
        rev_summary : str | pd.DataFrame
            Aggregated rev supply curve summary file. Str filepath or full df.
        gid_col : str
            Column label in rev_summary that contains the generation gids
            (data index in gen_fpath).
        cf_dset : str
            Dataset name to pull generation profiles from.
        weight : str | None
            Column in rev_summary used to apply weighted mean to profiles.
            The supply curve table data in the weight column should have
            weight values corresponding to the gid_col in the same row.
        fout : str, optional
            filepath to output h5 file, by default None
        scaled_precision : bool, optional
            Flag to scale cf_profiles by 1000 and save as uint16.,
            by default False
        max_workers : int, optional
            Number of parallel workers. 1 will run serial, None will use all
            available., by default None

        Returns
        -------
        profiles : dict
            dict of n_profile-keyed arrays with shape (time, n) for the
            representative profiles for each supply curve point (n).
            For the AggregateRepProfile class, this only has one key: 0.
        meta : pd.DataFrame
            Meta dataframe (reV supply curve summary).
        time_index : pd.DatatimeIndex
            Datetime Index for represntative profiles
        """

        arp = cls(gen_fpath, rev_summary, gid_col=gid_col, cf_dset=cf_dset,
                  weight=weight)

        arp._run(fout=fout, scaled_precision=scaled_precision,
                 max_workers=max_workers)

        return arp._profiles, arp._meta, arp._time_index
