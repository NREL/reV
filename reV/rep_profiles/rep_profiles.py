# -*- coding: utf-8 -*-
"""Representative profile extraction utilities.

Created on Thu Oct 31 12:49:23 2019

@author: gbuster
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import pandas as pd
import numpy as np
import os
import logging

from reV.handlers.resource import Resource
from reV.handlers.outputs import Outputs
from reV.utilities.exceptions import FileInputError
from reV.utilities.utilities import parse_year


logger = logging.getLogger(__name__)


class RepresentativeMethods:
    """Class for organizing the methods to determine representative-ness"""

    def __init__(self, profiles, rep_method='meanoid', err_method='rmse'):
        """
        Parameters
        ----------
        profiles : np.ndarray
            (time, sites) timeseries array of cf profile data.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        """
        self._rep_method = self.rep_methods[rep_method]
        self._err_method = self.err_methods[err_method]
        self._profiles = profiles

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
    def meanoid(profiles):
        """Find the mean profile across all sites.

        Parameters
        ----------
        profiles : np.ndarray
            (time, sites) timeseries array of cf profile data.

        Returns
        -------
        arr : np.ndarray
            (time, 1) timeseries of the mean of all cf profiles across sites.
        """
        arr = profiles.mean(axis=1).reshape((len(profiles), 1))
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

    @staticmethod
    def mbe(profiles, baseline, i_profile=0):
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
        i_rep = RepresentativeMethods.nargmin(mbe, i_profile)
        return profiles[:, i_rep], i_rep

    @staticmethod
    def mae(profiles, baseline, i_profile=0):
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
        i_rep = RepresentativeMethods.nargmin(mae, i_profile)
        return profiles[:, i_rep], i_rep

    @staticmethod
    def rmse(profiles, baseline, i_profile=0):
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
        i_rep = RepresentativeMethods.nargmin(rmse, i_profile)
        return profiles[:, i_rep], i_rep

    @classmethod
    def run(cls, profiles, rep_method='meanoid', err_method='rmse',
            i_profile=0):
        """Run representative profile methods.

        Parameters
        ----------
        profiles : np.ndarray
            (time, sites) timeseries array of cf profile data.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
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
        inst = cls(profiles, rep_method=rep_method, err_method=err_method)
        baseline = inst._rep_method(inst._profiles)
        profile, i_rep = inst._err_method(inst._profiles, baseline,
                                          i_profile=i_profile)
        return profile, i_rep


class RegionRepProfile:
    """Framework to handle rep profile for one resource region"""

    def __init__(self, gen_fpath, rev_summary, cf_dset='cf_profile',
                 rep_method='meanoid', err_method='rmse', i_profile=0):
        """
        Parameters
        ----------
        gen_fpath : str
            Filepath to reV gen output file to extract "cf_profile" from.
        rev_summary : pd.DataFrame
            Aggregated rev supply curve summary file trimmed to just one
            region to get a rep profile for.
        cf_dset : str
            Dataset name to pull generation profiles from.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        i_profile : int
            The index of the represntative profile being saved
            (for n_profiles). 0 is the most representative profile.
        """

        self._gen_fpath = gen_fpath
        self._rev_summary = rev_summary
        self._cf_dset = cf_dset
        self._profile = None
        self._i_rep = None
        self._rep_method = rep_method
        self._err_method = err_method
        self._err_method = err_method
        self._i_profile = i_profile

    def _get_profiles(self, gen_gids):
        """Retrieve the cf profile array from the generation h5 file.

        Parameters
        ----------
        gen_gids : list | np.ndarray
            GIDs corresponding to the column indexes in the generation file.

        Returns
        -------
        profiles : np.ndarray
            Timeseries array of cf profile data.
        """
        with Resource(self._gen_fpath) as res:
            profiles = res[self._cf_dset, :, sorted(gen_gids)]
        return profiles

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

        if isinstance(data[0], str):
            if '[' and ']' in data[0]:
                data = [json.loads(s) for s in data]
                data = [a for b in data for a in b]

        return data

    @property
    def rep_profile(self):
        """Get the representative profile of this region."""
        if self._profile is None:
            gids = self._get_region_attr(self._rev_summary, 'gen_gids')
            all_profiles = self._get_profiles(gids)
            self._profile, self._i_rep = RepresentativeMethods.run(
                all_profiles, rep_method=self._rep_method,
                err_method=self._err_method, i_profile=self._i_profile)
        return self._profile

    @property
    def i_rep(self):
        """Get the representative profile index of this region."""
        if self._i_rep is None:
            gids = self._get_region_attr(self._rev_summary, 'gen_gids')
            all_profiles = self._get_profiles(gids)
            self._profile, self._i_rep = RepresentativeMethods.run(
                all_profiles, rep_method=self._rep_method,
                err_method=self._err_method)
        return self._i_rep

    @property
    def gen_gid_rep(self):
        """Get the representative profile gen gid of this region."""
        gids = self._get_region_attr(self._rev_summary, 'gen_gids')
        gen_gid_rep = gids[self.i_rep]
        return gen_gid_rep

    @property
    def res_gid_rep(self):
        """Get the representative profile resource gid of this region."""
        gids = self._get_region_attr(self._rev_summary, 'res_gids')
        res_gid_rep = gids[self.i_rep]
        return res_gid_rep

    @classmethod
    def get_region_rep_profile(cls, gen_fpath, rev_summary,
                               cf_dset='cf_profile', rep_method='meanoid',
                               err_method='rmse', i_profile=0):
        """Class method for parallelization of rep profile calc.

        Parameters
        ----------
        gen_fpath : str
            Filepath to reV gen output file to extract "cf_profile" from.
        rev_summary : pd.DataFrame
            Aggregated rev supply curve summary file trimmed to just one
            region to get a rep profile for.
        cf_dset : str
            Dataset name to pull generation profiles from.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        i_profile : int
            The index of the represntative profile being saved
            (for n_profiles). 0 is the most representative profile.

        Returns
        -------
        rep_profile : np.ndarray
            (time, 1) array for the most representative profile
        i_rep : int
            Column Index in profiles of the representative profile.
        gen_gid_rep : int
            Generation gid of the representative profile.
        res_gid_rep : int
            Resource gid of the representative profile.
        """
        r = cls(gen_fpath, rev_summary, cf_dset=cf_dset, rep_method=rep_method,
                err_method=err_method, i_profile=i_profile)
        return r.rep_profile, r.i_rep, r.gen_gid_rep, r.res_gid_rep


class RepProfiles:
    """Framework for calculating the representative profiles."""

    def __init__(self, gen_fpath, rev_summary, reg_cols, cf_dset='cf_profile',
                 rep_method='meanoid', err_method='rmse'):
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
        cf_dset : str
            Dataset name to pull generation profiles from.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        """

        logger.debug('Running rep profiles with gen_fpath: "{}"'
                     .format(gen_fpath))
        logger.debug('Running rep profiles with rev_summary: "{}"'
                     .format(rev_summary))
        logger.debug('Running rep profiles with region columns: "{}"'
                     .format(reg_cols))
        logger.debug('Running rep profiles with representative method: "{}"'
                     .format(rep_method))
        logger.debug('Running rep profiles with error method: "{}"'
                     .format(err_method))

        if reg_cols is None:
            reg_cols = []
        elif isinstance(reg_cols, str):
            reg_cols = [reg_cols]

        self._check_rev_gen(gen_fpath, cf_dset)
        self._cf_dset = cf_dset
        self._gen_fpath = gen_fpath
        self._rev_summary = self._parse_rev_summary(rev_summary, reg_cols)
        self._reg_cols = reg_cols
        self._regions = {k: self._rev_summary[k].unique().tolist()
                         for k in self._reg_cols}
        self._time_index = None
        self._meta = None
        self._profiles = np.zeros((len(self.time_index), len(self.meta)),
                                  dtype=np.float32)
        self._rep_method = rep_method
        self._err_method = err_method

    @staticmethod
    def _parse_rev_summary(rev_summary, reg_cols):
        """Extract, parse, and check the rev summary table.

        Parameters
        ----------
        rev_summary : str | pd.DataFrame
            Aggregated rev supply curve summary file. Str filepath or full df.
        reg_cols : list
            Column label(s) for a region column to extract profiles for.

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

        e = 'Column label "{}" not found in rev_summary table!'
        req_cols = ['gen_gids'] + reg_cols
        for c in req_cols:
            if c not in rev_summary:
                logger.error(e.format(c))
                raise KeyError(e.format(c))

        return rev_summary

    @staticmethod
    def _check_rev_gen(gen_fpath, cf_dset):
        """Check rev gen file for requisite datasets.

        Parameters
        ----------
        gen_fpath : str
            Filepath to reV gen output file to extract "cf_profile" from.
        cf_dset : str
            Dataset name to pull generation profiles from.
        """
        with Resource(gen_fpath) as res:
            dsets = res.dsets
        if cf_dset not in dsets:
            raise KeyError('reV gen file needs to have "{}" '
                           'dataset to calculate representative profiles!'
                           .format(cf_dset))
        if 'time_index' not in str(dsets):
            raise KeyError('reV gen file needs to have "time_index" '
                           'dataset to calculate representative profiles!')

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
        if self._meta is None:
            self._meta = self._rev_summary[self._reg_cols].drop_duplicates()
            self._meta = self._meta.reset_index(drop=True)
            self._meta['rep_gen_gid'] = -1
            self._meta['rep_res_gid'] = -1
        return self._meta

    @property
    def profiles(self):
        """Get the array of representative CF profiles corresponding to meta.

        Returns
        -------
        profiles : np.ndarray
            (time, n) cf profiles. Array of zeros if not yet calculated.
        """
        return self._profiles

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

    def _init_fout(self, fout, n_profiles):
        """Initialize an output h5 file for n_profiles

        Parameters
        ----------
        fout : None | str
            None or filepath to output h5 file.
        n_profiles : int
            The number of profiles to be saved.
        """
        if fout is not None:
            dsets = []
            shapes = {}
            attrs = {}
            chunks = {}
            dtypes = {}
            for i in range(n_profiles):
                dset = 'rep_profiles_{}'.format(i)
                dsets.append(dset)
                shapes[dset] = self.profiles.shape
                attrs[dset] = None
                chunks[dset] = None
                dtypes[dset] = self.profiles.dtype

            Outputs.init_h5(fout, dsets, shapes, attrs, chunks, dtypes,
                            self.meta, time_index=self.time_index)

            with Outputs(fout, mode='a') as out:
                rev_sum = Outputs.to_records_array(self._rev_summary)
                out._create_dset('rev_summary', rev_sum.shape,
                                 rev_sum.dtype, data=rev_sum)

    def _write_fout(self, fout, i):
        """Write profiles and meta to an output file.

        Parameters
        ----------
        fout : None | str
            None or filepath to output h5 file.
        i : int
            The index of the represntative profile being saved (for n_profiles)
        """
        if fout is not None:
            with Outputs(fout, mode='a') as out:
                dset = 'rep_profiles_{}'.format(i)
                rev_sum = Outputs.to_records_array(self._rev_summary)
                out[dset] = self.profiles
                out['rev_summary'] = rev_sum

    def _run_serial(self, i_profile=0):
        """Compute all representative profiles in serial.

        Parameters
        ----------
        i_profile : int
            The index of the represntative profile being saved
            (for n_profiles). 0 is the most representative profile.
        """
        logger.info('Running {} rep profile calculations in serial.'
                    .format(len(self.meta)))
        meta_static = deepcopy(self.meta)
        for i, row in meta_static.iterrows():
            region_dict = {k: v for (k, v) in row.to_dict().items()
                           if k in self._reg_cols}

            mask = self._get_mask(region_dict)
            profile, _, ggid, rgid = RegionRepProfile.get_region_rep_profile(
                self._gen_fpath, self._rev_summary[mask],
                cf_dset=self._cf_dset, rep_method=self._rep_method,
                err_method=self._err_method, i_profile=i_profile)

            logger.debug('Selected gen gid {} for region: {}'
                         .format(ggid, region_dict))
            self._meta.at[i, 'rep_gen_gid'] = ggid
            self._meta.at[i, 'rep_res_gid'] = rgid
            self._profiles[:, i] = profile

    def _run_parallel(self, i_profile=0):
        """Compute all representative profiles in parallel

        Parameters
        ----------
        i_profile : int
            The index of the represntative profile being saved
            (for n_profiles). 0 is the most representative profile.
        """
        logger.info('Kicking off {} rep profile futures.'
                    .format(len(self.meta)))
        futures = {}
        with ProcessPoolExecutor() as exe:
            for i, row in self.meta.iterrows():
                region_dict = {k: v for (k, v) in row.to_dict().items()
                               if k in self._reg_cols}

                mask = self._get_mask(region_dict)

                future = exe.submit(RegionRepProfile.get_region_rep_profile,
                                    self._gen_fpath, self._rev_summary[mask],
                                    cf_dset=self._cf_dset,
                                    rep_method=self._rep_method,
                                    err_method=self._err_method,
                                    i_profile=i_profile)

                futures[future] = [i, region_dict]

            for future in as_completed(futures):
                i, region_dict = futures[future]
                profile, _, ggid, rgid = future.result()
                logger.debug('Selected gen gid {} for region {}.'
                             .format(ggid, region_dict))
                self._meta.at[i, 'rep_gen_gid'] = ggid
                self._meta.at[i, 'rep_res_gid'] = rgid
                self._profiles[:, i] = profile

    @classmethod
    def run(cls, gen_fpath, rev_summary, reg_cols, cf_dset='cf_profile',
            rep_method='meanoid', err_method='rmse', parallel=True, fout=None,
            n_profiles=1):
        """Run representative profiles.

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
        cf_dset : str
            Dataset name to pull generation profiles from.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        parallel : bool
            Flag to run in parallel.
        fout : None | str
            None or filepath to output h5 file.
        n_profiles : int
            Number of representative profiles to save to fout.

        Returns
        -------
        out_profiles : dict
            dict of n_profile-keyed arrays with shape (time, n) for the
            representative profiles for each region.
        out_meta : pd.DataFrame
            dict of n_profile-keyed Meta dataframes recording the regions and
            the selected rep profile gid.
        """

        out_profiles = {}
        out_meta = {}

        for i in range(n_profiles):
            logger.info('Starting representative profiles for i_profile #{}'
                        .format(i))
            rp = cls(gen_fpath, rev_summary, reg_cols, cf_dset=cf_dset,
                     rep_method=rep_method, err_method=err_method)
            if parallel:
                rp._run_parallel(i_profile=i)
            else:
                rp._run_serial(i_profile=i)

            if i == 0:
                rp._init_fout(fout, n_profiles)
            rp._write_fout(fout, i)

            out_profiles[i] = rp._profiles
            out_meta[i] = rp._meta

        logger.info('Representative profiles complete!')
        return out_profiles, out_meta
