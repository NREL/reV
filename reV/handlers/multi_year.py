# -*- coding: utf-8 -*-
"""
Classes to collect reV outputs from multiple annual files.
"""
import logging
import numpy as np
import os
import pandas as pd

from reV.handlers.outputs import Outputs
from reV.utilities.exceptions import HandlerRuntimeError
from reV.utilities.utilities import parse_year

logger = logging.getLogger(__name__)


class MultiYear(Outputs):
    """
    Class to handle multiple years of data and:
    - collect datasets from multiple years
    - compute multi-year means
    - compute multi-year standard deviations
    - compute multi-year coefficient of variations

    """
    def __init__(self, h5_file, group=None, **kwargs):
        """
        Parameters
        ----------
        h5_file : str
            Path to .h5 resource file
        group : str
            Group to collect datasets into
        kwargs : dict
            kwargs to initialize class
        """
        super().__init__(h5_file, **kwargs)
        self._group = group

    @staticmethod
    def create_dset_out(source_h5, dset):
        """
        Create output dataset by parsing year from source_h5 and appending
        to dset_in.

        Parameters
        ----------
        source_h5 : str
            Path to source .h5 file to copy data from
        dset : str
            Dataset to copy

        Returns
        -------
        dset_out : str
            Ouput dataset name
        """
        f_name = os.path.join(source_h5)
        year = parse_year(f_name)
        dset_out = "{}-{}".format(dset, year)
        return dset_out

    def _adjust_group(self, dset):
        """
        If group was provided during initialization, add group to dset

        Parameters
        ----------
        dset : str
            Dataset name

        Returns
        -------
        dset : str
            Modified dataset name
        """
        if self._group is not None:
            dset = '{}/{}'.format(self._group, dset)

        return dset

    def _copy_time_index(self, source_h5):
        """
        Copy time_index from source_h5 to time_index_{year} in multiyear .h5

        Parameters
        ----------
        source_h5 : str
            Path to source .h5 file to copy data from
        """
        dset = self._adjust_group('time_index')
        dset_out = self.create_dset_out(source_h5, dset)
        if dset_out not in self.dsets:
            logger.debug("- Collecting time_index from {}"
                         .format(os.path.basename(source_h5)))
            with Outputs(source_h5, mode='r') as f_in:
                time_index = f_in._h5['time_index'][...]

            self._create_dset(dset_out, time_index.shape, time_index.dtype,
                              data=time_index)

    def _copy_dset(self, source_h5, dset, meta=None):
        """
        Copy dset_in from source_h5 to multiyear .h5

        Parameters
        ----------
        source_h5 : str
            Path to source .h5 file to copy data from
        dset : str
            Dataset to copy
        meta : pandas.DataFrame
            If provided confirm that source meta matches given meta
        """
        dset_out = self._adjust_group(self.create_dset_out(source_h5, dset))
        if dset_out not in self.dsets:
            logger.debug("- Collecting {} from {}"
                         .format(dset, os.path.basename(source_h5)))
            with Outputs(source_h5, unscale=False, mode='r') as f_in:
                if meta is not None:
                    cols = ['latitude', 'longitude']
                    source_meta = f_in.meta
                    if not meta[cols].equals(source_meta[cols]):
                        raise HandlerRuntimeError('Coordinates do not match')

                _, ds_dtype, ds_chunks = f_in.get_dset_properties(dset)
                ds_attrs = f_in.get_attrs(dset=dset)
                ds_data = f_in[dset]

            self._create_dset(dset_out, ds_data.shape, ds_dtype,
                              chunks=ds_chunks, attrs=ds_attrs, data=ds_data)

    def collect(self, h5_files, dset, profiles=False):
        """
        Collect dataset dset from given list of h5 files

        Parameters
        ----------
        h5_files : list
            List of .h5 files to collect datasets from
            NOTE: .h5 file names much indicate the year the data pertains to
        dset : str
            Dataset to collect
        profiles : bool
            Boolean flag to indicate if profiles are being collected
            If True also collect time_index
        """
        with Outputs(h5_files[0], mode='r') as f_in:
            meta = f_in._h5['meta'][...]

        if 'meta' not in self.dsets:
            logger.debug("Copying meta")
            ds_name = self._adjust_group('meta')
            self._create_dset(ds_name, meta.shape, meta.dtype,
                              data=meta)

        meta = pd.DataFrame(meta)
        for year_h5 in h5_files:
            if profiles:
                self._copy_time_index(year_h5)

            self._copy_dset(year_h5, dset, meta=meta)

    def _get_source_dsets(self, dset_out):
        """
        Extract all available annual datasets associated with dset

        Parameters
        ----------
        dset_out : str
            Output dataset to find source datasets for

        Returns
        -------
        source_dsets : list
            List of annual datasets
        """
        dset = dset_out.split("-")[0]
        my_dset = ["{}-{}".format(dset, val) for val in ['means', 'std']]
        source_dsets = [ds for ds in self.dsets if dset in ds
                        and ds not in my_dset]
        if dset_out in source_dsets:
            source_dsets.remove(dset_out)

        return source_dsets

    def _update_dset(self, dset_out, dset_data):
        """
        Update dataset, create if needed

        Parameters
        ----------
        dset_out : str
            Dataset name
        dset_data : ndarray
            Dataset data to write to disc
        """
        dset_name = self._adjust_group(dset_out)
        if dset_name in self.dsets:
            logger.debug("- Updating {}".format(dset_name))
            self[dset_name] = dset_data
        else:
            logger.debug("- Creating {}".format(dset_name))
            source_dset = self._get_source_dsets(dset_out)[0]
            _, ds_dtype, ds_chunks = self.get_dset_properties(source_dset)
            ds_attrs = self.get_attrs(dset=source_dset)
            self._add_dset(dset_name, dset_data, ds_dtype,
                           chunks=ds_chunks, attrs=ds_attrs)

    def _compute_means(self, dset_out):
        """
        Compute multi-year means for given dataset

        Parameters
        ----------
        dset_out : str
            Multi-year means dataset name

        Returns
        -------
        my_means : ndarray
            Array of multi-year means
        """
        source_dsets = self._get_source_dsets(dset_out)

        MY_means = np.zeros(len(self), dtype='float32')
        for ds in source_dsets:
            if self._h5[ds].shape == MY_means.shape:
                MY_means += self[ds]
            else:
                raise HandlerRuntimeError("{} shape {} should be {}"
                                          .format(ds, self._h5[ds].shape,
                                                  MY_means.shape))
        MY_means /= len(source_dsets)
        self._update_dset(dset_out, MY_means)

        return MY_means

    def means(self, dset):
        """
        Extract or compute multi-year means for given source dset

        Parameters
        ----------
        dset : str
            Dataset of interest

        Returns
        -------
        MY_means : ndarray
            Array of multi-year means for dataset of interest
        """
        my_dset = "{}-means".format(dset)
        if my_dset in self.dsets:
            MY_means = self[my_dset]
        else:
            MY_means = self._compute_means(my_dset)

        return MY_means

    def _compute_std(self, dset_out, means=None):
        """
        Compute multi-year standard deviation for given dataset

        Parameters
        ----------
        dset_out : str
            Multi-year std dataset name
        means : ndarray
            Array of pre-computed means

        Returns
        -------
        my_std : ndarray
            Array of multi-year standard deviations
        """
        if means is None:
            means = self._compute_means("{}-means".format(dset_out))

        source_dsets = self._get_source_dsets(dset_out)

        MY_std = np.zeros(means.shape, dtype='float32')
        for ds in source_dsets:
            if self._h5[ds].shape == MY_std.shape:
                MY_std += (self[ds] - means)**2
            else:
                raise HandlerRuntimeError("{} shape {} should be {}"
                                          .format(ds, self._h5[ds].shape,
                                                  MY_std.shape))

        MY_std = np.sqrt(MY_std / len(source_dsets))
        self._update_dset(dset_out, MY_std)

        return MY_std

    def std(self, dset):
        """
        Extract or compute multi-year standard deviation for given source dset

        Parameters
        ----------
        dset : str
            Dataset of interest

        Returns
        -------
        MY_std : ndarray
            Array of multi-year standard deviation for dataset of interest
        """
        my_dset = "{}-std".format(dset)
        if my_dset in self.dsets:
            MY_std = self[my_dset]
        else:
            MY_means = self.means(dset)
            MY_std = self._compute_std(my_dset, means=MY_means)

        return MY_std

    def CV(self, dset):
        """
        Extract or compute multi-year coefficient of variation for given
        source dset

        Parameters
        ----------
        dset : str
            Dataset of interest

        Returns
        -------
        MY_cv : ndarray
            Array of multi-year coefficient of variation for
            dataset of interest
        """
        MY_cv = self.std(dset) / self.means(dset)
        return MY_cv

    @classmethod
    def collect_means(cls, my_file, h5_files, dset, group=None):
        """
        Collect and compute multi-year means for given dataset

        Parameters
        ----------
        my_file : str
            Path to multi-year .h5 file
        h5_files : list
            List of .h5 files to collect datasets from
        dset : str
            Dataset to collect
        group : str
            Group to collect datasets into
        """
        logger.info('Collecting {} into {} '.format(dset, my_file),
                    'and computing multi-year means and standard deviation.')
        with cls(my_file, mode='a', group=group) as my:
            my.collect(h5_files, dset)
            means = my._compute_means("{}-means".format(dset))
            my._compute_std("{}-std".format(dset), means=means)

    @classmethod
    def collect_profiles(cls, my_file, h5_files, dset, group=None):
        """
        Collect multi-year profiles associated with given dataset

        Parameters
        ----------
        my_file : str
            Path to multi-year .h5 file
        h5_files : list
            List of .h5 files to collect datasets from
        dset : str
            Profiles dataset to collect
        group : str
            Group to collect datasets into
        """
        logger.info('Collecting {} into {}'.format(dset, my_file))
        with cls(my_file, mode='a', group=group) as my:
            my.collect(h5_files, dset, profiles=True)
