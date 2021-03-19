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
from reV.utilities import log_versions

from rex.utilities.utilities import parse_year, get_lat_lon_cols

logger = logging.getLogger(__name__)


class MultiYear(Outputs):
    """
    Class to handle multiple years of data and:
    - collect datasets from multiple years
    - compute multi-year means
    - compute multi-year standard deviations
    - compute multi-year coefficient of variations

    """
    def __init__(self, h5_file, group=None, unscale=True, mode='r',
                 str_decode=True):
        """
        Parameters
        ----------
        h5_file : str
            Path to .h5 resource file
        group : str
            Group to collect datasets into
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        mode : str
            Mode to instantiate h5py.File instance
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        """
        log_versions(logger)
        super().__init__(h5_file, group=group, unscale=unscale, mode=mode,
                         str_decode=str_decode)

    @staticmethod
    def _create_dset_name(source_h5, dset):
        """
        Create output dataset name by parsing year from source_h5 and
        appending to source dataset name.

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
        f_name = os.path.basename(source_h5)
        year = parse_year(f_name)
        dset_out = "{}-{}".format(dset, year)
        return dset_out

    def _copy_time_index(self, source_h5):
        """
        Copy time_index from source_h5 to time_index_{year} in multiyear .h5

        Parameters
        ----------
        source_h5 : str
            Path to source .h5 file to copy data from
        """
        dset_out = self._create_dset_name(source_h5, 'time_index')
        if dset_out not in self.datasets:
            logger.debug("- Collecting time_index from {}"
                         .format(os.path.basename(source_h5)))
            with Outputs(source_h5, mode='r') as f_in:
                time_index = f_in.h5['time_index'][...]

            self._create_dset(dset_out, time_index.shape, time_index.dtype,
                              data=time_index)

    def _copy_dset(self, source_h5, dset, meta=None, pass_through=False):
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
        pass_through : bool
            Flag to just pass through dataset without name modifications
            (no differences between years, no means or stdevs)
        """
        if pass_through:
            dset_out = dset
        else:
            dset_out = self._create_dset_name(source_h5, dset)

        if dset_out not in self.datasets:
            logger.debug("- Collecting {} from {}"
                         .format(dset, os.path.basename(source_h5)))
            with Outputs(source_h5, unscale=False, mode='r') as f_in:
                if meta is not None:
                    cols = get_lat_lon_cols(meta)
                    source_meta = f_in.meta
                    if not meta[cols].equals(source_meta[cols]):
                        raise HandlerRuntimeError('Coordinates do not match')

                _, ds_dtype, ds_chunks = f_in.get_dset_properties(dset)
                ds_attrs = f_in.get_attrs(dset=dset)
                ds_data = f_in[dset]

            self._create_dset(dset_out, ds_data.shape, ds_dtype,
                              chunks=ds_chunks, attrs=ds_attrs, data=ds_data)

    def collect(self, source_files, dset, profiles=False, pass_through=False):
        """
        Collect dataset dset from given list of h5 files

        Parameters
        ----------
        source_files : list
            List of .h5 files to collect datasets from
            NOTE: .h5 file names much indicate the year the data pertains to
        dset : str
            Dataset to collect
        profiles : bool
            Boolean flag to indicate if profiles are being collected
            If True also collect time_index
        pass_through : bool
            Flag to just pass through dataset without name modifications
            (no differences between years, no means or stdevs)
        """
        with Outputs(source_files[0], mode='r') as f_in:
            meta = f_in.h5['meta'][...]

        if 'meta' not in self.datasets:
            logger.debug("Copying meta")
            self._create_dset('meta', meta.shape, meta.dtype,
                              data=meta)

        meta = pd.DataFrame(meta)
        for year_h5 in source_files:
            if profiles:
                self._copy_time_index(year_h5)

            self._copy_dset(year_h5, dset, meta=meta,
                            pass_through=pass_through)

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
        dset = os.path.basename(dset_out).split("-")[0]
        logger.debug('-- source_dset root = {}'.format(dset))
        my_dset = ["{}-{}".format(dset, val) for val in ['means', 'stdev']]
        source_dsets = [ds for ds in self.datasets if dset in ds
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
        if dset_out in self.datasets:
            logger.debug("- Updating {}".format(dset_out))
            self[dset_out] = dset_data
        else:
            logger.debug("- Creating {}".format(dset_out))
            source_dset = self._get_source_dsets(dset_out)[0]
            _, ds_dtype, ds_chunks = self.get_dset_properties(source_dset)
            ds_attrs = self.get_attrs(dset=source_dset)
            self._add_dset(dset_out, dset_data, ds_dtype,
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
        logger.debug('\t- Computing {} from {}'.format(dset_out, source_dsets))

        my_means = np.zeros(len(self), dtype='float32')
        for ds in source_dsets:
            if self.h5[ds].shape == my_means.shape:
                my_means += self[ds]
            else:
                raise HandlerRuntimeError("{} shape {} should be {}"
                                          .format(ds, self.h5[ds].shape,
                                                  my_means.shape))
        my_means /= len(source_dsets)
        self._update_dset(dset_out, my_means)

        return my_means

    def means(self, dset):
        """
        Extract or compute multi-year means for given source dset

        Parameters
        ----------
        dset : str
            Dataset of interest

        Returns
        -------
        my_means : ndarray
            Array of multi-year means for dataset of interest
        """
        my_dset = "{}-means".format(dset)
        if my_dset in self.datasets:
            my_means = self[my_dset]
        else:
            my_means = self._compute_means(my_dset)

        return my_means

    def _compute_stdev(self, dset_out, means=None):
        """
        Compute multi-year standard deviation for given dataset

        Parameters
        ----------
        dset_out : str
            Multi-year stdev dataset name
        means : ndarray
            Array of pre-computed means

        Returns
        -------
        my_stdev : ndarray
            Array of multi-year standard deviations
        """
        if means is None:
            means = self._compute_means("{}-means".format(dset_out))

        source_dsets = self._get_source_dsets(dset_out)

        my_stdev = np.zeros(means.shape, dtype='float32')
        for ds in source_dsets:
            if self.h5[ds].shape == my_stdev.shape:
                my_stdev += (self[ds] - means)**2
            else:
                raise HandlerRuntimeError("{} shape {} should be {}"
                                          .format(ds, self.h5[ds].shape,
                                                  my_stdev.shape))

        my_stdev = np.sqrt(my_stdev / len(source_dsets))
        self._update_dset(dset_out, my_stdev)

        return my_stdev

    def stdev(self, dset):
        """
        Extract or compute multi-year standard deviation for given source dset

        Parameters
        ----------
        dset : str
            Dataset of interest

        Returns
        -------
        my_stdev : ndarray
            Array of multi-year standard deviation for dataset of interest
        """
        my_dset = "{}-stdev".format(dset)
        if my_dset in self.datasets:
            my_stdev = self[my_dset]
        else:
            my_means = self.means(dset)
            my_stdev = self._compute_stdev(my_dset, means=my_means)

        return my_stdev

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
        my_cv : ndarray
            Array of multi-year coefficient of variation for
            dataset of interest
        """
        my_cv = self.stdev(dset) / self.means(dset)
        return my_cv

    @staticmethod
    def is_profile(source_files, dset):
        """
        Check dataset in source files to see if it is a profile.

        Parameters
        ----------
        source_files : list
            List of .h5 files to collect datasets from
        dset : str
            Dataset to collect

        Returns
        -------
        is_profile : bool
            True if profile, False if not.
        """
        with Outputs(source_files[0]) as f:
            if dset not in f.datasets:
                raise KeyError('Dataset "{}" not found in source file: "{}"'
                               .format(dset, source_files[0]))

            shape, _, _ = f.get_dset_properties(dset)

        return len(shape) == 2

    @classmethod
    def pass_through(cls, my_file, source_files, dset, group=None):
        """
        Pass through a dataset that is identical in all source files to a
        dataset of the same name in the output multi-year file.

        Parameters
        ----------
        my_file : str
            Path to multi-year .h5 file
        source_files : list
            List of .h5 files to collect datasets from
        dset : str
            Dataset to pass through (will also be the name of the output
            dataset in my_file)
        group : str
            Group to collect datasets into
        """
        logger.info('Passing through {} into {}.'
                    .format(dset, my_file))
        with cls(my_file, mode='a', group=group) as my:
            my.collect(source_files, dset, pass_through=True)

    @classmethod
    def collect_means(cls, my_file, source_files, dset, group=None):
        """
        Collect and compute multi-year means for given dataset

        Parameters
        ----------
        my_file : str
            Path to multi-year .h5 file
        source_files : list
            List of .h5 files to collect datasets from
        dset : str
            Dataset to collect
        group : str
            Group to collect datasets into
        """
        logger.info('Collecting {} into {} '
                    'and computing multi-year means and standard deviations.'
                    .format(dset, my_file))
        with cls(my_file, mode='a', group=group) as my:
            my.collect(source_files, dset)
            means = my._compute_means("{}-means".format(dset))
            my._compute_stdev("{}-stdev".format(dset), means=means)

    @classmethod
    def collect_profiles(cls, my_file, source_files, dset, group=None):
        """
        Collect multi-year profiles associated with given dataset

        Parameters
        ----------
        my_file : str
            Path to multi-year .h5 file
        source_files : list
            List of .h5 files to collect datasets from
        dset : str
            Profiles dataset to collect
        group : str
            Group to collect datasets into
        """
        logger.info('Collecting {} into {}'.format(dset, my_file))
        with cls(my_file, mode='a', group=group) as my:
            my.collect(source_files, dset, profiles=True)
