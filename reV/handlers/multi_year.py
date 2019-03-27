"""
Classes to handle capacity factor profiles and annual averages
"""
import logging
import numpy as np
import os
import pandas as pd

from reV.utilities.exceptions import HandlerRuntimeError
from reV.handlers.outputs import Outputs, parse_year

logger = logging.getLogger(__name__)


class MultiYear(Outputs):
    """
    Class to handle multiple years of data
    """
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
        if year is None:
            raise HandlerRuntimeError("Cannot parse year from {}"
                                      .format(f_name))

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
        dset = 'time_index'
        dset_out = self.create_dset_out(source_h5, dset)
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
        dset_out = self.create_dset_out(source_h5, dset)

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
        Collect dataset dset from all give list of h5 files

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
            self._create_dset('meta', meta.shape, meta.dtype,
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
        source_dsets = [ds for ds in self.dsets if dset in ds]
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
        if dset_out in self.dsets:
            self[dset_out] = dset_data
        else:
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

        MY_means = np.zeros(len(self))
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

        MY_stdev = np.zeros(means.shape)
        for ds in source_dsets:
            if self._h5[ds].shape == MY_stdev.shape:
                MY_stdev += (self[ds] - means)**2
            else:
                raise HandlerRuntimeError("{} shape {} should be {}"
                                          .format(ds, self._h5[ds].shape,
                                                  MY_stdev.shape))

        MY_stdev = np.sqrt(MY_stdev / len(source_dsets))
        self._update_dset(dset_out, MY_stdev)

        return MY_stdev

    def stdev(self, dset):
        """
        Extract or compute multi-year standard deviation for given source dset

        Parameters
        ----------
        dset : str
            Dataset of interest

        Returns
        -------
        MY_stdev : ndarray
            Array of multi-year standard deviation for dataset of interest
        """
        my_dset = "{}-stdev".format(dset)
        if my_dset in self.dsets:
            MY_stdev = self[my_dset]
        else:
            MY_means = self.means(dset)
            MY_stdev = self._compute_stdev(my_dset, means=MY_means)

        return MY_stdev

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
        MY_cv = self.stdev(dset) / self.means(dset)
        return MY_cv
