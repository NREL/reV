"""
Classes to handle capacity factor profiles and annual averages
"""
import logging
import os

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

        dset_out = "{}_{}".format(dset, year)
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

        self._add_dset(dset_out, ds_data, ds_dtype,
                       chunks=ds_chunks, attrs=ds_attrs)

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
        meta = None
        for year_h5 in h5_files:
            self._copy_dset(year_h5, dset, meta=meta)
            if profiles:
                self._copy_time_index(year_h5)

            if meta is None:
                with Outputs(year_h5, mode='r') as f:
                    meta = f.meta
