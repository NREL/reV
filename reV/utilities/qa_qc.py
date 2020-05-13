# -*- coding: utf-8 -*-
"""
Quality assurance and control module
"""
import logging
import numpy as np
import pandas as pd
from warnings import warn

from rex import Resource
from rex.utilities.execution import SpawnProcessPool

logger = logging.getLogger(__name__)


class QAQC:
    """
    reV Quality Assurance and Control Handler
    """
    def __init__(self, h5_file):
        self._h5_file = h5_file
        self._h5 = Resource(h5_file)

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self.h5_file)

        return msg

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._h5.close()

        if type is not None:
            raise

    @property
    def h5_file(self):
        """
        .h5 file path

        Returns
        -------
        str
        """
        return self._h5_file

    @staticmethod
    def _compute_sites_summary(h5_fhandle, ds_name, sites=None):
        """
        Compute summary stats for given sites of given dataset

        Parameters
        ----------
        h5_fhandle : Resource
            Resource handler object
        ds_name : str
            Dataset name of interest
        sites : list | slice, optional
            sites of interest, by default None

        Returns
        -------
        sites_summary : pandas.DataFrame
            Summary stats for given sites / dataset
        """
        if sites is None:
            sites = slice(None)

        sites_meta = h5_fhandle['meta', sites]
        sites_data = h5_fhandle[ds_name, :, sites]
        sites_summary = pd.DataFrame(sites_data, columns=sites_meta.index)
        sites_summary = sites_summary.describe().T.drop(columns=['count'])
        sites_summary['sum'] = sites_data.sum(axis=0)

        return sites_summary

    @staticmethod
    def _compute_ds_summary(h5_fhandle, ds_name):
        """
        Compute summary statistics for given dataset (assumed to be a vector)

        Parameters
        ----------
        h5_fhandle : Resource
            Resource handler object
        ds_name : str
            Dataset name of interest

        Returns
        -------
        ds_summary : pandas.DataFrame
            Summary statistics for dataset
        """
        ds_data = h5_fhandle[ds_name, :]
        ds_summary = pd.DataFrame(ds_data, columns=[ds_name])
        ds_summary = ds_summary.describe().drop(['count'])
        ds_summary.at['sum'] = ds_data.sum()

        return ds_summary

    def summarize_dset(self, ds_name, process_size=None, max_workers=None,
                       out_path=None):
        """
        Compute dataset summary. If dataset is 2D compute temporal statistics
        for each site

        Parameters
        ----------
        ds_name : str
            Dataset name of interest
        process_size : int, optional
            Number of sites to process at a time, by default None
        max_workers : int, optional
            Number of workers to use in parallel, if 1 run in serial,
            if None use all available cores, by default None
        out_path : str
            File path to save summary to

        Returns
        -------
        summary : pandas.DataFrame
            Summary table for dataset
        """
        ds_shape, _, ds_chunks = self._h5.get_dset_properties(ds_name)
        if len(ds_shape) > 1:
            sites = np.arange(ds_shape[1])
            if max_workers > 1:
                if process_size is None:
                    process_size = ds_chunks

                sites = np.array_split(sites,
                                       int(np.ceil(len(sites) / process_size)))
                loggers = [__name__]
                with SpawnProcessPool(max_workers=max_workers,
                                      loggers=loggers) as ex:
                    futures = []
                    for site_slice in sites:
                        futures.append(ex.submit(
                            self._compute_sites_summary,
                            self._h5,
                            ds_name,
                            site_slice))

                    summary = [future.result() for future in futures]

                summary = pd.concat(summary)
            else:
                if process_size is None:
                    summary = self._compute_ds_summary(self._h5, ds_name,
                                                       sites)
                else:
                    sites = np.array_split(
                        sites, int(np.ceil(len(sites) / process_size)))

                    summary = []
                    for site_slice in sites:
                        summary.append(self._compute_ds_summary(self._h5,
                                                                ds_name,
                                                                site_slice))

                    summary = pd.concat(summary)

            summary.index.name = 'gid'
        else:
            if process_size is not None or max_workers > 1:
                msg = ("Computing summary statistics for 1D datasets will "
                       "proceed in serial")
                logger.warning(msg)
                warn(msg)

            summary = self._compute_ds_summary(self._h5, ds_name)

        if out_path is not None:
            summary.to_csv(out_path)

        return summary

    def summarize_means(self, out_path=None):
        """
        [

        Parameters
        ----------
        out_path : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        meta = self._h5.meta
        meta.index.name = 'gid'
        meta = meta.reset_index()
        for ds_name in self._h5.datasets:
            if ds_name not in ['meta', 'time_index']:
                shape = self._h5.get_dset_properties(ds_name)[0]
                if len(shape) == 1:
                    meta[ds_name] = self._h5[ds_name]

        if out_path is not None:
            meta.to_csv(out_path)

        return meta
