# -*- coding: utf-8 -*-
"""
Base class to handle collection of profiles and means across multiple .h5 files
"""
import logging
import numpy as np
import os
import sys
import psutil
import pandas as pd
import time
import shutil
from warnings import warn

from reV.handlers.outputs import Outputs
from reV.utilities.exceptions import (CollectionRuntimeError,
                                      CollectionValueError,
                                      CollectionWarning)

from rex.utilities.loggers import log_mem

logger = logging.getLogger(__name__)


class DatasetCollector:
    """
    Class to collect single datasets from several source files into a final
    output file.
    """
    def __init__(self, h5_file, source_files, gids, dset_in, dset_out=None,
                 mem_util_lim=0.7):
        """
        Parameters
        ----------
        h5_file : str
            Path to h5_file into which dataset is to be collected
        source_files : list
            List of source filepaths.
        gids : list
            list of gids to be collected
        dset_in : str
            Dataset to collect
        dset_out : str
            Dataset into which collected data is to be written
        mem_util_lim : float
            Memory utilization limit (fractional). This sets how many sites
            will be collected at a time.
        """
        self._h5_file = h5_file
        self._source_files = source_files
        self._gids = gids

        self._dset_in = dset_in
        if dset_out is None:
            dset_out = dset_in
        self._dset_out = dset_out

        tot_mem = psutil.virtual_memory().total
        self._mem_avail = mem_util_lim * tot_mem
        self._attrs, self._axis, self._site_mem_req = self._pre_collect()

        logger.debug('Available memory for collection is {} bytes'
                     .format(self._mem_avail))
        logger.debug('Site memory requirement is: {} bytes'
                     .format(self._site_mem_req))

    @staticmethod
    def parse_meta(h5_file):
        """
        Extract and convert meta data from a rec.array to pandas.DataFrame

        Parameters
        ----------
        h5_file : str
            Path to .h5 file from which meta is to be parsed

        Returns
        -------
        meta : pandas.DataFrame
            Portion of meta data corresponding to sites in h5_file
        """
        with Outputs(h5_file, mode='r') as f:
            meta = f.meta

        return meta

    @staticmethod
    def _get_site_mem_req(shape, dtype, n=100):
        """Get the memory requirement to collect one site from a dataset of
        shape and dtype

        Parameters
        ----------
        shape : tuple
            Shape of dataset to be collected (n_time, n_sites)
        dtype : np.dtype
            Numpy dtype of dataset (disk dtype)
        n : int
            Number of sites to prototype the memory req with.

        Returns
        -------
        site_mem : float
            Memory requirement in bytes for one site from a dataset with
            shape and dtype.
        """
        m = 1
        if len(shape) > 1:
            m = shape[0]
        site_mem = sys.getsizeof(np.ones((m, n), dtype=dtype)) / n
        return site_mem

    def _pre_collect(self):
        """Run a pre-collection check and get relevant dset attrs.

        Returns
        -------
        attrs : dict
            Dictionary of dataset attributes for the dataset being collected.
        axis : int
            Axis size (1 is 1D array, 2 is 2D array)
        site_mem_req : float
            Memory requirement in bytes to collect a single site from one
            source file.
        """
        with Outputs(self._source_files[0], mode='r') as f:
            shape, dtype, chunks = f.get_dset_properties(self._dset_in)
            attrs = f.get_attrs(self._dset_in)
            axis = len(f[self._dset_in].shape)

        with Outputs(self._h5_file, mode='a') as f:
            if axis == 1:
                dset_shape = (len(f),)
            elif axis == 2:
                if 'time_index' in f.datasets:
                    dset_shape = f.shape
                else:
                    m = ("'time_index' must be combined "
                         "before profiles can be "
                         "combined.")
                    logger.error(m)
                    raise CollectionRuntimeError(m)
            else:
                m = ('Cannot collect dset "{}" with '
                     'axis {}'.format(self._dset_in, axis))
                logger.error(m)
                raise CollectionRuntimeError(m)

            if self._dset_out not in f.datasets:
                f._create_dset(self._dset_out, dset_shape, dtype,
                               chunks=chunks, attrs=attrs)

        site_mem_req = self._get_site_mem_req(shape, dtype)

        return attrs, axis, site_mem_req

    @staticmethod
    def _get_gid_slice(gids_out, source_gids, fn_source):
        """Find the site slice that the chunked set of source gids belongs to.

        Parameters
        ----------
        gids_out : list
            List of resource GIDS in the final output meta data f_out
        source_gids : list
            List of resource GIDS in one chunk of source data.
        fn_source : str
            Source filename for warning printout.

        Returns
        -------
        site_slice : slice | np.ndarray
            Slice in the final output file to write data to from source gids.
            If gids in destination file are non-sequential, a boolean array of
            indexes is returned and a warning is printed.
        """

        locs = np.where(np.isin(gids_out, source_gids))[0]
        if not any(locs):
            e = ('DatasetCollector could not locate source gids in '
                 'output gids. \n\t Source gids: {} \n\t Output gids: {}'
                 .format(source_gids, gids_out))
            logger.error(e)
            raise CollectionRuntimeError(e)
        sequential_locs = np.arange(locs.min(), locs.max() + 1)

        if not len(locs) == len(sequential_locs):
            w = ('GID indices for source file "{}" are not '
                 'sequential in destination file!'.format(fn_source))
            logger.warning(w)
            warn(w, CollectionWarning)
            site_slice = np.isin(gids_out, source_gids)
        else:
            site_slice = slice(locs.min(), locs.max() + 1)

        return site_slice

    def _get_source_gid_chunks(self, f_source):
        """Split the gids from the f_source into chunks based on memory req.

        Parameters
        ----------
        f_source : reV.handlers.outputs.Output
            Source file handler

        Returns
        -------
        all_source_gids : list
            List of all source gids to be collected
        source_gid_chunks : list
            List of source gid chunks to collect.
        """

        all_source_gids = f_source.get_meta_arr('gid')
        mem_req = (len(all_source_gids) * self._site_mem_req)

        if mem_req > self._mem_avail:
            n = 2
            while True:
                source_gid_chunks = np.array_split(all_source_gids, n)
                new_mem_req = (len(source_gid_chunks[0]) * self._site_mem_req)
                if new_mem_req > self._mem_avail:
                    n += 1
                else:
                    logger.debug('Collecting dataset "{}" in {} chunks with '
                                 'an estimated {} bytes in each chunk '
                                 '(mem avail limit is {} bytes).'
                                 .format(self._dset_in, n, new_mem_req,
                                         self._mem_avail))
                    break
        else:
            source_gid_chunks = [all_source_gids]

        return all_source_gids, source_gid_chunks

    def _collect_chunk(self, all_source_gids, source_gids, f_out,
                       f_source, fp_source):
        """Collect one set of source gids from f_source to f_out.

        Parameters
        ----------
        all_source_gids : list
            List of all source gids to be collected
        source_gids : np.ndarray | list
            Source gids to be collected
        f_out : reV.handlers.outputs.Output
            Output file handler
        f_source : reV.handlers.outputs.Output
            Source file handler
        fp_source : str
            Source filepath
        """
        out_slice = self._get_gid_slice(self._gids, source_gids,
                                        os.path.basename(fp_source))

        source_i0 = np.where(all_source_gids == np.min(source_gids))[0][0]
        source_i1 = np.where(all_source_gids == np.max(source_gids))[0][0]
        source_slice = slice(source_i0, source_i1 + 1)
        source_indexer = np.isin(source_gids, self._gids)

        logger.debug('\t- Running low mem collection of "{}" for '
                     'output site {} from source site {} and file : {}'
                     .format(self._dset_in, out_slice, source_slice,
                             os.path.basename(fp_source)))

        try:
            if self._axis == 1:
                data = f_source[self._dset_in, source_slice]
                if not all(source_indexer):
                    data = data[source_indexer]
                f_out[self._dset_out, out_slice] = data

            elif self._axis == 2:
                data = f_source[self._dset_in, :, source_slice]
                if not all(source_indexer):
                    data = data[:, source_indexer]
                f_out[self._dset_out, :, out_slice] = data

        except Exception as e:
            logger.exception('Failed to collect source file {}. '
                             'Raised the following exception:\n{}'
                             .format(os.path.basename(fp_source), e))
            raise e

    def _collect(self):
        """Simple & robust serial collection optimized for low memory usage."""
        with Outputs(self._h5_file, mode='a') as f_out:
            for fp in self._source_files:
                with Outputs(fp, mode='r') as f_source:

                    x = self._get_source_gid_chunks(f_source)
                    all_source_gids, source_gid_chunks = x

                    for source_gids in source_gid_chunks:
                        self._collect_chunk(all_source_gids, source_gids,
                                            f_out, f_source, fp)

                log_mem(logger, log_level='DEBUG')

    @classmethod
    def collect_dset(cls, h5_file, source_files, gids, dset_in, dset_out=None,
                     mem_util_lim=0.7):
        """Collect a single dataset from a list of source files into a final
        output file.

        Parameters
        ----------
        h5_file : str
            Path to h5_file into which dataset is to be collected
        source_files : list
            List of source filepaths.
        gids : list
            list of gids to be collected
        dset_in : str
            Dataset to collect
        dset_out : str
            Dataset into which collected data is to be written
        mem_util_lim : float
            Memory utilization limit (fractional). This sets how many sites
            will be collected at a time.
        """
        dc = cls(h5_file, source_files, gids, dset_in, dset_out=dset_out,
                 mem_util_lim=mem_util_lim)
        dc._collect()


class Collector:
    """
    Class to handle the collection and combination of .h5 files
    """
    def __init__(self, h5_file, h5_dir, project_points, file_prefix=None,
                 clobber=False):
        """
        Parameters
        ----------
        h5_file : str
            Path to .h5 file into which data will be collected
        h5_dir : str
            Root directory containing .h5 files to combine
        project_points : str | slice | list | pandas.DataFrame | None
            Project points that correspond to the full collection of points
            contained in the .h5 files to be collected. None if points list is
            to be ignored (collect all data in h5_files)
        file_prefix : str
            .h5 file prefix, if None collect all files in h5_dir
        clobber : bool
            Flag to purge .h5 file if it already exists
        """
        if clobber:
            if os.path.isfile(h5_file):
                warn('{} already exists and is being replaced'.format(h5_file),
                     CollectionWarning)
                os.remove(h5_file)

        self._h5_out = h5_file
        ignore = os.path.basename(self._h5_out)
        self._h5_files = self.find_h5_files(h5_dir, file_prefix=file_prefix,
                                            ignore=ignore)
        if project_points is not None:
            self._gids = self.parse_project_points(project_points)
        else:
            self._gids = self.parse_gids_from_files(self._h5_files)

        self.combine_meta()

    @staticmethod
    def find_h5_files(h5_dir, file_prefix=None, ignore=None):
        """
        Search h5_dir for .h5 file, return sorted
        If file_prefix is not None, only return .h5 files with given prefix

        Parameters
        ----------
        h5_dir : str
            Root directory to search
        file_prefix : str
            Prefix for .h5 file in h5_dir, if None return all .h5 files
        ignore : str | list | NoneType
            File name(s) to ignore.
        """
        if not isinstance(ignore, list):
            ignore = [ignore]
        h5_files = []
        logger.debug('Looking for source files in {}'.format(h5_dir))
        for file in os.listdir(h5_dir):
            if file.endswith('.h5'):
                if file_prefix is not None:
                    if file.startswith(file_prefix) and file not in ignore:
                        logger.debug('\t- Found source file to collect: {}'
                                     .format(file))
                        h5_files.append(os.path.join(h5_dir, file))
                elif file not in ignore:
                    logger.debug('\t- Found source file to collect: {}'
                                 .format(file))
                    h5_files.append(os.path.join(h5_dir, file))
        h5_files = sorted(h5_files)
        logger.debug('Final list of {} source files: {}'
                     .format(len(h5_files), h5_files))
        return h5_files

    @staticmethod
    def parse_project_points(project_points):
        """
        Extract resource gids from project points

        Parameters
        ----------
        project_points : str | slice | list | pandas.DataFrame
            Reference to resource points that were processed and need
            collecting

        Returns
        -------
        gids : list
            List of resource gids that are to be collected
        """
        if isinstance(project_points, str):
            gids = pd.read_csv(project_points)['gid'].values
        elif isinstance(project_points, pd.DataFrame):
            gids = project_points['gid'].values
        elif isinstance(project_points, list):
            gids = project_points
        elif isinstance(project_points, slice):
            s = project_points.start
            if s is None:
                s = 0

            e = project_points.stop
            if e is None:
                m = "slice must be bounded!"
                logger.error(m)
                raise CollectionValueError(m)

            step = project_points.step
            if step is None:
                step = 1

            gids = list(range(s, e, step))
        else:
            m = 'Cannot parse project_points'
            logger.error(m)
            raise CollectionValueError(m)
        gids = sorted([int(g) for g in gids])
        return gids

    @staticmethod
    def parse_gids_from_files(h5_files):
        """
        Extract a sorted gid list from a list of h5_files.

        Parameters
        ----------
        h5_files : list
            List of h5 files to be collected.

        Returns
        -------
        gids : list
            List of sorted resource gids to be collected.
        """

        meta = [DatasetCollector.parse_meta(file) for file in h5_files]
        meta = pd.concat(meta, axis=0)
        gids = list(set(meta['gid'].values.tolist()))
        gids = sorted([int(g) for g in gids])
        return gids

    def get_dset_shape(self, dset_name):
        """
        Extract the dataset shape from the first file in the collection list.

        Parameters
        ----------
        dset_name : str
            Dataset to be collected whose shape is in question.

        Returns
        -------
        shape : tuple
            Dataset shape tuple.
        """
        with Outputs(self.h5_files[0], mode='r') as f:
            shape, _, _ = f.get_dset_properties(dset_name)

        return shape

    @property
    def h5_files(self):
        """
        List of .h5 files to be combined

        Returns
        -------
        list
        """
        return self._h5_files

    @property
    def gids(self):
        """
        List of gids corresponding to all sites to be combined

        Returns
        -------
        list
        """
        return self._gids

    def combine_time_index(self):
        """
        Extract time_index, None if not present in .h5 files
        """
        with Outputs(self.h5_files[0], mode='r') as f:
            if 'time_index' in f.datasets:
                time_index = f.time_index
                attrs = f.get_attrs('time_index')
            else:
                time_index = None
                warn("'time_index' was not processed as it is not "
                     "present in .h5 files to be combined.",
                     CollectionWarning)

        if time_index is not None:
            with Outputs(self._h5_out, mode='a') as f:
                f._set_time_index('time_index', time_index, attrs=attrs)

    def _check_meta(self, meta):
        """
        Check combined meta against self._gids to make sure all sites
        are present in self._h5_files

        Parameters
        ----------
        meta : pandas.DataFrame
            DataFrame of combined meta from all files in self._h5_files

        Parameters
        ----------
        meta : pandas.DataFrame
            DataFrame of combined meta from all files in self._h5_files.
            Duplicate GIDs are dropped and a warning is raised.
        """
        meta_gids = meta['gid'].values
        gids = np.array(self.gids)
        missing = gids[~np.in1d(gids, meta_gids)]
        if any(missing):
            # TODO: Write missing gids to disk to allow for automated re-run
            m = "gids: {} are missing".format(missing)
            logger.error(m)
            raise CollectionRuntimeError(m)

        if len(set(meta_gids)) != len(meta):
            m = ('Meta of length {} has {} unique gids! '
                 'There are duplicate gids in the source file list: {}'
                 .format(len(meta), len(set(meta_gids)), self.h5_files))
            logger.warning(m)
            warn(m, CollectionWarning)
            meta = meta.drop_duplicates(subset='gid', keep='last')

        meta = meta.sort_values('gid')
        meta = meta.reset_index(drop=True)

        return meta

    def _purge_chunks(self):
        """Remove the chunked files (after collection). Will not delete files
        if any datasets were not collected."""

        with Outputs(self._h5_out, mode='r') as out:
            dsets_collected = out.datasets
        with Outputs(self.h5_files[0], mode='r') as out:
            dsets_source = out.datasets

        missing = [d for d in dsets_source if d not in dsets_collected]

        if any(missing):
            w = ('Not purging chunked output files. These dsets '
                 'have not been collected: {}'.format(missing))
            warn(w, CollectionWarning)
            logger.warning(w)
        else:
            for fpath in self.h5_files:
                os.remove(fpath)

    def _move_chunks(self, sub_dir):
        """Move the chunked files to a sub dir (after collection).

        Parameters
        ----------
        sub_dir : str | None
            Sub directory name to move chunks to. None to not move files.
        """

        if sub_dir is not None:
            for fpath in self.h5_files:
                base_dir, fn = os.path.split(fpath)
                new_dir = os.path.join(base_dir, sub_dir)
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                new_fpath = os.path.join(new_dir, fn)
                shutil.move(fpath, new_fpath)

    def combine_meta(self):
        """
        Load and combine meta data from .h5
        """
        with Outputs(self._h5_out, mode='a') as f:
            if 'meta' in f.datasets:
                self._check_meta(f.meta)
            else:
                with Outputs(self.h5_files[0], mode='r') as f_in:
                    global_attrs = f_in.get_attrs()
                    meta_attrs = f_in.get_attrs('meta')

                for key, value in global_attrs.items():
                    f._h5.attrs[key] = value

                meta = [DatasetCollector.parse_meta(file)
                        for file in self.h5_files]

                meta = pd.concat(meta, axis=0)
                meta = self._check_meta(meta)
                logger.info('Writing meta data with shape {}'
                            .format(meta.shape))

                f._set_meta('meta', meta, attrs=meta_attrs)

    @classmethod
    def collect(cls, h5_file, h5_dir, project_points, dset_name, dset_out=None,
                file_prefix=None, mem_util_lim=0.7):
        """
        Collect dataset from h5_dir to h5_file

        Parameters
        ----------
        h5_file : str
            Path to .h5 file into which data will be collected
        h5_dir : str
            Root directory containing .h5 files to combine
        project_points : str | slice | list | pandas.DataFrame | None
            Project points that correspond to the full collection of points
            contained in the .h5 files to be collected. None if points list is
            to be ignored (collect all data in h5_files)
        dset_name : str
            Dataset to be collected. If source shape is 2D, time index will be
            collected.
        dset_out : str
            Dataset to collect means into
        file_prefix : str
            .h5 file prefix, if None collect all files on h5_dir
        mem_util_lim : float
            Memory utilization limit (fractional). This sets how many sites
            will be collected at a time.
        """
        if file_prefix is None:
            h5_files = "*.h5"
        else:
            h5_files = "{}*.h5".format(file_prefix)

        logger.info('Collecting dataset "{}" from {} files in {} to {}'
                    .format(dset_name, h5_files, h5_dir, h5_file))
        ts = time.time()
        clt = cls(h5_file, h5_dir, project_points, file_prefix=file_prefix,
                  clobber=True)
        logger.debug("\t- 'meta' collected")

        dset_shape = clt.get_dset_shape(dset_name)
        if len(dset_shape) > 1:
            clt.combine_time_index()
            logger.debug("\t- 'time_index' collected")

        DatasetCollector.collect_dset(clt._h5_out, clt.h5_files, clt.gids,
                                      dset_name, dset_out=dset_out,
                                      mem_util_lim=mem_util_lim)

        logger.debug("\t- Collection of '{}' complete".format(dset_name))

        tt = (time.time() - ts) / 60
        logger.info('Collection complete')
        logger.debug('\t- Collection took {:.4f} minutes'
                     .format(tt))

    @classmethod
    def add_dataset(cls, h5_file, h5_dir, dset_name, dset_out=None,
                    file_prefix=None, mem_util_lim=0.7):
        """
        Collect and add dataset to h5_file from h5_dir

        Parameters
        ----------
        h5_file : str
            Path to .h5 file into which data will be collected
        h5_dir : str
            Root directory containing .h5 files to combine
        dset_name : str
            Dataset to be collected. If source shape is 2D, time index will be
            collected.
        dset_out : str
            Dataset to collect means into
        file_prefix : str
            .h5 file prefix, if None collect all files on h5_dir
        mem_util_lim : float
            Memory utilization limit (fractional). This sets how many sites
            will be collected at a time.
        """
        if file_prefix is None:
            h5_files = "*.h5"
        else:
            h5_files = "{}*.h5".format(file_prefix)

        logger.info('Collecting "{}" from {} files in {} and adding to {}'
                    .format(dset_name, h5_files, h5_dir, h5_file))
        ts = time.time()
        with Outputs(h5_file, mode='r') as f:
            points = f.meta

        clt = cls(h5_file, h5_dir, points, file_prefix=file_prefix)

        dset_shape = clt.get_dset_shape(dset_name)
        if len(dset_shape) > 1:
            clt.combine_time_index()
            logger.debug("\t- 'time_index' collected")

        DatasetCollector.collect_dset(clt._h5_out, clt.h5_files, clt.gids,
                                      dset_name, dset_out=dset_out,
                                      mem_util_lim=mem_util_lim)

        logger.debug("\t- Collection of '{}' complete".format(dset_name))

        tt = (time.time() - ts) / 60
        logger.info('{} collected'.format(dset_name))
        logger.debug('\t- Collection took {:.4f} minutes'
                     .format(tt))

    @classmethod
    def purge_chunks(cls, h5_file, h5_dir, project_points, file_prefix=None):
        """
        Purge (remove) chunked files from h5_dir (after collection).

        Parameters
        ----------
        h5_file : str
            Path to .h5 file into which data will be collected
        h5_dir : str
            Root directory containing .h5 files to combine
        project_points : str | slice | list | pandas.DataFrame
            Project points that correspond to the full collection of points
            contained in the .h5 files to be collected
        file_prefix : str
            .h5 file prefix, if None collect all files on h5_dir
        """

        clt = cls(h5_file, h5_dir, project_points, file_prefix=file_prefix)
        clt._purge_chunks()
        logger.info('Purged chunk files from {}'.format(h5_dir))

    @classmethod
    def move_chunks(cls, h5_file, h5_dir, project_points, file_prefix=None,
                    sub_dir='chunk_files'):
        """
        Move chunked files from h5_dir (after collection) to subdir.

        Parameters
        ----------
        h5_file : str
            Path to .h5 file into which data will be collected
        h5_dir : str
            Root directory containing .h5 files to combine
        project_points : str | slice | list | pandas.DataFrame
            Project points that correspond to the full collection of points
            contained in the .h5 files to be collected
        file_prefix : str
            .h5 file prefix, if None collect all files on h5_dir
        sub_dir : str | None
            Sub directory name to move chunks to. None to not move files.
        """

        clt = cls(h5_file, h5_dir, project_points, file_prefix=file_prefix)
        clt._move_chunks(sub_dir)
        logger.info('Moved chunk files from {} to sub_dir: {}'
                    .format(h5_dir, sub_dir))
