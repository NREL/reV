# -*- coding: utf-8 -*-
"""
Base class to handle collection of profiles and means across multiple .h5 files
"""
import logging
import numpy as np
import os
import pandas as pd
import time
from warnings import warn

from reV.handlers.outputs import Outputs
from reV.utilities.exceptions import (HandlerRuntimeError, HandlerValueError,
                                      HandlerWarning)
from reV.utilities.execution import execute_parallel, SmartParallelJob

logger = logging.getLogger(__name__)


class DatasetCollector:
    """
    Class to use with SmartParallelJob to handle dataset collection from
    .h5 files
    """
    def __init__(self, h5_file, gids, dset_in, dset_out=None):
        """
        Parameters
        ----------
        h5_file : str
            Path to h5_file into which dataset is to be collected
        gids : list
            list of gids to be collected
        dset_in : str
            Dataset to collect
        dset_out : str
            Dataset into which collected data is to be written
        """
        self._h5_file = h5_file
        self._gids = gids
        self._dset_in = dset_in
        if dset_out is None:
            dset_out = dset_in

        self._dset_out = dset_out
        self._out = None

    @property
    def out(self):
        """
        Extract output from futures

        Returns
        -------
        dset_slice : slice
            Slice of dataset that corresponds to dset_out
        dset_out : ndarray
            Array of collected output for dset_in
        """
        out = None
        out_slice = None
        if self._out is not None:
            gids = []
            out = []
            for ids, arr in self._out:
                gids.extend(ids)
                out.append(arr)

            out = np.hstack(out)
            out_slice = self._get_slice(gids, axis=len(out.shape))

        return out_slice, out

    @out.setter
    def out(self, results):
        """
        Add results from SmartParallelJob to out

        Parameters
        ----------
        results : list
            Results extracted by SmartParallelJob
        """
        if isinstance(results, list) and self._out is not None:
            self._out.extend(results)
        else:
            self._out = results

    def _get_slice(self, gids, axis):
        """
        Determine to which slice of the dataset gids corresponds.
        Confirm that gids is an inclusive list covering the slice.

        Parameters
        ----------
        gids : list
            List of gids collected

        Returns
        -------
        gid_slice : slice
            slice of full gids to be collected
        """
        pos = np.in1d(self._gids, gids)
        if np.all(pos):
            gid_slice = slice(None, None, None)
        else:
            start, end = np.where()[0][[0, -1]]
            gid_slice = slice(start, end + 1, None)

            if not np.array_equal(self._gids[gid_slice], gids):
                raise HandlerValueError('gids are not a valid slice of full '
                                        'set of _gids')
        if axis == 1:
            out_slice = (gid_slice,)
        else:
            out_slice = (slice(None, None, None), gid_slice)

        return out_slice

    def run(self, h5_file):
        """
        Base run method need for SmartParallelJob

        Parameters
        ----------
        h5_file : str
            .h5 file path for file to be run
        """
        with Outputs(h5_file, mode='r', unscale=False) as f:
            gids = f.meta['gid'].values
            dset_out = f[self._dset_in]

        return gids, dset_out

    def flush(self):
        """
        Base flush method needed for SmartParallelJob
        """
        dset_slice, dset_arr = self.out
        keys = (self._dset_out,) + dset_slice
        with Outputs(self._h5_file, mode='a') as f:
            f.__setitem__(keys, dset_arr)

    @classmethod
    def collect(cls, handler, gids, dset_in, h5_files, dset_out=None):
        """
        Extract dset_in from h5_files and write to disk

        Parameters
        ----------
        handler : Instance of Outputs
            Resource class to handle writing of collected dataset
        gids : list
            list of gids to be collected
        dset_in : str
            Dataset to collect
        h5_files : list
            h5 files from which dset is to be collected
        dset_out : str
            Dataset into which collected data is to be written
        """
        dset_cls = cls(handler, gids, dset_in, dset_out=dset_out)
        futures = []
        for file in h5_files:
            futures.append(dset_cls.run(file))

        dset_cls.out = futures
        dset_cls.flush()


class Collector:
    """
    Class to handle the collection and combination of .h5 files
    """
    def __init__(self, h5_file, h5_dir, project_points, file_prefix=None,
                 parallel=False, clobber=False):
        """
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
            .h5 file prefix, if None collect all files in h5_dir
        parallel : bool
            Option to run in parallel
        clobber : bool
            Flag to purge .h5 file if it already exists
        """
        if clobber:
            if os.path.isfile(h5_file):
                warn('{} already exists and is being replaced'.format(h5_file),
                     HandlerWarning)
                os.remove(h5_file)

        self._h5_out = h5_file
        ignore = os.path.basename(self._h5_out)
        self._h5_files = self.find_h5_files(h5_dir, file_prefix=file_prefix,
                                            ignore=ignore)
        self._gids = self.parse_project_points(project_points)
        self._parallel = parallel
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
        for file in os.listdir(h5_dir):
            if file.endswith('.h5'):
                if file_prefix is not None:
                    if file.startswith(file_prefix) and file not in ignore:
                        h5_files.append(os.path.join(h5_dir, file))
                elif file not in ignore:
                    h5_files.append(os.path.join(h5_dir, file))

        return sorted(h5_files)

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
                raise HandlerValueError("slice must be bounded!")

            step = project_points.step
            if step is None:
                step = 1

            gids = list(range(s, e, step))
        else:
            raise HandlerValueError('Cannot parse project_points')

        return gids

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
            if 'time_index' in f.dsets:
                time_index = f.time_index
            else:
                time_index = None
                warn("'time_index' was not processed as it is not "
                     "present in .h5 files to be combined.",
                     HandlerWarning)

        if time_index is not None:
            with Outputs(self._h5_out, mode='a') as f:
                f.time_index = time_index

    def _check_meta(self, meta):
        """
        Check combined meta against self._gids to make sure all sites
        are present in self._h5_files

        Parameters
        ----------
        meta : pandas.DataFrame
            DataFrame of combined meta from all files in self._h5_files
        """
        meta_gids = meta['gid'].values
        gids = np.array(self.gids)
        missing = gids[~np.in1d(gids, meta_gids)]
        if any(missing):
            # TODO: Convert HandlerRuntimeError to a custom collection error
            # TODO: Write missing gids to disk to allow for automated re-run
            raise HandlerRuntimeError("gids: {} are missing"
                                      .format(missing))

    def _purge_chunks(self):
        """Remove the chunked files (after collection). Will not delete files
        if any datasets were not collected."""

        with Outputs(self._h5_out, mode='r') as out:
            dsets_collected = out.dsets
        with Outputs(self.h5_files[0], mode='r') as out:
            dsets_source = out.dsets

        missing = [d for d in dsets_source if d not in dsets_collected]

        if any(missing):
            w = ('Not purging chunked output files. These dsets '
                 'have not been collected: {}'.format(missing))
            warn(w, HandlerWarning)
            logger.warning(w)
        else:
            for fpath in self.h5_files:
                os.remove(fpath)

    def combine_meta(self):
        """
        Load and combine meta data from .h5
        """
        with Outputs(self._h5_out, mode='a') as f:
            if 'meta' in f.dsets:
                self._check_meta(f.meta)
            else:
                if self._parallel:
                    meta = execute_parallel(self.parse_meta, self.h5_files)
                else:
                    meta = [self.parse_meta(file) for file in self.h5_files]

                meta = pd.concat(meta, axis=0)
                self._check_meta(meta)
                f.meta = meta

    def _pre_collect(self, dset, dset_out=None):
        """Run a pre-collection check and get relevant dset attrs.

        Parameters
        ----------
        dset : str
            dataset to collect
        dset_out : str
            name of dataset to be saved to disk, if None use dset

        Returns
        -------
        dset_out : str
            name of dataset to be saved to disk
        attrs : dict
            Dset attribute of dset from source files.
        axis : int
            Axis size (1 is 1D array, 2 is 2D array)
        """
        if dset_out is None:
            dset_out = dset
        with Outputs(self.h5_files[0], mode='r') as f:
            _, dtype, chunks = f.get_dset_properties(dset)
            attrs = f.get_attrs(dset)
            axis = len(f[dset].shape)

        with Outputs(self._h5_out, mode='a') as f:
            if axis == 1:
                dset_shape = (len(f),)
            elif axis == 2:
                if 'time_index' in f.dsets:
                    dset_shape = f.shape
                else:
                    raise HandlerRuntimeError("'time_index' must be combined "
                                              "before profiles can be "
                                              "combined.")
            else:
                raise HandlerRuntimeError('Cannot collect dset "{}" with '
                                          'axis {}'.format(dset, axis))
            if dset_out not in f.dsets:
                f._create_dset(dset_out, dset_shape, dtype, chunks=chunks,
                               attrs=attrs)
        return dset_out, attrs, axis

    def combine_dset(self, dset, dset_out=None):
        """
        Collect, combine and save dset to disk

        Parameters
        ----------
        dset : str
            dataset to collect
        dset_out : str
            name of dataset to be saved to disk, if None use dset
        """

        dset_out, _, _ = self._pre_collect(dset, dset_out=dset_out)

        if self._parallel:
            dset_collector = DatasetCollector(self._h5_out, self.gids, dset,
                                              dset_out=dset_out)
            SmartParallelJob.execute(dset_collector, self.h5_files)
        else:
            DatasetCollector.collect(self._h5_out, self.gids, dset,
                                     self.h5_files, dset_out=dset_out)

    def low_mem_collect(self, dset, dset_out=None):
        """Simple and robust serial collection optimized for low memory usage

        Parameters
        ----------
        dset : str
            dataset to collect
        dset_out : str
            name of dataset to be saved to disk, if None use dset
        """

        dset_out, _, axis = self._pre_collect(dset, dset_out=dset_out)

        with Outputs(self._h5_out, mode='a') as f_out:
            for fp in self.h5_files:

                with Outputs(fp, mode='r') as f_source:
                    source_gids = f_source.get_meta_arr('gid')
                    locs = np.where(np.isin(self.gids, source_gids))[0]
                    locs = slice(locs.min(), locs.max() + 1)
                    if axis == 1:
                        f_out[dset_out, locs] = f_source[dset]
                    elif axis == 2:
                        f_out[dset_out, slice(None), locs] = f_source[dset]

                logger.debug('Low memory collection of "{}" complete '
                             'from source: {}'.format(dset, fp))

    @classmethod
    def collect(cls, h5_file, h5_dir, project_points, dset_name,
                dset_out=None, file_prefix=None, parallel=False,
                low_mem=True):
        """
        Collect dataset from h5_dir to h5_file

        Parameters
        ----------
        h5_file : str
            Path to .h5 file into which data will be collected
        h5_dir : str
            Root directory containing .h5 files to combine
        project_points : str | slice | list | pandas.DataFrame
            Project points that correspond to the full collection of points
            contained in the .h5 files to be collected
        dset_name : str
            Dataset to be collected. If source shape is 2D, time index will be
            collected.
        dset_out : str
            Dataset to collect means into
        file_prefix : str
            .h5 file prefix, if None collect all files on h5_dir
        parallel : bool
            Option to run in parallel
        low_mem : bool
            Flag to run serial low memory collection (overrides parallel)
        """
        if file_prefix is None:
            h5_files = "*.h5"
        else:
            h5_files = "{}*.h5".format(file_prefix)

        logger.info('Collecting dataset "{}" from {} files in {} to {}'
                    .format(dset_name, h5_files, h5_dir, h5_file))
        ts = time.time()
        clt = cls(h5_file, h5_dir, project_points, file_prefix=file_prefix,
                  parallel=parallel, clobber=True)
        logger.debug("\t- 'meta' collected")

        dset_shape = clt.get_dset_shape(dset_name)
        if len(dset_shape) > 1:
            clt.combine_time_index()
            logger.debug("\t- 'time_index' collected")

        if low_mem:
            clt.low_mem_collect(dset_name, dset_out=dset_out)
        else:
            clt.combine_dset(dset_name, dset_out=dset_out)
        logger.debug("\t- '{}' collected".format(dset_name))

        tt = (time.time() - ts) / 60
        logger.info('Collection complete')
        logger.debug('\t- Colletion took {:.4f} minutes'
                     .format(tt))

    @classmethod
    def add_dataset(cls, h5_file, h5_dir, dset_name, dset_out=None,
                    file_prefix=None, parallel=False, low_mem=True):
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
        parallel : bool
            Option to run in parallel
        low_mem : bool
            Flag to run serial low memory collection (overrides parallel)
        """
        if file_prefix is None:
            h5_files = "*.h5"
        else:
            h5_files = "{}*.h5".format(file_prefix)

        logger.info('Collecting "{}" from {} files in {} and adding to {}'
                    .format(dset_name, h5_files, h5_dir, h5_file))
        ts = time.time()
        with Outputs(h5_file, mode='a') as f:
            points = f.meta

        clt = cls(h5_file, h5_dir, points, file_prefix=file_prefix,
                  parallel=parallel)

        dset_shape = clt.get_dset_shape(dset_name)
        if len(dset_shape) > 1:
            clt.combine_time_index()
            logger.debug("\t- 'time_index' collected")

        if low_mem:
            clt.low_mem_collect(dset_name, dset_out=dset_out)
        else:
            clt.combine_dset(dset_name, dset_out=dset_out)

        logger.debug("\t- '{}' collected".format(dset_name))

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
