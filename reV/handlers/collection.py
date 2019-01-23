"""
Base class to handle collection of profiles and means across multiple .h5 files
"""
import logging
import numpy as np
import os
import pandas as pd

from reV.utilities.execution import SuperParallelJob
from rev.handler.outputs import Outputs
from reV.utilities.exceptions import (HandlerRuntimeError, HandlerValueError)
from reV.utilities.execution import execute_futures

logger = logging.getLogger(__name__)


class DatasetCollector:
    """
    Class to use with SmartParallelJob to handle dataset collection from
    .h5 files
    """
    def __init__(self, handler, gids, dset_in, dset_out=None):
        """
        Parameters
        ----------
        handler : Instance of Outputs
            Resource class to handle writing of collected dataset
        gids : list
            list of gids to be collected
        dset_in : str
            Dataset to collect
        dset_out : str
            Dataset into which collected data is to be written
        """
        self._handler = handler
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
        if self._out is not None:
            gids = []
            out = []
            for ids, out in self._out:
                gids.extend(ids)
                out.append(out)

            out = np.concatenate(out)
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
            gids = f.meta['gids'].values
            dset_out = f[self._dset_in]

        return gids, dset_out

    def flush(self):
        """
        Base flush method needed for SmartParallelJob
        """
        dset_slice, dset_arr = self.out
        keys = (self._dset_out,) + dset_slice
        self._handler.__setitem__(keys, dset_arr)

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

        dset_cls.flush()


class Collector(Outputs):
    """
    Class to handle the collection and combination of .h5 files
    """
    def __init__(self, h5_file, h5_dir, project_points, file_prefix=None,
                 parallel=False):
        """
        Parameters
        ----------
        h5_dir : str
            Root directory containing .h5 files to combine
        project_points : str | slice | list | pandas.DataFrame
            Project points that correspond to the full collection of points
            contained in the .h5 files to be collected
        file_prefix : str
            .h5 file prefix, if None collect all files on h5_dir
        parallel : bool
            Option to run in parallel using dask
        handler_cls : Resource class or sub-class
            Class to handle loading of .h5 data
        """
        super().__init__(self, h5_file, mode='a')
        self._h5_files = self.find_h5_files(h5_dir, file_prefix=file_prefix)
        self._gids = self.parse_project_points(project_points)
        self._parallel = parallel
        self.combine_meta()

    @staticmethod
    def find_h5_files(h5_dir, file_prefix=None):
        """
        Search h5_dir for .h5 file, return sorted
        If file_prefix is not None, only return .h5 files with given prefix

        Parameters
        ----------
        h5_dir : str
            Root directory to search
        file_prefix : str
            Prefix for .h5 file in h5_dir, if None return all .h5 files
        """
        h5_files = []
        for file in os.listdir(h5_dir):
            if file.endswith('.h5'):
                if file_prefix is not None:
                    if file.startswith(file_prefix):
                        h5_files.append(os.path.join(h5_dir, file))
                else:
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

    def parse_meta(self, h5_file):
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

        Returns
        -------
        _time_index : pandas.DatetimeIndex
            Datetimestamps associated with profiles to be combined
        """
        with Outputs(self.h5_files[0], mode='r') as f:
            if 'time_index' in f.dsets:
                time_index = f.time_index
                self.time_index = time_index

    @property
    def time_index(self):
        """
        Extract time_index, None if not present in .h5 files

        Returns
        -------
        _time_index : pandas.DatetimeIndex
            Datetimestamps associated with profiles to be combined
        """
        if not hasattr(self, '_time_index'):
            with self._handler_cls(self._h5_files[0]) as res:
                if 'time_index' in res.dsets:
                    time_index = res.time_index
                else:
                    time_index = None

            self._time_index = time_index

        return self._time_index

    def _check_meta(self, meta):
        """
        Check combined meta against self._gids to make sure all sites
        are present in self._h5_files

        Parameters
        ----------
        meta : pandas.DataFrame
            DataFrame of combined meta from all files in self._h5_files
        """
        meta_gids = meta['gids'].values
        if not np.array_equal(meta_gids, self.gids):
            gids = np.array(self.gids)
            missing = gids[~np.in1d(gids, meta_gids)]
            raise HandlerRuntimeError("gids: {} are missing"
                                      .format(missing))

    def combine_meta(self):
        """
        Load and combine meta data from .h5

        Returns
        -------
        meta : pandas.DataFrame
            DataFrame of combined meta
        """
        if self._parallel:
            meta = execute_futures(self.parse_meta, self.h5_files)
        else:
            meta = [self.parse_meta(file) for file in self.h5_files]

        meta = pd.concat(meta, axis=0)
        self._check_meta(meta)
        self.meta = meta

        return meta

    def combine_dset(self, dset, dset_shape, dset_out=None):
        """
        Collect, combine and save dset to disk

        Parameters
        ----------
        dset : str
            dataset to collect
        dset_shape : tuple
            Shape of final collected dset array
        dset_out : str
            name of dataset to be saved to disk, if None use dset
        """
        with Outputs(self.h5_files[0], mode='r') as f:
            _, dtype, chunks = f.get_dset_properties(dset)
            attrs = f.get_attrs(dset)

        self._h5.create_ds(dset, dset_shape, dtype, chunks=chunks,
                           attrs=attrs)
        if self._parallel:
            dset_collector = DatasetCollector(self._h5, self.gids, dset,
                                              dset_out=dset_out)
            SuperParallelJob.execute(dset_collector, self.h5_files)
        else:
            DatasetCollector.collect(self._h5, self.gids, dset, self.h5_files,
                                     dset_out=dset_out)
