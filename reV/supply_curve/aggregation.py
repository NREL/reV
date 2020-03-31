# -*- coding: utf-8 -*-
"""
reV aggregation framework.
"""
from abc import ABC, abstractmethod
from concurrent.futures import as_completed
import os
import h5py
import numpy as np
import logging

from reV.handlers.outputs import Outputs
from reV.supply_curve.exclusions import ExclusionMaskFromDict
from reV.supply_curve.points import SupplyCurveExtent, SupplyCurvePoint
from reV.utilities.execution import SpawnProcessPool
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      FileInputError)


logger = logging.getLogger(__name__)


class AbstractAggFileHandler(ABC):
    """Simple framework to handle aggregation file context managers."""

    def __init__(self, excl_fpath, excl_dict=None, area_filter_kernel='queen',
                 min_area=None, check_excl_layers=False):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | None
            Minimum required contiguous area filter in sq-km
        check_excl_layers : bool
            Run a pre-flight check on each exclusion layer to ensure they
            contain un-excluded values
        """

        self._excl_fpath = excl_fpath
        self._excl = ExclusionMaskFromDict(excl_fpath, layers_dict=excl_dict,
                                           min_area=min_area,
                                           kernel=area_filter_kernel,
                                           check_layers=check_excl_layers)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if type is not None:
            raise

    @abstractmethod
    def close(self):
        """Close all file handlers."""
        self._excl.close()

    @property
    def exclusions(self):
        """Get the exclusions file handler object.

        Returns
        -------
        _excl : ExclusionMask
            Exclusions h5 handler object.
        """
        return self._excl

    @property
    def h5(self):
        "Placeholder for h5 property in AggFileHandler"
        pass


class AggFileHandler(AbstractAggFileHandler):
    """Simple framework to handle aggregation file context managers."""

    def __init__(self, excl_fpath, h5_fpath, excl_dict=None,
                 area_filter_kernel='queen', min_area=None,
                 check_excl_layers=False):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        h5_fpath : str
            Filepath to .h5 file to be aggregated
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | None
            Minimum required contiguous area filter in sq-km
        check_excl_layers : bool
            Run a pre-flight check on each exclusion layer to ensure they
            contain un-excluded values
        """
        super().__init__(excl_fpath, excl_dict=excl_dict,
                         area_filter_kernel=area_filter_kernel,
                         min_area=min_area,
                         check_excl_layers=check_excl_layers)

        self._h5 = Outputs(h5_fpath, mode='r')

    @property
    def h5(self):
        """
        Get the h5 file handler object.

        Returns
        -------
        _h5 : Outputs
            reV h5 outputs handler object.
        """
        return self._h5

    def close(self):
        """Close all file handlers."""
        self._excl.close()
        self._h5.close()


class AbstractAggregation(ABC):
    """Abstract supply points aggregation framework."""

    def __init__(self, excl_fpath, tm_dset, excl_dict=None,
                 area_filter_kernel='queen', min_area=None,
                 check_excl_layers=False, resolution=64, gids=None):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | None
            Minimum required contiguous area filter in sq-km
        check_excl_layers : bool
            Run a pre-flight check on each exclusion layer to ensure they
            contain un-excluded values
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        """

        self._excl_fpath = excl_fpath
        self._tm_dset = tm_dset
        self._excl_dict = excl_dict
        self._resolution = resolution
        self._area_filter_kernel = area_filter_kernel
        self._min_area = min_area
        self._check_excl_layers = check_excl_layers
        if check_excl_layers:
            logger.debug('Exclusions layers will be checked for un-excluded '
                         'values!')

        if gids is None:
            with SupplyCurveExtent(excl_fpath, resolution=resolution) as sc:
                gids = np.array(range(len(sc)), dtype=np.uint32)
        elif not isinstance(gids, np.ndarray):
            gids = np.array(gids)

        self._gids = gids

    @abstractmethod
    def _check_files(self):
        """Do a preflight check on input files"""

        if not os.path.exists(self._excl_fpath):
            raise FileNotFoundError('Could not find required input file: '
                                    '{}'.format(self._excl_fpath))

        with h5py.File(self._excl_fpath, 'r') as f:
            if self._tm_dset not in f:
                raise FileInputError('Could not find techmap dataset "{}" '
                                     'in exclusions file: {}'
                                     .format(self._tm_dset,
                                             self._excl_fpath))

    @staticmethod
    @abstractmethod
    def run_serial(sc_point_method, excl_fpath, tm_dset,
                   excl_dict=None, area_filter_kernel='queen',
                   min_area=None, check_excl_layers=False,
                   resolution=64, gids=None, close=False, args=None,
                   kwargs=None):
        """Standalone method to create agg summary - can be parallelized.

        Parameters
        ----------
        sc_point_method : method
            Supply Curve Point Method to operate on a single SC point.
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        tm_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | None
            Minimum required contiguous area filter in sq-km
        check_excl_layers : bool
            Run a pre-flight check on each exclusion layer to ensure they
            contain un-excluded values
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        close : bool
            Flag to close object file handlers on exit.
        args : list | None
            List of positional args for sc_point_method
        kwargs : dict | None
            Dict of kwargs for sc_point_method

        Returns
        -------
        output : list
            List of output objects from sc_point_method.
        """

        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        output = []

        with SupplyCurveExtent(excl_fpath, resolution=resolution) as sc:
            points = sc.points
            exclusion_shape = sc.exclusions.shape
            if gids is None:
                gids = range(len(sc))

        # pre-extract handlers so they are not repeatedly initialized
        file_kwargs = {'excl_dict': excl_dict,
                       'area_filter_kernel': area_filter_kernel,
                       'min_area': min_area,
                       'check_excl_layers': check_excl_layers}
        with AbstractAggFileHandler(excl_fpath, **file_kwargs) as fh:

            for gid in gids:
                try:
                    gid_out = sc_point_method(
                        gid,
                        fh.exclusions,
                        tm_dset,
                        *args,
                        excl_dict=excl_dict,
                        resolution=resolution,
                        exclusion_shape=exclusion_shape,
                        close=close,
                        **kwargs)

                except EmptySupplyCurvePointError:
                    pass

                else:
                    output.append(gid_out)

        return output

    @abstractmethod
    def run_parallel(self, sc_point_method, args=None, kwargs=None,
                     close=False, max_workers=None, chunk_point_len=1000):
        """
        Aggregate with sc_point_method in parallel

        Parameters
        ----------
        args : list | None
            List of positional args for sc_point_method
        kwargs : dict | None
            Dict of kwargs for sc_point_method
        close : bool
            Flag to close object file handlers on exit.
        max_workers : int | None
            Number of cores to run summary on. None is all
            available cpus.
        chunk_point_len : int
            Number of SC points to process on a single parallel worker.

        Returns
        -------
        summary : list
            List of outputs from sc_point_method.
        """

        chunks = np.array_split(
            self._gids, int(np.ceil(len(self._gids) / chunk_point_len)))

        logger.info('Running supply curve point aggregation for '
                    'points {} through {} at a resolution of {} '
                    'on {} cores in {} chunks.'
                    .format(self._gids[0], self._gids[-1], self._resolution,
                            max_workers, len(chunks)))

        n_finished = 0
        futures = []
        output = []

        with SpawnProcessPool(max_workers=max_workers) as executor:

            # iterate through split executions, submitting each to worker
            for gid_set in chunks:
                # submit executions and append to futures list
                futures.append(executor.submit(
                    self.run_serial,
                    sc_point_method, self._excl_fpath, self._tm_dset,
                    excl_dict=self._excl_dict,
                    area_filter_kernel=self._area_filter_kernel,
                    min_area=self._min_area,
                    check_excl_layers=self._check_excl_layers,
                    resolution=self._resolution,
                    gids=gid_set,
                    close=close,
                    args=args,
                    kwargs=kwargs))

            # gather results
            for future in as_completed(futures):
                n_finished += 1
                logger.info('Parallel aggregation futures collected: '
                            '{} out of {}'
                            .format(n_finished, len(chunks)))
                output += future.result()

        return output

    def aggregate(self, sc_point_method, args=None, kwargs=None,
                  close=False, max_workers=None, chunk_point_len=1000):
        """
        Aggregate with sc_point_method

        Parameters
        ----------
        args : list | None
            List of positional args for sc_point_method
        kwargs : dict | None
            Dict of kwargs for sc_point_method
        close : bool
            Flag to close object file handlers on exit.
        max_workers : int | None
            Number of cores to run summary on. None is all
            available cpus.
        chunk_point_len : int
            Number of SC points to process on a single parallel worker.

        Returns
        -------
        summary : list
            List of outputs from sc_point_method.
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        if max_workers == 1:
            agg = self.run_serial(sc_point_method, self._excl_fpath,
                                  self._tm_dset,
                                  excl_dict=self._excl_dict,
                                  area_filter_kernel=self._area_filter_kernel,
                                  min_area=self._min_area,
                                  check_excl_layers=self._check_excl_layers,
                                  resolution=self._resolution,
                                  gids=self._gids,
                                  close=close, args=args,
                                  kwargs=kwargs)
        else:
            agg = self.run_parallel(sc_point_method, args=args,
                                    kwargs=kwargs, close=close,
                                    max_workers=max_workers,
                                    chunk_point_len=chunk_point_len)

        if not any(agg):
            e = ('Supply curve aggregation found no non-excluded SC points. '
                 'Please check your exclusions or subset SC GID selection.')
            logger.error(e)
            raise EmptySupplyCurvePointError(e)

        return agg

    @classmethod
    def run(cls, excl_fpath, tm_dset, sc_point_method, excl_dict=None,
            area_filter_kernel='queen', min_area=None,
            check_excl_layers=False, resolution=64, gids=None,
            args=None, kwargs=None, close=False, max_workers=None,
            chunk_point_len=1000):
        """Get the supply curve points aggregation summary.

        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        sc_point_method : method
            Supply Curve Point Method to operate on a single SC point.
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | None
            Minimum required contiguous area filter in sq-km
        check_excl_layers : bool
            Run a pre-flight check on each exclusion layer to ensure they
            contain un-excluded values
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        sc_point_method : method
            Supply Curve Point Method to operate on a single SC point.
        args : list | None
            List of positional args for sc_point_method
        kwargs : dict | None
            Dict of kwargs for sc_point_method
        close : bool
            Flag to close object file handlers on exit.
        max_workers : int | None
            Number of cores to run summary on. None is all
            available cpus.
        chunk_point_len : int
            Number of SC points to process on a single parallel worker.

        Returns
        -------
        summary : DataFrame
            Summary of the SC points.
        """

        agg = cls(excl_fpath, tm_dset, excl_dict=excl_dict,
                  area_filter_kernel=area_filter_kernel, min_area=min_area,
                  check_excl_layers=check_excl_layers, resolution=resolution,
                  gids=gids)

        aggregation = agg.aggregate(sc_point_method, args=args, kwargs=kwargs,
                                    close=close, max_workers=max_workers,
                                    chunk_point_len=chunk_point_len)

        return aggregation


class Aggregation(AbstractAggregation):
    """Abstract supply points aggregation framework."""

    def __init__(self, excl_fpath, h5_fpath, tm_dset, *agg_dset,
                 excl_dict=None, area_filter_kernel='queen', min_area=None,
                 check_excl_layers=False, resolution=64, gids=None):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        h5_fpath : str
            Filepath to .h5 file to aggregate
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        agg_dset : str
            Dataset to aggreate, can supply multiple datasets
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | None
            Minimum required contiguous area filter in sq-km
        check_excl_layers : bool
            Run a pre-flight check on each exclusion layer to ensure they
            contain un-excluded values
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        """
        super().__init__(excl_fpath, tm_dset, excl_dict=excl_dict,
                         area_filter_kernel=area_filter_kernel,
                         min_area=min_area,
                         check_excl_layers=check_excl_layers,
                         resolution=resolution, gids=gids)

        self._h5_fpath = h5_fpath
        if isinstance(agg_dset, str):
            agg_dset = (agg_dset, )

        self._agg_dsets = agg_dset

        self._check_files()

    def _check_files(self):
        """Do a preflight check on input files"""

        if not os.path.exists(self._excl_fpath):
            raise FileNotFoundError('Could not find required exclusions file: '
                                    '{}'.format(self._excl_fpath))

        if not os.path.exists(self._h5_fpath):
            raise FileNotFoundError('Could not find required h5 file: '
                                    '{}'.format(self._h5_fpath))

        with h5py.File(self._excl_fpath, 'r') as f:
            if self._tm_dset not in f:
                raise FileInputError('Could not find techmap dataset "{}" '
                                     'in exclusions file: {}'
                                     .format(self._tm_dset,
                                             self._excl_fpath))

        with h5py.File(self._excl_fpath, 'r') as f:
            for dset in self._agg_dsets:
                if dset not in f:
                    raise FileInputError('Could not find provided dataset "{}"'
                                         ' in h5 file: {}'
                                         .format(dset, self._h5_fpath))

    @staticmethod
    def run_serial(excl_fpath, h5_fpath, tm_dset, *agg_dset,
                   agg_method='mean', excl_dict=None,
                   area_filter_kernel='queen', min_area=None,
                   check_excl_layers=False, resolution=64, gids=None,
                   close=False):
        """
        Standalone method to aggregate - can be parallelized.

        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        h5_fpath : str
            Filepath to .h5 file to aggregate
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        agg_dset : str
            Dataset to aggreate, can supply multiple datasets
        agg_method : str
            Aggregation method, either mean or sum/aggregate
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | None
            Minimum required contiguous area filter in sq-km
        check_excl_layers : bool
            Run a pre-flight check on each exclusion layer to ensure they
            contain un-excluded values
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        close : bool
            Flag to close object file handlers on exit.

        Returns
        -------
        agg_out : dict
            Aggregated values for each aggregation dataset
        """
        if agg_method.lower().startswith('mean'):
            agg_method = SupplyCurvePoint.sc_mean
        elif agg_method.lower().startswith(('sum', 'agg')):
            agg_method = SupplyCurvePoint.sc_sum
        else:
            msg = 'Aggregation method must be either mean or sum/aggregate'
            logger.error(msg)
            raise ValueError(msg)

        if isinstance(agg_dset, str):
            agg_dset = (agg_dset, )

        with SupplyCurveExtent(excl_fpath, resolution=resolution) as sc:
            exclusion_shape = sc.exclusions.shape
            if gids is None:
                gids = range(len(sc))

        agg_out = {}

        # pre-extract handlers so they are not repeatedly initialized
        file_kwargs = {'excl_dict': excl_dict,
                       'area_filter_kernel': area_filter_kernel,
                       'min_area': min_area,
                       'check_excl_layers': check_excl_layers}
        with AggFileHandler(excl_fpath, h5_fpath, **file_kwargs) as fh:
            for dset in agg_dset:
                output = []
                ds = fh.h5.open_dataset(dset)
                for gid in gids:
                    try:
                        gid_out = agg_method(gid, fh.exclusions, tm_dset, ds,
                                             excl_dict=excl_dict,
                                             resolution=resolution,
                                             exclusion_shape=exclusion_shape,
                                             close=close)

                    except EmptySupplyCurvePointError:
                        pass

                    else:
                        output.append(gid_out)

        return agg_out

    def run_parallel(self, agg_method='mean', close=False, max_workers=None,
                     chunk_point_len=1000):
        """
        Aggregate in parallel

        Parameters
        ----------
        agg_method : str
            Aggregation method, either mean or sum/aggregate
        close : bool
            Flag to close object file handlers on exit.
        max_workers : int | None
            Number of cores to run summary on. None is all
            available cpus.
        chunk_point_len : int
            Number of SC points to process on a single parallel worker.

        Returns
        -------
        agg_out : dict
            Aggregated values for each aggregation dataset
        """
        chunks = np.array_split(
            self._gids, int(np.ceil(len(self._gids) / chunk_point_len)))

        logger.info('Running supply curve point aggregation for '
                    'points {} through {} at a resolution of {} '
                    'on {} cores in {} chunks.'
                    .format(self._gids[0], self._gids[-1], self._resolution,
                            max_workers, len(chunks)))

        n_finished = 0
        futures = []
        agg_out = {ds: [] for ds in self._agg_dsets}
        with SpawnProcessPool(max_workers=max_workers) as executor:

            # iterate through split executions, submitting each to worker
            for gid_set in chunks:
                # submit executions and append to futures list
                futures.append(executor.submit(
                    self.run_serial,
                    self._excl_fpath,
                    self._h5_fpath,
                    self._tm_dset,
                    *self._agg_dsets,
                    agg_method=agg_method,
                    excl_dict=self._excl_dict,
                    area_filter_kernel=self._area_filter_kernel,
                    min_area=self._min_area,
                    check_excl_layers=self._check_excl_layers,
                    resolution=self._resolution,
                    gids=gid_set,
                    close=close))

            # gather results
            for future in as_completed(futures):
                n_finished += 1
                logger.info('Parallel aggregation futures collected: '
                            '{} out of {}'
                            .format(n_finished, len(chunks)))
                for k, v in future.results():
                    agg_out[k].append(v)

        return agg_out

    def aggregate(self, agg_method='mean', close=False, max_workers=None,
                  chunk_point_len=1000):
        """
        Aggregate with given agg_method

        Parameters
        ----------
        agg_method : str
            Aggregation method, either mean or sum/aggregate
        close : bool
            Flag to close object file handlers on exit.
        max_workers : int | None
            Number of cores to run summary on. None is all
            available cpus.
        chunk_point_len : int
            Number of SC points to process on a single parallel worker.

        Returns
        -------
        agg : dict
            Aggregated values for each aggregation dataset
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        if max_workers == 1:
            agg = self.run_serial(self._excl_fpath,
                                  self._h5_fpath,
                                  self._tm_dset,
                                  *self._agg_dsets,
                                  agg_method=agg_method,
                                  excl_dict=self._excl_dict,
                                  area_filter_kernel=self._area_filter_kernel,
                                  min_area=self._min_area,
                                  check_excl_layers=self._check_excl_layers,
                                  resolution=self._resolution,
                                  gids=self._gids,
                                  close=close)
        else:
            agg = self.run_parallel(agg_method=agg_method, close=close,
                                    max_workers=max_workers,
                                    chunk_point_len=chunk_point_len)

        if not any(agg):
            e = ('Supply curve aggregation found no non-excluded SC points. '
                 'Please check your exclusions or subset SC GID selection.')
            logger.error(e)
            raise EmptySupplyCurvePointError(e)

        return agg

    @classmethod
    def run(cls, excl_fpath, tm_dset, sc_point_method, excl_dict=None,
            area_filter_kernel='queen', min_area=None,
            check_excl_layers=False, resolution=64, gids=None,
            args=None, kwargs=None, close=False, max_workers=None,
            chunk_point_len=1000):
        """Get the supply curve points aggregation summary.

        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        sc_point_method : method
            Supply Curve Point Method to operate on a single SC point.
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | None
            Minimum required contiguous area filter in sq-km
        check_excl_layers : bool
            Run a pre-flight check on each exclusion layer to ensure they
            contain un-excluded values
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        sc_point_method : method
            Supply Curve Point Method to operate on a single SC point.
        args : list | None
            List of positional args for sc_point_method
        kwargs : dict | None
            Dict of kwargs for sc_point_method
        close : bool
            Flag to close object file handlers on exit.
        max_workers : int | None
            Number of cores to run summary on. None is all
            available cpus.
        chunk_point_len : int
            Number of SC points to process on a single parallel worker.

        Returns
        -------
        summary : DataFrame
            Summary of the SC points.
        """

        agg = cls(excl_fpath, tm_dset, excl_dict=excl_dict,
                  area_filter_kernel=area_filter_kernel, min_area=min_area,
                  check_excl_layers=check_excl_layers, resolution=resolution,
                  gids=gids)

        aggregation = agg.aggregate(sc_point_method, args=args, kwargs=kwargs,
                                    close=close, max_workers=max_workers,
                                    chunk_point_len=chunk_point_len)

        return aggregation
