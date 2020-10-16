# -*- coding: utf-8 -*-
"""
reV aggregation framework.
"""
from abc import ABC, abstractmethod, abstractstaticmethod
from concurrent.futures import as_completed
import h5py
import logging
import numpy as np
import os
import pandas as pd

from reV.handlers.outputs import Outputs
from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.exclusions import ExclusionMaskFromDict
from reV.supply_curve.points import (SupplyCurveExtent,
                                     AggregationSupplyCurvePoint)
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      FileInputError, SupplyCurveInputError)

from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem

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
        """
        Placeholder for h5 Resource handler
        """


class AggFileHandler(AbstractAggFileHandler):
    """
    Framework to handle aggregation file context manager:
    - exclusions .h5 file
    - h5 file to be aggregated
    """

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

        self._h5 = Resource(h5_fpath)

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
    """Abstract supply curve points aggregation framework based on only an
    exclusion file and techmap."""

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
                gids = sc.valid_sc_points(tm_dset)
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

    @abstractstaticmethod
    def run_serial(sc_point_method, excl_fpath, tm_dset,
                   excl_dict=None, area_filter_kernel='queen',
                   min_area=None, check_excl_layers=False,
                   resolution=64, gids=None, args=None,
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
            exclusion_shape = sc.exclusions.shape
            if gids is None:
                gids = range(len(sc))

        # pre-extract handlers so they are not repeatedly initialized
        file_kwargs = {'excl_dict': excl_dict,
                       'area_filter_kernel': area_filter_kernel,
                       'min_area': min_area,
                       'check_excl_layers': check_excl_layers}
        # pylint: disable=abstract-class-instantiated
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
                        close=False,
                        **kwargs)

                except EmptySupplyCurvePointError:
                    pass

                except Exception:
                    logger.exception('SC gid {} failed!'.format(gid))
                    raise

                else:
                    output.append(gid_out)

        return output

    @abstractmethod
    def run_parallel(self, sc_point_method, args=None, kwargs=None,
                     max_workers=None, chunk_point_len=1000):
        """
        Aggregate with sc_point_method in parallel

        Parameters
        ----------
        args : list | None
            List of positional args for sc_point_method
        kwargs : dict | None
            Dict of kwargs for sc_point_method
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
        loggers = [__name__, 'reV.supply_curve.points', 'reV']
        with SpawnProcessPool(max_workers=max_workers, loggers=loggers) as exe:

            # iterate through split executions, submitting each to worker
            for gid_set in chunks:
                # submit executions and append to futures list
                futures.append(exe.submit(
                    self.run_serial,
                    sc_point_method, self._excl_fpath, self._tm_dset,
                    excl_dict=self._excl_dict,
                    area_filter_kernel=self._area_filter_kernel,
                    min_area=self._min_area,
                    check_excl_layers=self._check_excl_layers,
                    resolution=self._resolution,
                    gids=gid_set,
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
                  max_workers=None, chunk_point_len=1000):
        """
        Aggregate with sc_point_method

        Parameters
        ----------
        args : list | None
            List of positional args for sc_point_method
        kwargs : dict | None
            Dict of kwargs for sc_point_method
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
                                  args=args,
                                  kwargs=kwargs)
        else:
            agg = self.run_parallel(sc_point_method, args=args,
                                    kwargs=kwargs, max_workers=max_workers,
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
            args=None, kwargs=None, max_workers=None, chunk_point_len=1000):
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
                                    max_workers=max_workers,
                                    chunk_point_len=chunk_point_len)

        return aggregation


class Aggregation(AbstractAggregation):
    """Concrete but generalized aggregation framework to aggregate ANY reV h5
    file to a supply curve grid (based on an aggregated exclusion grid)."""

    def __init__(self, excl_fpath, h5_fpath, tm_dset, *agg_dset,
                 excl_dict=None, area_filter_kernel='queen', min_area=None,
                 check_excl_layers=False, resolution=64, excl_area=None,
                 gids=None):
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
        excl_area : float | None
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath.
        gids : list | None
            List of gids to get aggregation for (can use to subset if running
            in parallel), or None for all gids in the SC extent.
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
        self._gen_index = self._parse_gen_index(self._h5_fpath)

        if excl_area is None:
            with ExclusionLayers(excl_fpath) as excl:
                excl_area = excl.pixel_area
        self._excl_area = excl_area
        if self._excl_area is None:
            e = ('No exclusion pixel area was input and could not parse '
                 'area from the exclusion file attributes!')
            logger.error(e)
            raise SupplyCurveInputError(e)

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

        with Resource(self._h5_fpath) as f:
            for dset in self._agg_dsets:
                if dset not in f:
                    raise FileInputError('Could not find provided dataset "{}"'
                                         ' in h5 file: {}'
                                         .format(dset, self._h5_fpath))

    @staticmethod
    def _parse_gen_index(h5_fpath):
        """Parse gen outputs for an array of generation gids corresponding to
        the resource gids.

        Parameters
        ----------
        h5_fpath : str
            Filepath to reV compliant .h5 file

        Returns
        -------
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        """

        with Resource(h5_fpath) as f:
            gen_index = f.meta

        if 'gid' in gen_index:
            gen_index = gen_index.rename(columns={'gid': 'res_gids'})
            gen_index['gen_gids'] = gen_index.index
            gen_index = gen_index[['res_gids', 'gen_gids']]
            gen_index = gen_index.set_index(keys='res_gids')
            gen_index = \
                gen_index.reindex(range(int(gen_index.index.max() + 1)))
            gen_index = gen_index['gen_gids'].values
            gen_index[np.isnan(gen_index)] = -1
            gen_index = gen_index.astype(np.int32)
        else:
            gen_index = None

        return gen_index

    @staticmethod
    def run_serial(excl_fpath, h5_fpath, tm_dset, *agg_dset,
                   agg_method='mean', excl_dict=None,
                   area_filter_kernel='queen', min_area=None,
                   check_excl_layers=False, resolution=64, excl_area=0.0081,
                   gids=None, gen_index=None):
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
        excl_area : float
            Area of an exclusion cell (square km).
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.

        Returns
        -------
        agg_out : dict
            Aggregated values for each aggregation dataset
        """
        with SupplyCurveExtent(excl_fpath, resolution=resolution) as sc:
            exclusion_shape = sc.exclusions.shape
            if gids is None:
                gids = sc.valid_sc_points(tm_dset)

        # pre-extract handlers so they are not repeatedly initialized
        file_kwargs = {'excl_dict': excl_dict,
                       'area_filter_kernel': area_filter_kernel,
                       'min_area': min_area,
                       'check_excl_layers': check_excl_layers}
        dsets = agg_dset + ('meta', )
        agg_out = {ds: [] for ds in dsets}
        with AggFileHandler(excl_fpath, h5_fpath, **file_kwargs) as fh:
            n_finished = 0
            for gid in gids:
                try:
                    gid_out = AggregationSupplyCurvePoint.run(
                        gid,
                        fh.exclusions,
                        fh.h5,
                        tm_dset,
                        *agg_dset,
                        agg_method=agg_method,
                        excl_dict=excl_dict,
                        resolution=resolution,
                        excl_area=excl_area,
                        exclusion_shape=exclusion_shape,
                        close=False,
                        gen_index=gen_index)

                except EmptySupplyCurvePointError:
                    logger.debug('SC gid {} is fully excluded or does not '
                                 'have any valid source data!'.format(gid))
                except Exception:
                    logger.exception('SC gid {} failed!'.format(gid))
                    raise
                else:
                    n_finished += 1
                    logger.debug('Serial aggregation: '
                                 '{} out of {} points complete'
                                 .format(n_finished, len(gids)))
                    log_mem(logger)
                    for k, v in gid_out.items():
                        agg_out[k].append(v)

        return agg_out

    def run_parallel(self, agg_method='mean', excl_area=0.0081,
                     max_workers=None, chunk_point_len=1000):
        """
        Aggregate in parallel

        Parameters
        ----------
        agg_method : str
            Aggregation method, either mean or sum/aggregate
        excl_area : float
            Area of an exclusion cell (square km).
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
        dsets = self._agg_dsets + ('meta', )
        agg_out = {ds: [] for ds in dsets}
        loggers = [__name__, 'reV.supply_curve.points', 'reV']
        with SpawnProcessPool(max_workers=max_workers, loggers=loggers) as exe:
            # iterate through split executions, submitting each to worker
            for gid_set in chunks:
                # submit executions and append to futures list
                futures.append(exe.submit(
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
                    excl_area=excl_area,
                    gids=gid_set,
                    gen_index=self._gen_index))

            # gather results
            for future in futures:
                n_finished += 1
                logger.info('Parallel aggregation futures collected: '
                            '{} out of {}'
                            .format(n_finished, len(chunks)))
                for k, v in future.result().items():
                    if v:
                        agg_out[k].extend(v)

        return agg_out

    def aggregate(self, agg_method='mean', max_workers=None,
                  chunk_point_len=1000):
        """
        Aggregate with given agg_method

        Parameters
        ----------
        agg_method : str
            Aggregation method, either mean or sum/aggregate
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
                                  excl_area=self._excl_area,
                                  gen_index=self._gen_index)
        else:
            agg = self.run_parallel(agg_method=agg_method,
                                    excl_area=self._excl_area,
                                    max_workers=max_workers,
                                    chunk_point_len=chunk_point_len)

        if not agg['meta']:
            e = ('Supply curve aggregation found no non-excluded SC points. '
                 'Please check your exclusions or subset SC GID selection.')
            logger.error(e)
            raise EmptySupplyCurvePointError(e)

        for k, v in agg.items():
            if k == 'meta':
                v = pd.concat(v, axis=1).T
                v = v.sort_values('sc_point_gid')
                v = v.reset_index(drop=True)
                v.index.name = 'sc_gid'
                agg[k] = v.reset_index()
            else:
                v = np.dstack(v)[0]
                if v.shape[0] == 1:
                    v = v.flatten()

                agg[k] = v

        return agg

    def save_agg_to_h5(self, out_fpath, aggregation):
        """
        Save aggregated data to disc in .h5 format

        Parameters
        ----------
        out_fpath : str
            Output .h5 file path
        aggregation : dict
            Aggregated values for each aggregation dataset
        """
        agg_out = aggregation.copy()
        meta = agg_out.pop('meta')
        for c in meta.columns:
            try:
                meta[c] = pd.to_numeric(meta[c])
            except (ValueError, TypeError):
                pass

        dsets = []
        shapes = {}
        attrs = {}
        chunks = {}
        dtypes = {}
        time_index = None
        with Resource(self._h5_fpath) as f:
            for dset, data in agg_out.items():
                dsets.append(dset)
                shape = data.shape
                shapes[dset] = shape
                if len(data.shape) == 2:
                    if ('time_index' in f) and (shape[0] == f.shape[0]):
                        if time_index is None:
                            time_index = f.time_index

                attrs[dset] = f.get_attrs(dset=dset)
                _, dtype, chunk = f.get_dset_properties(dset)
                chunks[dset] = chunk
                dtypes[dset] = dtype

        Outputs.init_h5(out_fpath, dsets, shapes, attrs, chunks, dtypes,
                        meta, time_index=time_index)

        with Outputs(out_fpath, mode='a') as out:
            for dset, data in agg_out.items():
                out[dset] = data

    @classmethod
    def run(cls, excl_fpath, h5_fpath, tm_dset, *agg_dset,
            excl_dict=None, area_filter_kernel='queen', min_area=None,
            check_excl_layers=False, resolution=64, gids=None,
            agg_method='mean', excl_area=None, max_workers=None,
            chunk_point_len=1000, out_fpath=None):
        """Get the supply curve points aggregation summary.

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
            List of gids to get aggregation for (can use to subset if running
            in parallel), or None for all gids in the SC extent.
        agg_method : str
            Aggregation method, either mean or sum/aggregate
        excl_area : float | None
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath.
        max_workers : int | None
            Number of cores to run summary on. None is all
            available cpus.
        chunk_point_len : int
            Number of SC points to process on a single parallel worker.
        out_fpath : str
            Output .h5 file path

        Returns
        -------
        agg : dict
            Aggregated values for each aggregation dataset
        """

        agg = cls(excl_fpath, h5_fpath, tm_dset, *agg_dset,
                  excl_dict=excl_dict, area_filter_kernel=area_filter_kernel,
                  min_area=min_area, check_excl_layers=check_excl_layers,
                  resolution=resolution, gids=gids, excl_area=excl_area)

        aggregation = agg.aggregate(agg_method=agg_method,
                                    max_workers=max_workers,
                                    chunk_point_len=chunk_point_len)

        if out_fpath is not None:
            agg.save_agg_to_h5(out_fpath, aggregation)

        return aggregation
