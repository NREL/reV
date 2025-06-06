# -*- coding: utf-8 -*-
"""reV aggregation framework."""

import logging
import os
from abc import ABC, abstractmethod

import h5py
import numpy as np
import pandas as pd
from rex.resource import Resource
from rex.utilities import check_res_file
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem

from reV.handlers.exclusions import ExclusionLayers
from reV.handlers.outputs import Outputs
from reV.supply_curve.exclusions import ExclusionMaskFromDict
from reV.supply_curve.extent import SupplyCurveExtent
from reV.supply_curve.points import AggregationSupplyCurvePoint
from reV.supply_curve.tech_mapping import TechMapping
from reV.utilities import ResourceMetaField, SupplyCurveField, log_versions
from reV.utilities.exceptions import (
    EmptySupplyCurvePointError,
    FileInputError,
    SupplyCurveInputError,
)

logger = logging.getLogger(__name__)


class AbstractAggFileHandler(ABC):
    """Simple framework to handle aggregation file context managers."""

    def __init__(
        self,
        excl_fpath,
        excl_dict=None,
        area_filter_kernel="queen",
        min_area=None,
    ):
        """
        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        excl_dict : dict | None
            Dictionary of exclusion keyword arugments of the format
            {layer_dset_name: {kwarg: value}} where layer_dset_name is a
            dataset in the exclusion h5 file and kwarg is a keyword argument to
            the reV.supply_curve.exclusions.LayerMask class.
            by default None
        area_filter_kernel : str, optional
            Contiguous area filter method to use on final exclusions mask,
            by default 'queen'
        min_area : float, optional
            Minimum required contiguous area filter in sq-km,
            by default None
        """
        self._excl_fpath = excl_fpath
        self._excl = ExclusionMaskFromDict(
            excl_fpath,
            layers_dict=excl_dict,
            min_area=min_area,
            kernel=area_filter_kernel,
        )

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

    DEFAULT_H5_HANDLER = Resource

    def __init__(
        self,
        excl_fpath,
        h5_fpath,
        excl_dict=None,
        area_filter_kernel="queen",
        min_area=None,
        h5_handler=None,
        **h5_handler_kwargs,
    ):
        """
        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        h5_fpath : str
            Filepath to .h5 file to be aggregated
        excl_dict : dict | None
            Dictionary of exclusion keyword arugments of the format
            {layer_dset_name: {kwarg: value}} where layer_dset_name is a
            dataset in the exclusion h5 file and kwarg is a keyword argument to
            the reV.supply_curve.exclusions.LayerMask class.
            by default None
        area_filter_kernel : str, optional
            Contiguous area filter method to use on final exclusions mask,
            by default 'queen'
        min_area : float, optional
            Minimum required contiguous area filter in sq-km, by default None
        h5_handler : rex.Resource | None
            Optional special handler similar to the rex.Resource handler which
            is default.
        **h5_handler_kwargs
            Optional keyword-value pairs to pass to the h5 handler.
        """
        super().__init__(
            excl_fpath,
            excl_dict=excl_dict,
            area_filter_kernel=area_filter_kernel,
            min_area=min_area,
        )

        if h5_handler is None:
            self._h5 = Resource(h5_fpath, **h5_handler_kwargs)
        else:
            self._h5 = h5_handler(h5_fpath, **h5_handler_kwargs)

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


class BaseAggregation(ABC):
    """Abstract supply curve points aggregation framework based on only an
    exclusion file and techmap."""

    def __init__(
        self,
        excl_fpath,
        tm_dset,
        excl_dict=None,
        area_filter_kernel="queen",
        min_area=None,
        resolution=64,
        excl_area=None,
        res_fpath=None,
        gids=None,
        pre_extract_inclusions=False,
    ):
        """
        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        excl_dict : dict | None
            Dictionary of exclusion keyword arugments of the format
            {layer_dset_name: {kwarg: value}} where layer_dset_name is a
            dataset in the exclusion h5 file and kwarg is a keyword argument to
            the reV.supply_curve.exclusions.LayerMask class.
            by default None
        area_filter_kernel : str, optional
            Contiguous area filter method to use on final exclusions mask,
            by default "queen"
        min_area : float, optional
            Minimum required contiguous area filter in sq-km,
            by default None
        resolution : int, optional
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead,
            by default None
        excl_area : float, optional
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath, by default None
        gids : list, optional
            List of supply curve point gids to get summary for (can use to
            subset if running in parallel), or None for all gids in the SC
            extent, by default None
        pre_extract_inclusions : bool, optional
            Optional flag to pre-extract/compute the inclusion mask from the
            provided excl_dict, by default False. Typically faster to compute
            the inclusion mask on the fly with parallel workers.
        """
        self._excl_fpath = excl_fpath
        self._tm_dset = tm_dset
        self._excl_dict = excl_dict
        self._resolution = resolution
        self._area_filter_kernel = area_filter_kernel
        self._min_area = min_area
        self._res_fpath = res_fpath
        self._gids = gids
        self._pre_extract_inclusions = pre_extract_inclusions
        self._excl_area = self._get_excl_area(excl_fpath, excl_area=excl_area)
        self._shape = None

        self._validate_tech_mapping()

        if pre_extract_inclusions:
            self._inclusion_mask = (
                ExclusionMaskFromDict.extract_inclusion_mask(
                    excl_fpath,
                    tm_dset,
                    excl_dict=excl_dict,
                    area_filter_kernel=area_filter_kernel,
                    min_area=min_area,
                )
            )
        else:
            self._inclusion_mask = None

    def _validate_tech_mapping(self):
        """Check that tech mapping exists and create it if it doesn't"""

        with ExclusionLayers(self._excl_fpath) as f:
            dsets = f.h5.dsets

        excl_fp_is_str = isinstance(self._excl_fpath, str)
        tm_in_excl = self._tm_dset in dsets
        if tm_in_excl:
            logger.info('Found techmap "{}".'.format(self._tm_dset))
        elif not tm_in_excl and not excl_fp_is_str:
            msg = (
                'Could not find techmap dataset "{}" and cannot run '
                "techmap with arbitrary multiple exclusion filepaths "
                "to write to: {}".format(self._tm_dset, self._excl_fpath)
            )
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            logger.info(
                'Could not find techmap "{}". Running techmap module.'.format(
                    self._tm_dset
                )
            )
            try:
                TechMapping.run(
                    self._excl_fpath, self._res_fpath, dset=self._tm_dset
                )
            except Exception as e:
                msg = (
                    "TechMapping process failed. Received the "
                    "following error:\n{}".format(e)
                )
                logger.exception(msg)
                raise RuntimeError(msg) from e

    @property
    def gids(self):
        """
        1D array of supply curve point gids to aggregate

        Returns
        -------
        ndarray
        """
        if self._gids is None:
            with SupplyCurveExtent(
                self._excl_fpath, resolution=self._resolution
            ) as sc:
                self._gids = sc.valid_sc_points(self._tm_dset)
        elif np.issubdtype(type(self._gids), np.number):
            self._gids = np.array([self._gids])
        elif not isinstance(self._gids, np.ndarray):
            self._gids = np.array(self._gids)

        return self._gids

    @property
    def shape(self):
        """Get the shape of the full exclusions raster.

        Returns
        -------
        tuple
        """
        if self._shape is None:
            with SupplyCurveExtent(
                self._excl_fpath, resolution=self._resolution
            ) as sc:
                self._shape = sc.exclusions.shape

        return self._shape

    @staticmethod
    def _get_excl_area(excl_fpath, excl_area=None):
        """
        Get exclusion area from excl_fpath pixel area. Confirm that the
        exclusion area is not None.

        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        excl_area : float, optional
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath, by default None

        Returns
        -------
        excl_area : float
            Area of an exclusion pixel in km2
        """
        if excl_area is None:
            logger.debug(
                "Setting the exclusion area from the area of a pixel "
                "in {}".format(excl_fpath)
            )
            with ExclusionLayers(excl_fpath) as excl:
                excl_area = excl.pixel_area

        if excl_area is None:
            e = (
                "No exclusion pixel area was input and could not parse "
                "area from the exclusion file attributes!"
            )
            logger.error(e)
            raise SupplyCurveInputError(e)

        return excl_area

    @staticmethod
    def _check_inclusion_mask(inclusion_mask, gids, excl_shape):
        """
        Check inclusion mask to ensure it has the proper shape

        Parameters
        ----------
        inclusion_mask : np.ndarray | dict | optional
            2D array pre-extracted inclusion mask where 1 is included and 0 is
            excluded. This must be either match the full exclusion shape or
            be a dict lookup of single-sc-point exclusion masks corresponding
            to the gids input and keyed by gids, by default None which will
            calculate exclusions on the fly for each sc point.
        gids : list | ndarray
            sc point gids corresponding to inclusion mask
        excl_shape : tuple
            Full exclusion layers shape
        """
        if isinstance(inclusion_mask, dict):
            assert len(inclusion_mask) == len(gids)
        elif isinstance(inclusion_mask, np.ndarray):
            assert inclusion_mask.shape == excl_shape
        elif inclusion_mask is not None:
            msg = (
                "Expected inclusion_mask to be dict or array but received "
                "{}".format(type(inclusion_mask))
            )
            logger.error(msg)
            raise SupplyCurveInputError(msg)

    @staticmethod
    def _get_gid_inclusion_mask(
        inclusion_mask, gid, slice_lookup, resolution=64
    ):
        """
        Get inclusion mask for desired gid

        Parameters
        ----------
        inclusion_mask : np.ndarray | dict | optional
            2D array pre-extracted inclusion mask where 1 is included and 0 is
            excluded. This must be either match the full exclusion shape or
            be a dict lookup of single-sc-point exclusion masks corresponding
            to the gids input and keyed by gids, by default None which will
            calculate exclusions on the fly for each sc point.
        gid : int
            sc_point_gid value, used to extract inclusion mask from 2D
            inclusion array
        slice_lookup : dict
            Mapping of sc_point_gids to exclusion/inclusion row and column
            slices
        resolution : int, optional
            supply curve extent resolution, by default 64

        Returns
        -------
        gid_inclusions : ndarray | None
            2D array of inclusions for desired gid, normalized from 0, excluded
            to 1 fully included, if inclusion mask is None gid_inclusions
            is None
        """
        gid_inclusions = None
        if isinstance(inclusion_mask, dict):
            gid_inclusions = inclusion_mask[gid]
            assert gid_inclusions.shape[0] <= resolution
            assert gid_inclusions.shape[1] <= resolution
        elif isinstance(inclusion_mask, np.ndarray):
            row_slice, col_slice = slice_lookup[gid]
            gid_inclusions = inclusion_mask[row_slice, col_slice]
        elif inclusion_mask is not None:
            msg = (
                "Expected inclusion_mask to be dict or array but received "
                "{}".format(type(inclusion_mask))
            )
            logger.error(msg)
            raise SupplyCurveInputError(msg)

        return gid_inclusions

    @staticmethod
    def _get_gid_zones(excl_fpath, zones_dset, gid, slice_lookup):
        """
        Get zones 2D array for desired gid.

        Parameters
        ----------
        excl_fpath : str | None, optional
            Filepath to HDF5 file containing `zones_dset`. If not specified,
            output of function will be an array containing all values equal to
            1.
        zones_dset : str | None, optional
            Dataset name in the `excl_fpath` file containing the zones to be
            loaded. If not specified, output of function will be an array
            containing all values equal to 1.
        gid : int
            sc_point_gid value, used to extract the applicable subset of zones.
        slice_lookup : dict
            Mapping of sc_point_gids to exclusion/inclusion row and column
            slices

        Returns
        -------
        zones : ndarray | None
            2D array of zones for desired gid.
        """

        row_slice, col_slice = slice_lookup[gid]
        if excl_fpath is not None and zones_dset is not None:
            with ExclusionLayers(excl_fpath) as fh:
                if zones_dset not in fh:
                    msg = (
                        f"Could not find zones_dset {zones_dset} in "
                        f"excl_fpath {excl_fpath}."
                    )
                    logger.error(msg)
                    raise FileInputError(msg)
                zones = fh[zones_dset, row_slice, col_slice]
        else:
            shape = (
                row_slice.stop - row_slice.start,
                col_slice.stop - col_slice.start
            )
            zones = np.ones(shape, dtype="uint8")

        return zones

    @staticmethod
    def _parse_gen_index(gen_fpath):
        """Parse gen outputs for an array of generation gids corresponding to
        the resource gids.

        Parameters
        ----------
        gen_fpath : str
            Filepath to reV generation output .h5 file. This can also be a csv
            filepath to a project points input file.

        Returns
        -------
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        """

        __, hsds = check_res_file(gen_fpath)
        if gen_fpath.endswith(".h5"):
            with Resource(gen_fpath, hsds=hsds) as f:
                gen_index = f.meta
        elif gen_fpath.endswith(".csv"):
            gen_index = pd.read_csv(gen_fpath)
        else:
            msg = (
                "Could not recognize gen_fpath input, needs to be reV gen "
                "output h5 or project points csv but received: {}".format(
                    gen_fpath
                )
            )
            logger.error(msg)
            raise FileInputError(msg)

        if ResourceMetaField.GID in gen_index:
            gen_index = gen_index.rename(
                columns={ResourceMetaField.GID: SupplyCurveField.RES_GIDS}
            )
            gen_index[SupplyCurveField.GEN_GIDS] = gen_index.index
            gen_index = gen_index[
                [SupplyCurveField.RES_GIDS, SupplyCurveField.GEN_GIDS]
            ]
            gen_index = gen_index.set_index(keys=SupplyCurveField.RES_GIDS)
            gen_index = gen_index.reindex(
                range(int(gen_index.index.max() + 1))
            )
            gen_index = gen_index[SupplyCurveField.GEN_GIDS].values
            gen_index[np.isnan(gen_index)] = -1
            gen_index = gen_index.astype(np.int32)
        else:
            gen_index = None

        return gen_index


class Aggregation(BaseAggregation):
    """Concrete but generalized aggregation framework to aggregate ANY reV h5
    file to a supply curve grid (based on an aggregated exclusion grid)."""

    def __init__(
        self,
        excl_fpath,
        tm_dset,
        *agg_dset,
        excl_dict=None,
        area_filter_kernel="queen",
        min_area=None,
        resolution=64,
        excl_area=None,
        gids=None,
        pre_extract_inclusions=False,
    ):
        """
        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        agg_dset : str
            Dataset to aggreate, can supply multiple datasets. The datasets
            should be scalar values for each site. This method cannot aggregate
            timeseries data.
        excl_dict : dict | None
            Dictionary of exclusion keyword arugments of the format
            {layer_dset_name: {kwarg: value}} where layer_dset_name is a
            dataset in the exclusion h5 file and kwarg is a keyword argument to
            the reV.supply_curve.exclusions.LayerMask class.
            by default None
        area_filter_kernel : str, optional
            Contiguous area filter method to use on final exclusions mask,
            by default "queen"
        min_area : float, optional
            Minimum required contiguous area filter in sq-km,
            by default None
        resolution : int, optional
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead,
            by default None
        excl_area : float, optional
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath,
            by default None
        gids : list, optional
            List of supply curve point gids to get summary for (can use to
            subset if running in parallel), or None for all gids in the SC
            extent, by default None
        pre_extract_inclusions : bool, optional
            Optional flag to pre-extract/compute the inclusion mask from the
            provided excl_dict, by default False. Typically faster to compute
            the inclusion mask on the fly with parallel workers.
        """
        log_versions(logger)
        logger.info("Initializing Aggregation...")
        logger.debug("Exclusion filepath: {}".format(excl_fpath))
        logger.debug("Exclusion dict: {}".format(excl_dict))

        super().__init__(
            excl_fpath,
            tm_dset,
            excl_dict=excl_dict,
            area_filter_kernel=area_filter_kernel,
            min_area=min_area,
            resolution=resolution,
            excl_area=excl_area,
            gids=gids,
            pre_extract_inclusions=pre_extract_inclusions,
        )

        if isinstance(agg_dset, str):
            agg_dset = (agg_dset,)

        self._agg_dsets = agg_dset

    def _check_files(self, h5_fpath):
        """Do a preflight check on input files"""

        if not os.path.exists(self._excl_fpath):
            raise FileNotFoundError(
                "Could not find required exclusions file: " "{}".format(
                    self._excl_fpath
                )
            )

        __, hsds = check_res_file(h5_fpath)
        if not hsds and not os.path.exists(h5_fpath):
            raise FileNotFoundError(
                "Could not find required h5 file: " "{}".format(h5_fpath)
            )

        with h5py.File(self._excl_fpath, "r") as f:
            if self._tm_dset not in f:
                raise FileInputError(
                    'Could not find techmap dataset "{}" '
                    "in exclusions file: {}".format(
                        self._tm_dset, self._excl_fpath
                    )
                )

        with Resource(h5_fpath, hsds=hsds) as f:
            for dset in self._agg_dsets:
                if dset not in f:
                    raise FileInputError(
                        'Could not find provided dataset "{}"'
                        " in h5 file: {}".format(dset, h5_fpath)
                    )

    @classmethod
    def run_serial(
        cls,
        excl_fpath,
        h5_fpath,
        tm_dset,
        *agg_dset,
        agg_method="mean",
        excl_dict=None,
        inclusion_mask=None,
        area_filter_kernel="queen",
        min_area=None,
        resolution=64,
        excl_area=0.0081,
        gids=None,
        gen_index=None,
    ):
        """
        Standalone method to aggregate - can be parallelized.

        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        h5_fpath : str
            Filepath to .h5 file to aggregate
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        agg_dset : str
            Dataset to aggreate, can supply multiple datasets. The datasets
            should be scalar values for each site. This method cannot aggregate
            timeseries data.
        agg_method : str, optional
            Aggregation method, either mean or sum/aggregate, by default "mean"
        excl_dict : dict | None
            Dictionary of exclusion keyword arugments of the format
            {layer_dset_name: {kwarg: value}} where layer_dset_name is a
            dataset in the exclusion h5 file and kwarg is a keyword argument to
            the reV.supply_curve.exclusions.LayerMask class.
            by default None
        inclusion_mask : np.ndarray, optional
            2D array pre-extracted inclusion mask where 1 is included and 0 is
            excluded. This must be either match the full exclusion shape or
            be a list of single-sc-point exclusion masks corresponding to the
            gids input, by default None
        area_filter_kernel : str, optional
            Contiguous area filter method to use on final exclusions mask,
            by default "queen"
        min_area : float, optional
            Minimum required contiguous area filter in sq-km,
            by default None
        resolution : int, optional
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead,
            by default 0.0081
        excl_area : float, optional
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath,
            by default None
        gids : list, optional
            List of supply curve point gids to get summary for (can use to
            subset if running in parallel), or None for all gids in the SC
            extent, by default None
        gen_index : np.ndarray, optional
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run, by default None

        Returns
        -------
        agg_out : dict
            Aggregated values for each aggregation dataset
        """
        with SupplyCurveExtent(excl_fpath, resolution=resolution) as sc:
            exclusion_shape = sc.exclusions.shape
            if gids is None:
                gids = sc.valid_sc_points(tm_dset)
            elif np.issubdtype(type(gids), np.number):
                gids = [gids]

            slice_lookup = sc.get_slice_lookup(gids)

        cls._check_inclusion_mask(inclusion_mask, gids, exclusion_shape)

        # pre-extract handlers so they are not repeatedly initialized
        file_kwargs = {
            "excl_dict": excl_dict,
            "area_filter_kernel": area_filter_kernel,
            "min_area": min_area,
            "hsds": check_res_file(h5_fpath)[1],
        }
        dsets = (
            *agg_dset,
            "meta",
        )
        agg_out = {ds: [] for ds in dsets}
        with AggFileHandler(excl_fpath, h5_fpath, **file_kwargs) as fh:
            n_finished = 0
            for gid in gids:
                gid_inclusions = cls._get_gid_inclusion_mask(
                    inclusion_mask, gid, slice_lookup, resolution=resolution
                )
                try:
                    gid_out = AggregationSupplyCurvePoint.run(
                        gid,
                        fh.exclusions,
                        fh.h5,
                        tm_dset,
                        *agg_dset,
                        agg_method=agg_method,
                        excl_dict=excl_dict,
                        inclusion_mask=gid_inclusions,
                        resolution=resolution,
                        excl_area=excl_area,
                        exclusion_shape=exclusion_shape,
                        close=False,
                        gen_index=gen_index,
                    )

                except EmptySupplyCurvePointError:
                    logger.debug(
                        "SC gid {} is fully excluded or does not "
                        "have any valid source data!".format(gid)
                    )
                except Exception as e:
                    msg = "SC gid {} failed!".format(gid)
                    logger.exception(msg)
                    raise RuntimeError(msg) from e
                else:
                    n_finished += 1
                    logger.debug(
                        "Serial aggregation: "
                        "{} out of {} points complete".format(
                            n_finished, len(gids)
                        )
                    )
                    log_mem(logger)
                    for k, v in gid_out.items():
                        agg_out[k].append(v)

        return agg_out

    def run_parallel(
        self,
        h5_fpath,
        agg_method="mean",
        excl_area=None,
        max_workers=None,
        sites_per_worker=100,
    ):
        """
        Aggregate in parallel

        Parameters
        ----------
        h5_fpath : str
            Filepath to .h5 file to aggregate
        agg_method : str, optional
            Aggregation method, either mean or sum/aggregate, by default "mean"
        excl_area : float, optional
            Area of an exclusion cell (square km), by default None
        max_workers : int, optional
            Number of cores to run summary on. None is all available cpus,
            by default None
        sites_per_worker : int, optional
            Number of SC points to process on a single parallel worker,
            by default 100

        Returns
        -------
        agg_out : dict
            Aggregated values for each aggregation dataset
        """

        self._check_files(h5_fpath)
        gen_index = self._parse_gen_index(h5_fpath)

        slice_lookup = None
        chunks = int(np.ceil(len(self.gids) / sites_per_worker))
        chunks = np.array_split(self.gids, chunks)

        if self._inclusion_mask is not None:
            with SupplyCurveExtent(
                self._excl_fpath, resolution=self._resolution
            ) as sc:
                assert sc.exclusions.shape == self._inclusion_mask.shape
                slice_lookup = sc.get_slice_lookup(self.gids)

        logger.info(
            "Running supply curve point aggregation for "
            "points {} through {} at a resolution of {} "
            "on {} cores in {} chunks.".format(
                self.gids[0],
                self.gids[-1],
                self._resolution,
                max_workers,
                len(chunks),
            )
        )

        n_finished = 0
        futures = []
        dsets = self._agg_dsets + ("meta",)
        agg_out = {ds: [] for ds in dsets}
        loggers = [__name__, "reV.supply_curve.points", "reV"]
        with SpawnProcessPool(max_workers=max_workers, loggers=loggers) as exe:
            # iterate through split executions, submitting each to worker
            for gid_set in chunks:
                # submit executions and append to futures list
                chunk_incl_masks = None
                if self._inclusion_mask is not None:
                    chunk_incl_masks = {}
                    for gid in gid_set:
                        rs, cs = slice_lookup[gid]
                        chunk_incl_masks[gid] = self._inclusion_mask[rs, cs]

                # submit executions and append to futures list
                futures.append(
                    exe.submit(
                        self.run_serial,
                        self._excl_fpath,
                        h5_fpath,
                        self._tm_dset,
                        *self._agg_dsets,
                        agg_method=agg_method,
                        excl_dict=self._excl_dict,
                        inclusion_mask=chunk_incl_masks,
                        area_filter_kernel=self._area_filter_kernel,
                        min_area=self._min_area,
                        resolution=self._resolution,
                        excl_area=excl_area,
                        gids=gid_set,
                        gen_index=gen_index,
                    )
                )

            # gather results
            for future in futures:
                n_finished += 1
                logger.info(
                    "Parallel aggregation futures collected: "
                    "{} out of {}".format(n_finished, len(chunks))
                )
                for k, v in future.result().items():
                    if v:
                        agg_out[k].extend(v)

        return agg_out

    def aggregate(
        self,
        h5_fpath,
        agg_method="mean",
        max_workers=None,
        sites_per_worker=100,
    ):
        """
        Aggregate with given agg_method

        Parameters
        ----------
        h5_fpath : str
            Filepath to .h5 file to aggregate
        agg_method : str, optional
            Aggregation method, either mean or sum/aggregate, by default "mean"
        max_workers : int, optional
            Number of cores to run summary on. None is all available cpus,
            by default None
        sites_per_worker : int, optional
            Number of SC points to process on a single parallel worker,
            by default 100

        Returns
        -------
        agg : dict
            Aggregated values for each aggregation dataset
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        if max_workers == 1:
            self._check_files(h5_fpath)
            gen_index = self._parse_gen_index(h5_fpath)
            agg = self.run_serial(
                self._excl_fpath,
                h5_fpath,
                self._tm_dset,
                *self._agg_dsets,
                agg_method=agg_method,
                excl_dict=self._excl_dict,
                gids=self.gids,
                inclusion_mask=self._inclusion_mask,
                area_filter_kernel=self._area_filter_kernel,
                min_area=self._min_area,
                resolution=self._resolution,
                excl_area=self._excl_area,
                gen_index=gen_index,
            )
        else:
            agg = self.run_parallel(
                h5_fpath=h5_fpath,
                agg_method=agg_method,
                excl_area=self._excl_area,
                max_workers=max_workers,
                sites_per_worker=sites_per_worker,
            )

        if not agg["meta"]:
            e = (
                "Supply curve aggregation found no non-excluded SC points. "
                "Please check your exclusions or subset SC GID selection."
            )
            logger.error(e)
            raise EmptySupplyCurvePointError(e)

        for k, v in agg.items():
            if k == "meta":
                v = pd.concat(v, axis=1).T
                v = v.sort_values(SupplyCurveField.SC_POINT_GID)
                v = v.reset_index(drop=True)
                v.index.name = SupplyCurveField.SC_GID
                agg[k] = v
            else:
                v = np.dstack(v)[0]
                if v.shape[0] == 1:
                    v = v.flatten()

                agg[k] = v

        return agg

    @staticmethod
    def save_agg_to_h5(h5_fpath, out_fpath, aggregation):
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
        meta = agg_out.pop("meta").reset_index()
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

        __, hsds = check_res_file(h5_fpath)
        with Resource(h5_fpath, hsds=hsds) as f:
            for dset, data in agg_out.items():
                dsets.append(dset)
                shape = data.shape
                shapes[dset] = shape
                if len(data.shape) == 2:
                    if ("time_index" in f) and (shape[0] == f.shape[0]):
                        if time_index is None:
                            time_index = f.time_index

                attrs[dset] = f.get_attrs(dset=dset)
                _, dtype, chunk = f.get_dset_properties(dset)
                chunks[dset] = chunk
                dtypes[dset] = dtype

        Outputs.init_h5(
            out_fpath,
            dsets,
            shapes,
            attrs,
            chunks,
            dtypes,
            meta,
            time_index=time_index,
        )

        with Outputs(out_fpath, mode="a") as out:
            for dset, data in agg_out.items():
                out[dset] = data

    @classmethod
    def run(
        cls,
        excl_fpath,
        h5_fpath,
        tm_dset,
        *agg_dset,
        excl_dict=None,
        area_filter_kernel="queen",
        min_area=None,
        resolution=64,
        excl_area=None,
        gids=None,
        pre_extract_inclusions=False,
        agg_method="mean",
        max_workers=None,
        sites_per_worker=100,
        out_fpath=None,
    ):
        """Get the supply curve points aggregation summary.

        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        h5_fpath : str
            Filepath to .h5 file to aggregate
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        agg_dset : str
            Dataset to aggreate, can supply multiple datasets. The datasets
            should be scalar values for each site. This method cannot aggregate
            timeseries data.
        excl_dict : dict | None
            Dictionary of exclusion keyword arugments of the format
            {layer_dset_name: {kwarg: value}} where layer_dset_name is a
            dataset in the exclusion h5 file and kwarg is a keyword argument to
            the reV.supply_curve.exclusions.LayerMask class.
            by default None
        area_filter_kernel : str, optional
            Contiguous area filter method to use on final exclusions mask,
            by default "queen"
        min_area : float, optional
            Minimum required contiguous area filter in sq-km,
            by default None
        resolution : int, optional
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead,
            by default None
        excl_area : float, optional
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath,
            by default None
        gids : list, optional
            List of supply curve point gids to get summary for (can use to
            subset if running in parallel), or None for all gids in the SC
            extent, by default None
        pre_extract_inclusions : bool, optional
            Optional flag to pre-extract/compute the inclusion mask from the
            provided excl_dict, by default False. Typically faster to compute
            the inclusion mask on the fly with parallel workers.
        agg_method : str, optional
            Aggregation method, either mean or sum/aggregate, by default "mean"
        max_workers : int, optional
            Number of cores to run summary on. None is all available cpus,
            by default None
        sites_per_worker : int, optional
            Number of SC points to process on a single parallel worker,
            by default 100
        out_fpath : str, optional
            Output .h5 file path, by default None

        Returns
        -------
        agg : dict
            Aggregated values for each aggregation dataset
        """

        agg = cls(
            excl_fpath,
            tm_dset,
            *agg_dset,
            excl_dict=excl_dict,
            area_filter_kernel=area_filter_kernel,
            min_area=min_area,
            resolution=resolution,
            excl_area=excl_area,
            gids=gids,
            pre_extract_inclusions=pre_extract_inclusions,
        )

        aggregation = agg.aggregate(
            h5_fpath=h5_fpath,
            agg_method=agg_method,
            max_workers=max_workers,
            sites_per_worker=sites_per_worker,
        )

        if out_fpath is not None:
            agg.save_agg_to_h5(h5_fpath, out_fpath, aggregation)

        return aggregation
