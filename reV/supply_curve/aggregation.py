# -*- coding: utf-8 -*-
"""reV supply curve point aggregation framework.

Created on Fri Jun 21 13:24:31 2019

@author: gbuster
"""
import concurrent.futures as cf
import os
import h5py
import numpy as np
import pandas as pd
from warnings import warn
import logging

from reV.handlers.outputs import Outputs
from reV.supply_curve.points import (ExclusionPoints, SupplyCurvePoint,
                                     SupplyCurveExtent)
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      OutputWarning, FileInputError)


logger = logging.getLogger(__name__)


class SupplyCurvePointSummary(SupplyCurvePoint):
    """Supply curve summary framework with extra methods for summary calc."""

    # technology-dependent power density estimates in MW/km2
    POWER_DENSITY = {'pv': 36, 'wind': 3}

    def __init__(self, gid, f_excl, f_gen, f_techmap, tm_dset_gen, tm_dset_res,
                 res_class_dset=None, res_class_bin=None, ex_area=0.0081,
                 power_density=None, dset_cf='cf_mean-means',
                 dset_lcoe='lcoe_fcr-means', resolution=64,
                 exclusion_shape=None,
                 close=False):
        """
        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        f_excl : str | ExclusionPoints
            Filepath to exclusions geotiff or ExclusionPoints file handler.
        f_gen : str | reV.handlers.Outputs
            Filepath to .h5 reV generation output results or reV Outputs file
            handler.
        f_techmap : str | h5py.File
            Filepath to tech mapping between exclusions and resource data
            (created using the reV TechMapping framework) or an h5py file
            handler object.
        tm_dset_gen : str
            Dataset name in the techmap file containing the
            exclusions-to-generation mapping data.
        tm_dset_res : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        res_class_dset : str | np.ndarray | None
            Dataset in the generation file dictating resource classes.
            Can be pre-extracted resource data in np.ndarray.
            None if no resource classes.
        res_class_bin : list | None
            Two-entry lists dictating the single resource class bin.
            None if no resource classes.
        ex_area : float
            Area of an exclusion cell (square km).
        power_density : float | None
            Power density in MW/km2. None will attempt to infer power density
            from the generation meta data technology.
        dset_cf : str | np.ndarray
            Dataset name from f_gen containing capacity factor mean values.
            Can be pre-extracted generation output data in np.ndarray.
        dset_lcoe : str | np.ndarray
            Dataset name from f_gen containing LCOE mean values.
            Can be pre-extracted generation output data in np.ndarray.
        resolution : int | None
            SC resolution, must be input in combination with gid.
        exclusion_shape : tuple
            Shape of the exclusions extent (rows, cols). Inputing this will
            speed things up considerably.
        close : bool
            Flag to close object file handlers on exit.
        """

        self._res_class_dset = res_class_dset
        self._res_class_bin = res_class_bin
        self._dset_cf = dset_cf
        self._dset_lcoe = dset_lcoe
        self._res_gid_set = None
        self._gen_gid_set = None
        self._mean_res = None
        self._res_data = None
        self._gen_data = None
        self._lcoe_data = None
        self._ex_area = ex_area
        self._power_density = power_density

        super().__init__(gid, f_excl, f_gen, f_techmap, tm_dset_gen,
                         tm_dset_res, resolution=resolution,
                         exclusion_shape=exclusion_shape, close=close)

        self._apply_exclusions()

    def _apply_exclusions(self):
        """Apply exclusions by masking the generation and resource gid arrays.
        This removes all res/gen entries that are masked by the exclusions or
        resource bin."""

        exclude = (self.excl_data == 0)
        exclude = self._resource_exclusion(exclude)

        self._gen_gids = self._gen_gids[~exclude]
        self._res_gids = self._res_gids[~exclude]

        if (self._gen_gids != -1).sum() == 0:
            msg = ('Supply curve point gid {} is completely excluded for res '
                   'bin: {}'.format(self._gid, self._res_class_bin))
            raise EmptySupplyCurvePointError(msg)

    def _resource_exclusion(self, boolean_exclude):
        """Include the resource exclusion into a pre-existing bool exclusion.

        Parameters
        ----------
        boolean_exclude : np.ndarray
            Boolean exclusion array (True is exclude).

        Returns
        -------
        boolean_exclude : np.ndarray
            Same as input but includes additional exclusions for resource
            outside of current resource class bin.
        """

        if (self._res_class_dset is not None and
                self._res_class_bin is not None):

            rex = ((self.res_data[self._gen_gids] <
                    np.min(self._res_class_bin)) |
                   (self.res_data[self._gen_gids] >=
                    np.max(self._res_class_bin)))

            boolean_exclude = (boolean_exclude | rex)

        return boolean_exclude

    @staticmethod
    def ordered_unique(seq):
        """Get a list of unique values in the same order as the input sequence.

        Parameters
        ----------
        seq : list | tuple
            Sequence of values.

        Returns
        -------
        seq : list
            List of unique values in seq input with original order.
        """

        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    @property
    def area(self):
        """Get the non-excluded resource area of the supply curve point in the
        current resource class.

        Returns
        -------
        area : float
            Non-excluded resource/generation area in square km.
        """
        return (self._res_gids != -1).sum() * self._ex_area

    @property
    def res_data(self):
        """Get the resource data array.

        Returns
        -------
        _res_data : np.ndarray
            Multi-year-mean resource data array for all sites in the
            generation data output file.
        """

        if isinstance(self._res_class_dset, np.ndarray):
            return self._res_class_dset

        else:
            if self._res_data is None:
                self._res_data = self.gen[self._res_class_dset]

        return self._res_data

    @property
    def gen_data(self):
        """Get the generation capacity factor data array.

        Returns
        -------
        _gen_data : np.ndarray
            Multi-year-mean capacity factor data array for all sites in the
            generation data output file.
        """

        if isinstance(self._dset_cf, np.ndarray):
            return self._dset_cf

        else:
            if self._gen_data is None:
                if self._dset_cf in self.gen.dsets:
                    self._gen_data = self.gen[self._dset_cf]

        return self._gen_data

    @property
    def lcoe_data(self):
        """Get the LCOE data array.

        Returns
        -------
        _lcoe_data : np.ndarray
            Multi-year-mean LCOE data array for all sites in the
            generation data output file.
        """

        if isinstance(self._dset_lcoe, np.ndarray):
            return self._dset_lcoe

        else:
            if self._lcoe_data is None:
                if self._dset_lcoe in self.gen.dsets:
                    self._lcoe_data = self.gen[self._dset_lcoe]

        return self._lcoe_data

    @property
    def latitude(self):
        """Get the SC point latitude"""
        return self.centroid[0]

    @property
    def longitude(self):
        """Get the SC point longitude"""
        return self.centroid[1]

    @property
    def res_gid_set(self):
        """Get the list of resource gids corresponding to this sc point.

        Returns
        -------
        res_gids : list
            List of resource gids.
        """
        if self._res_gid_set is None:
            self._res_gid_set = self.ordered_unique(self._res_gids)
            if -1 in self._res_gid_set:
                self._res_gid_set.remove(-1)
        return self._res_gid_set

    @property
    def gen_gid_set(self):
        """Get the list of generation gids corresponding to this sc point.

        Returns
        -------
        gen_gids : list
            List of generation gids.
        """
        if self._gen_gid_set is None:
            self._gen_gid_set = self.ordered_unique(self._gen_gids)
            if -1 in self._gen_gid_set:
                self._gen_gid_set.remove(-1)
        return self._gen_gid_set

    @property
    def gid_counts(self):
        """Get the number of exclusion pixels in each resource/generation gid
        corresponding to this sc point.

        Returns
        -------
        gid_counts : list
            List of exclusion pixels in each resource/generation gid.
        """
        return [(self._res_gids == gid).sum() for gid in self.res_gid_set]

    @property
    def mean_cf(self):
        """Get the mean capacity factor for the non-excluded data.

        Returns
        -------
        mean_cf : float
            Mean capacity factor value for the non-excluded data.
        """
        mean_cf = None
        if self.gen_data is not None:
            mean_cf = self.gen_data[self._gen_gids].mean()
        return mean_cf

    @property
    def mean_lcoe(self):
        """Get the mean LCOE for the non-excluded data.

        Returns
        -------
        mean_lcoe : float
            Mean LCOE value for the non-excluded data.
        """
        mean_lcoe = None
        if self.lcoe_data is not None:
            mean_lcoe = self.lcoe_data[self._gen_gids].mean()
        return mean_lcoe

    @property
    def power_density(self):
        """Get the estimated power density either from input or infered from
        generation output meta.

        Returns
        -------
        _power_density : float
            Estimated power density in MW/km2
        """

        if self._power_density is None:
            tech = self.gen.meta['reV_tech'][0]
            if tech in self.POWER_DENSITY:
                self._power_density = self.POWER_DENSITY[tech]
            else:
                warn('Could not recognize reV technology in generation meta '
                     'data: "{}". Cannot lookup an appropriate power density '
                     'to calculate SC point capacity.'.format(tech))
        return self._power_density

    @property
    def capacity(self):
        """Get the estimated capacity in MW of the supply curve point in the
        current resource class with the applied exclusions.

        Returns
        -------
        capacity : float
            Estimated capacity in MW of the supply curve point in the
            current resource class with the applied exclusions.
        """

        capacity = None
        if self.power_density is not None:
            capacity = self.area * self.power_density
        return capacity

    @classmethod
    def summary(cls, gid, fpath_excl, fpath_gen, fpath_techmap, dset_tm,
                gen_index, args=None, **kwargs):
        """Get a summary dictionary of a single supply curve point.

        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        fpath_techmap : str
            Filepath to tech mapping between exclusions and generation results
            (created using the reV TechMapping framework).
        dset_tm : str
            Dataset name in the techmap file containing the
            exclusions-to-generation mapping data.
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        args : tuple | list | None
            List of summary arguments to include. None defaults to all
            available args defined in the class attr.
        kwargs : dict
            Keyword args to init the SC point.

        Returns
        -------
        summary : dict
            Dictionary of summary outputs for this sc point.
        """

        with cls(gid, fpath_excl, fpath_gen, fpath_techmap, dset_tm, gen_index,
                 **kwargs) as point:

            ARGS = {'res_gids': point.res_gid_set,
                    'gen_gids': point.gen_gid_set,
                    'gid_counts': point.gid_counts,
                    'mean_cf': point.mean_cf,
                    'mean_lcoe': point.mean_lcoe,
                    'capacity': point.capacity,
                    'area_sq_km': point.area,
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    }

            if args is None:
                args = list(ARGS.keys())

            summary = {}
            for arg in args:
                if arg in ARGS:
                    summary[arg] = ARGS[arg]
                else:
                    warn('Cannot find "{}" as an available SC point summary '
                         'output', OutputWarning)

        return summary


class Aggregation:
    """Supply points aggregation framework."""

    def __init__(self, fpath_excl, fpath_gen, fpath_techmap, dset_tm,
                 res_class_dset=None, res_class_bins=None,
                 dset_cf='cf_mean-means', dset_lcoe='lcoe_fcr-means',
                 resolution=64, gids=None, n_cores=None):
        """
        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        fpath_techmap : str
            Filepath to tech mapping between exclusions and generation results
            The tech mapping module will be run if this file does not exist.
        dset_tm : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of two-entry lists dictating the resource class bins.
            None if no resource classes.
        dset_cf : str
            Dataset name from f_gen containing capacity factor mean values.
        dset_lcoe : str
            Dataset name from f_gen containing LCOE mean values.
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        n_cores : int | None
            Number of cores to run summary on. 1 is serial, None is all
            available cpus.
        """

        self._fpath_excl = fpath_excl
        self._fpath_gen = fpath_gen
        self._fpath_techmap = fpath_techmap
        self._dset_tm = dset_tm
        self._res_class_dset = res_class_dset
        self._res_class_bins = res_class_bins
        self._dset_cf = dset_cf
        self._dset_lcoe = dset_lcoe
        self._resolution = resolution

        if n_cores is None:
            n_cores = os.cpu_count()
        self._n_cores = n_cores

        if gids is None:
            with SupplyCurveExtent(fpath_excl, resolution=resolution) as sc:
                gids = np.array(range(len(sc)), dtype=np.uint32)
        elif not isinstance(gids, np.ndarray):
            gids = np.array(gids)
        self._gids = gids

        self._check_files()
        self._gen_index = self._parse_gen_index(self._fpath_gen)

    def _check_files(self):
        """Do a preflight check on input files"""

        for fpath in (self._fpath_excl, self._fpath_gen, self._fpath_techmap):
            if not os.path.exists(fpath):
                raise FileNotFoundError('Could not find required input file: '
                                        '{}'.format(fpath))

        with h5py.File(self._fpath_techmap, 'r') as f:
            if self._dset_tm not in f:
                raise FileInputError('Could not find "{}" in techmap file: {}'
                                     .format(self._dset_tm,
                                             self._fpath_techmap))

    @staticmethod
    def _parse_gen_index(fpath_gen):
        """Parse gen outputs for an array of generation gids corresponding to
        the resource gids.

        Parameters
        ----------
        fpath_gen : str
            Filepath to .h5 reV generation output results.

        Returns
        -------
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        """

        with Outputs(fpath_gen, mode='r') as gen:
            gen_index = gen.meta

        gen_index = gen_index.rename(columns={'gid': 'res_gids'})
        gen_index['gen_gids'] = gen_index.index
        gen_index = gen_index[['res_gids', 'gen_gids']]
        gen_index = gen_index.set_index(keys='res_gids')
        gen_index = gen_index.reindex(range(gen_index.index.max() + 1))
        gen_index = gen_index['gen_gids'].values
        gen_index[np.isnan(gen_index)] = -1
        gen_index = gen_index.astype(np.int32)

        return gen_index

    @staticmethod
    def _serial_summary(fpath_excl, fpath_gen, fpath_techmap, dset_tm,
                        gen_index, res_class_dset=None, res_class_bins=None,
                        dset_cf='cf_mean-means', dset_lcoe='lcoe_fcr-means',
                        resolution=64, gids=None, **kwargs):
        """Standalone method to create agg summary - can be parallelized.

        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        fpath_techmap : str
            Filepath to tech mapping between exclusions and generation results
            (created using the reV TechMapping framework).
        dset_tm : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of two-entry lists dictating the resource class bins.
            None if no resource classes.
        dset_cf : str
            Dataset name from f_gen containing capacity factor mean values.
        dset_lcoe : str
            Dataset name from f_gen containing LCOE mean values.
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        kwargs : dict
            Namespace of additional keyword args to init
            SupplyCurvePointSummary.

        Returns
        -------
        summary : list
            List of dictionaries, each being an SC point summary.
        """

        summary = []

        with SupplyCurveExtent(fpath_excl, resolution=resolution) as sc:

            exclusion_shape = sc.exclusions.shape

            if gids is None:
                gids = range(len(sc))

            # pre-extract handlers so they are not repeatedly initialized
            with ExclusionPoints(fpath_excl) as excl:
                with Outputs(fpath_gen, mode='r') as gen:
                    with h5py.File(fpath_techmap, 'r') as techmap:

                        # pre-extract data before iteration
                        _ = gen.meta

                        if res_class_dset is None:
                            res_data = None
                            res_class_bins = [None]
                        else:
                            res_data = gen[res_class_dset]

                        if dset_cf in gen.dsets:
                            cf_data = gen[dset_cf]
                        else:
                            cf_data = None
                            warn('Could not find cf dataset "{}" in '
                                 'generation file: {}'
                                 .format(dset_cf, fpath_gen), OutputWarning)
                        if dset_lcoe in gen.dsets:
                            lcoe_data = gen[dset_lcoe]
                        else:
                            lcoe_data = None
                            warn('Could not find lcoe dataset "{}" in '
                                 'generation file: {}'
                                 .format(dset_lcoe, fpath_gen), OutputWarning)

                        for gid in gids:

                            for ri, res_bin in enumerate(res_class_bins):
                                try:
                                    pointsum = SupplyCurvePointSummary.summary(
                                        gid, excl, gen, techmap,
                                        dset_tm, gen_index,
                                        res_class_dset=res_data,
                                        res_class_bin=res_bin,
                                        dset_cf=cf_data,
                                        dset_lcoe=lcoe_data,
                                        resolution=resolution,
                                        exclusion_shape=exclusion_shape,
                                        **kwargs)

                                except EmptySupplyCurvePointError:
                                    pass

                                else:
                                    pointsum['sc_gid'] = gid
                                    pointsum['sc_row_ind'] = \
                                        sc.points.loc[gid, 'row_ind']
                                    pointsum['sc_col_ind'] = \
                                        sc.points.loc[gid, 'col_ind']
                                    pointsum['res_class'] = ri

                                    summary.append(pointsum)

        return summary

    def _parallel_summary(self, **kwargs):
        """Get the supply curve points aggregation summary using futures.

        Parameters
        ----------
        kwargs : dict
            Namespace of additional keyword args to init
            SupplyCurvePointSummary.

        Returns
        -------
        summary : list
            List of dictionaries, each being an SC point summary.
        """

        chunks = np.array_split(self._gids,
                                int(np.ceil(len(self._gids) / 1000)))

        logger.info('Running supply curve point aggregation for '
                    'points {} through {} at a resolution of {} '
                    'on {} cores in {} chunks.'
                    .format(self._gids[0], self._gids[-1], self._resolution,
                            self._n_cores, len(chunks)))

        n_finished = 0
        futures = []
        summary = []

        with cf.ProcessPoolExecutor(max_workers=self._n_cores) as executor:

            # iterate through split executions, submitting each to worker
            for gid_set in chunks:
                # submit executions and append to futures list
                futures.append(executor.submit(
                    self._serial_summary,
                    self._fpath_excl, self._fpath_gen, self._fpath_techmap,
                    self._dset_tm, self._gen_index,
                    res_class_dset=self._res_class_dset,
                    res_class_bins=self._res_class_bins,
                    dset_cf=self._dset_cf, dset_lcoe=self._dset_lcoe,
                    resolution=self._resolution,
                    gids=gid_set, **kwargs))

            # gather results
            for future in cf.as_completed(futures):
                n_finished += 1
                logger.info('Parallel aggregation futures collected: '
                            '{} out of {}'
                            .format(n_finished, len(chunks)))
                summary += future.result()

        return summary

    @classmethod
    def summary(cls, fpath_excl, fpath_gen, fpath_techmap, dset_tm,
                res_class_dset=None, res_class_bins=None,
                dset_cf='cf_mean-means', dset_lcoe='lcoe_fcr-means',
                resolution=64, gids=None, n_cores=None, option='dataframe',
                **kwargs):
        """Get the supply curve points aggregation summary.

        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        fpath_techmap : str
            Filepath to tech mapping between exclusions and generation results
            The tech mapping module will be run if this file does not exist.
        dset_tm : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of two-entry lists dictating the resource class bins.
            None if no resource classes.
        dset_cf : str
            Dataset name from f_gen containing capacity factor mean values.
        dset_lcoe : str
            Dataset name from f_gen containing LCOE mean values.
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        n_cores : int | None
            Number of cores to run summary on. 1 is serial, None is all
            available cpus.
        option : str
            Output dtype option (dict, dataframe).
        kwargs : dict
            Namespace of additional keyword args to init
            SupplyCurvePointSummary.

        Returns
        -------
        summary : list | DataFrame
            Summary of the SC points.
        """

        agg = cls(fpath_excl, fpath_gen, fpath_techmap, dset_tm,
                  resolution=resolution, gids=gids,
                  res_class_dset=res_class_dset, res_class_bins=res_class_bins,
                  dset_cf=dset_cf, dset_lcoe=dset_lcoe, n_cores=n_cores)

        if n_cores == 1:
            summary = agg._serial_summary(agg._fpath_excl, agg._fpath_gen,
                                          agg._fpath_techmap,
                                          agg._dset_tm, agg._gen_index,
                                          res_class_dset=agg._res_class_dset,
                                          res_class_bins=agg._res_class_bins,
                                          dset_cf=agg._dset_cf,
                                          dset_lcoe=agg._dset_lcoe,
                                          resolution=agg._resolution,
                                          gids=gids, **kwargs)
        else:
            summary = agg._parallel_summary(**kwargs)

        if 'dataframe' in option.lower():
            summary = pd.DataFrame(summary)
            summary = summary.sort_values('sc_gid')

        return summary
