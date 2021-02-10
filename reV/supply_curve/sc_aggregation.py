# -*- coding: utf-8 -*-
"""reV supply curve aggregation framework.

Created on Fri Jun 21 13:24:31 2019

@author: gbuster
"""
from concurrent.futures import as_completed
import h5py
import logging
import numpy as np
import os
import pandas as pd
from scipy.spatial import cKDTree
from warnings import warn

from reV.generation.base import BaseGen
from reV.handlers.exclusions import ExclusionLayers
from reV.offshore.offshore import Offshore as OffshoreClass
from reV.supply_curve.aggregation import (AbstractAggFileHandler,
                                          AbstractAggregation,
                                          Aggregation)
from reV.supply_curve.exclusions import FrictionMask
from reV.supply_curve.points import SupplyCurveExtent
from reV.supply_curve.point_summary import SupplyCurvePointSummary
from reV.utilities.exceptions import (EmptySupplyCurvePointError,
                                      OutputWarning, FileInputError,
                                      InputWarning, SupplyCurveInputError)

from rex.resource import Resource
from rex.multi_file_resource import MultiFileResource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.utilities import get_lat_lon_cols

logger = logging.getLogger(__name__)


class SupplyCurveAggFileHandler(AbstractAggFileHandler):
    """
    Framework to handle aggregation summary context managers:
    - exclusions .h5 file
    - generation .h5 file
    - econ .h5 file (optional)
    - friction surface .h5 file (optional)
    - variable power density .csv (optional)
    """

    def __init__(self, excl_fpath, gen_fpath, econ_fpath=None,
                 data_layers=None, power_density=None, excl_dict=None,
                 friction_fpath=None, friction_dset=None,
                 area_filter_kernel='queen', min_area=None,
                 check_excl_layers=False):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        econ_fpath : str | None
            Filepath to .h5 reV econ output results. This is optional and only
            used if the lcoe_dset is not present in the gen_fpath file.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        power_density : float | str | None
            Power density in MW/km2 or filepath to variable power
            density file. None will attempt to infer a constant
            power density from the generation meta data technology.
            Variable power density csvs must have "gid" and "power_density"
            columns where gid is the resource gid (typically wtk or nsrdb gid)
            and the power_density column is in MW/km2.
        excl_dict : dict | None
            Dictionary of exclusion LayerMask arugments {layer: {kwarg: value}}
        friction_fpath : str | None
            Filepath to friction surface data (cost based exclusions).
            Must be paired with friction_dset. The friction data must be the
            same shape as the exclusions. Friction input creates a new output
            "mean_lcoe_friction" which is the nominal LCOE multiplied by the
            friction data.
        friction_dset : str | None
            Dataset name in friction_fpath for the friction surface data.
            Must be paired with friction_fpath. Must be same shape as
            exclusions.
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

        self._gen = self._open_gen_econ_resource(gen_fpath, econ_fpath)
        # pre-initialize any import attributes
        _ = self._gen.meta

        self._data_layers = self._open_data_layers(data_layers)
        self._power_density = power_density
        self._parse_power_density()

        self._friction_layer = None
        if friction_fpath is not None and friction_dset is not None:
            self._friction_layer = FrictionMask(friction_fpath, friction_dset)

            if not np.all(self._friction_layer.shape == self._excl.shape):
                e = ('Friction layer shape {} must match exclusions shape {}!'
                     .format(self._friction_layer.shape, self._excl.shape))
                logger.error(e)
                raise FileInputError(e)

    @staticmethod
    def _open_gen_econ_resource(gen_fpath, econ_fpath):
        """Open a rex resource file handler for the reV generation and
        (optionally) the reV econ output(s).

        Parameters
        ----------
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        econ_fpath : str | None
            Filepath to .h5 reV econ output results. This is optional and only
            used if the lcoe_dset is not present in the gen_fpath file.

        Returns
        -------
        handler : Resource | MultiFileResource
            Open resource handler initialized with gen_fpath and
            (optionally) econ_fpath.
        """

        if econ_fpath is None:
            handler = Resource(gen_fpath)
        else:
            handler = MultiFileResource([gen_fpath, econ_fpath],
                                        check_files=True)

        return handler

    def _open_data_layers(self, data_layers):
        """Open data layer Exclusion h5 handlers.

        Parameters
        ----------
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".

        Returns
        -------
        data_layers : None | dict
            Aggregation data layers. fobj is added to the dictionary of each
            layer.
        """

        if data_layers is not None:
            for name, attrs in data_layers.items():
                data_layers[name]['fobj'] = self._excl.excl_h5
                if 'fpath' in attrs:
                    if attrs['fpath'] != self._excl_fpath:
                        data_layers[name]['fobj'] = ExclusionLayers(
                            attrs['fpath'])

        return data_layers

    @staticmethod
    def _close_data_layers(data_layers):
        """Close all data layers with exclusion h5 handlers.

        Parameters
        ----------
        data_layers : None | dict
            Aggregation data layers. Must have fobj exclusion handlers to close
        """

        if data_layers is not None:
            for layer in data_layers.values():
                if 'fobj' in layer:
                    layer['fobj'].close()

    def _parse_power_density(self):
        """Parse the power density input. If file, open file handler."""

        if isinstance(self._power_density, str):
            self._pdf = self._power_density

            if self._pdf.endswith('.csv'):
                self._power_density = pd.read_csv(self._pdf)
                if ('gid' in self._power_density
                        and 'power_density' in self._power_density):
                    self._power_density = self._power_density.set_index('gid')
                else:
                    msg = ('Variable power density file must include "gid" '
                           'and "power_density" columns, but received: {}'
                           .format(self._power_density.columns.values))
                    logger.error(msg)
                    raise FileInputError(msg)
            else:
                msg = ('Variable power density file must be csv but received: '
                       '{}'.format(self._pdf))
                logger.error(msg)
                raise FileInputError(msg)

    def close(self):
        """Close all file handlers."""
        self._excl.close()
        self._gen.close()
        self._close_data_layers(self._data_layers)
        if self._friction_layer is not None:
            self._friction_layer.close()

    @property
    def gen(self):
        """Get the gen file handler object.

        Returns
        -------
        _gen : Outputs
            reV gen outputs handler object.
        """
        return self._gen

    @property
    def data_layers(self):
        """Get the data layers object.

        Returns
        -------
        _data_layers : dict
            Data layers namespace.
        """
        return self._data_layers

    @property
    def power_density(self):
        """Get the power density object.

        Returns
        -------
        _power_density : float | None | pd.DataFrame
            Constant power density float, None, or opened dataframe with
            (resource) "gid" and "power_density columns".
        """
        return self._power_density

    @property
    def friction_layer(self):
        """Get the friction layer (cost based exclusions).

        Returns
        -------
        friction_layer : None | FrictionMask
            Friction layer with scalar friction values if valid friction inputs
            were entered. Otherwise, None to not apply friction layer.
        """
        return self._friction_layer


class OffshoreAggregation:
    """Offshore aggregation utility methods."""

    @staticmethod
    def _get_res_class(res_class_bins, res_data, gen_gid):
        """Get the offshore resource class for one site if data is available.

        Parameters
        ----------
        res_class_bins : list
            List of resource class bins. Can be [None] if not resource
            classes are requested (output will be 0).
        res_data : np.ndarray | None
            Generation output array of site resource data indexed by gen_gid.
            None if no resource data was computed (output will be 0).
        gen_gid : int
            Generation gid (index) for site in question.
            This is used to index res_data.

        Returns
        -------
        res_class : int
            Resource class integer. Zero if required data is not available.
        """

        res_class = 0
        if (res_class_bins[0] is not None
                and res_data is not None):
            for ri, res_bin in enumerate(res_class_bins):

                c1 = res_data[gen_gid] > np.min(res_bin)
                c2 = res_data[gen_gid] < np.max(res_bin)

                if c1 and c2:
                    res_class = ri
                    break

        return res_class

    @staticmethod
    def _get_means(data, gen_gid):
        """Get the mean point data from the data array for gen_gid.

        Parameters
        ----------
        data : np.ndarray | None
            Array of mean data values or None if no data is available.
        gen_gid : int
            location to get mean data for

        Returns
        -------
        mean_data : float | None
            Mean data value or None if no data available.
        """

        mean_data = None
        if data is not None:
            mean_data = data[gen_gid]

        return mean_data

    @staticmethod
    def _agg_data_layers(data_layers, summary):
        """Agg categorical offshore data layers using NN to onshore points.

        Parameters
        ----------
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        summary : DataFrame
            Summary of the SC points.

        Returns
        -------
        summary : DataFrame
            Summary of the SC points with aggregated offshore data layers.
        """

        if 'offshore' in summary and data_layers is not None:
            cat_layers = [k for k, v in data_layers.items()
                          if v['method'].lower() == 'mode']

            if any(summary['offshore']) and any(cat_layers):
                logger.info('Aggregating the following columns for offshore '
                            'wind sites based on NN onshore sites: {}'
                            .format(cat_layers))
                offshore_mask = (summary['offshore'] == 1)
                offshore_summary = summary[offshore_mask]
                onshore_summary = summary[~offshore_mask]

                # pylint: disable=not-callable
                lat_lon_cols = get_lat_lon_cols(onshore_summary)
                tree = cKDTree(onshore_summary[lat_lon_cols])
                lat_lon_cols = get_lat_lon_cols(offshore_summary)
                _, nn = tree.query(offshore_summary[lat_lon_cols])

                for i, off_gid in enumerate(offshore_summary.index):
                    on_gid = onshore_summary.index.values[nn[i]]
                    logger.debug('Offshore gid {} is closest to onshore gid {}'
                                 .format(off_gid, on_gid))

                    for c in cat_layers:
                        summary.at[off_gid, c] = onshore_summary.at[on_gid, c]

        return summary

    @staticmethod
    def _parse_meta_cols(offshore_meta_cols, gen_meta):
        """Parse the offshore meta columns and return a validated list.

        Parameters
        ----------
        offshore_meta_cols : list | tuple | None
            Column labels from original offshore data file that were passed
            through to the offshore module output meta data. None will use
            Offshore class variable DEFAULT_META_COLS, and any
            additional requested cols will be added to DEFAULT_META_COLS.
        gen_meta : pd.DataFrame
            Meta data from the offshore generation output h5 file.

        Returns
        -------
        offshore_meta_cols : list
            List of offshore meta columns to include in aggregation summary,
            validated for columns that are present in gen_meta.
        """

        if offshore_meta_cols is None:
            offshore_meta_cols = list(OffshoreClass.DEFAULT_META_COLS)
        else:
            offshore_meta_cols = list(offshore_meta_cols)
            offshore_meta_cols += list(OffshoreClass.DEFAULT_META_COLS)
            offshore_meta_cols = list(set(offshore_meta_cols))

        missing = [c for c in offshore_meta_cols if c not in gen_meta]
        offshore_meta_cols = [c for c in offshore_meta_cols if c in gen_meta]

        if any(missing):
            msg = ('Requested offshore columns {} not found in '
                   'generation meta data. Not including in '
                   'aggregation output table.'.format(missing))
            logger.warning(msg)
            warn(msg)

        return offshore_meta_cols

    @staticmethod
    def _nearest_sc_points(excl_fpath, resolution, lat_lons):
        """
        Get nearest sc_points to offshore farms

        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        resolution : int
            SC resolution
        lat_lons : ndarray
            Offshore wind farm coordinates (lat, lon) pairs

        Returns
        -------
        points : pandas.DataFrame
            Nearest SC points (gid, row_ind, and col_ind)
        """
        with SupplyCurveExtent(excl_fpath, resolution=resolution) as f:
            points = f.points
            sc_lat_lons = f.lat_lon

        tree = cKDTree(sc_lat_lons)  # pylint: disable=not-callable
        _, pos = tree.query(lat_lons)

        return points.loc[pos].reset_index()

    @classmethod
    def run(cls, summary, handler, excl_fpath, res_data, res_class_bins,
            cf_data, lcoe_data, offshore_flag, h5_dsets_data=None,
            resolution=64, offshore_capacity=600, offshore_gid_counts=494,
            offshore_pixel_area=4, offshore_meta_cols=None):
        """Get the offshore supply curve point summary. Each offshore resource
        pixel will be summarized in its own supply curve point which will be
        added to the summary list.

        Parameters
        ----------
        summary : list
            List of dictionaries, each being an onshore SC point summary.
        handler : SupplyCurveAggFileHandler
            Instantiated SupplyCurveAggFileHandler.
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        res_data : np.ndarray | None
            Extracted resource data from res_class_dset
        res_class_bins : list
            List of resouce class bin ranges.
        cf_data : np.ndarray | None
            Capacity factor data extracted from cf_dset in gen
        lcoe_data : np.ndarray | None
            LCOE data extracted from lcoe_dset in gen
        offshore_flag : np.ndarray
            Array of offshore boolean flags if available from wind generation
            data. If this is input as None, this method has been called
            without offshore data in error and summary will be passed
            through un manipulated.
        h5_dsets_data : dict | None
            If additional h5_dsets are requested, this will be a dictionary
            keyed by the h5 dataset names. The corresponding values will be
            the extracted arrays from the h5 files.
        resolution : int, optional
            SC resolution, by default 64
        offshore_capacity : int | float
            Offshore resource pixel generation capacity in MW.
        offshore_gid_counts : int
            Approximate number of exclusion pixels that would fall into an
            offshore pixel area.
        offshore_pixel_area : int | float
            Approximate area of offshore resource pixels in km2.
        offshore_meta_cols : list | tuple | None
            Column labels from original offshore data file that were passed
            through to the offshore module output meta data. None will use
            Offshore class variable DEFAULT_META_COLS, and any
            additional requested cols will be added to DEFAULT_META_COLS.

        Returns
        -------
        summary : list
            List of dictionaries, each being an SC point summary, includng SC
            points for single offshore resource pixels.
        """

        if offshore_flag is None:
            return summary

        for i, _ in enumerate(summary):
            summary[i]['offshore'] = 0

        offshore_meta_cols = cls._parse_meta_cols(offshore_meta_cols,
                                                  handler.gen.meta)

        gen_gids = list(range(len(offshore_flag)))
        lat_lon_cols = get_lat_lon_cols(handler.gen.meta)
        lat_lon = handler.gen.meta.loc[gen_gids, lat_lon_cols].values
        sc_points = cls._nearest_sc_points(excl_fpath, resolution, lat_lon)

        for gen_gid, offshore in enumerate(offshore_flag):
            if offshore:
                if 'offshore_res_gids' not in handler.gen.meta:
                    e = ('Offshore sites found in wind data, but '
                         '"offshore_res_gids" not found in the '
                         'meta data. You must run the offshore wind '
                         'farm module before offshore supply curve.')
                    logger.error(e)
                    raise SupplyCurveInputError(e)

                # pylint: disable-msg=E1101
                farm_gid = handler.gen.meta.loc[gen_gid, 'gid']
                latitude = handler.gen.meta.loc[gen_gid, lat_lon_cols[0]]
                longitude = handler.gen.meta.loc[gen_gid, lat_lon_cols[1]]
                timezone = handler.gen.meta.loc[gen_gid, 'timezone']
                res_gids = handler.gen.meta\
                    .loc[gen_gid, 'offshore_res_gids']

                res_class = cls._get_res_class(res_class_bins, res_data,
                                               gen_gid)
                cf = cls._get_means(cf_data, gen_gid)
                lcoe = cls._get_means(lcoe_data, gen_gid)
                res = cls._get_means(res_data, gen_gid)

                pointsum = {'sc_point_gid': sc_points.loc[gen_gid, 'gid'],
                            'sc_row_ind': sc_points.loc[gen_gid, 'row_ind'],
                            'sc_col_ind': sc_points.loc[gen_gid, 'col_ind'],
                            'farm_gid': farm_gid,
                            'res_gids': res_gids,
                            'gen_gids': [gen_gid],
                            'gid_counts': [int(offshore_gid_counts)],
                            'mean_cf': cf,
                            'mean_lcoe': lcoe,
                            'mean_res': res,
                            'capacity': offshore_capacity,
                            'area_sq_km': offshore_pixel_area,
                            'latitude': latitude,
                            'longitude': longitude,
                            'res_class': res_class,
                            'timezone': timezone,
                            'elevation': 0,
                            'offshore': 1,
                            }

                for label in offshore_meta_cols:
                    pointsum[label] = handler.gen.meta.loc[gen_gid, label]

                if h5_dsets_data is not None:
                    for dset, data in h5_dsets_data.items():
                        label = 'mean_{}'.format(dset)
                        pointsum[label] = cls._get_means(data, gen_gid)

                summary.append(pointsum)

        return summary


class SupplyCurveAggregation(AbstractAggregation):
    """
    Supply curve points aggregation framework.

    Examples
    --------
    Standard outputs:

    sc_gid : int
        Unique supply curve gid. This is the enumerated supply curve points,
        which can have overlapping geographic locations due to different
        resource bins at the same geographic SC point.
    res_gids : list
        Stringified list of resource gids (e.g. original WTK or NSRDB resource
        GIDs) corresponding to each SC point.
    gen_gids : list
        Stringified list of generation gids (e.g. GID in the reV generation
        output, which corresponds to the reV project points and not
        necessarily the resource GIDs).
    gid_counts : list
        Stringified list of the sum of inclusion scalar values corresponding
        to each gen_gid and res_gid, where 1 is included, 0 is excluded, and
        0.7 is included with 70 percent of available land. Each entry in this
        list is associated with the corresponding entry in the gen_gids and
        res_gids lists.
    n_gids : int
        Total number of included pixels. This is a boolean sum and considers
        partial inclusions to be included (e.g. 1).
    mean_cf : float
        Mean capacity factor of each supply curve point (the arithmetic mean is
        weighted by the inclusion layer) (unitless).
    mean_lcoe : float
        Mean LCOE of each supply curve point (the arithmetic mean is weighted
        by the inclusion layer). Units match the reV econ output ($/MWh).
    mean_res : float
        Mean resource, the resource dataset to average is provided by the user
        in 'res_class_dset'. The arithmetic mean is weighted by the inclusion
        layer.
    capacity : float
        Total capacity of each supply curve point (MW). Units are contingent on
        the 'power_density' input units of MW/km2.
    area_sq_km : float
        Total included area for each supply curve point in km2. This is based
        on the nominal area of each exclusion pixel which by default is
        calculated from the exclusion profile attributes. The NREL reV default
        is 0.0081 km2 pixels (90m x 90m). The area sum considers partial
        inclusions.
    latitude : float
        Supply curve point centroid latitude coordinate, in degrees
        (does not consider exclusions).
    longitude : float
        Supply curve point centroid longitude coordinate, in degrees
        (does not consider exclusions).
    country : str
        Country of the supply curve point based on the most common country
        of the associated resource meta data. Does not consider exclusions.
    state : str
        State of the supply curve point based on the most common state
        of the associated resource meta data. Does not consider exclusions.
    county : str
        County of the supply curve point based on the most common county
        of the associated resource meta data. Does not consider exclusions.
    elevation : float
        Mean elevation of the supply curve point based on the mean elevation
        of the associated resource meta data. Does not consider exclusions.
    timezone : int
        UTC offset of local timezone based on the most common timezone of the
        associated resource meta data. Does not consider exclusions.
    sc_point_gid : int
        Spatially deterministic supply curve point gid. Duplicate sc_point_gid
        values can exist due to resource binning.
    sc_row_ind : int
        Row index of the supply curve point in the aggregated exclusion grid.
    sc_col_ind : int
        Column index of the supply curve point in the aggregated exclusion grid
    res_class : int
        Resource class for the supply curve gid. Each geographic supply curve
        point (sc_point_gid) can have multiple resource classes associated with
        it, resulting in multiple supply curve gids (sc_gid) associated with
        the same spatially deterministic supply curve point.


    Optional outputs:

    mean_friction : float
        Mean of the friction data provided in 'friction_fpath' and
        'friction_dset'. The arithmetic mean is weighted by boolean
        inclusions and considers partial inclusions to be included.
    mean_lcoe_friction : float
        Mean of the nominal LCOE multiplied by mean_friction value.
    mean_{dset} : float
        Mean input h5 dataset(s) provided by the user in 'h5_dsets'. These
        mean calculations are weighted by the partial inclusion layer.
    data_layers : float | int | str | dict
        Requested data layer aggregations, each data layer must be the same
        shape as the exclusion layers.
        - mode: int | str
            Most common value of a given data layer after applying the
            boolean inclusion mask.
        - mean : float
            Arithmetic mean value of a given data layer weighted by the
            scalar inclusion mask (considers partial inclusions).
        - min : float | int
            Minimum value of a given data layer after applying the
            boolean inclusion mask.
        - max : float | int
            Maximum value of a given data layer after applying the
            boolean inclusion mask.
        - sum : float
            Sum of a given data layer weighted by the scalar inclusion mask
            (considers partial inclusions).
        - category : dict
            Dictionary mapping the unique values in the data_layer to the
            sum of inclusion scalar values associated with all pixels with that
            unique value.
    """
    def __init__(self, excl_fpath, gen_fpath, tm_dset, econ_fpath=None,
                 excl_dict=None, area_filter_kernel='queen', min_area=None,
                 check_excl_layers=False, resolution=64, excl_area=None,
                 gids=None, res_class_dset=None, res_class_bins=None,
                 cf_dset='cf_mean-means', lcoe_dset='lcoe_fcr-means',
                 h5_dsets=None, data_layers=None, power_density=None,
                 friction_fpath=None, friction_dset=None, cap_cost_scale=None):
        """
        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        econ_fpath : str | None
            Filepath to .h5 reV econ output results. This is optional and only
            used if the lcoe_dset is not present in the gen_fpath file.
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
            List of gids to get summary for (can use to subset if running in
            parallel), or None for all gids in the SC extent.
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of floats or ints (bin edges) to convert to list of two-entry
            bin boundaries or list of two-entry bind boundaries in final format
        cf_dset : str
            Dataset name from f_gen containing capacity factor mean values.
        lcoe_dset : str
            Dataset name from f_gen containing LCOE mean values.
        h5_dsets : list | None
            Optional list of additional datasets from the source h5 gen/econ
            files to aggregate.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        power_density : float | str | None
            Power density in MW/km2 or filepath to variable power
            density file. None will attempt to infer a constant
            power density from the generation meta data technology.
            Variable power density csvs must have "gid" and "power_density"
            columns where gid is the resource gid (typically wtk or nsrdb gid)
            and the power_density column is in MW/km2.
        friction_fpath : str | None
            Filepath to friction surface data (cost based exclusions).
            Must be paired with friction_dset. The friction data must be the
            same shape as the exclusions. Friction input creates a new output
            "mean_lcoe_friction" which is the nominal LCOE multiplied by the
            friction data.
        friction_dset : str | None
            Dataset name in friction_fpath for the friction surface data.
            Must be paired with friction_fpath. Must be same shape as
            exclusions.
        cap_cost_scale : str | None
            Optional LCOE scaling equation to implement "economies of scale".
            Equations must be in python string format and return a scalar
            value to multiply the capital cost by. Independent variables in
            the equation should match the names of the columns in the reV
            supply curve aggregation table. This will not affect offshore
            wind LCOE.
        """

        super().__init__(excl_fpath, tm_dset, excl_dict=excl_dict,
                         area_filter_kernel=area_filter_kernel,
                         min_area=min_area,
                         check_excl_layers=check_excl_layers,
                         resolution=resolution, gids=gids)

        self._gen_fpath = gen_fpath
        self._econ_fpath = econ_fpath
        self._res_class_dset = res_class_dset
        self._res_class_bins = self._convert_bins(res_class_bins)
        self._cf_dset = cf_dset
        self._lcoe_dset = lcoe_dset
        self._h5_dsets = h5_dsets
        self._cap_cost_scale = cap_cost_scale
        self._power_density = power_density
        self._friction_fpath = friction_fpath
        self._friction_dset = friction_dset
        self._data_layers = data_layers

        logger.debug('Resource class bins: {}'.format(self._res_class_bins))

        if self._cap_cost_scale is not None:
            if self._h5_dsets is None:
                self._h5_dsets = []
            self._h5_dsets += list(BaseGen.LCOE_ARGS)
            self._h5_dsets = list(set(self._h5_dsets))

        if self._power_density is None:
            msg = ('Supply curve aggregation power density not specified. '
                   'Will try to infer based on lookup table: {}'
                   .format(SupplyCurvePointSummary.POWER_DENSITY))
            logger.warning(msg)
            warn(msg, InputWarning)

        self._check_data_layers()
        self._gen_index = Aggregation._parse_gen_index(self._gen_fpath)

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

        check_exists = [self._excl_fpath, self._gen_fpath]
        if self._econ_fpath is not None:
            check_exists.append(self._econ_fpath)

        for fpath in check_exists:
            if not os.path.exists(fpath):
                raise FileNotFoundError('Could not find input file: {}'
                                        .format(fpath))

        with h5py.File(self._excl_fpath, 'r') as f:
            if self._tm_dset not in f:
                raise FileInputError('Could not find techmap dataset "{}" '
                                     'in exclusions file: {}'
                                     .format(self._tm_dset,
                                             self._excl_fpath))

    def _check_data_layers(self, methods=('mean', 'max', 'min',
                           'mode', 'sum', 'category')):
        """Run pre-flight checks on requested aggregation data layers.

        Parameters
        ----------
        methods : list | tuple
            Data layer aggregation methods that are available to the user.
        """

        if self._data_layers is not None:

            with ExclusionLayers(self._excl_fpath) as f:
                shape_base = f.shape

            for k, v in self._data_layers.items():
                if 'dset' not in v:
                    raise KeyError('Data aggregation "dset" data layer "{}" '
                                   'must be specified.'.format(k))
                if 'method' not in v:
                    raise KeyError('Data aggregation "method" data layer "{}" '
                                   'must be specified.'.format(k))
                elif v['method'].lower() not in methods:
                    raise ValueError('Cannot recognize data layer agg method: '
                                     '"{}". Can only do: {}.'
                                     .format(v['method'], methods))
                if 'fpath' in v:
                    with ExclusionLayers(v['fpath']) as f:
                        if any(f.shape != shape_base):
                            msg = ('Data shape of data layer "{}" is {}, '
                                   'which does not match the baseline '
                                   'exclusions shape {}.'
                                   .format(k, f.shape, shape_base))
                            raise FileInputError(msg)

    @staticmethod
    def _get_input_data(gen, gen_fpath, econ_fpath, res_class_dset,
                        res_class_bins, cf_dset, lcoe_dset, h5_dsets):
        """Extract SC point agg input data args from higher level inputs.

        Parameters
        ----------
        gen : Resource | MultiFileResource
            Open rex resource handler initialized from gen_fpath and
            (optionally) econ_fpath.
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        econ_fpath : str
            Filepath to .h5 reV econ output results (optional argument if
            lcoe_dset is not found in gen_fpath).
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of two-entry lists dictating the resource class bins.
            None if no resource classes.
        cf_dset : str
            Dataset name from f_gen containing capacity factor mean values.
        lcoe_dset : str
            Dataset name from f_gen containing LCOE mean values.
        h5_dsets : list | None
            Optional list of additional datasets from the source h5 gen/econ
            files to aggregate.

        Returns
        -------
        res_data : np.ndarray | None
            Extracted resource data from res_class_dset
        res_class_bins : list
            List of resouce class bin ranges.
        cf_data : np.ndarray | None
            Capacity factor data extracted from cf_dset in gen
        lcoe_data : np.ndarray | None
            LCOE data extracted from lcoe_dset in gen
        offshore_flags : np.ndarray | None
            Array of offshore boolean flags if available from wind generation
            data. None if offshore flag is not available.
        h5_dsets_data : dict | None
            If additional h5_dsets are requested, this will be a dictionary
            keyed by the h5 dataset names. The corresponding values will be
            the extracted arrays from the h5 files.
        """

        if res_class_dset is None:
            res_data = None
            res_class_bins = [None]
        else:
            res_data = gen[res_class_dset]

        if res_class_bins is None:
            res_class_bins = [None]

        if cf_dset in gen.datasets:
            cf_data = gen[cf_dset]
        else:
            cf_data = None
            w = ('Could not find cf dataset "{}" in '
                 'generation file: {}. Available datasets: {}'
                 .format(cf_dset, gen_fpath, gen.datasets))
            logger.warning(w)
            warn(w, OutputWarning)

        if lcoe_dset in gen.datasets:
            lcoe_data = gen[lcoe_dset]
        else:
            lcoe_data = None
            w = ('Could not find lcoe dataset "{}" in generation file: {} or '
                 'econ file: {}. Available datasets: {}'
                 .format(lcoe_dset, gen_fpath, econ_fpath, gen.datasets))
            logger.warning(w)
            warn(w, OutputWarning)

        h5_dsets_data = None
        if h5_dsets is not None:
            h5_dsets_data = {}
            if not isinstance(h5_dsets, (list, tuple)):
                e = ('Additional h5_dsets argument must be a list or tuple '
                     'but received: {} {}'.format(type(h5_dsets), h5_dsets))
                logger.error(e)
                raise TypeError(e)
            else:
                for dset in h5_dsets:
                    if dset not in gen.datasets:
                        w = ('Could not find additional h5_dset "{}" in '
                             'generation file: {} or econ file: {}. '
                             'Available datasets: {}'
                             .format(dset, gen_fpath, econ_fpath,
                                     gen.datasets))
                        logger.warning(w)
                        warn(w, OutputWarning)
                    else:
                        h5_dsets_data[dset] = gen[dset]

        if 'offshore' in gen.meta:
            offshore_flag = gen.meta['offshore'].values
        else:
            offshore_flag = None

        return (res_data, res_class_bins, cf_data, lcoe_data, offshore_flag,
                h5_dsets_data)

    @classmethod
    def run_serial(cls, excl_fpath, gen_fpath, tm_dset, gen_index,
                   econ_fpath=None, excl_dict=None, area_filter_kernel='queen',
                   min_area=None, check_excl_layers=False, resolution=64,
                   gids=None, args=None, res_class_dset=None,
                   res_class_bins=None, cf_dset='cf_mean-means',
                   lcoe_dset='lcoe_fcr-means', h5_dsets=None, data_layers=None,
                   power_density=None, friction_fpath=None, friction_dset=None,
                   excl_area=0.0081, cap_cost_scale=None):
        """Standalone method to create agg summary - can be parallelized.

        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        tm_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        econ_fpath : str | None
            Filepath to .h5 reV econ output results. This is optional and only
            used if the lcoe_dset is not present in the gen_fpath file.
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
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of two-entry lists dictating the resource class bins.
            None if no resource classes.
        cf_dset : str
            Dataset name from f_gen containing capacity factor mean values.
        lcoe_dset : str
            Dataset name from f_gen containing LCOE mean values.
        h5_dsets : list | None
            Optional list of additional datasets from the source h5 gen/econ
            files to aggregate.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        power_density : float | str | None
            Power density in MW/km2 or filepath to variable power
            density file. None will attempt to infer a constant
            power density from the generation meta data technology.
            Variable power density csvs must have "gid" and "power_density"
            columns where gid is the resource gid (typically wtk or nsrdb gid)
            and the power_density column is in MW/km2.
        friction_fpath : str | None
            Filepath to friction surface data (cost based exclusions).
            Must be paired with friction_dset. The friction data must be the
            same shape as the exclusions. Friction input creates a new output
            "mean_lcoe_friction" which is the nominal LCOE multiplied by the
            friction data.
        friction_dset : str | None
            Dataset name in friction_fpath for the friction surface data.
            Must be paired with friction_fpath. Must be same shape as
            exclusions.
        excl_area : float
            Area of an exclusion cell (square km).
        cap_cost_scale : str | None
            Optional LCOE scaling equation to implement "economies of scale".
            Equations must be in python string format and return a scalar
            value to multiply the capital cost by. Independent variables in
            the equation should match the names of the columns in the reV
            supply curve aggregation table. This will not affect offshore
            wind LCOE.

        Returns
        -------
        summary : list
            List of dictionaries, each being an SC point summary.
        """

        summary = []

        with SupplyCurveExtent(excl_fpath, resolution=resolution) as sc:
            points = sc.points
            exclusion_shape = sc.exclusions.shape
            if gids is None:
                gids = sc.valid_sc_points(tm_dset)

        # pre-extract handlers so they are not repeatedly initialized
        file_kwargs = {'econ_fpath': econ_fpath,
                       'data_layers': data_layers,
                       'power_density': power_density,
                       'excl_dict': excl_dict,
                       'area_filter_kernel': area_filter_kernel,
                       'min_area': min_area,
                       'friction_fpath': friction_fpath,
                       'friction_dset': friction_dset,
                       'check_excl_layers': check_excl_layers}
        with SupplyCurveAggFileHandler(excl_fpath, gen_fpath,
                                       **file_kwargs) as fh:
            inputs = cls._get_input_data(fh.gen, gen_fpath, econ_fpath,
                                         res_class_dset, res_class_bins,
                                         cf_dset, lcoe_dset, h5_dsets)

            n_finished = 0
            for gid in gids:
                for ri, res_bin in enumerate(inputs[1]):
                    try:
                        pointsum = SupplyCurvePointSummary.summarize(
                            gid,
                            fh.exclusions,
                            fh.gen,
                            tm_dset,
                            gen_index,
                            res_class_dset=inputs[0],
                            res_class_bin=res_bin,
                            cf_dset=inputs[2],
                            lcoe_dset=inputs[3],
                            h5_dsets=inputs[5],
                            data_layers=fh.data_layers,
                            resolution=resolution,
                            exclusion_shape=exclusion_shape,
                            power_density=fh.power_density,
                            args=args,
                            excl_dict=excl_dict,
                            excl_area=excl_area,
                            close=False,
                            offshore_flags=inputs[4],
                            friction_layer=fh.friction_layer,
                            cap_cost_scale=cap_cost_scale)

                    except EmptySupplyCurvePointError:
                        pass

                    except Exception:
                        logger.exception('SC gid {} failed!'.format(gid))
                        raise

                    else:
                        pointsum['sc_point_gid'] = gid
                        pointsum['sc_row_ind'] = points.loc[gid, 'row_ind']
                        pointsum['sc_col_ind'] = points.loc[gid, 'col_ind']
                        pointsum['res_class'] = ri

                        summary.append(pointsum)
                        n_finished += 1
                        logger.debug('Serial aggregation: '
                                     '{} out of {} points complete'
                                     .format(n_finished, len(gids)))

        return summary

    def run_parallel(self, args=None, excl_area=0.0081, max_workers=None,
                     points_per_worker=10):
        """Get the supply curve points aggregation summary using futures.

        Parameters
        ----------
        args : tuple | list | None
            List of summary arguments to include. None defaults to all
            available args defined in the class attr.
        excl_area : float, optional
            Area of an exclusion cell (square km), by default 0.0081
        max_workers : int | None, optional
            Number of cores to run summary on. None is all
            available cpus, by default None
        points_per_worker : int
            Number of sc_points to summarize on each worker, by default 10

        Returns
        -------
        summary : list
            List of dictionaries, each being an SC point summary.
        """
        chunks = len(self._gids) // points_per_worker
        chunks = np.array_split(self._gids, chunks)

        logger.info('Running supply curve point aggregation for '
                    'points {} through {} at a resolution of {} '
                    'on {} cores in {} chunks.'
                    .format(self._gids[0], self._gids[-1], self._resolution,
                            max_workers, len(chunks)))

        n_finished = 0
        futures = []
        summary = []
        loggers = [__name__, 'reV.supply_curve.point_summary', 'reV']
        with SpawnProcessPool(max_workers=max_workers, loggers=loggers) as exe:

            # iterate through split executions, submitting each to worker
            for gid_set in chunks:
                # submit executions and append to futures list
                futures.append(exe.submit(
                    self.run_serial,
                    self._excl_fpath, self._gen_fpath,
                    self._tm_dset, self._gen_index,
                    econ_fpath=self._econ_fpath,
                    excl_dict=self._excl_dict,
                    res_class_dset=self._res_class_dset,
                    res_class_bins=self._res_class_bins,
                    cf_dset=self._cf_dset,
                    lcoe_dset=self._lcoe_dset,
                    h5_dsets=self._h5_dsets,
                    data_layers=self._data_layers,
                    resolution=self._resolution,
                    power_density=self._power_density,
                    friction_fpath=self._friction_fpath,
                    friction_dset=self._friction_dset,
                    area_filter_kernel=self._area_filter_kernel,
                    min_area=self._min_area,
                    gids=gid_set, args=args, excl_area=excl_area,
                    check_excl_layers=self._check_excl_layers,
                    cap_cost_scale=self._cap_cost_scale))

            # gather results
            for future in as_completed(futures):
                n_finished += 1
                logger.info('Parallel aggregation futures collected: '
                            '{} out of {}'
                            .format(n_finished, len(chunks)))
                summary += future.result()

        return summary

    def run_offshore(self, summary, offshore_capacity=600,
                     offshore_gid_counts=494, offshore_pixel_area=4,
                     offshore_meta_cols=None):
        """Get the offshore supply curve point summary. Each offshore resource
        pixel will be summarized in its own supply curve point.

        Parameters
        ----------
        summary : list
            List of dictionaries, each being an onshore SC point summary.
        offshore_capacity : int | float
            Offshore resource pixel generation capacity in MW.
        offshore_gid_counts : int
            Approximate number of exclusion pixels that would fall into an
            offshore pixel area.
        offshore_pixel_area : int | float
            Approximate area of offshore resource pixels in km2.
        offshore_meta_cols : list | tuple | None
            Column labels from original offshore data file that were passed
            through to the offshore module output meta data. None will use
            Offshore class variable DEFAULT_META_COLS, and any
            additional requested cols will be added to DEFAULT_META_COLS.

        Returns
        -------
        summary : list
            List of dictionaries, each being an SC point summary, includng SC
            points for single offshore resource pixels.
        """

        file_kwargs = {'data_layers': self._data_layers,
                       'econ_fpath': self._econ_fpath,
                       'power_density': self._power_density,
                       'excl_dict': self._excl_dict}
        with SupplyCurveAggFileHandler(self._excl_fpath, self._gen_fpath,
                                       **file_kwargs) as fh:
            inp = self._get_input_data(fh.gen, self._gen_fpath,
                                       self._econ_fpath, self._res_class_dset,
                                       self._res_class_bins, self._cf_dset,
                                       self._lcoe_dset, self._h5_dsets)

            res, bins, cf, lcoe, offshore_flag, h5_dsets_data = inp

            if offshore_flag is not None:
                if any(offshore_flag):
                    summary = OffshoreAggregation.run(
                        summary, fh, self._excl_fpath, res, bins,
                        cf, lcoe, offshore_flag,
                        h5_dsets_data=h5_dsets_data,
                        resolution=self._resolution,
                        offshore_capacity=offshore_capacity,
                        offshore_gid_counts=offshore_gid_counts,
                        offshore_pixel_area=offshore_pixel_area,
                        offshore_meta_cols=offshore_meta_cols)

        return summary

    @staticmethod
    def _convert_bins(bins):
        """Convert a list of floats or ints to a list of two-entry bin bounds.

        Parameters
        ----------
        bins : list | None
            List of floats or ints (bin edges) to convert to list of two-entry
            bin boundaries or list of two-entry bind boundaries in final format

        Returns
        -------
        bins : list
            List of two-entry bin boundaries
        """

        if bins is None:
            return None

        type_check = [isinstance(x, (list, tuple)) for x in bins]

        if all(type_check):
            return bins

        elif any(type_check):
            raise TypeError('Resource class bins has inconsistent '
                            'entry type: {}'.format(bins))

        else:
            bbins = []
            for i, b in enumerate(sorted(bins)):
                if i < len(bins) - 1:
                    bbins.append([b, bins[i + 1]])

            return bbins

    @staticmethod
    def _summary_to_df(summary):
        """Convert the agg summary list to a DataFrame.

        Parameters
        ----------
        summary : list
            List of dictionaries, each being an SC point summary.

        Returns
        -------
        summary : DataFrame
            Summary of the SC points.
        """
        summary = pd.DataFrame(summary)
        summary = summary.sort_values('sc_point_gid')
        summary = summary.reset_index(drop=True)
        summary.index.name = 'sc_gid'

        return summary

    def summarize(self, args=None, max_workers=None, points_per_worker=10,
                  offshore_capacity=600, offshore_gid_counts=494,
                  offshore_pixel_area=4, offshore_meta_cols=None):
        """
        Get the supply curve points aggregation summary

        Parameters
        ----------
        args : tuple | list | None
            List of summary arguments to include. None defaults to all
            available args defined in the class attr.
        max_workers : int | None, optional
            Number of cores to run summary on. None is all
            available cpus, by default None
        points_per_worker : int
            Number of sc_points to summarize on each worker, by default 10
        offshore_capacity : int | float
            Offshore resource pixel generation capacity in MW.
        offshore_gid_counts : int
            Approximate number of exclusion pixels that would fall into an
            offshore pixel area.
        offshore_pixel_area : int | float
            Approximate area of offshore resource pixels in km2.
        offshore_meta_cols : list | tuple | None
            Column labels from original offshore data file that were passed
            through to the offshore module output meta data. None will use
            Offshore class variable DEFAULT_META_COLS, and any
            additional requested cols will be added to DEFAULT_META_COLS.

        Returns
        -------
        summary : list
            List of dictionaries, each being an SC point summary.
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        if max_workers == 1:
            afk = self._area_filter_kernel
            chk = self._check_excl_layers
            summary = self.run_serial(self._excl_fpath, self._gen_fpath,
                                      self._tm_dset, self._gen_index,
                                      econ_fpath=self._econ_fpath,
                                      excl_dict=self._excl_dict,
                                      res_class_dset=self._res_class_dset,
                                      res_class_bins=self._res_class_bins,
                                      cf_dset=self._cf_dset,
                                      lcoe_dset=self._lcoe_dset,
                                      h5_dsets=self._h5_dsets,
                                      data_layers=self._data_layers,
                                      resolution=self._resolution,
                                      power_density=self._power_density,
                                      friction_fpath=self._friction_fpath,
                                      friction_dset=self._friction_dset,
                                      area_filter_kernel=afk,
                                      min_area=self._min_area,
                                      gids=self._gids, args=args,
                                      excl_area=self._excl_area,
                                      check_excl_layers=chk,
                                      cap_cost_scale=self._cap_cost_scale)
        else:
            summary = self.run_parallel(args=args, excl_area=self._excl_area,
                                        max_workers=max_workers,
                                        points_per_worker=points_per_worker)

        summary = self.run_offshore(summary,
                                    offshore_capacity=offshore_capacity,
                                    offshore_gid_counts=offshore_gid_counts,
                                    offshore_pixel_area=offshore_pixel_area,
                                    offshore_meta_cols=offshore_meta_cols)

        if not any(summary):
            e = ('Supply curve aggregation found no non-excluded SC points. '
                 'Please check your exclusions or subset SC GID selection.')
            logger.error(e)
            raise EmptySupplyCurvePointError(e)

        summary = self._summary_to_df(summary)
        summary = OffshoreAggregation._agg_data_layers(self._data_layers,
                                                       summary)

        return summary

    @classmethod
    def summary(cls, excl_fpath, gen_fpath, tm_dset, econ_fpath=None,
                excl_dict=None, area_filter_kernel='queen', min_area=None,
                check_excl_layers=False, resolution=64, gids=None,
                res_class_dset=None, res_class_bins=None,
                cf_dset='cf_mean-means', lcoe_dset='lcoe_fcr-means',
                h5_dsets=None, data_layers=None, power_density=None,
                friction_fpath=None, friction_dset=None,
                args=None, excl_area=None, max_workers=None,
                points_per_worker=10,
                cap_cost_scale=None, offshore_capacity=600,
                offshore_gid_counts=494, offshore_pixel_area=4,
                offshore_meta_cols=None):
        """Get the supply curve points aggregation summary.

        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        tm_dset : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        econ_fpath : str | None
            Filepath to .h5 reV econ output results. This is optional and only
            used if the lcoe_dset is not present in the gen_fpath file.
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
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of floats or ints (bin edges) to convert to list of two-entry
            bin boundaries or list of two-entry bind boundaries in final format
        cf_dset : str
            Dataset name from f_gen containing capacity factor mean values.
        lcoe_dset : str
            Dataset name from f_gen containing LCOE mean values.
        h5_dsets : list | None
            Optional list of additional datasets from the source h5 gen/econ
            files to aggregate.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        power_density : float | str | None
            Power density in MW/km2 or filepath to variable power
            density file. None will attempt to infer a constant
            power density from the generation meta data technology.
            Variable power density csvs must have "gid" and "power_density"
            columns where gid is the resource gid (typically wtk or nsrdb gid)
            and the power_density column is in MW/km2.
        friction_fpath : str | None
            Filepath to friction surface data (cost based exclusions).
            Must be paired with friction_dset. The friction data must be the
            same shape as the exclusions. Friction input creates a new output
            "mean_lcoe_friction" which is the nominal LCOE multiplied by the
            friction data.
        friction_dset : str | None
            Dataset name in friction_fpath for the friction surface data.
            Must be paired with friction_fpath. Must be same shape as
            exclusions.
        args : tuple | list | None
            List of summary arguments to include. None defaults to all
            available args defined in the class attr.
        excl_area : float | None
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath.
        max_workers : int | None, optional
            Number of cores to run summary on. None is all
            available cpus, by default None
        points_per_worker : int
            Number of sc_points to summarize on each worker, by default 10
        cap_cost_scale : str | None
            Optional LCOE scaling equation to implement "economies of scale".
            Equations must be in python string format and return a scalar
            value to multiply the capital cost by. Independent variables in
            the equation should match the names of the columns in the reV
            supply curve aggregation table. This will not affect offshore
            wind LCOE.
        offshore_capacity : int | float
            Offshore resource pixel generation capacity in MW.
        offshore_gid_counts : int
            Approximate number of exclusion pixels that would fall into an
            offshore pixel area.
        offshore_pixel_area : int | float
            Approximate area of offshore resource pixels in km2.
        offshore_meta_cols : list | tuple | None
            Column labels from original offshore data file that were passed
            through to the offshore module output meta data. None will use
            Offshore class variable DEFAULT_META_COLS, and any
            additional requested cols will be added to DEFAULT_META_COLS.

        Returns
        -------
        summary : DataFrame
            Summary of the SC points.
        """

        agg = cls(excl_fpath, gen_fpath, tm_dset,
                  econ_fpath=econ_fpath,
                  excl_dict=excl_dict,
                  res_class_dset=res_class_dset,
                  res_class_bins=res_class_bins,
                  cf_dset=cf_dset,
                  lcoe_dset=lcoe_dset,
                  h5_dsets=h5_dsets,
                  data_layers=data_layers,
                  resolution=resolution,
                  power_density=power_density,
                  gids=gids,
                  friction_fpath=friction_fpath,
                  friction_dset=friction_dset,
                  area_filter_kernel=area_filter_kernel,
                  min_area=min_area,
                  check_excl_layers=check_excl_layers,
                  excl_area=excl_area,
                  cap_cost_scale=cap_cost_scale)

        summary = agg.summarize(args=args,
                                max_workers=max_workers,
                                points_per_worker=points_per_worker,
                                offshore_capacity=offshore_capacity,
                                offshore_gid_counts=offshore_gid_counts,
                                offshore_pixel_area=offshore_pixel_area,
                                offshore_meta_cols=offshore_meta_cols)

        return summary
