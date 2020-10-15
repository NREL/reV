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
                tree = cKDTree(onshore_summary[['latitude', 'longitude']])
                _, nn = tree.query(offshore_summary[['latitude', 'longitude']])

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
        cols = ['latitude', 'longitude']
        lat_lon = handler.gen.meta.loc[gen_gids, cols].values
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
                latitude = handler.gen.meta.loc[gen_gid, 'latitude']
                longitude = handler.gen.meta.loc[gen_gid, 'longitude']
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
    """Supply points aggregation framework."""

    def __init__(self, excl_fpath, gen_fpath, tm_dset, econ_fpath=None,
                 excl_dict=None, area_filter_kernel='queen', min_area=None,
                 check_excl_layers=False, resolution=64, excl_area=None,
                 gids=None, res_class_dset=None, res_class_bins=None,
                 cf_dset='cf_mean-means', lcoe_dset='lcoe_fcr-means',
                 h5_dsets=None, data_layers=None, power_density=None,
                 friction_fpath=None, friction_dset=None):
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
        self._power_density = power_density
        self._friction_fpath = friction_fpath
        self._friction_dset = friction_dset
        self._data_layers = data_layers

        logger.debug('Resource class bins: {}'.format(self._res_class_bins))

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

    @staticmethod
    def run_serial(excl_fpath, gen_fpath, tm_dset, gen_index, econ_fpath=None,
                   excl_dict=None, area_filter_kernel='queen', min_area=None,
                   check_excl_layers=False, resolution=64, gids=None,
                   args=None, res_class_dset=None, res_class_bins=None,
                   cf_dset='cf_mean-means', lcoe_dset='lcoe_fcr-means',
                   h5_dsets=None, data_layers=None, power_density=None,
                   friction_fpath=None, friction_dset=None, excl_area=0.0081):
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
            inputs = SupplyCurveAggregation._get_input_data(fh.gen,
                                                            gen_fpath,
                                                            econ_fpath,
                                                            res_class_dset,
                                                            res_class_bins,
                                                            cf_dset,
                                                            lcoe_dset,
                                                            h5_dsets)

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
                            friction_layer=fh.friction_layer)

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

    def run_parallel(self, args=None, excl_area=0.0081, max_workers=None):
        """Get the supply curve points aggregation summary using futures.

        Parameters
        ----------
        args : tuple | list | None
            List of summary arguments to include. None defaults to all
            available args defined in the class attr.
        excl_area : float
            Area of an exclusion cell (square km).
        max_workers : int | None
            Number of cores to run summary on. None is all
            available cpus.

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
                    check_excl_layers=self._check_excl_layers))

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
            inp = SupplyCurveAggregation._get_input_data(fh.gen,
                                                         self._gen_fpath,
                                                         self._econ_fpath,
                                                         self._res_class_dset,
                                                         self._res_class_bins,
                                                         self._cf_dset,
                                                         self._lcoe_dset,
                                                         self._h5_dsets)

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

    def summarize(self, args=None, max_workers=None,
                  offshore_capacity=600, offshore_gid_counts=494,
                  offshore_pixel_area=4, offshore_meta_cols=None):
        """
        Get the supply curve points aggregation summary

        Parameters
        ----------
        args : tuple | list | None
            List of summary arguments to include. None defaults to all
            available args defined in the class attr.
        max_workers : int | None
            Number of cores to run summary on. None is all
            available cpus.
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
                                      check_excl_layers=chk)
        else:
            summary = self.run_parallel(args=args, excl_area=self._excl_area,
                                        max_workers=max_workers)

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
                offshore_capacity=600, offshore_gid_counts=494,
                offshore_pixel_area=4, offshore_meta_cols=None):
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
        max_workers : int | None
            Number of cores to run summary on. None is all
            available cpus.
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
                  excl_area=excl_area)

        summary = agg.summarize(args=args,
                                max_workers=max_workers,
                                offshore_capacity=offshore_capacity,
                                offshore_gid_counts=offshore_gid_counts,
                                offshore_pixel_area=offshore_pixel_area,
                                offshore_meta_cols=offshore_meta_cols)

        return summary
