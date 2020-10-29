# -*- coding: utf-8 -*-
"""
reV generation module.
"""
from concurrent.futures import TimeoutError
import logging
import numpy as np
import os
import pprint
import psutil
import sys
from warnings import warn

from reV.config.project_points import ProjectPoints, PointsControl
from reV.handlers.outputs import Outputs
from reV.SAM.generation import (Pvwattsv5, Pvwattsv7, TcsMoltenSalt, WindPower,
                                SolarWaterHeat, TroughPhysicalHeat,
                                LinearDirectSteam)
from reV.SAM.version_checker import PySamVersionChecker
from reV.utilities.exceptions import (OutputWarning, ExecutionError,
                                      ParallelExecutionWarning,
                                      ProjectPointsValueError)

from rex.resource import Resource
from rex.multi_file_resource import MultiFileResource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.utilities import check_res_file

logger = logging.getLogger(__name__)


class Gen:
    """Base class for reV generation."""

    # Mapping of reV technology strings to SAM generation objects
    OPTIONS = {'pvwattsv5': Pvwattsv5,
               'pvwattsv7': Pvwattsv7,
               'tcsmoltensalt': TcsMoltenSalt,
               'solarwaterheat': SolarWaterHeat,
               'troughphysicalheat': TroughPhysicalHeat,
               'lineardirectsteam': LinearDirectSteam,
               'windpower': WindPower,
               }

    # Mapping of reV generation outputs to scale factors and units.
    # Type is scalar or array and corresponds to the SAM single-site output
    OUT_ATTRS = {'other': {'scale_factor': 1, 'units': 'unknown',
                           'dtype': 'float32', 'chunks': None},
                 'cf_mean': {'scale_factor': 1000, 'units': 'unitless',
                             'dtype': 'uint16', 'chunks': None,
                             'type': 'scalar'},
                 'cf_profile': {'scale_factor': 1000, 'units': 'unitless',
                                'dtype': 'uint16', 'chunks': (None, 100),
                                'type': 'array'},
                 'dni': {'scale_factor': 1, 'units': 'W/m2',
                         'dtype': 'uint16', 'chunks': (None, 100),
                         'type': 'array'},
                 'dhi': {'scale_factor': 1, 'units': 'W/m2',
                         'dtype': 'uint16', 'chunks': (None, 100),
                         'type': 'array'},
                 'ghi': {'scale_factor': 1, 'units': 'W/m2',
                         'dtype': 'uint16', 'chunks': (None, 100),
                         'type': 'array'},
                 'dni_mean': {'scale_factor': 1000, 'units': 'kWh/m2/day',
                              'dtype': 'uint16', 'chunks': None,
                              'type': 'scalar'},
                 'dhi_mean': {'scale_factor': 1000, 'units': 'kWh/m2/day',
                              'dtype': 'uint16', 'chunks': None,
                              'type': 'scalar'},
                 'ghi_mean': {'scale_factor': 1000, 'units': 'kWh/m2/day',
                              'dtype': 'uint16', 'chunks': None,
                              'type': 'scalar'},
                 'air_temperature': {'scale_factor': 10, 'units': 'Celsius',
                                     'dtype': 'int16', 'chunks': (None, 100),
                                     'type': 'array'},
                 'surface_albedo': {'scale_factor': 100, 'units': 'unitless',
                                    'dtype': 'uint8', 'chunks': (None, 100),
                                    'type': 'array'},
                 'wind_speed': {'scale_factor': 100, 'units': 'm/s',
                                'dtype': 'uint16', 'chunks': (None, 100),
                                'type': 'array'},
                 'windspeed': {'scale_factor': 100, 'units': 'm/s',
                               'dtype': 'uint16', 'chunks': (None, 100),
                               'type': 'array'},
                 'temperature': {'scale_factor': 100, 'units': 'Celsius',
                                 'dtype': 'int16', 'chunks': (None, 100),
                                 'type': 'array'},
                 'pressure': {'scale_factor': 10, 'units': 'atm',
                              'dtype': 'uint16', 'chunks': (None, 100),
                              'type': 'array'},
                 'ws_mean': {'scale_factor': 1000, 'units': 'm/s',
                             'dtype': 'uint16', 'chunks': None,
                             'type': 'scalar'},
                 'annual_energy': {'scale_factor': 1, 'units': 'kWh',
                                   'dtype': 'float32', 'chunks': None,
                                   'type': 'scalar'},
                 'energy_yield': {'scale_factor': 1, 'units': 'kWh/kW',
                                  'dtype': 'float32', 'chunks': None,
                                  'type': 'scalar'},
                 'gen_profile': {'scale_factor': 1, 'units': 'kW',
                                 'dtype': 'float32', 'chunks': (None, 100),
                                 'type': 'array'},
                 'ac': {'scale_factor': 1, 'units': 'kW',
                        'dtype': 'float32', 'chunks': (None, 100),
                        'type': 'array'},
                 'dc': {'scale_factor': 1, 'units': 'kW',
                        'dtype': 'float32', 'chunks': (None, 100),
                        'type': 'array'},
                 'poa': {'scale_factor': 1, 'units': 'W/m2',
                         'dtype': 'float32', 'chunks': (None, 100),
                         'type': 'array'},
                 'ppa_price': {'scale_factor': 1, 'units': 'dol/MWh',
                               'dtype': 'float32', 'chunks': None,
                               'type': 'scalar'},
                 'lcoe_real': {'scale_factor': 1, 'units': 'dol/MWh',
                               'dtype': 'float32', 'chunks': None,
                               'type': 'scalar'},
                 'lcoe_nom': {'scale_factor': 1, 'units': 'dol/MWh',
                              'dtype': 'float32', 'chunks': None,
                              'type': 'scalar'},
                 'lcoe_fcr': {'scale_factor': 1, 'units': 'dol/MWh',
                              'dtype': 'float32', 'chunks': None,
                              'type': 'scalar'},
                 # Solar water heater
                 'T_amb': {'scale_factor': 1, 'units': 'C',
                           'dtype': 'int16', 'chunks': None,
                           'type': 'array'},
                 'T_cold': {'scale_factor': 1, 'units': 'C',
                            'dtype': 'float32', 'chunks': None,
                            'type': 'array'},
                 'T_deliv': {'scale_factor': 1, 'units': 'C',
                             'dtype': 'float32', 'chunks': None,
                             'type': 'array'},
                 'T_hot': {'scale_factor': 1, 'units': 'C',
                           'dtype': 'float32', 'chunks': None,
                           'type': 'array'},
                 'T_tank': {'scale_factor': 1, 'units': 'C',
                            'dtype': 'float32', 'chunks': None,
                            'type': 'array'},
                 'draw': {'scale_factor': 1, 'units': 'kg/hr',
                          'dtype': 'float32', 'chunks': None,
                          'type': 'array'},
                 'beam': {'scale_factor': 1, 'units': 'W/m2',
                          'dtype': 'uint16', 'chunks': None,
                          'type': 'array'},
                 'diffuse': {'scale_factor': 1, 'units': 'W/m2',
                             'dtype': 'uint16', 'chunks': None,
                             'type': 'array'},
                 'I_incident': {'scale_factor': 1, 'units': 'W/m2',
                                'dtype': 'float32', 'chunks': None,
                                'type': 'array'},
                 'I_transmitted': {'scale_factor': 1, 'units': 'W/m2',
                                   'dtype': 'float32', 'chunks': None,
                                   'type': 'array'},
                 'Q_deliv': {'scale_factor': 1, 'units': 'kW',
                             'dtype': 'float32', 'chunks': None,
                             'type': 'array'},
                 'annual_Q_deliv': {'scale_factor': 1, 'units': 'kWh',
                                    'dtype': 'float32', 'chunks': None,
                                    'type': 'scalar'},
                 'solar_fraction': {'scale_factor': 1, 'units': 'None',
                                    'dtype': 'float32', 'chunks': None,
                                    'type': 'scalar'},
                 # Linear Fresnel
                 'q_dot_to_heat_sink': {'scale_factor': 1, 'units': 'MWt',
                                        'dtype': 'float32', 'chunks': None,
                                        'type': 'array'},
                 'gen': {'scale_factor': 1, 'units': 'kW',
                         'dtype': 'float32', 'chunks': None,
                         'type': 'array'},
                 'm_dot_field': {'scale_factor': 1, 'units': 'kg/s',
                                 'dtype': 'float32', 'chunks': None,
                                 'type': 'array'},
                 'q_dot_sf_out': {'scale_factor': 1, 'units': 'MWt',
                                  'dtype': 'float32', 'chunks': None,
                                  'type': 'array'},
                 'W_dot_heat_sink_pump': {'scale_factor': 1, 'units': 'MWe',
                                          'dtype': 'float32', 'chunks': None,
                                          'type': 'array'},
                 'm_dot_loop': {'scale_factor': 1, 'units': 'kg/s',
                                'dtype': 'float32', 'chunks': None,
                                'type': 'array'},
                 'q_dot_rec_inc': {'scale_factor': 1, 'units': 'MWt',
                                   'dtype': 'float32', 'chunks': None,
                                   'type': 'array'},
                 'annual_field_energy': {'scale_factor': 1, 'units': 'kWh',
                                         'dtype': 'float32', 'chunks': None,
                                         'type': 'scalar'},
                 'annual_thermal_consumption': {'scale_factor': 1,
                                                'units': 'kWh',
                                                'dtype': 'float32',
                                                'chunks': None,
                                                'type': 'scalar'},

                 # Trough physical process heat
                 'T_field_cold_in': {'scale_factor': 1, 'units': 'C',
                                     'dtype': 'float32', 'chunks': None,
                                     'type': 'array'},
                 'T_field_hot_out': {'scale_factor': 1, 'units': 'C',
                                     'dtype': 'float32', 'chunks': None,
                                     'type': 'array'},
                 'm_dot_field_delivered': {'scale_factor': 1, 'units': 'kg/s',
                                           'dtype': 'float32', 'chunks': None,
                                           'type': 'array'},
                 'm_dot_field_recirc': {'scale_factor': 1, 'units': 'kg/s',
                                        'dtype': 'float32', 'chunks': None,
                                        'type': 'array'},
                 'q_dot_htf_sf_out': {'scale_factor': 1, 'units': 'MWt',
                                      'dtype': 'float32', 'chunks': None,
                                      'type': 'array'},
                 'qinc_costh': {'scale_factor': 1, 'units': 'MWt',
                                'dtype': 'float32', 'chunks': None,
                                'type': 'array'},
                 'dni_costh': {'scale_factor': 1, 'units': 'W/m2',
                               'dtype': 'float32', 'chunks': None,
                               'type': 'array'},
                 'annual_gross_energy': {'scale_factor': 1, 'units': 'kWh',
                                         'dtype': 'float32', 'chunks': None,
                                         'type': 'scalar'},
                 }

    def __init__(self, points_control, res_file, output_request=('cf_mean',),
                 fout=None, dirout='./gen_out', drop_leap=False,
                 mem_util_lim=0.4):
        """
        Parameters
        ----------
        points_control : reV.config.project_points.PointsControl
            Project points control instance for site and SAM config spec.
        res_file : str
            Filepath to single resource file, multi-h5 directory,
            or /h5_dir/prefix*suffix
        output_request : list | tuple
            Output variables requested from SAM.
        fout : str | None
            Optional .h5 output file specification.
        dirout : str | None
            Optional output directory specification. The directory will be
            created if it does not already exist.
        drop_leap : bool
            Drop leap day instead of final day of year during leap years
        mem_util_lim : float
            Memory utilization limit (fractional). This sets how many site
            results will be stored in-memory at any given time before flushing
            to disk.
        """

        self._points_control = points_control
        self._res_file = res_file
        self._site_limit = None
        self._site_mem = None
        self._fout = fout
        self._dirout = dirout
        self._fpath = None
        self._meta = None
        self._time_index = None
        self._year = None
        self._sam_obj_default = None
        self._sam_module = self.OPTIONS[self.tech]
        self._drop_leap = drop_leap
        self.mem_util_lim = mem_util_lim

        self._run_attrs = {'points_control': str(points_control),
                           'res_file': res_file,
                           'output_request': output_request,
                           'fout': str(fout),
                           'dirout': str(dirout),
                           'drop_leap': str(drop_leap),
                           'mem_util_lim': mem_util_lim,
                           'sam_module': self._sam_module.MODULE}

        self._output_request = self._parse_output_request(output_request)
        self._multi_h5_res, self._hsds = check_res_file(res_file)

        if self.tech not in self.OPTIONS:
            msg = ('Requested technology "{}" is not available. '
                   'reV generation can analyze the following '
                   'SAM technologies: {}'
                   .format(self.tech, list(self.OPTIONS.keys())))
            logger.error(msg)
            raise KeyError(msg)

        # pre-initialize output arrays to store results when available.
        self._out = {}
        self._finished_sites = []
        self._out_n_sites = 0
        self._out_chunk = ()
        self._check_sam_version_inputs()

        # initialize output file
        self._init_fpath()
        self._init_h5()
        self._init_out_arrays()

    @property
    def output_request(self):
        """Get the output variables requested from generation.

        Returns
        -------
        output_request : list
            Output variables requested from SAM.
        """
        return self._output_request

    @property
    def out_chunk(self):
        """Get the current output chunk index range (INCLUSIVE).

        Returns
        -------
        _out_chunk : tuple
            Two entry tuple (start, end) indicies (inclusive) for where the
            current data in-memory belongs in the final output.
        """
        return self._out_chunk

    @property
    def site_limit(self):
        """Get the number of sites results that can be stored in memory at once

        Returns
        -------
        _site_limit : int
            Number of site result sets that can be stored in memory at once
            without violating memory limits.
        """

        if self._site_limit is None:
            tot_mem = psutil.virtual_memory().total / 1e6
            avail_mem = self.mem_util_lim * tot_mem
            self._site_limit = int(np.floor(avail_mem / self.site_mem))
            logger.info('Generation limited to storing {0} sites in memory '
                        '({1:.1f} GB total hardware, {2:.1f} GB available '
                        'with {3:.1f}% utilization).'
                        .format(self._site_limit, tot_mem / 1e3,
                                avail_mem / 1e3, self.mem_util_lim * 100))

        return self._site_limit

    @property
    def site_mem(self):
        """Get the memory (MB) required to store all results for a single site.

        Returns
        -------
        _site_mem : float
            Memory (MB) required to store all results in requested in
            output_request for a single site.
        """

        if self._site_mem is None:
            # average the memory usage over n sites
            # (for better understanding of array overhead)
            n = 100
            self._site_mem = 0
            for request in self.output_request:
                dtype = 'float32'
                if request in self.OUT_ATTRS:
                    dtype = self.OUT_ATTRS[request].get('dtype', 'float32')

                shape = self._get_data_shape(request, n)
                self._site_mem += sys.getsizeof(np.ones(shape, dtype=dtype))

            self._site_mem = self._site_mem / 1e6 / n
            logger.info('Output results from a single site are calculated to '
                        'use {0:.1f} KB of memory.'
                        .format(self._site_mem / 1000))

        return self._site_mem

    @property
    def points_control(self):
        """Get project points controller.

        Returns
        -------
        points_control : reV.config.project_points.PointsControl
            Project points control instance for site and SAM config spec.
        """
        return self._points_control

    @property
    def project_points(self):
        """Get project points

        Returns
        -------
        project_points : reV.config.project_points.ProjectPoints
            Project points from the points control instance.
        """
        return self._points_control.project_points

    @property
    def sam_configs(self):
        """Get the sam config dictionary.

        Returns
        -------
        sam_configs : reV.config.sam.SAMGenConfig
            SAM config from the project points instance.
        """
        return self.project_points.sam_configs

    @property
    def sam_metas(self):
        """
        SAM configurations including runtime module

        Returns
        -------
        sam_metas : dict
            Nested dictionary of SAM configuration files with module used
            at runtime
        """
        sam_metas = self.sam_configs.copy()
        for v in sam_metas.values():
            v.update({'module': self._sam_module.MODULE})

        return sam_metas

    @property
    def tech(self):
        """Get the reV technology string.

        Returns
        -------
        tech : str
            SAM technology to analyze (pvwattsv7, windpower, tcsmoltensalt,
            solarwaterheat, troughphysicalheat, lineardirectsteam)
            The string should be lower-cased with spaces and _ removed.
        """
        return self.project_points.tech

    @property
    def res_file(self):
        """Get the resource filename and path.

        Returns
        -------
        res_file : str
            Filepath to single resource file, multi-h5 directory,
            or /h5_dir/prefix*suffix
        """
        return self._res_file

    @property
    def fout(self):
        """Get the target file output.

        Returns
        -------
        fout : str | None
            Optional .h5 output file specification.
        """
        return self._fout

    @property
    def dirout(self):
        """Get the target output directory.

        Returns
        -------
        dirout : str | None
            Optional output directory specification. The directory will be
            created if it does not already exist.
        """
        return self._dirout

    @property
    def meta(self):
        """Get resource meta for all sites in project points.

        Returns
        -------
        meta : pd.DataFrame
            Meta data df for sites in project points. Column names are meta
            data variables, rows are different sites. The row index
            does not indicate the site number if the project points are
            non-sequential or do not start from 0, so a 'gid' column is added.
        """
        if self._meta is None:
            if not self._multi_h5_res:
                res_cls = Resource
                kwargs = {'hsds': self._hsds}
            else:
                res_cls = MultiFileResource
                kwargs = {}

            with res_cls(self.res_file, **kwargs) as res:
                res_meta = res.meta

            if np.max(self.project_points.sites) > len(res_meta):
                msg = ('ProjectPoints has a max site gid of {} which is '
                       'out of bounds for the meta data of size {} from '
                       'resource file: {}'
                       .format(np.max(self.project_points.sites),
                               res_meta.shape, self.res_file))
                logger.error(msg)
                raise ProjectPointsValueError(msg)

            self._meta = res_meta.iloc[self.project_points.sites, :]
            self._meta.loc[:, 'gid'] = self.project_points.sites
            self._meta.loc[:, 'reV_tech'] = self.project_points.tech

        return self._meta

    @property
    def run_attrs(self):
        """
        Run time attributes (__init__ args and kwargs)

        Returns
        -------
        run_attrs : dict
            Dictionary of runtime args and kwargs
        """
        return self._run_attrs

    @property
    def time_index(self):
        """Get the generation resource time index data.

        Returns
        -------
        _time_index : pandas.DatetimeIndex
            Time-series datetime index
        """
        if self._time_index is None:
            if not self._multi_h5_res:
                res_cls = Resource
                kwargs = {'hsds': self._hsds}
            else:
                res_cls = MultiFileResource
                kwargs = {}

            with res_cls(self.res_file, **kwargs) as res:
                time_index = res.time_index

            downscale = self.project_points.sam_config_obj.downscale
            step = self.project_points.sam_config_obj.time_index_step
            if downscale is not None:
                from rex.utilities.downscale import make_time_index
                year = time_index.year[0]
                time_index = make_time_index(year, downscale)
                logger.info('reV solar generation running with temporal '
                            'downscaling frequency "{}" with final '
                            'time_index length {}'
                            .format(downscale, len(time_index)))
            elif step is not None:
                time_index = time_index[::2]

            self._time_index = self.handle_leap_ti(time_index,
                                                   drop_leap=self._drop_leap)

        return self._time_index

    @property
    def year(self):
        """Get the generation resource year.

        Returns
        -------
        _year : int
            Year of the time-series datetime index.
        """

        if self._year is None:
            self._year = int(self.time_index.year[0])

        return self._year

    @property
    def out(self):
        """Get the generation output results.

        Returns
        -------
        out : dict
            Dictionary of generation results from SAM.
        """
        out = {}
        for k, v in self._out.items():
            if k in Gen.OUT_ATTRS:
                scale_factor = Gen.OUT_ATTRS[k].get('scale_factor', 1)
            else:
                scale_factor = 1

            if scale_factor != 1:
                v = v.astype('float32')
                v /= scale_factor

            out[k] = v

        return out

    @out.setter
    def out(self, result):
        """Set the output attribute, unpack futures, clear output from mem.

        Parameters
        ----------
        result : list | dict | None
            Generation results to set to output dictionary. Use cases:
             - List input is interpreted as a futures list, which is unpacked
               before setting to the output dict.
             - Dictionary input is interpreted as an already unpacked result.
             - None is interpreted as a signal to clear the output dictionary.
        """
        if isinstance(result, list):
            # unpack futures list to dictionary first
            result = self.unpack_futures(result)

        if isinstance(result, dict):

            # iterate through dict where sites are keys and values are
            # corresponding results
            for site_gid, site_output in result.items():

                # check that the sites are stored sequentially then add to
                # the finished site list
                if self._finished_sites:
                    if int(site_gid) < np.max(self._finished_sites):
                        raise Exception('Site results are non sequential!')

                # unpack site output object
                self.unpack_output(site_gid, site_output)

                # add site gid to the finished list after outputs are unpacked
                self._finished_sites.append(site_gid)

        elif isinstance(result, type(None)):
            self._out.clear()
            self._finished_sites.clear()
        else:
            raise TypeError('Did not recognize the type of generation output. '
                            'Tried to set output type "{}", but requires '
                            'list, dict or None.'.format(type(result)))

    @staticmethod
    def _output_request_type_check(req):
        """Output request type check and ensure list for manipulation.

        Parameters
        ----------
        req : list | tuple | str
            Output request of variable type.

        Returns
        -------
        output_request : list
            Output request.
        """

        if isinstance(req, list):
            output_request = req
        elif isinstance(req, tuple):
            output_request = list(req)
        elif isinstance(req, str):
            output_request = [req]
        else:
            raise TypeError('Output request must be str, list, or tuple but '
                            'received: {}'.format(type(req)))

        return output_request

    @staticmethod
    def _add_out_reqs(output_request):
        """Add additional output requests as needed.

        Parameters
        ----------
        output_request : list
            Output variables requested from SAM.

        Returns
        -------
        output_request : list
            Output variable list with cf_mean and resource mean out vars.
        """

        if 'cf_mean' not in output_request:
            # ensure that cf_mean is requested from output
            output_request.append('cf_mean')

        return output_request

    @staticmethod
    def handle_leap_ti(ti, drop_leap=False):
        """Handle a time index for a leap year by dropping a day.

        Parameters
        ----------
        ti : pandas.DatetimeIndex
            Time-series datetime index with or without a leap day.
        drop_leap : bool
            Option to drop leap day (if True) or drop the last day of the year
            (if False).

        Returns
        -------
        ti : pandas.DatetimeIndex
            Time-series datetime index with length a multiple of 365.
        """

        # drop leap day or last day
        leap_day = ((ti.month == 2) & (ti.day == 29))
        last_day = ((ti.month == 12) & (ti.day == 31))
        if drop_leap:
            # preference is to drop leap day if exists
            ti = ti.drop(ti[leap_day])
        elif any(leap_day):
            # leap day exists but preference is to drop last day of year
            ti = ti.drop(ti[last_day])

        if len(ti) % 365 != 0:
            raise ValueError('Bad time index with length not a multiple of '
                             '365: {}'.format(ti))

        return ti

    @staticmethod
    def _pp_to_pc(points, points_range, sam_files, tech, sites_per_worker=None,
                  res_file=None, curtailment=None):
        """
        Create ProjectControl from ProjectPoints

        Parameters
        ----------
        points : slice | list | str | reV.config.project_points.PointsControl
            Slice specifying project points, or string pointing to a project
            points csv, or a fully instantiated PointsControl object.
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. To be taken from the reV.config.PointsControl.split_range
            property.
        sam_files : dict | str | list | SAMConfig
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s), top level value is the SAM path. Can also be a single
            config file str. If it's a list, it is mapped to the sorted list
            of unique configs requested by points csv. Can also be a
            pre loaded SAMConfig object.
        tech : str
            SAM technology to analyze (pvwattsv7, windpower, tcsmoltensalt,
            solarwaterheat, troughphysicalheat, lineardirectsteam)
            The string should be lower-cased with spaces and _ removed.
        sites_per_worker : int
            Number of sites to run in series on a worker. None defaults to the
            resource file chunk size.
        res_file : str
            Filepath to single resource file, multi-h5 directory,
            or /h5_dir/prefix*suffix
        curtailment : NoneType | dict | str | config.curtailment.Curtailment
            Inputs for curtailment parameters. If not None, curtailment inputs
            are expected. Can be:
                - Explicit namespace of curtailment variables (dict)
                - Pointer to curtailment config json file with path (str)
                - Instance of curtailment config object
                  (config.curtailment.Curtailment)

        Returns
        -------
        pc : reV.config.project_points.PointsControl
            PointsControl object instance.
        """
        if not isinstance(points, ProjectPoints):
            # make Project Points instance
            pp = ProjectPoints(points, sam_files, tech=tech,
                               res_file=res_file, curtailment=curtailment)
        else:
            pp = ProjectPoints(points.df, sam_files, tech=tech,
                               res_file=res_file, curtailment=curtailment)

        #  make Points Control instance
        if points_range is not None:
            # PointsControl is for just a subset of the project points...
            # this is the case if generation is being initialized on one
            # of many HPC nodes in a large project
            pc = PointsControl.split(points_range[0], points_range[1], pp,
                                     sites_per_split=sites_per_worker)
        else:
            # PointsControl is for all of the project points
            pc = PointsControl(pp, sites_per_split=sites_per_worker)

        return pc

    @staticmethod
    def get_pc(points, points_range, sam_files, tech, sites_per_worker=None,
               res_file=None, curtailment=None):
        """Get a PointsControl instance.

        Parameters
        ----------
        points : slice | list | str | reV.config.project_points.PointsControl
            Slice specifying project points, or string pointing to a project
            points csv, or a fully instantiated PointsControl object.
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. To be taken from the reV.config.PointsControl.split_range
            property.
        sam_files : dict | str | list | SAMConfig
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s), top level value is the SAM path. Can also be a single
            config file str. If it's a list, it is mapped to the sorted list
            of unique configs requested by points csv. Can also be a
            pre loaded SAMConfig object.
        tech : str
            SAM technology to analyze (pvwattsv7, windpower, tcsmoltensalt,
            solarwaterheat, troughphysicalheat, lineardirectsteam)
            The string should be lower-cased with spaces and _ removed.
        sites_per_worker : int
            Number of sites to run in series on a worker. None defaults to the
            resource file chunk size.
        res_file : str
            Filepath to single resource file, multi-h5 directory,
            or /h5_dir/prefix*suffix
        curtailment : NoneType | dict | str | config.curtailment.Curtailment
            Inputs for curtailment parameters. If not None, curtailment inputs
            are expected. Can be:
                - Explicit namespace of curtailment variables (dict)
                - Pointer to curtailment config json file with path (str)
                - Instance of curtailment config object
                  (config.curtailment.Curtailment)

        Returns
        -------
        pc : reV.config.project_points.PointsControl
            PointsControl object instance.
        """

        if tech not in Gen.OPTIONS and tech.lower() != 'econ':
            msg = ('Did not recognize reV-SAM technology string "{}". '
                   'Technology string options are: {}'
                   .format(tech, list(Gen.OPTIONS.keys())))
            logger.error(msg)
            raise KeyError(msg)

        if sites_per_worker is None:
            # get the optimal sites per split based on res file chunk size
            sites_per_worker = Gen.get_sites_per_worker(res_file)

        logger.debug('Sites per worker being set to {} for Gen/Econ '
                     'PointsControl.'.format(sites_per_worker))

        if isinstance(points, (slice, list, str, ProjectPoints)):
            pc = Gen._pp_to_pc(points, points_range, sam_files, tech,
                               sites_per_worker=sites_per_worker,
                               res_file=res_file, curtailment=curtailment)

        elif isinstance(points, PointsControl):
            # received a pre-intialized instance of pointscontrol
            pc = points
        else:
            raise TypeError('Points input type is unrecognized: '
                            '"{}"'.format(type(points)))

        return pc

    @staticmethod
    def get_sites_per_worker(res_file, default=100):
        """Get the nominal sites per worker (x-chunk size) for a given file.

        This is based on the concept that it is most efficient for one core to
        perform one read on one chunk of resource data, such that chunks will
        not have to be read into memory twice and no sites will be read
        redundantly.

        Parameters
        ----------
        res_file : str
            Filepath to single resource file, multi-h5 directory,
            or /h5_dir/prefix*suffix
        default : int
            Sites to be analyzed on a single core if the chunk size cannot be
            determined from res_file.

        Returns
        -------
        sites_per_worker : int
            Nominal sites to be analyzed per worker. This is set to the x-axis
            chunk size for windspeed and dni datasets for the WTK and NSRDB
            data, respectively.
        """
        if not res_file or not os.path.isfile(res_file):
            return default

        with Resource(res_file) as res:
            if 'wtk' in res_file.lower():
                for dset in res.datasets:
                    if 'speed' in dset:
                        # take nominal WTK chunks from windspeed
                        _, _, chunks = res.get_dset_properties(dset)
                        break
            elif 'nsrdb' in res_file.lower():
                # take nominal NSRDB chunks from dni
                _, _, chunks = res.get_dset_properties('dni')
            else:
                warn('Expected "nsrdb" or "wtk" to be in resource filename: {}'
                     .format(res_file))
                chunks = None

        if chunks is None:
            # if chunks not set, go to default
            sites_per_worker = default
            logger.debug('Sites per worker being set to {} (default) based on '
                         'no set chunk size in {}.'
                         .format(sites_per_worker, res_file))
        else:
            sites_per_worker = chunks[1]
            logger.debug('Sites per worker being set to {} based on chunk '
                         'size of {}.'.format(sites_per_worker, res_file))

        return sites_per_worker

    @staticmethod
    def unpack_futures(futures):
        """Combine list of futures results into their native dict format/type.

        Parameters
        ----------
        futures : list
            List of dictionary futures results.

        Returns
        -------
        out : dict
            Compiled results of the native future results type (dict).
        """

        out = {}
        for x in futures:
            out.update(x)

        return out

    @staticmethod
    def run(points_control, tech=None, res_file=None, output_request=None,
            scale_outputs=True):
        """Run a SAM generation analysis based on the points_control iterator.

        Parameters
        ----------
        points_control : reV.config.PointsControl
            A PointsControl instance dictating what sites and configs are run.
        tech : str
            SAM technology to analyze (pvwattsv7, windpower, tcsmoltensalt,
            solarwaterheat, troughphysicalheat, lineardirectsteam)
            The string should be lower-cased with spaces and _ removed.
        res_file : str
            Filepath to single resource file, multi-h5 directory,
            or /h5_dir/prefix*suffix
        output_request : list | tuple
            Output variables requested from SAM.
        scale_outputs : bool
            Flag to scale outputs in-place immediately upon Gen returning data.

        Returns
        -------
        out : dict
            Output dictionary from the SAM reV_run function. Data is scaled
            within this function to the datatype specified in Gen.OUT_ATTRS.
        """
        # run generation method for specified technology
        try:
            out = Gen.OPTIONS[tech].reV_run(points_control, res_file,
                                            output_request=output_request)
        except Exception as e:
            out = {}
            logger.exception('Worker failed for PC: {}'.format(points_control))
            raise e

        if scale_outputs:
            # dtype convert in-place so no float data is stored unnecessarily
            for site, site_output in out.items():
                for k in site_output.keys():
                    # iterate through variable names in each site's output dict
                    if k in Gen.OUT_ATTRS:
                        # get dtype and scale for output variable name
                        dtype = Gen.OUT_ATTRS[k].get('dtype', 'float32')
                        scale_factor = Gen.OUT_ATTRS[k].get('scale_factor', 1)

                        # apply scale factor and dtype
                        out[site][k] *= scale_factor
                        if np.issubdtype(dtype, np.integer):
                            # round after scaling if integer dtype
                            out[site][k] = np.round(out[site][k])

                        if isinstance(out[site][k], np.ndarray):
                            # simple astype for arrays
                            out[site][k] = out[site][k].astype(dtype)
                        else:
                            # use numpy array conversion for scalar values
                            out[site][k] = np.array([out[site][k]],
                                                    dtype=dtype)[0]

        return out

    def _parse_output_request(self, req):
        """Set the output variables requested from generation.

        Parameters
        ----------
        req : list | tuple
            Output variables requested from SAM.

        Returns
        -------
        output_request : list
            Output variables requested from SAM.
        """

        output_request = self._output_request_type_check(req)
        output_request = self._add_out_reqs(output_request)

        for request in output_request:
            if request not in self.OUT_ATTRS:
                msg = ('User output request "{}" not recognized. '
                       'Will attempt to extract from PySAM.'.format(request))
                logger.debug(msg)

        return list(set(output_request))

    def _get_data_shape(self, dset, n_sites):
        """Get the output array shape based on OUT_ATTRS or PySAM.Outputs.

        Parameters
        ----------
        dset : str
            Variable name to get shape for.
        n_sites : int
            Number of sites for this data shape.

        Returns
        -------
        shape : tuple
            1D or 2D shape tuple for dset.
        """

        if dset in self.OUT_ATTRS:
            if self.OUT_ATTRS[dset]['type'] == 'array':
                data_shape = (len(self.time_index), n_sites)
            else:
                data_shape = (n_sites, )

        elif dset in self.project_points.all_sam_input_keys:
            data_shape = (n_sites, )
            data = list(self.project_points.sam_configs.values())[0][dset]
            if isinstance(data, (list, tuple, np.ndarray, str)):
                msg = ('Cannot pass through non-scalar SAM input key "{}" '
                       'as an output_request!'.format(dset))
                logger.error(msg)
                raise ExecutionError(msg)

        else:
            if self._sam_obj_default is None:
                init_obj = self._sam_module()
                self._sam_obj_default = init_obj.default

            try:
                out_data = getattr(self._sam_obj_default.Outputs, dset)
            except AttributeError as e:
                msg = ('Could not get data shape for dset "{}" '
                       'from object "{}". '
                       'Received the following error: "{}"'
                       .format(dset, self._sam_obj_default, e))
                logger.error(msg)
                raise ExecutionError(msg)
            else:
                if isinstance(out_data, (int, float, str)):
                    data_shape = (n_sites, )
                elif len(out_data) % len(self.time_index) == 0:
                    data_shape = (len(self.time_index), n_sites)
                else:
                    data_shape = (len(out_data), n_sites)

        return data_shape

    def _init_fpath(self):
        """Combine directory and filename, ensure .h5 ext., make out dirs."""

        if self._fout is not None:

            # ensure output file is an h5
            if not self._fout .endswith('.h5'):
                self._fout += '.h5'

            # ensure year is in fout
            if str(self.year) not in self._fout:
                self._fout = self._fout.replace('.h5',
                                                '_{}.h5'.format(self.year))

            # create and use optional output dir
            if self._dirout:
                if not os.path.exists(self._dirout):
                    os.makedirs(self._dirout)

                # Add output dir to fout string
                self._fpath = os.path.join(self._dirout, self._fout)
            else:
                self._fpath = self._fout

    def _init_h5(self, mode='w'):
        """Initialize the single h5 output file with all output requests.

        Parameters
        ----------
        mode : str
            Mode to instantiate h5py.File instance
        """

        if self._fpath is not None:

            if 'w' in mode:
                logger.info('Initializing full output file: "{}" with mode: {}'
                            .format(self._fpath, mode))
            elif 'a' in mode:
                logger.info('Appending data to output file: "{}" with mode: {}'
                            .format(self._fpath, mode))

            attrs = {d: {} for d in self.output_request}
            chunks = {}
            dtypes = {}
            shapes = {}

            # flag to write time index if profiles are being output
            write_ti = False

            for dset in self.output_request:

                tmp = 'other'
                if dset in self.OUT_ATTRS:
                    tmp = dset

                attrs[dset]['units'] = self.OUT_ATTRS[tmp].get('units',
                                                               'unknown')
                attrs[dset]['scale_factor'] = \
                    self.OUT_ATTRS[tmp].get('scale_factor', 1)
                chunks[dset] = self.OUT_ATTRS[tmp].get('chunks', None)
                dtypes[dset] = self.OUT_ATTRS[tmp].get('dtype', 'float32')
                shapes[dset] = self._get_data_shape(dset, len(self.meta))
                if len(shapes[dset]) > 1:
                    write_ti = True

            # only write time index if profiles were found in output request
            if write_ti:
                ti = self.time_index
            else:
                ti = None

            Outputs.init_h5(self._fpath, self.output_request, shapes, attrs,
                            chunks, dtypes, self.meta, time_index=ti,
                            configs=self.sam_metas, run_attrs=self.run_attrs,
                            mode=mode)

    def _init_out_arrays(self, index_0=0):
        """Initialize output arrays based on the number of sites that can be
        stored in memory safely.

        Parameters
        ----------
        index_0 : int
            This is the site list index (not gid) for the first site in the
            output data. If a node cannot process all sites in-memory at once,
            this is used to segment the sites in the current output chunk.
        """

        self._out = {}
        self._finished_sites = []

        # Output chunk is the index range (inclusive) of this set of site outs
        self._out_chunk = (index_0, np.min((index_0 + self.site_limit,
                                            len(self.project_points) - 1)))
        self._out_n_sites = int(self.out_chunk[1] - self.out_chunk[0]) + 1

        logger.info('Initializing in-memory outputs for {} sites with gids '
                    '{} through {} inclusive (site list index {} through {})'
                    .format(self._out_n_sites,
                            self.project_points.sites[self.out_chunk[0]],
                            self.project_points.sites[self.out_chunk[1]],
                            self.out_chunk[0], self.out_chunk[1]))

        for request in self.output_request:
            dtype = 'float32'
            if request in self.OUT_ATTRS:
                dtype = self.OUT_ATTRS[request].get('dtype', 'float32')

            shape = self._get_data_shape(request, self._out_n_sites)
            # initialize the output request as an array of zeros
            self._out[request] = np.zeros(shape, dtype=dtype)

    def _check_sam_version_inputs(self):
        """Check the PySAM version and input keys. Fix where necessary."""
        for key, parameters in self.project_points.sam_configs.items():
            updated = PySamVersionChecker.run(self.tech, parameters)
            sam_obj = self._points_control._project_points._sam_config_obj
            sam_obj._inputs[key] = updated

    def unpack_output(self, site_gid, site_output):
        """Unpack a SAM SiteOutput object to the output attribute.

        Parameters
        ----------
        site_gid : int
            Resource-native site gid (index).
        site_output : dict
            SAM site output object.
        """

        # iterate through the site results
        for var, value in site_output.items():
            if var not in self._out:
                raise KeyError('Tried to collect output variable "{}", but it '
                               'was not yet initialized in the output '
                               'dictionary.')

            # get the index in the output array for the current site
            i = self.site_index(site_gid, out_index=True)

            # check to see if we have exceeded the current output chunk.
            # If so, flush data to disk and reset the output initialization
            if i + 1 > self._out_n_sites:
                self.flush()
                global_site_index = self.site_index(site_gid)
                self._init_out_arrays(index_0=global_site_index)
                i = self.site_index(site_gid, out_index=True)

            if isinstance(value, (list, tuple, np.ndarray)):
                if not isinstance(value, np.ndarray):
                    value = np.array(value)

                self._out[var][:, i] = value.T
            elif value != 0:
                self._out[var][i] = value

    def site_index(self, site_gid, out_index=False):
        """Get the index corresponding to the site gid.

        Parameters
        ----------
        site_gid : int
            Resource-native site index (gid).
        out_index : bool
            Option to get output index (if true) which is the column index in
            the current in-memory output array, or (if false) the global site
            index from the project points site list.

        Returns
        -------
        index : int
            Global site index if out_index=False, otherwise column index in
            the current in-memory output array.
        """

        # get the index for site_gid in the (global) project points site list.
        global_site_index = self.project_points.sites.index(site_gid)

        if not out_index:
            output_index = global_site_index
        else:
            output_index = global_site_index - self.out_chunk[0]
            if output_index < 0:
                raise ValueError('Attempting to set output data for site with '
                                 'gid {} to global site index {}, which was '
                                 'already set based on the current output '
                                 'index chunk of {}'
                                 .format(site_gid, global_site_index,
                                         self.out_chunk))

        return output_index

    def flush(self):
        """Flush generation data in self.out attribute to disk in .h5 format.

        The data to be flushed is accessed from the instance attribute
        "self.out". The disk target is based on the instance attributes
        "self._fpath". Data is not flushed if _fpath is None or if .out is
        empty.
        """

        # handle output file request if file is specified and .out is not empty
        if isinstance(self._fpath, str) and self._out:
            logger.info('Flushing outputs to disk, target file: "{}"'
                        .format(self._fpath))

            # get the slice of indices to write outputs to
            islice = slice(self.out_chunk[0], self.out_chunk[1] + 1)

            # open output file in append mode to add output results to
            with Outputs(self._fpath, mode='a') as f:

                # iterate through all output requests writing each as a dataset
                for dset, arr in self._out.items():
                    if len(arr.shape) == 1:
                        # write array of scalars
                        f[dset, islice] = arr
                    else:
                        # write 2D array of profiles
                        f[dset, :, islice] = arr

            logger.debug('Flushed generation output successfully to disk.')

    def _pre_split_pc(self, pool_size=(os.cpu_count() * 2)):
        """Pre-split project control iterator into sub chunks to further
        split the parallelization.

        Parameters
        ----------
        pool_size : int
            Number of futures to submit to a single process pool for
            parallel futures.

        Returns
        -------
        N : int
            Total number of points control split instances.
        pc_chunks : list
            List of lists of points control split instances.
        """
        N = 0
        pc_chunks = []
        i_chunk = []

        for i, split in enumerate(self.points_control):
            N += 1
            i_chunk.append(split)
            if (i + 1) % pool_size == 0:
                pc_chunks.append(i_chunk)
                i_chunk = []

        if i_chunk:
            pc_chunks.append(i_chunk)

        logger.debug('Pre-splitting points control into {} chunks with the '
                     'following chunk sizes: {}'
                     .format(len(pc_chunks), [len(x) for x in pc_chunks]))
        return N, pc_chunks

    def _parallel_run(self, max_workers=None, pool_size=(os.cpu_count() * 2),
                      timeout=1800, **kwargs):
        """Execute parallel compute.

        Parameters
        ----------
        max_workers : None | int
            Number of workers. None will default to cpu count.
        pool_size : int
            Number of futures to submit to a single process pool for
            parallel futures.
        timeout : int | float
            Number of seconds to wait for parallel run iteration to complete
            before returning zeros.
        kwargs : dict
            Keyword arguments to self.run().
        """

        logger.debug('Running parallel execution with max_workers={}'
                     .format(max_workers))
        i = 0
        N, pc_chunks = self._pre_split_pc(pool_size=pool_size)
        for j, pc_chunk in enumerate(pc_chunks):
            logger.debug('Starting process pool for points control '
                         'iteration {} out of {}'
                         .format(j + 1, len(pc_chunks)))

            failed_futures = False
            chunks = {}
            futures = []
            loggers = [__name__, 'reV.econ.econ', 'reV']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                for pc in pc_chunk:
                    future = exe.submit(self.run, pc, **kwargs)
                    futures.append(future)
                    chunks[future] = pc

                for future in futures:
                    i += 1
                    try:
                        result = future.result(timeout=timeout)
                    except TimeoutError:
                        failed_futures = True
                        sites = chunks[future].project_points.sites
                        result = self._handle_failed_future(future, i, sites,
                                                            timeout)

                    self.out = result

                    mem = psutil.virtual_memory()
                    m = ('Parallel run at iteration {0} out of {1}. '
                         'Memory utilization is {2:.3f} GB out of {3:.3f} GB '
                         'total ({4:.1f}% used, intended limit of {5:.1f}%)'
                         .format(i, N, mem.used / 1e9, mem.total / 1e9,
                                 100 * mem.used / mem.total,
                                 100 * self.mem_util_lim))
                    logger.info(m)

                if failed_futures:
                    logger.info('Forcing pool shutdown after failed futures.')
                    exe.shutdown(wait=False)
                    logger.info('Forced pool shutdown complete.')

        self.flush()

    def _handle_failed_future(self, future, i, sites, timeout):
        """Handle a failed future and return zeros.

        Parameters
        ----------
        future : concurrent.futures.Future
            Failed future to cancel.
        i : int
            Iteration number for logging
        sites : list
            List of site gids belonging to this failed future.
        timeout : int
            Number of seconds to wait for parallel run iteration to complete
            before returning zeros.
        """

        w = ('Iteration {} hit the timeout limit of {} seconds! Passing zeros.'
             .format(i, timeout))
        logger.warning(w)
        warn(w, OutputWarning)

        site_out = {k: 0 for k in self.output_request}
        result = {site: site_out for site in sites}

        try:
            cancelled = future.cancel()
        except Exception as e:
            w = 'Could not cancel future! Received exception: {}'.format(e)
            logger.warning(w)
            warn(w, ParallelExecutionWarning)

        if not cancelled:
            w = 'Could not cancel future!'
            logger.warning(w)
            warn(w, ParallelExecutionWarning)

        return result

    @classmethod
    def reV_run(cls, tech, points, sam_files, res_file,
                output_request=('cf_mean',), curtailment=None,
                max_workers=1, sites_per_worker=None,
                pool_size=(os.cpu_count() * 2), timeout=1800,
                points_range=None, fout=None,
                dirout='./gen_out', mem_util_lim=0.4, scale_outputs=True):
        """Execute a parallel reV generation run with smart data flushing.

        Parameters
        ----------
        tech : str
            SAM technology to analyze (pvwattsv7, windpower, tcsmoltensalt,
            solarwaterheat, troughphysicalheat, lineardirectsteam)
            The string should be lower-cased with spaces and _ removed.
        points : slice | list | str | reV.config.project_points.PointsControl
            Slice specifying project points, or string pointing to a project
            points csv, or a fully instantiated PointsControl object.
        sam_files : dict | str | list | SAMConfig
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s), top level value is the SAM path. Can also be a single
            config file str. If it's a list, it is mapped to the sorted list
            of unique configs requested by points csv. Can also be a
            pre loaded SAMConfig object.
        res_file : str
            Filepath to single resource file, multi-h5 directory,
            or /h5_dir/prefix*suffix
        output_request : list | tuple
            Output variables requested from SAM.
        curtailment : NoneType | dict | str | config.curtailment.Curtailment
            Inputs for curtailment parameters. If not None, curtailment inputs
            are expected. Can be:
                - Explicit namespace of curtailment variables (dict)
                - Pointer to curtailment config json file with path (str)
                - Instance of curtailment config object
                  (config.curtailment.Curtailment)
        max_workers : int
            Number of local workers to run on.
        sites_per_worker : int | None
            Number of sites to run in series on a worker. None defaults to the
            resource file chunk size.
        pool_size : int
            Number of futures to submit to a single process pool for
            parallel futures.
        timeout : int | float
            Number of seconds to wait for parallel run iteration to complete
            before returning zeros. Default is 1800 seconds.
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. To be taken from the reV.config.PointsControl.split_range
            property.
        fout : str | None
            Optional .h5 output file specification. Object will be returned
            if None.
        dirout : str | None
            Optional output directory specification. The directory will be
            created if it does not already exist.
        mem_util_lim : float
            Memory utilization limit (fractional). This will determine how many
            site results are stored in memory at any given time.
        scale_outputs : bool
            Flag to scale outputs in-place immediately upon Gen returning data.

        Returns
        -------
        gen : Gen
            Gen instance with outputs saved to gen.out dict
        """

        # get a points control instance
        pc = Gen.get_pc(points, points_range, sam_files, tech,
                        sites_per_worker=sites_per_worker, res_file=res_file,
                        curtailment=curtailment)

        # make a Gen class instance to operate with
        gen = cls(pc, res_file, output_request=output_request, fout=fout,
                  dirout=dirout, mem_util_lim=mem_util_lim,)

        kwargs = {'tech': gen.tech,
                  'res_file': gen.res_file,
                  'output_request': gen.output_request,
                  'scale_outputs': scale_outputs}

        logger.info('Running reV generation for: {}'.format(pc))
        logger.debug('The following project points were specified: "{}"'
                     .format(points))
        logger.debug('The following SAM configs are available to this run:\n{}'
                     .format(pprint.pformat(sam_files, indent=4)))
        logger.debug('The SAM output variables have been requested:\n{}'
                     .format(output_request))

        # use serial or parallel execution control based on max_workers
        try:
            if max_workers == 1:
                logger.debug('Running serial generation for: {}'.format(pc))
                for pc_sub in pc:
                    gen.out = gen.run(pc_sub, **kwargs)

                gen.flush()
            else:
                logger.debug('Running parallel generation for: {}'.format(pc))
                gen._parallel_run(max_workers=max_workers, pool_size=pool_size,
                                  timeout=timeout, **kwargs)

        except Exception as e:
            logger.exception('reV generation failed!')
            raise e

        return gen
