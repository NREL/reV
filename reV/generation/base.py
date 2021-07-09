# -*- coding: utf-8 -*-
"""
reV base gen and econ module.
"""
from abc import ABC, abstractmethod
import copy
from concurrent.futures import TimeoutError
import logging
import pandas as pd
import numpy as np
import os
import psutil
import json
import sys
from warnings import warn

from reV.config.project_points import ProjectPoints, PointsControl
from reV.handlers.outputs import Outputs
from reV.SAM.version_checker import PySamVersionChecker
from reV.utilities.exceptions import (OutputWarning, ExecutionError,
                                      ParallelExecutionWarning,
                                      OffshoreWindInputWarning)
from reV.utilities import log_versions

from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool

logger = logging.getLogger(__name__)


ATTR_DIR = os.path.dirname(os.path.realpath(__file__))
ATTR_DIR = os.path.join(ATTR_DIR, 'output_attributes')
with open(os.path.join(ATTR_DIR, 'other.json'), 'r') as f:
    OTHER_ATTRS = json.load(f)
with open(os.path.join(ATTR_DIR, 'lcoe_fcr.json'), 'r') as f:
    LCOE_ATTRS = json.load(f)
with open(os.path.join(ATTR_DIR, 'single_owner.json'), 'r') as f:
    SO_ATTRS = json.load(f)
with open(os.path.join(ATTR_DIR, 'windbos.json'), 'r') as f:
    BOS_ATTRS = json.load(f)
with open(os.path.join(ATTR_DIR, 'lcoe_fcr_inputs.json'), 'r') as f:
    LCOE_IN_ATTRS = json.load(f)


class BaseGen(ABC):
    """Base class for reV gen and econ classes to run SAM simulations."""

    # Mapping of reV requests to SAM objects that should be used for simulation
    OPTIONS = {}

    # Mapping of reV generation / econ outputs to scale factors and units.
    OUT_ATTRS = copy.deepcopy(OTHER_ATTRS)

    # Mapping of reV econ outputs to scale factors and units.
    # Type is scalar or array and corresponds to the SAM single-site output
    # This is the OUT_ATTRS class attr for Econ but should also be accessible
    # to rev generation
    ECON_ATTRS = copy.deepcopy(OTHER_ATTRS)
    ECON_ATTRS.update(LCOE_ATTRS)
    ECON_ATTRS.update(SO_ATTRS)
    ECON_ATTRS.update(BOS_ATTRS)
    ECON_ATTRS.update(LCOE_IN_ATTRS)

    # SAM argument names used to calculate LCOE
    # Note that system_capacity is not included here because it is never used
    # downstream and could be confused with the supply_curve point capacity
    LCOE_ARGS = ('fixed_charge_rate', 'capital_cost', 'fixed_operating_cost',
                 'variable_operating_cost')

    def __init__(self, points_control, output_request, site_data=None,
                 out_fpath=None, drop_leap=False, mem_util_lim=0.4,
                 scale_outputs=True):
        """
        Parameters
        ----------
        points_control : reV.config.project_points.PointsControl
            Project points control instance for site and SAM config spec.
        output_request : list | tuple
            Output variables requested from SAM.
        site_data : str | pd.DataFrame | None
            Site-specific input data for SAM calculation. String should be a
            filepath that points to a csv, DataFrame is pre-extracted data.
            Rows match sites, columns are input keys. Need a "gid" column.
            Input as None if no site-specific data.
        out_fpath : str, optional
            Output .h5 file path, by default None
        drop_leap : bool
            Drop leap day instead of final day of year during leap years.
        mem_util_lim : float
            Memory utilization limit (fractional). This sets how many site
            results will be stored in-memory at any given time before flushing
            to disk.
        scale_outputs : bool
            Flag to scale outputs in-place immediately upon Gen returning data.
        """
        log_versions(logger)
        self._points_control = points_control
        self._year = None
        self._site_limit = None
        self._site_mem = None
        self._out_fpath = out_fpath
        self._meta = None
        self._time_index = None
        self._sam_module = None
        self._sam_obj_default = None
        self._drop_leap = drop_leap
        self.mem_util_lim = mem_util_lim
        self.scale_outputs = scale_outputs

        self._run_attrs = {'points_control': str(points_control),
                           'output_request': output_request,
                           'out_fpath': str(out_fpath),
                           'site_data': str(site_data),
                           'drop_leap': str(drop_leap),
                           'mem_util_lim': mem_util_lim,
                           }

        self._site_data = self._parse_site_data(site_data)
        self.add_site_data_to_pp(self._site_data)
        self._output_request = self._parse_output_request(output_request)

        # pre-initialize output arrays to store results when available.
        self._out = {}
        self._finished_sites = []
        self._out_n_sites = 0
        self._out_chunk = ()
        self._check_sam_version_inputs()

    @property
    def output_request(self):
        """Get the output variables requested from the user.

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
    def site_data(self):
        """Get the site-specific inputs in dataframe format.

        Returns
        -------
        _site_data : pd.DataFrame
            Site-specific input data for gen or econ calculation. Rows match
            sites, columns are variables.
        """
        return self._site_data

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
            logger.info('Limited to storing {0} sites in memory '
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
        return self.project_points.sam_inputs

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
    def sam_module(self):
        """Get the SAM module class to be used for SAM simulations.

        Returns
        -------
        sam_module : object
            SAM object like PySAM.Pvwattsv7 or PySAM.Lcoefcr
        """
        return self._sam_module

    @property
    def out_fpath(self):
        """Get the output file path.

        Returns
        -------
        out_fpath : str | None
        """
        return self._out_fpath

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
        return self._meta

    @property
    def time_index(self):
        """Get the resource time index data.

        Returns
        -------
        _time_index : pandas.DatetimeIndex
            Time-series datetime index
        """
        return self._time_index

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
    def year(self):
        """Get the resource year.

        Returns
        -------
        _year : int
            Year of the time-series datetime index.
        """

        if self._year is None and self.time_index is not None:
            self._year = int(self.time_index.year[0])

        return self._year

    @property
    def tech(self):
        """Get the reV technology string.

        Returns
        -------
        tech : str
            SAM technology to analyze (pvwattsv7, windpower, tcsmoltensalt,
            solarwaterheat, troughphysicalheat, lineardirectsteam, econ)
            The string should be lower-cased with spaces and _ removed.
        """
        return self.project_points.tech

    @property
    def out(self):
        """Get the reV gen or econ output results.

        Returns
        -------
        out : dict
            Dictionary of gen or econ results from SAM.
        """
        out = {}
        for k, v in self._out.items():
            if k in self.OUT_ATTRS:
                scale_factor = self.OUT_ATTRS[k].get('scale_factor', 1)
            else:
                scale_factor = 1

            if scale_factor != 1 and self.scale_outputs:
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
            Gen or Econ results to set to output dictionary. Use cases:
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
            raise TypeError('Did not recognize the type of output. '
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
    def _pp_to_pc(points, points_range, sam_configs, tech,
                  sites_per_worker=None, res_file=None, curtailment=None):
        """
        Create ProjectControl from ProjectPoints

        Parameters
        ----------
        points : int | slice | list | str | pandas.DataFrame
                 | reV.config.project_points.PointsControl
            Single site integer,
            or slice or list specifying project points,
            or string pointing to a project points csv,
            or a pre-loaded project points DataFrame,
            or a fully instantiated PointsControl object.
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. To be taken from the reV.config.PointsControl.split_range
            property.
        sam_configs : dict | str | SAMConfig
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s) which map to the config column in the project points
            CSV. Values are either a JSON SAM config file or dictionary of SAM
            config inputs. Can also be a single config file path or a
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
            pp = ProjectPoints(points, sam_configs, tech=tech,
                               res_file=res_file, curtailment=curtailment)
        else:
            pp = ProjectPoints(points.df, sam_configs, tech=tech,
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

    @classmethod
    def get_pc(cls, points, points_range, sam_configs, tech,
               sites_per_worker=None, res_file=None, curtailment=None):
        """Get a PointsControl instance.

        Parameters
        ----------
        points : int | slice | list | str | pandas.DataFrame
                 | reV.config.project_points.PointsControl
            Single site integer,
            or slice or list specifying project points,
            or string pointing to a project points csv,
            or a pre-loaded project points DataFrame,
            or a fully instantiated PointsControl object.
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. To be taken from the reV.config.PointsControl.split_range
            property.
        sam_configs : dict | str | SAMConfig
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s) which map to the config column in the project points
            CSV. Values are either a JSON SAM config file or dictionary of SAM
            config inputs. Can also be a single config file path or a
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

        if tech not in cls.OPTIONS and tech.lower() != 'econ':
            msg = ('Did not recognize reV-SAM technology string "{}". '
                   'Technology string options are: {}'
                   .format(tech, list(cls.OPTIONS.keys())))
            logger.error(msg)
            raise KeyError(msg)

        if sites_per_worker is None:
            # get the optimal sites per split based on res file chunk size
            sites_per_worker = cls.get_sites_per_worker(res_file)

        logger.debug('Sites per worker being set to {} for Gen/Econ '
                     'PointsControl.'.format(sites_per_worker))

        if isinstance(points, PointsControl):
            # received a pre-intialized instance of pointscontrol
            pc = points
        else:
            pc = cls._pp_to_pc(points, points_range, sam_configs, tech,
                               sites_per_worker=sites_per_worker,
                               res_file=res_file, curtailment=curtailment)

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
                warn('Could not infer dataset chunk size as the resource type '
                     'could not be determined from the filename: {}'
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
    @abstractmethod
    def run(points_control, tech=None, res_file=None, output_request=None,
            scale_outputs=True):
        """Run a reV-SAM analysis based on the points_control iterator.

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
            Flag to scale outputs in-place immediately upon returning data.

        Returns
        -------
        out : dict
            Output dictionary from the SAM reV_run function. Data is scaled
            within this function to the datatype specified in cls.OUT_ATTRS.
        """

    def _parse_site_data(self, inp):
        """Parse site-specific data from input arg

        Parameters
        ----------
        inp : str | pd.DataFrame | None
            Site data in .csv or pre-extracted dataframe format. None signifies
            that there is no extra site-specific data and that everything is
            fully defined in the input h5 and SAM json configs.

        Returns
        -------
        site_data : pd.DataFrame
            Site-specific data for econ calculation. Rows correspond to sites,
            columns are variables.
        """

        if inp is None or inp is False:
            # no input, just initialize dataframe with site gids as index
            site_data = pd.DataFrame(index=self.project_points.sites)
            site_data.index.name = 'gid'
        else:
            # explicit input, initialize df
            if isinstance(inp, str):
                if inp.endswith('.csv'):
                    site_data = pd.read_csv(inp)
            elif isinstance(inp, pd.DataFrame):
                site_data = inp
            else:
                # site data was not able to be set. Raise error.
                raise Exception('Site data input must be .csv or '
                                'dataframe, but received: {}'.format(inp))

            if 'gid' not in site_data and site_data.index.name != 'gid':
                # require gid as column label or index
                raise KeyError('Site data input must have "gid" column '
                               'to match reV site gid.')

            if site_data.index.name != 'gid':
                # make gid the dataframe index if not already
                site_data = site_data.set_index('gid', drop=True)

        if 'offshore' in site_data:
            if site_data['offshore'].sum() > 1:
                w = ('Found offshore sites in econ site data input. '
                     'This functionality has been deprecated. '
                     'Please run the reV offshore module to '
                     'calculate offshore wind lcoe.')
                warn(w, OffshoreWindInputWarning)
                logger.warning(w)

        return site_data

    def add_site_data_to_pp(self, site_data):
        """Add the site df (site-specific inputs) to project points dataframe.

        This ensures that only the relevant site's data will be passed through
        to parallel workers when points_control is iterated and split.

        Parameters
        ----------
        site_data : pd.DataFrame
            Site-specific data for econ calculation. Rows correspond to sites,
            columns are variables.
        """
        self.project_points.join_df(site_data, key=self.site_data.index.name)

    @abstractmethod
    def _parse_output_request(self, req):
        """Set the output variables requested from the user.

        Parameters
        ----------
        req : list | tuple
            Output variables requested from SAM.

        Returns
        -------
        output_request : list
            Output variables requested from SAM.
        """

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
            data = list(self.project_points.sam_inputs.values())[0][dset]
            if isinstance(data, (list, tuple, np.ndarray, str)):
                msg = ('Cannot pass through non-scalar SAM input key "{}" '
                       'as an output_request!'.format(dset))
                logger.error(msg)
                raise ExecutionError(msg)

        else:
            if self._sam_obj_default is None:
                self._sam_obj_default = self.sam_module.default()

            try:
                out_data = getattr(self._sam_obj_default.Outputs, dset)
            except AttributeError as e:
                msg = ('Could not get data shape for dset "{}" '
                       'from object "{}". '
                       'Received the following error: "{}"'
                       .format(dset, self._sam_obj_default, e))
                logger.error(msg)
                raise ExecutionError(msg) from e
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

        if self._out_fpath is not None:

            # ensure output file is an h5
            if not self._out_fpath .endswith('.h5'):
                self._out_fpath += '.h5'

            # ensure year is in out_fpath
            if str(self.year) not in self._out_fpath:
                self._out_fpath = self._out_fpath.replace('.h5',
                                                          '_{}.h5'
                                                          .format(self.year))

            # create and use optional output dir
            dirout = os.path.dirname(self._out_fpath)
            if dirout and not os.path.exists(dirout):
                os.makedirs(dirout)

    def _init_h5(self, mode='w'):
        """Initialize the single h5 output file with all output requests.

        Parameters
        ----------
        mode : str
            Mode to instantiate h5py.File instance
        """

        if self._out_fpath is not None:

            if 'w' in mode:
                logger.info('Initializing full output file: "{}" with mode: {}'
                            .format(self._out_fpath, mode))
            elif 'a' in mode:
                logger.info('Appending data to output file: "{}" with mode: {}'
                            .format(self._out_fpath, mode))

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

            Outputs.init_h5(self._out_fpath, self.output_request, shapes,
                            attrs, chunks, dtypes, self.meta, time_index=ti,
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
            if request in self.OUT_ATTRS and self.scale_outputs:
                dtype = self.OUT_ATTRS[request].get('dtype', 'float32')

            shape = self._get_data_shape(request, self._out_n_sites)
            # initialize the output request as an array of zeros
            self._out[request] = np.zeros(shape, dtype=dtype)

    def _check_sam_version_inputs(self):
        """Check the PySAM version and input keys. Fix where necessary."""
        for key, parameters in self.project_points.sam_inputs.items():
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
        """Flush the output data in self.out attribute to disk in .h5 format.

        The data to be flushed is accessed from the instance attribute
        "self.out". The disk target is based on the instance attributes
        "self._out_fpath". Data is not flushed if _fpath is None or if .out is
        empty.
        """

        # handle output file request if file is specified and .out is not empty
        if isinstance(self._out_fpath, str) and self._out:
            logger.info('Flushing outputs to disk, target file: "{}"'
                        .format(self._out_fpath))

            # get the slice of indices to write outputs to
            islice = slice(self.out_chunk[0], self.out_chunk[1] + 1)

            # open output file in append mode to add output results to
            with Outputs(self._out_fpath, mode='a') as f:

                # iterate through all output requests writing each as a dataset
                for dset, arr in self._out.items():
                    if len(arr.shape) == 1:
                        # write array of scalars
                        f[dset, islice] = arr
                    else:
                        # write 2D array of profiles
                        f[dset, :, islice] = arr

            logger.debug('Flushed output successfully to disk.')

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
            loggers = [__name__, 'reV.gen', 'reV.econ', 'reV']
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
