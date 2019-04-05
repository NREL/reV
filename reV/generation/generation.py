"""
Generation
"""
import logging
import numpy as np
import os
import pprint
import re
import sys
import psutil
from warnings import warn

from reV.SAM.generation import PV, CSP, LandBasedWind, OffshoreWind
from reV.config.project_points import ProjectPoints, PointsControl
from reV.utilities.execution import (execute_parallel, execute_single,
                                     SmartParallelJob)
from reV.handlers.outputs import Outputs
from reV.handlers.resource import Resource


logger = logging.getLogger(__name__)


class Gen:
    """Base class for reV generation."""

    # Mapping of reV technology strings to SAM generation functions
    OPTIONS = {'pv': PV.reV_run,
               'csp': CSP.reV_run,
               'wind': LandBasedWind.reV_run,
               'landbasedwind': LandBasedWind.reV_run,
               'offshorewind': OffshoreWind.reV_run,
               }

    # Mapping of reV generation outputs to scale factors and units.
    # Type is scalar or array and corresponds to the SAM single-site output
    OUT_ATTRS = {'cf_mean': {'scale_factor': 1000, 'units': 'unitless',
                             'dtype': 'uint16', 'chunks': None,
                             'type': 'scalar'},
                 'cf_profile': {'scale_factor': 1000, 'units': 'unitless',
                                'dtype': 'uint16', 'chunks': (None, 100),
                                'type': 'array'},
                 'annual_energy': {'scale_factor': 1, 'units': 'kWh',
                                   'dtype': 'float32', 'chunks': None,
                                   'type': 'scalar'},
                 'energy_yield': {'scale_factor': 1, 'units': 'kWh/kW',
                                  'dtype': 'float32', 'chunks': None,
                                  'type': 'scalar'},
                 'gen_profile': {'scale_factor': 1, 'units': 'kW',
                                 'dtype': 'float32', 'chunks': (None, 100),
                                 'type': 'array'},
                 'poa': {'scale_factor': 1, 'units': 'W/m2',
                         'dtype': 'float32', 'chunks': (None, 100),
                         'type': 'array'},
                 'ppa_price': {'scale_factor': 1, 'units': 'dol/MWh',
                               'dtype': 'float32', 'chunks': None,
                               'type': 'scalar'},
                 'lcoe_fcr': {'scale_factor': 1, 'units': 'dol/MWh',
                              'dtype': 'float32', 'chunks': None,
                              'type': 'scalar'},
                 }

    def __init__(self, points_control, res_file, output_request=('cf_mean',),
                 fout=None, dirout='./gen_out', drop_leap=False,
                 mem_util_lim=0.7):
        """
        Parameters
        ----------
        points_control : reV.config.project_points.PointsControl
            Project points control instance for site and SAM config spec.
        res_file : str
            Resource file with path.
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
        self.output_request = output_request
        self._fout = fout
        self._dirout = dirout
        self._drop_leap = drop_leap
        self.mem_util_lim = mem_util_lim

        if self.tech not in self.OPTIONS:
            raise KeyError('Requested technology "{}" is not available. '
                           'reV generation can analyze the following '
                           'technologies: {}'
                           .format(self.tech, list(self.OPTIONS.keys())))

        # pre-initialize output arrays to store results when available.
        self.initialize_output_arrays()

    @property
    def output_request(self):
        """Get the output variables requested from generation.

        Returns
        -------
        output_request : tuple
            Output variables requested from SAM.
        """
        return self._output_request

    @output_request.setter
    def output_request(self, req):
        """Set the output variables requested from generation.

        Parameters
        ----------
        req : str | list | tuple
            Output variable(s) requested from SAM.
        """

        if isinstance(req, str):
            # single output request, make tuple
            self._output_request = (req,)
        elif isinstance(req, (list, tuple)):
            # ensure output request is tuple
            self._output_request = tuple(req)
        else:
            raise TypeError('Output request must be str, list, or tuple but '
                            'received: {}'.format(type(req)))

        if 'cf_mean' not in self._output_request:
            # ensure that cf_mean is requested from output
            self._output_request += ('cf_mean',)

        for request in self._output_request:
            if request not in self.OUT_ATTRS:
                raise ValueError('User output request "{}" not recognized. '
                                 'The following output requests are available '
                                 'in "{}": "{}"'
                                 .format(request, self.__class__,
                                         list(self.OUT_ATTRS.keys())))

    @property
    def site_limit(self):
        """Get the number of sites results that can be stored in memory at once

        Returns
        -------
        _site_limit : int
            Number of site result sets that can be stored in memory at once
            without violating memory limits.
        """

        if not hasattr(self, '_site_limit'):
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

        if not hasattr(self, '_site_mem'):
            # average the memory usage over n sites
            # (for better understanding of array overhead)
            n = 100
            self._site_mem = 0
            for request in self._output_request:
                dtype = self.OUT_ATTRS[request].get('dtype', 'float32')
                if self.OUT_ATTRS[request]['type'] == 'array':
                    ti_len = len(self.time_index)
                    shape = (ti_len, n)
                else:
                    shape = (n, )
                self._site_mem += sys.getsizeof(np.ones(shape, dtype=dtype))
            self._site_mem = self._site_mem / 1e6 / n
            logger.info('Output results from a single site are calculated to '
                        'use {0:.3f} MB of memory.'
                        .format(self._site_mem))
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
    def tech(self):
        """Get the reV technology string.

        Returns
        -------
        tech : str
            reV technology being executed (e.g. pv, csp, wind).
        """
        return self.project_points.tech

    @property
    def res_file(self):
        """Get the resource filename and path.

        Returns
        -------
        res_file : str
            Resource file with path.
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
        """Get resource meta for sites with results (in self.finished_sites).

        Returns
        -------
        _meta : pd.DataFrame
            Meta data df for sites that have completed results.
            Column names are variables, rows are different sites. The row index
            does not indicate the site number, so a 'gid' column is added.
        """

        with Resource(self.res_file) as res:
            self._meta = res['meta', self.finished_sites]
            self._meta.loc[:, 'gid'] = self.finished_sites
            self._meta.loc[:, 'reV_tech'] = self.project_points.tech
        return self._meta

    @property
    def time_index(self):
        """Get the generation resource time index data.

        Returns
        -------
        _time_index : pandas.DatetimeIndex
            Time-series datetime index
        """

        if not hasattr(self, '_time_index'):
            with Resource(self.res_file) as res:
                self._time_index = res.time_index

            # drop leap day or last day
            leap_day = ((self._time_index.month == 2) &
                        (self._time_index.day == 29))
            last_day = ((self._time_index.month == 12) &
                        (self._time_index.day == 31))
            if self._drop_leap:
                self._time_index = self._time_index.drop(
                    self._time_index[leap_day])
            elif any(leap_day):
                self._time_index = self._time_index.drop(
                    self._time_index[last_day])

        return self._time_index

    @staticmethod
    def get_pc(points, points_range, sam_files, tech, sites_per_split=None,
               res_file=None, curtailment=None):
        """Get a PointsControl instance.

        Parameters
        ----------
        points : slice | str | reV.config.project_points.PointsControl
            Slice specifying project points, or string pointing to a project
            points csv, or a fully instantiated PointsControl object.
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. To be taken from the reV.config.PointsControl.split_range
            property.
        sam_files : dict | str | list
            Dict contains SAM input configuration ID(s) and file path(s).
            Keys are the SAM config ID(s), top level value is the SAM path.
            Can also be a single config file str. If it's a list, it is mapped
            to the sorted list of unique configs requested by points csv.
        tech : str
            Technology to analyze (pv, csp, landbasedwind, offshorewind).
        sites_per_split : int
            Number of sites to run in series on a core. None defaults to the
            resource file chunk size.
        res_file : str
            Single resource file with path.
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

        if sites_per_split is None:
            # get the optimal sites per split based on res file chunk size
            sites_per_split = Gen.sites_per_core(res_file)

        if isinstance(points, (slice, str)):
            # make Project Points instance
            pp = ProjectPoints(points, sam_files, tech=tech, res_file=res_file,
                               curtailment=curtailment)

            #  make Points Control instance
            if points_range is None:
                # PointsControl is for all of the project points
                pc = PointsControl(pp, sites_per_split=sites_per_split)
            else:
                # PointsControl is for just a subset of the projec points...
                # this is the case if generation is being initialized on one
                # of many HPC nodes in a large project
                pc = PointsControl.split(points_range[0], points_range[1], pp,
                                         sites_per_split=sites_per_split)

        elif isinstance(points, PointsControl):
            # received a pre-intialized instance of pointscontrol
            pc = points
        else:
            raise TypeError('Points input type is unrecognized: '
                            '"{}"'.format(type(points)))
        return pc

    @staticmethod
    def sites_per_core(res_file, default=100):
        """Get the nominal sites per core (x-chunk size) for a given file.

        This is based on the concept that it is most efficient for one core to
        perform one read on one chunk of resource data, such that chunks will
        not have to be read into memory twice and no sites will be read
        redundantly.

        Parameters
        ----------
        res_file : str
            Full resource file path + filename.
        default : int
            Sites to be analyzed on a single core if the chunk size cannot be
            determined from res_file.

        Returns
        -------
        sites_per_core : int
            Nominal sites to be analyzed per core. This is set to the x-axis
            chunk size for windspeed and dni datasets for the WTK and NSRDB
            data, respectively.
        """
        if not res_file:
            return default

        with Resource(res_file) as res:
            if 'wtk' in res_file.lower():
                for dset in res.dsets:
                    if 'speed' in dset:
                        # take nominal WTK chunks from windspeed
                        chunks = res._h5[dset].chunks
                        break
            elif 'nsrdb' in res_file.lower():
                # take nominal NSRDB chunks from dni
                chunks = res._h5['dni'].chunks
            else:
                warn('Expected "nsrdb" or "wtk" to be in resource filename: {}'
                     .format(res_file))
                chunks = None

        if chunks is None:
            # if chunks not set, go to default
            sites_per_core = default
            logger.debug('Sites per core being set to {} (default) based on '
                         'no set chunk size in {}.'
                         .format(sites_per_core, res_file))
        else:
            sites_per_core = chunks[1]
            logger.debug('Sites per core being set to {} based on chunk size '
                         'of {}.'.format(sites_per_core, res_file))
        return sites_per_core

    def initialize_output_arrays(self, index_0=0):
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
        self.finished_sites = []

        # Output chunk is the index range (inclusive) of this set of site outs
        self.output_chunk = (index_0, np.min((index_0 + self.site_limit,
                                              len(self.project_points) - 1)))
        self.out_n_sites = int(self.output_chunk[1] - self.output_chunk[0]) + 1

        logger.info('Initializing generation outputs for {} sites with gids '
                    '{} through {} inclusive (site list index {} through {})'
                    .format(self.out_n_sites,
                            self.project_points.sites[self.output_chunk[0]],
                            self.project_points.sites[self.output_chunk[1]],
                            self.output_chunk[0], self.output_chunk[1]))

        for request in self.output_request:
            dtype = self.OUT_ATTRS[request].get('dtype', 'float32')
            if self.OUT_ATTRS[request]['type'] == 'array':
                shape = (len(self.time_index), self.out_n_sites)
            else:
                shape = (self.out_n_sites, )

            # initialize the output request as an array of zeros
            self._out[request] = np.zeros(shape, dtype=dtype)

    @property
    def out(self):
        """Get the generation output results.

        Returns
        -------
        out : dict
            Dictionary of generation results from SAM.
        """

        if not hasattr(self, '_out'):
            self.initialize_output_arrays()
        return self._out

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

        if not hasattr(self, '_out'):
            # initialize output dict and list of finished sites
            self.initialize_output_arrays()

        if isinstance(result, list):
            # unpack futures list to dictionary first
            result = self.unpack_futures(result)

        if isinstance(result, dict):

            # iterate through dict where sites are keys and values are
            # corresponding results
            for site_gid, site_output in result.items():

                # check that the sites are stored sequentially then add to
                # the finished site list
                if self.finished_sites:
                    if int(site_gid) < np.max(self.finished_sites):
                        raise Exception('Site results are non sequential!')

                # unpack site output object
                self.unpack_output(site_gid, site_output)

                # add site gid to the finished list after outputs are unpacked
                self.finished_sites.append(site_gid)

            # try to clear some memory
            del result

        elif isinstance(result, type(None)):
            self._out.clear()
            self.finished_sites.clear()
        else:
            raise TypeError('Did not recognize the type of generation output. '
                            'Tried to set output type "{}", but requires '
                            'list, dict or None.'.format(type(result)))

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

    def unpack_output(self, site_gid, site_output):
        """Unpack a SAM SiteOutput object to the output attribute.

        Parameters
        ----------
        site_gid : int
            Resource-native site gid (index).
        site_output : SAM.SiteOutput
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
            if i + 1 > self.out_n_sites:
                self.flush()
                global_site_index = self.site_index(site_gid)
                self.initialize_output_arrays(index_0=global_site_index)
                i = self.site_index(site_gid, out_index=True)

            if isinstance(value, np.ndarray):
                # set the new timeseries to the 2D array
                self._out[var][:, i] = value.T
            else:
                # set a scalar result to the list (1D array)
                self._out[var][i] = value

        # try to clear some memory
        del site_output

    def site_index(self, site_gid, out_index=False):
        """Get the index corresponding to the site gid.

        Parameters
        ----------
        site_gid : int
            Resource-native site index (gid).
        out_index : bool
            Option to get output index (if true) which is the column index in
            the current output array or the the global site index from the
            project points site list (if false).

        Returns
        -------
        index : int
            Global site index if out_index=False, otherwise column index in
            the output array.
        """

        # get the index for site_gid in the (global) project points site list.
        global_site_index = self.project_points.sites.index(site_gid)

        if not out_index:
            return global_site_index
        else:
            output_index = global_site_index - self.output_chunk[0]
            if output_index < 0:
                raise ValueError('Attempting to set output data for site with '
                                 'gid {} to global site index {}, which was '
                                 'already set based on the current output '
                                 'index chunk of {}'
                                 .format(site_gid, global_site_index,
                                         self.output_chunk))
            return output_index

    def get_dset_attrs(self, var):
        """Get dataset attributes associated with output variable.

        Parameters
        ----------
        var : str
            SAM variable name to be unpacked from gen_out, also the intended
            dataset name that will be written to disk.

        Returns
        -------
        data : np.ndarray
            1D array of scalar values sorted by site number or 2D array of
            profile outputs with rows matching the time series and columns
            matching sorted rows.
        dtype : str
            Target dataset datatype. Defaults to float32.
        chunks : list | tuple | NoneType
            Chunk shape for target dataset. Defaults to None.
        attrs : dict
            Additional dataset attributes including scale_factor and units.
        """

        data = self.out[var]
        dtype = self.OUT_ATTRS[var].get('dtype', 'float32')
        chunks = self.OUT_ATTRS[var].get('chunks', None)
        attrs = {k: self.OUT_ATTRS[var].get(k, 'None') for
                 k in ['scale_factor', 'units']}
        return data, dtype, chunks, attrs

    def gen_to_disk(self, fout='gen_out.h5'):
        """Save generation outputs to disk (all vars in self.output_request).

        Parameters
        ----------
        fout : str
            Target .h5 output file (with path).
        """

        with Outputs(fout, mode='w-') as f:
            # Save meta
            f['meta'] = self.meta
            logger.debug("\t- 'meta' saved to disc")

            if 'profile' in str(self.output_request):
                f['time_index'] = self.time_index
                logger.debug("\t- 'time_index' saved to disc")

            if self.sam_configs is not None:
                f.set_configs(self.sam_configs)
                logger.debug("\t- SAM configurations saved as attributes "
                             "on 'meta'")

            # iterate through all output requests writing each as a dataset
            for dset in self.output_request:
                # retrieve the dataset with associated attributes
                data, dtype, chunks, attrs = self.get_dset_attrs(dset)
                # Write output dataset to disk
                f._add_dset(dset_name=dset, data=data, dtype=dtype,
                            chunks=chunks, attrs=attrs)

    @staticmethod
    def get_unique_fout(fout):
        """Ensure a unique tag of format _x000 on the fout file name.

        Parameters
        ----------
        fout : str
            Target output directory joined with the INTENDED filename. Should
            contain a _x000 tag in the filename.

        Returns
        -------
        fout : str
            Target output directory joined with a UNIQUE filename. The
            extension in the original fout ("_x000") is incremented until the
            result is unique in the output directory.
        """

        if os.path.exists(fout):
            match = re.match(r'.*_x([0-9]{3})', fout)
            if match:
                new_tag = '_x' + str(int(match.group(1)) + 1).zfill(3)
                fout = fout.replace('_x' + match.group(1), new_tag)
                fout = Gen.get_unique_fout(fout)
        return fout

    @staticmethod
    def handle_fout(fout, dirout):
        """Ensure that the file+dir output exist and have unique names.

        Parameters
        ----------
        fout : str
            Target filename (with or without .h5 extension).
        dirout : str
            Target output directory.

        Returns
        -------
        fout : str
            Target output directory joined with the target filename. An
            extension is appended to the filename in the format
            "basename_x000.h5" where basename is the input fout and _x000
            creates a unique filename in the output directory.
        """

        # combine filename and path
        fout = Gen.make_h5_fpath(fout, dirout)

        # check to see if target already exists. If so, assign unique ID.
        fout = fout.replace('.h5', '_x000.h5')
        fout = Gen.get_unique_fout(fout)

        return fout

    @staticmethod
    def make_h5_fpath(fout, dirout):
        """Combine directory and filename and ensure .h5 extension.

        Parameters
        ----------
        fout : str
            Target filename (with or without .h5 extension).
        dirout : str
            Target output directory.

        Returns
        -------
        fout : str
            Target output directory joined with the target filename
            ending in .h5.
        """

        if not fout.endswith('.h5'):
            fout += '.h5'
        # create and use optional LCOE output dir
        if dirout:
            if not os.path.exists(dirout):
                os.makedirs(dirout)
            # Add output dir to LCOE fout string
            fout = os.path.join(dirout, fout)
        return fout

    def flush(self):
        """Flush generation data in self.out attribute to disk in .h5 format.

        The data to be flushed is accessed from the instance attribute
        "self.out". The disk target is based on the isntance attributes
        "self.fout" and "self.dirout". The flushed file is ensured to have a
        unique filename. Data is not flushed if fout is None or if .out is
        empty.
        """

        # use mutable copies of the properties
        fout = self.fout
        dirout = self.dirout

        # handle output file request if file is specified and .out is not empty
        if isinstance(fout, str) and self.out:
            fout = self.handle_fout(fout, dirout)

            logger.info('Flushing generation outputs to disk, target file: {}'
                        .format(fout))
            self.gen_to_disk(fout)

            logger.debug('Flushed generation output successfully to disk.')

    @staticmethod
    def run(points_control, tech=None, res_file=None, output_request=None,
            scale_outputs=True):
        """Run a SAM generation analysis based on the points_control iterator.

        Parameters
        ----------
        points_control : reV.config.PointsControl
            A PointsControl instance dictating what sites and configs are run.
            This function uses an explicit points_control input instance
            instead of an instance attribute so that the execute_futures
            can pass in a split instance of points_control. This is a
            @staticmethod to expedite submission to Dask client.
        tech : str
            Technology to analyze (pv, csp, landbasedwind, offshorewind).
        res_file : str
            Single resource file with path.
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
            out = Gen.OPTIONS[tech](points_control, res_file, output_request)
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

    @classmethod
    def run_direct(cls, tech=None, points=None, sam_files=None, res_file=None,
                   output_request=('cf_mean',), curtailment=None, n_workers=1,
                   sites_per_split=None, points_range=None, fout=None,
                   dirout='./gen_out', return_obj=True, scale_outputs=True):
        """Execute a generation run directly from source files without config.

        Parameters
        ----------
        tech : str
            Technology to analyze (pv, csp, landbasedwind, offshorewind).
        points : slice | str | reV.config.project_points.PointsControl
            Slice specifying project points, or string pointing to a project
            points csv, or a fully instantiated PointsControl object.
        sam_files : dict | str | list
            Dict contains SAM input configuration ID(s) and file path(s).
            Keys are the SAM config ID(s), top level value is the SAM path.
            Can also be a single config file str. If it's a list, it is mapped
            to the sorted list of unique configs requested by points csv.
        res_file : str
            Single resource file with path.
        output_request : list | tuple
            Output variables requested from SAM.
        curtailment : NoneType | dict | str | config.curtailment.Curtailment
            Inputs for curtailment parameters. If not None, curtailment inputs
            are expected. Can be:
                - Explicit namespace of curtailment variables (dict)
                - Pointer to curtailment config json file with path (str)
                - Instance of curtailment config object
                  (config.curtailment.Curtailment)
        n_workers : int
            Number of local workers to run on.
        sites_per_split : int
            Number of sites to run in series on a core. None defaults to the
            resource file chunk size.
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. To be taken from the reV.config.PointsControl.split_range
            property.
        fout : str | None
            Optional .h5 output file specification.
        dirout : str | None
            Optional output directory specification. The directory will be
            created if it does not already exist.
        return_obj : bool
            Option to return the Gen object instance.
        scale_outputs : bool
            Flag to scale outputs in-place immediately upon Gen returning data.

        Returns
        -------
        gen : reV.generation.Gen
            Generation object instance with outputs stored in .out attribute.
            Only returned if return_obj is True.
        """

        # get a points control instance
        pc = Gen.get_pc(points, points_range, sam_files, tech, sites_per_split,
                        res_file=res_file, curtailment=curtailment)

        # make a Gen class instance to operate with
        gen = cls(pc, res_file, output_request=output_request, fout=fout,
                  dirout=dirout)

        kwargs = {'tech': gen.tech, 'res_file': gen.res_file,
                  'output_request': gen.output_request,
                  'scale_outputs': scale_outputs}

        # use serial or parallel execution control based on n_workers
        if n_workers == 1:
            logger.debug('Running serial generation for: {}'.format(pc))
            out = execute_single(gen.run, pc, **kwargs)
        else:
            logger.debug('Running parallel generation for: {}'.format(pc))
            out = execute_parallel(gen.run, pc, n_workers=n_workers,
                                   loggers=[__name__, 'reV.SAM'], **kwargs)

        # save output data to object attribute
        gen.out = out

        # flush output data (will only write to disk if fout is a str)
        gen.flush()

        # optionally return Gen object (useful for debugging and hacking)
        if return_obj:
            return gen

    @classmethod
    def run_smart(cls, tech=None, points=None, sam_files=None, res_file=None,
                  output_request=('cf_mean',), curtailment=None, n_workers=1,
                  sites_per_split=None, points_range=None, fout=None,
                  dirout='./gen_out', mem_util_lim=0.7, scale_outputs=True):
        """Execute a generation run with smart data flushing.

        Parameters
        ----------
        tech : str
            Technology to analyze (pv, csp, landbasedwind, offshorewind).
        points : slice | str | reV.config.project_points.PointsControl
            Slice specifying project points, or string pointing to a project
            points csv, or a fully instantiated PointsControl object.
        sam_files : dict | str | list
            Dict contains SAM input configuration ID(s) and file path(s).
            Keys are the SAM config ID(s), top level value is the SAM path.
            Can also be a single config file str. If it's a list, it is mapped
            to the sorted list of unique configs requested by points csv.
        res_file : str
            Single resource file with path.
        output_request : list | tuple
            Output variables requested from SAM.
        curtailment : NoneType | dict | str | config.curtailment.Curtailment
            Inputs for curtailment parameters. If not None, curtailment inputs
            are expected. Can be:
                - Explicit namespace of curtailment variables (dict)
                - Pointer to curtailment config json file with path (str)
                - Instance of curtailment config object
                  (config.curtailment.Curtailment)
        n_workers : int
            Number of local workers to run on.
        sites_per_split : int | None
            Number of sites to run in series on a core. None defaults to the
            resource file chunk size.
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. To be taken from the reV.config.PointsControl.split_range
            property.
        fout : str | None
            Optional .h5 output file specification.
        dirout : str | None
            Optional output directory specification. The directory will be
            created if it does not already exist.
        mem_util_lim : float
            Memory utilization limit (fractional). This will determine how many
            site results are stored in memory at any given time.
        scale_outputs : bool
            Flag to scale outputs in-place immediately upon Gen returning data.
        """

        # get a points control instance
        pc = Gen.get_pc(points, points_range, sam_files, tech, sites_per_split,
                        res_file=res_file, curtailment=curtailment)

        # make a Gen class instance to operate with
        gen = cls(pc, res_file, output_request=output_request, fout=fout,
                  dirout=dirout, mem_util_lim=mem_util_lim)

        kwargs = {'tech': gen.tech, 'res_file': gen.res_file,
                  'output_request': gen.output_request,
                  'scale_outputs': scale_outputs}

        logger.info('Running parallel generation with smart data flushing '
                    'for: {}'.format(pc))
        logger.debug('The following project points were specified: "{}"'
                     .format(points))
        logger.debug('The following SAM configs are available to this run:\n{}'
                     .format(pprint.pformat(sam_files, indent=4)))
        logger.debug('The SAM output variables have been requested:\n{}'
                     .format(output_request))
        try:
            # use SmartParallelJob to manage runs, but set mem limit to 1
            # because Gen() will manage the sites in-memory
            SmartParallelJob.execute(gen, pc, n_workers=n_workers,
                                     loggers=['reV.generation', 'reV.SAM',
                                              'reV.utilities'],
                                     mem_util_lim=1.0, **kwargs)
        except Exception as e:
            logger.exception('SmartParallelJob.execute() failed.')
            raise e
