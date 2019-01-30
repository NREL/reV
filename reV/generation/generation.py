"""
Generation
"""
from copy import deepcopy
import logging
import numpy as np
import os
import pprint
import re
from warnings import warn

from reV.SAM.generation import PV, CSP, LandBasedWind, OffshoreWind
from reV.config.project_points import ProjectPoints, PointsControl
from reV.utilities.execution import (execute_parallel, execute_single,
                                     SmartParallelJob)
from reV.handlers.outputs import Outputs
from reV.handlers.resource import Resource


logger = logging.getLogger(__name__)


class Gen:
    """Base class for generation"""

    # Mapping of reV technology strings to SAM generation functions
    OPTIONS = {'pv': PV.reV_run,
               'csp': CSP.reV_run,
               'wind': LandBasedWind.reV_run,
               'landbasedwind': LandBasedWind.reV_run,
               'offshorewind': OffshoreWind.reV_run,
               }

    OUT_ATTRS = {'cf_means': {'scale_factor': 1000, 'units': 'unitless',
                              'dtype': 'uint16'},
                 'cf_profiles': {'scale_factor': 1000, 'units': 'unitless',
                                 'dtype': 'uint16'},
                 }

    def __init__(self, points_control, res_file, output_request=('cf_mean',),
                 fout=None, dirout='./gen_out'):
        """Initialize a generation instance.

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
        """

        self._points_control = points_control
        self._res_file = res_file
        self._output_request = output_request
        self._fout = fout
        self._dirout = dirout

        if self.tech not in self.OPTIONS:
            raise KeyError('Requested technology "{}" is not available. '
                           'reV generation can analyze the following '
                           'technologies: {}'
                           .format(self.tech, list(self.OPTIONS.keys())))

    @property
    def output_request(self):
        """Get the list of output variables requested from generation.

        Returns
        -------
        output_request : list | tuple
            Output variables requested from SAM.
        """
        return self._output_request

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
        """Get resource meta data for the analyzed sites stored in self._out.

        Returns
        -------
        _meta : pd.DataFrame
            Meta data df for sites that have completed results in self._out.
            Column names are variables, rows are different sites. The row index
            does not indicate the site number, so a 'gid' column is added.
        """

        if hasattr(self, '_out'):
            finished_sites = sorted(list(self._out.keys()))
        with Resource(self.res_file) as res:
            self._meta = res['meta', finished_sites]
            self._meta.loc[:, 'gid'] = finished_sites
            self._meta.loc[:, 'reV_tech'] = self.project_points.tech
        return self._meta

    @property
    def out(self):
        """Get the generation output results.

        Returns
        -------
        out : dict
            Dictionary of generation results from SAM.
        """

        if not hasattr(self, '_out'):
            self._out = {}
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
            self._out = {}
        if isinstance(result, list):
            self._out.update(self.unpack_futures(result))
        elif isinstance(result, dict):
            self._out.update(result)
        elif isinstance(result, type(None)):
            self._out.clear()
        else:
            raise TypeError('Did not recognize the type of generation output. '
                            'Tried to set output type "{}", but requires '
                            'list, dict or None.'.format(type(result)))

    @property
    def time_index(self, drop_leap=True):
        """Get the generation resource time index data.

        Parameters
        ----------
        drop_leap : bool
            Option to drop the leap day from the time_index.

        Returns
        -------
        _time_index : pd.DatetimeIndex
            Time index objects from the resource data (self.res_file).
        """
        if not hasattr(self, '_time_index'):
            with Resource(self.res_file) as res:
                self._time_index = res.time_index
            if drop_leap:
                leap_day = ((self._time_index.month == 2) &
                            (self._time_index.day == 29))
                self._time_index = self._time_index.drop(
                    self._time_index[leap_day])
        return self._time_index

    @staticmethod
    def get_pc(points, points_range, sam_files, tech, sites_per_split=None,
               res_file=None):
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

        Returns
        -------
        pc : reV.config.project_points.PointsControl
            PointsControl object instance.
        """

        if sites_per_split is None:
            sites_per_split = Gen.sites_per_core(res_file)

        if isinstance(points, (slice, str)):
            # make Project Points and Points Control instances
            pp = ProjectPoints(points, sam_files, tech=tech, res_file=res_file)
            if points_range is None:
                pc = PointsControl(pp, sites_per_split=sites_per_split)
            else:
                pc = PointsControl.split(points_range[0], points_range[1], pp,
                                         sites_per_split=sites_per_split)
        elif isinstance(points, PointsControl):
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
        {out.update(x) for x in futures}
        return out

    @staticmethod
    def unpack_scalars(gen_out, sam_var='cf_mean'):
        """Unpack a numpy 1darray of scalars from a gen output dictionary.

        Parameters
        ----------
        gen_out : dict
            Nested dictionary of SAM results. Top level key is site number,
            Next level key should include the target sam_var.
        sam_var : str
            SAM variable name to be unpacked from gen_out. The SAM outputs
            associated with this variable must be scalar values.

        Returns
        -------
        out : np.array
            1D array of scalar values sorted by site number.
        """

        sorted_keys = sorted(list(gen_out.keys()), key=float)
        out = np.array([gen_out[k][sam_var] for k in sorted_keys])
        return out

    @staticmethod
    def unpack_profiles(gen_out, sam_var='cf_profile'):
        """Unpack a numpy 2darray of profiles from a gen output dictionary.

        Parameters
        ----------
        gen_out : dict
            Nested dictionary of SAM results. Top level key is site number,
            Next level key should include the target sam_var.
        sam_var : str
            SAM variable name to be unpacked from gen_out. The SAM outputs
            associated with this variable must be profiles.

        Returns
        -------
        out : np.ndarray
            2D array of profiles. Columns are sorted by site
            number. Rows correspond to the profile timeseries.
        """

        sorted_keys = sorted(list(gen_out.keys()), key=float)
        out = np.array([gen_out[k][sam_var] for k in sorted_keys])
        return out.transpose()

    def means_to_disk(self, fout='gen_out.h5', mode='w'):
        """Save capacity factor means to disk.

        Parameters
        ----------
        fout : str
            Target .h5 output file (with path).
        mode : str
            .h5 file write mode (e.g. 'w', 'w-', 'a').
        """
        cf_means = self.unpack_scalars(self.out, sam_var='cf_mean')
        meta = self.meta
        meta.loc[:, 'cf_means'] = cf_means
        # get LCOE if in output request, otherwise default to None
        lcoe = None
        if 'lcoe' in str(self.output_request):
            lcoe = self.unpack_scalars(self.out, sam_var='lcoe_fcr')

        # get dset attributes
        attrs = deepcopy(self.OUT_ATTRS['cf_means'])
        dtype = attrs['dtype']
        del attrs['dtype']

        Outputs.write_means(fout, meta, 'cf', cf_means, attrs, dtype,
                            self.sam_configs, lcoe=lcoe, **{'mode': mode})

    def profiles_to_disk(self, fout='gen_out.h5', mode='w'):
        """Save capacity factor profiles to disk.

        Parameters
        ----------
        fout : str
            Target .h5 output file (with path).
        mode : str
            .h5 file write mode (e.g. 'w', 'w-', 'a').
        """
        cf_profiles = self.unpack_profiles(self.out, sam_var='cf_profile')
        meta = self.meta
        meta.loc[:, 'cf_means'] = self.unpack_scalars(self.out,
                                                      sam_var='cf_mean')
        # get LCOE if in output request, otherwise default to None
        lcoe = None
        if 'lcoe' in str(self.output_request):
            lcoe = self.unpack_scalars(self.out, sam_var='lcoe_fcr')

        # get dset attributes
        attrs = deepcopy(self.OUT_ATTRS['cf_profiles'])
        dtype = attrs['dtype']
        del attrs['dtype']

        Outputs.write_profiles(fout, meta, self.time_index, 'cf_profiles',
                               cf_profiles, attrs, dtype, self.sam_configs,
                               lcoe=lcoe, **{'mode': mode})

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

    def flush(self, mode='w'):
        """Flush generation data in self.out attribute to disk in .h5 format.

        The data to be flushed is accessed from the instance attribute
        "self.out". The disk target is based on the isntance attributes
        "self.fout" and "self.dirout". The flushed file is ensured to have a
        unique filename. Data is not flushed if fout is None or if .out is
        empty.

        Parameters
        ----------
        mode : str
            .h5 file write mode (e.g. 'w', 'a').
        """

        # use mutable copies of the properties
        fout = self.fout
        dirout = self.dirout

        # handle output file request if file is specified and .out is not empty
        if isinstance(fout, str) and self.out:
            fout = self.handle_fout(fout, dirout)

            logger.info('Flushing generation outputs to disk, target file: {}'
                        .format(fout))
            if 'profile' in str(self.output_request):
                self.profiles_to_disk(fout=fout, mode=mode)
            else:
                self.means_to_disk(fout=fout, mode=mode)
            logger.debug('Flushed generation output successfully to disk.')

    @staticmethod
    def run(points_control, tech=None, res_file=None, output_request=None):
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

        Returns
        -------
        out : dict
            Output dictionary from the SAM reV_run function.
        """

        try:
            out = Gen.OPTIONS[tech](points_control, res_file, output_request)
        except Exception as e:
            out = {}
            logger.exception('Worker failed for PC: {}'.format(points_control))
            raise e

        return out

    @classmethod
    def run_direct(cls, tech=None, points=None, sam_files=None, res_file=None,
                   output_request=('cf_mean',), n_workers=1,
                   sites_per_split=None, points_range=None, fout=None,
                   dirout='./gen_out', return_obj=True):
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

        Returns
        -------
        gen : reV.generation.Gen
            Generation object instance with outputs stored in .out attribute.
            Only returned if return_obj is True.
        """

        # always extract cf mean
        if 'cf_mean' not in output_request:
            output_request += ('cf_mean',)

        # get a points control instance
        pc = Gen.get_pc(points, points_range, sam_files, tech, sites_per_split,
                        res_file=res_file)

        # make a Gen class instance to operate with
        gen = cls(pc, res_file, output_request=output_request, fout=fout,
                  dirout=dirout)

        kwargs = {'tech': gen.tech, 'res_file': gen.res_file,
                  'output_request': gen.output_request}

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
                  output_request=('cf_mean',), n_workers=1,
                  sites_per_split=None, points_range=None, fout=None,
                  dirout='./gen_out', mem_util_lim=0.7):
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
            Memory utilization limit (fractional). If the used memory divided
            by the total memory is greater than this value, the obj.out will
            be flushed and the local node memory will be cleared.
        """

        # always extract cf mean
        if 'cf_mean' not in output_request:
            output_request += ('cf_mean',)

        # get a points control instance
        pc = Gen.get_pc(points, points_range, sam_files, tech, sites_per_split,
                        res_file=res_file)

        # make a Gen class instance to operate with
        gen = cls(pc, res_file, output_request=output_request, fout=fout,
                  dirout=dirout)

        kwargs = {'tech': gen.tech, 'res_file': gen.res_file,
                  'output_request': gen.output_request}

        logger.info('Running parallel generation with smart data flushing '
                    'for: {}'.format(pc))
        logger.debug('The following project points were specified: "{}"'
                     .format(points))
        logger.debug('The following SAM configs are available to this run:\n{}'
                     .format(pprint.pformat(sam_files, indent=4)))
        logger.debug('The SAM output variables have been requested:\n{}'
                     .format(output_request))
        try:
            SmartParallelJob.execute(gen, pc, n_workers=n_workers,
                                     loggers=['reV.generation',
                                              'reV.utilities'],
                                     mem_util_lim=mem_util_lim, **kwargs)
        except Exception as e:
            logger.exception('SmartParallelJob.execute() failed.')
            raise e
