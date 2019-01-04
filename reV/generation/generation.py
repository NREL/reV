"""
Generation
"""
import logging
import numpy as np
import os
import re
from warnings import warn

from reV.SAM.SAM import PV, CSP, LandBasedWind, OffshoreWind
from reV.config.project_points import ProjectPoints, PointsControl
from reV.execution.execution import (execute_parallel, execute_single,
                                     execute_smart_parallel)
from reV.handlers.capacity_factor import CapacityFactor
from reV.handlers.resource import Resource


logger = logging.getLogger(__name__)


class Gen:
    """Base class for generation"""
    def __init__(self, points_control, res_file, output_request=('cf_mean',),
                 fout=None, dirout='./gen_out'):
        """Initialize a generation instance.

        Parameters
        ----------
        points_control : reV.config.PointsControl
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

    @property
    def output_request(self):
        """Get the list of output variables requested from generation."""
        return self._output_request

    @property
    def points_control(self):
        """Get project points controller."""
        return self._points_control

    @property
    def project_points(self):
        """Get project points"""
        return self._points_control.project_points

    @property
    def sam_configs(self):
        """Get the sam config dictionary."""
        return self.project_points.sam_configs

    @property
    def tech(self):
        """Get the reV technology string."""
        return self.project_points.tech

    @property
    def res_file(self):
        """Get the resource filename and path."""
        return self._res_file

    @property
    def fout(self):
        """Get the target file output."""
        return self._fout

    @property
    def dirout(self):
        """Get the target output directory."""
        return self._dirout

    @property
    def meta(self):
        """Get the generation resource meta data."""
        if hasattr(self, '_out'):
            finished_sites = sorted(list(self._out.keys()))
        with Resource(self.res_file) as res:
            self._meta = res['meta', finished_sites]
            self._meta.loc[:, 'gid'] = finished_sites
            self._meta.loc[:, 'reV_tech'] = self.project_points.tech
        return self._meta

    @property
    def out(self):
        """Get the generation output results."""
        if not hasattr(self, '_out'):
            self._out = {}
        return self._out

    @out.setter
    def out(self, result):
        """Set the output attribute, unpack futures, clear output from mem.
        """
        if not hasattr(self, '_out'):
            self._out = {}
        if isinstance(result, list):
            self._out.update(self.unpack_futures(result))
        elif isinstance(result, dict):
            self._out.update(result)
        elif isinstance(result, type(None)):
            del self._out
        else:
            raise TypeError('Did not recognize the type of generation output. '
                            'Tried to set output type "{}", but requires '
                            'list, dict or None.'.format(type(result)))

    @property
    def time_index(self, drop_leap=True):
        """Get the generation resource time index data."""
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
    def unpack_cf_means(gen_out):
        """Unpack a numpy means 1darray from a gen output dictionary."""
        sorted_keys = sorted(list(gen_out.keys()), key=float)
        out = np.array([gen_out[k]['cf_mean'] for k in sorted_keys])
        return out

    @staticmethod
    def unpack_cf_profiles(gen_out):
        """Unpack a numpy profiles 2darray from a gen output dictionary."""
        sorted_keys = sorted(list(gen_out.keys()), key=float)
        out = np.array([gen_out[k]['cf_profile'] for k in sorted_keys])
        return out.transpose()

    def means_to_disk(self, gen_out, fout='gen_out.h5', mode='w'):
        """Save capacity factor means to disk."""
        cf_means = self.unpack_cf_means(gen_out)
        meta = self.meta
        meta.loc[:, 'cf_means'] = self.unpack_cf_means(gen_out)
        CapacityFactor.write_means(fout, meta, cf_means, self.sam_configs,
                                   **{'mode': mode})

    def profiles_to_disk(self, gen_out, fout='gen_out.h5', mode='w'):
        """Save capacity factor profiles to disk."""
        cf_profiles = self.unpack_cf_profiles(gen_out)
        meta = self.meta
        meta.loc[:, 'cf_means'] = self.unpack_cf_means(gen_out)
        CapacityFactor.write_profiles(fout, meta, self.time_index,
                                      cf_profiles, self.sam_configs,
                                      **{'mode': mode})

    @staticmethod
    def get_unique_fout(fout):
        """Ensure a unique tag of format _n000 on the fout file name."""
        if os.path.exists(fout):
            match = re.match(r'.*n([0-9]{3})', fout)
            if match:
                new_tag = str(int(match.group(1)) + 1).zfill(3)
                fout = fout.replace(match.group(1), new_tag)
                fout = Gen.get_unique_fout(fout)
        return fout

    @staticmethod
    def handle_fout(fout, dirout):
        """Ensure that the file+dir output exist and have unique names."""
        if not fout.endswith('.h5'):
            fout += '.h5'
            warn('Generation output file request must be .h5, '
                 'set to: "{}"'.format(fout))
        # create and use optional output dir
        if dirout:
            if not os.path.exists(dirout):
                os.makedirs(dirout)
            # Add output dir to fout string
            fout = os.path.join(dirout, fout)

        # check to see if target already exists. If so, assign unique ID.
        fout = fout.replace('.h5', '_n000.h5')
        fout = Gen.get_unique_fout(fout)

        return fout

    def flush(self, mode='w'):
        """Flush generation data in self.out attribute to disk in .h5 format.

        Parameters
        ----------
        mode : str
            .h5 file write mode (e.g. 'w', 'a').
        """
        # use mutable copies of the properties
        fout = self.fout
        dirout = self.dirout

        # handle output file request
        if isinstance(fout, str):
            fout = self.handle_fout(fout, dirout)

            logger.debug('Flushing generation outputs to disk, target file: {}'
                         .format(fout))
            if 'profile' in str(self.output_request):
                self.profiles_to_disk(self.out, fout=fout, mode=mode)
            else:
                self.means_to_disk(self.out, fout=fout, mode=mode)
            logger.debug('Flushed generation output successfully to disk.')

    def run(self, points_control):
        """Run a SAM generation analysis based on the points_control iterator.

        Parameters
        ----------
        points_control : reV.config.PointsControl
            A PointsControl instance dictating what sites and configs are run.
            This function uses an explicit points_control input instance
            instead of the Gen object property so that the execute_futures
            can pass in a split instance of points_control.

        Returns
        -------
        out : dict
            Output dictionary from the SAM reV_run function.
        """

        sam_funs = {'pv': PV.reV_run,
                    'csp': CSP.reV_run,
                    'landbasedwind': LandBasedWind.reV_run,
                    'offshorewind': OffshoreWind.reV_run,
                    }

        out = sam_funs[self.tech](points_control, self.res_file,
                                  output_request=self.output_request)

        return out

    @classmethod
    def run_direct(cls, tech=None, points=None, sam_files=None, res_file=None,
                   cf_profiles=True, n_workers=1, sites_per_split=100,
                   points_range=None, fout=None, dirout='./gen_out',
                   return_obj=True):
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
        cf_profiles : bool
            Enables capacity factor annual profile output. Capacity factor
            means output if this is False.
        n_workers : int
            Number of local workers to run on.
        sites_per_split : int
            Number of sites to run in series on a core.
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

        # create the output request tuple
        output_request = ('cf_mean',)
        if cf_profiles:
            output_request += ('cf_profile',)

        if isinstance(points, (slice, str)):
            # make Project Points and Points Control instances
            pp = ProjectPoints(points, sam_files, tech, res_file=res_file)
            if points_range is None:
                pc = PointsControl(pp, sites_per_split=sites_per_split)
            else:
                pc = PointsControl.split(points_range[0], points_range[1], pp,
                                         sites_per_split=sites_per_split)
        elif isinstance(points, PointsControl):
            pc = points
        else:
            raise TypeError('Generation Points input type is unrecognized: '
                            '"{}"'.format(type(points)))

        # make a Gen class instance to operate with
        gen = cls(pc, res_file, output_request=output_request, fout=fout,
                  dirout=dirout)

        # use serial or parallel execution control based on n_workers
        if n_workers == 1:
            logger.debug('Running serial generation for: {}'.format(pc))
            out = execute_single(gen.run, pc)
        else:
            logger.debug('Running parallel generation for: {}'.format(pc))
            out = execute_parallel(gen.run, pc, n_workers=n_workers,
                                   loggers=[__name__, 'reV.SAM'])

        # save output data to object attribute
        gen.out = out

        # flush output data (will only write to disk if fout is a str)
        gen.flush()

        # optionally return Gen object (useful for debugging and hacking)
        if return_obj:
            return gen

    @classmethod
    def run_smart(cls, tech=None, points=None, sam_files=None, res_file=None,
                  cf_profiles=True, n_workers=1, sites_per_split=100,
                  points_range=None, fout=None, dirout='./gen_out'):
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
        cf_profiles : bool
            Enables capacity factor annual profile output. Capacity factor
            means output if this is False.
        n_workers : int
            Number of local workers to run on.
        sites_per_split : int
            Number of sites to run in series on a core.
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. To be taken from the reV.config.PointsControl.split_range
            property.
        fout : str | None
            Optional .h5 output file specification.
        dirout : str | None
            Optional output directory specification. The directory will be
            created if it does not already exist.
        """

        # create the output request tuple
        output_request = ('cf_mean',)
        if cf_profiles:
            output_request += ('cf_profile',)

        if isinstance(points, (slice, str)):
            # make Project Points and Points Control instances
            pp = ProjectPoints(points, sam_files, tech, res_file=res_file)
            if points_range is None:
                pc = PointsControl(pp, sites_per_split=sites_per_split)
            else:
                pc = PointsControl.split(points_range[0], points_range[1], pp,
                                         sites_per_split=sites_per_split)
        elif isinstance(points, PointsControl):
            pc = points
        else:
            raise TypeError('Generation Points input type is unrecognized: '
                            '"{}"'.format(type(points)))

        # make a Gen class instance to operate with
        gen = cls(pc, res_file, output_request=output_request, fout=fout,
                  dirout=dirout)

        logger.debug('Running parallel generation with smart data flushing '
                     'for: {}'.format(pc))
        execute_smart_parallel(gen, pc, n_workers=n_workers,
                               loggers=[__name__, 'reV.SAM'])


if __name__ == '__main__':
    # TEST case on local machine
    import time
    t0 = time.time()
    tech = 'pv'
    points = slice(0, 100)
    sam_files = ('C:/sandbox/reV/git_reV2/tests/data/SAM/'
                 'naris_pv_1axis_inv13.json')
    res_file = 'C:/sandbox/reV/git_reV2/tests/data/nsrdb/ri_100_nsrdb_2012.h5'
    cf_profiles = True
    n_workers = 2
    sites_per_core = 1
    points_range = None
    fout = 'reV.h5'
    dirout = 'C:/sandbox/reV/test_output'
    Gen.run_smart(tech=tech,
                  points=points,
                  sam_files=sam_files,
                  res_file=res_file,
                  cf_profiles=cf_profiles,
                  n_workers=n_workers,
                  sites_per_split=sites_per_core,
                  points_range=points_range,
                  fout=fout,
                  dirout=dirout)

    print('reV generation local run complete. Total time elapsed: '
          '{0:.2f} minutes.'.format((time.time() - t0) / 60))
