"""
Generation
"""
import logging
import os
import numpy as np
from warnings import warn

from reV.SAM.SAM import PV, CSP, LandBasedWind, OffshoreWind
from reV.config.project_points import ProjectPoints, PointsControl
from reV.execution.execution import execute_parallel, execute_single
from reV.handlers.capacity_factor import CapacityFactor
from reV.handlers.resource import Resource


logger = logging.getLogger(__name__)


class Gen:
    """Base class for generation"""
    def __init__(self, points_control, res_file, output_request=('cf_mean',)):
        """Initialize a generation instance.

        Parameters
        ----------
        points_control : reV.config.PointsControl
            Project points control instance for site and SAM config spec.
        res_file : str
            Resource file with path.
        output_request : list | tuple
            Output variables requested from SAM.
        """

        self._points_control = points_control
        self._res_file = res_file
        self._output_request = output_request

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
    def meta(self):
        """Get the generation resource meta data."""
        if not hasattr(self, '_meta'):
            with Resource(self.res_file) as res:
                self._meta = res.meta[res.meta.index.isin(
                    self.project_points.sites)]
                self._meta['gid'] = self.project_points.sites
                self._meta['reV_tech'] = self.project_points.tech
                self._meta['sam_config'] = [self.project_points[site][0] for
                                            site in self.project_points.sites]
        return self._meta

    @meta.setter
    def meta(self, key_val):
        """Add a (key, value) pair as a column in the meta dataframe."""
        if not hasattr(self, '_meta'):
            self._meta = self.meta
        if len(key_val[1]) != len(self._meta):
            raise ValueError('Data can only be added to meta if it is the '
                             'same length. Meta length: {}, trying to add: '
                             '\n{}'.format(len(self._meta), key_val[1]))
        self._meta[key_val[0]] = key_val[1]

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
        logger.debug('Flushing generation annual means to disk to file: {}'
                     .format(fout))
        cf_means = self.unpack_cf_means(gen_out)
        self.meta = ('cf_means', cf_means)
        CapacityFactor.write_means(fout, self.meta, cf_means, self.sam_configs,
                                   **{'mode': mode})

    def profiles_to_disk(self, gen_out, fout='gen_out.h5', mode='w'):
        """Save capacity factor profiles to disk."""
        logger.debug('Flushing generation profiles to disk to file: {}'
                     .format(fout))
        cf_profiles = self.unpack_cf_profiles(gen_out)
        self.meta = ('cf_means', self.unpack_cf_means(gen_out))
        CapacityFactor.write_profiles(fout, self.meta, self.time_index,
                                      cf_profiles, self.sam_configs,
                                      **{'mode': mode})

    def flush(self, fout='gen_out.h5', dirout='./gen_out', mode='w'):
        """Flush generation data in self.out attribute to disk in .h5 format.

        Parameters
        ----------
        fout : str | None
            .h5 output file specification. Data will not be written to disk if
            this is None.
        dirout : str
            Output directory specification. The directory will be
            created if it does not already exist.
        mode : str
            .h5 file write mode (e.g. 'w', 'a').
        """

        # handle output file request
        if isinstance(fout, str):
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
            # write means or profiles to disk
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
        gen = cls(pc, res_file, output_request=output_request)

        # use serial or parallel execution control based on n_workers
        if n_workers == 1:
            logger.debug('Running serial generation for: {}'.format(pc))
            out = execute_single(gen.run, pc)
        else:
            logger.debug('Running parallel generation for: {}'.format(pc))
            out = execute_parallel(gen.run, pc, n_workers=n_workers,
                                   loggers=[__name__, 'reV.SAM'])
            out = gen.unpack_futures(out)

        # save output data to object attribute
        gen.out = out

        # flush output data (will only write to disk if fout is a str)
        gen.flush(fout=fout, dirout=dirout)

        # optionally return Gen object (useful for debugging and hacking)
        if return_obj:
            return gen
