"""
Generation
"""
import logging
import numpy as np
from warnings import warn

from reV.SAM.SAM import PV, CSP, LandBasedWind, OffshoreWind
from reV.config.config import ProjectPoints, PointsControl
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
                self._meta['rid'] = self.project_points.sites
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
        cf_means = self.unpack_cf_means(gen_out)
        self.meta = ('cf_means', cf_means)
        CapacityFactor.write_means(fout, self.meta, cf_means, self.sam_configs,
                                   **{'mode': mode})

    def profiles_to_disk(self, gen_out, fout='gen_out.h5', mode='w'):
        """Save capacity factor profiles to disk."""
        cf_profiles = self.unpack_cf_profiles(gen_out)
        self.meta = ('cf_means', self.unpack_cf_means(gen_out))
        CapacityFactor.write_profiles(fout, self.meta, self.time_index,
                                      cf_profiles, self.sam_configs,
                                      **{'mode': mode})

    @staticmethod
    def run(points_control, res_file=None, output_request=None, tech=None):
        """Run a generation analysis."""
        sam_funs = {'pv': PV.reV_run,
                    'csp': CSP.reV_run,
                    'landbasedwind': LandBasedWind.reV_run,
                    'offshorewind': OffshoreWind.reV_run,
                    }

        out = sam_funs[tech](points_control, res_file,
                             output_request=output_request)

        return out

    @classmethod
    def direct(cls, tech=None, points=None, sam_files=None, res_file=None,
               cf_profiles=True, n_workers=1, sites_per_split=100,
               points_range=None, fout=None, return_obj=True):
        """Execute a generation run directly from source files without config.

        Parameters
        ----------
        tech : str
            Technology to analyze (pv, csp, landbasedwind, offshorewind).
        points : slice | str
            Slice specifying project points or string pointing to a project
            points csv.
        sam_files : dict | str | list
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s), top level value is the SAM path. Can also be a single
            config file str. If it's a list, it is mapped to the sorted list
            of unique configs requested by points csv.
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
            analyze. To be taken from the
            reV.config.PointsControl.split_range property.
        fout : str | None
            Optional .h5 output file specification.
        return_obj : bool
            Return the Gen object instance.

        Returns
        -------
        out : dict
            Nested dictionaries where the top level key is the site index,
            the second level key is the variable name, second level value is
            the output variable value.
        """

        # create the output request tuple
        output_request = ('cf_mean',)
        if cf_profiles:
            output_request += ('cf_profile',)

        # make Project Points and Points Control instances
        pp = ProjectPoints(points, sam_files, tech, res_file=res_file)
        if points_range is None:
            pc = PointsControl(pp, sites_per_split=sites_per_split)
        else:
            pc = PointsControl.split(points_range[0], points_range[1], pp,
                                     sites_per_split=sites_per_split)

        # make a Gen class instance to operate with
        gen = cls(pc, res_file, output_request=output_request)

        # use serial or parallel execution control based on n_workers
        if n_workers == 1:
            logger.debug('Running serial generation for: {}'.format(pc))
            out = execute_single(gen.run, pc, res_file=res_file, tech=tech,
                                 output_request=output_request)
        else:
            logger.debug('Running parallel generation for: {}'.format(pc))
            out = execute_parallel(gen.run, pc, n_workers=n_workers,
                                   loggers=[__name__, 'reV.SAM'],
                                   res_file=res_file, tech=tech,
                                   output_request=output_request)
            out = gen.unpack_futures(out)

        # save output data to object attribute
        gen.out = out

        # handle output file request
        if isinstance(fout, str):
            if not fout.endswith('.h5'):
                fout += '.h5'
                warn('Generation output file request must be .h5, '
                     'set to: "{}"'.format(fout))
            if 'profile' in str(output_request):
                gen.profiles_to_disk(gen.out, fout=fout)
            else:
                gen.means_to_disk(gen.out, fout=fout)

        # optionally return Gen object (useful for debugging)
        if return_obj:
            return gen


if __name__ == '__main__':
    import h5py
    from reV import __testdatadir__
    import pandas as pd

    name = 'reV'
    tech = 'pv'
    points = slice(0, 10)
    sam_files = __testdatadir__ + '/SAM/naris_pv_1axis_inv13.json'
    res_file = __testdatadir__ + '/nsrdb/ri_100_nsrdb_2012.h5'
    output_request = ('cf_mean', 'cf_profile')
    sites_per_core = 100
    n_workers = 1
    verbose = True

    gen = Gen.direct(tech=tech,
                     points=points,
                     sam_files=sam_files,
                     res_file=res_file,
                     n_workers=n_workers,
                     sites_per_split=sites_per_core,
                     fout='test.h5')

    with h5py.File('test.h5', 'r') as f:
        var = 'cf_profiles'
        data = f[var][...]
        ti = pd.DataFrame(f['time_index'][...])
        meta = pd.DataFrame(f['meta'][...])
        print(list(f.keys()))
        print(data)
        print(list(f[var].attrs))
        print(f[var].attrs['scale_factor'])
        print(list(f['meta'].attrs))
        print(f['meta'].attrs['0'])
        print(meta)
