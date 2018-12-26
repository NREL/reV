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
    def res_file(self):
        """Get the resource filename and path."""
        return self._res_file

    @staticmethod
    def get_meta_vars(project_points):
        """Unpack meta and other auxiliary data from project points."""
        res = Resource(project_points.res_file)
        meta = res.meta[res.meta.index.isin(project_points.sites)]
        time_index = res.time_index
        sam_configs = project_points.sam_configs
        return meta, time_index, sam_configs

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
        for result in futures:
            for key, val in result.items():
                out[key] = val
        return out

    @staticmethod
    def unpack_cf_means(gen_out):
        """Unpack a numpy means 1darray from a gen output dictionary."""
        out = np.array([val['cf_mean'] for val in gen_out.values()])
        return out

    @staticmethod
    def unpack_cf_profiles(gen_out):
        """Unpack a numpy profiles 2darray from a gen output dictionary."""
        out = np.array([val['cf_profile']for val in gen_out.values()])
        return out.transpose()

    @staticmethod
    def means_to_disk(gen_out, meta, sam_configs, fout='gen_out.h5', mode='w'):
        """Save capacity factor means to disk."""
        cf_means = Gen.unpack_cf_means(gen_out)
        CapacityFactor.write_means(fout, meta, cf_means, sam_configs,
                                   **{'mode': mode})

    @staticmethod
    def profiles_to_disk(gen_out, meta, time_index, sam_configs,
                         fout='gen_out.h5', mode='w'):
        """Save capacity factor profiles to disk."""
        cf_profiles = Gen.unpack_cf_profiles(gen_out)
        CapacityFactor.write_profiles(fout, meta, time_index, cf_profiles,
                                      sam_configs, **{'mode': mode})

    @staticmethod
    def run(points_control, res_file=None, output_request=None, tech=None):
        """Run a generation compute."""

        if tech == 'pv':
            out = PV.reV_run(points_control, res_file,
                             output_request=output_request)
        elif tech == 'csp':
            out = CSP.reV_run(points_control, res_file,
                              output_request=output_request)
        elif tech == 'landbasedwind':
            out = LandBasedWind.reV_run(points_control, res_file,
                                        output_request=output_request)
        elif tech == 'offshorewind':
            out = OffshoreWind.reV_run(points_control, res_file,
                                       output_request=output_request)
        else:
            raise ValueError('Technology not recognized: {}'.format(tech))

        return out

    @staticmethod
    def direct(tech=None, points=None, sam_files=None, res_file=None,
               output_request=('cf_mean',), n_workers=1, sites_per_split=100,
               points_range=None, fout=None):
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
        output_request : list | tuple
            Output variables requested from SAM.
        n_workers : int
            Number of local workers to run on.
        sites_per_split : int
            Number of sites to run in series on a core.

        Returns
        -------
        out : dict
            Nested dictionaries where the top level key is the site index,
            the second level key is the variable name, second level value is
            the output variable value.
        """

        pp = ProjectPoints(points, sam_files, tech, res_file=res_file)

        if points_range is None:
            pc = PointsControl(pp, sites_per_split=sites_per_split)
        else:
            pc = PointsControl.split(points_range[0], points_range[1], pp,
                                     sites_per_split=sites_per_split)

        if n_workers == 1:
            logger.debug('Running serial generation for: {}'.format(pc))
            out = execute_single(Gen.run, pc, res_file=res_file, tech=tech,
                                 output_request=output_request)
        else:
            logger.debug('Running parallel generation for: {}'.format(pc))
            out = execute_parallel(Gen.run, pc, n_workers=n_workers,
                                   loggers=[__name__, 'reV.SAM'],
                                   res_file=res_file, tech=tech,
                                   output_request=output_request)
            out = Gen.unpack_futures(out)

        if isinstance(fout, str):
            if not fout.endswith('.h5'):
                fout += '.h5'
                warn('Generation output file request must be .h5, '
                     'set to: {}'.format(fout))
            meta, ti, sam_configs = Gen.get_meta_vars(pc.project_points)
            if 'profile' in str(output_request):
                Gen.profiles_to_disk(out, meta, ti, sam_configs, fout=fout)
            else:
                Gen.means_to_disk(out, meta, sam_configs, fout=fout)

        return out


if __name__ == '__main__':
    import h5py
    from reV import __testdatadir__
    from reV.rev_logger import init_logger
    import os

    name = 'reV'
    tech = 'pv'
    points = slice(0, 10)
    sam_files = __testdatadir__ + '/SAM/naris_pv_1axis_inv13.json'
    res_file = __testdatadir__ + '/nsrdb/ri_100_nsrdb_2012.h5'
    output_request = ('cf_mean', 'cf_profile')
    sites_per_core = 100
    n_workers = 1
    verbose = True

    if verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    log_modules = [__name__, 'reV.SAM', 'reV.config', 'reV.generation',
                   'reV.execution', 'reV.handlers']
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    for module in log_modules:
        init_logger(module, log_level=log_level,
                    log_file=os.path.join(log_dir, '{}.log'.format(name)))

    out = Gen.direct(tech=tech,
                     points=points,
                     sam_files=sam_files,
                     res_file=res_file,
                     output_request=output_request,
                     n_workers=n_workers,
                     sites_per_split=sites_per_core,
                     fout='test')

    with h5py.File('test.h5', 'r') as f:
        print(list(f.keys()))
        print(f['cf_means'][...])
        print(list(f['cf_means'].attrs))
        print(f['cf_means'].attrs['scale_factor'])
        print(f['meta'].attrs['0'])
        print(f['meta'][...])
