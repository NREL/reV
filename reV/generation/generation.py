"""
Generation
"""
import logging

from reV.SAM.SAM import PV, CSP, LandBasedWind, OffshoreWind
from reV.config.config import ProjectPoints, PointsControl
from reV.execution.execution import execute_parallel, execute_single


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
    def organize_futures(futures):
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
               points_range=None):
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
            out = Gen.organize_futures(out)
        return out
