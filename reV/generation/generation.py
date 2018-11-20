"""
Generation
"""
from dask.distributed import LocalCluster, Client
import logging
import os
import reV.SAM.SAM as SAM
from reV.config.config import Config
from reV import __dir__ as REVDIR
from reV.rev_logger import init_logger, REV_LOGGERS
from reV.handlers import resource


logger = logging.getLogger(__name__)


class Gen:
    """Base class for generation"""
    def __init__(self, config_file):
        """Initialize a generation instance."""

        self._output_request = None
        self._config = Config(config_file)

    @property
    def config(self):
        """Get the config object."""
        return self._config

    @property
    def output_request(self):
        """Get the list of output variables requested from generation."""
        if self._output_request is None:
            self._output_request = ['cf_mean']
            if self.config.SAM_gen.write_profiles:
                self._output_request += ['cf_profile']
        return self._output_request

    @property
    def project_points(self):
        """Get config project points"""
        return self._config.project_points

    @property
    def execution_control(self):
        """Get config project points"""
        return self._config.execution_control

    def execute_parallel(self, execution_control=None, res_files=None):
        """Execute a parallel generation compute."""
        if execution_control is None:
            execution_control = self.execution_control
        if res_files is None:
            res_files = self.config.res_files

        for res_file in res_files:

            # set the current level. split level will be one level down.
            execution_control.level = 'node'

            cluster = LocalCluster(n_workers=execution_control.N)
            with Client(cluster) as client:
                client.run(REV_LOGGERS.init_logger, __name__)
                client.run(REV_LOGGERS.init_logger, 'reV.SAM')
                futures = []

                for exec_slice in execution_control:
                    print(exec_slice.project_points.sites)
                    futures.append(
                        client.submit(self.execute_serial,
                                      execution_control=exec_slice,
                                      res_files=[res_file]))

                results = client.gather(futures)

        return results

    @staticmethod
    def test_dd(execution_control=None, res_files=None):
        """Simple test dask distributed."""
        project_points = execution_control.project_points

        for res_file in res_files:

            if 'nsrdb' in res_file:
                res_iter = SAM.ResourceManager(resource.NSRDB(res_file),
                                               project_points)
            elif 'wtk' in res_file:
                res_iter = SAM.ResourceManager(resource.WTK(res_file),
                                               project_points)

        return res_iter

    def execute_serial(self, execution_control=None, res_files=None,
                       output_request=None):
        """Exec generation sim for a single file and instance of project points
        """
        if execution_control is None:
            execution_control = self.execution_control
        if res_files is None:
            res_files = self.config.res_files
        if output_request is None:
            output_request = self.output_request

        project_points = execution_control.project_points

        logger.info('Running Gen serial for sites: {}'
                    .format(project_points.sites))

        for res_file in res_files:

            res_iter = self.get_sam_res(res_file, project_points)

            if self.config.tech == 'pv':
                out = SAM.PV.reV_run(res_iter, project_points,
                                     output_request=output_request)

            elif self.config.tech == 'csp':
                out = SAM.CSP.reV_run(res_iter, project_points,
                                      output_request=output_request)

            elif self.config.tech == 'landbasedwind':
                out = SAM.LandBasedWind.reV_run(
                    res_iter, project_points,
                    output_request=output_request)

            elif self.config.tech == 'offshorewind':
                out = SAM.OffshoreWind.reV_run(
                    res_iter, project_points,
                    output_request=output_request)

        return out

    @staticmethod
    def get_sam_res(res_file, project_points):
        """Get the SAM resource iterator object."""
        if 'nsrdb' in res_file:
            res_iter = SAM.ResourceManager(resource.NSRDB(res_file),
                                           project_points)
        elif 'wtk' in res_file:
            res_iter = SAM.ResourceManager(resource.WTK(res_file),
                                           project_points)
        return res_iter


if __name__ == '__main__':
    # temporary script based test will be merged into test.py later

    cfile = os.path.join(REVDIR, 'config/ini/ri_subset_pv_gentest.ini')
    gen = Gen(cfile)

    logger_list = [__name__, "reV.config", "reV.SAM", "reV.handlers"]
    loggers = {}
    handlers = {}
    for log in logger_list:
        loggers[log], handlers[log] = init_logger(log, log_level="INFO",
                                                  log_file='gen.log')

    config = gen.config
#    outs = gen.execute_serial()
    outs = gen.execute_parallel()
    config = gen.config
    exec_control = gen.config.execution_control
    pp = gen.project_points
    print(outs)
