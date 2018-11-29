"""
Generation
"""
from dask.distributed import Client, LocalCluster
from dask_jobqueue import PBSCluster
import logging
import os
import time
import timeit
import functools

from reV.SAM.SAM import PV, CSP, LandBasedWind, OffshoreWind
from reV.config.config import Config
from reV import __testdata__ as TESTDATA
from reV.rev_logger import init_logger, REV_LOGGERS
from reV.handlers.resource import NSRDB, WTK


logger = logging.getLogger(__name__)


def timer(fun):
    """Function timer decorator."""
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        fout = fun(*args, **kwargs)
        elapsed = timeit.default_timer() - start_time
        logger.debug('{0} took {1:.3f} seconds to execute.'
                     .format(fun, elapsed))
        return fout
    return wrapper


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
    def execution_control(self):
        """Get config project points"""
        return self._config.execution_control

    @property
    def output_request(self):
        """Get the list of output variables requested from generation."""
        if self._output_request is None:
            self._output_request = ['cf_mean']
            if self.config.sam_gen.write_profiles:
                self._output_request += ['cf_profile']
        return self._output_request

    @property
    def project_points(self):
        """Get config project points"""
        return self._config.execution_control.project_points

    @property
    def res_files(self):
        """Get the source resource filenames from config."""
        return self._config.res_files

    @timer
    def execute_hpc(self, execution_control=None, res_files=None):
        """Execute a multi-node multi-core job on an HPC cluster."""
        if execution_control is None:
            execution_control = self.execution_control
        if res_files is None:
            res_files = self.res_files

        for res_file in res_files:

            # start a PBS cluster and request nodes using scale
            cluster = PBSCluster(queue=execution_control.hpc_queue,
                                 project=execution_control.hpc_alloc,
                                 name=self.config.name,
                                 cores=execution_control.ppn,
                                 memory=execution_control.hpc_node_mem,
                                 walltime=execution_control.hpc_walltime,
                                 )

            cluster.scale(execution_control.nodes)
            logger.debug('Scaling PBS cluster to {} workers.'
                         .format(execution_control.nodes))

            results = self.execute_futures(cluster, execution_control,
                                           res_file)
        return results

    @timer
    def execute_parallel(self, execution_control=None, res_files=None):
        """Execute a parallel generation compute on a single node."""
        if execution_control is None:
            execution_control = self.execution_control
        if res_files is None:
            res_files = self.res_files

        for res_file in res_files:

            cluster = LocalCluster(n_workers=execution_control.ppn)

            results = self.execute_futures(cluster, execution_control,
                                           res_file)
        return results

    @timer
    def execute_futures(self, cluster, execution_control, res_file):
        """Execute concurrent futures with an established cluster."""

        futures = []

        with Client(cluster) as client:

            # initialize loggers on workers
            client.run(REV_LOGGERS.init_logger, __name__)
            client.run(REV_LOGGERS.init_logger, 'reV.SAM')

            # iterate through split executions submitting each to worker
            for i, exec_slice in enumerate(execution_control):

                logger.debug('Kicking off serial worker #{} for sites: {}'
                             .format(i, exec_slice.project_points.sites))

                futures.append(
                    client.submit(self.execute_serial,
                                  execution_control=exec_slice,
                                  res_files=[res_file], worker=i))

            if hasattr(cluster, 'pending_jobs'):
                # HPC cluster, make sure loggers are init as they get qsub'd
                last_running = list(cluster.running_jobs.keys())
                while True:
                    pending = list(cluster.pending_jobs.keys())
                    running = list(cluster.running_jobs.keys())
                    if not pending:
                        # no more pending jobs, all loggers should be init
                        break
                    else:
                        if last_running != running:
                            # some new jobs are running, init logger
                            last_running = running
                            client.run(REV_LOGGERS.init_logger, __name__)
                            client.run(REV_LOGGERS.init_logger, 'reV.SAM')
                    time.sleep(0.5)

            results = client.gather(futures)

        return results

    @timer
    def execute_serial(self, execution_control=None, res_files=None,
                       output_request=None, worker=0):
        """Execute a serial generation compute on a single core."""
        if execution_control is None:
            execution_control = self.execution_control
        if res_files is None:
            res_files = self.res_files
        if output_request is None:
            output_request = self.output_request

        project_points = execution_control.project_points

        logger.debug('Running Gen serial on worker #{} for sites: {} '
                     .format(worker, project_points.sites))

        for res_file in res_files:

            res_iter = self.get_sam_res(res_file, project_points)

            if self.config.tech == 'pv':
                out = PV.reV_run(res_iter, project_points,
                                 output_request=output_request)

            elif self.config.tech == 'csp':
                out = CSP.reV_run(res_iter, project_points,
                                  output_request=output_request)

            elif self.config.tech == 'landbasedwind':
                out = LandBasedWind.reV_run(res_iter, project_points,
                                            output_request=output_request)

            elif self.config.tech == 'offshorewind':
                out = OffshoreWind.reV_run(res_iter, project_points,
                                           output_request=output_request)

        return out

    @staticmethod
    def get_sam_res(res_file, project_points):
        """Get the SAM resource iterator object (single year, single file)."""
        if 'nsrdb' in res_file:
            res_iter = NSRDB.preload_SAM(res_file, project_points)
        elif 'wtk' in res_file:
            res_iter = WTK.preload_SAM(res_file, project_points)
        return res_iter


if __name__ == '__main__':
    # temporary script based test will be merged into test.py later

    cfile = os.path.join(TESTDATA, 'config_ini/ri_rev2_test.ini')
    gen = Gen(cfile)

    logger_list = [__name__, "reV.config", "reV.SAM", "reV.handlers"]
    for log in logger_list:
        init_logger(log, log_level="DEBUG", log_file='gen.log')

    # extract for debugging
    config = gen.config
    ec = gen.config.execution_control
    pp = gen.project_points

    if gen.execution_control.option == 'serial':
        outs = gen.execute_serial()
    elif gen.execution_control.option == 'parallel':
        outs = gen.execute_parallel()
    elif gen.execution_control.option == 'hpc':
        outs = gen.execute_hpc()
    else:
        raise ValueError('Run option not recognized: {}'
                         .format(gen.execution_control.option))

    print(outs)
