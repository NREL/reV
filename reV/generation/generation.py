"""
Generation
"""
from dask.distributed import Client, LocalCluster
from dask_jobqueue import PBSCluster
import logging
import time
import timeit
import functools

from reV.SAM.SAM import PV, CSP, LandBasedWind, OffshoreWind
from reV.config.config import Config
from reV.rev_logger import REV_LOGGERS


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
        """Initialize a generation instance.

        Parameters
        ----------
        config_file : str
            reV 2.0 user input configuration file (with full file path).
        """

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
        if not hasattr(self, '_output_request'):
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
        """Execute a multi-node multi-core job on an HPC cluster.

        Parameters
        ----------
        execution_control : reV.config.ExecutionControl
            reV 2.0 ExecutionControl instance.
        res_files : list
            Resource file list (with full paths) to analyze.

        Returns
        -------
        results : list
            List of generation futures results.
        """

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
        """Execute a parallel generation compute on a single node.

        Parameters
        ----------
        execution_control : reV.config.ExecutionControl
            reV 2.0 ExecutionControl instance.
        res_files : list
            Resource file list (with full paths) to analyze.

        Returns
        -------
        results : list
            List of generation futures results.
        """

        if execution_control is None:
            execution_control = self.execution_control
        if res_files is None:
            res_files = self.res_files

        for res_file in res_files:

            # start a local cluster on a personal comp or HPC single node
            cluster = LocalCluster(n_workers=execution_control.ppn)

            results = self.execute_futures(cluster, execution_control,
                                           res_file)
        return results

    @timer
    def execute_futures(self, cluster, execution_control, res_file):
        """Execute concurrent futures with an established cluster.

        Parameters
        ----------
        cluster : LocalCluster | PBSCluster
            Dask cluster object generated from either the LocalCluster or
            PBSCluster classes.
        execution_control : reV.config.ExecutionControl
            reV 2.0 ExecutionControl instance.
        res_file : str
            Single resource file (with full paths) to analyze.

        Returns
        -------
        results : list
            List of generation futures results.
        """

        futures = []

        # initialize a client based on the input cluster.
        with Client(cluster) as client:

            # initialize loggers on workers
            client.run(REV_LOGGERS.init_logger, __name__)
            client.run(REV_LOGGERS.init_logger, 'reV.SAM')

            # iterate through split executions, submitting each to worker
            for i, exec_slice in enumerate(execution_control):

                logger.debug('Kicking off serial worker #{} for sites: {}'
                             .format(i, exec_slice.project_points.sites))

                # submit executions and append to futures list
                futures.append(
                    client.submit(self.execute_serial,
                                  execution_control=exec_slice,
                                  res_files=[res_file], worker=i))

            if hasattr(cluster, 'pending_jobs'):
                # HPC cluster, make sure loggers are init as they get qsub'd
                pending = list(cluster.pending_jobs.keys())
                running = list(cluster.running_jobs.keys())
                last_running = list(cluster.running_jobs.keys())
                while True:
                    if not pending:
                        # no more pending jobs, all loggers should be init'd
                        break
                    else:
                        if last_running != running:
                            # new jobs are running, init logger
                            last_running = running
                            client.run(REV_LOGGERS.init_logger, __name__)
                            client.run(REV_LOGGERS.init_logger, 'reV.SAM')
                    time.sleep(0.5)

            # gather results
            results = client.gather(futures)

        return results

    @timer
    def execute_serial(self, project_points=None, res_files=None,
                       output_request=None, worker=0):
        """Execute a serial generation compute on a single core.

        Parameters
        ----------
        project_points : reV.config.ProjectPoints
            reV 2.0 ProjectPoints instance.
        res_files : list
            Resource file list (with full paths) to analyze.

        Returns
        -------
        out : dict
            Nested dictionaries where the top level key is the site index,
            the second level key is the variable name, second level value is
            the output variable value.
        """

        if project_points is None:
            project_points = self.project_points
        if res_files is None:
            res_files = self.res_files
        if output_request is None:
            output_request = self.output_request

        logger.debug('Running Gen serial on worker #{} for sites: {} '
                     .format(worker, project_points.sites))

        for res_file in res_files:

            if self.config.tech == 'pv':
                out = PV.reV_run(res_file, project_points,
                                 output_request=output_request)

            elif self.config.tech == 'csp':
                out = CSP.reV_run(res_file, project_points,
                                  output_request=output_request)

            elif self.config.tech == 'landbasedwind':
                out = LandBasedWind.reV_run(res_file, project_points,
                                            output_request=output_request)

            elif self.config.tech == 'offshorewind':
                out = OffshoreWind.reV_run(res_file, project_points,
                                           output_request=output_request)

        return out
