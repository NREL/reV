"""
Generation
"""
from dask.distributed import Client, LocalCluster
import logging

from reV.rev_logger import REV_LOGGERS


logger = logging.getLogger(__name__)


def execute_sub():
    """Execute a subprocess run on multiple nodes."""
    return {}


def execute_parallel(fun, execution_iter, loggers=[], n_workers=2, **kwargs):
    """Execute a parallel generation compute on a single node.

    Parameters
    ----------
    fun : function
        Python function object that will be submitted to futures. See
        downstream execution methods for arg passing structure.
    execution_iter : iter
        Python iterator that controls the futures submitted to dask.
    loggers : list
        List of logger names to initialize on the workers.
    n_workers : int
        Number of workers to scale the cluster to.
    **kwargs : dict
        Key word arguments passed to the fun.

    Returns
    -------
    results : list
        List of futures results.
    """

    # start a local cluster on a personal comp or HPC single node
    cluster = LocalCluster(n_workers=n_workers)

    results = execute_futures(fun, execution_iter, cluster, loggers=loggers,
                              **kwargs)

    return results


def execute_futures(fun, execution_iter, cluster, loggers=[], **kwargs):
    """Execute concurrent futures with an established cluster.

    Parameters
    ----------
    fun : function
        Python function object that will be submitted to futures. See
        downstream execution methods for arg passing structure.
    execution_iter : iter
        Python iterator that controls the futures submitted to dask.
    cluster : dask.distributed.LocalCluster
        Dask cluster object created from the LocalCluster() class.
    loggers : list
        List of logger names to initialize on the workers.
    **kwargs : dict
        Key word arguments passed to the fun.

    Returns
    -------
    results : list
        List of futures results.
    """

    futures = []

    # initialize a client based on the input cluster.
    with Client(cluster) as client:

        # initialize loggers on workers
        for logger_name in loggers:
            client.run(REV_LOGGERS.init_logger, logger_name)

        # iterate through split executions, submitting each to worker
        for i, exec_slice in enumerate(execution_iter):

            logger.debug('Kicking off serial worker #{} for: {}'
                         .format(i, exec_slice))

            # submit executions and append to futures list
            futures.append(client.submit(execute_single, fun, exec_slice,
                                         worker=i, **kwargs))

        # gather results
        results = client.gather(futures)

    return results


def execute_single(fun, input_obj, worker=0, **kwargs):
    """Execute a serial compute on a single core.

    Parameters
    ----------
    fun : function
        Function to execute.
    input_obj : object
        Object passed as first argument to fun. Typically a project control
        object that can be the result of iteration in the parallel execution
        framework.
    worker : int
        Worker number for debugging purposes.
    **kwargs : dict
        Key word arguments passed to fun.
    """

    logger.debug('Running single on worker #{} for: {}'
                 .format(worker, input_obj))

    out = fun(input_obj, **kwargs)

    return out
