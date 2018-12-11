"""
Generation
"""
from dask.distributed import Client, LocalCluster
from subprocess import Popen, PIPE
import logging
import os
import getpass
import shlex
from warnings import warn

from reV.rev_logger import REV_LOGGERS


logger = logging.getLogger(__name__)


class SubprocessManager:
    """Base class to handle subprocess execution."""

    # get username as class attribute.
    user = getpass.getuser()

    @staticmethod
    def make_path(d):
        """Make a directory if it doesn't exist."""
        if not os.path.exists(d):
            os.mkdir(d)

    @staticmethod
    def make_sh(fname, script):
        """Make a shell script to execute a subprocess."""
        with open(fname, 'w+') as f:
            logger.debug('The shell script "{n}" contains the following:\n'
                         '~~~~~~~~~~ {n} ~~~~~~~~~~\n'
                         '{s}\n'
                         '~~~~~~~~~~ {n} ~~~~~~~~~~'
                         .format(n=fname, s=script))
            f.write(script)

    @staticmethod
    def rm(fname):
        """Remove a file."""
        os.remove(fname)

    @staticmethod
    def submit(cmd):
        """Open a subprocess and submit a command. Capture out/error."""
        cmd = shlex.split(cmd)
        logger.debug('Submitting the following cmd as a subprocess:\n\t{}'
                     .format(cmd))

        # use subprocess to submit command and get piped o/e
        process = Popen(cmd, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        stderr = stderr.decode('ascii').rstrip()
        stdout = stdout.decode('ascii').rstrip()

        if stderr:
            raise Exception('Error occurred submitting job:\n{}'
                            .format(process.stderr))

        return stdout, stderr

    @staticmethod
    def s(s):
        """Format an object as string for python cli command entry."""
        if isinstance(s, (list, tuple, dict)):
            return '"{}"'.format(s)
        elif not isinstance(s, (int, float, type(None))):
            return "'{}'".format(s)
        else:
            return '{}'.format(s)

    @staticmethod
    def node_loggers(module_list, level='DEBUG', log_dir='logs',
                     file='reV.log'):
        """Get a string python command to init loggers in a shell submission"""
        SubprocessManager.make_path(log_dir)
        log_args = ''
        for module in module_list:
            log_args += ("init_logger({n}, log_level={level}, log_file={f});"
                         .format(n=PBS.s(module),
                                 level=PBS.s(level),
                                 f=PBS.s(os.path.join(log_dir, file))))
        return log_args


class PBS(SubprocessManager):
    """Subclass for PBS subprocess jobs."""

    def __init__(self, cmd, alloc='rev', queue='short', name='reV',
                 stdout_path='./stdout'):
        """Initialize and submit a PBS job."""
        self.make_path(stdout_path)
        self.id, self.err = self.qsub(cmd,
                                      alloc=alloc,
                                      queue=queue,
                                      name=name,
                                      stdout_path=stdout_path)

    @staticmethod
    def check_status(job, var='id'):
        """Check the status of this PBS job using qstat.

        Parameters
        ----------
        job : str
            Job name or ID number.
        var : str
            Identity/type of job identification input arg ('id' or 'name').

        Returns
        -------
        out : str or NoneType
            Qstat job status character or None if not found.
        """

        # column location of various job identifiers
        col_loc = {'id': 0, 'name': 3}
        qstat_rows = PBS.qstat()
        if qstat_rows is None:
            return None
        else:
            # reverse the list so most recent jobs are first
            qstat_rows = reversed(qstat_rows)

        # update job status from qstat list
        for row in qstat_rows:
            row = row.split()
            if len(row) > 10:
                if row[col_loc[var]].strip() == job.strip():
                    return row[-2]
        return None

    @staticmethod
    def qstat():
        """Run the PBS qstat command and return the stdout split to rows."""
        cmd = 'qstat -u {user}'.format(user=PBS.user)
        stdout, _ = PBS.submit(cmd)
        if not stdout:
            # No jobs are currently running.
            return None
        else:
            qstat_rows = stdout.split('\n')
            return qstat_rows

    @staticmethod
    def qsub(cmd, alloc='rev', queue='short', name='reV',
             stdout_path='./stdout', keep_sh=False):
        """Submit a PBS job via qsub command and PBS shell script."""

        status = PBS.check_status(name, var='name')

        if status == 'Q' or status == 'R':
            warn('Not submitting job "{}" because it is already in '
                 'qstat with status: "{}"'.format(name, status))
            out = None
            err = 'already_running'
        else:
            fname = '{}.sh'.format(name)
            script = ('#!/bin/bash\n'
                      '#PBS -N {n} # job name\n'
                      '#PBS -A {a} # allocation account\n'
                      '#PBS -q {q} # queue (debug, short, batch, or long)\n'
                      '#PBS -o {p}/{n}_$PBS_JOBID.o\n'
                      '#PBS -e {p}/{n}_$PBS_JOBID.e\n'
                      '{cmd}'
                      .format(n=name, a=alloc, q=queue, p=stdout_path,
                              cmd=cmd))

            # write the shell script file and submit as qsub job
            PBS.make_sh(fname, script)
            out, err = PBS.submit('qsub {script}'.format(script=fname))

            if not err:
                logger.debug('PBS job "{}" with id #{} submitted successfully'
                             .format(name, out))
                if not keep_sh:
                    PBS.rm(fname)

        return out, err


class SLURM(SubprocessManager):
    """Subclass for SLURM subprocess jobs."""
    def __init__(self, py=None, alloc=None, name='reV',
                 feature=None, stdout_path='./stdout'):
        """Initialize a SLURM job."""
        pass


def execute_parallel(fun, execution_iter, loggers=[], n_workers=None,
                     **kwargs):
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
    if n_workers:
        cluster = LocalCluster(n_workers=n_workers)
    else:
        cluster = None

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

    logger.debug('Running single serial execution on worker #{} for: {}'
                 .format(worker, input_obj))

    out = fun(input_obj, **kwargs)

    return out
