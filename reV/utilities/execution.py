"""
Generation
"""
from dask.distributed import Client, LocalCluster, wait
from subprocess import Popen, PIPE
import logging
import gc
from math import floor
import os
import psutil
import getpass
import shlex
from warnings import warn

from reV.utilities.loggers import REV_LOGGERS
from reV.utilities.exceptions import ExecutionError


logger = logging.getLogger(__name__)


def log_mem():
    """Print memory status to debug logger."""
    mem = psutil.virtual_memory()
    logger.debug('{0:.3f} GB used of {1:.3f} GB total ({2:.1f}% used) '
                 '({3:.3f} GB free) ({4:.3f} GB available).'
                 ''.format(mem.used / 1e9,
                           mem.total / 1e9,
                           100 * mem.used / mem.total,
                           mem.free / 1e9,
                           mem.available / 1e9))


class SubprocessManager:
    """Base class to handle subprocess execution."""

    # get username as class attribute.
    USER = getpass.getuser()

    @staticmethod
    def make_path(d):
        """Make a directory tree if it doesn't exist.

        Parameters
        ----------
        d : str
            Directory tree to check and potentially create.
        """
        if not os.path.exists(d):
            os.makedirs(d)

    @staticmethod
    def make_sh(fname, script):
        """Make a shell script (.sh file) to execute a subprocess.

        Parameters
        ----------
        fname : str
            Name of the .sh file to create.
        script : str
            Contents to be written into the .sh file.
        """
        logger.debug('The shell script "{n}" contains the following:\n'
                     '~~~~~~~~~~ {n} ~~~~~~~~~~\n'
                     '{s}\n'
                     '~~~~~~~~~~ {n} ~~~~~~~~~~'
                     .format(n=fname, s=script))
        with open(fname, 'w+') as f:
            f.write(script)

    @staticmethod
    def rm(fname):
        """Remove a file.

        Parameters
        ----------
        fname : str
            Filename (with path) to remove.
        """
        os.remove(fname)

    @staticmethod
    def submit(cmd):
        """Open a subprocess and submit a command.

        Parameters
        ----------
        cmd : str
            Command to be submitted using python subprocess.

        Returns
        -------
        stdout : str
            Subprocess standard output. This is decoded from the subprocess
            stdout with rstrip.
        stderr : str
            Subprocess standard error. This is decoded from the subprocess
            stderr with rstrip. After decoding/rstrip, this will be empty if
            the subprocess doesn't return an error.
        """

        cmd = shlex.split(cmd)

        # use subprocess to submit command and get piped o/e
        process = Popen(cmd, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        stderr = stderr.decode('ascii').rstrip()
        stdout = stdout.decode('ascii').rstrip()

        if stderr:
            raise Exception('Error occurred submitting job:\n{}'
                            .format(stderr))

        return stdout, stderr

    @staticmethod
    def s(s):
        """Format input as str w/ appropriate quote types for python cli entry.

        Examples
        --------
            list, tuple -> "['one', 'two']"
            dict -> "{'key': 'val'}"
            int, float, None -> '0'
            str, other -> 'string'
        """

        if isinstance(s, (list, tuple, dict)):
            return '"{}"'.format(s)
        elif not isinstance(s, (int, float, type(None))):
            return "'{}'".format(s)
        else:
            return '{}'.format(s)

    @staticmethod
    def walltime(hours):
        """Get the SLURM walltime string in format "HH:MM:SS"

        Parameters
        ----------
        hours : float | int
            Requested number of job hours.

        Returns
        -------
        walltime : str
            SLURM walltime request in format "HH:MM:SS"
        """

        m_str = '{0:02d}'.format(round(60 * (hours % 1)))
        h_str = '{0:02d}'.format(floor(hours))
        return '{}:{}:00'.format(h_str, m_str)


class PBS(SubprocessManager):
    """Subclass for PBS subprocess jobs."""

    def __init__(self, cmd, alloc, queue, name='reV',
                 feature=None, stdout_path='./stdout'):
        """Initialize and submit a PBS job.

        Parameters
        ----------
        cmd : str
            Command to be submitted in PBS shell script. Example:
                'python -m reV.generation.cli_gen'
        alloc : str
            HPC allocation account. Example: 'rev'.
        queue : str
            HPC queue to submit job to. Example: 'short', 'batch-h', etc...
        name : str
            PBS job name.
        feature : str | None
            PBS feature request (-l=feature). Example: '24core', '64GB', etc...
        stdout_path : str
            Path to print .stdout and .stderr files.
        """

        self.make_path(stdout_path)
        self.id, self.err = self.qsub(cmd,
                                      alloc=alloc,
                                      queue=queue,
                                      name=name,
                                      feature=feature,
                                      stdout_path=stdout_path)

    def check_status(self, job, var='id'):
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
            Common status codes: Q, R, C (queued, running, complete).
        """

        # column location of various job identifiers
        col_loc = {'id': 0, 'name': 3}
        qstat_rows = self.qstat()
        if qstat_rows is None:
            return None
        else:
            # reverse the list so most recent jobs are first
            qstat_rows = reversed(qstat_rows)

        # update job status from qstat list
        for row in qstat_rows:
            row = row.split()
            # make sure the row is long enough to be a job status listing
            if len(row) > 10:
                if row[col_loc[var]].strip() == job.strip():
                    # Job status is located at the -2 index
                    status = row[-2]
                    logger.debug('Job with {} "{}" has status: "{}"'
                                 .format(var, job, status))
                    return status
        return None

    def qstat(self):
        """Run the PBS qstat command and return the stdout split to rows.

        Returns
        -------
        qstat_rows : list | None
            List of strings where each string is a row in the qstat printout.
            Returns None if qstat is empty.
        """

        cmd = 'qstat -u {user}'.format(user=self.USER)
        stdout, _ = self.submit(cmd)
        if not stdout:
            # No jobs are currently running.
            return None
        else:
            qstat_rows = stdout.split('\n')
            return qstat_rows

    def qsub(self, cmd, alloc, queue, name='reV', feature=None,
             stdout_path='./stdout', keep_sh=False):
        """Submit a PBS job via qsub command and PBS shell script

        Parameters
        ----------
        cmd : str
            Command to be submitted in PBS shell script. Example:
                'python -m reV.generation.cli_gen'
        alloc : str
            HPC allocation account. Example: 'rev'.
        queue : str
            HPC queue to submit job to. Example: 'short', 'batch-h', etc...
        name : str
            PBS job name.
        feature : str | None
            PBS feature request (-l=feature). Example: '24core', '64GB', etc...
        stdout_path : str
            Path to print .stdout and .stderr files.
        keep_sh : bool
            Boolean to keep the .sh files. Default is to remove these files
            after job submission.

        Returns
        -------
        out : str
            qsub standard output, this is typically the PBS job ID.
        err : str
            qsub standard error, this is typically an empty string if the job
            was submitted successfully.
        """

        status = self.check_status(name, var='name')

        if status == 'Q' or status == 'R':
            warn('Not submitting job "{}" because it is already in '
                 'qstat with status: "{}"'.format(name, status))
            out = None
            err = 'already_running'
        else:
            feature_str = '#PBS -l feature={}\n'.format(feature)
            fname = '{}.sh'.format(name)
            script = ('#!/bin/bash\n'
                      '#PBS -N {n} # job name\n'
                      '#PBS -A {a} # allocation account\n'
                      '#PBS -q {q} # queue (debug, short, batch, or long)\n'
                      '#PBS -o {p}/{n}_$PBS_JOBID.o\n'
                      '#PBS -e {p}/{n}_$PBS_JOBID.e\n'
                      '{L}'
                      '{cmd}'
                      .format(n=name, a=alloc, q=queue, p=stdout_path,
                              L=feature_str if feature else '',
                              cmd=cmd))

            # write the shell script file and submit as qsub job
            self.make_sh(fname, script)
            out, err = self.submit('qsub {script}'.format(script=fname))

            if not err:
                logger.debug('PBS job "{}" with id #{} submitted successfully'
                             .format(name, out))
                if not keep_sh:
                    self.rm(fname)

        return out, err


class SLURM(SubprocessManager):
    """Subclass for SLURM subprocess jobs."""

    def __init__(self, cmd, alloc, memory, walltime, name='reV',
                 stdout_path='./stdout'):
        """Initialize and submit a PBS job.

        Parameters
        ----------
        cmd : str
            Command to be submitted in PBS shell script. Example:
                'python -m reV.generation.cli_gen'
        alloc : str
            HPC project (allocation) handle. Example: 'rev'.
        memory : int
            Node memory request in GB.
        walltime : float
            Node walltime request in hours.
        name : str
            SLURM job name.
        stdout_path : str
            Path to print .stdout and .stderr files.
        """

        self.make_path(stdout_path)
        self.out, self.err = self.sbatch(cmd,
                                         alloc=alloc,
                                         memory=memory,
                                         walltime=walltime,
                                         name=name,
                                         stdout_path=stdout_path)
        if self.out:
            self.id = self.out.split(' ')[-1]
        else:
            self.id = None

    def check_status(self, job, var='id'):
        """Check the status of this PBS job using qstat.

        Parameters
        ----------
        job : str
            Job name or ID number.
        var : str
            Identity/type of job identification input arg ('id' or 'name').

        Returns
        -------
        out : str | NoneType
            squeue job status str or None if not found.
            Common status codes: PD, R, CG (pending, running, complete).
        """

        # column location of various job identifiers
        col_loc = {'id': 0, 'name': 2}

        if var == 'name':
            # check for specific name
            squeue_rows = self.squeue(name=job)
        else:
            squeue_rows = self.squeue()

        if squeue_rows is None:
            return None
        else:
            # reverse the list so most recent jobs are first
            squeue_rows = reversed(squeue_rows)

        # update job status from qstat list
        for row in squeue_rows:
            row = row.split()
            # make sure the row is long enough to be a job status listing
            if len(row) > 7:
                if row[col_loc[var]].strip() in job.strip():
                    # Job status is located at the 4 index
                    status = row[4]
                    logger.debug('Job with {} "{}" has status: "{}"'
                                 .format(var, job, status))
                    return row[4]
        return None

    def squeue(self, name=None):
        """Run the SLURM squeue command and return the stdout split to rows.

        Parameters
        ----------
        name : str | None
            Optional to check the squeue for a specific job name (not limited
            to the 8 shown characters) or show users whole squeue.

        Returns
        -------
        squeue_rows : list | None
            List of strings where each string is a row in the squeue printout.
            Returns None if squeue is empty.
        """

        cmd = ('squeue -u {user}{job_name}'
               .format(user=self.USER,
                       job_name=' -n {}'.format(name) if name else ''))
        stdout, _ = self.submit(cmd)
        if not stdout:
            # No jobs are currently running.
            return None
        else:
            squeue_rows = stdout.split('\n')
            return squeue_rows

    def sbatch(self, cmd, alloc, memory, walltime, name='reV',
               stdout_path='./stdout', keep_sh=False):
        """Submit a SLURM job via sbatch command and SLURM shell script

        Parameters
        ----------
        cmd : str
            Command to be submitted in PBS shell script. Example:
                'python -m reV.generation.cli_gen'
        alloc : str
            HPC project (allocation) handle. Example: 'rev'.
        memory : int
            Node memory request in GB.
        walltime : float
            Node walltime request in hours.
        name : str
            SLURM job name.
        stdout_path : str
            Path to print .stdout and .stderr files.
        keep_sh : bool
            Boolean to keep the .sh files. Default is to remove these files
            after job submission.

        Returns
        -------
        out : str
            sbatch standard output, this is typically the SLURM job ID.
        err : str
            sbatch standard error, this is typically an empty string if the job
            was submitted successfully.
        """

        status = self.check_status(name, var='name')

        if status == 'PD' or status == 'R':
            warn('Not submitting job "{}" because it is already in '
                 'squeue with status: "{}"'.format(name, status))
            out = None
            err = 'already_running'
        else:
            fname = '{}.sh'.format(name)
            script = ('#!/bin/bash\n'
                      '#SBATCH --account={a} # allocation account\n'
                      '#SBATCH --time={t} # walltime\n'
                      '#SBATCH --job-name={n} # job name\n'
                      '#SBATCH --nodes=1 # number of nodes\n'
                      '#SBATCH --mem={m} # node RAM in MB\n'
                      '#SBATCH --output={p}/{n}_%j.o\n'
                      '#SBATCH --error={p}/{n}_%j.e\n'
                      '{cmd}'
                      .format(a=alloc, t=self.walltime(walltime), n=name,
                              m=int(memory * 1000), p=stdout_path, cmd=cmd))

            # write the shell script file and submit as qsub job
            self.make_sh(fname, script)
            out, err = self.submit('sbatch {script}'.format(script=fname))

            if not err:
                logger.debug('SLURM job "{}" with id #{} submitted '
                             'successfully'.format(name, out))
                if not keep_sh:
                    self.rm(fname)

        return out, err


def execute_parallel(fun, execution_iter, loggers=[], n_workers=None,
                     **kwargs):
    """Execute a parallel compute on a single node.

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


def execute_futures(fun, execution_iter, cluster=None, loggers=[], **kwargs):
    """Execute concurrent futures with an established cluster.

    Parameters
    ----------
    fun : function
        Python function object that will be submitted to futures. See
        downstream execution methods for arg passing structure.
    execution_iter : iter
        Python iterator that controls the futures submitted to dask.
    cluster : dask.distributed.LocalCluster | None
        Dask cluster object created from the LocalCluster() class. None creates
        a client with all available workers.
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
        Worker number (for debugging purposes).
    **kwargs : dict
        Key word arguments passed to fun.
    """

    logger.debug('Running single serial execution on worker #{} for: {}'
                 .format(worker, input_obj))
    out = fun(input_obj, **kwargs)
    log_mem()

    return out


class SmartParallelJob:
    """Single node parallel compute manager with smart data flushing."""

    def __init__(self, obj, execution_iter, loggers=[], n_workers=None,
                 mem_util_lim=0.7):
        """Single node parallel compute manager with smart data flushing.

        Parameters
        ----------
        obj : object
            Python object that will be submitted to futures. Must have methods
            run(arg) and flush(). run(arg) must take the iteration result of
            execution_iter as the single positional argument. Additionally,
            the results of obj.run(arg) will be passed to obj.out. obj.out
            will be passed None when the memory is to be cleared. It is
            advisable that obj.run() be a @staticmethod for dramatically
            faster submission to the Dask client.
        execution_iter : iter
            Python iterator that controls the futures submitted to dask.
        loggers : list
            List of logger names to initialize on the workers.
        n_workers : int
            Number of workers to scale the cluster to. None will use all
            available workers in a local cluster.
        mem_util_lim : float
            Memory utilization limit (fractional). If the used memory divided
            by the total memory is greater than this value, the obj.out will
            be flushed and the local node memory will be cleared.
        """

        self._obj = obj
        self._execution_iter = execution_iter
        self._loggers = loggers
        self._n_workers = n_workers
        self._mem_util_lim = mem_util_lim

    @property
    def cluster(self):
        """Get a Dask LocalCluster object.

        Returns
        -------
        _cluster : dask.distributed.LocalCluster
            Local cluster object with all available workers or  self._n_workers
            (if specified).
        """

        if not hasattr(self, '_cluster'):
            # start a local cluster on a personal comp or HPC single node
            if self._n_workers is None:
                self._cluster = LocalCluster(n_workers=None)
            elif isinstance(self._n_workers, int):
                self._cluster = LocalCluster(n_workers=self._n_workers)
            else:
                raise ExecutionError('Bad number of workers: {}'
                                     .format(self._n_workers))
        return self._cluster

    @property
    def execution_iter(self):
        """Get the iterator object that controls the parallel execution.

        Returns
        -------
        _execution_iter : iterable
            Iterable object that controls the processes of the parallel job.
        """
        return self._execution_iter

    @property
    def mem_util_lim(self):
        """Get the memory utilization limit (fractional).

        Returns
        -------
        _mem_util_lim : float
            Fractional memory utilization limit. If the used memory divided
            by the total memory is greater than this value, the obj.out will
            be flushed and the local node memory will be cleared.
        """
        return self._mem_util_lim

    @property
    def n_workers(self):
        """Get the number of workers in the local cluster.

        Returns
        -------
        _n_workers : int | NoneType
            Number of workers. Will be calculated based on the local cluster if
            self._cluster is an attribute.
        """

        if hasattr(self, '_cluster') and self._n_workers is None:
            self._n_workers = len(self.cluster.workers)
        return self._n_workers

    @property
    def obj(self):
        """Get the main python object that will be submitted to futures.

        Returns
        -------
        _obj : Object
            Python object that will be submitted to futures. Must have methods
            run(arg) and flush(). run(arg) must take the iteration result of
            execution_iter as the single positional argument. Additionally,
            the results of obj.run(arg) will be passed to obj.out. obj.out
            will be passed None when the memory is to be cleared. It is
            advisable that obj.run() be a @staticmethod for dramatically
            faster submission to the Dask client.
        """
        return self._obj

    @obj.setter
    def obj(self, inp_obj):
        """Verify the input object and set to protected property.

        Parameters
        ----------
        inp_obj : Object
            Python object that will be submitted to futures. Must have methods
            run(arg) and flush(). run(arg) must take the iteration result of
            execution_iter as the single positional argument. Additionally,
            the results of obj.run(arg) will be passed to obj.out. obj.out
            will be passed None when the memory is to be cleared. It is
            advisable that obj.run() be a @staticmethod for dramatically
            faster submission to the Dask client.
        """
        if not hasattr(inp_obj, 'run') or not hasattr(inp_obj, 'flush'):
            raise ExecutionError('Parallel execution with object: "{}" '
                                 'failed. The target object must have methods '
                                 'run() and flush()'.format(inp_obj))
        else:
            self._obj = inp_obj

    @property
    def loggers(self):
        """Get the list of logger names."""
        return self._loggers

    def init_loggers(self, client):
        """Initialize loggers on all workers

        Parameters
        ----------
        client : dask.distributed.Client
            Local Dask client with a local cluster. Loggers will be initialized
            on the workers in this client.
        """
        for logger_name in self.loggers:
            client.run(REV_LOGGERS.init_logger, logger_name)

    def flush(self):
        """Flush obj.out to disk, set obj.out=None, and garbage collect."""
        # memory utilization limit exceeded, flush memory to disk
        self.obj.flush()
        self.obj.out = None
        gc.collect()

    def gather_and_flush(self, i, client, futures, force_flush=False):
        """Wait on futures, potentially update obj.out and flush to disk.

        Parameters
        ----------
        i : int | str
            Iteration number (for logging purposes).
        client : dask.distributed.Client
            Dask client.
        futures : list
            List of Dask futures to wait on or gather.
        force_flush : bool
            Option to force a disk flush. Useful for end-of-iteration. If this
            is False, will only flush to disk if the memory utilization exceeds
            the mem_util_lim.

        Returns
        -------
        futures : list
            List of Dask futures. If the memory was flushed, this is a
            cleared list: futures.clear()
        client : dask.distributed.Client
            Dask client. If the memory was flushed, the client is restarted
            before returning: client.restart()
        """

        wait(futures)
        mem = psutil.virtual_memory()
        logger.debug('Parallel run at iteration {0}. '
                     'Results are stored in memory for {1} futures '
                     'and memory usage is {2:.3f} GB out of {3:.3f} GB '
                     'total ({4:.1f}% used)'
                     .format(i, len(futures),
                             mem.used / 1e9,
                             mem.total / 1e9,
                             100 * mem.used / mem.total))

        if ((mem.used / mem.total) >= self.mem_util_lim) or force_flush:
            logger.debug('Flushing memory to disk. The memory utilization is '
                         '{0:.2f}% and the limit is {1:.2f}%.'
                         .format(100 * (mem.used / mem.total),
                                 100 * self.mem_util_lim))
            # send gathered futures to object output
            # (obj.out should be a property setter that will append new data.)
            self.obj.out = client.gather(futures)
            futures.clear()
            client.restart()
            logger.debug('Restarted Dask client.')
            self.flush()
        return futures, client

    @classmethod
    def execute(cls, obj, execution_iter, loggers=[], n_workers=None,
                mem_util_lim=0.7, **kwargs):
        """Execute the smart parallel run with data flushing.

        Parameters
        ----------
        obj : object
            Python object that will be submitted to futures. Must have methods
            run(arg) and flush(). run(arg) must take the iteration result of
            execution_iter as the single positional argument. Additionally,
            the results of obj.run(arg) will be passed to obj.out. obj.out
            will be passed None when the memory is to be cleared. It is
            advisable that obj.run() be a @staticmethod for dramatically
            faster submission to the Dask client.
        execution_iter : iter
            Python iterator that controls the futures submitted to dask.
        loggers : list
            List of logger names to initialize on the workers.
        n_workers : int
            Number of workers to scale the cluster to. None will use all
            available workers in a local cluster.
        mem_util_lim : float
            Memory utilization limit (fractional). If the used memory divided
            by the total memory is greater than this value, the obj.out will
            be flushed and the local node memory will be cleared.
        kwargs : dict
            Keyword arguments to be passed to obj.run(). Makes it easier to
            have obj.run() as a @staticmethod.
        """

        manager = cls(obj, execution_iter, loggers=loggers,
                      n_workers=n_workers, mem_util_lim=mem_util_lim)

        # start a local cluster on a personal comp or HPC single node
        cluster = manager.cluster
        # Get the number of workers in case it was input as None
        n_workers = manager.n_workers

        logger.info('Executing parallel run on cluster with {0} workers. '
                    .format(n_workers))
        log_mem()

        # initialize a client based on the input cluster.
        with Client(cluster) as client:
            futures = []
            manager.init_loggers(client)

            # iterate through split executions, submitting each to worker
            for i, exec_slice in enumerate(manager.execution_iter):
                logger.debug('Kicking off serial worker #{0} for: {1}. '
                             .format(i, exec_slice))

                # submit executions and append to futures list
                futures.append(client.submit(obj.run, exec_slice, **kwargs))

                # Take a pause after one complete set of workers
                if (i + 1) % n_workers == 0:
                    futures, client = manager.gather_and_flush(i, client,
                                                               futures)

            # All futures complete
            futures = manager.gather_and_flush('END', client, futures,
                                               force_flush=True)
            logger.debug('Smart parallel job complete. Returning execution '
                         'control to higher level processes.')
            log_mem()
        manager.cluster.close()
