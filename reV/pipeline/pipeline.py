# -*- coding: utf-8 -*-
"""
reV data pipeline architecture.
"""
import time
import os
import numpy as np
import logging
from warnings import warn

from reV.config.base_analysis_config import AnalysisConfig
from reV.config.pipeline import PipelineConfig
from reV.pipeline.status import Status
from reV.utilities.exceptions import ExecutionError

from rex.utilities.execution import SubprocessManager
from rex.utilities.hpc import SLURM
from rex.utilities.loggers import init_logger
from rex.utilities.utilities import safe_json_load

logger = logging.getLogger(__name__)


class Pipeline:
    """reV pipeline execution framework."""

    CMD_BASE = 'python -m reV.cli -c {fp_config} {command}'

    COMMANDS = ('generation',
                'econ',
                'offshore',
                'collect',
                'multi-year',
                'supply-curve-aggregation',
                'supply-curve',
                'rep-profiles',
                'qa-qc',
                )

    RETURN_CODES = {0: 'successful',
                    1: 'running',
                    2: 'failed',
                    3: 'complete'}

    def __init__(self, pipeline, monitor=True, verbose=False):
        """
        Parameters
        ----------
        pipeline : str | dict
            Pipeline config file path or dictionary.
        monitor : bool
            Flag to perform continuous monitoring of the pipeline.
        verbose : bool
            Flag to submit pipeline steps with -v flag for debug logging
        """
        self.monitor = monitor
        self.verbose = verbose
        self._config = PipelineConfig(pipeline)
        self._run_list = self._config.pipeline
        self._init_status()

        # init logger for pipeline module if requested in input config
        if 'logging' in self._config:
            init_logger('reV', **self._config.logging)

    def _init_status(self):
        """Initialize the status json in the output directory."""

        status = self._get_status_obj()

        for i, step in enumerate(self._run_list):

            for module in step.keys():
                module_dict = {module: {'pipeline_index': i}}
                status.data = status.update_dict(status.data, module_dict)

        status._dump()

    def _cancel_all_jobs(self):
        """Cancel all jobs in this pipeline via SLURM scancel."""
        status = self._get_status_obj()
        s = SLURM()
        for job_id in status.job_ids:
            s.scancel(job_id)
        logger.info('Pipeline job "{}" cancelled.'.format(self._config.name))

    def _main(self):
        """Iterate through run list submitting steps while monitoring status"""

        i = 0

        for i, step in enumerate(self._run_list):
            return_code = self._check_step_completed(i)

            if return_code == 0:
                logger.debug('Successful: "{}".'.format(list(step.keys())[0]))
            else:
                return_code = 1
                self._submit_step(i)

                # do not enter while loop for continuous monitoring
                if not self.monitor:
                    break

                time.sleep(1)
                while return_code == 1 and self.monitor:
                    time.sleep(5)
                    return_code = self._check_step_completed(i)

                    if return_code == 2:
                        module, f_config = self._get_command_config(i)
                        raise ExecutionError('Pipeline failed at step '
                                             '{} "{}" {}'
                                             .format(i, module, f_config))

        if i + 1 == len(self._run_list) and return_code == 0:
            logger.info('Pipeline job "{}" is complete.'
                        .format(self._config.name))
            logger.debug('Output directory is: "{}"'
                         .format(self._config.dirout))

    def _submit_step(self, i):
        """Submit a step in the pipeline.

        Parameters
        ----------
        i : int
            Step index in the pipeline run list.
        """

        command, f_config = self._get_command_config(i)
        cmd = self._get_cmd(command, f_config, verbose=self.verbose)

        logger.info('Pipeline submitting: "{}" for job "{}"'
                    .format(command, self._config.name))
        logger.debug('Pipeline submitting subprocess call:\n\t"{}"'
                     .format(cmd))

        try:
            stderr = SubprocessManager.submit(cmd)[1]
        except OSError as e:
            logger.exception('Pipeline subprocess submission returned an '
                             'error: \n{}'.format(e))
            raise e

        if stderr:
            logger.warning('Subprocess received stderr: \n{}'.format(stderr))

    def _check_step_completed(self, i):
        """Check if a pipeline step has been completed.

        Parameters
        ----------
        i : int
            Step index in the pipeline run list.

        Returns
        -------
        return_code : int
            Pipeline step return code.
        """

        module, _ = self._get_command_config(i)
        status = self._get_status_obj()
        submitted = self._check_jobs_submitted(status, module)
        if not submitted:
            return_code = 1
        else:
            return_code = self._get_module_return_code(status, module)

        return return_code

    @staticmethod
    def _check_jobs_submitted(status, module):
        """Check whether jobs have been submitted for a given module.

        Parameters
        ----------
        status : reV.pipeline.status.Status
            reV job status object.
        module : str
            reV module.

        Returns
        -------
        submitted : bool
            Boolean check to see if jobs have been submitted for the module arg
        """

        submitted = False
        if module in status.data:
            jobs = status.data[module]
            for job in jobs.keys():
                if job != 'pipeline_index':
                    submitted = True
                    break
        return submitted

    @staticmethod
    def _get_config_obj(f_config):
        """Get an analysis config object form a config json file.

        Parameters
        ----------
        f_config : str
            File path for config.

        Returns
        -------
        config_obj : AnalysisConfig
            reV analysis config object.
        """

        config_dict = safe_json_load(f_config)
        return AnalysisConfig(config_dict, check_keys=False)

    def _get_status_obj(self):
        """Get a reV pipeline status object.

        Returns
        -------
        status : reV.pipeline.status.Status
            reV job status object.
        """

        status = Status(self._config.dirout, name=self._config.name,
                        hardware=self._config.hardware)
        return status

    def _get_module_return_code(self, status, module):
        """Get a return code for a full module based on a status object.

        Parameters
        ----------
        status : reV.pipeline.status.Status
            reV job status object.
        module : str
            reV module.

        Returns
        -------
        return_code : int
            Pipeline step return code (for the full module in the pipeline
            step).
        """

        # initialize return code array
        arr = []
        check_failed = False

        if module not in status.data:
            # assume running
            arr = [1]
        else:
            for job_name in status.data[module].keys():
                if job_name != 'pipeline_index':

                    # update the job status and get the status string
                    status._update_job_status(module, job_name)
                    js = status.data[module][job_name]['job_status']

                    if js == 'successful':
                        arr.append(0)
                    elif js == 'failed':
                        arr.append(2)
                        check_failed = True
                    elif js is None:
                        arr.append(3)
                    else:
                        arr.append(1)

            status._dump()

        return_code = self._parse_code_array(arr)

        status = self.RETURN_CODES[return_code]
        fail_str = ''
        if check_failed and status != 'failed':
            fail_str = ', but some jobs have failed'
        logger.info('Module "{}" for job "{}" is {}{}.'
                    .format(module, self._config.name, status, fail_str))

        return return_code

    @staticmethod
    def _parse_code_array(arr):
        """Parse array of return codes to get single return code for module.

        Parameters
        ----------
        arr : list | np.ndarray
            List or array of integer return codes.

        Returns
        -------
        return_code : int
            Single return code for the module represented by the input array of
            return codes.
        """

        # check to see if all have completed, or any have failed
        check_success = all(np.array(arr) == 0)
        check_complete = all(np.array(arr) != 1)
        check_failed = any(np.array(arr) == 2)

        # only return success if all have succeeded.
        if check_success:
            return_code = 0
        # Only return failed when all have finished.
        elif check_complete & check_failed:
            return_code = 2
        # only return complete when all have completed
        # (but some should have succeeded or failed)
        elif check_complete:
            return_code = 3
        # otherwise, jobs are still running
        else:
            return_code = 1

        return return_code

    def _get_command_config(self, i):
        """Get the (command, config) key pair.

        Parameters
        ----------
        i : int
            Step index in the pipeline run list.

        Returns
        -------
        key_pair : list
            Two-entry list containing [command, config_file].
        """
        key_pair = list(self._run_list[i].items())[0]
        return key_pair

    @classmethod
    def _get_cmd(cls, command, f_config, verbose=False):
        """Get the python cli call string based on the command and config arg.

        Parameters
        ----------
        command : str
            reV cli command which should be a reV module.
        f_config : str
            File path for the config file corresponding to the command.
        verbose : bool
            Flag to submit pipeline steps with -v flag for debug logging

        Returns
        -------
        cmd : str
            Python reV CLI call string.
        """
        if command not in cls.COMMANDS:
            raise KeyError('Could not recongize command "{}". '
                           'Available commands are: {}'
                           .format(command, cls.COMMANDS))
        cmd = cls.CMD_BASE.format(fp_config=f_config, command=command)
        if verbose:
            cmd += ' -v'

        return cmd

    @staticmethod
    def _get_module_status(status, i):
        """Get the status dict for the module with the given pipeline index.

        Parameters
        ----------
        status : reV.pipeline.status.Status
            reV job status object.
        i : int
            pipeline index of desired module.

        Returns
        -------
        out : dict
            Status dictionary for the module with pipeline index i.
        """

        # iterate through modules and find the one that was run previously
        for module_status in status.data.values():
            i_current = module_status.get('pipeline_index', -99)
            if str(i) == str(i_current):
                out = module_status
                break

        return out

    @staticmethod
    def _get_job_status(module_status, option='all'):
        """Get a job status dict from the module status dict.

        Parameters
        ----------
        module_status : dict
            Status dictionary for a full reV module containing one or more
            job status dict.
        option : str
            Option to retrieve one or many jobs from the module status dict.

        Returns
        -------
        out : dict | list
            Job status(es).
        """

        # find the preceding job (1st is used, should be one job in most cases)
        if option == 'first':
            for job, job_status in module_status.items():
                if job != 'pipeline_index':
                    out = job_status
                    break
        elif option == 'all':
            out = []
            for job, job_status in module_status.items():
                if job != 'pipeline_index':
                    out.append(job_status)
        else:
            raise KeyError('Did not recognize pipeline job status request '
                           'for "{}"'.format(option))
        return out

    @classmethod
    def parse_previous(cls, status_dir, module, target='fpath',
                       target_module=None):
        """Parse output file paths from the previous pipeline step.

        Parameters
        ----------
        status_dir : str
            Directory containing the status file to parse.
        module : str
            Current module (i.e. current pipeline step).
        target : str
            Parsing target of previous module.
        target_module : str | None
            Optional name of module to pull target data from.

        Returns
        -------
        out : list
            Arguments parsed from the status file in status_dir from
            the module preceding the input module arg.
        """

        status = Status(status_dir)
        msg = ('Could not parse data regarding "{}" from status file in '
               '"{}".'.format(module, status_dir))
        if module in status.data:
            if 'pipeline_index' in status.data[module]:
                msg = None
        if msg:
            raise KeyError(msg)

        i1 = int(status.data[module]['pipeline_index'])
        i0 = i1 - 1

        if i0 < 0:
            i0 = 0
            warn('Module "{0}" is attempting to parse a previous pipeline '
                 'step, but it appears to be the first step. Attempting to '
                 'parse data from {0}.'.format(module))

        if target_module is None:
            module_status = cls._get_module_status(status, i0)
            job_statuses = cls._get_job_status(module_status)
        else:
            if target_module not in status.data:
                raise KeyError('Target module "{}" not found in pipeline '
                               'status dictionary.'.format(target_module))
            else:
                module_status = status.data[target_module]
                job_statuses = cls._get_job_status(module_status)

        out = []
        if target == 'fpath':
            for status in job_statuses:
                out.append(os.path.join(status['dirout'], status['fout']))
        else:
            for status in job_statuses:
                out.append(status[target])

        return out

    @classmethod
    def cancel_all(cls, pipeline):
        """Cancel all jobs via SLURM scancel corresponding to pipeline.

        Parameters
        ----------
        pipeline : str | dict
            Pipeline config file path or dictionary.
        """

        pipe = cls(pipeline)
        pipe._cancel_all_jobs()

    @classmethod
    def run(cls, pipeline, monitor=True, verbose=False):
        """Run the reV pipeline.

        Parameters
        ----------
        pipeline : str | dict
            Pipeline config file path or dictionary.
        monitor : bool
            Flag to perform continuous monitoring of the pipeline.
        verbose : bool
            Flag to submit pipeline steps with -v flag for debug logging
        """

        pipe = cls(pipeline, monitor=monitor, verbose=verbose)
        pipe._main()
