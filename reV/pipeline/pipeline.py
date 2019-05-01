"""
reV date pipeline architecture.
"""
import time
import json
import os
import logging

from reV.config.analysis_configs import AnalysisConfig
from reV.utilities.execution import SubprocessManager
from reV.utilities.exceptions import ExecutionError
from reV.pipeline.status import Status


logger = logging.getLogger(__name__)


class Pipeline:
    """reV pipeline execution framework."""

    COMMANDS = ('generation', 'econ')
    RETURNCODE = {0: 'successful',
                  1: 'running',
                  2: 'failed'}

    def __init__(self, run_list):
        """
        Parameters
        ----------
        run_list : list
            List of reV pipeline steps. Each pipeline step entry must have
            the following format:
                run_list[0] = {"rev_module": "module_config_file"}
        """

        self._run_list = run_list

    def _main(self):
        """Iterate through run list submitting steps while monitoring status"""

        for i, step in enumerate(self._run_list):
            returncode = self._check_step_completed(i)

            if returncode == 0:
                logger.info('Based on successful end state in reV status '
                            'file, not running pipeline step {}: {}.'
                            .format(i, step))
            else:
                returncode = 1
                self._submit_step(i)
                while returncode == 1:
                    time.sleep(5)
                    returncode = self._check_step_completed(i)

                    if returncode == 2:
                        module, f_config = self._get_command_config(i)
                        raise ExecutionError('reV pipeline failed at step '
                                             '{} "{}" {}'
                                             .format(i, module, f_config))

    def _submit_step(self, i):
        """Submit a step in the pipeline.

        Parameters
        ----------
        i : int
            Step index in the pipeline run list.
        """

        command, f_config = self._get_command_config(i)
        cmd = self._get_cmd(command, f_config)
        logger.info('reV pipeline submitting subprocess:\n\t"{}"'.format(cmd))
        SubprocessManager.submit(cmd)

    def _check_step_completed(self, i):
        """Check if a pipeline step has been completed.

        Parameters
        ----------
        i : int
            Step index in the pipeline run list.

        Returns
        -------
        returncode : int
            Pipeline step return code.
        """

        module, f_config = self._get_command_config(i)

        with open(f_config, 'r') as f:
            config_dict = json.load(f)
        config_obj = AnalysisConfig(config_dict)

        status = Status(config_obj.dirout)

        if os.path.isfile(status._fpath):
            returncode = self._get_return_code(status, module)
        else:
            # file does not yet exist. assume job is not yet running.
            returncode = 1

        return returncode

    def _get_return_code(self, status, module):
        """Get a return code for a full module based on a status object.

        Parameters
        ----------
        status : reV.pipeline.status.Status
            reV job status object.
        module : str
            reV module.

        Returns
        -------
        returncode : int
            Pipeline step return code.
        """

        returncode = 0
        if module not in status.data:
            returncode = 1
        else:
            for job_name, job_attrs in status.data[module].items():
                status._update_job_status(module, job_name)
                status._dump()

                logger.debug('reV pipeline job "{}" has status "{}".'
                             .format(job_name, job_attrs['job_status']))

                # if return code is changed to 1 or 2, do not update again
                if returncode == 0:
                    if job_attrs['job_status'] == 'failed':
                        returncode = 2
                    elif job_attrs['job_status'] is None:
                        status.set_job_status(status._path, module,
                                              job_name, 'failed')
                        returncode = 2
                    elif job_attrs['job_status'] not in status.FROZEN_STATUS:
                        returncode = 1
        return returncode

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

    @staticmethod
    def _get_cmd(command, f_config):
        """Get the python cli call string based on the command and config arg.

        Parameters
        ----------
        command : str
            reV cli command which should be a reV module.
        f_config : str
            File path for the config file corresponding to the command.

        Returns
        -------
        cmd : str
            Python reV CLI call string.
        """
        if command not in Pipeline.COMMANDS:
            raise KeyError('Could not recongize command "{}". '
                           'Available commands are: {}'
                           .format(command, Pipeline.COMMANDS))
        cmd = 'python -m reV.cli -c {} {}'.format(f_config, command)
        return cmd

    @classmethod
    def run(cls, run_list):
        """Run the reV pipeline.

        Parameters
        ----------
        run_list : list
            List of reV pipeline steps. Each pipeline step entry must have
            the following format:
                run_list[0] = {"rev_module": "module_config_file"}
        """
        pipe = cls(run_list)
        pipe._main()
