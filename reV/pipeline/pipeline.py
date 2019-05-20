"""
reV date pipeline architecture.
"""
import time
import json
import os
import logging

from reV.config.base_analysis_config import AnalysisConfig
from reV.utilities.execution import SubprocessManager
from reV.utilities.exceptions import ExecutionError
from reV.pipeline.status import Status


logger = logging.getLogger(__name__)


class Pipeline:
    """reV pipeline execution framework."""

    COMMANDS = ('generation', 'econ', 'collect')
    RETURNCODE = {0: 'successful',
                  1: 'running',
                  2: 'failed'}

    def __init__(self, run_list, status_dir=None):
        """
        Parameters
        ----------
        run_list : list
            List of reV pipeline steps. Each pipeline step entry must have
            the following format:
                run_list[0] = {"rev_module": "module_config_file"}
        status_dir : str
            Optional directory to save status file. Default will be dirout for
            each module.
        """

        self._run_list = run_list
        self._status_dir = status_dir

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
        cmd = self._get_cmd(command, f_config, status_dir=self._status_dir)

        config_obj = self._get_config_obj(f_config)
        status = self._get_status_obj(dirout=config_obj.dirout,
                                      status_dir=self._status_dir)
        if command in status.data:
            status.data[command]['pipeline_index'] = i
        else:
            status.data[command] = {'pipeline_index': i}
        status._dump()

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
        config_obj = self._get_config_obj(f_config)
        status = self._get_status_obj(dirout=config_obj.dirout,
                                      status_dir=self._status_dir)

        if os.path.isfile(status._fpath):
            returncode = self._get_return_code(status, module)
        else:
            # file does not yet exist. assume job is not yet running.
            returncode = 1

        return returncode

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

        with open(f_config, 'r') as f:
            config_dict = json.load(f)
        return AnalysisConfig(config_dict)

    @staticmethod
    def _get_status_obj(dirout=None, status_dir=None):
        """Get a reV pipeline status object.

        Parameters
        ----------
        dirout : str
            Output directory which will be used if no status directory.
        status_dir : str
            Status directory which is the prefered location.

        Returns
        -------
        status : reV.pipeline.status.Status
            reV job status object.
        """

        if status_dir is None:
            status = Status(dirout)
        else:
            status = Status(status_dir)
        return status

    @staticmethod
    def _get_return_code(status, module):
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
            for job_name in status.data[module].keys():
                if job_name != 'pipeline_index':
                    status._update_job_status(module, job_name)
                    status._dump()
                    js = status.data[module][job_name]['job_status']

                    logger.debug('reV pipeline job "{}" has status "{}".'
                                 .format(job_name, js))

                    # if return code is changed to 1 or 2, do not update again
                    if returncode == 0:
                        if js == 'failed':
                            returncode = 2
                        elif js is None:
                            status.set_job_status(status._path, module,
                                                  job_name, 'failed')
                            returncode = 2
                        elif js not in status.FROZEN_STATUS:
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
    def _get_cmd(command, f_config, status_dir=None):
        """Get the python cli call string based on the command and config arg.

        Parameters
        ----------
        command : str
            reV cli command which should be a reV module.
        f_config : str
            File path for the config file corresponding to the command.
        status_dir : str
            Optional directory to save status file. Default will be dirout for
            each module.

        Returns
        -------
        cmd : str
            Python reV CLI call string.
        """
        if command not in Pipeline.COMMANDS:
            raise KeyError('Could not recongize command "{}". '
                           'Available commands are: {}'
                           .format(command, Pipeline.COMMANDS))
        sdir_str = ''
        if status_dir:
            sdir_str = '-st {}'.format(status_dir)
        cmd = ('python -m reV.cli -c {} {} {}'
               .format(f_config, sdir_str, command))
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

    @staticmethod
    def parse_previous(status_dir, module, target='fpath'):
        """Parse output file paths from the previous pipeline step.

        Parameters
        ----------
        status_dir : str
            Directory containing the status file to parse.
        module : str
            Current module (i.e. current pipeline step).
        target : str
            Parsing target of previous module.

        Returns
        -------
        out : list
            Arguments parsed from the status file in status_dir from
            the module preceding the input module arg.
        """

        status = Pipeline._get_status_obj(status_dir=status_dir)
        msg = ('Could not parse data regarding "{}" from reV status file in '
               '"{}".'.format(module, status_dir))
        if module in status.data:
            if 'pipeline_index' in status.data[module]:
                msg = None
        if msg:
            raise KeyError(msg)

        i1 = int(status.data[module]['pipeline_index'])
        i0 = i1 - 1

        module_status = Pipeline._get_module_status(status, i0)
        job_statuses = Pipeline._get_job_status(module_status)

        out = []
        if target == 'fpath':
            for status in job_statuses:
                out.append(os.path.join(status['dirout'], status['fout']))
        else:
            for status in job_statuses:
                out.append(status[target])

        return out

    @classmethod
    def run(cls, run_list, status_dir=None):
        """Run the reV pipeline.

        Parameters
        ----------
        run_list : list
            List of reV pipeline steps. Each pipeline step entry must have
            the following format:
                run_list[0] = {"rev_module": "module_config_file"}
        status_dir : str
            Optional directory to save status file. Default will be dirout for
            each module.
        """
        pipe = cls(run_list, status_dir=status_dir)
        pipe._main()
