"""
reV Base Configuration Frameworks
"""
import os
import json
import logging
from warnings import warn

from reV.utilities.execution import SLURM, PBS


logger = logging.getLogger(__name__)


class Status(dict):
    """Base class for reV data pipeline health and status information."""

    FROZEN_STATUS = ('successful', 'failed')

    def __init__(self, path, name='rev_status.json'):
        """
        Parameters
        ----------
        path : str
            Path to json status file.
        name : str
            Filename for rev status file.
        """

        self._path = path
        if path.endswith('.json'):
            self._fpath = self._path
        else:
            if not name.endswith('.json'):
                name += '.json'
            self._fpath = os.path.join(path, name)
        self.data = self._load()

    def _load(self):
        """Load status json.

        Returns
        -------
        data : dict
            JSON file contents loaded as a python dictionary.
        """
        if os.path.isfile(self._fpath):
            with open(self._fpath, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        return data

    def _dump(self):
        """Load json config into config class instance.

        Parameters
        ----------
        data : dict
            reV data pipeline status info.
        """
        with open(self._fpath, 'w') as f:
            json.dump(self.data, f, sort_keys=True, indent=4,
                      separators=(',', ': '))

    @staticmethod
    def _get_check_method(hardware='eagle'):
        """Get a method to check job status on the specified hardware.

        Parameters
        ----------
        hardware : str
            Hardware specification that determines how jobs are monitored.
            Options are found in the options dictionary below.
        """
        options = {'eagle': SLURM.check_status,
                   'peregrine': PBS.check_status}
        try:
            method = options[hardware]
        except KeyError as _:
            raise KeyError('Could not check job on the requested hardware: '
                           '"{}".'.format(hardware))
        return method

    @staticmethod
    def _get_job_status(job_id, hardware='eagle'):
        """Get the job status using pre-defined hardware-specific methods.

        Parameters
        ----------
        job_id : str | int
            SLURM or PBS job submission id.
        hardware : str
            Hardware option, either eagle or peregrine.

        Returns
        -------
        status : str | None
            Job status from qstat/squeue. None if no job found.
        """
        status = None
        if job_id:
            method = Status._get_check_method(hardware=hardware)
            status = method(job_id)
        return status

    @staticmethod
    def _check_completion_file(path, job_name):
        """Look for a completion file in the target path.

        Parameters
        ----------
        path : str
            Directory to look for completion file.
        job_name : str
            Job name.

        Returns
        -------
        status : str | None
            Job status if completion file found.
        """
        status = None
        for fname in os.listdir(path):
            if str(job_name) in fname and fname.endswith(Status.FROZEN_STATUS):
                os.remove(os.path.join(path, fname))
                status = fname.split('.')[-1]
                break
        return status

    def _update_job_status(self, module, job_name, hardware='eagle'):
        """Update HPC job and respective job status to the status obj instance.

        Parameters
        ----------
        module : str
            reV module that the job belongs to.
        job_name : str
            Unique job name identification.
        hardware : str
            Hardware option, either eagle or peregrine.
        """

        if module not in self.data:
            raise KeyError('reV status has not yet been initialized for "{}".'
                           .format(module))
        if job_name not in self.data[module]:
            raise KeyError('reV status has not yet been initialized for "{}".'
                           .format(job_name))
        else:
            previous = self.data[module][job_name]['job_status']

        job_id = self.data[module][job_name].get('job_id', None)
        hardware = self.data[module][job_name].get('hardware', hardware)

        # look for completion file.
        current = self._check_completion_file(self._path, job_name)
        if current not in self.FROZEN_STATUS:
            # check job status if completion file not present.
            current = self._get_job_status(job_id, hardware=hardware)

        # do not overwrite a successful or failed job status.
        if (current != previous and previous not in self.FROZEN_STATUS):
            self.data[module][job_name]['job_status'] = current

    def _set_job_status(self, module, job_name, status):
        """Set an updated job status when finished. Must be a status string in
        the FROZEN_STATUS class attribute.

        Parameters
        ----------
        module : str
            reV module that the job belongs to.
        job_name : str
            Unique job name identification.
        status : str
            Status string to set. Must be a status string in
            the FROZEN_STATUS class attribute.
        """

        if module not in self.data:
            raise KeyError('reV pipeline status has not been initialized '
                           'for "{}".'.format(module))
        if job_name not in self.data[module]:
            raise KeyError('reV pipeline status has not been initialized '
                           'for "{}: {}".'.format(module, job_name))

        if status in self.FROZEN_STATUS:
            self.data[module][job_name]['job_status'] = status
        else:
            warn('Can only force-set a job status to one of the following: {}'
                 .format(self.FROZEN_STATUS))

    @staticmethod
    def make_completion_file(path, job_name, status):
        """Make a temporary file recording the status of a job.

        Parameters
        ----------
        path : str
            Path to json status file.
        job_name : str
            Unique job name identification.
        status : str
            Status string to set. Must be a status string in
            the FROZEN_STATUS class attribute.
        """
        if status in Status.FROZEN_STATUS:
            open(os.path.join(path, job_name + '.{}'.format(status)), 'w')

    @classmethod
    def add_job(cls, path, module, job_name, replace=True, job_attrs=None):
        """Add or update job status using pre-defined methods.

        Parameters
        ----------
        path : str
            Path to json status file.
        module : str
            reV module that the job belongs to.
        job_name : str
            Unique job name identification.
        replace : bool
            Flag to replace pre-existing job status.
        job_attrs : dict
            Job attributes. Should include 'job_id' if running on HPC.
        """

        obj = cls(path)

        if job_attrs is None:
            job_attrs = {}

        if 'hardware' in job_attrs:
            if job_attrs['hardware'] in ('eagle', 'peregrine'):
                if 'job_id' not in job_attrs:
                    warn('Key "job_id" should be in kwargs for "{}" if '
                         'adding job from an eagle or peregrine node.'
                         .format(job_name))

        if module not in obj.data:
            obj.data[module] = {job_name: job_attrs}
        if replace:
            obj.data[module][job_name] = job_attrs

        obj.data[module][job_name]['job_status'] = 'submitted'

        obj._dump()

    @classmethod
    def retrieve_job_status(cls, path, module, job_name):
        """Update and retrieve job status using pre-defined methods.

        Parameters
        ----------
        path : str
            Path to json status file.
        module : str
            reV module that the job belongs to.
        job_name : str
            Unique job name identification.
        """

        obj = cls(path)
        obj._update_job_status(module, job_name)
        obj._dump()
        return obj.data[module][job_name]['job_status']

    @classmethod
    def set_job_status(cls, path, module, job_name, status):
        """Force set a job status to a frozen status and save to status file.

        Parameters
        ----------
        path : str
            Path to json status file.
        module : str
            reV module that the job belongs to.
        job_name : str
            Unique job name identification.
        status : str
            Status string to set. Must be a status string in
            the FROZEN_STATUS class attribute.
        """

        obj = cls(path)
        obj._set_job_status(module, job_name, status)
        obj._dump()

    @classmethod
    def update(cls, path):
        """Update all job statuses and dump to json.

        Parameters
        ----------
        path : str
            Path to json status file.
        """

        obj = cls(path)
        for module in obj.data.keys():
            for job_name in obj.data[module].keys():
                if job_name != 'pipeline_index':
                    obj._update_job_status(module, job_name)
        obj._dump()
