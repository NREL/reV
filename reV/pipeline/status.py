"""
reV Base Configuration Frameworks
"""
import copy
import os
import json
import logging
import time
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
        if str(path).endswith('.json'):
            self._fpath = self._path
        else:
            if not str(name).endswith('.json'):
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

    def _check_all_job_files(self, path):
        """Look for all single-job job status files in the target path and
        update status.

        Parameters
        ----------
        path : str
            Directory to look for completion file.
        """

        for fname in os.listdir(path):
            if fname.startswith('status_') and fname.endswith('.json'):
                # wait one second to make sure file is finished being written
                time.sleep(0.1)
                with open(os.path.join(path, fname), 'r') as f:
                    status = json.load(f)
                self.data = self.update_dict(self.data, status)
                os.remove(os.path.join(path, fname))

    @staticmethod
    def _check_job_file(path, job_name):
        """Look for a single-job job status file in the target path.

        Parameters
        ----------
        path : str
            Directory to look for completion file.
        job_name : str
            Job name.

        Returns
        -------
        status : dict | None
            Job status dictionary if completion file found.
        """
        status = None
        target_fname = 'status_{}.json'.format(job_name)
        for fname in os.listdir(path):
            if fname == target_fname:
                # wait one second to make sure file is finished being written
                time.sleep(0.1)
                with open(os.path.join(path, fname), 'r') as f:
                    status = json.load(f)
                os.remove(os.path.join(path, fname))
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

        # init defaults in case job/module not in status file yet
        previous = None
        job_id = None
        if module in self.data:
            if job_name in self.data[module]:
                previous = self.data[module][job_name].get('job_status', None)
                job_id = self.data[module][job_name].get('job_id', None)
                hardware = self.data[module][job_name].get('hardware',
                                                           hardware)

        # look for completion file.
        current = self._check_job_file(self._path, job_name)

        # Update status data dict recursively if job file was found
        if current is not None:
            self.data = Status.update_dict(self.data, current)

        # check job status via hardware if job file not found.
        else:
            # job exists
            if job_name in self.data[module]:

                current = self._get_job_status(job_id, hardware=hardware)

                # No current status and job was not successful: failed!
                if current is None and previous != 'successful':
                    self.data[module][job_name]['job_status'] = 'failed'

                # do not overwrite a successful or failed job status.
                elif (current != previous and
                      previous not in self.FROZEN_STATUS):
                    self.data[module][job_name]['job_status'] = current

            # job does not yet exist
            else:
                self.data[module][job_name] = {}

    def _set_job_status(self, module, job_name, status):
        """Set an updated job status to the object instance.

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

        self.data[module][job_name]['job_status'] = status

    @staticmethod
    def update_dict(d, u):
        """Update a dictionary recursively.

        Parameters
        ----------
        d : dict
            Base dictionary to update.
        u : dict
            New dictionary with data to add to d.

        Returns
        -------
        d : dict
            d with data updated from u.
        """

        d = copy.deepcopy(d)

        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = Status.update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    @staticmethod
    def make_job_file(path, module, job_name, attrs):
        """Make a json file recording the status of a single job.

        Parameters
        ----------
        path : str
            Path to json status file.
        module : str
            reV module that the job belongs to.
        job_name : str
            Unique job name identification.
        attrs : str
            Dictionary of job attributes that represent the job status
            attributes.
        """
        if job_name.endswith('.h5'):
            job_name = job_name.replace('.h5', '')
        status = {module: {job_name: attrs}}
        fpath = os.path.join(path, 'status_{}.json'.format(job_name))
        with open(fpath, 'w') as f:
            json.dump(status, f, sort_keys=True, indent=4,
                      separators=(',', ': '))

    @classmethod
    def add_job(cls, path, module, job_name, replace=False, job_attrs=None):
        """Add a job to status json.

        Parameters
        ----------
        path : str
            Path to json status file.
        module : str
            reV module that the job belongs to.
        job_name : str
            Unique job name identification.
        replace : bool
            Flag to force replacement of pre-existing job status.
        job_attrs : dict
            Job attributes. Should include 'job_id' if running on HPC.
        """
        if job_name.endswith('.h5'):
            job_name = job_name.replace('.h5', '')

        obj = cls(path)

        if job_attrs is None:
            job_attrs = {}

        if 'hardware' in job_attrs:
            if job_attrs['hardware'] in ('eagle', 'peregrine'):
                if 'job_id' not in job_attrs:
                    warn('Key "job_id" should be in kwargs for "{}" if '
                         'adding job from an eagle or peregrine node.'
                         .format(job_name))

        # check to see if job exists yet
        exists = obj.job_exists(path, job_name)

        # job exists and user has requested forced replacement
        if replace and exists:
            del obj.data[module][job_name]

        # new job attribute data will be written if either:
        #  A) the user requested forced replacement or
        #  B) if the job does not exist
        if replace or not exists:

            if module not in obj.data:
                obj.data[module] = {job_name: job_attrs}
            else:
                obj.data[module][job_name] = job_attrs

            if 'job_status' not in job_attrs:
                obj.data[module][job_name]['job_status'] = 'submitted'

            obj._dump()

    @classmethod
    def job_exists(cls, path, job_name):
        """Check whether a job exists and return a bool.

        Parameters
        ----------
        path : str
            Path to json status file.
        job_name : str
            Unique job name identification.

        Returns
        -------
        exists : bool
            True if the job exists in the status json.
        """
        if job_name.endswith('.h5'):
            job_name = job_name.replace('.h5', '')

        obj = cls(path)
        if obj.data:
            for jobs in obj.data.values():
                if jobs:
                    for name in jobs.keys():
                        if name == job_name:
                            return True

        return False

    @classmethod
    def retrieve_job_status(cls, path, module, job_name):
        """Update and retrieve job status.

        Parameters
        ----------
        path : str
            Path to json status file.
        module : str
            reV module that the job belongs to.
        job_name : str
            Unique job name identification.

        Returns
        -------
        status : str | None
            Status string or None if job/module not found.
        """
        if job_name.endswith('.h5'):
            job_name = job_name.replace('.h5', '')

        obj = cls(path)
        obj._update_job_status(module, job_name)

        status = obj.data[module][job_name].get('job_status', None)

        return status

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
        if job_name.endswith('.h5'):
            job_name = job_name.replace('.h5', '')

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
        obj._check_all_job_files(path)
        obj._dump()
