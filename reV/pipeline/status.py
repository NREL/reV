# -*- coding: utf-8 -*-
"""
reV job status manager.
"""
import copy
import os
import json
import logging
import time
from warnings import warn
import shutil

from reV.utilities import safe_json_load
from reV.utilities.execution import SLURM, PBS


logger = logging.getLogger(__name__)


class Status(dict):
    """Base class for reV data pipeline health and status information."""

    FROZEN_STATUS = ('successful', 'failed')

    def __init__(self, status_dir, name=None):
        """
        Parameters
        ----------
        status_dir : str
            Directory to with json status file.
        name : str | None
            Optional job name for status. Will look for the file
            "{name}_status.json" in the status_dir.
        """

        self._status_dir = status_dir
        self._fpath = self._parse_fpath(status_dir, name)
        self.data = self._load(self._fpath)

    @staticmethod
    def _parse_fpath(status_dir, name):
        """Get the status filepath from the status directory and jobname.

        Parameters
        ----------
        status_dir : str
            Directory to with json status file.
        name : str | None
            Optional job name for status. Will look for the file
            "{name}_status.json" in the status_dir.

        Returns
        -------
        fpath : str
            Filepath to job status json.
        """

        if str(status_dir).endswith('.json'):
            raise TypeError('Need a directory containing a status json, '
                            'not a status json: {}'.format(status_dir))

        if name is None:
            fpath = os.path.join(status_dir, 'rev_status.json')
            for fn in os.listdir(status_dir):
                if fn.endswith('_status.json'):
                    fpath = os.path.join(status_dir, fn)
                    break
        else:
            fpath = os.path.join(status_dir, '{}_status.json'.format(name))

        return fpath

    @staticmethod
    def _load(fpath):
        """Load status json.

        Parameters
        -------
        fpath : str
            Filepath to job status json.

        Returns
        -------
        data : dict
            JSON file contents loaded as a python dictionary.
        """
        if os.path.isfile(fpath):
            data = safe_json_load(fpath)
        else:
            data = {}
        return data

    def _dump(self):
        """Dump status json w/ backup file in case process gets killed."""
        backup = self._fpath.replace('.json', '_backup.json')
        self._sort_by_index()
        if os.path.exists(self._fpath):
            shutil.copy(self._fpath, backup)
        with open(self._fpath, 'w') as f:
            json.dump(self.data, f, indent=4, separators=(',', ': '))
        if os.path.exists(backup):
            os.remove(backup)

    def _sort_by_index(self):
        """Sort modules in data dictionary by pipeline index."""

        sortable = True

        for value in self.data.values():
            if 'pipeline_index' not in value:
                sortable = False
                break

        if sortable:
            sorted_keys = sorted(self.data, key=lambda x:
                                 self.data[x]['pipeline_index'])
            self.data = {k: self.data[k] for k in sorted_keys}

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
                   'peregrine': PBS.check_status,
                   'local': None}
        try:
            method = options[hardware]
        except KeyError:
            raise KeyError('Could not check job on the requested hardware: '
                           '"{}".'.format(hardware))
        return method

    @staticmethod
    def _get_job_status(job_id, hardware='local'):
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
            if method is None:
                status = None
            else:
                status = method(job_id)
        return status

    def _check_all_job_files(self, status_dir):
        """Look for all single-job job status files in the target status_dir
        and update status.

        Parameters
        ----------
        status_dir : str
            Directory to look for completion file.
        """

        for fname in os.listdir(status_dir):
            if fname.startswith('jobstatus_') and fname.endswith('.json'):
                # wait one second to make sure file is finished being written
                time.sleep(0.01)
                status = safe_json_load(os.path.join(status_dir, fname))
                self.data = self.update_dict(self.data, status)
                os.remove(os.path.join(status_dir, fname))

    @staticmethod
    def _check_job_file(status_dir, job_name):
        """Look for a single-job job status file in the target status_dir.

        Parameters
        ----------
        status_dir : str
            Directory to look for completion file.
        job_name : str
            Job name.

        Returns
        -------
        status : dict | None
            Job status dictionary if completion file found.
        """
        status = None
        target_fname = 'jobstatus_{}.json'.format(job_name)
        for fname in os.listdir(status_dir):
            if fname == target_fname:
                # wait one second to make sure file is finished being written
                time.sleep(0.01)
                status = safe_json_load(os.path.join(status_dir, fname))
                os.remove(os.path.join(status_dir, fname))
                break
        return status

    def _update_job_status(self, module, job_name, hardware='local'):
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

        # look for completion file.
        current = self._check_job_file(self._status_dir, job_name)

        # Update status data dict recursively if job file was found
        if current is not None:
            self.data = Status.update_dict(self.data, current)

        # check job status via hardware if job file not found.
        elif module in self.data:
            # job exists
            if job_name in self.data[module]:

                # init defaults in case job/module not in status file yet
                previous = self.data[module][job_name].get('job_status', None)
                job_id = self.data[module][job_name].get('job_id', None)
                hardware = self.data[module][job_name].get('hardware',
                                                           hardware)

                # get job status from hardware
                current = self._get_job_status(job_id, hardware=hardware)

                # No current status and job was not successful: failed!
                if current is None and previous != 'successful':
                    self.data[module][job_name]['job_status'] = 'failed'

                # do not overwrite a successful or failed job status.
                elif (current != previous
                      and previous not in self.FROZEN_STATUS):
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
    def _get_attr_list(inp, key='job_id'):
        """Get all job attribute values from the status data dict.

        Parameters
        ----------
        inp : dict
            Job status dictionary.
        key : str
            Key to get values for.

        Returns
        -------
        out : list
            List of values corresponding to the input key for all jobs in inp.
        """

        out = []

        if isinstance(inp, dict):
            if key in inp:
                out = inp[key]
            else:
                for v in inp.values():
                    temp = Status._get_attr_list(v, key=key)

                    if isinstance(temp, list):
                        if any(temp):
                            out += temp
                    elif isinstance(temp, (int, str)):
                        out.append(temp)
        return out

    @property
    def job_ids(self):
        """Get list of job ids."""
        return self._get_attr_list(self.data, key='job_id')

    @property
    def hardware(self):
        """Get list of job hardware."""
        return self._get_attr_list(self.data, key='hardware')

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
    def make_job_file(status_dir, module, job_name, attrs):
        """Make a json file recording the status of a single job.

        Parameters
        ----------
        status_dir : str
            Directory to put json status file.
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
        fpath = os.path.join(status_dir, 'jobstatus_{}.json'.format(job_name))
        with open(fpath, 'w') as f:
            json.dump(status, f, sort_keys=True, indent=4,
                      separators=(',', ': '))

    @classmethod
    def add_job(cls, status_dir, module, job_name, replace=False,
                job_attrs=None):
        """Add a job to status json.

        Parameters
        ----------
        status_dir : str
            Directory containing json status file.
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

        obj = cls(status_dir)

        if job_attrs is None:
            job_attrs = {}

        if 'hardware' in job_attrs:
            if job_attrs['hardware'] in ('eagle', 'peregrine'):
                if 'job_id' not in job_attrs:
                    warn('Key "job_id" should be in kwargs for "{}" if '
                         'adding job from an eagle or peregrine node.'
                         .format(job_name))

        # check to see if job exists yet
        exists = obj.job_exists(status_dir, job_name)

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
    def job_exists(cls, status_dir, job_name):
        """Check whether a job exists and return a bool.

        Parameters
        ----------
        status_dir : str
            Directory containing json status file.
        job_name : str
            Unique job name identification.

        Returns
        -------
        exists : bool
            True if the job exists in the status json.
        """
        if job_name.endswith('.h5'):
            job_name = job_name.replace('.h5', '')

        obj = cls(status_dir)
        exists = False
        if obj.data:
            for jobs in obj.data.values():
                if jobs:
                    for name in jobs.keys():
                        if name == job_name:
                            exists = True
                            break

        return exists

    @classmethod
    def retrieve_job_status(cls, status_dir, module, job_name):
        """Update and retrieve job status.

        Parameters
        ----------
        status_dir : str
            Directory containing json status file.
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

        obj = cls(status_dir)
        obj._update_job_status(module, job_name)

        try:
            status = obj.data[module][job_name].get('job_status', None)
        except KeyError:
            status = None

        return status

    @classmethod
    def set_job_status(cls, status_dir, module, job_name, status):
        """Force set a job status to a frozen status and save to status file.

        Parameters
        ----------
        status_dir : str
            Directory containing json status file.
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

        obj = cls(status_dir)
        obj._set_job_status(module, job_name, status)
        obj._dump()

    @classmethod
    def update(cls, status_dir):
        """Update all job statuses and dump to json.

        Parameters
        ----------
        status_dir : str
            Directory containing json status file.
        """

        obj = cls(status_dir)
        for module in obj.data.keys():
            for job_name in obj.data[module].keys():
                if job_name != 'pipeline_index':
                    obj._update_job_status(module, job_name)
        obj._check_all_job_files(status_dir)
        obj._dump()
