# -*- coding: utf-8 -*-
"""
reV Configuration for Execution Options
"""
import logging

from reV.config.base_config import BaseConfig


logger = logging.getLogger(__name__)


class BaseExecutionConfig(BaseConfig):
    """Base class to handle execution configuration"""

    def __init__(self, config_dict):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        self._option = None
        self._nodes = None
        self._max_workers = None
        self._sites_per_worker = None
        self._mem_util_lim = None
        super().__init__(config_dict)

    @property
    def option(self):
        """Get the hardware run option.

        Returns
        -------
        _option : str
            Execution control option, e.g. local, peregrine, eagle...
        """

        if self._option is None:
            # default option if not specified
            self._option = 'local'
            if 'option' in self:
                if self['option']:
                    # config-specified, set to attribute
                    self._option = self['option'].lower()
        return self._option

    @property
    def nodes(self):
        """Get the number of nodes property.

        Returns
        -------
        _nodes : int
            Number of available nodes. Default is 1 node.
        """
        if self._nodes is None:
            # set default option if not specified
            self._nodes = 1
            if 'nodes' in self:
                self._nodes = self['nodes']
        return self._nodes

    @property
    def max_workers(self):
        """Get the process per node (max_workers) property.

        Returns
        -------
        _max_workers : int
            Processes per node. Default is 1 max_workers.
        """
        if self._max_workers is None:
            # set default option if not specified
            self._max_workers = 1
            if 'max_workers' in self:
                self._max_workers = self['max_workers']
        return self._max_workers

    @property
    def sites_per_worker(self):
        """Get the number of sites to run per worker.

        Returns
        -------
        _sites_per_worker : int | None
            Number of sites to run per worker in a parallel scheme.
        """
        if self._sites_per_worker is None:
            if 'sites_per_worker' in self:
                self._sites_per_worker = self['sites_per_worker']
        return self._sites_per_worker

    @property
    def mem_util_lim(self):
        """Get the node memory utilization limit property.

        Returns
        -------
        _mem_util_lim : float
            Memory utilization limit (fractional). Key in the config json is
            "memory_utilization_limit".
        """

        if self._mem_util_lim is None:
            if 'memory_utilization_limit' in self:
                self._mem_util_lim = self['memory_utilization_limit']
            else:
                self._mem_util_lim = 0.4
        return self._mem_util_lim


class HPCConfig(BaseExecutionConfig):
    """Class to handle HPC configuration inputs."""

    def __init__(self, config_dict):
        self._hpc_alloc = None
        self._feature = None
        super().__init__(config_dict)

    @property
    def alloc(self):
        """Get the HPC allocation property.

        Returns
        -------
        _hpc_alloc : str
            Name of the HPC allocation account for the specified job.
        """
        if self._hpc_alloc is None:
            # default option if not specified
            self._hpc_alloc = 'rev'
            if 'allocation' in self:
                if self['allocation']:
                    # config-specified, set to attribute
                    self._hpc_alloc = self['allocation']
        return self._hpc_alloc

    @property
    def feature(self):
        """Get feature request str.

        Returns
        -------
        _feature : str | NoneType
            Feature request string.

            For EAGLE, a full additional flag.
            Config should look like:
                "feature": "--qos=high"
                "feature": "--depend=[state:job_id]"

            For PEREGRINE, everything following the -l flag.
            Config should look like:
                "feature": "qos=high"
                "feature": "feature=256GB"
        """
        if self._feature is None:
            # default option if not specified
            if 'feature' in self:
                if self['feature']:
                    # config-specified, set to attribute
                    self._feature = self['feature']
        return self._feature


class PeregrineConfig(HPCConfig):
    """Class to handle Peregrine configuration inputs."""

    def __init__(self, config_dict):
        self._hpc_node_mem = None
        self._hpc_walltime = None
        self._hpc_queue = None
        super().__init__(config_dict)

    @property
    def node_mem(self):
        """Get the Peregrine node memory property.

        Returns
        -------
        _hpc_node_mem : str
            Single node memory request, e.g. 32GB, 64GB, etc...
        """
        defaults = {'short': '32GB',
                    'debug': '32GB',
                    'batch': '32GB',
                    'batch-h': '64GB',
                    'long': '32GB',
                    'bigmem': '64GB',
                    'data-transfer': '32GB',
                    }
        if self._hpc_node_mem is None:
            # default option if not specified
            self._hpc_node_mem = defaults[self.queue]
            if 'memory' in self:
                if self['memory']:
                    # config-specified, set to attribute
                    self._hpc_node_mem = self['memory']
        return self._hpc_node_mem

    @property
    def walltime(self):
        """Get the Peregrine node walltime property.

        Returns
        -------
        _hpc_walltime : str
            Single node job time request, e.g. '04:00:00'.
        """
        defaults = {'short': '04:00:00',
                    'debug': '01:00:00',
                    'batch': '48:00:00',
                    'batch-h': '48:00:00',
                    'long': '240:00:00',
                    'bigmem': '240:00:00',
                    'data-transfer': '120:00:00',
                    }
        if self._hpc_walltime is None:
            # default option if not specified
            self._hpc_walltime = defaults[self.queue]
            if 'walltime' in self:
                if self['walltime']:
                    # config-specified, set to attribute
                    self._hpc_walltime = self['walltime']
        return self._hpc_walltime

    @property
    def queue(self):
        """Get the Peregrine queue property.

        Returns
        -------
        _hpc_queue : str
            Peregrine queue request, e.g. 'short' or 'long'.
        """
        if self._hpc_queue is None:
            # default option if not specified
            self._hpc_queue = 'short'
            if 'queue' in self:
                if self['queue']:
                    # config-specified, set to attribute
                    self._hpc_queue = self['queue']
        return self._hpc_queue


class EagleConfig(HPCConfig):
    """Class to handle Eagle configuration inputs."""

    def __init__(self, config_dict):
        self._hpc_node_mem = None
        self._hpc_walltime = None
        super().__init__(config_dict)

    @property
    def node_mem(self):
        """Get the requested Eagle node memory property.

        Returns
        -------
        _hpc_node_mem : int
            Requested node memory in GB.
        """
        if self._hpc_node_mem is None:
            if 'memory' in self:
                if self['memory']:
                    # config-specified, set to attribute
                    self._hpc_node_mem = self['memory']
        return self._hpc_node_mem

    @property
    def walltime(self):
        """Get the requested Eagle node walltime property.

        Returns
        -------
        _hpc_walltime : int
            Requested single node job time in hours.
        """
        if self._hpc_walltime is None:
            # default option if not specified
            self._hpc_walltime = 1
            if 'walltime' in self:
                if self['walltime']:
                    # config-specified, set to attribute
                    self._hpc_walltime = self['walltime']
        return self._hpc_walltime
