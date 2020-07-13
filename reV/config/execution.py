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
        super().__init__(config_dict)

        self._option = 'local'
        self._nodes = 1
        self._max_workers = None
        self._sites_per_worker = None
        self._mem_util_lim = 0.4

    @property
    def option(self):
        """Get the hardware run option.

        Returns
        -------
        option : str
            Execution control option, e.g. local, peregrine, eagle...
        """
        self._option = str(self.get('option', self._option)).lower()
        return self._option

    @property
    def nodes(self):
        """Get the number of nodes property.

        Returns
        -------
        nodes : int
            Number of available nodes. Default is 1 node.
        """
        self._nodes = int(self.get('nodes', self._nodes))
        return self._nodes

    @property
    def max_workers(self):
        """Get the max_workers property (1 runs in serial, None is all workers)

        Returns
        -------
        max_workers : int | None
            Processes per node. Default is None max_workers (all available).
        """
        self._max_workers = self.get('max_workers', self._max_workers)
        return self._max_workers

    @property
    def sites_per_worker(self):
        """Get the number of sites to run per worker.

        Returns
        -------
        sites_per_worker : int | None
            Number of sites to run per worker in a parallel scheme.
        """
        self._sites_per_worker = self.get('sites_per_worker',
                                          self._sites_per_worker)
        return self._sites_per_worker

    @property
    def mememory_utilization_limit(self):
        """Get the node memory utilization limit property. Key in the config
        json is "memory_utilization_limit".

        Returns
        -------
        mem_util_lim : float
            Memory utilization limit (fractional). Key in the config json is
            "memory_utilization_limit".
        """
        self._mem_util_lim = self.get('memory_utilization_limit',
                                      self._mem_util_lim)
        return self._mem_util_lim


class HPCConfig(BaseExecutionConfig):
    """Class to handle HPC configuration inputs."""

    def __init__(self, config_dict):
        """
        Parameters
        ----------
        config_dict : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """

        super().__init__(config_dict)

        self._hpc_alloc = 'rev'
        self._feature = None
        self._module = None
        self._conda_env = None

    @property
    def allocation(self):
        """Get the HPC allocation property.

        Returns
        -------
        hpc_alloc : str
            Name of the HPC allocation account for the specified job.
        """
        self._hpc_alloc = self.get('allocation', self._hpc_alloc)
        return self._hpc_alloc

    @property
    def feature(self):
        """Get feature request str.

        Returns
        -------
        feature : str | NoneType
            Feature request string.

            For EAGLE, a full additional flag.
            Config should look like:
                "feature": "--qos=high"
                "feature": "--depend=[state:job_id]"
        """
        self._feature = self.get('feature', self._feature)
        return self._feature

    @property
    def module(self):
        """
        Get module to load if given

        Returns
        -------
        module : str
            Module to load on node
        """
        self._module = self.get('module', self._module)
        return self._module

    @property
    def conda_env(self):
        """
        Get conda environment to activate

        Returns
        -------
        conda_env : str
            Conda environment to activate
        """
        self._conda_env = self.get('conda_env', self._conda_env)
        return self._conda_env


class SlurmConfig(HPCConfig):
    """Class to handle SLURM (Eagle) configuration inputs."""

    def __init__(self, config_dict):
        """
        Parameters
        ----------
        config_dict : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """

        super().__init__(config_dict)

        self._hpc_node_mem = None
        self._hpc_walltime = 1

    @property
    def memory(self):
        """Get the requested Eagle node "memory" value in GB or can be None.

        Returns
        -------
        _hpc_node_mem : int | None
            Requested node memory in GB.
        """
        self._hpc_node_mem = self.get('memory', self._hpc_node_mem)
        return self._hpc_node_mem

    @property
    def walltime(self):
        """Get the requested Eagle node "walltime" value.

        Returns
        -------
        _hpc_walltime : int
            Requested single node job time in hours.
        """
        self._hpc_walltime = int(self.get('walltime', self._hpc_walltime))
        return self._hpc_walltime
