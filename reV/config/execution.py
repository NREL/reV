# -*- coding: utf-8 -*-
"""
reV Configuration for Execution Options
"""
import logging

from reV.config.base_config import BaseConfig
from reV.utilities.exceptions import ConfigError

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

        self._default_option = 'local'
        self._default_nodes = 1
        self._default_mem_util_lim = 0.4
        super().__init__(config_dict)

    @property
    def option(self):
        """Get the hardware run option.

        Returns
        -------
        option : str
            Execution control option, e.g. local, peregrine, eagle...
        """
        return str(self.get('option', self._default_option)).lower()

    @property
    def nodes(self):
        """Get the number of nodes property.

        Returns
        -------
        nodes : int
            Number of available nodes. Default is 1 node.
        """
        return int(self.get('nodes', self._default_nodes))

    @property
    def max_workers(self):
        """Get the max_workers property (1 runs in serial, None is all workers)

        Returns
        -------
        max_workers : int | None
            Processes per node. Default is None max_workers (all available).
        """
        return self.get('max_workers', None)

    @property
    def sites_per_worker(self):
        """Get the number of sites to run per worker.

        Returns
        -------
        sites_per_worker : int | None
            Number of sites to run per worker in a parallel scheme.
        """
        return self.get('sites_per_worker', None)

    @property
    def memory_utilization_limit(self):
        """Get the node memory utilization limit property. Key in the config
        json is "memory_utilization_limit".

        Returns
        -------
        mem_util_lim : float
            Memory utilization limit (fractional). Key in the config json is
            "memory_utilization_limit".
        """
        mem_util_lim = self.get('memory_utilization_limit',
                                self._default_mem_util_lim)

        return mem_util_lim

    @property
    def sh_script(self):
        """Get the "sh_script" entry which is a string that contains extra
        shell script commands to run before the reV commands.

        Returns
        -------
        str
        """
        return self.get('sh_script', '')


class HPCConfig(BaseExecutionConfig):
    """Class to handle HPC configuration inputs."""

    @property
    def allocation(self):
        """Get the HPC allocation property.

        Returns
        -------
        hpc_alloc : str
            Name of the HPC allocation account for the specified job.
        """

        return self.get('allocation', None)

    @property
    def feature(self):
        """Get feature request str.

        Returns
        -------
        feature : str | NoneType
            Feature request string. For EAGLE, a full additional flag.
            Config should look like:
            ``"feature": "--depend=[state:job_id]"``
        """
        return self.get('feature', None)

    @property
    def module(self):
        """
        Get module to load if given

        Returns
        -------
        module : str
            Module to load on node
        """
        return self.get('module', None)

    @property
    def conda_env(self):
        """
        Get conda environment to activate

        Returns
        -------
        conda_env : str
            Conda environment to activate
        """
        return self.get('conda_env', None)


class SlurmConfig(HPCConfig):
    """Class to handle SLURM (Eagle) configuration inputs."""

    def _preflight(self):
        """Run a preflight check on the config."""
        if self.option in {'eagle', 'kestrel', 'slurm'}:
            if self.allocation is None:
                msg = 'HPC execution config must have an "allocation" input'
                logger.error(msg)
                raise ConfigError(msg)
            if self.walltime is None:
                msg = 'HPC execution config must have a "walltime" input'
                logger.error(msg)
                raise ConfigError(msg)

        super()._preflight()

    @property
    def memory(self):
        """Get the requested Eagle node "memory" value in GB or can be None.

        Returns
        -------
        _hpc_node_mem : int | None
            Requested node memory in GB.
        """
        return self.get('memory', None)

    @property
    def walltime(self):
        """Get the requested Eagle node "walltime" value.

        Returns
        -------
        _hpc_walltime : int
            Requested single node job time in hours.
        """
        return self.get('walltime', None)
