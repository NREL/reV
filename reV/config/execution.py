"""
reV Configuration
"""
import logging

from reV.config.base_config import BaseConfig


logger = logging.getLogger(__name__)


class BaseExecutionConfig(BaseConfig):
    """Base class to handle execution configuration"""

    def __init__(self, config_dict):
        super().__init__(config_dict)

    @property
    def option(self):
        """Get the hardware run option.

        Available options:
            - local
            - peregrine
        """

        default = 'local'
        if not hasattr(self, '_option'):
            if 'option' in self:
                if self['option']:
                    self._option = self['option'].lower()
                else:
                    # default option if not specifiedserial
                    self._option = default
            else:
                # default option if not specified
                self._option = default

        return self._option

    @property
    def nodes(self):
        """Get the number of nodes property. Default is 1 node."""
        if not hasattr(self, '_nodes'):
            if 'nodes' in self:
                self._nodes = self['nodes']
            else:
                self._nodes = 1
        return self._nodes

    @property
    def ppn(self):
        """Get the process per node (ppn) property. Default is 1 ppn."""
        if not hasattr(self, '_ppn'):
            if 'ppn' in self:
                self._ppn = self['ppn']
            else:
                self._ppn = 1
        return self._ppn


class HPCConfig(BaseExecutionConfig):
    """Class to handle HPC configuration inputs."""

    def __init__(self, config_dict):
        super().__init__(config_dict)

    @property
    def alloc(self):
        """Get the HPC allocation property."""
        default = 'rev'
        if not hasattr(self, '_hpc_alloc'):
            if 'allocation' in self:
                if self['allocation']:
                    self._hpc_alloc = self['allocation']
                else:
                    # default option if not specified
                    self._hpc_alloc = default
            else:
                # default option if not specified
                self._hpc_alloc = default

        return self._hpc_alloc


class PeregrineConfig(HPCConfig):
    """Class to handle Peregrine configuration inputs."""

    def __init__(self, config_dict):
        super().__init__(config_dict)

    @property
    def node_mem(self):
        """Get the HPC node memory property."""
        defaults = {'short': '32GB',
                    'debug': '32GB',
                    'batch': '32GB',
                    'batch-h': '64GB',
                    'long': '32GB',
                    'bigmem': '64GB',
                    'data-transfer': '32GB',
                    }
        if not hasattr(self, '_hpc_node_mem'):
            if 'memory' in self:
                if self['memory']:
                    self._hpc_node_mem = self['memory']
                else:
                    # default option if not specified
                    self._hpc_node_mem = defaults[self.queue]
            else:
                # default option if not specified
                self._hpc_node_mem = defaults[self.queue]

        return self._hpc_node_mem

    @property
    def walltime(self):
        """Get the HPC node walltime property."""
        defaults = {'short': '04:00:00',
                    'debug': '01:00:00',
                    'batch': '48:00:00',
                    'batch-h': '48:00:00',
                    'long': '240:00:00',
                    'bigmem': '240:00:00',
                    'data-transfer': '120:00:00',
                    }
        if not hasattr(self, '_hpc_walltime'):
            if 'walltime' in self:
                if self['walltime']:
                    self._hpc_walltime = self['walltime']
                else:
                    # default option if not specified
                    self._hpc_walltime = defaults[self.queue]
            else:
                # default option if not specified
                self._hpc_walltime = defaults[self.queue]

        return self._hpc_walltime

    @property
    def queue(self):
        """Get the Peregrine queue property."""
        default = 'short'
        if not hasattr(self, '_hpc_queue'):
            if 'queue' in self:
                if self['queue']:
                    self._hpc_queue = self['queue']
                else:
                    # default option if not specified
                    self._hpc_queue = default
            else:
                # default option if not specified
                self._hpc_queue = default

        return self._hpc_queue

    @property
    def feature(self):
        """Get feature request str. Cores or memory. Mem is prioritized."""
        if not hasattr(self, '_feature'):
            self._feature = None
            if 'memory' in self:
                if self['memory']:
                    self._feature = self['memory']
            elif 'ppn' in self:
                if self['ppn']:
                    self._feature = '{}core'.format(self['ppn'])
        return self._feature
