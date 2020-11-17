# -*- coding: utf-8 -*-
"""
reV file multi-year config
"""
import logging
import os

from reV.config.base_analysis_config import AnalysisConfig
from reV.config.output_request import SAMOutputRequest
from reV.pipeline.pipeline import Pipeline
from reV.utilities.exceptions import ConfigError

logger = logging.getLogger(__name__)


class MultiYearConfig(AnalysisConfig):
    """File collection config."""

    NAME = 'multi-year'

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        super().__init__(config)
        self._groups = None

    @property
    def my_file(self):
        """
        Returns
        -------
        my_file : str
            MultiYear output .h5 file path
        """
        my_file = os.path.join(self.dirout, self.name + ".h5")
        return my_file

    @property
    def group_names(self):
        """
        Returns
        -------s
        group_names : list
            List of group names
        """
        if self._groups is None:
            self._groups = MultiYearGroup.factory(self.dirout, self['groups'])

        return list(self._groups)

    @property
    def group_params(self):
        """
        Returns
        -------
        group_params : dict
            Dictionary of group parameters: name, source_files, dsets
        """
        group_params = {}
        for name in self.group_names:
            group = self._groups[name]
            group_params[name] = {'group': group.name,
                                  'dsets': group.dsets,
                                  'source_files': group.source_files}

        return group_params


class MultiYearGroup:
    """
    Handle group parameters for MultiYearConfig
    """
    def __init__(self, name, out_dir, source_files="PIPELINE",
                 source_dir=None, source_prefix=None,
                 dsets=('cf_mean',)):
        """
        Parameters
        ----------
        name : str
            Group name
        out_dir : str
            Output directory, used for Pipeline handling
        source_files : str | list | NoneType
            List of source files
            If "PIPELINE" extract from collection status file
            If None, use source_dir and source_prefix
        source_dir : str | NoneType
            Directory to extract source files from
        source_prefix : str | NoneType
            File prefix to search for in source directory
        dsets : list | tuple
            List of datasets to collect
        """
        self._name = name
        self._dirout = out_dir
        self._source_files = source_files
        self._source_dir = source_dir
        self._source_prefix = source_prefix
        self._dsets = SAMOutputRequest(dsets)

    @property
    def name(self):
        """
        Returns
        -------
        name : str
            Group name
        """
        name = self._name if self._name.lower() != "none" else None
        return name

    @property
    def source_files(self):
        """
        Returns
        -------
        source_files : list
            list of source files to collect from
        """
        if self._source_files is not None:
            if isinstance(self._source_files, (list, tuple)):
                source_files = self._source_files
            elif self._source_files == "PIPELINE":
                source_files = Pipeline.parse_previous(self._dirout,
                                                       'multi-year',
                                                       target='fpath')
            else:
                raise ConfigError("source_files must be a list, tuple, "
                                  "or 'PIPELINE'")
        else:
            if self._source_dir and self._source_prefix:
                source_files = []
                for file in os.listdir(self._source_dir):
                    if (file.startswith(self._source_prefix)
                            and file.endswith('.h5') and '_node' not in file):
                        source_files.append(os.path.join(self._source_dir,
                                                         file))
            else:
                raise ConfigError("source_files or both source_dir and "
                                  "source_prefix must be provided")

        if not any(source_files):
            raise FileNotFoundError('Could not find any source files for '
                                    'multi-year collection group: "{}"'
                                    .format(self.name))

        return source_files

    @property
    def dsets(self):
        """
        Returns
        -------
        _dsets :list | tuple
            Datasets to collect
        """
        return self._dsets

    @classmethod
    def factory(cls, out_dir, groups_dict):
        """
        Generate dictionary of MultiYearGroup objects for all groups in groups

        Parameters
        ----------
        out_dir : str
            Output directory, used for Pipeline handling
        groups_dict : dict
            Dictionary of group parameters, parsed from multi-year config file

        Returns
        -------
        groups : dict
            Dictionary of MultiYearGroup objects for each group in groups
        """
        groups = {}
        for name, kwargs in groups_dict.items():
            groups[name] = cls(name, out_dir, **kwargs)

        return groups
