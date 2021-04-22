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

from rex.utilities.utilities import get_class_properties

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
    def groups(self):
        """Get the multi year collection groups

        Returns
        -------
        dict
        """
        return self._groups

    @property
    def group_names(self):
        """
        Returns
        -------
        group_names : list
            List of group names
        """
        if self._groups is None:
            self._groups = MultiYearGroup._factory(self.dirout, self['groups'])

        return list(self._groups)

    @property
    def group_params(self):
        """Dictionary of all groups and their respective parameters:
        {group_name1: {group: None, source_files: [], dsets: []}}

        Returns
        -------
        dict
        """
        group_params = {}
        for name in self.group_names:
            group = self._groups[name]
            group_params[name] = group._dict_rep()

        return group_params


class MultiYearGroup:
    """
    Handle group parameters for MultiYearConfig
    """
    def __init__(self, name, out_dir, source_files=None,
                 source_dir=None, source_prefix=None,
                 dsets=('cf_mean',), pass_through_dsets=None):
        """
        Parameters
        ----------
        name : str
            Group name
        out_dir : str
            Output directory, used for Pipeline handling
        source_files : str | list | NoneType
            Explicit list of source files - either use this OR
            source_dir + source_prefix
            If this arg is "PIPELINE", determine source_files from
            the status file of the previous pipeline step.
            If None, use source_dir and source_prefix
        source_dir : str | NoneType
            Directory to extract source files from
            (must be paired with source_prefix)
        source_prefix : str | NoneType
            File prefix to search for in source directory
            (must be paired with source_dir)
        dsets : list | tuple
            List of datasets to collect
        pass_through_dsets : list | tuple | None
            Optional list of datasets that are identical in the multi-year
            files (e.g. input datasets that don't vary from year to year) that
            should be copied to the output multi-year file once without a
            year suffix or means/stdev calculation
        """
        self._name = name
        self._dirout = out_dir
        self._source_files = source_files
        self._source_dir = source_dir
        self._source_prefix = source_prefix
        self._dsets = SAMOutputRequest(dsets)
        self._pass_through_dsets = None
        if pass_through_dsets is not None:
            self._pass_through_dsets = SAMOutputRequest(pass_through_dsets)

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
                e = "source_files must be a list, tuple, or 'PIPELINE'"
                logger.error(e)
                raise ConfigError(e)
        else:
            if self._source_dir and self._source_prefix:
                source_files = []
                for file in os.listdir(self._source_dir):
                    if (file.startswith(self._source_prefix)
                            and file.endswith('.h5') and '_node' not in file):
                        source_files.append(os.path.join(self._source_dir,
                                                         file))
            else:
                e = ("source_files or both source_dir and "
                     "source_prefix must be provided")
                logger.error(e)
                raise ConfigError(e)

        if not any(source_files):
            e = ('Could not find any source files for '
                 'multi-year collection group: "{}"'
                 .format(self.name))
            logger.error(e)
            raise FileNotFoundError(e)

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

    @property
    def pass_through_dsets(self):
        """Optional list of datasets that are identical in the multi-year
        files (e.g. input datasets that don't vary from year to year) that
        should be copied to the output multi-year file once without a
        year suffix or means/stdev calculation

        Returns
        -------
        list | tuple | None
        """
        return self._pass_through_dsets

    def _dict_rep(self):
        """Get a dictionary representation of this multi year collection group

        Returns
        -------
        dict
        """
        props = get_class_properties(self.__class__)
        out = {k: getattr(self, k) for k in props}
        out['group'] = self.name
        return out

    @classmethod
    def _factory(cls, out_dir, groups_dict):
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
