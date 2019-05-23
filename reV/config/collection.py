# -*- coding: utf-8 -*-
"""
reV analysis configs (generation, lcoe, etc...)

Created on Mon Jan 28 11:43:27 2019

@author: gbuster
"""
import logging

from reV.config.analysis_configs import AnalysisConfig


logger = logging.getLogger(__name__)


class CollectionConfig(AnalysisConfig):
    """Base analysis config (generation, lcoe, etc...)."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        self._parallel = False
        self._dsets = None
        self._file_prefixes = None
        self._ec = None
        super().__init__(config)

    @property
    def coldir(self):
        """Get the directory to collect files from.

        Returns
        -------
        _coldir : str
            Target path to collect h5 files from.
        """
        return self['directories']['collect_directory']

    @property
    def project_points(self):
        """Get the collection project points.

        Returns
        -------
        _project_points : str
            Target path for project points file.
        """
        return self['project_points']

    @property
    def parallel(self):
        """Get the flag to do a parallel collection.

        Returns
        -------
        _parallel : bool
            Flag to collect data in parallel.
        """
        if 'parallel' in self['project_control']:
            self._parallel = self['project_control']['parallel']
        return self._parallel

    @property
    def dsets(self):
        """Get dset names to collect.

        Returns
        -------
        _dsets : list
            list of dset names to collect.
        """

        if self._dsets is None:
            self._dsets = self['project_control']['dsets']
            if not isinstance(self._dsets, list):
                self._dsets = list(self._dsets)
        return self._dsets

    @property
    def file_prefixes(self):
        """Get the file prefixes to collect.

        Returns
        -------
        _file_prefixes : list
            list of file prefixes to collect.
        """

        if self._file_prefixes is None:
            self._file_prefixes = self['project_control']['file_prefixes']
            if not isinstance(self._file_prefixes, list):
                self._file_prefixes = list(self._file_prefixes)
        return self._file_prefixes
