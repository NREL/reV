# -*- coding: utf-8 -*-
"""
reV analysis configs (generation, lcoe, etc...)

Created on Mon Jan 28 11:43:27 2019

@author: gbuster
"""
import logging

from reV.pipeline.pipeline import Pipeline
from reV.config.base_analysis_config import AnalysisConfig


logger = logging.getLogger(__name__)


class CollectionConfig(AnalysisConfig):
    """Base analysis config (generation, lcoe, etc...)."""

    def __init__(self, config):
<<<<<<< HEAD
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
=======
        self._coldir = None
>>>>>>> minor refactor of pipeline framework so that all pipeline parsing is done in the config. Additions/bug fixes:
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
        if self._coldir is None:
            self._coldir = self['directories']['collect_directory']
            if 'PIPELINE' in self._coldir:
                dirs = Pipeline.parse_previous(self.dirout, 'collect',
                                               target='dirout')
                self._coldir = dirs[0]
        return self._coldir

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

    def _parse_pipeline_prefixes(self):
        """Parse reV pipeline for file prefixes from previous module."""
        files = Pipeline.parse_previous(self.dirout, 'collect',
                                        target='fout')
        for i, fname in enumerate(files):
            files[i] = '_'.join([c for c in fname.split('_')
                                 if '.h5' not in c and 'node' not in c])
        file_prefixes = list(set(files))
        return file_prefixes

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
            if 'PIPELINE' in self._file_prefixes:
                self._file_prefixes = self._parse_pipeline_prefixes()
            if not isinstance(self._file_prefixes, list):
                self._file_prefixes = list(self._file_prefixes)
        return self._file_prefixes
