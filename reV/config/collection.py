# -*- coding: utf-8 -*-
"""
reV file collection config

Created on Mon Jan 28 11:43:27 2019

@author: gbuster
"""
import logging

from reV.config.base_analysis_config import AnalysisConfig
from reV.config.output_request import SAMOutputRequest
from reV.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


class CollectionConfig(AnalysisConfig):
    """File collection config."""

    NAME = 'collect'
    REQUIREMENTS = ('dsets', 'file_prefixes')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        super().__init__(config)

        self._purge = False
        self._dsets = None
        self._file_prefixes = None
        self._ec = None
        self._coldir = self.dirout

    @property
    def coldir(self):
        """Get the directory to collect files from.

        Returns
        -------
        coldir : str
            Target path to collect h5 files from.
        """
        self._coldir = self['directories'].get('collect_directory',
                                               self._coldir)

        if self._coldir == 'PIPELINE':
            self._coldir = Pipeline.parse_previous(self.dirout, 'collect',
                                                   target='dirout')[0]
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
    def purge_chunks(self):
        """Get the flag to delete chunk files. Default is False which just
        moves chunk files to a sub dir.

        Returns
        -------
        purge : bool
            Flag to delete chunk files. Default is False which just
            moves chunk files to a sub dir.
        """
        self._purge = self.get('purge_chunks', self._purge)
        return self._purge

    @property
    def dsets(self):
        """Get dset names to collect.

        Returns
        -------
        dsets : list
            list of dset names to collect.
        """

        if self._dsets is None:
            self._dsets = SAMOutputRequest(self['dsets'])
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
        file_prefixes : list
            list of file prefixes to collect.
        """

        if self._file_prefixes is None:
            self._file_prefixes = self['file_prefixes']

            if 'PIPELINE' in self._file_prefixes:
                self._file_prefixes = self._parse_pipeline_prefixes()

            if isinstance(self._file_prefixes, str):
                self._file_prefixes = [self._file_prefixes]
            else:
                self._file_prefixes = list(self._file_prefixes)

        return self._file_prefixes
