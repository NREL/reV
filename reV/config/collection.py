# -*- coding: utf-8 -*-
"""
reV file collection config

Created on Mon Jan 28 11:43:27 2019

@author: gbuster
"""
import os
import logging

from rex import Resource

from reV.config.base_analysis_config import AnalysisConfig
from reV.config.output_request import SAMOutputRequest
from reV.pipeline.pipeline import Pipeline
from reV.utilities import ModuleName

logger = logging.getLogger(__name__)


class CollectionConfig(AnalysisConfig):
    """File collection config."""

    NAME = ModuleName.COLLECT
    REQUIREMENTS = ('dsets', 'collect_pattern')

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
        """Get dset names to collect. This can be set as "PIPELINE" in the
        config, which will pull all of the non-time-index and non-meta dataset
        names from one of the output files in the previous pipeline step.

        Returns
        -------
        dsets : list
            list of dset names to collect.
        """

        if self._dsets is None:
            self._dsets = self['dsets']
            if isinstance(self._dsets, str) and self._dsets == 'PIPELINE':
                files = Pipeline.parse_previous(self.dirout, 'collect',
                                                target='fout')
                with Resource(files[0]) as res:
                    self._dsets = [d for d in res
                                   if not d.startswith('time_index')
                                   and d != 'meta']

            self._dsets = SAMOutputRequest(self._dsets)

        return self._dsets

    def _parse_pipeline_prefixes(self):
        """Parse reV pipeline for file prefixes from previous module."""
        files = Pipeline.parse_previous(self.dirout, module=ModuleName.COLLECT,
                                        target='fout')
        for i, fname in enumerate(files):
            files[i] = '_'.join([c for c in fname.split('_')
                                 if '.h5' not in c and 'node' not in c])
        file_prefixes = list(set(files))
        return file_prefixes

    @property
    def fn_out_names(self):
        """Get a list of output filenames ordered to correspond to the list of
        collection patterns.

        This can also be set to PIPELINE or just not set at all if this is a
        pipeline job and collect_pattern="PIPELINE"
        """

        fn_out_names = self.get('fn_out_names', None)
        collect_pattern = self['collect_pattern']

        if (str(fn_out_names) == 'PIPELINE'
                or str(collect_pattern) == 'PIPELINE'):
            fn_out_names = self._parse_pipeline_prefixes()
            fn_out_names = [fn if fn.endswith('.h5') else fn + '.h5'
                            for fn in fn_out_names]

        elif fn_out_names is None and str(collect_pattern) != 'PIPELINE':
            fn_out_names = [os.path.basename(fp).replace('*', '')
                            for fp in self.collect_pattern]

        if isinstance(fn_out_names, str):
            fn_out_names = [fn_out_names]
        elif fn_out_names is None:
            msg = ('Failed to parse "fn_out_names" from collect config!')
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            fn_out_names = list(fn_out_names)

        return fn_out_names

    @property
    def collect_pattern(self):
        """Get a list of one or more unix-style /filepath/patterns*.h5, each of
        which is a separate collection job. This should correspond to the
        fn_out_names, unless both are set to PIPELINE.
        """

        collect_pattern = self['collect_pattern']

        if str(collect_pattern) == 'PIPELINE':
            coldir = Pipeline.parse_previous(self.dirout,
                                             module=ModuleName.COLLECT,
                                             target='dirout')[0]
            prefixes = self._parse_pipeline_prefixes()
            fn_patterns = [fn if fn.endswith('.h5') else fn + '*.h5'
                           for fn in prefixes]
            collect_pattern = [os.path.join(coldir, fn) for fn in fn_patterns]

        if isinstance(collect_pattern, str):
            collect_pattern = [collect_pattern]
        elif collect_pattern is None:
            msg = ('Failed to parse "collect_pattern" from collect config!')
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            collect_pattern = list(collect_pattern)

        return collect_pattern
