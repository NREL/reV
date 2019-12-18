# -*- coding: utf-8 -*-
"""
reV offshore wind aggregation config.

@author: gbuster
"""
import logging

from reV.utilities.exceptions import ConfigError
from reV.config.base_analysis_config import AnalysisConfig
from reV.pipeline.pipeline import Pipeline


logger = logging.getLogger(__name__)


class OffshoreConfig(AnalysisConfig):
    """Offshore wind aggregation config."""

    NAME = 'offshore'
    REQUIREMENTS = ('gen_fpath', 'offshore_fpath', 'project_points',
                    'sam_files')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        super().__init__(config)
        self._preflight()

    def _preflight(self):
        """Perform pre-flight checks on the SC agg config inputs"""
        missing = []
        for req in self.REQUIREMENTS:
            if self.get(req, None) is None:
                missing.append(req)
        if any(missing):
            raise ConfigError('SC offshore config missing the following '
                              'keys: {}'.format(missing))

    @property
    def gen_fpath(self):
        """Get the generation data filepath(s)"""

        fpath = self['gen_fpath']

        if fpath == 'PIPELINE':
            fpath = Pipeline.parse_previous(
                self.dirout, 'aggregation', target='fpath',
                target_module='generation')

        return fpath

    @property
    def offshore_fpath(self):
        """Get the offshore data filepath"""
        return self['offshore_fpath']

    @property
    def project_points(self):
        """Get the project points filepath"""
        return self['project_points']

    @property
    def sam_files(self):
        """Get the sam files dict"""
        return self['sam_files']
