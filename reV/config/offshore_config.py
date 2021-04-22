# -*- coding: utf-8 -*-
"""
reV offshore wind aggregation config.

@author: gbuster
"""
import logging

from reV.config.base_analysis_config import AnalysisConfig
from reV.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


class OffshoreConfig(AnalysisConfig):
    """Offshore wind aggregation config."""

    NAME = 'offshore'
    REQUIREMENTS = ('gen_fpath', 'offshore_fpath', 'project_points',
                    'sam_files', 'nrwal_configs')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        super().__init__(config)

    @property
    def gen_fpath(self):
        """
        Base generation fpath
        """
        return self['gen_fpath']

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

    @property
    def nrwal_configs(self):
        """Get the nrwal configs dict"""
        return self['nrwal_configs']

    @property
    def offshore_meta_cols(self):
        """Column labels from offshore_fpath to pass through to the output meta
        data. None (default) will use the offshore class variable
        DEFAULT_META_COLS, and any additional cols requested here will be added
        to DEFAULT_META_COLS."""
        return self.get('offshore_meta_cols', None)

    @property
    def offshore_nrwal_keys(self):
        """Get keys from the offshore nrwal configs to pass through as new
        datasets in the reV output h5"""
        return self.get('offshore_nrwal_keys', None)

    def parse_gen_fpaths(self):
        """
        Get a list of generation data filepaths

        Returns
        -------
        list
        """
        fpaths = self.gen_fpath
        if fpaths == 'PIPELINE':
            fpaths = Pipeline.parse_previous(
                self.dirout, 'offshore', target='fpath',
                target_module='generation')

        if isinstance(fpaths, str):
            fpaths = [fpaths]

        return fpaths
