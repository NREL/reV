# -*- coding: utf-8 -*-
"""
reV-NRWAL config.

@author: gbuster
"""
import logging

from reV.config.base_analysis_config import AnalysisConfig
from reV.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


class OffshoreConfig(AnalysisConfig):
    """Offshore wind aggregation config."""

    NAME = 'nrwal'
    REQUIREMENTS = ('gen_fpath', 'site_data', 'project_points',
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
    def site_data(self):
        """Get the site-specific spatial data filepath"""
        return self['site_data']

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
    def site_meta_cols(self):
        """Column labels from site_data to pass through to the output meta
        data."""
        return self.get('site_meta_cols', None)

    @property
    def output_request(self):
        """Get keys from the nrwal configs to pass through as new datasets in
        the reV output h5"""
        return self.get('output_request', None)

    @property
    def run_offshore(self):
        """Get the flag to run nrwal for offshore sites only based on the
        'offshore' flag in the meta data"""
        return bool(self.get('run_offshore', False))

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
                self.dirout, 'nrwal', target='fpath',
                target_module='generation')

        if isinstance(fpaths, str):
            fpaths = [fpaths]

        return fpaths
