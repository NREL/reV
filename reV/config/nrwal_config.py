# -*- coding: utf-8 -*-
"""
reV-NRWAL config.

@author: gbuster
"""
from glob import glob
import logging

from reV.config.base_analysis_config import AnalysisConfig
from reV.pipeline.pipeline import Pipeline
from reV.utilities import ModuleName

logger = logging.getLogger(__name__)


class RevNrwalConfig(AnalysisConfig):
    """Offshore wind aggregation config."""

    NAME = ModuleName.NRWAL
    REQUIREMENTS = ('gen_fpath', 'site_data', 'sam_files', 'nrwal_configs',
                    'output_request')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        super().__init__(config)
        self._gen_fpath = self._parse_gen_fpath()

    def _parse_gen_fpath(self):
        """Get one or more generation data filepaths

        Returns
        -------
        list | str
        """
        fpaths = self['gen_fpath']
        if fpaths == 'PIPELINE':
            fpaths = Pipeline.parse_previous(
                self.dirout, module=ModuleName.NRWAL, target='fpath'
            )

        if isinstance(fpaths, str) and '*' in fpaths:
            fpaths = glob(fpaths)
            if not any(fpaths):
                msg = ('Could not find any file paths for '
                       'gen_fpath glob pattern.')
                logger.error(msg)
                raise RuntimeError(msg)

        if len(fpaths) == 1:
            fpaths = fpaths[0]

        return fpaths

    @property
    def gen_fpath(self):
        """Base generation fpath(s) used as input data files. Anything in the
        output_request is added and/or manipulated in this file(s)."""
        return self._gen_fpath

    @property
    def site_data(self):
        """Get the site-specific spatial data filepath"""
        return self['site_data']

    @property
    def sam_files(self):
        """Get the sam files dict"""
        return self['sam_files']

    @property
    def nrwal_configs(self):
        """Get the nrwal configs dict"""
        return self['nrwal_configs']

    @property
    def output_request(self):
        """Get keys from the nrwal configs to pass through as new datasets in
        the reV output h5. If you want to manipulate a dset like cf_mean from
        gen_fpath and include it in the output_request, you should set
        save_raw=True and then in the NRWAL equations use cf_mean_raw as the
        input and then define cf_mean as the manipulated data that will be
        included in the output_request.
        """
        return self['output_request']

    @property
    def save_raw(self):
        """Get the flag to save raw datasets in gen_fpath in the case that they
        are manipulated then requested in output_request. Optional, default is
        True."""
        return bool(self.get('save_raw', True))

    @property
    def meta_gid_col(self):
        """The column in the gen_fpath meta data that corresponds to the "gid"
        column in the site_data input. Optional, default is "gid".
        """
        return self.get('meta_gid_col', 'gid')

    @property
    def site_meta_cols(self):
        """Column labels from site_data to pass through to the output meta
        data. Optional, default is None."""
        return self.get('site_meta_cols', None)
