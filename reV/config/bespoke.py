# -*- coding: utf-8 -*-
"""
reV bespoke wind plant optimization config

Created on Jan 2022

@author: gbuster
"""
import logging

from reV.config.output_request import SAMOutputRequest
from reV.config.base_analysis_config import AnalysisConfig
from reV.utilities.exceptions import ConfigError
from reV.utilities import ModuleName

logger = logging.getLogger(__name__)


class BespokeConfig(AnalysisConfig):
    """SAM-based analysis config (generation, lcoe, etc...)."""

    NAME = ModuleName.BESPOKE

    REQUIREMENTS = ('excl_fpath', 'res_fpath', 'tm_dset', 'objective_function',
                    'cost_function', 'project_points', 'sam_files',
                    )

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        super().__init__(config)

    def _preflight(self):
        """Run preflight checks for missing REQUIREMENTS and also check special
        bespoke inputs."""
        super()._preflight()

        if self.project_points is None and len(self.sam_files) > 1:
            msg = ('If project_points is None, only one sam_files entry '
                   'should be present, but received {} sam_files: {}'
                   .format(len(self.sam_files), self.sam_files))
            logger.error(msg)
            raise ConfigError(msg)

    @property
    def excl_fpath(self):
        """Get the exclusions filepath(s), Required."""
        return self['excl_fpath']

    @property
    def res_fpath(self):
        """
        Wind resource h5 filepath in NREL WTK format. Can also include
        unix-style wildcards like /dir/wind_*.h5 for multiple years of
        resource data. Required.

        Returns
        -------
        str
        """
        return self['res_fpath']

    @property
    def tm_dset(self):
        """Get the techmap dataset, required."""
        return self['tm_dset']

    @property
    def objective_function(self):
        """Get the bespoke optimization objective function"""
        return self['objective_function']

    @property
    def cost_function(self):
        """Get the bespoke optimization cost function"""
        return self['cost_function']

    @property
    def project_points(self):
        """This can be None to use all available reV supply curve points or a
        string pointing to a project points csv. Points csv should have 'gid'
        and 'config' column, the config maps to the sam_configs dict keys.

        Returns
        -------
        pp : ProjectPoints
            ProjectPoints object
        """
        return self['project_points']

    @property
    def points_range(self):
        """An optional input that specifies the (start, end) index (inclusive,
        exclusive) of the project points to analyze. If this is specified, the
        requested points are analyzed on a single worker.
        """
        return self.get('points_range', None)

    @property
    def sam_files(self):
        """SAM config files. This should be a dictionary mapping config ids
        (keys) to config filepaths. If points is None, only one entry should be
        present and can be a single string filepath (no dict necessary).

        Returns
        -------
        dict
        """
        sf = self['sam_files']
        if isinstance(sf, str):
            sf = {'default': sf}
        return sf

    @property
    def min_spacing(self):
        """Minimum spacing between turbines in meters. Can also be a string
        like "5x" (default) which is interpreted as 5 times the turbine rotor
        diameter.
        """
        return str(self.get('min_spacing', '5x'))

    @property
    def ga_time(self):
        """Cutoff time for single-plant genetic algorithm optimization in
        seconds. Default is 20 seconds.
        """
        return float(self.get('ga_time', 20))

    @property
    def output_request(self):
        """Get the list of requested output variables. default is
        ('system_capacity', 'cf_mean')
        """

        default = ('system_capacity', 'cf_mean')
        _output_request = self.get('output_request', default)
        _output_request = SAMOutputRequest(_output_request)

        return _output_request

    @property
    def ws_bins(self):
        """Get the windspeed binning arguments This should be a 3-entry list
        with [start, stop, step] for the windspeed binning of the wind joint
        probability distribution. The stop value is inclusive, so ws_bins=[0,
        20, 5] would result in four bins with bin edges [0, 5, 10, 15, 20].
        Default is [0, 20, 5].
        """
        return self.get('ws_bins', [0, 20, 5])

    @property
    def wd_bins(self):
        """Get the winddirection binning arguments This should be a 3-entry
        list with [start, stop, step] for the winddirection binning of the
        wind joint probability distribution. The stop value is inclusive, so
        ws_bins=[0, 360, 90] would result in four bins with bin edges [0, 90,
        180, 270, 360]. Default is [0, 360, 45].
        """
        return self.get('wd_bins', [0, 360, 45])

    @property
    def excl_dict(self):
        """Get the exclusions dictionary. Default is None (all included)."""
        return self.get('excl_dict', None)

    @property
    def area_filter_kernel(self):
        """Get the minimum area filter kernel name ('queen' or 'rook'). Default
        is queen but the area filter kernel isnt active unless min_area != 0"""
        return self.get('area_filter_kernel', 'queen')

    @property
    def min_area(self):
        """Get the minimum area filter minimum area in km2. Default is None
        (not active)."""
        return self.get('min_area', None)

    @property
    def resolution(self):
        """Get the SC resolution. Default is 64."""
        return self.get('resolution', 64)

    @property
    def excl_area(self):
        """Get the exclusion pixel area in km2. Default is None which will
        determine the area from the exclusion file projection profile."""
        return self.get('excl_area', None)

    @property
    def pre_extract_inclusions(self):
        """Optional flag to pre-extract/compute the inclusion mask from the
        provided excl_dict, by default False. Typically faster to compute
        the inclusion mask on the fly with parallel workers.
        """
        return self.get('pre_extract_inclusions', False)
