# -*- coding: utf-8 -*-
"""
reV bespoke wind plant optimization config

Created on Jan 2022

@author: gbuster
"""
import logging
import numpy as np

from reV.config.output_request import SAMOutputRequest
from reV.config.base_analysis_config import AnalysisConfig
from reV.config.sam_config import SAMConfig
from reV.config.project_points import PointsControl, ProjectPoints

logger = logging.getLogger(__name__)


class BespokeConfig(AnalysisConfig):
    """SAM-based analysis config (generation, lcoe, etc...)."""
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
        self._pc = None

    def parse_sam_config(self):
        """Get the SAM configuration object.

        Returns
        -------
        sam_gen : reV.config.sam.SAMConfig
            SAM config object. This object emulates a dictionary.
        """
        return SAMConfig(self['sam_files'])

    def parse_points_control(self):
        """Get the generation points control object.

        Returns
        -------
        points_control : reV.config.project_points.PointsControl
            PointsControl object based on specified project points and
            execution control option.
        """
        if self._pc is None:
            # make an instance of project points
            pp = ProjectPoints(self.project_points, self['sam_files'],
                               tech='windpower')

            sites_per_worker = int(1e9)
            if (self.execution_control.option == 'peregrine'
                    or self.execution_control.option == 'eagle'):
                # sites per split on peregrine or eagle is the number of sites
                # in project points / number of nodes. This is for the initial
                # division of the project sites between HPC nodes (jobs)
                sites_per_worker = int(np.ceil(
                    len(pp) / self.execution_control.nodes))

            elif self.execution_control.option == 'local':
                # sites per split on local is number of sites / # of processes
                sites_per_worker = int(np.ceil(
                    len(pp) / self.execution_control.max_workers))

            # make an instance of points control and set to protected attribute
            self._pc = PointsControl(pp, sites_per_split=sites_per_worker)

        return self._pc

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
        return self['resource_file']

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
        """
        project_points input

        Returns
        -------
        pp : ProjectPoints
            ProjectPoints object
        """
        return self['project_points']

    @property
    def sam_files(self):
        """
        SAM config files

        Returns
        -------
        dict
        """
        return self['sam_files']

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
        """Get the windspeed binning arguments This should be a 3-entry tuple
        with (start, stop, step) for the windspeed binning of the wind joint
        probability distribution. The stop value is inclusive, so ws_bins=(0,
        20, 5) would result in four bins with bin edges (0, 5, 10, 15, 20).
        Default is (0, 20, 5).
        """
        return self.get('ws_bins', (0, 20, 5))

    @property
    def wd_bins(self):
        """Get the winddirection binning arguments This should be a 3-entry
        tuple with (start, stop, step) for the winddirection binning of the
        wind joint probability distribution. The stop value is inclusive, so
        ws_bins=(0, 360, 90) would result in four bins with bin edges (0, 90,
        180, 270, 360). Default is (0, 360, 45).
        """
        return self.get('wd_bins', (0, 360, 45))

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
