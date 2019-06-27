# -*- coding: utf-8 -*-
"""reV-to-SAM interface module.

Relies heavily upon the SAM Simulation Core (SSC) API module (sscapi) from the
SAM software development kit (SDK).
"""
import copy
import gc
import logging
import numpy as np
import pandas as pd
from warnings import warn

from reV.handlers.resource import Resource
from reV.utilities.exceptions import SAMInputWarning, SAMExecutionError
from reV.utilities.curtailment import curtail
from reV.utilities.utilities import mean_irrad
from reV.SAM.SAM import SAM, SiteOutput
from reV.SAM.econ import LCOE, SingleOwner


logger = logging.getLogger(__name__)


class Generation(SAM):
    """Base class for SAM generation simulations."""

    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None):
        """Initialize a SAM generation object."""

        # check timezone input and set missing timezone from json if necessary
        meta = self.tz_check(parameters, meta)

        super().__init__(resource=resource, meta=meta, parameters=parameters,
                         output_request=output_request)

    @staticmethod
    def get_res_mean(res_file, res_df, output_request):
        """Get the resource annual means.

        Parameters
        ----------
        res_file : str
            Resource file with full path.
        res_df : pd.DataFrame
            2D table with resource data. Available columns must have solar_vars
        output_request : list
            Outputs to retrieve from SAM.

        Returns
        -------
        res_mean : SiteOutput
            SiteOutput object with variables for resource means.
        out_req_nomeans : list
            Output request list with the resource mean entries removed.
        """

        out_req_nomeans = copy.deepcopy(output_request)
        res_mean = None

        if 'ws_mean' in out_req_nomeans:
            out_req_nomeans.remove('ws_mean')
            res_mean = SiteOutput()
            res_mean['ws_mean'] = res_df['windspeed'].mean()

        else:
            if 'dni_mean' in out_req_nomeans:
                out_req_nomeans.remove('dni_mean')
                res_mean = SiteOutput()
                res_mean['dni_mean'] = mean_irrad(res_df['dni'])

            if 'ghi_mean' in out_req_nomeans:
                out_req_nomeans.remove('ghi_mean')
                if res_mean is None:
                    res_mean = SiteOutput()

                if 'ghi' in res_df:
                    res_mean['ghi_mean'] = mean_irrad(res_df['ghi'])
                else:
                    with Resource(res_file) as res:
                        res_mean['ghi_mean'] = mean_irrad(
                            res['ghi', :, res_df.name])

        return res_mean, out_req_nomeans

    @staticmethod
    def tz_check(parameters, meta):
        """Check timezone input and use json config tz if not in resource meta.

        Parameters
        ----------
        parameters : dict
            SAM model input parameters.
        meta : pd.DataFrame
            1D table with resource meta data.

        Returns
        -------
        meta : pd.DataFrame
            1D table with resource meta data. If meta was not originally set in
            the resource meta data, but was set as "tz" or "timezone" in the
            SAM model input parameters json file, timezone will be added to
            this instance of meta.
        """

        if meta is not None:
            if 'timezone' not in meta:
                if 'tz' in parameters:
                    meta['timezone'] = int(parameters['tz'])
                elif 'timezone' in parameters:
                    meta['timezone'] = int(parameters['timezone'])
                else:
                    msg = ('Need timezone input to run SAM gen. Not found in '
                           'resource meta or technology json input config.')
                    raise SAMExecutionError(msg)
        return meta

    def gen_exec(self, module_to_run):
        """Run SAM generation with possibility for follow on econ analysis.

        Parameters
        ----------
        module_to_run : str
            SAM module name (e.g., 'pvwattsv5', 'tcsmolten_salt', 'windpower').
        """

        self.set_parameters()

        if 'lcoe_fcr' in self.output_request:
            # econ outputs requested, run LCOE model after generation.
            self.execute(module_to_run, close=False)
            lcoe = LCOE(self.ssc, self.data, self.parameters,
                        output_request=('lcoe_fcr',))
            lcoe.execute(LCOE.MODULE)
            self.outputs.update(lcoe.outputs)

        elif 'ppa_price' in self.output_request:
            # econ outputs requested, run SingleOwner model after generation.
            self.execute(module_to_run, close=False)
            so = SingleOwner(self.ssc, self.data, self.parameters,
                             output_request=('ppa_price',))
            so.execute(so.MODULE)
            self.outputs.update(so.outputs)

        else:
            # normal run, no econ analysis
            self.execute(module_to_run, close=True)

    @classmethod
    def reV_run(cls, points_control, res_file, output_request=('cf_mean',),
                downscale=None):
        """Execute SAM generation based on a reV points control instance.

        Parameters
        ----------
        points_control : config.PointsControl
            PointsControl instance containing project points site and SAM
            config info.
        res_file : str
            Resource file with full path.
        output_request : list | tuple
            Outputs to retrieve from SAM.
        downscale : NoneType | str
            Option for NSRDB resource downscaling to higher temporal
            resolution. Expects a string in the Pandas frequency format,
            e.g. '5min'.

        Returns
        -------
        out : dict
            Nested dictionaries where the top level key is the site index,
            the second level key is the variable name, second level value is
            the output variable value.
        """

        # initialize output dictionary
        out = {}

        # Get the SAM resource object
        resources = SAM.get_sam_res(res_file,
                                    points_control.project_points,
                                    points_control.project_points.tech,
                                    downscale=downscale)

        # run resource through curtailment filter if applicable
        curtailment = points_control.project_points.curtailment
        if curtailment is not None:
            resources = curtail(resources, curtailment)

        # Use resource object iterator
        for res_df, meta in resources:

            # get SAM inputs from project_points based on the current site
            site = res_df.name
            _, inputs = points_control.project_points[site]

            res_mean, out_req_nomeans = cls.get_res_mean(res_file, res_df,
                                                         output_request)

            # iterate through requested sites.
            sim = cls(resource=res_df, meta=meta, parameters=inputs,
                      output_request=out_req_nomeans)
            sim.gen_exec(cls.MODULE)

            # collect outputs to dictout
            out[site] = sim.outputs
            if res_mean is not None:
                out[site].update(res_mean)

            del res_df, meta, sim

        del resources
        gc.collect()
        return out


class Solar(Generation):
    """Base Class for Solar generation from SAM
    """

    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None, drop_leap=False):
        """Initialize a SAM solar object.

        Parameters
        ----------
        resource : pd.DataFrame
            2D table with resource data. Available columns must have solar_vars
        meta : pd.DataFrame
            1D table with resource meta data.
        parameters : dict or ParametersManager()
            SAM model input parameters.
        output_request : list
            Requested SAM outputs (e.g., 'cf_mean', 'annual_energy',
            'cf_profile', 'gen_profile', 'energy_yield', 'ppa_price',
            'lcoe_fcr').
        drop_leap : bool
            Drops February 29th from the resource data.
        """

        # drop the leap day
        if drop_leap:
            resource = self.drop_leap(resource)

        # set PV tilt to latitude if applicable
        parameters = self.set_latitude_tilt_az(parameters, meta)

        # don't pass resource to base class, set in set_nsrdb instead.
        super().__init__(resource=None, meta=meta, parameters=parameters,
                         output_request=output_request)

        # Set the site number using resource
        if isinstance(resource, pd.DataFrame):
            self._site = resource.name
        else:
            self._site = None

        if resource is None or meta is None:
            # if no resource input data is specified, you need a resource file
            self.parameters.require_resource_file(res_type='solar')

        elif resource is not None and meta is not None:
            self.set_nsrdb(resource)

    def set_latitude_tilt_az(self, parameters, meta):
        """Check if tilt is specified as latitude and set tilt=lat, az=180 or 0

        Parameters
        ----------
        parameters : dict
            SAM model input parameters.
        meta : pd.DataFrame
            1D table with resource meta data.

        Returns
        -------
        parameters : dict
            SAM model input parameters. If for a pv simulation the "tilt"
            parameter was originally not present or set to 'lat' or 'latitude',
            the tilt will be set to the absolute value of the latitude found
            in meta and the azimuth will be 180 if lat>0, 0 if lat<0.
        """

        set_tilt = False
        if 'pv' in self.MODULE:
            if parameters is not None and meta is not None:
                if 'tilt' not in parameters:
                    warn('No tilt specified, setting at latitude.',
                         SAMInputWarning)
                    set_tilt = True
                else:
                    if (parameters['tilt'] == 'lat'
                            or parameters['tilt'] == 'latitude'):
                        set_tilt = True

        if set_tilt:
            # set tilt to abs(latitude)
            parameters['tilt'] = np.abs(meta['latitude'])
            if meta['latitude'] > 0:
                # above the equator, az = 180
                parameters['azimuth'] = 180
            else:
                # below the equator, az = 0
                parameters['azimuth'] = 0
            logger.debug('Tilt specified at "latitude", setting tilt to: {}, '
                         'azimuth to: {}'
                         .format(parameters['tilt'], parameters['azimuth']))
        return parameters

    def set_nsrdb(self, resource):
        """Set SSC NSRDB resource data arrays.

        Parameters
        ----------
        resource : pd.DataFrame
            2D table with resource data. Available columns must have var_list.
        """

        # call generic set resource method from the base class
        super().set_resource(resource=resource)

        # map resource data names to SAM required data names
        var_map = {'dni': 'dn',
                   'dhi': 'df',
                   'ghi': 'gh',
                   'wind_speed': 'wspd',
                   'air_temperature': 'tdry',
                   }

        # set resource variables
        for var in resource.columns.values:
            if var != 'time_index':
                # ensure that resource array length is multiple of 8760
                res_arr = np.roll(
                    self.ensure_res_len(resource[var]),
                    int(self.meta['timezone'] * self.time_interval))
                if var in ['dni', 'dhi', 'ghi']:
                    if np.min(res_arr) < 0:
                        warn('Solar irradiance variable "{}" has a minimum '
                             'value of {}. Truncating to zero.'
                             .format(var, np.min(res_arr)), SAMInputWarning)
                        res_arr = np.where(res_arr < 0, 0, res_arr)

                self.ssc.data_set_array(self.res_data, var_map[var], res_arr)

        # add resource data to self.data and clear
        self.ssc.data_set_table(self.data, 'solar_resource_data',
                                self.res_data)
        self.ssc.data_free(self.res_data)


class PV(Solar):
    """Photovoltaic (PV) generation with pvwattsv5.
    """
    MODULE = 'pvwattsv5'

    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None):
        """Initialize a SAM solar PV object.

        Parameters
        ----------
        resource : pd.DataFrame
            2D table with resource data. Available columns must have solar_vars
        meta : pd.DataFrame
            1D table with resource meta data.
        parameters : dict or ParametersManager()
            SAM model input parameters.
        output_request : list
            Requested SAM outputs (e.g., 'cf_mean', 'annual_energy',
            'cf_profile', 'gen_profile', 'energy_yield', 'ppa_price',
            'lcoe_fcr').
        """
        super().__init__(resource=resource, meta=meta, parameters=parameters,
                         output_request=output_request)


class CSP(Solar):
    """Concentrated Solar Power (CSP) generation
    """
    MODULE = 'tcsmolten_salt'

    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None):
        """Initialize a SAM concentrated solar power (CSP) object.
        """
        super().__init__(resource=resource, meta=meta, parameters=parameters,
                         output_request=output_request)


class Wind(Generation):
    """Base class for Wind generation from SAM
    """

    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None, drop_leap=False):
        """Initialize a SAM wind object.

        Parameters
        ----------
        resource : pd.DataFrame
            2D table with resource data. Available columns must have wind_vars
        meta : pd.DataFrame
            1D table with resource meta data.
        parameters : dict
            SAM model input parameters.
        output_request : list
            Requested SAM outputs (e.g., 'cf_mean', 'annual_energy',
            'cf_profile', 'gen_profile', 'energy_yield', 'ppa_price',
            'lcoe_fcr').
        drop_leap : bool
            Drops February 29th from the resource data.
        """

        # drop the leap day
        if drop_leap:
            resource = self.drop_leap(resource)

        # don't pass resource to base class, set in set_wtk instead.
        super().__init__(resource=None, meta=meta, parameters=parameters,
                         output_request=output_request)

        # Set the site number using resource
        if isinstance(resource, pd.DataFrame):
            self._site = resource.name
        else:
            self._site = None

        if resource is None or meta is None:
            # if no resource input data is specified, you need a resource file
            self.parameters.require_resource_file(res_type='wind')

        elif resource is not None and meta is not None:
            self.set_wtk(resource)

    def set_wtk(self, resource):
        """Set SSC WTK resource data arrays.

        Parameters
        ----------
        resource : pd.DataFrame
            2D table with resource data. Available columns must have var_list.
        """

        # call generic set resource method from the base class
        super().set_resource(resource=resource)

        self.ssc.data_set_array(self.res_data, 'fields', [1, 2, 3, 4])
        self.ssc.data_set_array(self.res_data, 'heights',
                                4 * [self.parameters['wind_turbine_hub_ht']])

        # must be set as matrix in [temperature, pres, speed, direction] order
        # ensure that resource array length is multiple of 8760
        # roll the truncated resource array to local timezone
        temp = np.roll(
            self.ensure_res_len(resource[['temperature', 'pressure',
                                          'windspeed',
                                          'winddirection']].values),
            int(self.meta['timezone'] * self.time_interval), axis=0)
        self.ssc.data_set_matrix(self.res_data, 'data', temp)

        # add resource data to self.data and clear
        self.ssc.data_set_table(self.data, 'wind_resource_data', self.res_data)
        self.ssc.data_free(self.res_data)


class LandBasedWind(Wind):
    """Onshore wind generation
    """
    MODULE = 'windpower'

    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None):
        """Initialize a SAM land based wind object.
        """
        super().__init__(resource=resource, meta=meta, parameters=parameters,
                         output_request=output_request)


class OffshoreWind(LandBasedWind):
    """Offshore wind generation
    """
    MODULE = 'windpower'

    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None):
        """Initialize a SAM offshore wind object.
        """
        super().__init__(resource=resource, meta=meta, parameters=parameters,
                         output_request=output_request)
