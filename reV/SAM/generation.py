#!/usr/bin/env python
"""reV-to-SAM interface module.

Relies heavily upon the SAM Simulation Core (SSC) API module (sscapi) from the
SAM software development kit (SDK).
"""
import gc
import logging
import numpy as np

from reV.SAM.SAM import SAM
from reV.SAM.econ import LCOE, SingleOwner


logger = logging.getLogger(__name__)


class Generation(SAM):
    """Base class for SAM generation simulations."""

    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None):
        """Initialize a SAM generation object."""
        super().__init__(resource=resource, meta=meta, parameters=parameters,
                         output_request=output_request)

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
                        output_request=self.output_request)
            lcoe.execute(LCOE.MODULE)
            self.outputs = lcoe.outputs

        elif 'ppa_price' in self.output_request:
            # econ outputs requested, run SingleOwner model after generation.
            self.execute(module_to_run, close=False)
            so = SingleOwner(self.ssc, self.data, self.parameters,
                             output_request=self.output_request)
            so.execute(so.MODULE)
            self.outputs = so.outputs

        else:
            # normal run, no econ analysis
            self.execute(module_to_run, close=True)

    @classmethod
    def reV_run(cls, points_control, res_file, output_request=('cf_mean',)):
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
        return_meta : bool
            Adds meta key/value pair to dictionary output. Additional reV
            variables added to the meta series.

        Returns
        -------
        out : dict
            Nested dictionaries where the top level key is the site index,
            the second level key is the variable name, second level value is
            the output variable value.
        """

        out = {}

        resources = SAM.get_sam_res(res_file, points_control.project_points,
                                    points_control.project_points.tech)

        for res_df, meta in resources:
            # get SAM inputs from project_points based on the current site
            site = res_df.name
            config, inputs = points_control.project_points[site]
            # iterate through requested sites.
            sim = cls(resource=res_df, meta=meta, parameters=inputs,
                      output_request=output_request)
            sim.gen_exec(cls.MODULE)
            out[site] = sim.outputs

            logger.debug('Outputs for site {} with config "{}", \n\t{}...'
                         .format(site, config, str(out[site])[:100]))
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

        # don't pass resource to base class, set in set_nsrdb instead.
        super().__init__(resource=None, meta=meta, parameters=parameters,
                         output_request=output_request)

        # Set the site number using resource
        self.site = resource

        if resource is None or meta is None:
            # if no resource input data is specified, you need a resource file
            self.parameters.require_resource_file(res_type='solar')

        elif resource is not None and meta is not None:
            self.set_nsrdb(resource)

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
                res_arr = self.ensure_res_len(np.roll(resource[var],
                                              int(self.meta['timezone'] *
                                                  self.time_interval)))
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
        self.site = resource

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
        temp = self.ensure_res_len(resource[['temperature', 'pressure',
                                             'windspeed',
                                             'winddirection']].values)
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
