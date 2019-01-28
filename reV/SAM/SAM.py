#!/usr/bin/env python
"""reV-to-SAM interface module.

Relies heavily upon the SAM Simulation Core (SSC) API module (sscapi) from the
SAM software development kit (SDK).
"""
import gc
import json
import logging
import numpy as np
import os
import pandas as pd
from warnings import warn

from ORCA.system import System as ORCASystem
from ORCA.data import Data as ORCAData

from reV.handlers.resource import WTK, NSRDB
from reV.SAM.PySSC import PySSC
from reV.utilities.exceptions import SAMInputWarning, SAMExecutionError


logger = logging.getLogger(__name__)


def is_num(n):
    """Check if input is a number (returns True/False)"""
    try:
        float(n)
        return True
    except Exception as _:
        return False


def is_str(s):
    """Check if input is a string and not a number (returns True/False)"""
    if is_num(s) is True:
        return False
    else:
        try:
            str(s)
            return True
        except Exception as _:
            return False


def is_array(a):
    """Check if input is an array (returns True/False)"""
    if (isinstance(a, (list, np.ndarray)) and ~isinstance(a, str)):
        return True
    else:
        return False


def is_2D_list(a):
    """Check if input is a nested (2D) list (returns True/False)"""
    if isinstance(a, list):
        if isinstance(a[0], list):
            return True
    return False


class ParametersManager:
    """Class to manage SAM input parameters, requirements, and defaults."""

    def __init__(self, parameters, module, verify=True, set_def=True):
        """Initialize the SAM input parameters class.

        Parameters
        ----------
        parameters : dict
            SAM model input parameters.
        module : str
            SAM module ('pvwatts', 'tcsmolten_salt', etc...)
        verify : bool
            Flag on whether to verify input parameters.
        set_def : bool
            Flag on whether to set defaults for missing input parameters
            (prints SAMInputWarning if defaults get set).
        """

        # set the parameters and module properties
        self._module = module
        self.parameters = parameters

        # get requirements and verify that all are satisfied
        self._requirements = self.get_requirements(self.module)
        if verify:
            self.verify_inputs(set_def=set_def)

    def __getitem__(self, key):
        """Get parameters property"""
        return self.parameters[key]

    def __setitem__(self, key, value):
        """Set dictionary entry in parameters property"""
        self._parameters[key] = value

    def keys(self):
        """Return the dict keys representation of the parameters property"""
        return self.parameters.keys()

    @property
    def module(self):
        """Get module property."""
        return self._module

    @property
    def parameters(self):
        """Get the parameters property"""
        return self._parameters

    @parameters.setter
    def parameters(self, p):
        """Set the parameters property"""
        if isinstance(p, dict):
            self._parameters = p
        else:
            warn('Input parameters for {} SAM module need to '
                 'be input as a dictionary but were input as '
                 'a {}'.format(self.module, type(p)), SAMInputWarning)
            warn('No input parameters specified for '
                 '{} SAM module, defaults will be set'
                 .format(self.module), SAMInputWarning)
            self._parameters = {}

    @property
    def requirements(self):
        """Get the requirements property"""
        return self._requirements

    @staticmethod
    def get_requirements(module, req_folder='requirements'):
        """Retrieve the module-specific input data requirements from json.

        Parameters
        ----------
        module : str
            SAM module name. Must match .json filename in _path.
        req_folder : str
            Path containing requirement json's.

        Returns
        -------
        req : list
            SAM requirement input variable names and possible datatypes.
            Format is: req = [[input1, [dtype1a, dtype1b]],
                              [input2, [dtype2a]]]
        """

        type_map = {'int': int, 'float': float, 'str': str,
                    'np.ndarray': np.ndarray, 'list': list}

        jf = os.path.join(SAM.DIR, req_folder, module + '.json')

        with open(jf, 'r') as f:
            req = json.load(f)

        for i, [_, dtypes] in enumerate(req):
            for j, dtype in enumerate(dtypes):
                req[i][1][j] = type_map[dtype]
        return req

    @staticmethod
    def get_defaults(module, def_folder='defaults'):
        """Retrieve the module-specific default inputs.

        Parameters
        ----------
        module : str
            SAM module name. Must match .json filename in _path.
        def_folder : str
            Path containing default input json's.

        Returns
        -------
        defaults : dict
            SAM defaults for the specified module.
        """

        jf = os.path.join(SAM.DIR, def_folder, module + '.json')

        with open(jf, 'r') as f:
            # get unit test inputs
            defaults = json.load(f)

        return defaults

    def set_defaults(self):
        """Set missing values to module defaults.
        """
        defaults = self.get_defaults(self.module)
        for new, _ in self.requirements:
            if new not in self.parameters.keys():
                self.__setitem__(new, defaults[new])
                warn('Setting default value for "{}"'
                     .format(new), SAMInputWarning)

    def require_resource_file(self, res_type):
        """Enforce requirement of a resource file if res data is not input"""
        if res_type.lower() == 'solar':
            self._requirements.append(['solar_resource_file', [str]])
        elif res_type.lower() == 'wind':
            self._requirements.append(['wind_resource_filename', [str]])

        # re-verify inputs with new requirement. Will also set the default
        # resource file if one is not provided.
        self.verify_inputs()

    def verify_inputs(self, set_def=True):
        """Verify that required inputs are available and have correct dtype.
        Also set missing inputs to default values.

        Prints logger warnings when variables are missing, set to default,
        or are of the incorrect datatype.

        Parameters
        ----------
        set_def : bool
            Flag on whether to set defaults for missing input parameters
            (prints SAMInputWarning if defaults get set).
        """
        missing_inputs = False
        for name, dtypes in self.requirements:
            if name not in self.parameters.keys():
                warn('SAM input parameters must contain "{}"'
                     .format(name), SAMInputWarning)
                missing_inputs = True
            else:
                p = self.parameters[name]
                if not any([isinstance(p, d) for d in dtypes]):
                    warn('SAM input parameter "{}" must be of '
                         'type {} but is of type {}'
                         .format(name, dtypes, type(p)), SAMInputWarning)
        if missing_inputs and set_def:
            self.set_defaults()

    def update(self, more_parameters):
        """Add more parameters to this class.

        Parameters
        ----------
        more_parameters : dict
            New key-value pairs to add to this instance of SAM Parameters.
        """

        if isinstance(more_parameters, dict):
            self._parameters.update(more_parameters)
        else:
            warn('Attempting to update SAM input parameters with non-dict '
                 'input. Cannot perform update operation. Proceeding without '
                 'additional inputs: {}'.format(more_parameters),
                 SAMInputWarning)


class SAM:
    """Base class for SAM simulations (generation and econ)."""

    DIR = os.path.dirname(os.path.realpath(__file__))
    MODULE = None

    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None):
        """Initialize a SAM object.

        Parameters
        ----------
        meta : pd.DataFrame
            1D table with resource meta data.
        parameters : dict or ParametersManager()
            SAM model input parameters.
        output_request : list
            Requested SAM outputs (e.g., 'cf_mean', 'annual_energy',
            'cf_profile', 'gen_profile', 'energy_yield', 'ppa_price',
            'lcoe_fcr').
        """

        # set meta attribute as protected
        self._meta = meta

        # Initialize SAM Simulation Core (SSC) object as protected
        self._ssc = PySSC()

        # Initialize protected attribute for data setting
        self._data = self.ssc.data_create()

        # Initialize protected attribute for resource data setting
        self._res_data = self.ssc.data_create()

        # Use Parameters class to manage inputs, defaults, and requirements.
        if isinstance(parameters, ParametersManager):
            self.parameters = parameters
        else:
            self.parameters = ParametersManager(parameters, self.module)

        # Save output request as attribute
        self.output_request = output_request

        # Set resource data
        self.set_resource(resource=resource)

    @property
    def data(self):
        """Get data property."""
        return self._data

    @property
    def meta(self):
        """Get meta data property."""
        return self._meta

    @property
    def module(self):
        """Get module property."""
        return self.MODULE

    @property
    def res_data(self):
        """Get resource data property"""
        return self._res_data

    @property
    def ssc(self):
        """Get SAM simulation core (SSC) property"""
        return self._ssc

    @property
    def site(self):
        """Get the site number for this SAM simulation."""
        return self._site

    @site.setter
    def site(self, inp):
        """Set the site number based on resource input or integer."""
        if not hasattr(self, '_site'):
            if hasattr(inp, 'name'):
                # Set the protected property with the site number from resource
                self._site = inp.name
            elif isinstance(inp, int):
                self._site = inp
            else:
                # resource site number not found, set as N/A
                self._site = 'N/A'

    @staticmethod
    def get_sam_res(res_file, project_points, module=''):
        """Get the SAM resource iterator object (single year, single file).

        Parameters
        ----------
        res_file : str
            Single resource file (with full path) to retrieve.
        project_points : reV.config.ProjectPoints
            reV 2.0 Project Points instance used to retrieve resource data at a
            specific set of sites.
        module : str
            Optional SAM module name or reV technology to force interpretation
            of the resource file type.
            Example: if the resource file does not have 'nsrdb' in the
            name, module can be set to 'pvwatts' or 'tcsmolten' to force
            interpretation of the resource file as a solar resource.

        Returns
        -------
        res : reV.resource.SAMResource
            Resource iterator object to pass to SAM.
        """

        if ('nsrdb' in res_file or 'pv' in module or 'tcsmolten' in module or
                'csp' in module):
            res = NSRDB.preload_SAM(res_file, project_points)
        elif 'wtk' in res_file or 'wind' in module:
            res = WTK.preload_SAM(res_file, project_points)
        else:
            raise SAMExecutionError('Cannot interpret the type of resource '
                                    'file being input: {}. Should have nsrdb '
                                    'or wtk in the name, or specify which SAM '
                                    'module is being run.'.format(res_file))
        return res

    def set_resource(self, resource=None):
        """Generic resource setting utility.

        Parameters
        ----------
        resource : pd.DataFrame
            2D table with resource data. Must have time_index column.
        """

        if resource is not None:
            # set meta data
            self.set_meta()
            # set time variables
            self.time_interval = self.get_time_interval(resource.index)
            self.set_time_index(resource.index)

    def set_parameters(self, keys_to_set='all'):
        """Set SAM inputs using either a subset of keys or all parameter keys.

        Parameters
        ----------
        keys_to_set : str, list, or iterable
            This defaults to 'all', which will set all parameters in the
            dictionary self.parameters. Otherwise, only parameter keys in
            keys_to_set will be set.
        """
        if keys_to_set == 'all':
            keys_to_set = self.parameters.keys()

        for key in keys_to_set:
            # Set data to SSC using appropriate logic
            if is_num(self.parameters[key]) is True:
                self.ssc.data_set_number(self.data, key,
                                         self.parameters[key])

            elif is_2D_list(self.parameters[key]) is True:
                self.ssc.data_set_matrix(self.data, key,
                                         self.parameters[key])

            elif is_array(self.parameters[key]) is True:
                self.ssc.data_set_array(self.data, key,
                                        self.parameters[key])

            elif is_str(self.parameters[key]) is True:
                self.ssc.data_set_string(self.data, key,
                                         self.parameters[key])

    def set_meta(self, meta_vars=('latitude', 'longitude', 'elevation')):
        """Set the base SAM meta data variables when using resource data.

        Parameters
        ----------
        meta_vars : list | tuple
            List of meta variable names to set.
        """
        self.ssc.data_set_number(self.res_data, 'tz',
                                 int(self.meta['timezone']))

        # map resource data names to SAM required data names
        var_map = {'latitude': 'lat',
                   'longitude': 'lon',
                   'elevation': 'elev'}

        for var in meta_vars:
            self.ssc.data_set_number(self.res_data, var_map[var],
                                     self.meta[var])

    @staticmethod
    def drop_leap(resource):
        """Drop Feb 29th from all dataframes in resource dict."""
        if hasattr(resource, 'index'):
            if hasattr(resource.index, 'month') and hasattr(resource, 'day'):
                leap_day = ((resource.index.month == 2) &
                            (resource.index.day == 29))
                resource = resource.drop(resource.index[leap_day])
        return resource

    def set_time_index(self, time_index, time_vars=('year', 'month', 'day',
                                                    'hour', 'minute')):
        """Set the SAM time index variables.

        Parameters
        ----------
        time_index : pd.series
            Datetime series. Must have a dt attribute to access datetime
            properties (added using make_datetime method).
        time_vars : list | tuple
            List of time variable names to set.
        """

        time_index = self.make_datetime(time_index)
        time_index = self.ensure_res_len(time_index)

        for var in time_vars:
            self.ssc.data_set_array(self.res_data, var,
                                    getattr(time_index.dt, var).values)

    @staticmethod
    def ensure_res_len(res_arr, base=8760):
        """Ensure that the length of resource array is a multiple of base.

        Parameters
        ----------
        res_arr : array-like
            Array of resource data.
        base : int
            Ensure that length of resource array is a multiple of this value.

        Returns
        -------
        res_arr : array-like
            Truncated array of resource data such that length(res_arr)%base=0.
        """

        if len(res_arr) % base != 0:
            div = np.floor(len(res_arr) / 8760)
            target_len = int(div * 8760)
            warn('Resource array length is {}, but SAM requires a multiple of '
                 '8760. Truncating the timeseries to length {}.'
                 .format(len(res_arr), target_len), SAMInputWarning)
            if len(res_arr.shape) == 1:
                res_arr = res_arr[0:target_len]
            else:
                res_arr = res_arr[0:target_len, :]
        return res_arr

    @staticmethod
    def make_datetime(series):
        """Ensure that pd series is a datetime series with dt accessor"""
        if not hasattr(series, 'dt'):
            series = pd.to_datetime(pd.Series(series))
        return series

    @staticmethod
    def get_time_interval(time_index):
        """Get the time interval.

        Parameters
        ----------
        time_index : pd.series
            Datetime series. Must have a dt attribute to access datetime
            properties (added using make_datetime method).

        Returns
        -------
        time_interval : int:
            This value is the number of indices over which an hour is counted.
            So if the timestep is 0.5 hours, time_interval is 2.
        """

        time_index = SAM.make_datetime(time_index)
        x = time_index.dt.hour.diff()
        time_interval = 0

        # iterate through the hourly time diffs and count indices between flips
        for t in x[1:]:
            if t == 1.0:
                time_interval += 1
                break
            elif t == 0.0:
                time_interval += 1
        return int(time_interval)

    @property
    def cf_mean(self):
        """Get mean capacity factor (fractional) from SAM.

        Returns
        -------
        output : float
            Mean capacity factor (fractional).
        """
        return self.ssc.data_get_number(self.data, 'capacity_factor') / 100

    @property
    def cf_profile(self):
        """Get hourly capacity factor (frac) profile in orig timezone.

        Returns
        -------
        cf_profile : np.ndarray
            1D numpy array of capacity factor profile.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return self.gen_profile / self.parameters['system_capacity']

    @property
    def annual_energy(self):
        """Get annual energy generation value in kWh from SAM.

        Returns
        -------
        output : float
            Annual energy generation (kWh).
        """
        return self.ssc.data_get_number(self.data, 'annual_energy')

    @property
    def energy_yield(self):
        """Get annual energy yield value in kwh/kw from SAM.

        Returns
        -------
        output : float
            Annual energy yield (kwh/kw).
        """
        return self.ssc.data_get_number(self.data, 'kwh_per_kw')

    @property
    def gen_profile(self):
        """Get AC inverter power generation profile (orig timezone) in kW.

        Returns
        -------
        output : np.ndarray
            1D array of hourly AC inverter power generation in kW.
            Datatype is float32 and array length is 8760*time_interval.
        """
        gen = np.array(self.ssc.data_get_array(self.data, 'gen'),
                       dtype=np.float32)
        # Roll back to native timezone if resource meta has a timezone
        if hasattr(self, '_meta'):
            if self._meta is not None:
                if 'timezone' in self.meta:
                    gen = np.roll(gen, -1 * int(self.meta['timezone'] *
                                                self.time_interval))
        return gen

    @property
    def ppa_price(self):
        """Get PPA price ($/MWh). Native units are cents/kWh."""
        return self.ssc.data_get_number(self.data, 'ppa') * 10

    @property
    def lcoe_fcr(self):
        """Get LCOE ($/MWh). Native units are $/kWh, mult by 1000 for $/MWh."""
        return self.ssc.data_get_number(self.data, 'lcoe_fcr') * 1000

    def execute(self, module_to_run, close=True):
        """Execute a single SAM simulation core by module name.

        Parameters
        ----------
        module_to_run : str
            SAM module name (e.g., 'pvwattsv5', 'tcsmolten_salt', 'windpower',
            'singleowner', 'lcoefcr'...)
        close : boolean
            close=True (default) runs a single simulation and clears the data,
            storing only the requested outputs as an attribute. If this is
            False, the simulation core is not cleared and self.ssc can be
            passed to downstream modules. In this case, output collection is
            also not executed.
        """

        logger.debug('Running SAM module "{}" for site #{}'
                     .format(module_to_run, self.site))
        module = self.ssc.module_create(module_to_run.encode())
        self.ssc.module_exec_set_print(0)
        if self.ssc.module_exec(module, self.data) == 0:
            msg = ('SAM Simulation Error in "{}" for site #{}'
                   .format(module_to_run, self.site))
            logger.exception(msg)
            idx = 0
            while msg is not None:
                msg = self.ssc.module_log(module, idx)
                raise SAMExecutionError('SAM error message: "{}"'
                                        .format(msg.decode('utf-8')))
                logger.exception(msg)
                idx = idx + 1
            raise Exception(msg)
        self.ssc.module_free(module)

        if close is True:
            self.outputs = self.collect_outputs()
            self.ssc.data_free(self.data)

    def collect_outputs(self):
        """Collect SAM output_request.

        Returns
        -------
        output : list
            Zipped list of output requests (self.output_request) and SAM
            numerical results from the respective result functions.
        """
        results = {}
        for request in self.output_request:
            if request == 'cf_mean':
                results[request] = self.cf_mean
            elif request == 'cf_profile':
                results[request] = self.cf_profile
            elif request == 'annual_energy':
                results[request] = self.annual_energy
            elif request == 'energy_yield':
                results[request] = self.energy_yield
            elif request == 'gen_profile':
                results[request] = self.gen_profile
            elif request == 'ppa_price':
                results[request] = self.ppa_price
            elif request == 'lcoe_fcr':
                results[request] = self.lcoe_fcr

        return results


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
                                    module=points_control.project_points.tech)

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


class Economic(SAM):
    """Base class for SAM economic models.
    """
    MODULE = None

    def __init__(self, ssc, data, parameters, site_parameters=None,
                 output_request='lcoe_fcr'):
        """Initialize a SAM economic model object.

        Parameters
        ----------
        ssc : PySSC() | None
            Python SAM Simulation Core (SSC) object. Can be passed from a
            technology generation class after the SAM technology generation
            simulation has been run. This can be None, signifying that a new
            LCOE analysis is to be performed, not based on a SAM generation
            instance.
        data : PySSC.data_create() | None
            SSC data creation object. If passed from a technology generation
            class, do not run ssc.data_free(data) until after the Economic
            model has been run. This can be None, signifying that a new
            LCOE analysis is to be performed, not based on a SAM generation
            instance.
        parameters : dict | ParametersManager()
            Site-agnostic SAM model input parameters.
        site_parameters : dict
            Optional set of site-specific parameters to complement the
            site-agnostic 'parameters' input arg. Must have an 'offshore'
            column with boolean dtype if running ORCA.
        output_request : list | tuple | str
            Requested SAM output(s) (e.g., 'ppa_price', 'lcoe_fcr').
        """

        # set attribute to store site number
        self.site = None

        if ssc is None and data is None:
            # SAM generation simulation core not passed in. Create new SSC.
            self._ssc = PySSC()
            self._data = self._ssc.data_create()
        else:
            # Received SAM generation SSC.
            self._ssc = ssc
            self._data = data

        # check if offshore wind
        offshore = False
        if site_parameters is not None:
            if 'offshore' in site_parameters:
                offshore = bool(site_parameters['offshore'])

        if isinstance(output_request, (list, tuple)):
            self.output_request = output_request
        else:
            self.output_request = (output_request,)

        # Use Parameters class to manage inputs, defaults, and requirements.
        if isinstance(parameters, ParametersManager):
            self.parameters = parameters
        elif isinstance(parameters, dict) and offshore:
            # use parameters manager for offshore but do not verify or
            # set defaults (ORCA handles this)
            self.parameters = ParametersManager(parameters, self.module,
                                                verify=False, set_def=False)
        else:
            self.parameters = ParametersManager(parameters, self.module)

        # handle site-specific parameters
        if offshore:
            # offshore ORCA parameters will be handled seperately
            self._site_parameters = site_parameters
        # Non-offshore parameters can be added to ParametersManager class
        else:
            self.parameters.update(site_parameters)

    def execute(self, module_to_run, close=True):
        """Execute a SAM economic model calculation.
        """
        self.set_parameters()
        super().execute(module_to_run, close=close)

    @staticmethod
    def parse_sys_cap(site, inputs, site_df):
        """Find the system capacity variable in either inputs or df.

        Parameters
        ----------
        site : int
            Site gid.
        inputs : dict
            Generic system inputs (not site-specific).
        site_df : pd.DataFrame
            Site-specific inputs table with index = site gid's

        Returns
        -------
        sys_cap : int | float
            System nameplate capacity in native units (SAM is kW, ORCA is MW).
        """

        if ('system_capacity' not in inputs and
                'turbine_capacity' not in inputs and
                'system_capacity' not in site_df and
                'turbine_capacity' not in site_df):
            raise SAMExecutionError('Input parameter "system_capacity" '
                                    'or "turbine_capacity" '
                                    'must be included in the SAM config '
                                    'inputs or site-specific inputs in '
                                    'order to calculate annual energy '
                                    'yield for LCOE. Received the following '
                                    'inputs, site_df:\n{}\n{}'
                                    .format(inputs, site_df.head()))

        if 'system_capacity' in inputs:
            sys_cap = inputs['system_capacity']
        elif 'turbine_capacity' in inputs:
            sys_cap = inputs['turbine_capacity']
        elif 'system_capacity' in site_df:
            sys_cap = site_df.loc[site, 'system_capacity']
        elif 'turbine_capacity' in site_df:
            sys_cap = site_df.loc[site, 'turbine_capacity']

        return sys_cap

    @classmethod
    def reV_run(cls, points_control, site_df, output_request='lcoe_fcr'):
        """Execute SAM simulations based on a reV points control instance.

        Parameters
        ----------
        points_control : config.PointsControl
            PointsControl instance containing project points site and SAM
            config info.
        site_df : pd.DataFrame
            Dataframe of site-specific input variables. Row index corresponds
            to site number/gid (via df.loc not df.iloc), column labels are the
            variable keys that will be passed forward as SAM parameters.
        output_request : list | tuple | str
            Output(s) to retrieve from SAM.

        Returns
        -------
        out : dict
            Nested dictionaries where the top level key is the site index,
            the second level key is the variable name, second level value is
            the output variable value.
        """

        out = {}

        calc_aey = False
        if 'annual_energy' not in site_df:
            # annual energy yield has not been input, flag to calculate
            site_df.loc[:, 'annual_energy'] = np.nan
            calc_aey = True

        for site in points_control.sites:
            # get SAM inputs from project_points based on the current site
            config, inputs = points_control.project_points[site]

            # check to see if offshore
            offshore = False
            if 'offshore' in site_df:
                offshore = site_df.loc[site, 'offshore']

            # calculate the annual energy yield if not input
            if calc_aey and not offshore:
                if site_df.loc[site, 'capacity_factor'] > 1:
                    warn('Capacity factor > 1. Dividing by 100.')
                    cf = site_df.loc[site, 'capacity_factor'] / 100
                else:
                    cf = site_df.loc[site, 'capacity_factor']

                # get the system capacity
                sys_cap = cls.parse_sys_cap(site, inputs, site_df)

                # Calc annual energy, mult by 8760 to convert kW to kWh
                aey = sys_cap * cf * 8760

                # add aey to site-specific inputs
                site_df.loc[site, 'annual_energy'] = aey

            # iterate through requested sites.
            sim = cls(ssc=None, data=None, parameters=inputs,
                      site_parameters=dict(site_df.loc[site, :]),
                      output_request=output_request)
            sim.execute(cls.MODULE)
            out[site] = sim.outputs

            logger.debug('Outputs for site {} with config "{}", \n\t{}...'
                         .format(site, config, str(out[site])[:100]))
        return out


class LCOE(Economic):
    """SAM LCOE model.
    """
    MODULE = 'lcoefcr'

    def __init__(self, ssc, data, parameters, site_parameters=None,
                 output_request=('lcoe_fcr',)):
        """Initialize a SAM LCOE economic model object.
        """
        super().__init__(ssc, data, parameters,
                         site_parameters=site_parameters,
                         output_request=output_request)

    def execute(self, module_to_run, close=True):
        """Execute a SAM economic model calculation.
        """
        # check to see if there is an offshore flag and set for this run
        offshore = False
        if hasattr(self, '_site_parameters'):
            if 'offshore' in self._site_parameters:
                offshore = bool(self._site_parameters['offshore'])

        if offshore:
            # execute ORCA here for offshore wind LCOE
            out = self.ORCA_LCOE(self.parameters, self._site_parameters)
            self.outputs = {'lcoe_fcr': out}
        else:
            # run SAM LCOE normally for non-offshore technologies
            super().execute(module_to_run, close=close)

    @staticmethod
    def ORCA_LCOE(config_inputs, site_parameters):
        """Execute an ORCA LCOE calculation for single-site offshore wind.

        Parameters
        ----------
        config_inputs : dict | ParametersManager
            System/technology configuration inputs (non-site-specific).
        site_parameters : dict | pd.DataFrame
            Site-specific inputs.

        Returns
        -------
        lcoe_result : float
            LCOE value. Units: $/MWh.
        """

        ORCA_arg_map = {'capacity_factor': 'gcf', 'cf': 'gcf'}

        # extract config inputs as dict if ParametersManager was received
        if isinstance(config_inputs, ParametersManager):
            config_inputs = config_inputs.parameters

        # convert site parameters to dataframe if necessary
        if not isinstance(site_parameters, pd.DataFrame):
            site_parameters = pd.DataFrame(site_parameters, index=(0,))

        # rename any SAM kwargs to match ORCA requirements
        site_parameters = site_parameters.rename(index=str,
                                                 columns=ORCA_arg_map)

        # make an ORCA tech system instance
        system = ORCASystem(config_inputs)

        # make a site-specific data structure
        orca_data_struct = ORCAData(site_parameters)

        # calculate LCOE
        lcoe_result = system.lcoe(orca_data_struct)
        return lcoe_result[0]


class SingleOwner(Economic):
    """SAM single owner economic model.
    """
    MODULE = 'singleowner'

    def __init__(self, ssc, data, parameters, site_parameters=None,
                 output_request=('ppa_price',)):
        """Initialize a SAM single owner economic model object.
        """
        super().__init__(ssc, data, parameters,
                         site_parameters=site_parameters,
                         output_request=output_request)
