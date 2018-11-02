#!/usr/bin/env python
"""reV-to-SAM interface module.

Relies heavily upon the SAM Simulation Core (SSC) API module (sscapi) from the
SAM software development kit (SDK).
"""
import json
import logging
import numpy as np
import os
import pandas as pd

from reV.SAM.PySSC import PySSC
from reV.handlers import resource


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


class ResourceManager:
    """Class to manage SAM input resource data."""

    def __init__(self, res, sites, h=None):
        """Initialize a resource data iterator object for one or more sites.

        Parameters
        ----------
        res : reV.handlers.resource
            reV resource handler instance.
        sites : list | iter | slice | int
            Resource site indices to retrieve.
        h : int, float, or None
            Hub height for WTK data, None for NSRDB.
        """

        log_name = '{}.{}'.format(self.__module__, self.__class__.__name__)
        self._logger = logging.getLogger(log_name)

        self.index = sites
        self._N = len(self.index)

        if h is None:
            # no hub height specified, get NSRDB solar data
            self.res_type = 'solar'
            var_list = ['dni', 'dhi', 'wind_speed', 'air_temperature']
            self.retrieve(res, sites, var_list)
        elif isinstance(h, (int, float)):
            # hub height specified, get WTK wind data.
            self.res_type = 'wind'
            var_list = ['pressure', 'temperature', 'winddirection',
                        'windspeed']
            self.retrieve(res, sites, var_list, h=h)

            # wind sometimes needs pressure to be converted to ATM
            self.convert_pressure()
        else:
            raise TypeError('Hub height arg "h" specified but of wrong type: '
                            '{}. Could not retrieve WTK or NSRDB resource '
                            'data.'.format(type(h)))

    def __iter__(self):
        """Iterator initialization dunder."""
        self._i = -1
        return self

    def __next__(self):
        """Iterate through and return next site resource data.

        Returns
        -------
        ind : int
            Requested site index - matches the sites init argument.
        df : pd.DataFrame
            Single-site resource dataframe with columns from the var_list plus
            time_index.
        meta : pd.DataFrame
            Single-site meta data.
        """

        self._i += 1

        if self._i < self.N:
            # setup single site dataframe by indexing and slicing mult_res
            df = pd.DataFrame({key: value[:, self._i]
                              for key, value in self.mult_res.items()})
            df['time_index'] = self.time_index
            return self.index[self._i], df, self.meta.iloc[self._i]
        else:
            # iter attribute equal or greater than iter limit
            raise StopIteration

    @property
    def N(self):
        """Get the iterator limit."""
        return self._N

    @property
    def index(self):
        """Get the resource-native site index (requested site indices)."""
        return self._index

    @index.setter
    def index(self, sites):
        """Set the resource-native site index (requested site indices)."""
        if isinstance(sites, slice):
            self._index = list(range(*sites.indices(sites.stop)))
        elif isinstance(sites, int):
            self._index = [sites]
        else:
            self._index = list(sites)

    def retrieve(self, res, sites, var_list, h=None):
        """Setup a multi-site resource dictionary from a reV handler instance.

        Parameters
        ----------
        res : reV.handlers.resource
            reV resource handler instance.
        sites : list | iter | slice | int
            Site list to retrieve.
        var_list : list
            List of strings of variables names to pull from the resource file.

        Attributes
        ----------
        self.mult_res : dict
            Multiple-site resource. Keys are variables in var_list. Each
            item contains a dataframe with columns for each requested site.
        self.meta : pd.DataFrame
            Multiple-site meta data, each index is a requested site.
        self.time_index : pd.DataFrame
            Single year time_index (same for all sites).
        """

        self.mult_res = {}

        self.meta = res['meta', sites]
        self.time_index = res['time_index', :]

        for var in var_list:
            if self.res_type == 'wind' and h is not None:
                # request special ds names for wind hub heights
                var_hm = var + '_{}m'.format(h)
                self.mult_res[var] = res[var_hm, :, sites]
            else:
                # non-wind needs no special dataset name formatting
                self.mult_res[var] = res[var, :, sites]

            if len(self.mult_res[var].shape) < 2:
                # ensure that the array is 2D to ease downstream indexing
                self.mult_res[var] = self.mult_res[var].reshape(
                    (self.mult_res[var].shape[0], 1))

    def convert_pressure(self):
        """If pressure is in resource vars, convert to ATM if in Pa."""
        if 'pressure' in self.mult_res.keys():
            if np.min(self.mult_res['pressure']) > 1e3:
                # convert pressure from Pa to ATM
                self.mult_res['pressure'] *= 9.86923e-6


class ParametersManager:
    """Class to manage SAM input parameters, requirements, and defaults."""

    def __init__(self, parameters, module):
        """Initialize the SAM input parameters class.

        Parameters
        ----------
        parameters : dict
            SAM model input parameters.
        module : str
            SAM module ('pvwatts', 'tcsmolten_salt', etc...)
        """

        log_name = '{}.{}'.format(self.__module__, self.__class__.__name__)
        self._logger = logging.getLogger(log_name)

        # set the parameters and module properties
        self._module = module
        self.parameters = parameters

        # get requirements and verify that all are satisfied
        self._requirements = self.get_requirements(self.module)
        self.verify_inputs()

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
            self._logger.warning('Input parameters for {} SAM module need to '
                                 'be input as a dictionary but were input as '
                                 'a {}'.format(self.module, type(p)))
            self._logger.warning('No input parameters specified for '
                                 '{} SAM module, defaults will be set'
                                 .format(self.module))
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

        jf = os.path.join(SAM._sam_dir, req_folder, module + '.json')

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

        jf = os.path.join(SAM._sam_dir, def_folder, module + '.json')

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
                self._logger.warning('Setting default value for "{}"'
                                     ''.format(new))

    def require_resource_file(self, res_type='solar'):
        """Enforce requirement of a resource file if res data is not input"""
        if res_type.lower() == 'solar':
            self._requirements.append(['solar_resource_file', [str]])
        elif res_type.lower() == 'wind':
            self._requirements.append(['wind_resource_filename', [str]])

        # re-verify inputs with new requirement. Will also set the default
        # resource file if one is not provided.
        self.verify_inputs()

    def verify_inputs(self):
        """Verify that required inputs are available and have correct dtype.
        Also set missing inputs to default values.

        Prints logger warnings when variables are missing, set to default,
        or are of the incorrect datatype.
        """
        missing_inputs = False
        for name, dtypes in self.requirements:
            self._logger.debug('Verifying input parameter: "{}" '
                               'of viable types: {}'.format(name, dtypes))
            if name not in self.parameters.keys():
                self._logger.warning('SAM input parameters must contain "{}"'
                                     ''.format(name))
                missing_inputs = True
            else:
                p = self.parameters[name]
                if not any([isinstance(p, d) for d in dtypes]):
                    self._logger.warning('SAM input parameter "{}" must be of '
                                         'type {} but is of type {}'
                                         ''.format(name, dtypes, type(p)))
        if missing_inputs:
            self.set_defaults()


class SAM(object):
    """Base class for SAM derived generation.
    """
    _sam_dir = os.path.dirname(os.path.realpath(__file__))
    _module = None
    _available_modules = ['pvwattsv5', 'tcsmolten_salt', 'lcoefcr',
                          'singleowner', 'windpower']

    def __init__(self, meta, parameters, output_request):
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

        log_name = '{}.{}'.format(self.__module__, self.__class__.__name__)
        self._logger = logging.getLogger(log_name)
        self._logger.debug('SAM base class initializing...')

        # set attribute to store site number
        self.site = None

        # set meta attribute as protected
        self._meta = meta

        # Initialize SAM Simulation Core (SSC) object as protected
        self._ssc = PySSC()

        # Initialize protected attribute for data setting
        self._data = self.ssc.data_create()

        # Initialize protected attribute for resource data setting
        self._res_data = self.ssc.data_create()

        # Use Parameters class to manage inputs, defaults, and requirements.
        if parameters.__class__.__name__ == 'ParametersManager':
            self.parameters = parameters
        else:
            self.parameters = ParametersManager(parameters, self.module)

        # Save output request as attribute
        self.output_request = output_request

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
        return self._module

    @module.setter
    def module(self, m):
        """Set module property."""
        if m in self._available_modules:
            self._module = m
        else:
            raise KeyError('Module is not available: {}'.format(m))

    @property
    def res_data(self):
        """Get resource data property"""
        return self._res_data

    @property
    def ssc(self):
        """Get SAM simulation core (SSC) property"""
        return self._ssc

    def set_parameters(self, keys_to_set='all'):
        """Set SAM inputs using either a subset of keys or all parameter keys.

        Parameters
        ----------
        keys_to_set : str, list, or iterable
            This defaults to 'all', which will set all parameters in the
            dictionary self.parameters. Otherwise, only parameter keys in
            keys_to_set will be set.
        """
        self._logger.debug('Setting SAM input parameters.')
        if keys_to_set == 'all':
            self._logger.debug(self.parameters.keys())
            keys_to_set = self.parameters.keys()

        for key in keys_to_set:
            self._logger.debug('Setting parameter: {} = {}...'
                               .format(key, str(self.parameters[key])[:20]))
            self._logger.debug('Parameter {} has type: {}'
                               .format(key, type(self.parameters[key])))

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

    def set_meta(self, meta_vars=['latitude', 'longitude', 'elevation']):
        """Set the base SAM meta data variables when using resource data.

        Parameters
        ----------
        meta_vars : list
            List of meta variable names to set.
        """
        self.ssc.data_set_number(self.res_data, 'tz',
                                 int(self.meta['timezone']))

        # map resource data names to SAM required data names
        var_map = {'latitude': 'lat',
                   'longitude': 'lon',
                   'elevation': 'elev'}

        for var in meta_vars:
            self._logger.debug('Setting {} meta data.'.format(var))
            self.ssc.data_set_number(self.res_data, var_map[var],
                                     self.meta[var])

        self.site = self.meta.name

    def set_time_index(self, time_index, time_vars=['year', 'month', 'day',
                                                    'hour', 'minute']):
        """Set the SAM time index variables.

        Parameters
        ----------
        time_index : pd.series
            Datetime series. Must have a dt attribute to access datetime
            properties.
        time_vars : list
            List of time variable names to set.
        """
        for var in time_vars:
            self._logger.debug('Setting {} time index data.'.format(var))
            self.ssc.data_set_array(self.res_data, var,
                                    getattr(time_index.dt, var).values)

    @staticmethod
    def get_time_interval(time_index):
        """Get the time interval.

        Returns
        -------
        time_interval : int:
            This value is the number of indices over which an hour is counted.
            So if the timestep is 0.5 hours, time_interval is 2.
        """

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
        """Get hourly capacity factor (fractional) profile from SAM.

        Returns
        -------
        cf_profile : np.ndarray
            1D numpy array of capacity factor profile.
            Datatype is float32 and array length is 8760.
        """
        gen_array = np.array(self.ssc.data_get_array(self.data, 'gen'),
                             dtype=np.float32)
        cf_profile = gen_array / self.parameters['system_capacity']
        return cf_profile

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
        """Get hourly (8760) AC inverter power generation profile in kW.

        Returns
        -------
        output : np.ndarray
            1D array of hourly AC inverter power generation in kW.
            Datatype is float32 and array length is 8760.
        """
        return np.array(self.ssc.data_get_array(self.data, 'gen'),
                        dtype=np.float32)

    @property
    def ppa_price(self):
        """Get PPA price (cents/kWh)
        """
        return self.ssc.data_get_number(self.data, 'ppa')

    @property
    def lcoe_fcr(self):
        """Get LCOE (cents/kWh).
        """
        return 100 * self.ssc.data_get_number(self.data, 'lcoe_fcr')

    def execute(self, modules_to_run, close=True):
        """Execute a SAM simulation by module name.

        Parameters
        ----------
        modules_to_run : str or list
            SAM module names (e.g., 'pvwattsv5', 'tcsmolten_salt', 'windpower',
            'singleowner', 'lcoefcr'...)
        close : boolean
            close=True (default) runs a single simulation and clears the data,
            storing only the requested outputs as an attribute. If this is
            False, the simulation core is not cleared and self.ssc can be
            passed to downstream modules. In this case, output collection is
            also not executed.
        """
        if isinstance(modules_to_run, str):
            modules_to_run = [modules_to_run]

        for m in modules_to_run:
            self._logger.info('Running SAM module "{}" for site #{}'
                              .format(m, self.site))
            module = self.ssc.module_create(m.encode())
            self.ssc.module_exec_set_print(0)
            if self.ssc.module_exec(module, self.data) == 0:
                msg = ('SAM Simulation Error in "{}" for site #{}'
                       .format(m, self.site))
                self._logger.error(msg)
                idx = 1
                msg = self.ssc.module_log(module, 0)
                while msg is not None:
                    self._logger.error('    : ' + msg.decode('utf-8'))
                    msg = self.ssc.module_log(module, idx)
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

        _kw_out = {'cf_mean': self.cf_mean,
                   'cf_profile': self.cf_profile,
                   'annual_energy': self.annual_energy,
                   'energy_yield': self.energy_yield,
                   'gen_profile': self.gen_profile,
                   'ppa_price': self.ppa_price,
                   'lcoe_fcr': self.lcoe_fcr,
                   }

        results = [_kw_out[request.lower()]
                   for request in self.output_request]

        return dict(zip(self.output_request, results))

    @classmethod
    def reV_run(cls, res_file, sites, inputs, output_request=['cf_mean']):
        """Execute a SAM simulation for a single site with default reV outputs.

        Parameters
        ----------
        res_file : str
            H5 file containing resource data.
        sites : list | iter | slice | int
            Site indices from the resource file (res_file) to run for SAM.
        inputs : dict
            Required SAM simulation input parameters.

        Returns
        -------
        out : dict
            Nested dictionaries where the top level key is the site index,
            the second level key is the variable name, second level value is
            the output variable value.
        """

        out = {}

        if 'wind_turbine_hub_ht' in inputs.keys():
            # get wind resource.
            h = inputs['wind_turbine_hub_ht']
            res = resource.WTK(res_file)
        else:
            # Wind hub height not specified, NSRDB will be retrieved.
            h = None
            res = resource.NSRDB(res_file)

        resources = ResourceManager(res, sites, h=h)

        for site, res_df, meta in resources:
            # iterate through requested sites.
            sim = cls(resource=res_df, meta=meta, parameters=inputs,
                      output_request=output_request)
            sim.execute()
            out[site] = sim.outputs
            sim._logger.debug('Site {}, outputs: {}'.format(site, out[site]))

        return out


class Solar(SAM):
    """Base Class for Solar generation from SAM
    """

    def __init__(self, resource, meta, parameters, output_request):
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
        """

        super().__init__(meta, parameters, output_request)
        self._logger.debug('SAM Solar class initializing...')

        if resource is None or meta is None:
            # if no resource input data is specified, you need a resource file
            self.parameters.require_resource_file(res_type='solar')

        elif resource is not None and meta is not None:
            self._logger.debug('Setting resource and meta data for Solar.')
            self.set_nsrdb(resource)

    def set_nsrdb(self, resource, name='solar_resource_data',
                  var_list=['dni', 'dhi', 'wind_speed', 'air_temperature']):
        """Set SSC NSRDB resource data arrays.

        Parameters
        ----------
        resource : pd.DataFrame
            2D table with resource data. Available columns must have var_list.
        name : str
            Name of resource dataset for SAM. examples: 'solar_resource_data'
        var_list : list
            List of resource variable names to set. Must correspond with
            column names in resource.
        """

        # set meta data
        self.set_meta()
        self.time_interval = self.get_time_interval(resource['time_index'])
        self.set_time_index(resource['time_index'])

        # map resource data names to SAM required data names
        var_map = {'dni': 'dn',
                   'dhi': 'df',
                   'wind_speed': 'wspd',
                   'air_temperature': 'tdry',
                   }

        # set resource variables
        for var in var_list:
            self._logger.debug('Setting {} resource data.'.format(var))
            self.ssc.data_set_array(self.res_data, var_map[var],
                                    np.roll(resource[var],
                                            int(self.meta['timezone'] *
                                                self.time_interval)))

        # add resource data to self.data and clear
        self.ssc.data_set_table(self.data, name.encode(), self.res_data)
        self.ssc.data_free(self.res_data)


class PV(Solar):
    """Photovoltaic (PV) generation with pvwattsv5.
    """

    def __init__(self, resource, meta, parameters, output_request):
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
        self.module = 'pvwattsv5'
        super().__init__(resource, meta, parameters, output_request)
        self._logger.debug('SAM PV class initializing...')

    def execute(self):
        """Execute a SAM PV solar simulation.
        """
        self.set_parameters()

        if 'lcoe_fcr' in self.output_request:
            # econ outputs requested, run LCOE model after pvwatts.
            super().execute(self.module, close=False)
            lcoe = LCOE(self.ssc, self.data, self.parameters,
                        self.output_request)
            lcoe.execute()
            self.outputs = lcoe.outputs
        else:
            super().execute(self.module, close=True)


class CSP(Solar):
    """Concentrated Solar Power (CSP) generation
    """

    def __init__(self, resource, meta, parameters, output_request):
        """Initialize a SAM concentrated solar power (CSP) object.
        """
        self.module = 'tcsmolten_salt'
        super().__init__(resource, meta, parameters, output_request)
        self._logger.debug('SAM CSP class initializing...')

    def execute(self):
        """Execute a SAM CSP solar simulation.
        """
        self.set_parameters()

        if 'ppa_price' in self.output_request:
            # econ outputs requested, run single owner model after csp.
            super().execute(self.module, close=False)
            so = SingleOwner(self.ssc, self.data, self.parameters,
                             self.output_request)
            so.execute()
            self.outputs = so.outputs
        else:
            super().execute(self.module, close=True)


class Wind(SAM):
    """Base class for Wind generation from SAM
    """

    def __init__(self, resource, meta, parameters, output_request):
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
        """

        super().__init__(meta, parameters, output_request)
        self._logger.debug('SAM Wind class initializing...')

        if resource is None or meta is None:
            # if no resource input data is specified, you need a resource file
            self.parameters.require_resource_file(res_type='wind')

        elif resource is not None and meta is not None:
            self._logger.debug('Setting resource and meta data for Wind.')
            self.set_wtk(resource)

    def set_wtk(self, resource, name='wind_resource_data'):
        """Set SSC WTK resource data arrays.

        Parameters
        ----------
        resource : pd.DataFrame
            2D table with resource data. Available columns must have var_list.
        name : str
            Name of resource dataset for SAM. examples: 'solar_resource_data'
        """

        # set meta data
        self.set_meta()
        self.time_interval = self.get_time_interval(resource['time_index'])
        self.set_time_index(resource['time_index'])

        self.ssc.data_set_array(self.res_data, 'fields', [1, 2, 3, 4])
        self.ssc.data_set_array(self.res_data, 'heights',
                                4 * [self.parameters['wind_turbine_hub_ht']])

        # must be set as matrix in [temp, pres, speed, direction] order
        self.ssc.data_set_matrix(self.res_data, 'data',
                                 resource[['temperature', 'pressure',
                                           'windspeed',
                                           'winddirection']].values)

        # add resource data to self.data and clear
        self.ssc.data_set_table(self.data, name.encode(), self.res_data)
        self.ssc.data_free(self.res_data)


class LandBasedWind(Wind):
    """Onshore wind generation
    """

    def __init__(self, resource, meta, parameters, output_request):
        """Initialize a SAM land based wind object.
        """
        self.module = 'windpower'
        super().__init__(resource, meta, parameters, output_request)
        self._logger.debug('SAM land-based wind class initializing...')

    def execute(self):
        """Execute a SAM land based wind simulation.
        """
        self.set_parameters()

        if 'lcoe_fcr' in self.output_request:
            # econ outputs requested, run LCOE model after pvwatts.
            super().execute(self.module, close=False)
            lcoe = LCOE(self.ssc, self.data, self.parameters,
                        self.output_request)
            lcoe.execute()
            self.outputs = lcoe.outputs
        else:
            super().execute(self.module, close=True)


class OffshoreWind(LandBasedWind):
    """Offshore wind generation
    """

    def __init__(self, resource, meta, parameters, output_request):
        """Initialize a SAM offshore wind object.
        """
        self.module = 'windpower'
        super().__init__(resource, meta, parameters, output_request)
        self._logger.debug('SAM offshore wind class initializing...')


class Economic(SAM):
    """Base class for SAM economic models.
    """

    def __init__(self, ssc, data, parameters, output_request):
        """Initialize a SAM economic model object.

        Parameters
        ----------
        ssc : PySSC()
            Python SAM Simulation Core (SSC) object. Can be passed from a
            technology generation class after the SAM technology generation
            simulation has been run.
        data : PySSC.data_create()
            SSC data creation object. If passed from a technology generation
            class, do not run ssc.data_free(data) until after the Economic
            model has been run.
        parameters : dict or ParametersManager()
            SAM model input parameters.
        output_request : list
            Requested SAM outputs (e.g., 'cf_mean', 'annual_energy',
            'cf_profile', 'gen_profile', 'energy_yield', 'ppa_price',
            'lcoe_fcr').
        """

        log_name = '{}.{}'.format(self.__module__, self.__class__.__name__)
        self._logger = logging.getLogger(log_name)

        self._logger.debug('SAM Economomic class initializing...')

        # set attribute to store site number
        self.site = None

        self._ssc = ssc
        self._data = data
        self.output_request = output_request

        # Use Parameters class to manage inputs, defaults, and requirements.
        if parameters.__class__.__name__ == 'ParametersManager':
            self.parameters = parameters
        else:
            self.parameters = ParametersManager(parameters, self.module)

    def execute(self):
        """Execute a SAM single owner model calculation.
        """
        self.set_parameters()
        super().execute(self.module)


class LCOE(Economic):
    """SAM LCOE model.
    """

    def __init__(self, ssc, data, parameters, output_request):
        """Initialize a SAM LCOE economic model object.
        """
        self.module = 'lcoefcr'
        super().__init__(ssc, data, parameters, output_request)
        self._logger.debug('SAM LCOE class initializing...')


class SingleOwner(Economic):
    """SAM single owner economic model.
    """

    def __init__(self, ssc, data, parameters, output_request):
        """Initialize a SAM single owner economic model object.
        """
        self.module = 'singleowner'
        super().__init__(ssc, data, parameters, output_request)
        self._logger.debug('SAM LCOE class initializing...')
