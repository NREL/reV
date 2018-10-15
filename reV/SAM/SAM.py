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
            self._logger.warning('No input parameters specified for '
                                 '{} SAM module'.format(self.module))
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
                          'singleowner']

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

    @staticmethod
    def setup_resource_df(f, i, var_list):
        """Setup a single-site resource dataframe from an hdf.

        Parameters
        ----------
        f : str
            File path and name of resource data hdf.
        i : int
            Single site index to retrieve.
        var_list : list
            List of strings of variables names to pull from the resource file.

        Returns
        -------
        res : pd.DataFrame
            Single-site resource dataframe with columns from the var_list plus
            time_index.
        meta : pd.DataFrame
            Single-site meta data.
        """

        res = pd.DataFrame()
        with resource.NSRDB(f) as r:
            meta = r['meta', i]

            time_index = r['time_index', :]
            res['time_index'] = time_index

            for var in var_list:
                res[var] = r[var, :, i]

        return res, meta

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
            self._logger.debug('Setting parameter: {} = {}'
                               ''.format(key, self.parameters[key]))
            self._logger.debug('Parameter {} has type: {}'
                               ''.format(key, type(self.parameters[key])))
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

    def set_resource(self, resource, var_list, name):
        """Set SSC solar resource data arrays.

        Parameters
        ----------
        resource : pd.DataFrame
            2D table with resource data. Available columns must have var_list.
        var_list : list
            List of resource variable names to set. Must correspond with
            column names in resource.
        name : str
            Name of resource dataset for SAM. examples: 'solar_resource_data'
        """

        # set meta data
        self.set_meta()
        self.time_interval = self.get_time_interval(resource['time_index'])
        self.set_time_index(resource['time_index'])

        # map resource data names to SAM required data names
        var_map = {'dni': 'dn',
                   'dhi': 'df',
                   'wind_speed': 'wspd',
                   'air_temperature': 'tdry'}

        # set resource variables
        for var in var_list:
            self._logger.debug('Setting {} resource data.'.format(var))
            self.ssc.data_set_array(self.res_data, var_map[var],
                                    np.roll(resource[var],
                                            self.meta['timezone'] *
                                            self.time_interval))

        # add resource data to self.data and clear
        self.ssc.data_set_table(self.data, name.encode(), self.res_data)
        self.ssc.data_free(self.res_data)

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
        time_interval = -1

        # iterate through the hourly time diffs and count indices between flips
        for t in x:
            if t == 0.0 and time_interval < 0:
                time_interval = 0
            elif t == 0.0 and time_interval >= 0:
                time_interval += 1
                break
            elif t == 1.0:
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
            self._logger.info('Running SAM module: {}'.format(m))
            module = self.ssc.module_create(m.encode())
            self.ssc.module_exec_set_print(0)
            if self.ssc.module_exec(module, self.data) == 0:
                print('SAM Simulation Error in {}'.format(m))
                idx = 1
                msg = self.ssc.module_log(module, 0)
                while msg is not None:
                    print('    : ' + msg.decode('utf-8'))
                    msg = self.ssc.module_log(module, idx)
                    idx = idx + 1
                raise Exception("SAM Simulation Error in {}".format(m))
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
            var_list = ['dni', 'dhi', 'wind_speed', 'air_temperature']
            self.set_resource(resource, var_list, 'solar_resource_data')


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
            pv_lcoe = LCOE(self.ssc, self.data, self.parameters,
                           self.output_request)
            pv_lcoe.execute()
            self.outputs = pv_lcoe.outputs
        else:
            super().execute(self.module, close=True)

    @classmethod
    def reV_run(cls, site, res_file, inputs):
        """Execute a SAM simulation for a single site with default reV outputs.

        Parameters
        ----------
        site : int
            Site index from the resource file (res_file) to run for SAM.
        res_file : str
            H5 file containing NSRDB resource data.
        inputs : dict
            Required SAM simulation input parameters.

        Returns
        -------
        output : list
            Zipped list of output requests (self.output_request) and SAM
            numerical results from the respective result functions.
        """
        res, meta = cls.setup_resource_df(res_file, site, ['dni', 'dhi',
                                                           'wind_speed',
                                                           'air_temperature'])
        sim = cls(resource=res, meta=meta, parameters=inputs,
                  output_request=['cf_mean', 'cf_profile',
                                  'annual_energy', 'energy_yield',
                                  'gen_profile', 'lcoe_fcr'])
        sim.execute()
        return sim.outputs


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
            csp_so = SingleOwner(self.ssc, self.data, self.parameters,
                                 self.output_request)
            csp_so.execute()
            self.outputs = csp_so.outputs
        else:
            super().execute(self.module, close=True)

    @classmethod
    def reV_run(cls, site, res_file, inputs):
        """Execute a SAM simulation for a single site with default reV outputs.

        Parameters
        ----------
        site : int
            Site index from the resource file (res_file) to run for SAM.
        res_file : str
            H5 file containing NSRDB resource data.
        inputs : dict
            Required SAM simulation input parameters.

        Returns
        -------
        output : list
            Zipped list of output requests (self.output_request) and SAM
            numerical results from the respective result functions.
        """
        res, meta = cls.setup_resource_df(res_file, site, ['dni', 'dhi',
                                                           'wind_speed',
                                                           'air_temperature'])
        sim = cls(resource=res, meta=meta, parameters=inputs,
                  output_request=['cf_mean', 'cf_profile',
                                  'annual_energy', 'energy_yield',
                                  'gen_profile', 'ppa_price'])
        sim.execute()
        return sim.outputs


class Wind(SAM):
    """Base class for Wind generation from SAM
    """

    def __init__(self, resource, meta, parameters, output_request):
        """Initialize a SAM wind object.
        """
        super().__init__(resource, meta, parameters, output_request)

        self.set_meta()
        self.set_time_index(parameters['time_index'])
        data = self.ssc.data_create()
        self.ssc.data_set_table(data, 'wind_resource_data', self.data)
        self.ssc.data_free(self.data)


class LandBasedWind(Wind):
    """Onshore wind generation
    """

    def __init__(self, resource, meta, parameters, output_request):
        """Initialize a SAM land based wind object.
        """
        super().__init__(resource, meta, parameters, output_request)


class OffshoreWind(Wind):
    """Offshore wind generation
    """

    def __init__(self, resource, meta, parameters, output_request):
        """Initialize a SAM offshore wind object.
        """
        super().__init__(resource, meta, parameters, output_request)


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
