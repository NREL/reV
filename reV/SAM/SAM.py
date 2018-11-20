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


class ResourceManager:
    """Class to manage SAM input resource data."""

    def __init__(self, res, project_points, var_list=None):
        """Initialize a resource data iterator object for one or more sites.

        Parameters
        ----------
        res : reV.handlers.resource
            reV resource handler instance.
        project_points : config.ProjectPoints
            ProjectPoints instance with sites and configs.
        """

        self._var_list = var_list
        sites = project_points.sites

        if 'wind' in project_points.tech.lower():
            # get WTK wind data.
            self.res_type = 'wind'
            self.wind(res, project_points)

            # wind sometimes needs pressure to be converted to ATM
            self.convert_pressure()

        else:
            # get NSRDB solar data
            self.res_type = 'solar'
            self.retrieve(res, sites)

        self._N = None

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
            return self.meta.iloc[self._i].name, df, self.meta.iloc[self._i]
        else:
            # iter attribute equal or greater than iter limit
            raise StopIteration

    @property
    def var_list(self):
        """Get the resource variable list."""
        if self._var_list is None and self.res_type == 'wind':
            self._var_list = ('pressure', 'temperature', 'winddirection',
                              'windspeed')
        elif self._var_list is None and self.res_type == 'solar':
            self._var_list = ('dni', 'dhi', 'wind_speed', 'air_temperature')

        return self._var_list

    @property
    def N(self):
        """Get the iterator limit."""
        if self._N is None:
            self._N = len(self.meta)
        return self._N

    def wind(self, res, project_points, h_var='wind_turbine_hub_ht'):
        """Setup wind resource with consideration for multi-hub-height interp.

        Parameters
        ----------
        res : reV.handlers.resource
            reV resource handler instance.
        project_points : config.ProjectPoints
            ProjectPoints instance with sites and their corresponding configs.
        h_var : str
            Name of the hub height variable in the SAM input configs.
        """

        # get the config dict of format: config_dict[config_id] = SAM_config
        config_dict = project_points.sam_configs
        # get list of config ID's
        config_keys = list(config_dict.keys())

        if len(config_keys) == 1:
            # only one set of SAM inputs, no problem!
            h = config_dict[config_keys[0]][h_var]
            self.retrieve(res, project_points.sites, h=h)
        else:
            self.wind_mult_hh(res, project_points, config_dict, config_keys,
                              h_var)

    def wind_mult_hh(self, res, project_points, config_dict, config_keys,
                     h_var):
        """Set wind resource for multi-hub-height project points.

        Parameters
        ----------
        res : reV.handlers.resource
            reV resource handler instance.
        project_points : config.ProjectPoints
            ProjectPoints instance with sites and configs.
        config_dict : dict
            Multi-config dictionary from ProjectPoints. Keys are config ID's
            and values are SAM input parameter dictionaries.
        config_keys : list
            List of config ID's.
        h_var : str
            Name of the hub height variable in the SAM input configs.
        """

        h_to_config = {}
        for i, con in enumerate(config_keys):
            # make a dict (h_to_config) of lists of unique hub heights
            # (dict keys) corresponding to the list of implementing
            # config ID's (dict values)
            h = config_dict[con][h_var]
            if h in h_to_config:
                h_to_config[h] += [con]
            else:
                h_to_config[h] = [con]
        logger.debug('Hub height to config dict: {}'.format(h_to_config))

        # There are multiple configs, potentially multiple hub heights
        # init list that will save correct order of sites as they are added
        all_sites = []
        for i, (h, configs) in enumerate(h_to_config.items()):
            # iter thru unique hub heights with potentially multiple configs
            new_sites = []

            for con in configs:
                # add new sites corresponding to configs at a single hub h
                con_sites = project_points.get_sites_from_config(con)
                new_sites += con_sites
                new_sites = sorted(new_sites, key=float)

            all_sites += new_sites
            all_sites = sorted(all_sites, key=float)

            # retrieve data from the sites corresponding to the config(s) and
            # hub height. only do as many sites as necessary per hub height
            # to minimize redundant interpolations.
            logger.info('Retriving sites {} at hub height: {}m'
                        .format(new_sites, h))
            meta, time_index, mult_res = self.retrieve(res,
                                                       new_sites,
                                                       h=h,
                                                       option='return')
            if i == 0:
                # first resource import, initialize object attributes
                self.meta = meta
                self.time_index = time_index
                self.mult_res = mult_res
            else:
                # sort and append new data to attributes
                self.meta = self.meta.append(meta).sort_index()
                for j, site in enumerate(new_sites):
                    # find location of new site in the sorted site list
                    loc = all_sites.index(site)
                    for var in self.var_list:
                        # insert new site data into object attribute for
                        # all variable and all sites.
                        self.mult_res[var] = np.insert(self.mult_res[var],
                                                       loc,
                                                       mult_res[var][:, j],
                                                       axis=1)

    def retrieve(self, res, sites, h=None, option='attribute'):
        """Setup a multi-site resource dictionary from a reV handler instance.

        Parameters
        ----------
        res : reV.handlers.resource
            reV resource handler instance.
        sites : list | iter | slice | int
            Site list to retrieve.
        h : int | float
            Hub height for wind. If set, interpolation may be performed.
        option : str
            Can be either 'attribute' to set results as object attributes or
            'return' to return values outside this method for additional
            processing.

        Returns
        -------
        meta : pd.DataFrame
            Multiple-site meta data, each index is a requested site.
        time_index : pd.DataFrame
            Single year time_index (same for all sites).
        mult_res : dict
            Multiple-site resource. Keys are variables in var_list. Each
            item contains a dataframe with columns for each requested site.
        """

        mult_res = {}

        meta = res['meta', sites]
        time_index = res['time_index', :]

        for var in self.var_list:
            if self.res_type == 'wind' and h is not None:
                # request special ds names for wind hub heights
                var_hm = var + '_{}m'.format(h)
                mult_res[var] = res[var_hm, :, sites]
            else:
                # non-wind needs no special dataset name formatting
                mult_res[var] = res[var, :, sites]

            if len(mult_res[var].shape) < 2:
                # ensure that the array is 2D to ease downstream indexing
                mult_res[var] = mult_res[var].reshape(
                    (mult_res[var].shape[0], 1))

        if option == 'attribute':
            self.meta = meta
            self.time_index = time_index
            self.mult_res = mult_res
        elif option == 'return':
            return meta, time_index, mult_res

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
            logger.warning('Input parameters for {} SAM module need to '
                           'be input as a dictionary but were input as '
                           'a {}'.format(self.module, type(p)))
            logger.warning('No input parameters specified for '
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
                logger.warning('Setting default value for "{}"'.format(new))

    def require_resource_file(self, res_type):
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
            logger.debug('Verifying input parameter: "{}" '
                         'of viable types: {}'.format(name, dtypes))
            if name not in self.parameters.keys():
                logger.warning('SAM input parameters must contain "{}"'
                               .format(name))
                missing_inputs = True
            else:
                p = self.parameters[name]
                if not any([isinstance(p, d) for d in dtypes]):
                    logger.warning('SAM input parameter "{}" must be of '
                                   'type {} but is of type {}'
                                   .format(name, dtypes, type(p)))
        if missing_inputs:
            self.set_defaults()


class SAM:
    """Base class for SAM derived generation.
    """
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

        logger.debug('SAM base class initializing...')

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
            self.time_interval = self.get_time_interval(resource['time_index'])
            self.set_time_index(resource['time_index'])

    def set_parameters(self, keys_to_set='all'):
        """Set SAM inputs using either a subset of keys or all parameter keys.

        Parameters
        ----------
        keys_to_set : str, list, or iterable
            This defaults to 'all', which will set all parameters in the
            dictionary self.parameters. Otherwise, only parameter keys in
            keys_to_set will be set.
        """
        logger.debug('Setting SAM input parameters.')
        if keys_to_set == 'all':
            logger.debug(self.parameters.keys())
            keys_to_set = self.parameters.keys()

        for key in keys_to_set:
            logger.debug('Setting parameter: {} = {}...'
                         .format(key, str(self.parameters[key])[:20]))
            logger.debug('Parameter {} has type: {}'
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
            logger.debug('Setting {} meta data.'.format(var))
            self.ssc.data_set_number(self.res_data, var_map[var],
                                     self.meta[var])

        self.site = self.meta.name

    def set_time_index(self, time_index, time_vars=('year', 'month', 'day',
                                                    'hour', 'minute')):
        """Set the SAM time index variables.

        Parameters
        ----------
        time_index : pd.series
            Datetime series. Must have a dt attribute to access datetime
            properties.
        time_vars : list | tuple
            List of time variable names to set.
        """
        for var in time_vars:
            logger.debug('Setting {} time index data.'.format(var))
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
            logger.info('Running SAM module "{}" for site #{}'
                        .format(m, self.site))
            module = self.ssc.module_create(m.encode())
            self.ssc.module_exec_set_print(0)
            if self.ssc.module_exec(module, self.data) == 0:
                msg = ('SAM Simulation Error in "{}" for site #{}'
                       .format(m, self.site))
                logger.error(msg)
                idx = 1
                msg = self.ssc.module_log(module, 0)
                while msg is not None:
                    logger.error('{}'.format(msg.decode('utf-8')))
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

    @classmethod
    def reV_run(cls, resources, project_points, output_request=('cf_mean',)):
        """Execute a SAM simulation for a single site with default reV outputs.

        Parameters
        ----------
        resources : ResourceManager
            ResourceManager instance that emulates an iterator.
        project_points : config.ProjectPoints
            ProjectPoints instance containing site and SAM config info.
        output_request : list | tuple
            Outputs to retrieve from SAM.

        Returns
        -------
        out : dict
            Nested dictionaries where the top level key is the site index,
            the second level key is the variable name, second level value is
            the output variable value.
        """

        out = {}

        for site, res_df, meta in resources:
            # get SAM inputs from project_points based on the current site
            config, inputs = project_points[site]
            # iterate through requested sites.
            sim = cls(resource=res_df, meta=meta, parameters=inputs,
                      output_request=output_request)
            sim.execute(cls.MODULE)
            out[site] = sim.outputs

            logger.info('Site {} with config "{}", \n\toutputs: {}...'
                        .format(site, config, str(out[site])[:20]))

        return out


class Solar(SAM):
    """Base Class for Solar generation from SAM
    """

    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None):
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

        # don't pass resource to base class, set in set_nsrdb instead.
        super().__init__(resource=None, meta=meta, parameters=parameters,
                         output_request=output_request)
        logger.debug('SAM Solar class initializing...')

        if resource is None or meta is None:
            # if no resource input data is specified, you need a resource file
            self.parameters.require_resource_file(res_type='solar')

        elif resource is not None and meta is not None:
            logger.debug('Setting resource and meta data for Solar.')
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
                logger.debug('Setting {} resource data.'.format(var))
                self.ssc.data_set_array(self.res_data, var_map[var],
                                        np.roll(resource[var],
                                                int(self.meta['timezone'] *
                                                    self.time_interval)))

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
        logger.debug('SAM PV class initializing...')

    def execute(self, modules_to_run, close=True):
        """Execute a SAM PV solar simulation.
        """
        self.set_parameters()

        if 'lcoe_fcr' in self.output_request:
            # econ outputs requested, run LCOE model after pvwatts.
            super().execute(modules_to_run, close=False)
            lcoe = LCOE(self.ssc, self.data, self.parameters,
                        self.output_request)
            lcoe.execute(LCOE.MODULE)
            self.outputs = lcoe.outputs
        else:
            super().execute(modules_to_run, close=close)


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
        logger.debug('SAM CSP class initializing...')

    def execute(self, modules_to_run, close=True):
        """Execute a SAM CSP solar simulation.
        """
        self.set_parameters()

        if 'ppa_price' in self.output_request:
            # econ outputs requested, run single owner model after csp.
            super().execute(modules_to_run, close=False)
            so = SingleOwner(self.ssc, self.data, self.parameters,
                             self.output_request)
            so.execute(SingleOwner.MODULE)
            self.outputs = so.outputs
        else:
            super().execute(modules_to_run, close=close)


class Wind(SAM):
    """Base class for Wind generation from SAM
    """

    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None):
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

        # don't pass resource to base class, set in set_wtk instead.
        super().__init__(resource=None, meta=meta, parameters=parameters,
                         output_request=output_request)
        logger.debug('SAM Wind class initializing...')

        if resource is None or meta is None:
            # if no resource input data is specified, you need a resource file
            self.parameters.require_resource_file(res_type='wind')

        elif resource is not None and meta is not None:
            logger.debug('Setting resource and meta data for Wind.')
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

        # must be set as matrix in [temp, pres, speed, direction] order
        self.ssc.data_set_matrix(self.res_data, 'data',
                                 resource[['temperature', 'pressure',
                                           'windspeed',
                                           'winddirection']].values)

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
        logger.debug('SAM land-based wind class initializing...')

    def execute(self, modules_to_run, close=True):
        """Execute a SAM land based wind simulation.
        """
        self.set_parameters()

        if 'lcoe_fcr' in self.output_request:
            # econ outputs requested, run LCOE model after pvwatts.
            super().execute(modules_to_run, close=False)
            lcoe = LCOE(self.ssc, self.data, self.parameters,
                        self.output_request)
            lcoe.execute(LCOE.MODULE)
            self.outputs = lcoe.outputs
        else:
            super().execute(modules_to_run, close=close)


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
        logger.debug('SAM offshore wind class initializing...')


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

        logger.debug('SAM Economomic class initializing...')

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

    def execute(self, modules_to_run, close=True):
        """Execute a SAM single owner model calculation.
        """
        self.set_parameters()
        super().execute(modules_to_run, close=close)


class LCOE(Economic):
    """SAM LCOE model.
    """
    MODULE = 'lcoefcr'

    def __init__(self, ssc, data, parameters, output_request):
        """Initialize a SAM LCOE economic model object.
        """
        super().__init__(ssc, data, parameters, output_request)
        logger.debug('SAM LCOE class initializing...')


class SingleOwner(Economic):
    """SAM single owner economic model.
    """
    MODULE = 'singleowner'

    def __init__(self, ssc, data, parameters, output_request):
        """Initialize a SAM single owner economic model object.
        """
        super().__init__(ssc, data, parameters, output_request)
        logger.debug('SAM LCOE class initializing...')
