# -*- coding: utf-8 -*-
"""reV-to-SAM interface module.

Relies heavily upon the SAM Simulation Core (SSC) API module (sscapi) from the
SAM software development kit (SDK).
"""
import json
import logging
import numpy as np
import os
import pandas as pd
from warnings import warn

from reV.handlers.resource import WindResource, SolarResource, NSRDB
from reV.SAM.PySSC import PySSC
from reV.utilities.exceptions import (SAMInputWarning, SAMExecutionError,
                                      ResourceError)
from reV.utilities.slots import SlottedDict


logger = logging.getLogger(__name__)


def is_num(n):
    """Check if input is a number (returns True/False)

    Parameters
    ----------
    n : obj
        Input object

    Returns
    -------
    out : bool
        Boolean flag indicating if input is a number
    """
    try:
        float(n)
        out = True
    except Exception as _:
        out = False

    return out


def is_str(s):
    """Check if input is a string and not a number (returns True/False)

    Parameters
    ----------
    n : obj
        Input object

    Returns
    -------
    out : bool
        Boolean flag indicating if input is a string
    """
    if is_num(s) is True:
        out = False
    else:
        try:
            str(s)
            out = True
        except Exception as _:
            out = False

    return out


def is_array(a):
    """Check if input is an array (returns True/False)

    Parameters
    ----------
    n : obj
        Input object

    Returns
    -------
    out : bool
        Boolean flag indicating if input is an array
    """
    if (isinstance(a, (list, np.ndarray)) and ~isinstance(a, str)):
        out = True
    else:
        out = False

    return out


def is_2D_list(a):
    """Check if input is a nested (2D) list (returns True/False)

    Parameters
    ----------
    n : obj
        Input object

    Returns
    -------
    out : bool
        Boolean flag indicating if input is a 2D list
    """
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
        self._parameters = self._check_parameters(parameters)

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

    def _check_parameters(self, p):
        """Check parameters

        Parameters
        ----------
        p : dict
            Dictionary of Parameters

        Returns
        -------
        p : dict
            Dictionary of Parameters
        """
        if not isinstance(p, dict):
            warn('Input parameters for {} SAM module need to '
                 'be input as a dictionary but were input as '
                 'a {}'.format(self.module, type(p)), SAMInputWarning)
            warn('No input parameters specified for '
                 '{} SAM module, defaults will be set'
                 .format(self.module), SAMInputWarning)
            p = {}

        return p

    @property
    def parameters(self):
        """Get the parameters property

        Returns
        -------
        _parameters : dict
            Dictionary of Parameters
        """
        return self._parameters

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
        more_parameters : dict | None
            New key-value pairs to add to this instance of SAM Parameters.
        """
        if more_parameters is not None:
            if isinstance(more_parameters, dict):
                self._parameters.update(more_parameters)
            else:
                warn('Attempting to update SAM input parameters with non-dict '
                     'input. Cannot perform update operation. Proceeding '
                     'without additional inputs: {}'.format(more_parameters),
                     SAMInputWarning)


class SiteOutput(SlottedDict):
    """Slotted memory dictionary emulator for SAM single-site outputs."""

    # make attribute slots for all SAM output variable names
    __slots__ = ['cf_mean', 'cf_profile', 'annual_energy', 'energy_yield',
                 'gen_profile', 'poa', 'ppa_price', 'lcoe_fcr', 'npv',
                 'lcoe_nom', 'lcoe_real']


class SAM:
    """Base class for SAM simulations (generation and econ)."""

    DIR = os.path.dirname(os.path.realpath(__file__))
    MODULE = None

    # Mapping for reV technology and SAM module to h5 resource handler type
    # SolarResource is swapped for NSRDB if the res_file contains "nsrdb"
    RESOURCE_TYPES = {'pv': SolarResource, 'pvwattsv5': SolarResource,
                      'csp': SolarResource, 'tcsmolten_salt': SolarResource,
                      'wind': WindResource, 'landbasedwind': WindResource,
                      'offshorewind': WindResource, 'windpower': WindResource,
                      }

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

        self._site = None
        self.outputs = None

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

    @staticmethod
    def get_sam_res(res_file, project_points, module, downscale=None):
        """Get the SAM resource iterator object (single year, single file).

        Parameters
        ----------
        res_file : str
            Single resource file (with full path) to retrieve.
        project_points : reV.config.ProjectPoints
            reV 2.0 Project Points instance used to retrieve resource data at a
            specific set of sites.
        module : str
            SAM module name or reV technology to force interpretation
            of the resource file type.
            Example: module set to 'pvwatts' or 'tcsmolten' means that this
            expects a SolarResource file. If 'nsrdb' is in the res_file name,
            the NSRDB handler will be used.
        downscale : NoneType | str
            Option for NSRDB resource downscaling to higher temporal
            resolution. Expects a string in the Pandas frequency format,
            e.g. '5min'.

        Returns
        -------
        res : reV.resource.SAMResource
            Resource iterator object to pass to SAM.
        """

        try:
            res_handler = SAM.RESOURCE_TYPES[module.lower()]
        except KeyError:
            msg = ('Cannot interpret what kind of resource handler the SAM '
                   'module or reV technology "{}" requires. Expecting one of '
                   'the following SAM modules or reV technologies: {}'
                   .format(module, list(SAM.RESOURCE_TYPES.keys())))
            raise SAMExecutionError(msg)

        if res_handler == SolarResource and 'nsrdb' in res_file.lower():
            # Use NSRDB handler if definitely an NSRDB file
            res_handler = NSRDB

        # additional arguments for special cases.
        kwargs = {}
        if res_handler == SolarResource or res_handler == NSRDB:
            # check for clearsky irradiation analysis for NSRDB
            kwargs = {'clearsky': project_points.sam_config_obj.clearsky}

        elif res_handler == WindResource:
            if project_points.curtailment is not None:
                if project_points.curtailment.precipitation:
                    # make precip rate available for curtailment analysis
                    kwargs = {'precip_rate': True}

        else:
            raise TypeError('Did not recongize resource type "{}", '
                            'must be Wind, Solar, or NSRDB resource class.'
                            .format(res_handler))

        # check for downscaling request
        if downscale is not None:
            # make sure that downscaling is only requested for NSRDB resource
            if res_handler != NSRDB:
                msg = ('Downscaling was requested for a non-NSRDB '
                       'resource file. reV does not have this capability at '
                       'the current time. Please contact a developer for '
                       'more information on this feature.')
                raise SAMInputWarning(msg)
            else:
                # pass through the downscaling request
                kwargs['downscale'] = downscale

        res = res_handler.preload_SAM(res_file, project_points, **kwargs)

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
        """Drop Feb 29th from resource df with time index.

        Parameters
        ----------
        resource : pd.DataFrame
            Resource dataframe with an index containing a pandas
            time index object with month and day attributes.

        Returns
        -------
        resource : pd.DataFrame
            Resource dataframe with all February 29th timesteps removed.
        """

        if hasattr(resource, 'index'):
            if (hasattr(resource.index, 'month') and
                    hasattr(resource.index, 'day')):
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

        if len(res_arr) < 8760:
            raise ResourceError('Resource timeseries must be hourly. '
                                'Received timeseries of length {}.'
                                .format(len(res_arr)))

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

    def cf_mean(self):
        """Get mean capacity factor (fractional) from SAM.

        Returns
        -------
        output : float
            Mean capacity factor (fractional).
        """
        return self.ssc.data_get_number(self.data, 'capacity_factor') / 100

    def cf_profile(self):
        """Get hourly capacity factor (frac) profile in orig timezone.

        Returns
        -------
        cf_profile : np.ndarray
            1D numpy array of capacity factor profile.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return self.gen_profile() / self.parameters['system_capacity']

    def annual_energy(self):
        """Get annual energy generation value in kWh from SAM.

        Returns
        -------
        output : float
            Annual energy generation (kWh).
        """
        return self.ssc.data_get_number(self.data, 'annual_energy')

    def energy_yield(self):
        """Get annual energy yield value in kwh/kw from SAM.

        Returns
        -------
        output : float
            Annual energy yield (kwh/kw).
        """
        return self.ssc.data_get_number(self.data, 'kwh_per_kw')

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
        if self._meta is not None:
            if 'timezone' in self.meta:
                gen = np.roll(gen, -1 * int(self.meta['timezone'] *
                                            self.time_interval))
        return gen

    def poa(self):
        """Get plane-of-array irradiance profile (orig timezone) in W/m2.

        Returns
        -------
        output : np.ndarray
            1D array of plane-of-array irradiance in W/m2.
            Datatype is float32 and array length is 8760*time_interval.
        """
        poa = np.array(self.ssc.data_get_array(self.data, 'poa'),
                       dtype=np.float32)
        # Roll back to native timezone if resource meta has a timezone
        if self._meta is not None:
            if 'timezone' in self.meta:
                poa = np.roll(poa, -1 * int(self.meta['timezone'] *
                                            self.time_interval))
        return poa

    def ppa_price(self):
        """Get PPA price ($/MWh).

        Native units are cents/kWh, mult by 10 for $/MWh.
        """
        return self.ssc.data_get_number(self.data, 'ppa') * 10

    def npv(self):
        """Get net present value (NPV) ($).

        Native units are dollars.
        """
        return self.ssc.data_get_number(self.data, 'npv')

    def lcoe_fcr(self):
        """Get LCOE ($/MWh).

        Native units are $/kWh, mult by 1000 for $/MWh.
        """
        return self.ssc.data_get_number(self.data, 'lcoe_fcr') * 1000

    def lcoe_nom(self):
        """Get nominal LCOE ($/MWh) (from PPA/SingleOwner model).

        Native units are cents/kWh, mult by 10 for $/MWh.
        """
        return self.ssc.data_get_number(self.data, 'lcoe_nom') * 10

    def lcoe_real(self):
        """Get real LCOE ($/MWh) (from PPA/SingleOwner model).

        Native units are cents/kWh, mult by 10 for $/MWh.
        """
        return self.ssc.data_get_number(self.data, 'lcoe_real') * 10

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
                logger.exception(msg)
                idx = idx + 1
                raise SAMExecutionError('SAM error message: "{}"'
                                        .format(msg.decode('utf-8')))

            raise Exception(msg)
        self.ssc.module_free(module)

        self.outputs = self.collect_outputs()

        if close is True:
            self.ssc.data_free(self.data)

    def collect_outputs(self):
        """Collect SAM output_request.

        Returns
        -------
        output : SAM.SiteOutput
            Slotted dictionary emulator keyed by SAM variable names with SAM
            numerical results.
        """

        OUTPUTS = {'cf_mean': self.cf_mean,
                   'cf_profile': self.cf_profile,
                   'annual_energy': self.annual_energy,
                   'energy_yield': self.energy_yield,
                   'gen_profile': self.gen_profile,
                   'poa': self.poa,
                   'ppa_price': self.ppa_price,
                   'npv': self.npv,
                   'lcoe_fcr': self.lcoe_fcr,
                   'lcoe_nom': self.lcoe_nom,
                   'lcoe_real': self.lcoe_real,
                   }

        results = SiteOutput()
        for request in self.output_request:
            if request in OUTPUTS:
                results[request] = OUTPUTS[request]()
            else:
                msg = ('Requested SAM variable "{}" is not available. The '
                       'following output variables are available: "{}".'
                       .format(request, OUTPUTS.keys()))
                raise SAMExecutionError(msg)

        return results
