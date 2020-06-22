# -*- coding: utf-8 -*-
"""reV-to-SAM interface module.

Wraps the NREL-PySAM library with additional reV features.
"""
import logging
import numpy as np
import os
import pandas as pd
from warnings import warn
import PySAM.GenericSystem as generic

from reV.utilities.exceptions import (SAMInputWarning, SAMInputError,
                                      SAMExecutionError, ResourceError)

from rex.resource import MultiFileResource
from rex.renewable_resource import (WindResource, SolarResource, NSRDB,
                                    MultiFileWTK, MultiFileNSRDB)
from rex.utilities.utilities import check_res_file


logger = logging.getLogger(__name__)


class SamResourceRetriever:
    """Factory utility to get the SAM resource handler."""

    @staticmethod
    def _get_base_handler(res_file, module):
        """Get the base SAM resource handler, raise error if module not found.

        Parameters
        ----------
        res_file : str
            Single resource file (with full path) to retrieve.
        module : str
            SAM module name or reV technology to force interpretation
            of the resource file type.
            Example: module set to 'pvwatts' or 'tcsmolten' means that this
            expects a SolarResource file. If 'nsrdb' is in the res_file name,
            the NSRDB handler will be used.

        Returns
        -------
        res_handler : SolarResource | WindResource | NSRDB
            Solar or Wind resource handler based on input.
        """

        try:
            res_handler = RevPySam.RESOURCE_TYPES[module.lower()]

        except KeyError:
            msg = ('Cannot interpret what kind of resource handler the SAM '
                   'module or reV technology "{}" requires. Expecting one of '
                   'the following SAM modules or reV technologies: {}'
                   .format(module, list(RevPySam.RESOURCE_TYPES.keys())))
            logger.exception(msg)
            raise SAMExecutionError(msg)

        if res_handler == SolarResource and 'nsrdb' in res_file.lower():
            # Use NSRDB handler if definitely an NSRDB file
            res_handler = NSRDB

        return res_handler

    @staticmethod
    def _make_solar_kwargs(res_handler, project_points, output_request,
                           downscale=None):
        """Make kwargs dict for Solar | NSRDB resource handler initialization.

        Parameters
        ----------
        res_handler : SolarResource | NSRDB
            Solar resource handler.
        project_points : reV.config.ProjectPoints
            reV Project Points instance used to retrieve resource data at a
            specific set of sites.
        output_request : list
            Outputs to retrieve from SAM.
        downscale : NoneType | str
            Option for NSRDB resource downscaling to higher temporal
            resolution. Expects a string in the Pandas frequency format,
            e.g. '5min'.

        Returns
        -------
        kwargs : dict
            Extra input args to preload sam resource.
        args : tuple
            Args for res_handler.preload_SAM class method
        res_handler : SolarResource | NSRDB
            Solar resource handler.
        """
        args = (project_points.sites,)

        kwargs = {}
        # check for clearsky irradiation analysis for NSRDB
        kwargs['clearsky'] = project_points.sam_config_obj.clearsky
        kwargs['tech'] = project_points.tech
        # Check for resource means:
        mean_keys = ['dni_mean', 'ghi_mean', 'dhi_mean']
        if any([x in output_request for x in mean_keys]):
            kwargs['means'] = True

        # check for downscaling request
        if downscale is not None:
            # make sure that downscaling is only requested for NSRDB resource
            if res_handler != NSRDB:
                msg = ('Downscaling was requested for a non-NSRDB '
                       'resource file. reV does not have this capability at '
                       'the current time. Please contact a developer for '
                       'more information on this feature.')
                logger.warning(msg)
                warn(msg, SAMInputWarning)
            else:
                # pass through the downscaling request
                kwargs['downscale'] = downscale

        return kwargs, args, res_handler

    @staticmethod
    def _make_wind_kwargs(res_handler, project_points, output_request):
        """Make kwargs dict for Wind resource handler initialization.

        Parameters
        ----------
        res_handler : SolarResource | NSRDB
            Wind resource handler.
        project_points : reV.config.ProjectPoints
            reV Project Points instance used to retrieve resource data at a
            specific set of sites.
        output_request : list
            Outputs to retrieve from SAM.

        Returns
        -------
        kwargs : dict
            Extra input args to preload sam resource.
        args : tuple
            Args for res_handler.preload_SAM class method
        res_handler : WindResource | MultiFileWTK
            Wind resource handler.
        """
        args = (project_points.sites, project_points.h)
        kwargs = {}
        kwargs['icing'] = project_points.sam_config_obj.icing
        if project_points.curtailment is not None:
            if project_points.curtailment.precipitation:
                # make precip rate available for curtailment analysis
                kwargs['precip_rate'] = True

        # Check for resource means:
        if 'ws_mean' in output_request:
            kwargs['means'] = True

        return kwargs, args, res_handler

    @staticmethod
    def _multi_file_mods(res_handler, kwargs, res_file):
        """
        Check if res_file is a multi-file resource dir and update handler

        Parameters
        ----------
        res_handler : Resource
            Resource handler.
        kwargs : dict
            Key word arguments for resource init.
        res_file : str
            Single resource file (with full path) or multi h5 dir.

        Returns
        -------
        res_handler : Resource | MultiFileResource
            Resource handler, replaced by the multi file resource handler if
            necessary.
        kwargs : dict
            Key word arguments for resource init with h5_dir, prefix,
            and suffix.
        res_file : str
            Single resource file (with full path) or multi h5 dir.
        """
        if res_handler == WindResource:
            res_handler = MultiFileWTK
        elif res_handler == NSRDB:
            res_handler = MultiFileNSRDB
        else:
            res_handler = MultiFileResource

        return res_handler, kwargs, res_file

    @classmethod
    def get(cls, res_file, project_points, module,
            output_request=('cf_mean', ), downscale=None):
        """Get the SAM resource iterator object (single year, single file).

        Parameters
        ----------
        res_file : str
            Single resource file (with full path) to retrieve.
        project_points : reV.config.ProjectPoints
            reV Project Points instance used to retrieve resource data at a
            specific set of sites.
        module : str
            SAM module name or reV technology to force interpretation
            of the resource file type.
            Example: module set to 'pvwatts' or 'tcsmolten' means that this
            expects a SolarResource file. If 'nsrdb' is in the res_file name,
            the NSRDB handler will be used.
        output_request : list | tuple, optional
            Outputs to retrieve from SAM, by default ('cf_mean', )
        downscale : NoneType | str, optional
            Option for NSRDB resource downscaling to higher temporal
            resolution. Expects a string in the Pandas frequency format,
            e.g. '5min', by default None

        Returns
        -------
        res : reV.resource.SAMResource
            Resource iterator object to pass to SAM.
        """

        res_handler = cls._get_base_handler(res_file, module)

        if res_handler in (SolarResource, NSRDB):
            kwargs, args, res_handler = cls._make_solar_kwargs(
                res_handler, project_points, output_request,
                downscale=downscale)

        elif res_handler == WindResource:
            kwargs, args, res_handler = cls._make_wind_kwargs(
                res_handler, project_points, output_request)

        multi_h5_res, hsds = check_res_file(res_file)
        if multi_h5_res:
            res_handler, kwargs, res_file = cls._multi_file_mods(res_handler,
                                                                 kwargs,
                                                                 res_file)
        else:
            kwargs['hsds'] = hsds

        res = res_handler.preload_SAM(res_file, *args, **kwargs)

        return res


class Sam:
    """reV wrapper on the PySAM framework."""

    # PySAM object wrapped by this class
    PYSAM = generic

    # callable attributes to be ignored in the get/set logic
    IGNORE_ATTRS = ['assign', 'execute', 'export']

    def __init__(self):
        self._pysam = self.PYSAM.new()
        self._attr_dict = None
        self._default = None
        self._inputs = []
        if 'constant' in self.input_list:
            self['constant'] = 0.0

    def __getitem__(self, key):
        """Get the value of a PySAM attribute (either input or output).

        Parameters
        ----------
        key : str
            Lowest level attribute name.

        Returns
        -------
        out : object
            PySAM data.
        """

        group = self._get_group(key)
        try:
            out = getattr(getattr(self.pysam, group), key)
        except Exception:
            out = None
        return out

    def __setitem__(self, key, value):
        """Set a PySAM input data attribute.

        Parameters
        ----------
        key : str
            Lowest level attribute name.
        value : object
            Data to set to the key.
        """

        if key not in self.input_list:
            msg = ('Could not set input key "{}". Attribute not '
                   'found in PySAM object: "{}"'
                   .format(key, self.pysam))
            logger.exception(msg)
            raise SAMInputError(msg)
        else:
            group = self._get_group(key, outputs=False)
            try:
                setattr(getattr(self.pysam, group), key, value)
            except Exception as e:
                msg = ('Could not set input key "{}" to '
                       'group "{}" in "{}".\n'
                       'Data is: {} ({})\n'
                       'Received the following error: "{}"'
                       .format(key, group, self.pysam, value, type(value), e))
                logger.exception(msg)
                raise SAMInputError(msg)

    @property
    def pysam(self):
        """Get the pysam object."""
        return self._pysam

    @property
    def default(self):
        """Get the executed default pysam object.

        Returns
        -------
        _default : PySAM.GenericSystem
            Executed generic system pysam object.
        """
        if self._default is None:
            self._default = self.PYSAM.default('GenericSystemNone')
            self._default.execute()

        return self._default

    @property
    def attr_dict(self):
        """Get the heirarchical PySAM object attribute dictionary.

        Returns
        -------
        _attr_dict : dict
            Dictionary with:
               keys: variable groups
               values: lowest level attribute/variable names
        """
        if self._attr_dict is None:
            keys = self._get_pysam_attrs(self.pysam)
            self._attr_dict = {k: self._get_pysam_attrs(getattr(self.pysam, k))
                               for k in keys}

        return self._attr_dict

    @property
    def input_list(self):
        """Get the list of lowest level input attribute/variable names.

        Returns
        -------
        _inputs : list
            List of lowest level input attributes.
        """
        if not any(self._inputs):
            for k, v in self.attr_dict.items():
                if k.lower() != 'outputs':
                    self._inputs += v

        return self._inputs

    def _get_group(self, key, outputs=True):
        """Get the group that the input key belongs to.

        Parameters
        ----------
        key : str
            Lowest level PySAM attribute/variable name.
        outputs : bool
            Flag if this key might be in outputs group. False ignores the
            outputs group (looks for inputs only).

        Returns
        -------
        group : str | None
            PySAM attribute group that key belongs to. None if not found.
        """
        group = None

        temp = self.attr_dict
        if not outputs:
            temp = {k: v for (k, v) in temp.items()
                    if k.lower() != 'outputs'}

        for k, v in temp.items():
            if key in v:
                group = k
                break

        return group

    def _get_pysam_attrs(self, obj):
        """Get a list of attributes from obj with ignore logic.

        Parameters
        ----------
        obj : PySAM object
            PySAM object to get attribute list from.

        Returns
        -------
        attrs : list
            List of attrs belonging to obj with dunder attrs and IGNORE_ATTRS
            not included.
        """
        attrs = [a for a in dir(obj) if not a.startswith('__')
                 and a not in self.IGNORE_ATTRS]
        return attrs

    def execute(self):
        """Call the PySAM execute method. Raise SAMExecutionError if error."""
        try:
            self.pysam.execute()
        except Exception as e:
            msg = 'PySAM raised an error while executing: "{}"'.format(e)
            logger.exception(msg)
            raise SAMExecutionError(msg)

    @staticmethod
    def _filter_inputs(key):
        """Perform any necessary filtering of input keys for PySAM.

        Parameters
        ----------
        key : str
            SAM input key.

        Returns
        -------
        key : str
            Filtered SAM input key.
        """

        if '.' in key:
            key = key.replace('.', '_')

        if ':constant' in key and 'adjust:' in key:
            key = key.replace('adjust:', '')

        return key

    def assign_inputs(self, inputs, raise_warning=False):
        """Assign a flat dictionary of inputs to the PySAM object.

        Parameters
        ----------
        inputs : dict
            Flat (single-level) dictionary of PySAM inputs.
        raise_warning : bool
            Flag to raise a warning for inputs that are not set because they
            are not found in the PySAM object.
        """
        for k, v in inputs.items():
            k = self._filter_inputs(k)
            if k in self.input_list:
                self[k] = v
            elif raise_warning:
                wmsg = ('Not setting input "{}". Not found in PySAM inputs.'
                        .format(k))
                warn(wmsg, SAMInputWarning)
                logger.warning(wmsg)


class RevPySam(Sam):
    """Base class for reV-SAM simulations (generation and econ)."""

    DIR = os.path.dirname(os.path.realpath(__file__))
    MODULE = None

    # Mapping for reV technology and SAM module to h5 resource handler type
    # SolarResource is swapped for NSRDB if the res_file contains "nsrdb"
    RESOURCE_TYPES = {'pvwattsv5': SolarResource,
                      'pvwattsv7': SolarResource,
                      'tcsmoltensalt': SolarResource,
                      'solarwaterheat': SolarResource,
                      'troughphysicalheat': SolarResource,
                      'lineardirectsteam': SolarResource,
                      'windpower': WindResource,
                      }

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

        super().__init__()
        self._meta = meta
        self._site = None
        self.outputs = None
        self.time_interval = 1
        self.outputs = {}
        self.parameters = parameters
        self.output_request = output_request

    @property
    def meta(self):
        """Get meta data property."""
        return self._meta

    @property
    def module(self):
        """Get module property."""
        return self.MODULE

    @property
    def site(self):
        """Get the site number for this SAM simulation."""
        return self._site

    @staticmethod
    def get_sam_res(*args, **kwargs):
        """Get the SAM resource iterator object (single year, single file)."""
        return SamResourceRetriever.get(*args, **kwargs)

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
            if (hasattr(resource.index, 'month')
                    and hasattr(resource.index, 'day')):
                leap_day = ((resource.index.month == 2)
                            & (resource.index.day == 29))
                resource = resource.drop(resource.index[leap_day])

        return resource

    @staticmethod
    def ensure_res_len(res_arr, base=8760):
        """Ensure that the length of resource array is a multiple of base.

        Parameters
        ----------
        res_arr : np.ndarray
            Array of resource data.
        base : int
            Ensure that length of resource array is a multiple of this value.

        Returns
        -------
        res_arr : array-like
            Truncated array of resource data such that length(res_arr)%base=0.
        """

        if len(res_arr) < base:
            msg = ('Received timeseries of length {}, expected timeseries to'
                   'be at least {}'.format(len(res_arr), base))
            logger.exception(msg)
            raise ResourceError(msg)

        if len(res_arr) % base != 0:
            div = np.floor(len(res_arr) / base)
            target_len = int(div * base)
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

        time_index = RevPySam.make_datetime(time_index)
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

    @staticmethod
    def _is_arr_like(val):
        """Returns true if SAM data is array-like. False if scalar."""
        if isinstance(val, (int, float, str)):
            return False
        else:
            try:
                len(val)
            except TypeError:
                return False
            else:
                return True

    @staticmethod
    def _is_hourly(val):
        """Returns true if SAM data is hourly or sub-hourly. False otherise."""
        if not RevPySam._is_arr_like(val):
            return False
        else:
            L = len(val)
            return L >= 8760

    def outputs_to_utc_arr(self):
        """Convert array-like SAM outputs to UTC np.ndarrays"""
        if self.outputs is not None:
            for key, output in self.outputs.items():
                if self._is_arr_like(output):
                    output = np.asarray(output)

                    if output.dtype == np.float64:
                        output = output.astype(np.float32)
                    elif output.dtype == np.int64:
                        output = output.astype(np.int32)

                    if self._is_hourly(output):
                        output = np.roll(output, int(-1 * self.meta['timezone']
                                                     * self.time_interval))

                    self.outputs[key] = output

    def collect_outputs(self, output_lookup):
        """Collect SAM output_request.

        Parameters
        ----------
        output_lookup : dict
            Lookup dictionary mapping output keys to special output methods.
        """
        bad_requests = []
        for req in self.output_request:
            if req in output_lookup:
                self.outputs[req] = output_lookup[req]()
            else:
                try:
                    self.outputs[req] = getattr(self.pysam.Outputs, req)
                except AttributeError:
                    bad_requests.append(req)

        if any(bad_requests):
            msg = ('Could not retrieve outputs "{}" from PySAM object "{}".'
                   .format(bad_requests, self.pysam))
            logger.error(msg)
            raise SAMExecutionError(msg)

    def assign_inputs(self):
        """Assign the self.parameters attribute to the PySAM object."""
        super().assign_inputs(self.parameters)
