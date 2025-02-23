# -*- coding: utf-8 -*-
"""reV-to-SAM interface module.

Wraps the NREL-PySAM library with additional reV features.
"""

import copy
import json
import logging
import os
from warnings import warn

import numpy as np
import pandas as pd
import PySAM.CustomGeneration as generic
from rex.multi_file_resource import (
    MultiFileNSRDB,
    MultiFileResource,
    MultiFileWTK,
)
from rex.multi_res_resource import MultiResolutionResource
from rex.renewable_resource import (
    NSRDB,
    GeothermalResource,
    SolarResource,
    WaveResource,
    WindResource,
)
from rex.utilities.utilities import check_res_file

from reV.utilities import ResourceMetaField
from reV.utilities.exceptions import (
    ResourceError,
    SAMExecutionError,
    SAMInputError,
    SAMInputWarning,
)

logger = logging.getLogger(__name__)


class SamResourceRetriever:
    """Factory utility to get the SAM resource handler."""

    # Mapping for reV technology and SAM module to h5 resource handler type
    # SolarResource is swapped for NSRDB if the res_file contains "nsrdb"
    RESOURCE_TYPES = {
        "geothermal": GeothermalResource,
        "pvwattsv5": SolarResource,
        "pvwattsv7": SolarResource,
        "pvwattsv8": SolarResource,
        "pvsamv1": SolarResource,
        "tcsmoltensalt": SolarResource,
        "solarwaterheat": SolarResource,
        "lineardirectsteam": SolarResource,
        "windpower": WindResource,
        "mhkwave": WaveResource,
    }

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
            res_handler = SamResourceRetriever.RESOURCE_TYPES[module.lower()]

        except KeyError as e:
            msg = (
                "Cannot interpret what kind of resource handler the SAM "
                'module or reV technology "{}" requires. Expecting one of '
                "the following SAM modules or reV technologies: {}".format(
                    module, list(SamResourceRetriever.RESOURCE_TYPES.keys())
                )
            )
            logger.exception(msg)
            raise SAMExecutionError(msg) from e

        if res_handler == SolarResource and "nsrdb" in res_file.lower():
            # Use NSRDB handler if definitely an NSRDB file
            res_handler = NSRDB

        return res_handler

    @staticmethod
    def _parse_gid_map_sites(gen_gids, gid_map=None):
        """Parse resource gids based on the generation gids used by
        project_points and a gid_map. If gid_map is None, the input gen_gids
        are just passed through as the res_gids.

        Parameters
        ----------
        gen_gids : list
            List of project_points "sites" that are the generation gids.
        gid_map : None | dict
            Mapping of unique integer generation gids (keys) to single integer
            resource gids (values). This enables the user to input unique
            generation gids in the project points that map to non-unique
            resource gids. This can be None or a pre-extracted dict.

        Returns
        -------
        res_gids : list
            List of resource gids corresponding to the generation gids used by
            project points. If gid_map is None, then this is the same as the
            input gen_gids.
        """
        if gid_map is None:
            res_gids = gen_gids
        else:
            res_gids = [gid_map[i] for i in gen_gids]
        return res_gids

    @classmethod
    def _make_res_kwargs(
        cls, res_handler, project_points, output_request, gid_map
    ):
        """
        Make Resource.preloadSam args and kwargs

        Parameters
        ----------
        res_handler : Resource handler
            Wind resource handler.
        project_points : reV.config.ProjectPoints
            reV Project Points instance used to retrieve resource data at a
            specific set of sites.
        output_request : list
            Outputs to retrieve from SAM.
        gid_map : None | dict
            Mapping of unique integer generation gids (keys) to single integer
            resource gids (values). This enables the user to input unique
            generation gids in the project points that map to non-unique
            resource gids. This can be None or a pre-extracted dict.

        Returns
        -------
        kwargs : dict
            Extra input args to preload sam resource.
        args : tuple
            Args for res_handler.preload_SAM class method
        """
        sites = cls._parse_gid_map_sites(project_points.sites, gid_map=gid_map)
        args = (sites,)

        kwargs = {}
        if res_handler in (SolarResource, NSRDB):
            # check for clearsky irradiation analysis for NSRDB
            kwargs["clearsky"] = project_points.sam_config_obj.clearsky
            kwargs["bifacial"] = project_points.sam_config_obj.bifacial
            kwargs["tech"] = project_points.tech

            downscale = project_points.sam_config_obj.downscale
            # check for downscaling request
            if downscale is not None:
                # make sure that downscaling is only requested for NSRDB
                # resource
                if res_handler != NSRDB:
                    msg = (
                        "Downscaling was requested for a non-NSRDB "
                        "resource file. reV does not have this capability "
                        "at the current time. Please contact a developer "
                        "for more information on this feature."
                    )
                    logger.warning(msg)
                    warn(msg, SAMInputWarning)
                else:
                    # pass through the downscaling request
                    kwargs["downscale"] = downscale

        elif res_handler == WindResource:
            args += (project_points.h,)
            kwargs["icing"] = project_points.sam_config_obj.icing
            if (
                project_points.curtailment is not None
                and any(
                    config.precipitation
                    for config in project_points.curtailment.values()
                )
            ):
                # make precip rate available for curtailment analysis
                kwargs["precip_rate"] = True

            sam_configs = project_points.sam_inputs.values()
            needs_wd = any(_sam_config_contains_turbine_layout(sam_config)
                           for sam_config in sam_configs)
            kwargs["require_wind_dir"] = needs_wd

        elif res_handler == GeothermalResource:
            args += (project_points.d,)

        # Check for resource means
        if any(req.endswith("_mean") for req in output_request):
            kwargs["means"] = True

        return kwargs, args

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
        elif res_handler in (NSRDB, SolarResource):
            res_handler = MultiFileNSRDB
        else:
            res_handler = MultiFileResource

        return res_handler, kwargs, res_file

    @classmethod
    def get(
        cls,
        res_file,
        project_points,
        module,
        output_request=("cf_mean",),
        gid_map=None,
        lr_res_file=None,
        nn_map=None,
        bias_correct=None,
    ):
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
        gid_map : None | dict
            Mapping of unique integer generation gids (keys) to single integer
            resource gids (values). This enables the user to input unique
            generation gids in the project points that map to non-unique
            resource gids. This can be None or a pre-extracted dict.
        lr_res_file : str | None
            Optional low resolution resource file that will be dynamically
            mapped+interpolated to the nominal-resolution res_file. This
            needs to be of the same format as resource_file, e.g. they both
            need to be handled by the same rex Resource handler such as
            WindResource
        nn_map : np.ndarray
            Optional 1D array of nearest neighbor mappings associated with the
            res_file to lr_res_file spatial mapping. For details on this
            argument, see the rex.MultiResolutionResource docstring.
        bias_correct : None | pd.DataFrame
            Optional DataFrame or CSV filepath to a wind or solar
            resource bias correction table. This has columns:

                - ``gid``: GID of site (can be index name of dataframe)
                - ``method``: function name from ``rex.bias_correction`` module

            The ``gid`` field should match the true resource ``gid`` regardless
            of the optional ``gid_map`` input. Only ``windspeed`` **or**
            ``GHI`` + ``DNI`` + ``DHI`` are corrected, depending on the
            technology (wind for the former, PV or CSP for the latter). See the
            functions in the ``rex.bias_correction`` module for available
            inputs for ``method``. Any additional kwargs required for the
            requested ``method`` can be input as additional columns in the
            ``bias_correct`` table e.g., for linear bias correction functions
            you can include ``scalar`` and ``adder`` inputs as columns in the
            ``bias_correct`` table on a site-by-site basis. If ``None``, no
            corrections are applied. By default, ``None``.


        Returns
        -------
        res : reV.resource.SAMResource
            Resource iterator object to pass to SAM.
        """

        res_handler = cls._get_base_handler(res_file, module)
        kwargs, args = cls._make_res_kwargs(
            res_handler, project_points, output_request, gid_map
        )

        multi_h5_res, hsds = check_res_file(res_file)
        if multi_h5_res:
            res_handler, kwargs, res_file = cls._multi_file_mods(
                res_handler, kwargs, res_file
            )
        else:
            kwargs["hsds"] = hsds

        kwargs["time_index_step"] = (
            project_points.sam_config_obj.time_index_step
        )

        if lr_res_file is None:
            res = res_handler.preload_SAM(res_file, *args, **kwargs)
        else:
            kwargs["handler_class"] = res_handler
            kwargs["nn_map"] = nn_map
            res = MultiResolutionResource.preload_SAM(
                res_file, lr_res_file, *args, **kwargs
            )

        if bias_correct is not None:
            res.bias_correct(bias_correct)

        return res


class Sam:
    """reV wrapper on the PySAM framework."""

    # PySAM object wrapped by this class
    PYSAM = generic

    # callable attributes to be ignored in the get/set logic
    IGNORE_ATTRS = ["assign", "execute", "export"]

    def __init__(self):
        self._pysam = self.PYSAM.new()
        self._attr_dict = None
        self._inputs = []
        self.sam_sys_inputs = {}
        if "constant" in self.input_list:
            self["constant"] = 0.0

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
            msg = (
                'Could not set input key "{}". Attribute not '
                'found in PySAM object: "{}"'.format(key, self.pysam)
            )
            logger.exception(msg)
            raise SAMInputError(msg)

        if (key == "total_installed_cost" and isinstance(value, str)
            and value.casefold() == "windbos"):
            # "windbos" is a special reV key to tell reV to compute
            # total installed costs using WindBOS module. If detected,
            # don't try to set it as a PySAM attribute
            return

        self.sam_sys_inputs[key] = value
        group = self._get_group(key, outputs=False)
        try:
            setattr(getattr(self.pysam, group), key, value)
        except Exception as e:
            msg = (
                'Could not set input key "{}" to '
                'group "{}" in "{}".\n'
                "Data is: {} ({})\n"
                'Received the following error: "{}"'.format(
                    key, group, self.pysam, value, type(value), e
                )
            )
            logger.exception(msg)
            raise SAMInputError(msg) from e

    @property
    def pysam(self):
        """Get the pysam object."""
        return self._pysam

    @classmethod
    def default(cls):
        """Get the executed default pysam object.

        Returns
        -------
        PySAM.CustomGeneration
        """
        obj = cls.PYSAM.default("CustomGenerationProfileNone")
        obj.execute()

        return obj

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
            self._attr_dict = {
                k: self._get_pysam_attrs(getattr(self.pysam, k)) for k in keys
            }

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
                if k.lower() != "outputs":
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
            temp = {k: v for (k, v) in temp.items() if k.lower() != "outputs"}

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
        attrs = [
            a
            for a in dir(obj)
            if not a.startswith("__") and a not in self.IGNORE_ATTRS
        ]
        try:
            # adjustment factors are "dynamic" as of PySAM 5+
            # Not found by dir() function, so must check for them
            # explicitly
            __ = obj.AdjustmentFactors
            attrs.append("AdjustmentFactors")
        except AttributeError:
            pass
        return attrs

    def execute(self):
        """Call the PySAM execute method. Raise SAMExecutionError if error."""
        try:
            self.pysam.execute()
        except Exception as e:
            msg = 'PySAM raised an error while executing: "{}"'.format(e)
            logger.exception(msg)
            raise SAMExecutionError(msg) from e

    @staticmethod
    def _filter_inputs(key, value):
        """Perform any necessary filtering of input keys and values for PySAM.

        Parameters
        ----------
        key : str
            SAM input key.
        value : str | int | float | list | np.ndarray
            Input value associated with key.

        Returns
        -------
        key : str
            Filtered SAM input key.
        value : str | int | float | list | np.ndarray
            Filtered Input value associated with key.
        """

        if "." in key:
            key = key.replace(".", "_")

        if "adjust:" in key:
            msg = ("The 'adjust:' syntax is deprecated in PySAm 6+. Please"
                   "use 'adjust_' instead (e.g. 'adjust:hourly' -> "
                   "'adjust_hourly')")
            logger.warning(msg)
            warn(msg)
            key = key.replace(":", "_")

        if isinstance(value, str) and "[" in value and "]" in value:
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                msg = (
                    'Found a weird SAM config input for "{}" that looks '
                    "like a stringified-list but could not run through "
                    "json.loads() so skipping: {}".format(key, value)
                )
                logger.warning(msg)
                warn(msg)

        return key, value

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
            k, v = self._filter_inputs(k, v)
            if k in self.input_list and v is not None:
                self[k] = v
            elif raise_warning:
                wmsg = 'Not setting input "{}" to: {}.'.format(k, v)
                warn(wmsg, SAMInputWarning)
                logger.warning(wmsg)


class RevPySam(Sam):
    """Base class for reV-SAM simulations (generation and econ)."""

    DIR = os.path.dirname(os.path.realpath(__file__))
    MODULE = None

    def __init__(
        self, meta, sam_sys_inputs, output_request, site_sys_inputs=None
    ):
        """Initialize a SAM object.

        Parameters
        ----------
        meta : pd.DataFrame | pd.Series | None
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude, elevation,
            and timezone. Can be None for econ runs.
        sam_sys_inputs : dict
            Site-agnostic SAM system model inputs arguments.
        output_request : list
            Requested SAM outputs (e.g., 'cf_mean', 'annual_energy',
            , 'gen_profile', 'energy_yield', 'ppa_price',
            'lcoe_fcr').
        site_sys_inputs : dict
            Optional set of site-specific SAM system inputs to complement the
            site-agnostic inputs.
        """

        super().__init__()
        self._site = None
        self.time_interval = 1
        self.outputs = {}
        self.sam_sys_inputs = sam_sys_inputs
        self.site_sys_inputs = site_sys_inputs
        self.output_request = output_request
        if self.output_request is None:
            self.output_request = []

        self._meta = self._parse_meta(meta)
        self._parse_site_sys_inputs(site_sys_inputs)
        _add_cost_defaults(self.sam_sys_inputs)
        _add_sys_capacity(self.sam_sys_inputs)

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

        if hasattr(resource, "index"):
            if hasattr(resource.index, "month") and hasattr(
                resource.index, "day"
            ):
                leap_day = (resource.index.month == 2) & (
                    resource.index.day == 29
                )
                resource = resource.drop(resource.index[leap_day])

        return resource

    @staticmethod
    def ensure_res_len(arr, time_index):
        """
        Ensure time_index has a constant time-step and only covers 365 days
        (no leap days). If not remove last day

        Parameters
        ----------
        arr : ndarray
            Array to truncate if time_index has a leap day
        time_index : pandas.DatatimeIndex
            Time index associated with arr, used to check time-series
            frequency and number of days

        Returns
        -------
        arr : ndarray
            Truncated array of data such that there are 365 days
        """
        msg = (
            "A valid time_index must be supplied to ensure the proper "
            "resource length! Instead {} was supplied".format(type(time_index))
        )
        assert isinstance(time_index, pd.DatetimeIndex)

        msg = "arr length {} does not match time_index length {}!".format(
            len(arr), len(time_index)
        )
        assert len(arr) == len(time_index)

        if time_index.is_leap_year.all():
            mask = time_index.month == 2
            mask &= time_index.day == 29
            if not mask.any():
                mask = time_index.month == 2
                mask &= time_index.day == 28
                s = np.where(mask)[0][-1]

                freq = pd.infer_freq(time_index[:s])
                msg = "frequencies do not match before and after 2/29"
                assert freq == pd.infer_freq(time_index[s + 1:]), msg
            else:
                freq = pd.infer_freq(time_index)
        else:
            freq = pd.infer_freq(time_index)

        if freq is None:
            msg = (
                "Resource time_index does not have a consistent time-step "
                "(frequency)!"
            )
            logger.error(msg)
            raise ResourceError(msg)

        doy = time_index.dayofyear
        n_doy = len(doy.unique())

        if n_doy > 365:
            # Drop last day of year
            doy_max = doy.max()
            mask = doy != doy_max
            arr = arr[mask]

        return arr

    @staticmethod
    def make_datetime(series):
        """Ensure that pd series is a datetime series with dt accessor"""
        if not hasattr(series, "dt"):
            series = pd.to_datetime(pd.Series(series))

        return series

    @classmethod
    def get_time_interval(cls, time_index):
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

        time_index = cls.make_datetime(time_index)
        x = time_index.dt.hour.diff()
        time_interval = 0

        # iterate through the hourly time diffs and count indices between flips
        for t in x[1:]:
            if t == 1.0:
                time_interval += 1
                break
            if t == 0.0:
                time_interval += 1

        return int(time_interval)

    @staticmethod
    def _parse_meta(meta):
        """Make sure the meta data corresponds to a single location and convert
        to pd.Series.

        Parameters
        ----------
        meta : pd.DataFrame | pd.Series | None
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude, elevation,
            and timezone. Can be None for econ runs.

        Parameters
        ----------
        meta : pd.Series | None
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude, elevation,
            and timezone. Can be None for econ runs.
        """
        if isinstance(meta, pd.DataFrame):
            msg = (
                "Meta data must only be for a single site but received: "
                f"{meta}"
            )
            assert len(meta) == 1, msg
            meta = meta.iloc[0]

        if meta is not None:
            assert isinstance(meta, pd.Series)

        return meta

    def _parse_site_sys_inputs(self, site_sys_inputs):
        """Parse site-specific parameters and add to parameter dict.

        Parameters
        ----------
        site_sys_inputs : dict
            Optional set of site-specific SAM system inputs to complement the
            site-agnostic inputs.
        """

        if site_sys_inputs is not None:
            for k, v in site_sys_inputs.items():
                if isinstance(v, float) and np.isnan(v):
                    pass
                else:
                    self.sam_sys_inputs[k] = v

    @staticmethod
    def _is_arr_like(val):
        """Returns true if SAM data is array-like. False if scalar."""
        if isinstance(val, (int, float, str)):
            return False
        try:
            len(val)
        except TypeError:
            return False
        else:
            return True

    @classmethod
    def _is_hourly(cls, val):
        """Returns true if SAM data is hourly or sub-hourly. False otherise."""
        if not cls._is_arr_like(val):
            return False
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
                        n_roll = int(
                            -1
                            * self.meta[ResourceMetaField.TIMEZONE]
                            * self.time_interval
                        )
                        output = np.roll(output, n_roll)

                    self.outputs[key] = output

    def collect_outputs(self, output_lookup):
        """Collect SAM output_request, convert timeseries outputs to UTC, and
        save outputs to self.outputs property.

        Parameters
        ----------
        output_lookup : dict
            Lookup dictionary mapping output keys to special output methods.
        """
        bad_requests = []
        for req in self.output_request:
            if req in output_lookup:
                self.outputs[req] = output_lookup[req]()
            elif req in self.sam_sys_inputs:
                self.outputs[req] = self.sam_sys_inputs[req]
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

        self.outputs_to_utc_arr()

    def assign_inputs(self):
        """Assign the self.sam_sys_inputs attribute to the PySAM object."""
        super().assign_inputs(copy.deepcopy(self.sam_sys_inputs))

    def execute(self):
        """Call the PySAM execute method. Raise SAMExecutionError if error.
        Include the site index if available.
        """
        try:
            self.pysam.execute()
        except Exception as e:
            msg = 'PySAM raised an error while executing: "{}"'.format(
                self.module
            )
            if self.site is not None:
                msg += " for site {}".format(self.site)
            logger.exception(msg)
            raise SAMExecutionError(msg) from e


def _add_cost_defaults(sam_inputs):
    """Add default values for required cost outputs if they are missing. """
    if sam_inputs.get("__already_added_cost_defaults"):
        return

    sam_inputs.setdefault("fixed_charge_rate", None)

    reg_mult = sam_inputs.setdefault("capital_cost_multiplier", 1)
    capital_cost = sam_inputs.setdefault("capital_cost", None)
    fixed_operating_cost = sam_inputs.setdefault("fixed_operating_cost", None)
    variable_operating_cost = sam_inputs.setdefault(
        "variable_operating_cost", None)

    sam_inputs["base_capital_cost"] = capital_cost
    sam_inputs["base_fixed_operating_cost"] = fixed_operating_cost
    sam_inputs["base_variable_operating_cost"] = variable_operating_cost
    if capital_cost is not None:
        sam_inputs["capital_cost"] = capital_cost * reg_mult
    else:
        sam_inputs["capital_cost"] = None

    sam_inputs["__already_added_cost_defaults"] = True


def _add_sys_capacity(sam_inputs):
    """Add system capacity SAM input if it is missing. """
    cap = sam_inputs.get("system_capacity")
    if cap is None:
        cap = sam_inputs.get("turbine_capacity")

    if cap is None:
        cap = sam_inputs.get("wind_turbine_powercurve_powerout")
        if cap is not None:
            cap = max(cap)

    if cap is None:
        cap = sam_inputs.get("nameplate")

    sam_inputs["system_capacity"] = cap


def _sam_config_contains_turbine_layout(sam_config):
    """Detect wether SAM config contains multiple turbines in layout. """
    return len(sam_config.get("wind_farm_xCoordinates", ())) > 1
