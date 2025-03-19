# -*- coding: utf-8 -*-
"""reV-to-SAM generation interface module.

Wraps the NREL-PySAM pvwattsv5, windpower, and tcsmolensalt modules with
additional reV features.
"""

import copy
import logging
import os
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
from warnings import warn

import numpy as np
import pandas as pd
import PySAM.Geothermal as PySamGeothermal
import PySAM.LinearFresnelDsgIph as PySamLds
import PySAM.MhkWave as PySamMhkWave
import PySAM.Pvsamv1 as PySamDetailedPv
import PySAM.Pvwattsv5 as PySamPv5
import PySAM.Pvwattsv7 as PySamPv7
import PySAM.Pvwattsv8 as PySamPv8
import PySAM.Swh as PySamSwh
import PySAM.TcsmoltenSalt as PySamCSP
import PySAM.Windpower as PySamWindPower

from reV.losses import PowerCurveLossesMixin, ScheduledLossesMixin
from reV.SAM.defaults import (
    DefaultGeothermal,
    DefaultLinearFresnelDsgIph,
    DefaultMhkWave,
    DefaultPvSamv1,
    DefaultPvWattsv5,
    DefaultPvWattsv8,
    DefaultSwh,
    DefaultTcsMoltenSalt,
    DefaultWindPower,
)
from reV.SAM.econ import LCOE, SingleOwner
from reV.SAM.SAM import RevPySam
from reV.utilities import ResourceMetaField, SupplyCurveField
from reV.utilities.curtailment import curtail
from reV.utilities.exceptions import (
    InputError,
    SAMExecutionError,
    SAMInputWarning,
)

logger = logging.getLogger(__name__)


class AbstractSamGeneration(RevPySam, ScheduledLossesMixin, ABC):
    """Base class for SAM generation simulations."""

    _GEN_KEY = "gen"

    def __init__(
        self,
        resource,
        meta,
        sam_sys_inputs,
        site_sys_inputs=None,
        output_request=None,
        drop_leap=False,
    ):
        """Initialize a SAM generation object.

        Parameters
        ----------
        resource : pd.DataFrame
            Timeseries solar or wind resource data for a single location with a
            pandas DatetimeIndex.  There must be columns for all the required
            variables to run the respective SAM simulation. Remapping will be
            done to convert typical NSRDB/WTK names into SAM names (e.g. DNI ->
            dn and wind_speed -> windspeed)
        meta : pd.DataFrame | pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude, elevation,
            and timezone.
        sam_sys_inputs : dict
            Site-agnostic SAM system model inputs arguments.
        site_sys_inputs : dict
            Optional set of site-specific SAM system inputs to complement the
            site-agnostic inputs.
        output_request : list
            Requested SAM outputs (e.g., 'cf_mean', 'annual_energy',
            'cf_profile', 'gen_profile', 'energy_yield', 'ppa_price',
            'lcoe_fcr').
        drop_leap : bool
            Drops February 29th from the resource data. If False, December
            31st is dropped from leap years.
        """

        # drop the leap day
        if drop_leap:
            resource = self.drop_leap(resource)

        # make sure timezone and elevation are in the meta data
        meta = self.tz_elev_check(sam_sys_inputs, site_sys_inputs, meta)

        # don't pass resource to base class,
        # set in concrete generation classes instead
        super().__init__(
            meta,
            sam_sys_inputs,
            output_request,
            site_sys_inputs=site_sys_inputs,
        )

        # Set the site number using resource
        if hasattr(resource, "name"):
            self._site = resource.name
        else:
            self._site = None

        # let children pass in None resource
        if resource is not None:
            self.check_resource_data(resource)
            self.set_resource_data(resource, meta)

        self.add_scheduled_losses(resource)

    @classmethod
    def _get_res(cls, res_df, output_request):
        """Get the resource arrays and pass through for output (single site).

        Parameters
        ----------
        res_df : pd.DataFrame
            2D table with resource data.
        output_request : list
            Outputs to retrieve from SAM.

        Returns
        -------
        res_mean : dict | None
            Dictionary object with variables for resource arrays.
        out_req_cleaned : list
            Output request list with the resource request entries removed.
        """

        out_req_cleaned = copy.deepcopy(output_request)
        res_out = None

        res_reqs = []
        ti = res_df.index
        for req in out_req_cleaned:
            if req in res_df:
                res_reqs.append(req)
                if res_out is None:
                    res_out = {}
                res_out[req] = cls.ensure_res_len(res_df[req].values, ti)

        for req in res_reqs:
            out_req_cleaned.remove(req)

        return res_out, out_req_cleaned

    @staticmethod
    def _get_res_mean(resource, res_gid, output_request):
        """Get the resource annual means (single site).

        Parameters
        ----------
        resource : rex.sam_resource.SAMResource
            SAM resource object for WIND resource
        res_gid : int
            Site to extract means for
        output_request : list
            Outputs to retrieve from SAM.

        Returns
        -------
        res_mean : dict | None
            Dictionary object with variables for resource means.
        out_req_nomeans : list
            Output request list with the resource mean entries removed.
        """

        out_req_nomeans = copy.deepcopy(output_request)
        res_mean = None
        idx = resource.sites.index(res_gid)
        irrad_means = (
            "dni_mean",
            "dhi_mean",
            "ghi_mean",
            "clearsky_dni_mean",
            "clearsky_dhi_mean",
            "clearsky_ghi_mean",
        )

        if "ws_mean" in out_req_nomeans:
            out_req_nomeans.remove("ws_mean")
            res_mean = {}
            res_mean["ws_mean"] = resource["mean_windspeed", idx]

        else:
            for var in resource.var_list:
                label_1 = "{}_mean".format(var)
                label_2 = "mean_{}".format(var)
                if label_1 in out_req_nomeans:
                    out_req_nomeans.remove(label_1)
                    if res_mean is None:
                        res_mean = {}
                    res_mean[label_1] = resource[label_2, idx]

                    if label_1 in irrad_means:
                        # convert to kWh/m2/day
                        res_mean[label_1] /= 1000
                        res_mean[label_1] *= 24

        return res_mean, out_req_nomeans

    def check_resource_data(self, resource):
        """Check resource dataframe for NaN values

        Parameters
        ----------
        resource : pd.DataFrame
            Timeseries solar or wind resource data for a single location with a
            pandas DatetimeIndex.  There must be columns for all the required
            variables to run the respective SAM simulation. Remapping will be
            done to convert typical NSRDB/WTK names into SAM names (e.g. DNI ->
            dn and wind_speed -> windspeed)
        """
        if pd.isna(resource).any().any():
            bad_vars = pd.isna(resource).any(axis=0)
            bad_vars = resource.columns[bad_vars].values.tolist()
            msg = "Found NaN values for site {} in variables {}".format(
                self.site, bad_vars
            )
            logger.error(msg)
            raise InputError(msg)

        if len(resource) < 8760:
            msg = (f"Detected resource time series of length {len(resource)}, "
                   "which is less than 8760. This may yield unexpected "
                   "results or fail altogether. If this is not intentional, "
                   "try setting 'time_index_step: 1' in your SAM config or "
                   "double check the resource input you're using.")
            logger.warning(msg)
            warn(msg)

    @abstractmethod
    def set_resource_data(self, resource, meta):
        """Placeholder for resource data setting (nsrdb or wtk)"""

    @staticmethod
    def tz_elev_check(sam_sys_inputs, site_sys_inputs, meta):
        """Check timezone+elevation input and use json config
        timezone+elevation if not in resource meta.

        Parameters
        ----------
        sam_sys_inputs : dict
            Site-agnostic SAM system model inputs arguments.
        site_sys_inputs : dict
            Optional set of site-specific SAM system inputs to complement the
            site-agnostic inputs.
        meta : pd.DataFrame | pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude, elevation,
            and timezone.

        Returns
        -------
        meta : pd.DataFrame | pd.Series
            Dataframe or series for a single site. Will include "timezone"
            and "elevation" from the sam and site system inputs if found.
        """

        if meta is not None:
            axis = 0 if isinstance(meta, pd.core.series.Series) else 1
            meta = meta.rename(
                SupplyCurveField.map_to(ResourceMetaField), axis=axis
            )
            if sam_sys_inputs is not None:
                if ResourceMetaField.ELEVATION in sam_sys_inputs:
                    meta[ResourceMetaField.ELEVATION] = sam_sys_inputs[
                        ResourceMetaField.ELEVATION
                    ]
                if ResourceMetaField.TIMEZONE in sam_sys_inputs:
                    meta[ResourceMetaField.TIMEZONE] = int(
                        sam_sys_inputs[ResourceMetaField.TIMEZONE]
                    )

            # site-specific inputs take priority over generic system inputs
            if site_sys_inputs is not None:
                if ResourceMetaField.ELEVATION in site_sys_inputs:
                    meta[ResourceMetaField.ELEVATION] = site_sys_inputs[
                        ResourceMetaField.ELEVATION
                    ]
                if ResourceMetaField.TIMEZONE in site_sys_inputs:
                    meta[ResourceMetaField.TIMEZONE] = int(
                        site_sys_inputs[ResourceMetaField.TIMEZONE]
                    )

            if ResourceMetaField.TIMEZONE not in meta:
                msg = (
                    "Need timezone input to run SAM gen. Not found in "
                    "resource meta or technology json input config."
                )
                raise SAMExecutionError(msg)

        return meta

    @property
    def has_timezone(self):
        """Returns true if instance has a timezone set"""
        if self._meta is not None and ResourceMetaField.TIMEZONE in self.meta:
            return True

        return False

    def cf_mean(self):
        """Get mean capacity factor (fractional) from SAM.

        Returns
        -------
        output : float
            Mean capacity factor (fractional).
        """
        return self["capacity_factor"] / 100

    def cf_profile(self):
        """Get hourly capacity factor (frac) profile in local timezone.
        See self.outputs attribute for collected output data in UTC.

        Returns
        -------
        cf_profile : np.ndarray
            1D numpy array of capacity factor profile.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return self.gen_profile() / self.sam_sys_inputs["system_capacity"]

    def annual_energy(self):
        """Get annual energy generation value in kWh from SAM.

        Returns
        -------
        output : float
            Annual energy generation (kWh).
        """
        return self["annual_energy"]

    def energy_yield(self):
        """Get annual energy yield value in kwh/kw from SAM.

        Returns
        -------
        output : float
            Annual energy yield (kwh/kw).
        """
        return self["kwh_per_kw"]

    def gen_profile(self):
        """Get power generation profile (local timezone) in kW.
        See self.outputs attribute for collected output data in UTC.

        Returns
        -------
        output : np.ndarray
            1D array of hourly power generation in kW.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return np.array(self[self._GEN_KEY], dtype=np.float32)

    def collect_outputs(self, output_lookup=None):
        """Collect SAM output_request, convert timeseries outputs to UTC, and
        save outputs to self.outputs property.


        Parameters
        ----------
        output_lookup : dict | None
            Lookup dictionary mapping output keys to special output methods.
            None defaults to generation default outputs.
        """

        if output_lookup is None:
            output_lookup = {
                "cf_mean": self.cf_mean,
                "cf_profile": self.cf_profile,
                "annual_energy": self.annual_energy,
                "energy_yield": self.energy_yield,
                "gen_profile": self.gen_profile,
            }

        super().collect_outputs(output_lookup=output_lookup)

    def run_gen_and_econ(self):
        """Run SAM generation with possibility for follow on econ analysis."""

        lcoe_out_reqs = None
        so_out_reqs = None
        lcoe_vars = (
            "lcoe_fcr",
            "fixed_charge_rate",
            "capital_cost",
            "fixed_operating_cost",
            "variable_operating_cost",
        )
        so_vars = (
            "ppa_price",
            "lcoe_real",
            "lcoe_nom",
            "project_return_aftertax_npv",
            "flip_actual_irr",
            "gross_revenue",
        )
        if "lcoe_fcr" in self.output_request:
            lcoe_out_reqs = [r for r in self.output_request if r in lcoe_vars]
            self.output_request = [
                r for r in self.output_request if r not in lcoe_out_reqs
            ]
        elif any(x in self.output_request for x in so_vars):
            so_out_reqs = [r for r in self.output_request if r in so_vars]
            self.output_request = [
                r for r in self.output_request if r not in so_out_reqs
            ]

        # Execute the SAM generation compute module (pvwattsv7, windpower, etc)
        self.run()

        # Execute a follow-on SAM econ compute module
        # (lcoe_fcr, singleowner, etc)
        if lcoe_out_reqs is not None:
            self.sam_sys_inputs["annual_energy"] = self.annual_energy()
            lcoe = LCOE(self.sam_sys_inputs, output_request=lcoe_out_reqs)
            lcoe.assign_inputs()
            lcoe.execute()
            lcoe.collect_outputs()
            self.outputs.update(lcoe.outputs)

        elif so_out_reqs is not None:
            self.sam_sys_inputs["gen"] = self.gen_profile()
            so = SingleOwner(self.sam_sys_inputs, output_request=so_out_reqs)
            so.assign_inputs()
            so.execute()
            so.collect_outputs()
            self.outputs.update(so.outputs)

    def run(self):
        """Run a reV-SAM generation object by assigning inputs, executing the
        SAM simulation, collecting outputs, and converting all arrays to UTC.
        """
        self.assign_inputs()
        self.execute()
        self.collect_outputs()

    @classmethod
    def reV_run(
        cls,
        points_control,
        res_file,
        site_df,
        lr_res_file=None,
        output_request=("cf_mean",),
        drop_leap=False,
        gid_map=None,
        nn_map=None,
        bias_correct=None,
    ):
        """Execute SAM generation based on a reV points control instance.

        Parameters
        ----------
        points_control : config.PointsControl
            PointsControl instance containing project points site and SAM
            config info.
        res_file : str
            Resource file with full path.
        site_df : pd.DataFrame
            Dataframe of site-specific input variables. Row index corresponds
            to site number/gid (via df.loc not df.iloc), column labels are the
            variable keys that will be passed forward as SAM parameters.
        lr_res_file : str | None
            Optional low resolution resource file that will be dynamically
            mapped+interpolated to the nominal-resolution res_file. This
            needs to be of the same format as resource_file, e.g. they both
            need to be handled by the same rex Resource handler such as
            WindResource
        output_request : list | tuple
            Outputs to retrieve from SAM.
        drop_leap : bool
            Drops February 29th from the resource data. If False, December
            31st is dropped from leap years.
        gid_map : None | dict
            Mapping of unique integer generation gids (keys) to single integer
            resource gids (values). This enables the user to input unique
            generation gids in the project points that map to non-unique
            resource gids. This can be None or a pre-extracted dict.
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
        out : dict
            Nested dictionaries where the top level key is the site index,
            the second level key is the variable name, second level value is
            the output variable value.
        """
        # initialize output dictionary
        out = {}
        points = points_control.project_points

        # Get the RevPySam resource object
        resources = RevPySam.get_sam_res(
            res_file,
            points,
            points.tech,
            output_request=output_request,
            gid_map=gid_map,
            lr_res_file=lr_res_file,
            nn_map=nn_map,
            bias_correct=bias_correct,
        )

        # run resource through curtailment filter if applicable
        curtailment = points.curtailment
        if curtailment is not None:
            for curtail_type, curtail_config in curtailment.items():
                curtail_sites = points.get_sites_from_curtailment(curtail_type)
                if not curtail_sites:
                    continue
                resources = curtail(resources, curtail_config, curtail_sites,
                                    random_seed=curtail_config.random_seed)

        # iterate through project_points gen_gid values
        for gen_gid in points.sites:
            # Lookup the resource gid if there's a mapping and get the resource
            # data from the SAMResource object using the res_gid.
            res_gid = gen_gid if gid_map is None else gid_map[gen_gid]
            site_res_df, site_meta = resources._get_res_df(res_gid)

            # drop the leap day
            if drop_leap:
                site_res_df = cls.drop_leap(site_res_df)

            _, inputs = points[gen_gid]

            # get resource data pass-throughs and resource means
            res_outs, out_req_cleaned = cls._get_res(
                site_res_df, output_request
            )
            res_mean, out_req_cleaned = cls._get_res_mean(
                resources, res_gid, out_req_cleaned
            )

            # iterate through requested sites.
            sim = cls(
                resource=site_res_df,
                meta=site_meta,
                sam_sys_inputs=inputs,
                output_request=out_req_cleaned,
                site_sys_inputs=dict(site_df.loc[gen_gid, :]),
            )
            sim.run_gen_and_econ()

            # collect outputs to dictout
            out[gen_gid] = sim.outputs

            if res_outs is not None:
                out[gen_gid].update(res_outs)

            if res_mean is not None:
                out[gen_gid].update(res_mean)

        return out


class AbstractSamGenerationFromWeatherFile(AbstractSamGeneration, ABC):
    """Base class for running sam generation with a weather file on disk."""

    WF_META_DROP_COLS = {
        ResourceMetaField.LATITUDE,
        ResourceMetaField.LONGITUDE,
        ResourceMetaField.ELEVATION,
        ResourceMetaField.TIMEZONE,
        ResourceMetaField.COUNTRY,
        ResourceMetaField.STATE,
        ResourceMetaField.COUNTY,
        "urban",
        "population",
        "landcover",
    }

    @property
    @abstractmethod
    def PYSAM_WEATHER_TAG(self):
        """Name of the weather file input used by SAM generation module."""
        raise NotImplementedError

    def set_resource_data(self, resource, meta):
        """Generate the weather file and set the path as an input.

        Some PySAM models require a data file, not raw data. This method
        generates the weather data, writes it to a file on disk, and
        then sets the file as an input to the generation module. The
        function
        :meth:`~AbstractSamGenerationFromWeatherFile.run_gen_and_econ`
        deletes the file on disk after a run is complete.

        Parameters
        ----------
        resource : pd.DataFrame
            Time series resource data for a single location with a
            pandas DatetimeIndex. There must be columns for all the
            required variables to run the respective SAM simulation.
            Remapping will be done to convert typical NSRDB/WTK names
            into SAM names (e.g. DNI -> dn and wind_speed -> windspeed).
        meta : pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude,
            elevation, and timezone.
        """
        meta = self._parse_meta(meta)
        self.time_interval = self.get_time_interval(resource.index.values)
        pysam_w_fname = self._create_pysam_wfile(resource, meta)
        self[self.PYSAM_WEATHER_TAG] = pysam_w_fname

    def _create_pysam_wfile(self, resource, meta):
        """Create PySAM weather input file.

        Parameters
        ----------
        resource : pd.DataFrame
            Time series resource data for a single location with a
            pandas DatetimeIndex. There must be columns for all the
            required variables to run the respective SAM simulation.
            Remapping will be done to convert typical NSRDB/WTK names
            into SAM names (e.g. DNI -> dn and wind_speed -> windspeed).
        meta : pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude,
            elevation, and timezone.

        Returns
        -------
        fname : str
            Name of weather csv file.

        Notes
        -----
        PySAM will not accept data on Feb 29th. For leap years,
        December 31st is dropped and time steps are shifted to relabel
        Feb 29th as March 1st, March 1st as March 2nd, etc.
        """
        # pylint: disable=attribute-defined-outside-init,consider-using-with
        self._temp_dir = TemporaryDirectory()
        fname = os.path.join(self._temp_dir.name, "weather.csv")
        logger.debug("Creating PySAM weather data file: {}".format(fname))

        # ------- Process metadata
        m = pd.DataFrame(meta).T
        timezone = m[ResourceMetaField.TIMEZONE]
        m["Source"] = "NSRDB"
        m["Location ID"] = meta.name
        m["City"] = "-"
        m["State"] = m["state"].apply(lambda x: "-" if x == "None" else x)
        m["Country"] = m["country"].apply(lambda x: "-" if x == "None" else x)
        m["Latitude"] = m[ResourceMetaField.LATITUDE]
        m["Longitude"] = m[ResourceMetaField.LONGITUDE]
        m["Time Zone"] = timezone
        m["Elevation"] = m[ResourceMetaField.ELEVATION]
        m["Local Time Zone"] = timezone
        m["Dew Point Units"] = "c"
        m["DHI Units"] = "w/m2"
        m["DNI Units"] = "w/m2"
        m["Temperature Units"] = "c"
        m["Pressure Units"] = "mbar"
        m["Wind Speed"] = "m/s"
        keep_cols = [c for c in m.columns if c not in self.WF_META_DROP_COLS]
        m[keep_cols].to_csv(fname, index=False, mode="w")

        # --------- Process data
        var_map = {
            "dni": "DNI",
            "dhi": "DHI",
            "wind_speed": "Wind Speed",
            "air_temperature": "Temperature",
            "dew_point": "Dew Point",
            "surface_pressure": "Pressure",
        }
        resource = resource.rename(mapper=var_map, axis="columns")

        time_index = resource.index
        # Adjust from UTC to local time
        local = np.roll(
            resource.values, int(timezone * self.time_interval), axis=0
        )
        resource = pd.DataFrame(
            local, columns=resource.columns, index=time_index
        )
        mask = (time_index.month == 2) & (time_index.day == 29)
        time_index = time_index[~mask]

        df = pd.DataFrame(index=time_index)
        df["Year"] = time_index.year
        df["Month"] = time_index.month
        df["Day"] = time_index.day
        df["Hour"] = time_index.hour
        df["Minute"] = time_index.minute
        df = df.join(resource.loc[~mask])

        df.to_csv(fname, index=False, mode="a")

        return fname

    def run_gen_and_econ(self):
        """Run SAM generation and possibility follow-on econ analysis."""
        try:
            super().run_gen_and_econ()
        finally:
            temp_dir = getattr(self, "_temp_dir", None)
            if temp_dir is not None:
                temp_dir.cleanup()


class AbstractSamSolar(AbstractSamGeneration, ABC):
    """Base Class for Solar generation from SAM"""

    @staticmethod
    def agg_albedo(time_index, albedo):
        """Aggregate a timeseries of albedo data to monthly values w len 12 as
        required by pysam Pvsamv1

        Tech spec from pysam docs:
        https://nrel-pysam.readthedocs.io/en/master/modules/Pvsamv1.html
        #PySAM.Pvsamv1.Pvsamv1.SolarResource.albedo

        Parameters
        ----------
        time_index : pd.DatetimeIndex
            Timeseries solar resource datetimeindex
        albedo : list
            Timeseries Albedo data to be aggregated. Should be 0-1 and likely
            hourly or less.

        Returns
        -------
        monthly_albedo : list
            1D list of monthly albedo values with length 12
        """
        monthly_albedo = np.zeros(12).tolist()
        albedo = np.array(albedo)
        for month in range(1, 13):
            m = np.where(time_index.month == month)[0]
            monthly_albedo[int(month - 1)] = albedo[m].mean()

        return monthly_albedo

    def set_resource_data(self, resource, meta):
        """Set NSRDB resource data arrays.

        Parameters
        ----------
        resource : pd.DataFrame
            Timeseries solar or wind resource data for a single location with a
            pandas DatetimeIndex.  There must be columns for all the required
            variables to run the respective SAM simulation. Remapping will be
            done to convert typical NSRDB/WTK names into SAM names (e.g. DNI ->
            dn and wind_speed -> windspeed)
        meta : pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude, elevation,
            and timezone.
        """

        meta = self._parse_meta(meta)
        time_index = resource.index
        self.time_interval = self.get_time_interval(resource.index.values)

        # map resource data names to SAM required data names
        var_map = {
            "dni": "dn",
            "dhi": "df",
            "ghi": "gh",
            "clearskydni": "dn",
            "clearskydhi": "df",
            "clearskyghi": "gh",
            "windspeed": "wspd",
            "airtemperature": "tdry",
            "temperature": "tdry",
            "temp": "tdry",
            "dewpoint": "tdew",
            "surfacepressure": "pres",
            "pressure": "pres",
            "surfacealbedo": "albedo",
        }
        lower_case = {
            k: k.lower().replace(" ", "").replace("_", "")
            for k in resource.columns
        }
        irrad_vars = ["dn", "df", "gh"]

        resource = resource.rename(mapper=lower_case, axis="columns")
        resource = resource.rename(mapper=var_map, axis="columns")
        time_index = resource.index
        resource = {
            k: np.array(v)
            for (k, v) in resource.to_dict(orient="list").items()
        }

        # set resource variables
        for var, arr in resource.items():
            if var != "time_index":
                # ensure that resource array length is multiple of 8760
                arr = self.ensure_res_len(arr, time_index)
                n_roll = int(
                    self._meta[ResourceMetaField.TIMEZONE] * self.time_interval
                )
                arr = np.roll(arr, n_roll)

                if var in irrad_vars and np.min(arr) < 0:
                    warn(
                        'Solar irradiance variable "{}" has a minimum '
                        "value of {}. Truncating to zero.".format(
                            var, np.min(arr)
                        ),
                        SAMInputWarning,
                    )
                    arr = np.where(arr < 0, 0, arr)

                resource[var] = arr.tolist()

        resource["lat"] = meta[ResourceMetaField.LATITUDE]
        resource["lon"] = meta[ResourceMetaField.LONGITUDE]
        resource["tz"] = meta[ResourceMetaField.TIMEZONE]

        resource["elev"] = meta.get(ResourceMetaField.ELEVATION, 0.0)

        time_index = self.ensure_res_len(time_index, time_index)
        resource["minute"] = time_index.minute
        resource["hour"] = time_index.hour
        resource["month"] = time_index.month
        resource["year"] = time_index.year
        resource["day"] = time_index.day

        if "albedo" in resource:
            self["albedo"] = self.agg_albedo(
                time_index, resource.pop("albedo")
            )

        self["solar_resource_data"] = resource


class AbstractSamPv(AbstractSamSolar, ABC):
    """Photovoltaic (PV) generation with either pvwatts of detailed pv."""

    # set these class attrs in concrete subclasses
    MODULE = None
    PYSAM = None

    # pylint: disable=line-too-long
    def __init__(
        self,
        resource,
        meta,
        sam_sys_inputs,
        site_sys_inputs=None,
        output_request=None,
        drop_leap=False,
    ):
        """Initialize a SAM solar object.

        See the PySAM :py:class:`~PySAM.Pvwattsv8.Pvwattsv8` (or older
        version model) or :py:class:`~PySAM.Pvsamv1.Pvsamv1` documentation for
        the configuration keys required in the `sam_sys_inputs` config for the
        respective models. Some notable keys include the following to enable a
        lifetime simulation (non-exhaustive):

            - ``system_use_lifetime_output`` : Integer flag indicating whether
              or not to run a full lifetime model (0 for off, 1 for on). If
              running a lifetime model, the resource file will be repeated
              for the number of years specified as the lifetime of the
              plant and a performance degradation term will be used to
              simulate reduced performance over time.
            - ``analysis_period`` : Integer representing the number of years
              to include in the lifetime of the model generator. Required if
              ``system_use_lifetime_output`` is set to 1.
            - ``dc_degradation`` : List of percentage values representing the
              annual DC degradation of capacity factors. Maybe a single value
              that will be compound each year or a vector of yearly rates.
              Required if ``system_use_lifetime_output`` is set to 1.

        You may also include the following ``reV``-specific keys:

            - ``reV_outages`` : Specification for ``reV``-scheduled
              stochastic outage losses. For example::

                    outage_info = [
                        {
                            'count': 6,
                            'duration': 24,
                            'percentage_of_capacity_lost': 100,
                            'allowed_months': ['January', 'March'],
                            'allow_outage_overlap': True
                        },
                        {
                            'count': 10,
                            'duration': 1,
                            'percentage_of_capacity_lost': 10,
                            'allowed_months': ['January'],
                            'allow_outage_overlap': False
                        },
                        ...
                    ]

              See the description of
              :meth:`~reV.losses.scheduled.ScheduledLossesMixin.add_scheduled_losses`
              or the
              `reV losses demo notebook <https://tinyurl.com/4d7uutt3/>`_
              for detailed instructions on how to specify this input.
            - ``reV_outages_seed`` : Integer value used to seed the RNG
              used to compute stochastic outage losses.
            - ``time_index_step`` : Integer representing the step size
              used to sample the ``time_index`` in the resource data.
              This can be used to reduce temporal resolution (i.e. for
              30 minute NSRDB input data, ``time_index_step=1`` yields
              the full 30 minute time series as output, while
              ``time_index_step=2`` yields hourly output, and so forth).

              .. Note:: The reduced data shape (i.e. after applying a
                        step size of `time_index_step`) must still be an
                        integer multiple of 8760, or the execution will
                        fail.

            - ``clearsky`` : Boolean flag value indicating wether
              computation should use clearsky resource data to compute
              generation data.

        Parameters
        ----------
        resource : pd.DataFrame
            Timeseries solar or wind resource data for a single location with a
            pandas DatetimeIndex.  There must be columns for all the required
            variables to run the respective SAM simulation. Remapping will be
            done to convert typical NSRDB/WTK names into SAM names (e.g. DNI ->
            dn and wind_speed -> windspeed)
        meta : pd.DataFrame | pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude, elevation,
            and timezone.
        sam_sys_inputs : dict
            Site-agnostic SAM system model inputs arguments.
        site_sys_inputs : dict
            Optional set of site-specific SAM system inputs to complement the
            site-agnostic inputs.
        output_request : list
            Requested SAM outputs (e.g., 'cf_mean', 'annual_energy',
            'cf_profile', 'gen_profile', 'energy_yield', 'ppa_price',
            'lcoe_fcr').
        drop_leap : bool
            Drops February 29th from the resource data. If False, December
            31st is dropped from leap years.
        """

        # need to check tilt=lat and azimuth for pv systems
        meta = self._parse_meta(meta)
        sam_sys_inputs = self.set_latitude_tilt_az(sam_sys_inputs, meta)

        super().__init__(
            resource,
            meta,
            sam_sys_inputs,
            site_sys_inputs=site_sys_inputs,
            output_request=output_request,
            drop_leap=drop_leap,
        )

    def set_resource_data(self, resource, meta):
        """Set NSRDB resource data arrays.

        Parameters
        ----------
        resource : pd.DataFrame
            Timeseries solar or wind resource data for a single location with a
            pandas DatetimeIndex.  There must be columns for all the required
            variables to run the respective SAM simulation. Remapping will be
            done to convert typical NSRDB/WTK names into SAM names (e.g. DNI ->
            dn and wind_speed -> windspeed)
        meta : pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude, elevation,
            and timezone.

        Raises
        ------
        ValueError : If lat/lon outside of -90 to 90 and -180 to 180,
                     respectively.

        """
        bad_location_input = (
            (meta[ResourceMetaField.LATITUDE] < -90)
            | (meta[ResourceMetaField.LATITUDE] > 90)
            | (meta[ResourceMetaField.LONGITUDE] < -180)
            | (meta[ResourceMetaField.LONGITUDE] > 180)
        )
        if bad_location_input.any():
            raise ValueError(
                "Detected latitude/longitude values outside of "
                "the range -90 to 90 and -180 to 180, "
                "respectively. Please ensure input resource data"
                "locations conform to these ranges. "
            )
        return super().set_resource_data(resource, meta)

    @staticmethod
    def set_latitude_tilt_az(sam_sys_inputs, meta):
        """Check if tilt is specified as latitude and set tilt=lat, az=180 or 0

        Parameters
        ----------
        sam_sys_inputs : dict
            Site-agnostic SAM system model inputs arguments.
        meta : pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude, elevation,
            and timezone.

        Returns
        -------
        sam_sys_inputs : dict
            Site-agnostic SAM system model inputs arguments.
            If for a pv simulation the "tilt" parameter was originally not
            present or set to 'lat' or MetaKeyName.LATITUDE, the tilt will be
            set to the absolute value of the latitude found in meta and the
            azimuth will be 180 if lat>0, 0 if lat<0.
        """

        set_tilt = False
        if sam_sys_inputs is not None and meta is not None:
            if "tilt" not in sam_sys_inputs:
                warn(
                    "No tilt specified, setting at latitude.", SAMInputWarning
                )
                set_tilt = True
            elif (
                sam_sys_inputs["tilt"] == "lat"
                or sam_sys_inputs["tilt"] == ResourceMetaField.LATITUDE
            ) or (
                sam_sys_inputs["tilt"] == "lat"
                or sam_sys_inputs["tilt"] == ResourceMetaField.LATITUDE
            ):
                set_tilt = True

        if set_tilt:
            # set tilt to abs(latitude)
            sam_sys_inputs["tilt"] = np.abs(meta[ResourceMetaField.LATITUDE])
            if meta[ResourceMetaField.LATITUDE] > 0:
                # above the equator, az = 180
                sam_sys_inputs["azimuth"] = 180
            else:
                # below the equator, az = 0
                sam_sys_inputs["azimuth"] = 0

            logger.debug(
                'Tilt specified at "latitude", setting tilt to: {}, '
                "azimuth to: {}".format(
                    sam_sys_inputs["tilt"], sam_sys_inputs["azimuth"]
                )
            )
        return sam_sys_inputs

    def system_capacity_ac(self):
        """Get AC system capacity from SAM inputs.

        NOTE: AC nameplate = DC nameplate / ILR

        Returns
        -------
        cf_profile : float
            AC nameplate = DC nameplate / ILR
        """
        return (
            self.sam_sys_inputs["system_capacity"]
            / self.sam_sys_inputs["dc_ac_ratio"]
        )

    def cf_mean(self):
        """Get mean capacity factor (fractional) from SAM.

        NOTE: PV capacity factor is the AC power production / the DC nameplate

        Returns
        -------
        output : float
            Mean capacity factor (fractional).
            PV CF is calculated as AC power / DC nameplate.
        """
        return self["capacity_factor"] / 100

    def cf_mean_ac(self):
        """Get mean AC capacity factor (fractional) from SAM.

        NOTE: This value only available in PVWattsV8 and up.

        Returns
        -------
        output : float
            Mean AC capacity factor (fractional).
            PV AC CF is calculated as AC power / AC nameplate.
        """
        return self["capacity_factor_ac"] / 100

    def cf_profile(self):
        """Get hourly capacity factor (frac) profile in local timezone.
        See self.outputs attribute for collected output data in UTC.

        NOTE: PV capacity factor is the AC power production / the DC nameplate

        Returns
        -------
        cf_profile : np.ndarray
            1D numpy array of capacity factor profile.
            Datatype is float32 and array length is 8760*time_interval.
            PV CF is calculated as AC power / DC nameplate.
        """
        return self.gen_profile() / self.sam_sys_inputs["system_capacity"]

    def cf_profile_ac(self):
        """Get hourly AC capacity factor (frac) profile in local timezone.
        See self.outputs attribute for collected output data in UTC.

        NOTE: PV AC capacity factor is the AC power production / the AC
        nameplate. AC nameplate = DC nameplate / ILR

        Returns
        -------
        cf_profile : np.ndarray
            1D numpy array of capacity factor profile.
            Datatype is float32 and array length is 8760*time_interval.
            PV AC CF is calculated as AC power / AC nameplate.
        """
        return self.gen_profile() / self.system_capacity_ac()

    def gen_profile(self):
        """Get AC inverter power generation profile (local timezone) in kW.
        This is an alias of the "ac" SAM output variable if PySAM version>=3.
        See self.outputs attribute for collected output data in UTC.

        Returns
        -------
        output : np.ndarray
            1D array of AC inverter power generation in kW.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return np.array(self["gen"], dtype=np.float32)

    def ac(self):
        """Get AC inverter power generation profile (local timezone) in kW.
        See self.outputs attribute for collected output data in UTC.

        Returns
        -------
        output : np.ndarray
            1D array of AC inverter power generation in kW.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return np.array(self["ac"], dtype=np.float32) / 1000

    def dc(self):
        """
        Get DC array power generation profile (local timezone) in kW.
        See self.outputs attribute for collected output data in UTC.

        Returns
        -------
        output : np.ndarray
            1D array of DC array power generation in kW.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return np.array(self["dc"], dtype=np.float32) / 1000

    def clipped_power(self):
        """
        Get the clipped DC power generated behind the inverter
        (local timezone) in kW.
        See self.outputs attribute for collected output data in UTC.

        Returns
        -------
        clipped : np.ndarray
            1D array of clipped DC power in kW.
            Datatype is float32 and array length is 8760*time_interval.
        """
        ac = self.ac()
        dc = self.dc()

        return np.where(ac < ac.max(), 0, dc - ac)

    @staticmethod
    @abstractmethod
    def default():
        """Get the executed default pysam object."""

    def collect_outputs(self, output_lookup=None):
        """Collect SAM output_request, convert timeseries outputs to UTC, and
        save outputs to self.outputs property.

        Parameters
        ----------
        output_lookup : dict | None
            Lookup dictionary mapping output keys to special output methods.
            None defaults to generation default outputs.
        """

        if output_lookup is None:
            output_lookup = {
                "cf_mean": self.cf_mean,
                "cf_mean_ac": self.cf_mean_ac,
                "cf_profile": self.cf_profile,
                "cf_profile_ac": self.cf_profile_ac,
                "annual_energy": self.annual_energy,
                "energy_yield": self.energy_yield,
                "gen_profile": self.gen_profile,
                "ac": self.ac,
                "dc": self.dc,
                "clipped_power": self.clipped_power,
                "system_capacity_ac": self.system_capacity_ac,
            }

        super().collect_outputs(output_lookup=output_lookup)


class PvWattsv5(AbstractSamPv):
    """Photovoltaic (PV) generation with pvwattsv5."""

    MODULE = "pvwattsv5"
    PYSAM = PySamPv5

    @staticmethod
    def default():
        """Get the executed default pysam PVWATTSV5 object.

        Returns
        -------
        PySAM.Pvwattsv5
        """
        return DefaultPvWattsv5.default()


class PvWattsv7(AbstractSamPv):
    """Photovoltaic (PV) generation with pvwattsv7."""

    MODULE = "pvwattsv7"
    PYSAM = PySamPv7

    @staticmethod
    def default():
        """Get the executed default pysam PVWATTSV7 object.

        Returns
        -------
        PySAM.Pvwattsv7
        """
        raise NotImplementedError("Pvwattsv7 default file no longer exists!")


class PvWattsv8(AbstractSamPv):
    """Photovoltaic (PV) generation with pvwattsv8."""

    MODULE = "pvwattsv8"
    PYSAM = PySamPv8

    @staticmethod
    def default():
        """Get the executed default pysam PVWATTSV8 object.

        Returns
        -------
        PySAM.Pvwattsv8
        """
        return DefaultPvWattsv8.default()


class PvSamv1(AbstractSamPv):
    """Detailed PV model"""

    MODULE = "Pvsamv1"
    PYSAM = PySamDetailedPv

    def ac(self):
        """Get AC inverter power generation profile (local timezone) in kW.
        See self.outputs attribute for collected output data in UTC.

        Returns
        -------
        output : np.ndarray
            1D array of AC inverter power generation in kW.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return np.array(self["gen"], dtype=np.float32)

    def dc(self):
        """
        Get DC array power generation profile (local timezone) in kW.
        See self.outputs attribute for collected output data in UTC.

        Returns
        -------
        output : np.ndarray
            1D array of DC array power generation in kW.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return np.array(self["dc_net"], dtype=np.float32)

    @staticmethod
    def default():
        """Get the executed default pysam Pvsamv1 object.

        Returns
        -------
        PySAM.Pvsamv1
        """
        return DefaultPvSamv1.default()


class TcsMoltenSalt(AbstractSamSolar):
    """Concentrated Solar Power (CSP) generation with tower molten salt"""

    MODULE = "tcsmolten_salt"
    PYSAM = PySamCSP

    def cf_profile(self):
        """Get absolute value hourly capacity factor (frac) profile in
        local timezone.
        See self.outputs attribute for collected output data in UTC.

        Returns
        -------
        cf_profile : np.ndarray
            1D numpy array of capacity factor profile.
            Datatype is float32 and array length is 8760*time_interval.
        """
        x = np.abs(self.gen_profile() / self.sam_sys_inputs["system_capacity"])
        return x

    @staticmethod
    def default():
        """Get the executed default pysam CSP object.

        Returns
        -------
        PySAM.TcsmoltenSalt
        """
        return DefaultTcsMoltenSalt.default()


class SolarWaterHeat(AbstractSamGenerationFromWeatherFile):
    """
    Solar Water Heater generation
    """

    MODULE = "solarwaterheat"
    PYSAM = PySamSwh
    PYSAM_WEATHER_TAG = "solar_resource_file"

    @staticmethod
    def default():
        """Get the executed default pysam swh object.

        Returns
        -------
        PySAM.Swh
        """
        return DefaultSwh.default()


class LinearDirectSteam(AbstractSamGenerationFromWeatherFile):
    """
    Process heat linear Fresnel direct steam generation
    """

    MODULE = "lineardirectsteam"
    PYSAM = PySamLds
    PYSAM_WEATHER_TAG = "file_name"
    _GEN_KEY = "gen_heat"

    def cf_mean(self):
        """Calculate mean capacity factor (fractional) from SAM.

        Returns
        -------
        output : float
            Mean capacity factor (fractional).
        """
        net_power = (
            self["annual_field_energy"] - self["annual_thermal_consumption"]
        )  # kW-hr
        # q_pb_des is in MW, convert to kW-hr
        name_plate = self["q_pb_des"] * 8760 * 1000

        return net_power / name_plate

    @staticmethod
    def default():
        """Get the executed default pysam linear Fresnel object.

        Returns
        -------
        PySAM.LinearFresnelDsgIph
        """
        return DefaultLinearFresnelDsgIph.default()


# pylint: disable=line-too-long
class Geothermal(AbstractSamGenerationFromWeatherFile):
    """reV-SAM geothermal generation.

    As of 12/20/2022, the resource potential input in SAM is only used
    to calculate the number of well replacements during the lifetime of
    a geothermal plant. It was decided that reV would not model well
    replacements. Therefore, reV sets the resource potential to match
    (or be just above) the gross potential so that SAM does not throw
    any errors.

    Also as of 12/20/2022, the SAM GETEM module requires a weather file,
    but does not actually require any weather data to run. Therefore,
    reV currently generates an empty weather file to pass to SAM. This
    behavior can be easily updated in the future should the SAM GETEM
    module start using weather data.

    See the PySAM :py:class:`~PySAM.Geothermal.Geothermal` documentation
    for the configuration keys required in the `sam_sys_inputs` config.
    Some notable keys include (non-exhaustive):

        - ``resource_type`` : Integer flag representing either
          Hydrothermal (0) or EGS (1) resource. Only values of 0 or 1
          allowed.
        - ``resource_potential`` : Total resource potential at location
          (in MW).

          .. Important:: ``reV`` automatically sets the resource
             potential to match the gross potential (see documentation
             above), so this key should be left out of the config (it
             will be overridden in any case).

        - ``resource_temp`` : Temperature of resource (in C).

          .. Important:: This value is set by ``reV`` based on the
             user's geothermal resource data input. To override this
             behavior, users *may* specify their own ``resource_temp``
             value (either a single value for all sites in the SAM
             geothermal config or a  site-dependent value in the project
             points CSV). In this case, the resource temperature from
             the input data will be ignored completely, and the
             temperature at each location will be determined solely from
             this input.

        - ``resource_depth`` : Depth to geothermal resource (in m).
        - ``analysis_type`` : Integer flag representing the plant
          configuration. If the ``nameplate`` input is to be used to
          specify the plant capacity, then this flag should be set to 0
          (this is the default ``reV`` assumption). Otherwise, if the
          ``num_wells`` input is to be used to specify the plant site,
          then this flag should be set to 1. Only values of 0 or 1
          allowed.

        - ``nameplate`` : Geothermal plant size (in kW). Only affects
          the output if ``analysis_type=0``.

          .. Important:: Unlike wind or solar, ``reV`` geothermal
             dynamically sets the size of a geothermal plant. In
             particular, the plant capacity is set to match the resource
             potential (obtained from the input data) for each site. For
             this to work, users **must** leave out the ``nameplate``
             key from the SAM config.

             Alternatively, users *may* specify their own ``nameplate``
             capacity value (either a single value for all sites in the
             SAM geothermal config or a site-dependent value in the
             project points CSV). In this case, the resource potential
             from the input data will be ignored completely, and the
             capacity at each location will be determined solely from
             this input.

        - ``num_wells`` : Number of wells at each plant. This value is
          used to determined plant capacity if ``analysis_type=1``.
          Otherwise this input has no effect.
        - ``num_wells_getem`` : Number of wells assumed at each plant
          for power block calculations. Only affects power block outputs
          if ``analysis_type=0`` (otherwise the ``num_wells`` input is
          used in power block calculations).

          .. Note:: ``reV`` does not currently adjust this value based
             on the resource input (as it probably should). If any
             power block outputs are required in the future, there may
             need to be extra development to set this value based on
             the dynamically calculated plant size.

        - ``conversion_type`` : Integer flag representing the conversion
          plant type. Either Binary (0) or Flash (1). Only values of 0
          or 1 allowed.
        - ``geotherm.cost.inj_prod_well_ratio`` : Fraction representing
          the injection to production well ratio (0-1). SAM GUI defaults
          to 0.5 for this value, but it is recommended to set this to
          the GETEM default of 0.75.


    You may also include the following ``reV``-specific keys:

        - ``num_confirmation_wells`` : Number of confirmation wells that
          can also be used as production wells. This number is used to
          determined to total number of wells required at each plant,
          and therefore the total drilling costs. This value defaults to
          2 (to match the SAM GUI as of 8/1/2023). However, the default
          value can lead to negative costs if the plant size is small
          (e.g. only 1 production well is needed, so the costs equal
          -1 * ``drill_cost_per_well``). This is a limitation of the
          SAM calculations (as of 8/1/2023), and it is therefore useful
          to set ``num_confirmation_wells=0`` when performing ``reV``
          runs for small plant sizes.
        - ``capital_cost_per_kw`` : Capital cost values in $/kW. If
          this value is specified in the config, reV calculates and
          overrides the total ``capital_cost`` value based on the
          geothermal plant size (capacity) at each location.
        - ``fixed_operating_cost`` : Fixed operating cost values in
          $/kW. If this value is specified in the config, reV calculates
          and overrides the total ``fixed_operating_cost`` value based
          on the geothermal plant size (capacity) at each location.
        - ``drill_cost_per_well`` : Drilling cost per well, in $. If
          this value is specified in the config, reV calculates the
          total drilling costs based on the number of wells that need to
          be drilled at each location. The drilling costs are added to
          the total ``capital_cost`` at each location.
        - ``reV_outages`` : Specification for ``reV``-scheduled
          stochastic outage losses. For example::

                outage_info = [
                    {
                        'count': 6,
                        'duration': 24,
                        'percentage_of_capacity_lost': 100,
                        'allowed_months': ['January', 'March'],
                        'allow_outage_overlap': True
                    },
                    {
                        'count': 10,
                        'duration': 1,
                        'percentage_of_capacity_lost': 10,
                        'allowed_months': ['January'],
                        'allow_outage_overlap': False
                    },
                    ...
                ]

          See the description of
          :meth:`~reV.losses.scheduled.ScheduledLossesMixin.add_scheduled_losses`
          or the
          `reV losses demo notebook <https://tinyurl.com/4d7uutt3/>`_
          for detailed instructions on how to specify this input.
        - ``reV_outages_seed`` : Integer value used to seed the RNG
          used to compute stochastic outage losses.
        - ``time_index_step`` : Integer representing the step size
          used to sample the ``time_index`` in the resource data.
          This can be used to reduce temporal resolution (i.e. for
          30 minute NSRDB input data, ``time_index_step=1`` yields
          the full 30 minute time series as output, while
          ``time_index_step=2`` yields hourly output, and so forth).

    """

    MODULE = "geothermal"
    PYSAM = PySamGeothermal
    PYSAM_WEATHER_TAG = "file_name"
    _RESOURCE_POTENTIAL_MULT = 1.001
    _DEFAULT_NUM_CONFIRMATION_WELLS = 2  # SAM GUI default as of 5/26/23

    @staticmethod
    def default():
        """Get the executed default PySAM Geothermal object.

        Returns
        -------
        PySAM.Geothermal
        """
        return DefaultGeothermal.default()

    def cf_profile(self):
        """Get hourly capacity factor (frac) profile in local timezone.
        See self.outputs attribute for collected output data in UTC.

        Returns
        -------
        cf_profile : np.ndarray
            1D numpy array of capacity factor profile.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return self.gen_profile() / self.sam_sys_inputs["nameplate"]

    def assign_inputs(self):
        """Assign the self.sam_sys_inputs attribute to the PySAM object."""
        if self.sam_sys_inputs.get("ui_calculations_only"):
            msg = (
                "reV requires model run - cannot set "
                '"ui_calculations_only" to `True` (1). Automatically '
                "setting to `False` (0)!"
            )
            logger.warning(msg)
            warn(msg)
            self.sam_sys_inputs["ui_calculations_only"] = 0
        super().assign_inputs()

    def set_resource_data(self, resource, meta):
        """Generate the weather file and set the path as an input.

        The Geothermal PySAM model requires a data file, not raw data.
        This method generates the weather data, writes it to a file on
        disk, and then sets the file as an input to the Geothermal
        generation module. The function
        :meth:`~AbstractSamGenerationFromWeatherFile.run_gen_and_econ`
        deletes the file on disk after a run is complete.

        Parameters
        ----------
        resource : pd.DataFrame
            Time series resource data for a single location with a
            pandas DatetimeIndex. There must be columns for all the
            required variables to run the respective SAM simulation.
        meta : pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude,
            elevation, and timezone.
        """
        super().set_resource_data(resource, meta)
        self._set_resource_temperature(resource)
        self._set_egs_plant_design_temperature()
        self._set_nameplate_to_match_resource_potential(resource)
        self._set_resource_potential_to_match_gross_output()
        self._set_costs()

    def _set_resource_temperature(self, resource):
        """Set resource temp from data if user did not specify it."""

        if "resource_temp" in self.sam_sys_inputs:
            logger.debug(
                "Found 'resource_temp' value in SAM config: {:.2f}".format(
                    self.sam_sys_inputs["resource_temp"]
                )
            )
            return

        val = set(resource["temperature"].unique())
        logger.debug(
            "Found {} value(s) for 'temperature' in resource data".format(
                len(val)
            )
        )
        if len(val) > 1:
            msg = (
                "Found multiple values for 'temperature' for site "
                "{}: {}".format(self.site, val)
            )
            logger.error(msg)
            raise InputError(msg)

        val = val.pop()
        logger.debug(
            "Input 'resource_temp' not found in SAM config - setting "
            "to {:.2f} based on input resource data.".format(val)
        )
        self.sam_sys_inputs["resource_temp"] = val

    def _set_egs_plant_design_temperature(self):
        """Set the EGS plant temp to match resource (avoids cf > 1)"""
        if self.sam_sys_inputs.get("resource_type") != 1:
            return  # Not EGS run

        resource_temp = self.sam_sys_inputs["resource_temp"]
        logger.debug("Setting EGS plant design temperature to match "
                     "resource temperature ({}C)".format(resource_temp))
        self.sam_sys_inputs["design_temp"] = resource_temp

    def _set_nameplate_to_match_resource_potential(self, resource):
        """Set the nameplate capacity to match the resource potential."""

        if "nameplate" in self.sam_sys_inputs:
            msg = (
                'Found "nameplate" input in config! Resource potential '
                "from input data will be ignored. Nameplate capacity is "
                "{}".format(self.sam_sys_inputs["nameplate"])
            )
            logger.info(msg)
            # required for downstream LCOE calcs
            self.sam_sys_inputs["system_capacity"] = (
                self.sam_sys_inputs["nameplate"]
            )
            return

        val = set(resource["potential_MW"].unique())
        if len(val) > 1:
            msg = (
                'Found multiple values for "potential_MW" for site '
                "{}: {}".format(self.site, val)
            )
            logger.error(msg)
            raise InputError(msg)

        val = val.pop() * 1000

        logger.debug("Setting the nameplate to {}".format(val))
        self.sam_sys_inputs["nameplate"] = val
        # required for downstream LCOE calcs
        self.sam_sys_inputs["system_capacity"] = val

    def _set_resource_potential_to_match_gross_output(self):
        """Set the resource potential input to match the gross generation.

        If SAM throws an error during the UI calculation of the gross
        output, the resource_potential is simply set to -1 since
        SAM will error out for this point regardless of the
        resource_potential input.
        """

        super().assign_inputs()
        self["ui_calculations_only"] = 1
        try:
            self.execute()
        except SAMExecutionError:
            self["ui_calculations_only"] = 0
            self.sam_sys_inputs["resource_potential"] = -1
            return

        gross_gen = (
            self.pysam.Outputs.gross_output * self._RESOURCE_POTENTIAL_MULT
        )
        if "resource_potential" in self.sam_sys_inputs:
            msg = (
                'Setting "resource_potential" is not allowed! Updating '
                "user input of {} to match the gross generation: {}".format(
                    self.sam_sys_inputs["resource_potential"], gross_gen
                )
            )
            logger.warning(msg)
            warn(msg)

        logger.debug(
            "Setting the resource potential to {} MW".format(gross_gen)
        )
        self.sam_sys_inputs["resource_potential"] = gross_gen

        ncw = self.sam_sys_inputs.pop(
            "num_confirmation_wells", self._DEFAULT_NUM_CONFIRMATION_WELLS
        )
        self.sam_sys_inputs["prod_and_inj_wells_to_drill"] = (
            self.pysam.Outputs.num_wells_getem_output
            - ncw
            + self.pysam.Outputs.num_wells_getem_inj
        )
        self["ui_calculations_only"] = 0

    def _set_costs(self):
        """Set the costs based on plant size"""
        plant_size_kw = self.sam_sys_inputs["nameplate"]

        cc_per_kw = self.sam_sys_inputs.pop("capital_cost_per_kw", None)
        if cc_per_kw is not None:
            capital_cost = cc_per_kw * plant_size_kw
            logger.debug(
                "Setting the capital_cost to ${:,.2f}".format(capital_cost)
            )
            reg_mult = self.sam_sys_inputs.get("capital_cost_multiplier", 1)
            self.sam_sys_inputs["base_capital_cost"] = capital_cost
            self.sam_sys_inputs["capital_cost"] = capital_cost * reg_mult

        dc_per_well = self.sam_sys_inputs.pop("drill_cost_per_well", None)
        num_wells = self.sam_sys_inputs.pop(
            "prod_and_inj_wells_to_drill", None
        )
        if dc_per_well is not None:
            if num_wells is None:
                msg = (
                    "Could not determine number of wells to be drilled. "
                    "No drilling costs added!"
                )
                logger.warning(msg)
                warn(msg)
            else:
                capital_cost = self.sam_sys_inputs["capital_cost"]
                drill_cost = dc_per_well * num_wells
                logger.debug(
                    "Setting the drilling cost to ${:,.2f} "
                    "({:.2f} wells at ${:,.2f} per well)".format(
                        drill_cost, num_wells, dc_per_well
                    )
                )
                reg_mult = self.sam_sys_inputs.get(
                    "capital_cost_multiplier", 1
                )
                base_cc = capital_cost / reg_mult
                new_base_cc = base_cc + drill_cost
                self.sam_sys_inputs["base_capital_cost"] = new_base_cc
                self.sam_sys_inputs["capital_cost"] = new_base_cc * reg_mult

        foc_per_kw = self.sam_sys_inputs.pop(
            "fixed_operating_cost_per_kw", None
        )
        if foc_per_kw is not None:
            foc = foc_per_kw * plant_size_kw
            logger.debug(
                "Setting the fixed_operating_cost to ${:,.2f}".format(foc)
            )
            self.sam_sys_inputs["base_fixed_operating_cost"] = foc
            self.sam_sys_inputs["fixed_operating_cost"] = foc

        voc_per_kw = self.sam_sys_inputs.pop(
            "variable_operating_cost_per_kw", None
        )
        if voc_per_kw is not None:
            voc = voc_per_kw * plant_size_kw
            logger.debug(
                "Setting the variable_operating_cost to ${:,.2f}".format(voc)
            )
            self.sam_sys_inputs["base_variable_operating_cost"] = voc
            self.sam_sys_inputs["variable_operating_cost"] = voc

    def _create_pysam_wfile(self, resource, meta):
        """Create PySAM weather input file.

        Geothermal module requires a weather file, but does not actually
        require any weather data to run. Therefore, an empty file is
        generated and passed through.

        Parameters
        ----------
        resource : pd.DataFrame
            Time series resource data for a single location with a
            pandas DatetimeIndex. There must be columns for all the
            required variables to run the respective SAM simulation.
        meta : pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude,
            and timezone.

        Returns
        -------
        fname : str
            Name of weather csv file.

        Notes
        -----
        PySAM will not accept data on Feb 29th. For leap years,
        December 31st is dropped and time steps are shifted to relabel
        Feb 29th as March 1st, March 1st as March 2nd, etc.
        """
        # pylint: disable=attribute-defined-outside-init, consider-using-with
        self._temp_dir = TemporaryDirectory()
        fname = os.path.join(self._temp_dir.name, "weather.csv")
        logger.debug("Creating PySAM weather data file: {}".format(fname))

        # ------- Process metadata
        m = pd.DataFrame(meta).T
        m = m.rename(
            {
                "latitude": "Latitude",
                "longitude": "Longitude",
                "timezone": "Time Zone",
            },
            axis=1,
        )

        m[["Latitude", "Longitude", "Time Zone"]].to_csv(
            fname, index=False, mode="w"
        )

        # --------- Process data, blank for geothermal
        time_index = resource.index
        mask = (time_index.month == 2) & (time_index.day == 29)
        time_index = time_index[~mask]

        df = pd.DataFrame(index=time_index)
        df["Year"] = time_index.year
        df["Month"] = time_index.month
        df["Day"] = time_index.day
        df["Hour"] = time_index.hour
        df["Minute"] = time_index.minute
        df.to_csv(fname, index=False, mode="a")

        return fname

    def run_gen_and_econ(self):
        """Run SAM generation and possibility follow-on econ analysis."""
        try:
            super().run_gen_and_econ()
        except SAMExecutionError as e:
            logger.error(
                "Skipping site {}; received sam error: {}".format(
                    self._site, str(e)
                )
            )
            self.outputs = {}


class AbstractSamWind(AbstractSamGeneration, PowerCurveLossesMixin, ABC):
    """AbstractSamWind"""

    # pylint: disable=line-too-long
    def __init__(self, *args, **kwargs):
        """Wind generation from SAM.

        See the PySAM :py:class:`~PySAM.Windpower.Windpower`
        documentation for the configuration keys required in the
        `sam_sys_inputs` config. You may also include the following
        ``reV``-specific keys:

            - ``reV_power_curve_losses`` : A dictionary that can be used
              to initialize
              :class:`~reV.losses.power_curve.PowerCurveLossesInput`.
              For example::

                    reV_power_curve_losses = {
                        'target_losses_percent': 9.8,
                        'transformation': 'exponential_stretching'
                    }

              See the description of the class mentioned above or the
              `reV losses demo notebook <https://tinyurl.com/4d7uutt3/>`_
              for detailed instructions on how to specify this input.
            - ``reV_outages`` : Specification for ``reV``-scheduled
              stochastic outage losses. For example::

                    outage_info = [
                        {
                            'count': 6,
                            'duration': 24,
                            'percentage_of_capacity_lost': 100,
                            'allowed_months': ['January', 'March'],
                            'allow_outage_overlap': True
                        },
                        {
                            'count': 10,
                            'duration': 1,
                            'percentage_of_capacity_lost': 10,
                            'allowed_months': ['January'],
                            'allow_outage_overlap': False
                        },
                        ...
                    ]

              See the description of
              :meth:`~reV.losses.scheduled.ScheduledLossesMixin.add_scheduled_losses`
              or the
              `reV losses demo notebook <https://tinyurl.com/4d7uutt3/>`_
              for detailed instructions on how to specify this input.
            - ``reV_outages_seed`` : Integer value used to seed the RNG
              used to compute stochastic outage losses.
            - ``time_index_step`` : Integer representing the step size
              used to sample the ``time_index`` in the resource data.
              This can be used to reduce temporal resolution (i.e. for
              30 minute input data, ``time_index_step=1`` yields the
              full 30 minute time series as output, while
              ``time_index_step=2`` yields hourly output, and so forth).

              .. Note:: The reduced data shape (i.e. after applying a
                        step size of `time_index_step`) must still be
                        an integer multiple of 8760, or the execution
                        will fail.

        Parameters
        ----------
        resource : pd.DataFrame
            Timeseries solar or wind resource data for a single location with a
            pandas DatetimeIndex.  There must be columns for all the required
            variables to run the respective SAM simulation. Remapping will be
            done to convert typical NSRDB/WTK names into SAM names (e.g. DNI ->
            dn and wind_speed -> windspeed)
        meta : pd.DataFrame | pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude, elevation,
            and timezone.
        sam_sys_inputs : dict
            Site-agnostic SAM system model inputs arguments.
        site_sys_inputs : dict
            Optional set of site-specific SAM system inputs to complement the
            site-agnostic inputs.
        output_request : list
            Requested SAM outputs (e.g., 'cf_mean', 'annual_energy',
            'cf_profile', 'gen_profile', 'energy_yield', 'ppa_price',
            'lcoe_fcr').
        drop_leap : bool
            Drops February 29th from the resource data. If False, December
            31st is dropped from leap years.
        """
        super().__init__(*args, **kwargs)
        self.add_power_curve_losses()


class WindPower(AbstractSamWind):
    """Class for Wind generation from SAM"""

    MODULE = "windpower"
    PYSAM = PySamWindPower

    def set_resource_data(self, resource, meta):
        """Set WTK resource data arrays.

        Parameters
        ----------
        resource : pd.DataFrame
            Timeseries solar or wind resource data for a single location with a
            pandas DatetimeIndex.  There must be columns for all the required
            variables to run the respective SAM simulation. Remapping will be
            done to convert typical NSRDB/WTK names into SAM names (e.g. DNI ->
            dn and wind_speed -> windspeed)
        meta : pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude, elevation,
            and timezone.
        """

        meta = self._parse_meta(meta)

        # map resource data names to SAM required data names
        var_map = {
            "speed": "windspeed",
            "direction": "winddirection",
            "airtemperature": "temperature",
            "temp": "temperature",
            "surfacepressure": "pressure",
            "relativehumidity": "rh",
            "humidity": "rh",
        }
        lower_case = {
            k: k.lower().replace(" ", "").replace("_", "")
            for k in resource.columns
        }
        resource = resource.rename(mapper=lower_case, axis="columns")
        resource = resource.rename(mapper=var_map, axis="columns")

        data_dict = {}
        var_list = ["temperature", "pressure", "windspeed", "winddirection"]
        if "winddirection" not in resource:
            resource["winddirection"] = 0.0

        time_index = resource.index
        self.time_interval = self.get_time_interval(resource.index.values)

        data_dict["fields"] = [1, 2, 3, 4]
        data_dict["heights"] = 4 * [self.sam_sys_inputs["wind_turbine_hub_ht"]]

        if "rh" in resource:
            # set relative humidity for icing.
            rh = self.ensure_res_len(resource["rh"].values, time_index)
            n_roll = int(meta[ResourceMetaField.TIMEZONE] * self.time_interval)
            rh = np.roll(rh, n_roll, axis=0)
            data_dict["rh"] = rh.tolist()

        # must be set as matrix in [temperature, pres, speed, direction] order
        # ensure that resource array length is multiple of 8760
        # roll the truncated resource array to local timezone
        temp = self.ensure_res_len(resource[var_list].values, time_index)
        n_roll = int(meta[ResourceMetaField.TIMEZONE] * self.time_interval)
        temp = np.roll(temp, n_roll, axis=0)
        data_dict["data"] = temp.tolist()

        data_dict["lat"] = float(meta[ResourceMetaField.LATITUDE])
        data_dict["lon"] = float(meta[ResourceMetaField.LONGITUDE])
        data_dict["tz"] = int(meta[ResourceMetaField.TIMEZONE])
        data_dict["elev"] = float(meta[ResourceMetaField.ELEVATION])

        time_index = self.ensure_res_len(time_index, time_index)
        data_dict["minute"] = time_index.minute.tolist()
        data_dict["hour"] = time_index.hour.tolist()
        data_dict["year"] = time_index.year.tolist()
        data_dict["month"] = time_index.month.tolist()
        data_dict["day"] = time_index.day.tolist()

        # add resource data to self.data and clear
        self["wind_resource_data"] = data_dict
        self["wind_resource_model_choice"] = 0

    @staticmethod
    def default():
        """Get the executed default pysam WindPower object.

        Returns
        -------
        PySAM.Windpower
        """
        return DefaultWindPower.default()


# pylint: disable=too-many-ancestors
class WindPowerPD(AbstractSamGeneration, PowerCurveLossesMixin):
    """WindPower analysis with wind speed/direction joint probabilty
    distrubtion input"""

    MODULE = "windpower"
    PYSAM = PySamWindPower

    def __init__(
        self,
        ws_edges,
        wd_edges,
        wind_dist,
        meta,
        sam_sys_inputs,
        site_sys_inputs=None,
        output_request=None,
    ):
        """Initialize a SAM generation object for windpower with a
        speed/direction joint probability distribution.

        Parameters
        ----------
        ws_edges : np.ndarray
            1D array of windspeed (m/s) values that set the bin edges for the
            wind probability distribution. Same len as wind_dist.shape[0] + 1
        wd_edges : np.ndarray
            1D array of winddirections (deg) values that set the bin edges
            for the wind probability dist. Same len as wind_dist.shape[1] + 1
        wind_dist : np.ndarray
            2D array probability distribution of (windspeed, winddirection).
        meta : pd.DataFrame | pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude, elevation,
            and timezone.
        sam_sys_inputs : dict
            Site-agnostic SAM system model inputs arguments.
        site_sys_inputs : dict
            Optional set of site-specific SAM system inputs to complement the
            site-agnostic inputs.
        output_request : list
            Requested SAM outputs (e.g., 'cf_mean', 'annual_energy',
            'cf_profile', 'gen_profile', 'energy_yield', 'ppa_price',
            'lcoe_fcr').
        """

        # make sure timezone and elevation are in the meta data
        meta = self.tz_elev_check(sam_sys_inputs, site_sys_inputs, meta)

        # don't pass resource to base class,
        # set in concrete generation classes instead
        super().__init__(
            None,
            meta,
            sam_sys_inputs,
            site_sys_inputs=site_sys_inputs,
            output_request=output_request,
            drop_leap=False,
        )

        # Set the site number using meta data
        if hasattr(meta, "name"):
            self._site = meta.name
        else:
            self._site = None

        self.set_resource_data(ws_edges, wd_edges, wind_dist)
        self.add_power_curve_losses()

    def set_resource_data(self, ws_edges, wd_edges, wind_dist):
        """Send wind PD to pysam

        Parameters
        ----------
        ws_edges : np.ndarray
            1D array of windspeed (m/s) values that set the bin edges for the
            wind probability distribution. Same len as wind_dist.shape[0] + 1
        wd_edges : np.ndarray
            1D array of winddirections (deg) values that set the bin edges
            for the wind probability dist. Same len as wind_dist.shape[1] + 1
        wind_dist : np.ndarray
            2D array probability distribution of (windspeed, winddirection).
        """

        assert len(ws_edges) == wind_dist.shape[0] + 1
        assert len(wd_edges) == wind_dist.shape[1] + 1

        wind_dist /= wind_dist.sum()

        # SAM wants the midpoints of the sample bins
        ws_points = ws_edges[:-1] + np.diff(ws_edges) / 2
        wd_points = wd_edges[:-1] + np.diff(wd_edges) / 2

        wd_points, ws_points = np.meshgrid(wd_points, ws_points)
        vstack = (
            ws_points.flatten(),
            wd_points.flatten(),
            wind_dist.flatten(),
        )
        wrd = np.vstack(vstack).T.tolist()

        self["wind_resource_model_choice"] = 2
        self["wind_resource_distribution"] = wrd


class MhkWave(AbstractSamGeneration):
    """Class for Wave generation from SAM"""

    MODULE = "mhkwave"
    PYSAM = PySamMhkWave

    def set_resource_data(self, resource, meta):
        """Set Hindcast US Wave resource data arrays.

        Parameters
        ----------
        resource : pd.DataFrame
            Timeseries resource data for a single location with a
            pandas DatetimeIndex. There must be columns for all the required
            variables to run the respective SAM simulation.
        meta : pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude, elevation,
            and timezone.
        """

        meta = self._parse_meta(meta)

        # map resource data names to SAM required data names
        var_map = {
            "significantwaveheight": "significant_wave_height",
            "waveheight": "significant_wave_height",
            "height": "significant_wave_height",
            "swh": "significant_wave_height",
            "energyperiod": "energy_period",
            "waveperiod": "energy_period",
            "period": "energy_period",
            "ep": "energy_period",
        }
        lower_case = {
            k: k.lower().replace(" ", "").replace("_", "")
            for k in resource.columns
        }
        resource = resource.rename(mapper=lower_case, axis="columns")
        resource = resource.rename(mapper=var_map, axis="columns")

        data_dict = {}

        time_index = resource.index
        self.time_interval = self.get_time_interval(resource.index.values)

        # must be set as matrix in [temperature, pres, speed, direction] order
        # ensure that resource array length is multiple of 8760
        # roll the truncated resource array to local timezone
        for var in ["significant_wave_height", "energy_period"]:
            arr = self.ensure_res_len(resource[var].values, time_index)
            n_roll = int(meta[ResourceMetaField.TIMEZONE] * self.time_interval)
            data_dict[var] = np.roll(arr, n_roll, axis=0).tolist()

        data_dict["lat"] = meta[ResourceMetaField.LATITUDE]
        data_dict["lon"] = meta[ResourceMetaField.LONGITUDE]
        data_dict["tz"] = meta[ResourceMetaField.TIMEZONE]

        time_index = self.ensure_res_len(time_index, time_index)
        data_dict["minute"] = time_index.minute
        data_dict["hour"] = time_index.hour
        data_dict["year"] = time_index.year
        data_dict["month"] = time_index.month
        data_dict["day"] = time_index.day

        # add resource data to self.data and clear
        self["wave_resource_data"] = data_dict

    @staticmethod
    def default():
        """Get the executed default PySAM MhkWave object.

        Returns
        -------
        PySAM.MhkWave
        """
        return DefaultMhkWave.default()
