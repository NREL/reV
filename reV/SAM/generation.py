# -*- coding: utf-8 -*-
"""reV-to-SAM generation interface module.

Wraps the NREL-PySAM pvwattsv5, windpower, and tcsmolensalt modules with
additional reV features.
"""
from abc import ABC, abstractmethod
import copy
import os
import logging
import numpy as np
import pandas as pd
from warnings import warn
import PySAM.Pvwattsv5 as PySamPv5
import PySAM.Pvwattsv7 as PySamPv7
import PySAM.Pvsamv1 as PySamDetailedPv
import PySAM.Windpower as PySamWindPower
import PySAM.TcsmoltenSalt as PySamCSP
import PySAM.Swh as PySamSwh
import PySAM.TroughPhysicalProcessHeat as PySamTpph
import PySAM.LinearFresnelDsgIph as PySamLds

from reV.SAM.defaults import (DefaultPvWattsv5,
                              DefaultPvWattsv7,
                              DefaultPvSamv1,
                              DefaultWindPower,
                              DefaultTcsMoltenSalt,
                              DefaultSwh,
                              DefaultTroughPhysicalProcessHeat,
                              DefaultLinearFresnelDsgIph)
from reV.utilities.exceptions import SAMInputWarning, SAMExecutionError
from reV.utilities.curtailment import curtail
from reV.SAM.SAM import RevPySam
from reV.SAM.econ import LCOE, SingleOwner

logger = logging.getLogger(__name__)


class AbstractSamGeneration(RevPySam, ABC):
    """Base class for SAM generation simulations."""

    def __init__(self, resource, meta, sam_sys_inputs, site_sys_inputs=None,
                 output_request=None, drop_leap=False):
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
        super().__init__(meta, sam_sys_inputs, output_request,
                         site_sys_inputs=site_sys_inputs)

        # Set the site number using resource
        if hasattr(resource, 'name'):
            self._site = resource.name
        else:
            self._site = None

        self.set_resource_data(resource, meta)

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
        for req in out_req_cleaned:
            if req in res_df:
                res_reqs.append(req)
                if res_out is None:
                    res_out = {}
                res_out[req] = cls.ensure_res_len(res_df[req].values)

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

        if 'ws_mean' in out_req_nomeans:
            out_req_nomeans.remove('ws_mean')
            res_mean = {}
            res_mean['ws_mean'] = resource['mean_windspeed', idx]

        else:
            for var in ('dni', 'dhi', 'ghi'):
                label_1 = '{}_mean'.format(var)
                label_2 = 'mean_{}'.format(var)
                if label_1 in out_req_nomeans:
                    out_req_nomeans.remove(label_1)
                    if res_mean is None:
                        res_mean = {}
                    res_mean[label_1] = resource[label_2, idx] / 1000 * 24

        return res_mean, out_req_nomeans

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
        meta : pd.DataFrame
            1D table with resource meta data.

        Returns
        -------
        meta : pd.DataFrame
            1D table with resource meta data. Will include "timezone"
            and "elevation" from the sam and site system inputs if found.
        """

        if meta is not None:
            if sam_sys_inputs is not None:
                if 'elevation' in sam_sys_inputs:
                    meta['elevation'] = sam_sys_inputs['elevation']
                if 'timezone' in sam_sys_inputs:
                    meta['timezone'] = int(sam_sys_inputs['timezone'])

            # site-specific inputs take priority over generic system inputs
            if site_sys_inputs is not None:
                if 'elevation' in site_sys_inputs:
                    meta['elevation'] = site_sys_inputs['elevation']
                if 'timezone' in site_sys_inputs:
                    meta['timezone'] = int(site_sys_inputs['timezone'])

            if 'timezone' not in meta:
                msg = ('Need timezone input to run SAM gen. Not found in '
                       'resource meta or technology json input config.')
                raise SAMExecutionError(msg)

        return meta

    @property
    def has_timezone(self):
        """ Returns true if instance has a timezone set """
        if self._meta is not None:
            if 'timezone' in self.meta:
                return True

        return False

    def cf_mean(self):
        """Get mean capacity factor (fractional) from SAM.

        Returns
        -------
        output : float
            Mean capacity factor (fractional).
        """
        return self['capacity_factor'] / 100

    def cf_profile(self):
        """Get hourly capacity factor (frac) profile in orig timezone.

        Returns
        -------
        cf_profile : np.ndarray
            1D numpy array of capacity factor profile.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return self.gen_profile() / self.sam_sys_inputs['system_capacity']

    def annual_energy(self):
        """Get annual energy generation value in kWh from SAM.

        Returns
        -------
        output : float
            Annual energy generation (kWh).
        """
        return self['annual_energy']

    def energy_yield(self):
        """Get annual energy yield value in kwh/kw from SAM.

        Returns
        -------
        output : float
            Annual energy yield (kwh/kw).
        """
        return self['kwh_per_kw']

    def gen_profile(self):
        """Get power generation profile (orig timezone) in kW.

        Returns
        -------
        output : np.ndarray
            1D array of hourly power generation in kW.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return np.array(self['gen'], dtype=np.float32)

    def collect_outputs(self, output_lookup=None):
        """
        Collect SAM gen output_request. Rolls outputs to UTC if appropriate.

        Parameters
        ----------
        output_lookup : dict | None
            Lookup dictionary mapping output keys to special output methods.
            None defaults to generation default outputs.
        """

        if output_lookup is None:
            output_lookup = {'cf_mean': self.cf_mean,
                             'cf_profile': self.cf_profile,
                             'annual_energy': self.annual_energy,
                             'energy_yield': self.energy_yield,
                             'gen_profile': self.gen_profile,
                             }

        super().collect_outputs(output_lookup=output_lookup)

    def run_gen_and_econ(self):
        """Run SAM generation with possibility for follow on econ analysis."""

        lcoe_out_reqs = None
        so_out_reqs = None
        lcoe_vars = ('lcoe_fcr', 'fixed_charge_rate', 'capital_cost',
                     'fixed_operating_cost', 'variable_operating_cost')
        so_vars = ('ppa_price', 'lcoe_real', 'lcoe_nom',
                   'project_return_aftertax_npv', 'flip_actual_irr',
                   'gross_revenue')
        if 'lcoe_fcr' in self.output_request:
            lcoe_out_reqs = [r for r in self.output_request if r in lcoe_vars]
            self.output_request = [r for r in self.output_request
                                   if r not in lcoe_out_reqs]
        elif any(x in self.output_request for x in so_vars):
            so_out_reqs = [r for r in self.output_request if r in so_vars]
            self.output_request = [r for r in self.output_request
                                   if r not in so_out_reqs]

        # Execute the SAM generation compute module (pvwattsv7, windpower, etc)
        self.run()

        # Execute a follow-on SAM econ compute module
        # (lcoe_fcr, singleowner, etc)
        if lcoe_out_reqs is not None:
            self.sam_sys_inputs['annual_energy'] = self.annual_energy()
            lcoe = LCOE(self.sam_sys_inputs, output_request=lcoe_out_reqs)
            lcoe.assign_inputs()
            lcoe.execute()
            lcoe.collect_outputs()
            lcoe.outputs_to_utc_arr()
            self.outputs.update(lcoe.outputs)

        elif so_out_reqs is not None:
            self.sam_sys_inputs['gen'] = self.gen_profile()
            so = SingleOwner(self.sam_sys_inputs, output_request=so_out_reqs)
            so.assign_inputs()
            so.execute()
            so.collect_outputs()
            so.outputs_to_utc_arr()
            self.outputs.update(so.outputs)

    def run(self):
        """Run a reV-SAM generation object by assigning inputs, executing the
        SAM simulation, collecting outputs, and converting all arrays to UTC.
        """
        self.assign_inputs()
        self.execute()
        self.collect_outputs()
        self.outputs_to_utc_arr()

    @classmethod
    def reV_run(cls, points_control, res_file, site_df,
                output_request=('cf_mean',), drop_leap=False,
                gid_map=None):
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

        Returns
        -------
        out : dict
            Nested dictionaries where the top level key is the site index,
            the second level key is the variable name, second level value is
            the output variable value.
        """
        # initialize output dictionary
        out = {}

        # Get the RevPySam resource object
        resources = RevPySam.get_sam_res(res_file,
                                         points_control.project_points,
                                         points_control.project_points.tech,
                                         output_request=output_request,
                                         gid_map=gid_map)

        # run resource through curtailment filter if applicable
        curtailment = points_control.project_points.curtailment
        if curtailment is not None:
            resources = curtail(resources, curtailment,
                                random_seed=curtailment.random_seed)

        # iterate through project_points gen_gid values
        for gen_gid in points_control.project_points.sites:

            # Lookup the resource gid if there's a mapping and get the resource
            # data from the SAMResource object using the res_gid.
            res_gid = gen_gid if gid_map is None else gid_map[gen_gid]
            site_res_df, site_meta = resources._get_res_df(res_gid)

            # drop the leap day
            if drop_leap:
                site_res_df = cls.drop_leap(site_res_df)

            _, inputs = points_control.project_points[gen_gid]

            # get resource data pass-throughs and resource means
            res_outs, out_req_cleaned = cls._get_res(site_res_df,
                                                     output_request)
            res_mean, out_req_cleaned = cls._get_res_mean(resources, res_gid,
                                                          out_req_cleaned)

            # iterate through requested sites.
            sim = cls(resource=site_res_df, meta=site_meta,
                      sam_sys_inputs=inputs, output_request=out_req_cleaned,
                      site_sys_inputs=dict(site_df.loc[gen_gid, :]))
            sim.run_gen_and_econ()

            # collect outputs to dictout
            out[gen_gid] = sim.outputs

            if res_outs is not None:
                out[gen_gid].update(res_outs)

            if res_mean is not None:
                out[gen_gid].update(res_mean)

        return out


class AbstractSamSolar(AbstractSamGeneration, ABC):
    """Base Class for Solar generation from SAM"""

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
        meta : pd.DataFrame | pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude, elevation,
            and timezone.
        """

        time_index = resource.index
        self.time_interval = self.get_time_interval(resource.index.values)

        # map resource data names to SAM required data names
        var_map = {'dni': 'dn',
                   'dhi': 'df',
                   'ghi': 'gh',
                   'clearskydni': 'dn',
                   'clearskydhi': 'df',
                   'clearskyghi': 'gh',
                   'windspeed': 'wspd',
                   'airtemperature': 'tdry',
                   'temperature': 'tdry',
                   'temp': 'tdry',
                   'dewpoint': 'tdew',
                   'surfacepressure': 'pres',
                   'pressure': 'pres',
                   'surfacealbedo': 'albedo',
                   }
        lower_case = {k: k.lower().replace(' ', '').replace('_', '')
                      for k in resource.columns}
        irrad_vars = ['dn', 'df', 'gh']

        resource = resource.rename(mapper=lower_case, axis='columns')
        resource = resource.rename(mapper=var_map, axis='columns')
        resource = {k: np.array(v) for (k, v) in
                    resource.to_dict(orient='list').items()}

        # set resource variables
        for var, arr in resource.items():
            if var != 'time_index':

                # ensure that resource array length is multiple of 8760
                arr = np.roll(
                    self.ensure_res_len(arr),
                    int(self._meta['timezone'] * self.time_interval))

                if var in irrad_vars:
                    if np.min(arr) < 0:
                        warn('Solar irradiance variable "{}" has a minimum '
                             'value of {}. Truncating to zero.'
                             .format(var, np.min(arr)), SAMInputWarning)
                        arr = np.where(arr < 0, 0, arr)

                resource[var] = arr.tolist()

        resource['lat'] = meta['latitude']
        resource['lon'] = meta['longitude']
        resource['tz'] = meta['timezone']

        if 'elevation' in meta:
            resource['elev'] = meta['elevation']
        else:
            resource['elev'] = 0.0

        ti_8760 = self.ensure_res_len(time_index)
        resource['minute'] = ti_8760.minute
        resource['hour'] = ti_8760.hour
        resource['year'] = ti_8760.year
        resource['month'] = ti_8760.month
        resource['day'] = ti_8760.day

        if 'albedo' in resource:
            self['albedo'] = resource.pop('albedo')

        self['solar_resource_data'] = resource


class AbstractSamPv(AbstractSamSolar, ABC):
    """Photovoltaic (PV) generation with either pvwatts of detailed pv.
    """

    # set these class attrs in concrete subclasses
    MODULE = None
    PYSAM = None

    def __init__(self, resource, meta, sam_sys_inputs, site_sys_inputs=None,
                 output_request=None, drop_leap=False):
        """Initialize a SAM solar object.

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
        sam_sys_inputs = self.set_latitude_tilt_az(sam_sys_inputs, meta)

        super().__init__(resource, meta, sam_sys_inputs,
                         site_sys_inputs=site_sys_inputs,
                         output_request=output_request,
                         drop_leap=drop_leap)

    @staticmethod
    def set_latitude_tilt_az(sam_sys_inputs, meta):
        """Check if tilt is specified as latitude and set tilt=lat, az=180 or 0

        Parameters
        ----------
        sam_sys_inputs : dict
            Site-agnostic SAM system model inputs arguments.
        meta : pd.DataFrame
            1D table with resource meta data.

        Returns
        -------
        sam_sys_inputs : dict
            Site-agnostic SAM system model inputs arguments.
            If for a pv simulation the "tilt" parameter was originally not
            present or set to 'lat' or 'latitude', the tilt will be set to
            the absolute value of the latitude found in meta and the azimuth
            will be 180 if lat>0, 0 if lat<0.
        """

        set_tilt = False
        if sam_sys_inputs is not None and meta is not None:
            if 'tilt' not in sam_sys_inputs:
                warn('No tilt specified, setting at latitude.',
                     SAMInputWarning)
                set_tilt = True
            else:
                if (sam_sys_inputs['tilt'] == 'lat'
                        or sam_sys_inputs['tilt'] == 'latitude'):
                    set_tilt = True

        if set_tilt:
            # set tilt to abs(latitude)
            sam_sys_inputs['tilt'] = np.abs(meta['latitude'])
            if meta['latitude'] > 0:
                # above the equator, az = 180
                sam_sys_inputs['azimuth'] = 180
            else:
                # below the equator, az = 0
                sam_sys_inputs['azimuth'] = 0

            logger.debug('Tilt specified at "latitude", setting tilt to: {}, '
                         'azimuth to: {}'
                         .format(sam_sys_inputs['tilt'],
                                 sam_sys_inputs['azimuth']))
        return sam_sys_inputs

    def cf_mean(self):
        """Get mean capacity factor (fractional) from SAM.

        NOTE: PV capacity factor is the AC power production / the DC nameplate

        Returns
        -------
        output : float
            Mean capacity factor (fractional).
            PV CF is calculated as AC power / DC nameplate.
        """
        return self['capacity_factor'] / 100

    def cf_profile(self):
        """Get hourly capacity factor (frac) profile in orig timezone.

        NOTE: PV capacity factor is the AC power production / the DC nameplate

        Returns
        -------
        cf_profile : np.ndarray
            1D numpy array of capacity factor profile.
            Datatype is float32 and array length is 8760*time_interval.
            PV CF is calculated as AC power / DC nameplate.
        """
        return self.gen_profile() / self.sam_sys_inputs['system_capacity']

    def gen_profile(self):
        """Get AC inverter power generation profile (orig timezone) in kW.
        This is an alias of the "ac" SAM output variable.

        Returns
        -------
        output : np.ndarray
            1D array of AC inverter power generation in kW.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return self.ac()

    def ac(self):
        """Get AC inverter power generation profile (orig timezone) in kW.

        Returns
        -------
        output : np.ndarray
            1D array of AC inverter power generation in kW.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return np.array(self['ac'], dtype=np.float32) / 1000

    def dc(self):
        """
        Get DC array power generation profile (orig timezone) in kW.

        Returns
        -------
        output : np.ndarray
            1D array of DC array power generation in kW.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return np.array(self['dc'], dtype=np.float32) / 1000

    def clipped_power(self):
        """
        Get the clipped DC power generated behind the inverter
        (orig timezone) in kW.

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
        """Collect SAM gen output_request.

        Parameters
        ----------
        output_lookup : dict | None
            Lookup dictionary mapping output keys to special output methods.
            None defaults to generation default outputs.
        """

        if output_lookup is None:
            output_lookup = {'cf_mean': self.cf_mean,
                             'cf_profile': self.cf_profile,
                             'annual_energy': self.annual_energy,
                             'energy_yield': self.energy_yield,
                             'gen_profile': self.gen_profile,
                             'ac': self.ac,
                             'dc': self.dc,
                             'clipped_power': self.clipped_power
                             }

        super().collect_outputs(output_lookup=output_lookup)


class PvWattsv5(AbstractSamPv):
    """Photovoltaic (PV) generation with pvwattsv5.
    """
    MODULE = 'pvwattsv5'
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
    """Photovoltaic (PV) generation with pvwattsv7.
    """
    MODULE = 'pvwattsv7'
    PYSAM = PySamPv7

    @staticmethod
    def default():
        """Get the executed default pysam PVWATTSV7 object.

        Returns
        -------
        PySAM.Pvwattsv7
        """
        return DefaultPvWattsv7.default()


class PvSamv1(AbstractSamPv):
    """Detailed PV model"""

    MODULE = 'Pvsamv1'
    PYSAM = PySamDetailedPv

    def ac(self):
        """Get AC inverter power generation profile (orig timezone) in kW.

        Returns
        -------
        output : np.ndarray
            1D array of AC inverter power generation in kW.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return np.array(self['gen'], dtype=np.float32)

    def dc(self):
        """
        Get DC array power generation profile (orig timezone) in kW.

        Returns
        -------
        output : np.ndarray
            1D array of DC array power generation in kW.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return np.array(self['dc_net'], dtype=np.float32)

    @staticmethod
    def default():
        """Get the executed default pysam Pvsamv1 object.

        Returns
        -------
        PySAM.Pvsamv1
        """
        return DefaultPvSamv1.default()


class TcsMoltenSalt(AbstractSamSolar):
    """Concentrated Solar Power (CSP) generation with tower molten salt
    """
    MODULE = 'tcsmolten_salt'
    PYSAM = PySamCSP

    def cf_profile(self):
        """Get absolute value hourly capacity factor (frac) profile in
        orig timezone.

        Returns
        -------
        cf_profile : np.ndarray
            1D numpy array of capacity factor profile.
            Datatype is float32 and array length is 8760*time_interval.
        """
        x = np.abs(self.gen_profile() / self.sam_sys_inputs['system_capacity'])
        return x

    @staticmethod
    def default():
        """Get the executed default pysam CSP object.

        Returns
        -------
        PySAM.TcsmoltenSalt
        """
        return DefaultTcsMoltenSalt.default()


class AbstractSamSolarThermal(AbstractSamSolar, ABC):
    """Base class for solar thermal """
    PYSAM_WEATHER_TAG = None

    def set_resource_data(self, resource, meta):
        """
        Set NSRDB resource file. Overloads Solar.set_resource_data(). Solar
        thermal PySAM models require a data file, not raw data.

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
        """
        self.time_interval = self.get_time_interval(resource.index.values)
        pysam_w_fname = self._create_pysam_wfile(resource, meta)
        # pylint: disable=E1101
        self[self.PYSAM_WEATHER_TAG] = pysam_w_fname

    def _create_pysam_wfile(self, resource, meta):
        """
        Create PySAM weather input file. PySAM will not accept data on Feb
        29th. For leap years, December 31st is dropped and time steps are
        shifted to relabel Feb 29th as March 1st, March 1st as March 2nd, etc.

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

        Returns
        -------
        fname : string
            Name of weather csv file
        """
        fname = '{}_weather.csv'.format(self._site)
        logger.debug('Creating PySAM weather data file: {}'.format(fname))

        # ------- Process metadata
        m = pd.DataFrame(meta).T
        timezone = m.timezone
        m['Source'] = 'NSRDB'
        m['Location ID'] = meta.name
        m['City'] = '-'
        m['State'] = m.state.apply(lambda x: '-' if x == 'None' else x)
        m['Country'] = m.country.apply(lambda x: '-' if x == 'None' else x)
        m['Latitude'] = m.latitude
        m['Longitude'] = m.longitude
        m['Time Zone'] = m.timezone
        m['Elevation'] = m.elevation
        m['Local Time Zone'] = m.timezone
        m['Dew Point Units'] = 'c'
        m['DHI Units'] = 'w/m2'
        m['DNI Units'] = 'w/m2'
        m['Temperature Units'] = 'c'
        m['Pressure Units'] = 'mbar'
        m['Wind Speed'] = 'm/s'
        m = m.drop(['elevation', 'timezone', 'country', 'state', 'county',
                    'urban', 'population', 'landcover', 'latitude',
                    'longitude'], axis=1)
        m.to_csv(fname, index=False, mode='w')

        # --------- Process data
        # Adjust from UTC to local time
        local = np.roll(resource.values, int(timezone * self.time_interval),
                        axis=0)
        res = pd.DataFrame(local, columns=resource.columns,
                           index=resource.index)
        leap_mask = (res.index.month == 2) & (res.index.day == 29)
        no_leap_index = res.index[~leap_mask]
        df = pd.DataFrame(index=no_leap_index)
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['Hour'] = df.index.hour
        df['Minute'] = df.index.minute
        df['DNI'] = self.ensure_res_len(res.dni.values)
        df['DHI'] = self.ensure_res_len(res.dhi.values)
        df['Wind Speed'] = self.ensure_res_len(res.wind_speed.values)
        df['Temperature'] = self.ensure_res_len(res.air_temperature.values)
        df['Dew Point'] = self.ensure_res_len(res.dew_point.values)
        df['Pressure'] = self.ensure_res_len(res.surface_pressure.values)
        df.to_csv(fname, index=False, mode='a')

        return fname

    def run_gen_and_econ(self, delete_wfile=True):
        """
        Run SAM generation with possibility for follow on econ analysis.

        Parameters
        ----------
        delete_wfile : bool
            Delete PySAM weather file after processing is complete
        """
        super().run_gen_and_econ()

        pysam_w_fname = self[self.PYSAM_WEATHER_TAG]
        if delete_wfile and os.path.exists(pysam_w_fname):
            os.remove(pysam_w_fname)


class SolarWaterHeat(AbstractSamSolarThermal):
    """
    Solar Water Heater generation
    """
    MODULE = 'solarwaterheat'
    PYSAM = PySamSwh
    PYSAM_WEATHER_TAG = 'solar_resource_file'

    @staticmethod
    def default():
        """Get the executed default pysam swh object.

        Returns
        -------
        PySAM.Swh
        """
        return DefaultSwh.default()


class LinearDirectSteam(AbstractSamSolarThermal):
    """
    Process heat linear Fresnel direct steam generation
    """
    MODULE = 'lineardirectsteam'
    PYSAM = PySamLds
    PYSAM_WEATHER_TAG = 'file_name'

    def cf_mean(self):
        """Calculate mean capacity factor (fractional) from SAM.

        Returns
        -------
        output : float
            Mean capacity factor (fractional).
        """
        net_power = self['annual_field_energy'] \
            - self['annual_thermal_consumption']  # kW-hr
        # q_pb_des is in MW, convert to kW-hr
        name_plate = self['q_pb_des'] * 8760 * 1000

        return net_power / name_plate

    @staticmethod
    def default():
        """Get the executed default pysam linear Fresnel object.

        Returns
        -------
        PySAM.LinearFresnelDsgIph
        """
        return DefaultLinearFresnelDsgIph.default()


class TroughPhysicalHeat(AbstractSamSolarThermal):
    """
    Trough Physical Process Heat generation
    """
    MODULE = 'troughphysicalheat'
    PYSAM = PySamTpph
    PYSAM_WEATHER_TAG = 'file_name'

    def cf_mean(self):
        """Calculate mean capacity factor (fractional) from SAM.

        Returns
        -------
        output : float
            Mean capacity factor (fractional).
        """
        net_power = self['annual_gross_energy'] \
            - self['annual_thermal_consumption']  # kW-hr
        # q_pb_des is in MW, convert to kW-hr
        name_plate = self['q_pb_design'] * 8760 * 1000

        return net_power / name_plate

    @staticmethod
    def default():
        """Get the executed default pysam trough object.

        Returns
        -------
        PySAM.TroughPhysicalProcessHeat
        """
        return DefaultTroughPhysicalProcessHeat.default()


class WindPower(AbstractSamGeneration):
    """Class for Wind generation from SAM
    """
    MODULE = 'windpower'
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
        meta : pd.DataFrame | pd.Series
            Meta data corresponding to the resource input for the single
            location. Should include values for latitude, longitude, elevation,
            and timezone.
        """

        # map resource data names to SAM required data names
        var_map = {'speed': 'windspeed',
                   'direction': 'winddirection',
                   'airtemperature': 'temperature',
                   'temp': 'temperature',
                   'surfacepressure': 'pressure',
                   'relativehumidity': 'rh',
                   'humidity': 'rh',
                   }
        lower_case = {k: k.lower().replace(' ', '').replace('_', '')
                      for k in resource.columns}
        resource = resource.rename(mapper=lower_case, axis='columns')
        resource = resource.rename(mapper=var_map, axis='columns')

        data_dict = {}
        var_list = ['temperature', 'pressure', 'windspeed', 'winddirection']
        if 'winddirection' not in resource:
            resource['winddirection'] = 0.0

        time_index = resource.index
        self.time_interval = self.get_time_interval(resource.index.values)

        data_dict['fields'] = [1, 2, 3, 4]
        data_dict['heights'] = 4 * [self.sam_sys_inputs['wind_turbine_hub_ht']]

        if 'rh' in resource:
            # set relative humidity for icing.
            rh = np.roll(self.ensure_res_len(resource['rh'].values),
                         int(meta['timezone'] * self.time_interval),
                         axis=0)
            data_dict['rh'] = rh.tolist()

        # must be set as matrix in [temperature, pres, speed, direction] order
        # ensure that resource array length is multiple of 8760
        # roll the truncated resource array to local timezone
        temp = np.roll(self.ensure_res_len(resource[var_list].values),
                       int(meta['timezone'] * self.time_interval), axis=0)
        data_dict['data'] = temp.tolist()

        resource['lat'] = meta['latitude']
        resource['lon'] = meta['longitude']
        resource['tz'] = meta['timezone']
        resource['elev'] = meta['elevation']

        data_dict['minute'] = self.ensure_res_len(time_index.minute)
        data_dict['hour'] = self.ensure_res_len(time_index.hour)
        data_dict['year'] = self.ensure_res_len(time_index.year)
        data_dict['month'] = self.ensure_res_len(time_index.month)
        data_dict['day'] = self.ensure_res_len(time_index.day)

        # add resource data to self.data and clear
        self['wind_resource_data'] = data_dict

    @staticmethod
    def default():
        """Get the executed default pysam WindPower object.

        Returns
        -------
        PySAM.Windpower
        """
        return DefaultWindPower.default()
