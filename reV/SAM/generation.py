# -*- coding: utf-8 -*-
"""reV-to-SAM generation interface module.

Wraps the NREL-PySAM pvwattsv5, windpower, and tcsmolensalt modules with
additional reV features.
"""
from abc import ABC
import copy
import os
import logging
import numpy as np
import pandas as pd
from warnings import warn
import PySAM.Pvwattsv5 as PySamPV5
import PySAM.Pvwattsv7 as PySamPV7
import PySAM.Windpower as PySamWindPower
import PySAM.TcsmoltenSalt as PySamCSP
import PySAM.Swh as PySamSWH
import PySAM.TroughPhysicalProcessHeat as PySamTPPH
import PySAM.LinearFresnelDsgIph as PySamLDS

from reV.SAM.defaults import (DefaultPvwattsv5, DefaultPvwattsv7,
                              DefaultWindPower, DefaultTcsMoltenSalt,
                              DefaultSwh, DefaultTroughPhysicalProcessHeat,
                              DefaultLinearFresnelDsgIph)
from reV.utilities.exceptions import SAMInputWarning, SAMExecutionError
from reV.utilities.curtailment import curtail
from reV.SAM.SAM import RevPySam
from reV.SAM.econ import LCOE, SingleOwner

logger = logging.getLogger(__name__)


class Generation(RevPySam, ABC):
    """Base class for SAM generation simulations."""

    @staticmethod
    def _get_res(res_df, output_request):
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
                res_out[req] = Generation.ensure_res_len(res_df[req].values)
        for req in res_reqs:
            out_req_cleaned.remove(req)

        return res_out, out_req_cleaned

    @staticmethod
    def _get_res_mean(resource, site, output_request):
        """Get the resource annual means (single site).

        Parameters
        ----------
        resource : rex.sam_resource.SAMResource
            SAM resource object for WIND resource
        site : int
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
        idx = resource.sites.index(site)

        if 'ws_mean' in out_req_nomeans:
            out_req_nomeans.remove('ws_mean')
            res_mean = {}
            res_mean['ws_mean'] = resource['mean_windspeed', idx]

        else:
            if 'dni_mean' in out_req_nomeans:
                out_req_nomeans.remove('dni_mean')
                res_mean = {}
                res_mean['dni_mean'] = resource['mean_dni', idx] / 1000 * 24

            if 'ghi_mean' in out_req_nomeans:
                out_req_nomeans.remove('ghi_mean')
                if res_mean is None:
                    res_mean = {}
                res_mean['ghi_mean'] = resource['mean_ghi', idx] / 1000 * 24

        return res_mean, out_req_nomeans

    @staticmethod
    def tz_check(parameters, meta):
        """Check timezone input and use json config tz if not in resource meta.

        Parameters
        ----------
        parameters : dict
            SAM model input parameters.
        meta : pd.DataFrame
            1D table with resource meta data.

        Returns
        -------
        meta : pd.DataFrame
            1D table with resource meta data. If meta was not originally set in
            the resource meta data, but was set as "tz" or "timezone" in the
            SAM model input parameters json file, timezone will be added to
            this instance of meta.
        """

        if meta is not None:
            if 'timezone' not in meta:
                if 'tz' in parameters:
                    meta['timezone'] = int(parameters['tz'])
                elif 'timezone' in parameters:
                    meta['timezone'] = int(parameters['timezone'])
                else:
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
        return self.gen_profile() / self.parameters['system_capacity']

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

    def _gen_exec(self):
        """Run SAM generation with possibility for follow on econ analysis."""

        lcoe_out_req = None
        so_out_req = None
        if 'lcoe_fcr' in self.output_request:
            lcoe_out_req = self.output_request.pop(
                self.output_request.index('lcoe_fcr'))
        else:
            so_reqs = ('ppa_price', 'lcoe_real', 'lcoe_nom')
            reqs = [r for r in so_reqs if r in self.output_request]
            if len(reqs) > 1:
                raise KeyError('Cannot request more than one single owner '
                               'output in Generation module. Found the '
                               'following {} single owner output requests in '
                               'the generation run: {}'
                               .format(len(reqs), reqs))
            elif len(reqs) == 1:
                so_out_req = self.output_request.pop(
                    self.output_request.index(reqs[0]))

        self.assign_inputs()
        self.execute()
        self.collect_outputs()
        self.outputs_to_utc_arr()

        if lcoe_out_req is not None:
            self.parameters['annual_energy'] = self.annual_energy()
            lcoe = LCOE(self.parameters, output_request=(lcoe_out_req,))
            lcoe.assign_inputs()
            lcoe.execute()
            lcoe.collect_outputs()
            lcoe.outputs_to_utc_arr()
            self.outputs.update(lcoe.outputs)

        elif so_out_req is not None:
            self.parameters['gen'] = self.gen_profile()
            so = SingleOwner(self.parameters, output_request=(so_out_req,))
            so.assign_inputs()
            so.execute()
            so.collect_outputs()
            so.outputs_to_utc_arr()
            self.outputs.update(so.outputs)

    @classmethod
    def reV_run(cls, points_control, res_file, output_request=('cf_mean',),
                downscale=None, drop_leap=False):
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
        downscale : NoneType | str
            Option for NSRDB resource downscaling to higher temporal
            resolution. Expects a string in the Pandas frequency format,
            e.g. '5min'.
        drop_leap : bool
            Drops February 29th from the resource data. If False, December
            31st is dropped from leap years.

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
                                         downscale=downscale,
                                         output_request=output_request)

        # run resource through curtailment filter if applicable
        curtailment = points_control.project_points.curtailment
        if curtailment is not None:
            resources = curtail(resources, curtailment,
                                random_seed=curtailment.random_seed)

        # Use resource object iterator
        for res_df, meta in resources:

            # drop the leap day
            if drop_leap:
                res_df = cls.drop_leap(res_df)

            # get SAM inputs from project_points based on the current site
            site = res_df.name
            _, inputs = points_control.project_points[site]

            res_outs, out_req_cleaned = cls._get_res(res_df, output_request)
            res_mean, out_req_cleaned = cls._get_res_mean(resources, site,
                                                          out_req_cleaned)

            # iterate through requested sites.
            sim = cls(resource=res_df, meta=meta, parameters=inputs,
                      output_request=out_req_cleaned)
            sim._gen_exec()

            # collect outputs to dictout
            out[site] = sim.outputs

            if res_outs is not None:
                out[site].update(res_outs)

            if res_mean is not None:
                out[site].update(res_mean)

        return out


class Solar(Generation, ABC):
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
            Drops February 29th from the resource data. If False, December
            31st is dropped from leap years.
        """

        # drop the leap day
        if drop_leap:
            resource = self.drop_leap(resource)

        parameters = self.set_latitude_tilt_az(parameters, meta)
        meta = self.tz_check(parameters, meta)

        # don't pass resource to base class, set in set_nsrdb instead.
        super().__init__(meta, parameters, output_request)

        # Set the site number using resource
        if isinstance(resource, pd.DataFrame):
            self._site = resource.name
        else:
            self._site = None

        if resource is not None and meta is not None:
            self.set_nsrdb(resource)

    def set_latitude_tilt_az(self, parameters, meta):
        """Check if tilt is specified as latitude and set tilt=lat, az=180 or 0

        Parameters
        ----------
        parameters : dict
            SAM model input parameters.
        meta : pd.DataFrame
            1D table with resource meta data.

        Returns
        -------
        parameters : dict
            SAM model input parameters. If for a pv simulation the "tilt"
            parameter was originally not present or set to 'lat' or 'latitude',
            the tilt will be set to the absolute value of the latitude found
            in meta and the azimuth will be 180 if lat>0, 0 if lat<0.
        """

        set_tilt = False
        if 'pv' in self.MODULE:  # pylint: disable=unsupported-membership-test
            if parameters is not None and meta is not None:
                if 'tilt' not in parameters:
                    warn('No tilt specified, setting at latitude.',
                         SAMInputWarning)
                    set_tilt = True
                else:
                    if (parameters['tilt'] == 'lat'
                            or parameters['tilt'] == 'latitude'):
                        set_tilt = True

        if set_tilt:
            # set tilt to abs(latitude)
            parameters['tilt'] = np.abs(meta['latitude'])
            if meta['latitude'] > 0:
                # above the equator, az = 180
                parameters['azimuth'] = 180
            else:
                # below the equator, az = 0
                parameters['azimuth'] = 0

            logger.debug('Tilt specified at "latitude", setting tilt to: {}, '
                         'azimuth to: {}'
                         .format(parameters['tilt'], parameters['azimuth']))
        return parameters

    def set_nsrdb(self, resource):
        """Set NSRDB resource data arrays.

        Parameters
        ----------
        resource : pd.DataFrame
            2D table with resource data. Available columns must have var_list.
        """
        time_index = resource.index
        self.time_interval = self.get_time_interval(resource.index.values)

        # map resource data names to SAM required data names
        var_map = {'dni': 'dn',
                   'dhi': 'df',
                   'ghi': 'gh',
                   'clearsky_dni': 'dn',
                   'clearsky_dhi': 'df',
                   'clearsky_ghi': 'gh',
                   'wind_speed': 'wspd',
                   'air_temperature': 'tdry',
                   'dew_point': 'tdew',
                   'surface_pressure': 'pres',
                   }

        irrad_vars = ['dn', 'df', 'gh']

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

        resource['lat'] = self.meta['latitude']
        resource['lon'] = self.meta['longitude']
        resource['tz'] = self.meta['timezone']

        if 'elevation' in self.meta:
            resource['elev'] = self.meta['elevation']
        else:
            resource['elev'] = 0.0

        ti_8760 = self.ensure_res_len(time_index)
        resource['minute'] = ti_8760.minute
        resource['hour'] = ti_8760.hour
        resource['year'] = ti_8760.year
        resource['month'] = ti_8760.month
        resource['day'] = ti_8760.day

        self['solar_resource_data'] = resource


class Pvwatts(Solar, ABC):
    """Photovoltaic (PV) generation with pvwattsv5.
    """
    MODULE = 'pvwattsv7'
    PYSAM = PySamPV7

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
        return self.gen_profile() / self.parameters['system_capacity']

    def gen_profile(self):
        """Get AC inverter power generation profile (orig timezone) in kW.

        Returns
        -------
        output : np.ndarray
            1D array of hourly AC inverter power generation in kW.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return np.array(self['ac'], dtype=np.float32) / 1000

    @property
    def default(self):
        """Get the executed default pysam PVWATTS object."""

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
                             }

        super().collect_outputs(output_lookup=output_lookup)


class Pvwattsv5(Pvwatts):
    """Photovoltaic (PV) generation with pvwattsv5.
    """
    MODULE = 'pvwattsv5'
    PYSAM = PySamPV5

    @property
    def default(self):
        """Get the executed default pysam PVWATTSV5 object.

        Returns
        -------
        _default : PySAM.Pvwattsv5
            Executed pvwatts pysam object.
        """
        if self._default is None:
            self._default = DefaultPvwattsv5.default()

        return self._default


class Pvwattsv7(Pvwatts):
    """Photovoltaic (PV) generation with pvwattsv7.
    """
    MODULE = 'pvwattsv7'
    PYSAM = PySamPV7

    @property
    def default(self):
        """Get the executed default pysam PVWATTSV7 object.

        Returns
        -------
        _default : PySAM.Pvwattsv7
            Executed pvwatts pysam object.
        """
        if self._default is None:
            self._default = DefaultPvwattsv7.default()

        return self._default


class TcsMoltenSalt(Solar):
    """Concentrated Solar Power (CSP) generation with tower molten salt
    """
    MODULE = 'tcsmolten_salt'
    PYSAM = PySamCSP

    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None):
        """Initialize a SAM concentrated solar power (CSP) object.
        """
        super().__init__(resource=resource, meta=meta, parameters=parameters,
                         output_request=output_request)

    def cf_profile(self):
        """Get absolute value hourly capacity factor (frac) profile in
        orig timezone.

        Returns
        -------
        cf_profile : np.ndarray
            1D numpy array of capacity factor profile.
            Datatype is float32 and array length is 8760*time_interval.
        """
        return np.abs(self.gen_profile() / self.parameters['system_capacity'])

    @property
    def default(self):
        """Get the executed default pysam CSP object.

        Returns
        -------
        _default : PySAM.TcsmoltenSalt
            Executed TcsmoltenSalt pysam object.
        """
        if self._default is None:
            self._default = DefaultTcsMoltenSalt.default()

        return self._default


class SolarThermal(Solar, ABC):
    """ Base class for solar thermal """
    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None, drop_leap=False):
        """Initialize a SAM solar thermal object

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
            Drops February 29th from the resource data. If False, December
            31st is dropped from leap years. PySAM will not accept csv data on
            Feb 29th. For leap years, December 31st is dropped and time steps
            are shifted to relabel Feb 29th as March 1st, March 1st as March
            2nd, etc.
        """
        self._drop_leap = drop_leap
        super().__init__(resource=resource, meta=meta, parameters=parameters,
                         output_request=output_request, drop_leap=False)

    def set_nsrdb(self, resource):
        """
        Set NSRDB resource file. Overloads Solar.set_nsrdb(). Solar thermal
        PySAM models require a data file, not raw data.

        Parameters
        ----------
        resource : pd.DataFrame
            2D table with resource data. Available columns must have var_list.
        """
        self.time_interval = self.get_time_interval(resource.index.values)
        self._pysam_w_fname = self._create_pysam_wfile(self._meta, resource)
        # pylint: disable=E1101
        self[self._pysam_weather_tag] = self._pysam_w_fname

    def _create_pysam_wfile(self, meta, resource):
        """
        Create PySAM weather input file. PySAM will not accept data on Feb
        29th. For leap years, December 31st is dropped and time steps are
        shifted to relabel Feb 29th as March 1st, March 1st as March 2nd, etc.

        Parameters
        ----------
        meta : pd.DataFrame
            1D table with resource meta data.
        resource : pd.DataFrame
            2D table with resource data. Available columns must have var_list.

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

    def _gen_exec(self, delete_wfile=True):
        """
        Run SAM generation with possibility for follow on econ analysis.

        Parameters
        ----------
        delete_wfile : bool
            Delete PySAM weather file after processing is complete
        """
        super()._gen_exec()

        if delete_wfile and os.path.exists(self._pysam_w_fname):
            os.remove(self._pysam_w_fname)


class SolarWaterHeat(SolarThermal):
    """
    Solar Water Heater generation
    """
    MODULE = 'solarwaterheat'
    PYSAM = PySamSWH

    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None, drop_leap=False):
        """Initialize a SAM solar water heater object.

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
            Drops February 29th from the resource data. If False, December
            31st is dropped from leap years. PySAM will not accept csv data on
            Feb 29th. For leap years, December 31st is dropped and time steps
            are shifted to relabel Feb 29th as March 1st, March 1st as March
            2nd, etc.
        """
        self._pysam_weather_tag = 'solar_resource_file'
        super().__init__(resource=resource, meta=meta, parameters=parameters,
                         output_request=output_request, drop_leap=drop_leap)

    @property
    def default(self):
        """Get the executed default pysam swh object.
        Returns
        -------
        _default : PySAM.
            Executed  pysam object.
        """
        if self._default is None:
            self._default = DefaultSwh.default()

        return self._default


class LinearDirectSteam(SolarThermal):
    """
    Process heat linear Fresnel direct steam generation
    """
    MODULE = 'lineardirectsteam'
    PYSAM = PySamLDS

    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None, drop_leap=False):
        """Initialize a SAM process heat linear Fresnel direct steam object.

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
            Drops February 29th from the resource data. If False, December
            31st is dropped from leap years. PySAM will not accept csv data on
            Feb 29th. For leap years, December 31st is dropped and time steps
            are shifted to relabel Feb 29th as March 1st, March 1st as March
            2nd, etc.
        """
        self._pysam_weather_tag = 'file_name'
        super().__init__(resource=resource, meta=meta, parameters=parameters,
                         output_request=output_request, drop_leap=drop_leap)

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

    @property
    def default(self):
        """Get the executed default pysam linear Fresnel object.
        Returns
        -------
        _default : PySAM.
            Executed  pysam object.
        """
        if self._default is None:
            self._default = DefaultLinearFresnelDsgIph.default()

        return self._default


class TroughPhysicalHeat(SolarThermal):
    """
    Trough Physical Process Heat generation
    """
    MODULE = 'troughphysicalheat'
    PYSAM = PySamTPPH

    def __init__(self, resource=None, meta=None, parameters=None,
                 output_request=None, drop_leap=False):
        """Initialize a SAM trough physical process heat object.

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
            Drops February 29th from the resource data. If False, December
            31st is dropped from leap years. PySAM will not accept csv data on
            Feb 29th. For leap years, December 31st is dropped and time steps
            are shifted to relabel Feb 29th as March 1st, March 1st as March
            2nd, etc.
        """
        self._pysam_weather_tag = 'file_name'
        super().__init__(resource=resource, meta=meta, parameters=parameters,
                         output_request=output_request, drop_leap=drop_leap)

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

    @property
    def default(self):
        """Get the executed default pysam trough object.
        Returns
        -------
        _default : PySAM.
            Executed  pysam object.
        """
        if self._default is None:
            self._default = DefaultTroughPhysicalProcessHeat.default()

        return self._default


class WindPower(Generation):
    """Base class for Wind generation from SAM
    """
    MODULE = 'windpower'
    PYSAM = PySamWindPower

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
            Drops February 29th from the resource data. If False, December
            31st is dropped from leap years.
        """

        # drop the leap day
        if drop_leap:
            resource = self.drop_leap(resource)

        meta = self.tz_check(parameters, meta)

        # don't pass resource to base class, set in set_wtk instead.
        super().__init__(meta, parameters, output_request)

        # Set the site number using resource
        if isinstance(resource, pd.DataFrame):
            self._site = resource.name
        else:
            self._site = None

        if resource is not None and meta is not None:
            self.set_wtk(resource)

    def set_wtk(self, resource):
        """Set WTK resource data arrays.

        Parameters
        ----------
        resource : pd.DataFrame
            2D table with resource data. Available columns must have var_list.
        """

        data_dict = {}
        var_list = ['temperature', 'pressure', 'windspeed', 'winddirection']
        if 'winddirection' not in resource:
            resource['winddirection'] = 0.0

        time_index = resource.index
        self.time_interval = self.get_time_interval(resource.index.values)

        data_dict['fields'] = [1, 2, 3, 4]
        data_dict['heights'] = 4 * [self.parameters['wind_turbine_hub_ht']]

        if 'rh' in resource:
            # set relative humidity for icing.
            rh = np.roll(self.ensure_res_len(resource['rh'].values),
                         int(self.meta['timezone'] * self.time_interval),
                         axis=0)
            data_dict['rh'] = rh.tolist()

        # must be set as matrix in [temperature, pres, speed, direction] order
        # ensure that resource array length is multiple of 8760
        # roll the truncated resource array to local timezone
        temp = np.roll(self.ensure_res_len(resource[var_list].values),
                       int(self.meta['timezone'] * self.time_interval), axis=0)
        data_dict['data'] = temp.tolist()

        resource['lat'] = self.meta['latitude']
        resource['lon'] = self.meta['longitude']
        resource['tz'] = self.meta['timezone']
        resource['elev'] = self.meta['elevation']

        data_dict['minute'] = self.ensure_res_len(time_index.minute)
        data_dict['hour'] = self.ensure_res_len(time_index.hour)
        data_dict['year'] = self.ensure_res_len(time_index.year)
        data_dict['month'] = self.ensure_res_len(time_index.month)
        data_dict['day'] = self.ensure_res_len(time_index.day)

        # add resource data to self.data and clear
        self['wind_resource_data'] = data_dict

    @property
    def default(self):
        """Get the executed default pysam WindPower object.

        Returns
        -------
        _default : PySAM.Windpower
            Executed Windpower pysam object.
        """
        if self._default is None:
            self._default = DefaultWindPower.default()

        return self._default
