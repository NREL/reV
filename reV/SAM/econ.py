# -*- coding: utf-8 -*-
"""reV-to-SAM econ interface module.

Wraps the NREL-PySAM lcoefcr and singleowner modules with
additional reV features.
"""
import os
from copy import deepcopy
import logging
import numpy as np
import pandas as pd
from warnings import warn
import PySAM.Pvwattsv5 as pysam_pv
import PySAM.Lcoefcr as pysam_lcoe
import PySAM.Singleowner as pysam_so

from reV import TESTDATADIR
from reV.SAM.windbos import WindBos
from reV.handlers.outputs import Outputs
from reV.SAM.SAM import SAM
from reV.utilities.exceptions import SAMExecutionError


logger = logging.getLogger(__name__)


class Economic(SAM):
    """Base class for SAM economic models."""
    MODULE = None

    def __init__(self, parameters=None, site_parameters=None,
                 output_request='lcoe_fcr'):
        """Initialize a SAM economic model object.

        Parameters
        ----------
        parameters : dict | ParametersManager()
            Site-agnostic SAM model input parameters.
        site_parameters : dict
            Optional set of site-specific parameters to complement the
            site-agnostic 'parameters' input arg. Must have an 'offshore'
            column with boolean dtype if running ORCA.
        output_request : list | tuple | str
            Requested SAM output(s) (e.g., 'ppa_price', 'lcoe_fcr').
        """

        self._site = None
        self.parameters = parameters

        if isinstance(output_request, (list, tuple)):
            self.output_request = output_request
        else:
            self.output_request = (output_request,)

        self._parse_site_parameters(site_parameters)

        super().__init__(meta=None, parameters=parameters,
                         output_request=output_request)

    def _parse_site_parameters(self, site_parameters):
        """Parse site-specific parameters including offshore flags.

        Parameters
        ----------
        site_parameters : dict
            Optional set of site-specific parameters to complement the
            site-agnostic 'parameters' input arg. Must have an 'offshore'
            column with boolean dtype if running ORCA.
        """
        # check if offshore wind
        offshore = False
        if site_parameters is not None:
            if 'offshore' in site_parameters:
                offshore = (bool(site_parameters['offshore'])
                            and not np.isnan(site_parameters['offshore']))

        # handle site-specific parameters
        if offshore:
            # offshore ORCA parameters will be handled seperately
            self._site_parameters = site_parameters
        # Non-offshore parameters can be added to ParametersManager class
        else:
            self._site_parameters = None
            if site_parameters is not None:
                self.parameters.update(site_parameters)

    @staticmethod
    def _parse_sys_cap(site, inputs, site_df):
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

        if ('system_capacity' not in inputs
                and 'turbine_capacity' not in inputs
                and 'system_capacity' not in site_df
                and 'turbine_capacity' not in site_df):
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

    @staticmethod
    def _get_annual_energy(site, site_df, site_gids, cf_arr, inputs, calc_aey):
        """Get the single-site cf and annual energy and add to site_df.

        Parameters
        ----------
        site : int
            Site gid.
        site_df : pd.DataFrame
            Dataframe of site-specific input variables. Row index corresponds
            to site number/gid (via df.loc not df.iloc), column labels are the
            variable keys that will be passed forward as SAM parameters.
        site_gids : list
            List of all site gid values from the cf_file.
        cf_arr : np.ndarray
            Array of cf_mean values for all sites in the cf_file for the
            given year.
        inputs : dict
            Dictionary of SAM input parameters.
        calc_aey : bool
            Flag to add annual_energy to df (should be false for ORCA).

        Returns
        -------
        site_df : pd.DataFrame
            Same as input but with added labels "capacity_factor" and
            "annual_energy" (latter is dependent on calc_aey flag).
        """

        # check to see if this site is offshore
        offshore = False
        if 'offshore' in site_df:
            offshore = (bool(site_df.loc[site, 'offshore'])
                        and not np.isnan(site_df.loc[site, 'offshore']))

        # get the index location of the site in question
        isite = site_gids.index(site)

        # calculate the capacity factor
        cf = cf_arr[isite]
        if cf > 1:
            warn('Capacity factor > 1. Dividing by 100.')
            cf /= 100
        site_df.loc[site, 'capacity_factor'] = cf

        # calculate the annual energy yield if not input;
        # offshore requires that ORCA does the aey calc
        if calc_aey and not offshore:
            # get the system capacity
            sys_cap = Economic._parse_sys_cap(site, inputs, site_df)

            # Calc annual energy, mult by 8760 to convert kW to kWh
            aey = sys_cap * cf * 8760

            # add aey to site-specific inputs
            site_df.loc[site, 'annual_energy'] = aey
        return site_df

    @staticmethod
    def _get_gen_profile(site, site_df, cf_file, cf_year, inputs):
        """Get the single-site generation time series and add to inputs dict.

        Parameters
        ----------
        site : int
            Site gid.
        site_df : pd.DataFrame
            Dataframe of site-specific input variables. Row index corresponds
            to site number/gid (via df.loc not df.iloc), column labels are the
            variable keys that will be passed forward as SAM parameters.
        cf_file : str
            reV generation capacity factor output file with path.
        cf_year : int | str | None
            reV generation year to calculate econ for. Looks for cf_mean_{year}
            or cf_profile_{year}. None will default to a non-year-specific cf
            dataset (cf_mean, cf_profile).
        inputs : dict
            Dictionary of SAM input parameters.

        Returns
        -------
        inputs : dict
            Dictionary of SAM input parameters with the generation profile
            added.
        """

        # get the system capacity
        sys_cap = Economic._parse_sys_cap(site, inputs, site_df)

        # Retrieve the generation profile for single owner input
        with Outputs(cf_file) as cfh:

            # get the index location of the site in question
            site_gids = list(cfh.meta['gid'])
            isite = site_gids.index(site)

            # look for the cf_profile dataset
            if 'cf_profile' in cfh.dsets:
                gen = cfh['cf_profile', :, isite] * sys_cap
            elif 'cf_profile-{}'.format(cf_year) in cfh.dsets:
                gen = (cfh['cf_profile-{}'.format(cf_year), :, isite]
                       * sys_cap)
            elif 'cf_profile_{}'.format(cf_year) in cfh.dsets:
                gen = (cfh['cf_profile_{}'.format(cf_year), :, isite]
                       * sys_cap)
            else:
                raise KeyError('Could not find cf_profile values for '
                               'SingleOwner. Available datasets: {}'
                               .format(cfh.dsets))
        # add to input dict
        inputs['gen'] = gen

        return inputs

    def ppa_price(self):
        """Get PPA price ($/MWh).

        Native units are cents/kWh, mult by 10 for $/MWh.
        """
        return self['ppa'] * 10

    def npv(self):
        """Get net present value (NPV) ($).

        Native units are dollars.
        """
        return self['project_return_aftertax_npv']

    def lcoe_fcr(self):
        """Get LCOE ($/MWh).

        Native units are $/kWh, mult by 1000 for $/MWh.
        """
        if 'lcoe_fcr' in self.outputs:
            lcoe = self.outputs['lcoe_fcr']
        else:
            lcoe = self['lcoe_fcr'] * 1000
        return lcoe

    def lcoe_nom(self):
        """Get nominal LCOE ($/MWh) (from PPA/SingleOwner model).

        Native units are cents/kWh, mult by 10 for $/MWh.
        """
        return self['lcoe_nom'] * 10

    def lcoe_real(self):
        """Get real LCOE ($/MWh) (from PPA/SingleOwner model).

        Native units are cents/kWh, mult by 10 for $/MWh.
        """
        return self['lcoe_real'] * 10

    def flip_actual_irr(self):
        """Get actual IRR (from PPA/SingleOwner model).

        Native units are %.
        """
        return self['flip_actual_irr']

    def gross_revenue(self):
        """Get cash flow total revenue (from PPA/SingleOwner model).

        Native units are $.
        """
        cf_tr = np.array(self['cf_total_revenue'], dtype=np.float32)
        cf_tr = np.sum(cf_tr, axis=0)
        return cf_tr

    def collect_outputs(self):
        """Collect SAM econ output_request."""

        output_lookup = {'ppa_price': self.ppa_price,
                         'project_return_aftertax_npv': self.npv,
                         'lcoe_fcr': self.lcoe_fcr,
                         'lcoe_nom': self.lcoe_nom,
                         'lcoe_real': self.lcoe_real,
                         'flip_actual_irr': self.flip_actual_irr,
                         'gross_revenue': self.gross_revenue,
                         }

        super().collect_outputs(output_lookup)

    @classmethod
    def reV_run(cls, site, site_df, inputs, output_request):
        """Run the SAM econ model for a single site.

        Parameters
        ----------
        site : int
            Site gid.
        site_df : pd.DataFrame
            Dataframe of site-specific input variables. Row index corresponds
            to site number/gid (via df.loc not df.iloc), column labels are the
            variable keys that will be passed forward as SAM parameters.
        inputs : dict
            Dictionary of SAM input parameters.
        output_request : list | tuple | str
            Requested SAM output(s) (e.g., 'ppa_price', 'lcoe_fcr').

        Returns
        -------
        sim.outputs : SAM.SiteOutput
            Slotted dictionary emulator keyed by SAM variable names with SAM
            numerical results.
        """

        # Create SAM econ instance and calculate requested output.
        sim = cls(parameters=inputs,
                  site_parameters=dict(site_df.loc[site, :]),
                  output_request=output_request)
        sim._site = site

        sim.assign_inputs()
        sim.execute()
        sim.collect_outputs()

        return sim.outputs


class LCOE(Economic):
    """SAM LCOE model.
    """
    MODULE = 'lcoefcr'
    PYSAM = pysam_lcoe

    def __init__(self, parameters=None, site_parameters=None,
                 output_request=('lcoe_fcr',)):
        """Initialize a SAM LCOE economic model object."""
        super().__init__(parameters, site_parameters=site_parameters,
                         output_request=output_request)

    @staticmethod
    def _parse_lcoe_inputs(site_df, cf_file, cf_year):
        """Parse for non-site-specific LCOE inputs.

        Parameters
        ----------
        site_df : pd.DataFrame
            Dataframe of site-specific input variables. Row index corresponds
            to site number/gid (via df.loc not df.iloc), column labels are the
            variable keys that will be passed forward as SAM parameters.
        cf_file : str
            reV generation capacity factor output file with path.
        cf_year : int | str | None
            reV generation year to calculate econ for. Looks for cf_mean_{year}
            or cf_profile_{year}. None will default to a non-year-specific cf
            dataset (cf_mean, cf_profile).

        Returns
        -------
        site_gids : list
            List of all site gid values from the cf_file.
        calc_aey : bool
            Flag to require calculation of the annual energy yield before
            running LCOE.
        cf_arr : np.ndarray
            Array of cf_mean values for all sites in the cf_file for the
            given year.
        """

        # get the cf_file meta data gid's to use as indexing tools
        with Outputs(cf_file) as cfh:
            site_gids = list(cfh.meta['gid'])

        calc_aey = False
        if 'annual_energy' not in site_df:
            # annual energy yield has not been input, flag to calculate
            site_df.loc[:, 'annual_energy'] = np.nan
            calc_aey = True

        # make sure capacity factor is present in site-specific data
        if 'capacity_factor' not in site_df:
            site_df.loc[:, 'capacity_factor'] = np.nan

        # pull all cf mean values for LCOE calc
        with Outputs(cf_file) as cfh:
            if 'cf_mean' in cfh.dsets:
                cf_arr = cfh['cf_mean']
            elif 'cf_mean-{}'.format(cf_year) in cfh.dsets:
                cf_arr = cfh['cf_mean-{}'.format(cf_year)]
            elif 'cf_mean_{}'.format(cf_year) in cfh.dsets:
                cf_arr = cfh['cf_mean_{}'.format(cf_year)]
            elif 'cf' in cfh.dsets:
                cf_arr = cfh['cf']
            else:
                raise KeyError('Could not find cf_mean values for LCOE. '
                               'Available datasets: {}'.format(cfh.dsets))
        return site_gids, calc_aey, cf_arr

    def execute(self):
        """Execute a SAM economic model calculation."""
        # check to see if there is an offshore flag and set for this run
        offshore = False
        if self._site_parameters is not None:
            if 'offshore' in self._site_parameters:
                offshore = bool(self._site_parameters['offshore'])

        if offshore:
            # execute ORCA here for offshore wind LCOE
            orca = ORCA_LCOE(self.parameters, self._site_parameters)
            self.outputs = {'lcoe_fcr': orca.lcoe}
        else:
            # run SAM LCOE normally for non-offshore technologies
            super().execute()

    @property
    def default(self):
        """Get the executed default pysam LCOE FCR object.

        Returns
        -------
        _default : PySAM.Lcoefcr
            Executed Lcoefcr pysam object.
        """
        if self._default is None:
            res_file = os.path.join(
                TESTDATADIR,
                'SAM/USA AZ Phoenix Sky Harbor Intl Ap (TMY3).csv')
            x = pysam_pv.default('PVWattsLCOECalculator')
            x.LocationAndResource.solar_resource_file = res_file
            x.execute()

            self._default = pysam_lcoe.default('PVWattsLCOECalculator')
            self._default.SimpleLCOE.annual_energy = x.Outputs.annual_energy
            self._default.execute()
        return self._default

    @classmethod
    def reV_run(cls, points_control, site_df, cf_file, cf_year,
                output_request=('lcoe_fcr',)):
        """Execute SAM LCOE simulations based on a reV points control instance.

        Parameters
        ----------
        points_control : config.PointsControl
            PointsControl instance containing project points site and SAM
            config info.
        site_df : pd.DataFrame
            Dataframe of site-specific input variables. Row index corresponds
            to site number/gid (via df.loc not df.iloc), column labels are the
            variable keys that will be passed forward as SAM parameters.
        cf_file : str
            reV generation capacity factor output file with path.
        cf_year : int | str | None
            reV generation year to calculate econ for. Looks for cf_mean_{year}
            or cf_profile_{year}. None will default to a non-year-specific cf
            dataset (cf_mean, cf_profile).
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

        site_gids, calc_aey, cf_arr = cls._parse_lcoe_inputs(site_df, cf_file,
                                                             cf_year)

        for site in points_control.sites:
            # get SAM inputs from project_points based on the current site
            _, inputs = points_control.project_points[site]

            site_df = cls._get_annual_energy(site, site_df, site_gids, cf_arr,
                                             inputs, calc_aey)

            out[site] = super().reV_run(site, site_df, inputs, output_request)

        return out


class ORCA_LCOE:
    """reV-to-ORCA interface framework."""

    # Argument mapping, keys are reV var names, values are ORCA var names
    ARG_MAP = {'capacity_factor': 'gcf', 'cf': 'gcf'}

    def __init__(self, system_inputs, site_data):
        """Initialize an ORCA LCOE module for a single offshore wind site.

        Parameters
        ----------
        system_inputs : dict | ParametersManager
            System/technology configuration inputs (non-site-specific).
        site_data : dict | pd.DataFrame
            Site-specific inputs.
        """
        from ORCA.system import System as ORCASystem
        from ORCA.data import Data as ORCAData

        # make an ORCA tech system instance
        self._system_inputs = system_inputs
        self.system = ORCASystem(self.system_inputs)

        # make a site-specific data structure
        self._site_data = self._parse_site_data(site_data)
        self.orca_data_struct = ORCAData(self.site_data)

    @staticmethod
    def _parse_site_data(inp):
        """Parse the site-specific inputs for ORCA.

        Parameters
        ----------
        inp : dict | pd.DataFrame
            Site-specific inputs.

        Returns
        -------
        inp : pd.DataFrame
            Site-specific inputs.
        """
        # convert site parameters to dataframe if necessary
        if not isinstance(inp, pd.DataFrame):
            inp = pd.DataFrame(inp, index=(0,))

        # rename any SAM kwargs to match ORCA requirements
        return inp.rename(index=str, columns=ORCA_LCOE.ARG_MAP)

    @property
    def system_inputs(self):
        """Get the system (site-agnostic) inputs.

        Returns
        -------
        _system_inputs : dict
            System/technology configuration inputs (non-site-specific).
        """
        return self._system_inputs

    @property
    def site_data(self):
        """Get the site-specific inputs.

        Returns
        -------
        site_data : pd.DataFrame
            Site-specific inputs.
        """
        return self._site_data

    @property
    def lcoe(self):
        """Get the single-site LCOE.

        Returns
        -------
        lcoe_result : float
            Site LCOE value with units: $/MWh.
        """
        lcoe_result = self.system.lcoe(self.orca_data_struct)
        return lcoe_result[0]


class SingleOwner(Economic):
    """SAM single owner economic model.
    """
    MODULE = 'singleowner'
    PYSAM = pysam_so

    def __init__(self, parameters=None, site_parameters=None,
                 output_request=('ppa_price',)):
        """Initialize a SAM single owner economic model object.
        """
        super().__init__(parameters, site_parameters=site_parameters,
                         output_request=output_request)

        # run balance of system cost model if required
        self.parameters, self.windbos_outputs = \
            self._windbos(self.parameters)

    @staticmethod
    def _windbos(inputs):
        """Run SAM Wind Balance of System cost model if requested.

        Parameters
        ----------
        inputs : dict
            Dictionary of SAM key-value pair inputs.
            "total_installed_cost": "windbos" will trigger the windbos method.

        Returns
        -------
        inputs : dict
            Dictionary of SAM key-value pair inputs with the total installed
            cost replaced with WindBOS values if requested.
        output : dict
            Dictionary of windbos cost breakdowns.
        """

        outputs = {}
        if isinstance(inputs['total_installed_cost'], str):
            if inputs['total_installed_cost'].lower() == 'windbos':
                wb = WindBos(inputs)
                inputs['total_installed_cost'] = wb.total_installed_cost
                outputs = wb.output
        return inputs, outputs

    @property
    def default(self):
        """Get the executed default pysam Single Owner object.

        Returns
        -------
        _default : PySAM.Singleowner
            Executed Singleowner pysam object.
        """
        if self._default is None:
            res_file = os.path.join(
                TESTDATADIR,
                'SAM/USA AZ Phoenix Sky Harbor Intl Ap (TMY3).csv')
            x = pysam_pv.default('PVWattsSingleOwner')
            x.LocationAndResource.solar_resource_file = res_file
            x.execute()

            self._default = pysam_so.default('PVWattsSingleOwner')
            self._default.SystemOutput.gen = x.Outputs.ac
            self._default.execute()
        return self._default

    def collect_outputs(self):
        """Collect SAM output_request, including windbos results."""

        windbos_out_vars = [v for v in self.output_request
                            if v in self.windbos_outputs]
        self.output_request = [v for v in self.output_request
                               if v not in windbos_out_vars]

        super().collect_outputs()

        windbos_results = {}
        for request in windbos_out_vars:
            windbos_results[request] = self.windbos_outputs[request]

        self.outputs.update(windbos_results)

    @classmethod
    def reV_run(cls, points_control, site_df, cf_file, cf_year,
                output_request=('ppa_price',)):
        """Execute SAM SingleOwner simulations based on reV points control.

        Parameters
        ----------
        points_control : config.PointsControl
            PointsControl instance containing project points site and SAM
            config info.
        site_df : pd.DataFrame
            Dataframe of site-specific input variables. Row index corresponds
            to site number/gid (via df.loc not df.iloc), column labels are the
            variable keys that will be passed forward as SAM parameters.
        cf_file : str
            reV generation capacity factor output file with path.
        cf_year : int | str | None
            reV generation year to calculate econ for. Looks for cf_mean_{year}
            or cf_profile_{year}. None will default to a non-year-specific cf
            dataset (cf_mean, cf_profile).
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

        for site in points_control.sites:
            # get SAM inputs from project_points based on the current site
            _, inputs = points_control.project_points[site]

            # ensure that site-specific data is not persisted to other sites
            site_inputs = deepcopy(inputs)

            # set the generation profile as an input.
            site_inputs = cls._get_gen_profile(site, site_df, cf_file, cf_year,
                                               site_inputs)

            out[site] = super().reV_run(site, site_df, site_inputs,
                                        output_request)

        return out
