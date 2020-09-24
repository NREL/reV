# -*- coding: utf-8 -*-
"""reV-to-SAM econ interface module.

Wraps the NREL-PySAM lcoefcr and singleowner modules with
additional reV features.
"""
from copy import deepcopy
import logging
import numpy as np
from warnings import warn
import PySAM.Lcoefcr as PySamLCOE
import PySAM.Singleowner as PySamSingleOwner

from reV.SAM.defaults import DefaultSingleOwner, DefaultLCOE
from reV.handlers.outputs import Outputs
from reV.SAM.windbos import WindBos
from reV.SAM.SAM import RevPySam
from reV.utilities.exceptions import SAMExecutionError

logger = logging.getLogger(__name__)


class Economic(RevPySam):
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
            site-agnostic 'parameters' input arg.
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
        """Parse site-specific parameters and add to parameter dict.

        Parameters
        ----------
        site_parameters : dict
            Optional set of site-specific parameters to complement the
            site-agnostic 'parameters' input arg.
        """
        self._site_parameters = site_parameters
        if self._site_parameters is not None:
            self.parameters.update(self._site_parameters)

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
            System nameplate capacity in native units (SAM is kW).
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
            Flag to add annual_energy to df.

        Returns
        -------
        site_df : pd.DataFrame
            Same as input but with added labels "capacity_factor" and
            "annual_energy" (latter is dependent on calc_aey flag).
        """

        # get the index location of the site in question
        isite = site_gids.index(site)

        # calculate the capacity factor
        cf = cf_arr[isite]
        if cf > 1:
            warn('Capacity factor > 1. Dividing by 100.')
            cf /= 100
        site_df.loc[site, 'capacity_factor'] = cf

        # calculate the annual energy yield if not input;
        if calc_aey:
            # get the system capacity
            sys_cap = Economic._parse_sys_cap(site, inputs, site_df)

            # Calc annual energy, mult by 8760 to convert kW to kWh
            aey = sys_cap * cf * 8760

            # add aey to site-specific inputs
            site_df.loc[site, 'annual_energy'] = aey
        return site_df

    @staticmethod
    def _get_cf_profiles(sites, cf_file, cf_year):
        """Get the multi-site capacity factor time series profiles.

        Parameters
        ----------
        sites : list
            List of all site GID's to get gen profiles for.
        cf_file : str
            reV generation capacity factor output file with path.
        cf_year : int | str | None
            reV generation year to calculate econ for. Looks for cf_mean_{year}
            or cf_profile_{year}. None will default to a non-year-specific cf
            dataset (cf_mean, cf_profile).

        Returns
        -------
        profiles : np.ndarray
            2D array (time, n_sites) of all capacity factor profiles for all
            the requested sites.
        """

        # Retrieve the generation profile for single owner input
        with Outputs(cf_file) as cfh:

            # get the index location of the site in question
            site_gids = list(cfh.get_meta_arr('gid'))
            isites = [site_gids.index(s) for s in sites]

            # look for the cf_profile dataset
            if 'cf_profile' in cfh.datasets:
                dset = 'cf_profile'
            elif 'cf_profile-{}'.format(cf_year) in cfh.datasets:
                dset = 'cf_profile-{}'.format(cf_year)
            elif 'cf_profile_{}'.format(cf_year) in cfh.datasets:
                dset = 'cf_profile_{}'.format(cf_year)
            else:
                msg = ('Could not find cf_profile values for '
                       'input to SingleOwner. Available datasets: {}'
                       .format(cfh.datasets))
                logger.error(msg)
                raise KeyError(msg)

            profiles = cfh[dset, :, isites]

        return profiles

    @staticmethod
    def _make_gen_profile(isite, site, profiles, site_df, inputs):
        """Get the single-site generation time series and add to inputs dict.

        Parameters
        ----------
        isite : int
            Site index in the profiles array.
        site : int
            Site resource GID.
        profiles : np.ndarray
            2D array (time, n_sites) of all capacity factor profiles for all
            the requested sites.
        site_df : pd.DataFrame
            Dataframe of site-specific input variables. Row index corresponds
            to site number/gid (via df.loc not df.iloc), column labels are the
            variable keys that will be passed forward as SAM parameters.
        inputs : dict
            Dictionary of SAM input parameters.

        Returns
        -------
        inputs : dict
            Dictionary of SAM input parameters with the generation profile
            added.
        """

        sys_cap = Economic._parse_sys_cap(site, inputs, site_df)
        inputs['gen'] = profiles[:, isite] * sys_cap

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
        sim.outputs : dict
            Dictionary keyed by SAM variable names with SAM numerical results.
        """

        # Create SAM econ instance and calculate requested output.
        sim = cls(parameters=inputs,
                  site_parameters=dict(site_df.loc[site, :]),
                  output_request=output_request)
        sim._site = site

        sim.assign_inputs()
        sim.execute()
        sim.collect_outputs()
        sim.outputs_to_utc_arr()

        return sim.outputs


class LCOE(Economic):
    """SAM LCOE model.
    """
    MODULE = 'lcoefcr'
    PYSAM = PySamLCOE

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
            if 'cf_mean' in cfh.datasets:
                cf_arr = cfh['cf_mean']
            elif 'cf_mean-{}'.format(cf_year) in cfh.datasets:
                cf_arr = cfh['cf_mean-{}'.format(cf_year)]
            elif 'cf_mean_{}'.format(cf_year) in cfh.datasets:
                cf_arr = cfh['cf_mean_{}'.format(cf_year)]
            elif 'cf' in cfh.datasets:
                cf_arr = cfh['cf']
            else:
                raise KeyError('Could not find cf_mean values for LCOE. '
                               'Available datasets: {}'.format(cfh.datasets))
        return site_gids, calc_aey, cf_arr

    @property
    def default(self):
        """Get the executed default pysam LCOE FCR object.

        Returns
        -------
        _default : PySAM.Lcoefcr
            Executed Lcoefcr pysam object.
        """
        if self._default is None:
            self._default = DefaultLCOE.default()
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


class SingleOwner(Economic):
    """SAM single owner economic model.
    """
    MODULE = 'singleowner'
    PYSAM = PySamSingleOwner

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
        if inputs is not None:
            if 'total_installed_cost' in inputs:
                if isinstance(inputs['total_installed_cost'], str):
                    if inputs['total_installed_cost'].lower() == 'windbos':
                        wb = WindBos(inputs)
                        inputs['total_installed_cost'] = \
                            wb.total_installed_cost
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
            self._default = DefaultSingleOwner.default()

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

        profiles = cls._get_cf_profiles(points_control.sites, cf_file, cf_year)

        for i, site in enumerate(points_control.sites):
            # get SAM inputs from project_points based on the current site
            _, inputs = points_control.project_points[site]

            # ensure that site-specific data is not persisted to other sites
            site_inputs = deepcopy(inputs)

            # set the generation profile as an input.
            site_inputs = cls._make_gen_profile(i, site, profiles, site_df,
                                                site_inputs)

            out[site] = super().reV_run(site, site_df, site_inputs,
                                        output_request)

        return out
