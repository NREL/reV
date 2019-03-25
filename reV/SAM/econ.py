#!/usr/bin/env python
"""reV-to-SAM interface module.

Relies heavily upon the SAM Simulation Core (SSC) API module (sscapi) from the
SAM software development kit (SDK).
"""
import logging
import numpy as np
import pandas as pd
from warnings import warn

from reV.handlers.outputs import Outputs
from reV.SAM.PySSC import PySSC
from reV.SAM.SAM import SAM, ParametersManager
from reV.utilities.exceptions import SAMExecutionError


logger = logging.getLogger(__name__)


class Economic(SAM):
    """Base class for SAM economic models.
    """
    MODULE = None

    def __init__(self, ssc, data, parameters, site_parameters=None,
                 output_request='lcoe_fcr'):
        """Initialize a SAM economic model object.

        Parameters
        ----------
        ssc : PySSC() | None
            Python SAM Simulation Core (SSC) object. Can be passed from a
            technology generation class after the SAM technology generation
            simulation has been run. This can be None, signifying that a new
            LCOE analysis is to be performed, not based on a SAM generation
            instance.
        data : PySSC.data_create() | None
            SSC data creation object. If passed from a technology generation
            class, do not run ssc.data_free(data) until after the Economic
            model has been run. This can be None, signifying that a new
            LCOE analysis is to be performed, not based on a SAM generation
            instance.
        parameters : dict | ParametersManager()
            Site-agnostic SAM model input parameters.
        site_parameters : dict
            Optional set of site-specific parameters to complement the
            site-agnostic 'parameters' input arg. Must have an 'offshore'
            column with boolean dtype if running ORCA.
        output_request : list | tuple | str
            Requested SAM output(s) (e.g., 'ppa_price', 'lcoe_fcr').
        """

        # set attribute to store site number
        self.site = None

        if ssc is None and data is None:
            # SAM generation simulation core not passed in. Create new SSC.
            self._ssc = PySSC()
            self._data = self._ssc.data_create()
        else:
            # Received SAM generation SSC.
            self._ssc = ssc
            self._data = data

        # check if offshore wind
        offshore = False
        if site_parameters is not None:
            if 'offshore' in site_parameters:
                offshore = bool(site_parameters['offshore'])

        if isinstance(output_request, (list, tuple)):
            self.output_request = output_request
        else:
            self.output_request = (output_request,)

        # Use Parameters class to manage inputs, defaults, and requirements.
        if isinstance(parameters, ParametersManager):
            self.parameters = parameters
        elif isinstance(parameters, dict) and offshore:
            # use parameters manager for offshore but do not verify or
            # set defaults (ORCA handles this)
            self.parameters = ParametersManager(parameters, self.module,
                                                verify=False, set_def=False)
        else:
            self.parameters = ParametersManager(parameters, self.module)

        # handle site-specific parameters
        if offshore:
            # offshore ORCA parameters will be handled seperately
            self._site_parameters = site_parameters
        # Non-offshore parameters can be added to ParametersManager class
        else:
            self.parameters.update(site_parameters)

    def execute(self, module_to_run, close=True):
        """Execute a SAM economic model calculation.
        """
        self.set_parameters()
        super().execute(module_to_run, close=close)

    @staticmethod
    def parse_sys_cap(site, inputs, site_df):
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

        if ('system_capacity' not in inputs and
                'turbine_capacity' not in inputs and
                'system_capacity' not in site_df and
                'turbine_capacity' not in site_df):
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


class LCOE(Economic):
    """SAM LCOE model.
    """
    MODULE = 'lcoefcr'

    def __init__(self, ssc, data, parameters, site_parameters=None,
                 output_request=('lcoe_fcr',)):
        """Initialize a SAM LCOE economic model object.
        """
        super().__init__(ssc, data, parameters,
                         site_parameters=site_parameters,
                         output_request=output_request)

    def execute(self, module_to_run, close=True):
        """Execute a SAM economic model calculation.
        """
        # check to see if there is an offshore flag and set for this run
        offshore = False
        if hasattr(self, '_site_parameters'):
            if 'offshore' in self._site_parameters:
                offshore = bool(self._site_parameters['offshore'])

        if offshore:
            # execute ORCA here for offshore wind LCOE
            orca = ORCA_LCOE(self.parameters, self._site_parameters)
            self.outputs = {'lcoe_fcr': orca.lcoe}
        else:
            # run SAM LCOE normally for non-offshore technologies
            super().execute(module_to_run, close=close)

    @classmethod
    def reV_run(cls, points_control, site_df, output_request=('lcoe_fcr',)):
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

        calc_aey = False
        if 'annual_energy' not in site_df:
            # annual energy yield has not been input, flag to calculate
            site_df.loc[:, 'annual_energy'] = np.nan
            calc_aey = True

        for site in points_control.sites:
            # get SAM inputs from project_points based on the current site
            config, inputs = points_control.project_points[site]

            # check to see if this site is offshore
            offshore = False
            if 'offshore' in site_df:
                offshore = bool(site_df.loc[site, 'offshore'])

            # calculate the annual energy yield if not input;
            # offshore requires that ORCA does the aey calc
            if calc_aey and not offshore:
                if site_df.loc[site, 'capacity_factor'] > 1:
                    warn('Capacity factor > 1. Dividing by 100.')
                    cf = site_df.loc[site, 'capacity_factor'] / 100
                else:
                    cf = site_df.loc[site, 'capacity_factor']

                # get the system capacity
                sys_cap = cls.parse_sys_cap(site, inputs, site_df)

                # Calc annual energy, mult by 8760 to convert kW to kWh
                aey = sys_cap * cf * 8760

                # add aey to site-specific inputs
                site_df.loc[site, 'annual_energy'] = aey

            # Create SAM econ instance and calculate requested output.
            sim = cls(ssc=None, data=None, parameters=inputs,
                      site_parameters=dict(site_df.loc[site, :]),
                      output_request=output_request)
            sim.execute(cls.MODULE)
            out[site] = sim.outputs

            logger.debug('Outputs for site {} with config "{}", \n\t{}...'
                         .format(site, config, str(out[site])[:100]))
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
        self.system_inputs = system_inputs
        self.system = ORCASystem(self.system_inputs)

        # make a site-specific data structure
        self.site_data = site_data
        self.orca_data_struct = ORCAData(self.site_data)

    @property
    def system_inputs(self):
        """Get the system (site-agnostic) inputs.

        Returns
        -------
        _system_inputs : dict
            System/technology configuration inputs (non-site-specific).
        """
        return self._system_inputs

    @system_inputs.setter
    def system_inputs(self, inp):
        """Set the system (site-agnostic) inputs.

        Parameters
        ----------
        inp : dict | ParametersManager
            System/technology configuration inputs (non-site-specific).
        """
        # extract config inputs as dict if ParametersManager was received
        if isinstance(inp, ParametersManager):
            inp = inp.parameters
        self._system_inputs = inp

    @property
    def site_data(self):
        """Get the site-specific inputs.

        Returns
        -------
        site_data : pd.DataFrame
            Site-specific inputs.
        """
        return self._site_data

    @site_data.setter
    def site_data(self, inp):
        """Set the site-specific inputs.

        Parameters
        ----------
        inp : dict | pd.DataFrame
            Site-specific inputs.
        """
        # convert site parameters to dataframe if necessary
        if not isinstance(inp, pd.DataFrame):
            inp = pd.DataFrame(inp, index=(0,))

        # rename any SAM kwargs to match ORCA requirements
        self._site_data = inp.rename(index=str, columns=self.ARG_MAP)

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

    def __init__(self, ssc, data, parameters, site_parameters=None,
                 output_request=('ppa_price',)):
        """Initialize a SAM single owner economic model object.
        """
        super().__init__(ssc, data, parameters,
                         site_parameters=site_parameters,
                         output_request=output_request)

    def set_gen(self, gen):
        """Set the generation profile (kW) for single owner calculation.

        Parameters
        ----------
        gen : np.ndarray
            Generation profile (8760) in kW.
        """
        self.ssc.data_set_array(self.data, 'gen', gen)

    @classmethod
    def reV_run(cls, points_control, site_df, cf_file,
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
            reV generation h5 output file with path. Generation profiles must
            be included in this file for SingleOwner calculation.
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

        # get the cf_file meta data gid's to use as indexing tools
        with Outputs(cf_file) as cfh:
            site_gids = list(cfh.meta['gid'])

        for site in points_control.sites:
            # get SAM inputs from project_points based on the current site
            config, inputs = points_control.project_points[site]

            # get the system capacity
            sys_cap = cls.parse_sys_cap(site, inputs, site_df)

            # get the index location of the site in question
            isite = site_gids.index(site)

            # Calc generation profile for single owner input
            with Outputs(cf_file) as cfh:
                if 'cf_profile' in cfh.dsets:
                    gen = cfh['cf_profile', :, isite] * sys_cap

            # Create SAM econ instance and calculate requested output.
            sim = cls(ssc=None, data=None, parameters=inputs,
                      site_parameters=dict(site_df.loc[site, :]),
                      output_request=output_request)
            sim.set_gen(gen)
            sim.execute(cls.MODULE)
            out[site] = sim.outputs

            logger.debug('Outputs for site {} with config "{}", \n\t{}...'
                         .format(site, config, str(out[site])[:100]))
        return out
