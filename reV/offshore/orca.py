# -*- coding: utf-8 -*-
"""reV wrapper for ORCA offshore wind LCOE calculations.

Created on Fri Dec 13 10:03:35 2019

@author: gbuster
"""
import pandas as pd


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
