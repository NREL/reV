# -*- coding: utf-8 -*-
"""reV wrapper for ORCA offshore wind LCOE calculations.

Created on Fri Dec 13 10:03:35 2019

@author: gbuster
"""
from copy import deepcopy
import numpy as np
import pandas as pd
from warnings import warn
import logging

from reV.utilities.exceptions import OrcaOutputWarning

logger = logging.getLogger(__name__)


class ORCA_LCOE:
    """reV-to-ORCA interface framework."""

    # Argument mapping, keys are reV var names, values are ORCA var names
    ARG_MAP = {'capacity_factor': 'gcf', 'cf': 'gcf'}

    def __init__(self, system_inputs, site_data, site_gid=0):
        """Initialize an ORCA LCOE module for a single offshore wind site.

        Parameters
        ----------
        system_inputs : dict
            System/technology configuration inputs (non-site-specific).
        site_data : dict | pd.DataFrame
            Site-specific inputs.
        site_gid : int
            Optional site gid for logging and debugging.
        """
        from ORCA.system import System as ORCASystem
        from ORCA.data import Data as ORCAData

        self._gid = site_gid

        self._system_inputs, self._site_data = \
            self._parse_site_data(system_inputs, site_data, site_gid=site_gid)

        # make an ORCA tech system instance
        self.system = ORCASystem(self.system_inputs)

        # make a site-specific data structure
        self.orca_data_struct = ORCAData(self.site_data)

    @classmethod
    def _parse_site_data(cls, system_inputs, site_data, site_gid=0):
        """Parse the site-specific inputs for ORCA.

        Parameters
        ----------
        system_inputs : dict
            System inputs (non site specific).
        site_data : dict | pd.DataFrame
            Site-specific inputs.
        site_gid : int
            Optional site gid for logging and debugging.

        Returns
        -------
        system_inputs : dict
            System inputs (non site specific).
        site_data : pd.DataFrame
            Site-specific inputs.
        """
        # deep copy so not to modify global inputs
        system_inputs = deepcopy(system_inputs)

        # convert site parameters to dataframe if necessary
        if not isinstance(site_data, pd.DataFrame):
            site_data = pd.DataFrame(site_data, index=(0,))

        # rename any SAM kwargs to match ORCA requirements
        site_data = site_data.rename(index=str, columns=cls.ARG_MAP)

        for c in site_data.columns:
            if c in system_inputs:
                system_inputs[c] = site_data[c].values[0]
                logger.debug('Overwriting "{}" for site gid {} with input: {}'
                             .format(c, site_gid, system_inputs[c]))

        return system_inputs, site_data

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

    @staticmethod
    def _filter_lcoe(lcoe, gid, valid_range=(0, 1000)):
        """Filter bad and out of range lcoe values.

        Parameters
        ----------
        lcoe : float
            LCOE value
        gid : int
            Site gid for logging and debugging.
        valid_range : tuple
            Valid range of LCOE values.
        """
        w = ('ORCA LCOE for site {} is {}, out of valid range {}. '
             'Setting to: {}'
             .format(gid, lcoe, valid_range, np.max(valid_range)))

        if lcoe > np.max(valid_range) or lcoe < np.min(valid_range):
            logger.warning(w)
            warn(w, OrcaOutputWarning)
            lcoe = np.max(valid_range)
        return lcoe

    @property
    def lcoe(self):
        """Get the single-site LCOE.

        Returns
        -------
        lcoe_result : float
            Site LCOE value with units: $/MWh.
        """
        lcoe_result = self.system.lcoe(self.orca_data_struct)
        lcoe_result = self._filter_lcoe(lcoe_result[0], self._gid)
        return lcoe_result
