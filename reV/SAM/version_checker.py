# -*- coding: utf-8 -*-
"""Module to check PySAM versions and correct input keys to new SAM 2 keys.

Created on Mon Feb  3 14:40:42 2020

@author: gbuster
"""
import logging
from warnings import warn
from pkg_resources import get_distribution
from packaging import version
from reV.utilities.exceptions import PySAMVersionError, PySAMVersionWarning


logger = logging.getLogger(__name__)


class PySamVersionChecker:
    """Check the PySAM version and modify input keys if required."""

    WIND = {'wind_farm_losses_percent': 'turb_generic_loss'}
    V2_CORRECTION_KEYS = {'windpower': WIND}

    def __init__(self, requirement='2'):
        """
        Parameters
        ----------
        requirement : str
            PySAM version requirement.
        """
        self._requirement = requirement
        self._check_version()

    def _check_version(self, exception=True):
        """Check the PySAM version and raise exception or warning."""
        check = (version.parse(self.pysam_version)
                 < version.parse(self._requirement))
        if check:
            m = ('Bad PySAM version "{}". Requires: "{}".'
                 .format(self.pysam_version, self._requirement))
            if exception:
                logger.error(m)
                raise PySAMVersionError(m)
            else:
                logger.warning(m)
                warn(m, PySAMVersionWarning)

    def _check_inputs(self, tech, parameters):
        """Check PySAM inputs and modify keys to reflect different
        PySAM versions. Currently set to only correct inputs for PySAM v2.

        Parameters
        ----------
        tech : str
            reV-SAM technology string and key to the V2_CORRECTION_KEYS dict
        parameters : dict
            SAM input dictionary.

        Returns
        -------
        parameters : dict
            Updated input parameters dictionary
        """

        if version.parse(self.pysam_version) >= version.parse('2'):
            parameters = self._check_inputs_v2(tech, parameters)

        return parameters

    def _check_inputs_v2(self, tech, parameters):
        """Check PySAM inputs and modify keys to reflect PySAM 2.

        Parameters
        ----------
        tech : str
            reV-SAM technology string and key to the V2_CORRECTION_KEYS dict
        parameters : dict
            SAM input dictionary. Will be checked for valid keys if
            PySAM version > 2.

        Returns
        -------
        parameters : dict
            Updated input parameters dictionary
        """

        corrections = None
        for key, value in self.V2_CORRECTION_KEYS.items():
            if key in tech:
                corrections = value
                break

        if corrections is not None:
            for key in corrections:
                if key in parameters:
                    new_key = corrections[key]
                    parameters[new_key] = parameters.pop(key)
                    m = ('It appears old SAM v1 keys are being used. '
                         'Updated key "{}" to "{}".'.format(key, new_key))
                    logger.warning(m)
                    warn(m, PySAMVersionWarning)

        return parameters

    @property
    def pysam_version(self):
        """Get the PySAM distribution version"""
        return str(get_distribution('nrel-pysam')).split(' ')[1]

    @classmethod
    def run(cls, tech, parameters):
        """Run PySAM version and inputs checker and modify keys to reflect
        PySAM 2 updates.

        Parameters
        ----------
        tech : str
            reV-SAM technology string and key to the V2_CORRECTION_KEYS dict
        parameters : dict
            SAM input dictionary. Will be checked for valid keys if
            PySAM version > 2.

        Returns
        -------
        parameters : dict
            Updated input parameters dictionary
        """
        x = cls()
        parameters = x._check_inputs(tech, parameters)
        return parameters
