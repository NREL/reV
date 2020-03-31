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
    V2_KEYS = {'wind': WIND}

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
        if version.parse(self.version) < version.parse(self._requirement):
            m = ('Bad PySAM version "{}". Requires: "{}".'
                 .format(self.version, self._requirement))
            if exception:
                logger.error(m)
                raise PySAMVersionError(m)
            else:
                logger.warning(m)
                warn(m, PySAMVersionWarning)
        else:
            logger.info('reV is using PySAM version {}'.format(self.version))

    def _check_inputs_v2(self, tech, parameters):
        """Check PySAM inputs and modify keys to reflect PySAM 2.

        Parameters
        ----------
        tech : str
            reV technology and key to the SAM_2_KEY_CORRECTIONS dict.
        parameters : dict
            SAM input dictionary. Will be checked for valid keys if
            PySAM version > 2.

        Returns
        -------
        parameters : dict
            Updated input parameters dictionary
        """

        corrections = None
        for key, value in self.V2_KEYS.items():
            if key in tech:
                corrections = value
                break

        if corrections is not None:
            if version.parse(self.version) >= version.parse('2'):
                for key in parameters.keys():
                    if key in corrections:
                        new_key = corrections[key]
                        parameters[new_key] = parameters.pop(key)
                        m = ('It appears old SAM v1 keys are being used. '
                             'Updated key "{}" to "{}".'.format(key, new_key))
                        logger.warning(m)
                        warn(m, PySAMVersionWarning)

        return parameters

    @property
    def version(self):
        """Get the PySAM distribution version"""
        return str(get_distribution('nrel-pysam')).split(' ')[1]

    @classmethod
    def run(cls, tech, parameters):
        """Run PySAM version and inputs checker and modify keys to reflect
        PySAM 2 updates.

        Parameters
        ----------
        tech : str
            reV technology and key to the SAM_2_KEY_CORRECTIONS dict.
        parameters : dict
            SAM input dictionary. Will be checked for valid keys if
            PySAM version > 2.

        Returns
        -------
        parameters : dict
            Updated input parameters dictionary
        """
        x = cls()
        parameters = x._check_inputs_v2(tech, parameters)
        return parameters
