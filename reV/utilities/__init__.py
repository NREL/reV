# -*- coding: utf-8 -*-
"""
reV utilities.
"""
from enum import Enum
import PySAM
from rex.utilities.loggers import log_versions as rex_log_versions
from reV.version import __version__


class ModuleName(str, Enum):
    """A collection of the module names available in reV.

    Each module name should match the name of the click command
    that will be used to invoke its respective cli. As of 3/1/2022,
    this means that all commands are lowercase with underscores
    replaced by dashes.

    Reference
    ---------
    See this line in the click source code to get the most up-to-date
    click name conversions:
        https://tinyurl.com/4rehbsvf

    """

    BESPOKE = 'bespoke'
    COLLECT = 'collect'
    ECON = 'econ'
    GENERATION = 'generation'
    HYBRIDS = 'hybrids'
    MULTI_YEAR = 'multi-year'
    NRWAL = 'nrwal'
    QA_QC = 'qa-qc'
    REP_PROFILES = 'rep-profiles'
    SUPPLY_CURVE = 'supply-curve'
    SUPPLY_CURVE_AGGREGATION = 'supply-curve-aggregation'

    @classmethod
    def all_names(cls):
        """All module names.

        Returns
        -------
        set
            The set of all module name strings.
        """
        # pylint: disable=no-member
        return {v.value for v in cls.__members__.values()}


def log_versions(logger):
    """Log package versions:
    - rex and reV to info
    - h5py, numpy, pandas, scipy, and PySAM to debug

    Parameters
    ----------
    logger : logging.Logger
        Logger object to log memory message to.
    """
    logger.info('Running with reV version {}'.format(__version__))
    rex_log_versions(logger)
    logger.debug('- PySAM version {}'.format(PySAM.__version__))
