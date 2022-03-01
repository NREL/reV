# -*- coding: utf-8 -*-
"""
reV utilities.
"""
from enum import Enum
import PySAM
from rex.utilities.loggers import log_versions as rex_log_versions
from reV.version import __version__


class ModuleName(str, Enum):
    COLLECT = 'collect'
    ECON = 'econ'
    GENERATION = 'generation'
    MULTI_YEAR = 'multi-year'
    NRWAL = 'nrwal'
    QA_QC = 'qa-qc'
    REP_PROFILES = 'rep-profiles'


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
