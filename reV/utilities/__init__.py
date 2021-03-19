# -*- coding: utf-8 -*-
"""
reV utilities.
"""
import PySAM
from rex.utilities.loggers import log_versions as rex_log_versions
from reV.version import __version__


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
