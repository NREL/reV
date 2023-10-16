# -*- coding: utf-8 -*-
"""
General CLI utility functions.
"""
import logging
from warnings import warn

from gaps.pipeline import Status
from rex.utilities.loggers import init_mult

from reV.utilities import ModuleName
from reV.utilities.exceptions import ConfigWarning, PipelineError


logger = logging.getLogger(__name__)


def init_cli_logging(name, log_directory, verbose):
    """Initialize CLI logger

    Parameters
    ----------
    name : str
        The name to use for the log file written to disk.
    log_directory : str
        Path to log file output directory.
    verbose : bool
        Option to make logger verbose (DEBUG).
    """
    init_mult(name, log_directory, modules=['reV', 'rex'], verbose=verbose)
    logger.info("Initialized reV/rex {}loggers with name {!r} and log "
                "directory {!r}"
                .format("verbose " if verbose else "", name,
                        str(log_directory)))


def format_analysis_years(analysis_years=None):
    """Format user's analysis_years input

    Parameters
    ----------
    analysis_years : int | str | list, optional
        Years to run reV analysis on. Can be an integer or string, or a
        list of integers or strings (or ``None``). This input will get
        converted to a list of values automatically. If ``None``, a
        ``ConfigWarning`` will be thrown. By default, ``None``.

    Returns
    -------
    list
        List of analysis years. This list will never be empty, but it
        can contain ``None`` as the only value.
    """

    if not isinstance(analysis_years, list):
        analysis_years = [analysis_years]

    if analysis_years[0] is None:
        warn('Years may not have been specified, may default '
             'to available years in inputs files.', ConfigWarning)

    return analysis_years


def parse_from_pipeline(config, out_dir, config_key, target_modules):
    """Parse the out file from target modules and set as the values for key.

    This function only updates the ``config_key`` input if it is set to
    ``"PIPELINE"``.

    Parameters
    ----------
    config : dict
        Configuration dictionary. The ``config_key`` will be updated in
        this dictionary if it is set to ``"PIPELINE"``.
    out_dir : str
        Path to pipeline project directory where config and status files
        are located. The status file is expected to be in this
        directory.
    config_key : str
        Key in config files to replace with ``"out_file"`` value(s) from
        previous pipeline step.
    target_modules : list of str | list of `ModuleName`
        List of (previous) target modules to parse for the
        ``config_key``.

    Returns
    -------
    dict
        Input config dictionary with updated ``config_key`` input.

    Raises
    ------
    PipelineError
        If ``"out_file"`` not found in previous target module status
        files.
    """
    if config.get(config_key, None) == 'PIPELINE':
        for target_module in target_modules:
            gen_config_key = "gen" in config_key
            module_sca = target_module == ModuleName.SUPPLY_CURVE_AGGREGATION
            if gen_config_key and module_sca:
                target_key = "gen_fpath"
            else:
                target_key = "out_file"
            val = Status.parse_step_status(out_dir, target_module, target_key)
            if len(val) == 1:
                break
        else:
            raise PipelineError('Could not parse {} from previous '
                                'pipeline jobs.'.format(config_key))

        config[config_key] = val[0]
        logger.info('Config using the following pipeline input for {}: {}'
                    .format(config_key, val[0]))

    return config
