# -*- coding: utf-8 -*-
"""
Generation CLI utility functions.
"""
import pprint
import logging

from gaps.cli import as_click_command, CLICommandFromClass

from reV.generation.generation import Gen
from reV.utilities import ModuleName
from reV.utilities.cli_functions import format_analysis_years, init_cli_logging
from reV.utilities.exceptions import ConfigError


logger = logging.getLogger(__name__)


def _preprocessor(config, job_name, log_directory, resource_file, verbose,
                  analysis_years=None):
    """Preprocess generation config user input.

    Parameters
    ----------
    config : dict
        User configuration file input as (nested) dict.
    job_name : str
        Name of ``reV`` job being run.
    log_directory : str
        Path to log output directory.
    verbose : bool
        Flag to signal ``DEBUG`` verbosity (``verbose=True``).
    resource_file : str | list
        Filepath to resource data. This input can be path to a
        single resource HDF5 file or a path including a wildcard input
        like ``/h5_dir/prefix*suffix``.  This path can contain brackets
        ``{}``, which will be filled in by unique values from the
        `analysis_years` input (which must not be null in this case).
        Alternatively, this input can be a list of explicit files to
        process. In this case, the length of the list must match the
        length of the `analysis_years` input exactly, and the path are
        assumed to align with the `analysis_years` (i.e. the first path
        corresponds to the first analysis year, the second path
        corresponds to the second analysis year, and so on). In all
        cases, the resource data must be readable by
        :py:class:`rex.resource.Resource`
        or :py:class:`rex.multi_file_resource.MultiFileResource`.
        (i.e. the resource data conform to the
        `rex data format <https://tinyurl.com/3fy7v5kx>`_). This
        means the data file(s) must contain a 1D ``time_index``
        dataset indicating the UTC time of observation, a 1D
        ``meta`` dataset represented by a DataFrame with
        site-specific columns, and 2D resource datasets that match
        the dimensions of (``time_index``, ``meta``). The time index
        must start at 00:00 of January 1st of the year under
        consideration, and its shape must be a multiple of 8760.

        .. Important:: If you are using custom resource data (i.e.
            not NSRDB/WTK/Sup3rCC, etc.), ensure the following:

                - The data conforms to the
                `rex data format <https://tinyurl.com/3fy7v5kx>`_.
                - The ``meta`` DataFrame is organized such that every
                row is a pixel and at least the columns
                ``latitude``, ``longitude``, ``timezone``, and
                ``elevation`` are given for each location.
                - The time index and associated temporal data is in
                UTC.
                - The latitude is between -90 and 90 and longitude is
                between -180 and 180.
                - For solar data, ensure the DNI/DHI are not zero. You
                can calculate one of these these inputs from the
                other using the relationship

                .. math:: GHI = DNI * cos(SZA) + DHI

    analysis_years : int | list, optional
        A single year or list of years to perform analysis for. These
        years will be used to fill in any brackets ``{}`` in the
        `resource_file` input. If ``None``, the `resource_file`
        input is assumed to be the full path to the single resource
        file to be processed.  By default, ``None``.

    Returns
    -------
    dict
        Updated config file.
    """
    init_cli_logging(job_name, log_directory, verbose)
    config.get("execution_control", {}).setdefault("max_workers")
    analysis_years = format_analysis_years(analysis_years)

    config["resource_file"] = _parse_res_files(resource_file, analysis_years)
    lr_res_file = config.get("low_res_resource_file")
    if lr_res_file is None:
        config["low_res_resource_file"] = [None] * len(analysis_years)
    else:
        config["low_res_resource_file"] = _parse_res_files(lr_res_file,
                                                           analysis_years)

    config['technology'] = (config['technology'].lower()
                            .replace(' ', '').replace('_', ''))
    _log_generation_cli_inputs(config)
    return config


def _parse_res_files(res_fps, analysis_years):
    """Parse the base resource file input into correct ordered list format
    with year imputed in the {} format string"""

    # get base filename, may have {} for year format
    if isinstance(res_fps, str) and '{}' in res_fps:
        # need to make list of res files for each year
        res_fps = [res_fps.format(year) for year in analysis_years]
    elif isinstance(res_fps, str):
        # only one resource file request, still put in list
        res_fps = [res_fps]
    elif not isinstance(res_fps, (list, tuple)):
        msg = ('Bad "resource_file" type, needed str, list, or tuple '
               'but received: {}, {}'
               .format(res_fps, type(res_fps)))
        logger.error(msg)
        raise ConfigError(msg)

    if len(res_fps) != len(analysis_years):
        msg = ('The number of resource files does not match '
               'the number of analysis years!'
               '\n\tResource files: \n\t\t{}'
               '\n\tYears: \n\t\t{}'
               .format(res_fps, analysis_years))
        logger.error(msg)
        raise ConfigError(msg)

    return res_fps


def _log_generation_cli_inputs(config):
    """Log initial generation CLI inputs"""

    logger.info('The following project points were specified: "{}"'
                .format(config.get('project_points', None)))
    logger.info('The following SAM configs are available to this run:\n{}'
                .format(pprint.pformat(config.get('sam_files', None),
                                       indent=4)))
    logger.info('The following is being used for site specific input data: '
                '"{}"'.format(config.get("site_data")))


gen_command = CLICommandFromClass(Gen, method="run",
                                  name=str(ModuleName.GENERATION),
                                  add_collect=False,
                                  split_keys=["project_points",
                                              ("resource_file",
                                               "low_res_resource_file")],
                                  config_preprocessor=_preprocessor)
main = as_click_command(gen_command)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Generation CLI.')
        raise
