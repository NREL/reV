# -*- coding: utf-8 -*-
"""
reV Representative Profiles CLI utility functions.
"""
import os
import glob
import logging
from warnings import warn

from rex.utilities.utilities import parse_year
from gaps.cli import as_click_command, CLICommandFromClass

from reV.hybrids.hybrids import Hybridization
from reV.utilities.exceptions import PipelineError
from reV.utilities import ModuleName


logger = logging.getLogger(__name__)


def _preprocessor(config, out_dir, job_name):
    """Preprocess hybrids config user input.

    Parameters
    ----------
    config : dict
        User configuration file input as (nested) dict.
    out_dir : str
        Path to output file directory.
    job_name : str
        Name of hybrids job. This will be included in the output
        file name.

    Returns
    -------
    dict
        Updated config file.
    """
    for key in ["solar_fpath", "wind_fpath"]:
        config[key] = _glob_to_yearly_dict(config[key])

    config = _set_paths(config, out_dir, job_name)
    return config


def _glob_to_yearly_dict(fpath):
    """Glob the filepaths into a dictionary based on years. """
    _raise_err_if_pipeline(fpath)
    paths = {}
    for fp in glob.glob(fpath):
        fname = os.path.basename(fp)

        try:
            year = parse_year(fname)
        except RuntimeError:
            year = None

        paths.setdefault(year, []).append(fp)

    return paths


def _raise_err_if_pipeline(fpath):
    """Raise error if fpath input is 'PIPELINE'. """

    if fpath == 'PIPELINE':
        msg = ('Hybrids module cannot infer fpath from "PIPELINE" - '
               'input is ambiguous. Please specify both the solar and '
               'wind fpath before running hybrids module.')
        logger.error(msg)
        raise PipelineError(msg)


def _set_paths(config, out_dir, job_name):
    """Pair solar and wind files and corresponding process names. """

    solar_glob_paths = config["solar_fpath"]
    wind_glob_paths = config["wind_fpath"]
    all_years = set(solar_glob_paths) | set(wind_glob_paths)
    common_years = set(solar_glob_paths) & set(wind_glob_paths)
    if not all_years:
        msg = "No files found that match the input: {!r} and/or {!r}"
        e = msg.format(config['solar_fpath'], config['wind_fpath'])
        logger.error(e)
        raise RuntimeError(e)

    solar_fpaths = []
    wind_fpaths = []
    out_files = []
    for year in all_years:
        if year not in common_years:
            msg = ("No corresponding {} file found for {} input file "
                   "(year: '{}'): {!r}. No hybridization performed for "
                   "this input!")
            resources = (['solar', 'wind'] if year not in solar_glob_paths else
                         ['wind', 'solar'])
            paths = (solar_glob_paths.get(year, [])
                     + wind_glob_paths.get(year, []))
            w = msg.format(*resources, paths, year)
            logger.warning(w)
            warn(w)
            continue

        for fpaths in (solar_glob_paths, wind_glob_paths):
            if len(fpaths[year]) > 1:
                msg = ("Ambiguous number of files found for year '{}': {!r} "
                       "Please ensure there is only one input file per year. "
                       "No hybridization performed for this input!")
                w = msg.format(year, fpaths[year])
                logger.warning(w)
                warn(w)
                break
        else:
            solar_fpaths += solar_glob_paths[year]
            wind_fpaths += wind_glob_paths[year]
            out_fn = ("{}.h5".format(job_name)
                      if year is None
                      else "{}_{}.h5".format(job_name, year))
            out_files += [os.path.join(out_dir, out_fn)]

    if not solar_fpaths or not wind_fpaths:
        msg = "No files found that match the input: {!r} and/or {!r}"
        e = msg.format(config['solar_fpath'], config['wind_fpath'])
        logger.error(e)
        raise RuntimeError(e)

    config["solar_fpath"] = solar_fpaths
    config["wind_fpath"] = wind_fpaths
    config["fout"] = out_files

    return config


SPLIT_KEYS = [("solar_fpath", "wind_fpath", "fout")]
hybrids_command = CLICommandFromClass(Hybridization, method="run",
                                      name=str(ModuleName.HYBRIDS),
                                      add_collect=False, split_keys=SPLIT_KEYS,
                                      config_preprocessor=_preprocessor,
                                      skip_doc_params=["fout"])
main = as_click_command(hybrids_command)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Hybrids CLI.')
        raise
