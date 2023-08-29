# -*- coding: utf-8 -*-
"""
reV Representative Profiles CLI utility functions.
"""
import os
import logging
from warnings import warn

from gaps.cli import as_click_command, CLICommandFromClass

from reV.rep_profiles.rep_profiles import RepProfiles
from reV.utilities import ModuleName
from reV.utilities.cli_functions import (format_analysis_years,
                                         parse_from_pipeline)


logger = logging.getLogger(__name__)


def _preprocessor(config, out_dir, job_name, analysis_years=None):
    """Preprocess rep-profiles config user input.

    Parameters
    ----------
    config : dict
        User configuration file input as (nested) dict.
    out_dir : str
        Path to output file directory.
    job_name : str
        Name of rep-profiles job. This will be included in the output
        file name.
    analysis_years : int | list, optional
        A single year or list of years to perform analysis for. These
        years will be used to fill in any brackets ``{}`` in the
        `cf_dset` or `gen_fpath` inputs. If ``None``, the
        `cf_dset` and `gen_fpath` inputs are assumed to be the full
        dataset name and the full path to the single resource
        file to be processed, respectively. Note that only one of
        `cf_dset` or `gen_fpath` are allowed to contain brackets
        (``{}``) to be filled in by the analysis years.
        By default, ``None``.

    Returns
    -------
    dict
        Updated config file.
    """
    analysis_years = format_analysis_years(analysis_years)

    reg_cols = config.get('reg_cols', None)
    if isinstance(reg_cols, str):
        config["reg_cols"] = [reg_cols]

    key_to_modules = {"gen_fpath": [ModuleName.MULTI_YEAR,
                                    ModuleName.COLLECT,
                                    ModuleName.GENERATION,
                                    ModuleName.SUPPLY_CURVE_AGGREGATION],
                      "rev_summary": [ModuleName.SUPPLY_CURVE_AGGREGATION,
                                      ModuleName.SUPPLY_CURVE]}
    for key, modules in key_to_modules.items():
        config = parse_from_pipeline(config, out_dir, key, modules)

    config = _set_split_keys(config, out_dir, job_name, analysis_years)

    if config.get("aggregate_profiles"):
        check_keys = ['rep_method', 'err_method', 'n_profiles',
                      'save_rev_summary']
        no_effect = [key for key in check_keys if key in config]
        if no_effect:
            msg = ('The following key(s) have no effect when running '
                   'supply curve with "aggregate_profiles=True": "{}". '
                   'To silence this warning, please remove them from the '
                   'config'.format(', '.join(no_effect)))
            logger.warning(msg)
            warn(msg)

    return config


def _set_split_keys(config, out_dir, job_name, analysis_years):
    """Set the gen_fpath, fout, and cf_dset keys"""

    job_name = job_name.replace("rep_profiles", "rep-profiles")
    cf_dset = config.get("cf_dset")
    gen_fpath = config.get("gen_fpath")
    if analysis_years[0] is not None and '{}' in cf_dset:
        config["gen_fpath"] = [gen_fpath for _ in analysis_years]
        config["fout"] = [os.path.join(out_dir, '{}_{}.h5'.format(job_name, y))
                          for y in analysis_years]
        config["cf_dset"] = [cf_dset.format(y) for y in analysis_years]
    elif analysis_years[0] is not None and '{}' in gen_fpath:
        config["gen_fpath"] = [gen_fpath.format(y) for y in analysis_years]
        config["fout"] = [os.path.join(out_dir, '{}_{}.h5'.format(job_name, y))
                          for y in analysis_years]
        config["cf_dset"] = [cf_dset for _ in analysis_years]

    else:
        config["gen_fpath"] = [gen_fpath]
        config["fout"] = [os.path.join(out_dir, '{}.h5'.format(job_name))]
        config["cf_dset"] = [cf_dset]

    return config


SPLIT_KEYS = [("gen_fpath", "fout", "cf_dset")]
rep_profiles_command = CLICommandFromClass(RepProfiles, method="run",
                                           name=str(ModuleName.REP_PROFILES),
                                           add_collect=False,
                                           split_keys=SPLIT_KEYS,
                                           config_preprocessor=_preprocessor,
                                           skip_doc_params=["fout"])
main = as_click_command(rep_profiles_command)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Rep Profiles CLI.')
        raise
