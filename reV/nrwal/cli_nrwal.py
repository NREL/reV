# -*- coding: utf-8 -*-
"""
reV-NRWAL module CLI utility functions.
"""
import glob
import logging

from gaps.cli import as_click_command, CLICommandFromClass
from gaps.pipeline import parse_previous_status

from reV.nrwal.nrwal import RevNrwal
from reV.utilities import ModuleName


logger = logging.getLogger(__name__)


def _preprocessor(config, out_dir):
    """Preprocess NRWAL config user input.

    Parameters
    ----------
    config : dict
        User configuration file input as (nested) dict.
    out_dir : str
        Path to output file directory.

    Returns
    -------
    dict
        Updated config file.
    """
    config = _parse_gen_fpath(config, out_dir)
    return config


def _parse_gen_fpath(config, out_dir):
    """Parse gen_fpath user input and convert to list"""

    fpaths = config['gen_fpath']
    if fpaths == 'PIPELINE':
        fpaths = parse_previous_status(out_dir, ModuleName.NRWAL)

    if isinstance(fpaths, str) and '*' in fpaths:
        fpaths = glob.glob(fpaths)
        if not any(fpaths):
            msg = ('Could not find any file paths for '
                   'gen_fpath glob pattern.')
            logger.error(msg)
            raise RuntimeError(msg)

    if isinstance(fpaths, str):
        fpaths = [fpaths]

    config['gen_fpath'] = fpaths
    return config


nrwal_command = CLICommandFromClass(RevNrwal, method="run",
                                    name=str(ModuleName.NRWAL),
                                    add_collect=False,
                                    split_keys=["gen_fpath"],
                                    config_preprocessor=_preprocessor)
main = as_click_command(nrwal_command)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV NRWAL CLI.')
        raise
