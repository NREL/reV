# -*- coding: utf-8 -*-
"""
Bespoke wind plant optimization CLI utility functions.
"""
import logging
from reV.bespoke.bespoke import BespokeWindPlants
from reV.utilities import ModuleName
from gaps.cli import as_click_command,CLICommandFromClass


logger = logging.getLogger(__name__)


def _preprocessor(config):
    """Preprocess bespoke config user input.

    Parameters
    ----------
    config : dict
        User configuration file input as (nested) dict.

    Returns
    -------
    dict
        Updated config file.
    """
    if isinstance(config["sam_files"], str):
        config["sam_files"] = {'default': config["sam_files"]}

    return config


bespoke_command = CLICommandFromClass(BespokeWindPlants, method="run",
                                      name=str(ModuleName.BESPOKE),
                                      add_collect=False,
                                      split_keys=["project_points"],
                                      config_preprocessor=_preprocessor)
main = as_click_command(bespoke_command)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Bespoke CLI.')
        raise
