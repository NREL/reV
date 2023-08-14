# -*- coding: utf-8 -*-
"""
File collection CLI utility functions.
"""
import logging

from gaps.cli.collect import collect
from gaps.cli import as_click_command, CLICommandFromFunction
from gaps.cli.preprocessing import preprocess_collect_config as _preprocessor

from reV.utilities import ModuleName


logger = logging.getLogger(__name__)


SPLIT_KEYS = [("_out_path", "_pattern")]
collect_command = CLICommandFromFunction(collect, name=str(ModuleName.COLLECT),
                                         split_keys=SPLIT_KEYS,
                                         config_preprocessor=_preprocessor)
main = as_click_command(collect_command)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Collect CLI.')
        raise
