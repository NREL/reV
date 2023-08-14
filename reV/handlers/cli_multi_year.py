# -*- coding: utf-8 -*-
"""
Multi-year means CLI utility functions.
"""
import logging
from gaps.cli import as_click_command, CLICommandFromFunction

from reV.handlers.multi_year import my_collect_groups
from reV.utilities import ModuleName


logger = logging.getLogger(__name__)


my_command = CLICommandFromFunction(my_collect_groups,
                                    name=str(ModuleName.MULTI_YEAR),
                                    split_keys=None)
main = as_click_command(my_command)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Multi-Year collect CLI.')
        raise
