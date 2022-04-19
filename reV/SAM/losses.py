# -*- coding: utf-8 -*-
"""reV-losses module.

"""


def format_month_name(month_name):
    """Format a month name to match the names in the `calendar` module.

    In particular, any extra spaces at the beginning or end of the
    string are stripped, and the name is converted to a title (first
    letter is uppercase).

    Parameters
    ----------
    month_name : str
        Name of month.

    Returns
    -------
    str
        Name of month, formatted to match the month names in the
        `calendar` module.
    """
    return month_name.strip().title()
