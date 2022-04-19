# -*- coding: utf-8 -*-
"""reV-losses module.

"""

import calendar


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


def full_month_name_from_abbr(month_name):
    """Convert a month abbreviation to a full month name.

    Parameters
    ----------
    month_name : str
        Abbreviated month name. Must be one of:
            - "Jan"
            - "Feb"
            - "Mar"
            - "Apr"
            - "May"
            - "Jun"
            - "Jul"
            - "Aug"
            - "Sep"
            - "Oct"
            - "Nov"
            - "Dec"
        If the input does not match one of these, this function returns
        `None`.


    Returns
    -------
    str | None
        Unabbreviated month name, or `None` if input abbreviation is not
        understood.
    """
    for month_index in range(1, 13):
        if calendar.month_abbr[month_index] == month_name:
            return calendar.month_name[month_index]
