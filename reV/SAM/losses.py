# -*- coding: utf-8 -*-
"""reV-losses module.

"""
import calendar


def month_index(month_name):
    """Convert a month name (as string) to an index (0-11) of the month.

    Parameters
    ----------
    month_name : str
        Name of month to corresponding to desired index. This input
        can also be a 3-letter abbreviation of the month name.

    Returns
    -------
    int
        The 0-index of the month, or -1 if the month name is not
        understood.

    Examples
    --------
    >>> month_index("June")
    5
    >>> month_index("jul")
    6
    >>> month_index("Abc")
    -1
    """
    month_name = format_month_name(month_name)
    month_name = full_month_name_from_abbr(month_name) or month_name
    for month_ind in range(12):
        if calendar.month_name[month_ind + 1] == month_name:
            return month_ind

    return -1


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

    Examples
    --------
    >>> format_month_name("June")
    "June"
    >>> format_month_name("aprIl")
    "April"
    >>> format_month_name(" aug  ")
    "Aug"
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

    Examples
    --------
    >>> full_month_name_from_abbr("Jun")
    "June"
    >>> full_month_name_from_abbr("June") is None
    True
    >>> full_month_name_from_abbr('Abcdef') is None
    True
    """
    for month_index in range(1, 13):
        if calendar.month_abbr[month_index] == month_name:
            return calendar.month_name[month_index]
