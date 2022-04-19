# -*- coding: utf-8 -*-
"""reV-losses module.

"""
import calendar


def convert_to_full_month_names(month_names):
    """Format an iterable of month names to match those in `calendar`.

    This function will format each input name to match the formatting
    in `calendar.month_name` (upper case, no extra whitespace), and it
    will convert all abbreviations to full month names. No other
    assumptions are made about the inputs, so an input string "  abc "
    will get formatted and passed though as "Abc".

    Parameters
    ----------
    month_names : iter
        An iterable of strings representing the input month names.
        Month names can be unformatted and contain 3-letter month
        abbreviations.

    Returns
    -------
    list
        A list of month names matching the formatting of
        `calendar.month_name` (upper case, no extra whitespace).
        Abbreviations are also converted to a full month name.

    Examples
    --------
    >>> input_names = ['March', ' aprIl  ', 'Jun', 'jul', '  abc ']
    >>> convert_to_full_month_names(input_names)
    ['March', 'April', 'June', 'July', 'Abc']
    """
    formatted_names = []
    for name in month_names:
        month_name = format_month_name(name)
        month_name = full_month_name_from_abbr(month_name) or month_name
        formatted_names.append(month_name)
    return formatted_names


def month_index(month_name):
    """Convert a month name (as string) to an index (0-11) of the month.

    Parameters
    ----------
    month_name : str
        Name of month to corresponding to desired index. This input
        must match the formatting in `calendar.month_name` (upper case,
        no extra whitespace).

    Returns
    -------
    int
        The 0-index of the month, or -1 if the month name is not
        understood.

    Examples
    --------
    >>> month_index("June")
    5
    >>> month_index("July")
    6
    >>> month_index("Jun")
    -1
    >>> month_index("july")
    -1
    """
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
