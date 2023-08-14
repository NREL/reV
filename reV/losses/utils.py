# -*- coding: utf-8 -*-
"""reV-losses utilities.

"""
import calendar
import logging

import numpy as np

from reV.utilities.exceptions import reVLossesValueError

logger = logging.getLogger(__name__)

# 1900 is just a representative year, since a year input is required
DAYS_PER_MONTH = [calendar.monthrange(1900, i)[1] for i in range(1, 13)]
FIRST_DAY_INDEX_OF_MONTH = np.cumsum([0] + DAYS_PER_MONTH[:-1])


def convert_to_full_month_names(month_names):
    """Format an iterable of month names to match those in :mod:`calendar`.

    This function will format each input name to match the formatting
    in :obj:`calendar.month_name` (upper case, no extra whitespace), and
    it will convert all abbreviations to full month names. No other
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
    :obj:`list`
        A list of month names matching the formatting of
        :obj:`calendar.month_name` (upper case, no extra whitespace).
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


def filter_unknown_month_names(month_names):
    """Split the input into known and unknown month names.

    Parameters
    ----------
    month_names : iter
        An iterable of strings representing the input month names. Month
        names must match the formatting in :obj:`calendar.month_name`
        (upper case, no extra whitespace), otherwise they will be placed
        into the ``unknown_months`` return list.

    Returns
    -------
    known_months : :obj:`list`
        List of known month names.
    unknown_months : :obj:`list`
        List of unknown month names.
    """
    known_months, unknown_months = [], []
    for name in month_names:
        if name in calendar.month_name:
            known_months.append(name)
        else:
            unknown_months.append(name)

    return known_months, unknown_months


def hourly_indices_for_months(month_names):
    """Convert month names into a list of hourly indices.

    Given a list of month names, this function will return a list
    of indices such that any index value corresponds to an hour within
    the input months.

    Parameters
    ----------
    month_names : iter
        An iterable of month names for the desired starting indices.
        The month names must match the formatting in
        :obj:`calendar.month_name` (upper case, no extra whitespace),
        otherwise their hourly indices will not be included in the
        output.

    Returns
    -------
    :obj:`list`
        A list of hourly index values such that any index corresponds to
        an hour within the input months.
    """

    indices = []
    for ind in sorted(month_indices(month_names)):
        start_index = FIRST_DAY_INDEX_OF_MONTH[ind] * 24
        hours_in_month = DAYS_PER_MONTH[ind] * 24
        indices += list(range(start_index, start_index + hours_in_month))

    return indices


def month_indices(month_names):
    """Convert input month names to an indices (0-11) of the months.

    Parameters
    ----------
    month_names : iter
        An iterable of month names for the desired starting indices.
        The month names must match the formatting in
        :obj:`calendar.month_name` (upper case, no extra whitespace),
        otherwise their index will not be included in the output.

    Returns
    -------
    :obj:`set`
        A set of month indices for the input month names. Unknown
        month indices (-1) are removed.
    """
    return {month_index(name) for name in month_names} - {-1}


def month_index(month_name):
    """Convert a month name (as string) to an index (0-11) of the month.

    Parameters
    ----------
    month_name : str
        Name of month to corresponding to desired index. This input
        must match the formatting in :obj:`calendar.month_name`
        (upper case, no extra whitespace).

    Returns
    -------
    :obj:`int`
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
    """Format a month name to match the names in the :mod:`calendar` module.

    In particular, any extra spaces at the beginning or end of the
    string are stripped, and the name is converted to a title (first
    letter is uppercase).

    Parameters
    ----------
    month_name : str
        Name of month.

    Returns
    -------
    :obj:`str`
        Name of month, formatted to match the month names in the
        :mod:`calendar` module.

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
        :obj:`None`.


    Returns
    -------
    :obj:`str` | :obj:`None`
        Unabbreviated month name, or :obj:`None` if input abbreviation
        is not understood.

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


def _validate_arrays_not_empty(obj, array_names=None):
    """Validate that the input data arrays are not empty. """
    array_names = array_names or []
    for name in array_names:
        try:
            arr = getattr(obj, name)
        except AttributeError:
            continue
        if not arr.size:
            msg = "Invalid {} input: Array is empty! - {}"
            msg = msg.format(name.replace('_', ' '), arr)
            logger.error(msg)
            raise reVLossesValueError(msg)
