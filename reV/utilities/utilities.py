"""
Collection of helpful functions
"""
import re


def parse_year(path):
    """
    Attempt to parse

    Parameters
    ----------
    path : str
        File path or file name from which year is to be parsed

    Returns
    -------
    year : int
        Year parsed from path
    """
    match = re.match(r'.*([1-2][0-9]{3})', path)
    if match:
        year = int(match.group(1))
    else:
        raise RuntimeError('Cannot parse year from {}'.format(path))

    return year
