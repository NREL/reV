# -*- coding: utf-8 -*-
"""
Collection of helpful functions
"""
import os
import re
import json

from reV.utilities.exceptions import JSONError


def safe_json_load(fpath):
    """Perform a json file load with better exception handling.

    Parameters
    ----------
    fpath : str
        Filepath to .json file.

    Returns
    -------
    j : dict
        Loaded json dictionary.
    """

    if not isinstance(fpath, str):
        raise TypeError('Filepath must be str to load json: {}'.format(fpath))

    if not fpath.endswith('.json'):
        raise JSONError('Filepath must end in .json to load json: {}'
                        .format(fpath))

    if not os.path.isfile(fpath):
        raise JSONError('Could not find json file to load: {}'.format(fpath))

    try:
        with open(fpath, 'r') as f:
            j = json.load(f)
    except json.decoder.JSONDecodeError as e:
        emsg = ('JSON Error:\n{}\nCannot read json file: '
                '"{}"'.format(e, fpath))
        raise JSONError(emsg)

    return j


def parse_year(inp, option='raise'):
    """
    Attempt to parse a year out of a string.

    Parameters
    ----------
    inp : str
        String from which year is to be parsed
    option : str
        Return option:
         - "bool" will return True if year is found, else False.
         - Return year int / raise a RuntimeError otherwise

    Returns
    -------
    out : int | bool
        Year int parsed from inp,
        or boolean T/F (if found and option is bool).
    """

    # char leading year cannot be 0-9
    # char trailing year can be end of str or not 0-9
    regex = r".*[^0-9]([1-2][0-9]{3})($|[^0-9])"

    match = re.match(regex, inp)

    if match:
        out = int(match.group(1))

        if 'bool' in option:
            out = True

    else:
        if 'bool' in option:
            out = False
        else:
            raise RuntimeError('Cannot parse year from {}'.format(inp))

    return out


def mean_irrad(arr):
    """Calc the annual irradiance at a site given an irradiance timeseries.

    Parameters
    ----------
    arr : np.ndarray | pd.Series
        Annual irradiance array in W/m2. Row dimension is time.

    Returns
    -------
    mean : float | np.ndarray
        Mean irradiance values in kWh/m2/day. Float if the input array is
        1D, 1darray if the input array is 2D (multi-site).
    """

    mean = arr.mean(axis=0) / 1000 * 24
    return mean


def check_res_file(res_file):
    """
    Check resource to see if the given path
    - It belongs to a multi-file handler
    - Is on local disk
    - Is a hsds path

    Parameters
    ----------
    res_file : str
        Filepath to single resource file, multi-h5 directory,
        or /h5_dir/prefix*suffix

    Returns
    -------
    multi_h5_res : bool
        Boolean flag to use a MultiFileResource handler
    hsds : bool
        Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
        behind HSDS
    """
    multi_h5_res = False
    hsds = False
    if os.path.isdir(res_file) or ('*' in res_file):
        multi_h5_res = True
    else:
        if not os.path.isfile(res_file):
            try:
                import h5pyd
                with h5pyd.Folder(os.path.dirname(res_file) + '/') as f:
                    hsds = True
                    if res_file not in f:
                        msg = ('{} is not a valid HSDS file path!'
                               .format(res_file))
                        raise FileNotFoundError(msg)
            except Exception as ex:
                msg = ("{} is not a valid file path, and HSDS "
                       "cannot be check for a file at this path:{}!"
                       .format(res_file, ex))
                raise FileNotFoundError(msg)

    return multi_h5_res, hsds
