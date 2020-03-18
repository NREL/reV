# -*- coding: utf-8 -*-
"""
Handler utility to parse slicing keys.
"""


def parse_slice(ds_slice):
    """
    Parse dataset slice

    Parameters
    ----------
    ds_slice : tuple | int | slice | list
        Slice to extract from dataset

    Returns
    -------
    ds_slice : tuple
        slice for axis (0, 1)
    """
    if not isinstance(ds_slice, tuple):
        ds_slice = (ds_slice,)

    return ds_slice


def parse_keys(keys):
    """
    Parse keys for complex __getitem__ and __setitem__

    Parameters
    ----------
    keys : string | tuple
        key or key and slice to extract

    Returns
    -------
    key : string
        key to extract
    key_slice : slice | tuple
        Slice or tuple of slices of key to extract
    """
    if isinstance(keys, tuple):
        key = keys[0]
        key_slice = keys[1:]
    else:
        key = keys
        key_slice = (slice(None),)

    return key, key_slice
