# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:23:03 2018

@author: gbuster
"""
from copy import deepcopy
import numpy as np
import os

TESTDIR = os.path.dirname(os.path.realpath(__file__))
TOLERANCE = 0.001


def jsonify_key(key):
    """Make a dict key json compatible."""
    if isinstance(key, (int, float)):
        key = str(key)
    return key


def jsonify_data(data):
    """Make a dataset json compatible."""
    new = deepcopy(data)
    if isinstance(new, np.ndarray):
        if np.max(new) > 1e4:
            new = np.around(new.tolist(), decimals=1).tolist()
        elif np.max(new) > 1e3:
            new = np.around(new.tolist(), decimals=2).tolist()
        else:
            new = np.around(new.tolist(), decimals=4).tolist()
    elif isinstance(new, (float)):
        new = round(new, 4)
    return new


def jsonify(outputs):
    """Convert outputs dictionary to JSON compatitble format."""
    new = {}
    for key, data in outputs.items():
        new[jsonify_key(key)] = jsonify_data(data)
    return new


def get_shared_items(x, y):
    """Get a dict of shared values between the two input dicts."""
    shared_items = {}
    for k, v in x.items():
        if k in y:
            if isinstance(v, dict) and isinstance(y[k], dict):
                # recursion! go one level deeper.
                shared_items_2 = get_shared_items(v, y[k])
                if shared_items_2:
                    shared_items[k] = v
            elif (isinstance(v, (np.ndarray, list))
                    and isinstance(y[k], (np.ndarray, list))):
                if np.allclose(v, y[k], atol=TOLERANCE, rtol=TOLERANCE):
                    shared_items[k] = v
            elif x[k] == y[k]:
                shared_items[k] = v
    return shared_items


def dicts_match(x, y):
    """Check whether two dictionaries match."""
    if len(list(x.keys())) == len(list(y.keys())):
        # dicts have the same number of keys (good sign)
        shared_items = get_shared_items(x, y)
        if len(shared_items) == len(list(x.keys())):
            # everything matches
            return True, list(shared_items.keys())
        else:
            # values in keys do not match
            bad_items = {k: x[k] for k in x if k in y and x[k] != y[k]}
            return False, list(bad_items.keys())

    else:
        # keys are missing
        x = set(x.keys())
        y = set(y.keys())
        return False, list(x.symmetric_difference(y))
