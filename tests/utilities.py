# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:23:03 2018

@author: gbuster
"""
import numpy as np


def jsonify(outputs):
    """Convert outputs dictionary to JSON compatitble format."""
    orig_key_list = list(outputs.keys())
    for key in orig_key_list:
        if isinstance(outputs[key], np.ndarray):
            outputs[key] = outputs[key].tolist()
        if isinstance(key, (int, float)):
            outputs[str(key)] = outputs[key]
            del outputs[key]
    return outputs


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
