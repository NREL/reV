# -*- coding: utf-8 -*-
"""
Exclusions unit test module
"""
import numpy as np
import os
import pytest

from reV import TESTDATADIR
from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.exclusions import (LayerMask, ExclusionMask,
                                         ExclusionMaskFromDict)


CONFIGS = {'urban_pv': {'ri_smod': {'exclude_values': [1, ]},
                        'ri_srtm_slope': {'inclusion_range': (0, 5)}},
           'rural_pv': {'ri_smod': {'include_values': [1, ]},
                        'ri_srtm_slope': {'inclusion_range': (0, 5)}},
           'wind': {'ri_smod': {'include_values': [1, ]},
                    'ri_padus': {'exclude_values': [1, ]},
                    'ri_srtm_slope': {'inclusion_range': (0, 20)}},
           'weighted': {'ri_smod': {'include_values': [1, ]},
                        'ri_padus': {'exclude_values': [1, ], 'weight': 0.5},
                        'ri_srtm_slope': {'inclusion_range': (0, 20)}}}

AREA = {'urban_pv': 0.018, 'rural_pv': 1}


def mask_data(data, inclusion_range, exclude_values, include_values,
              weight):
    """
    Apply proper mask to data

    Parameters
    ----------
    data : ndarray
        data to mask
    inclusion_range : tuple
        (min threshold, max threshold) for values to include
    exclude_values : list
        list of values to exclude
        Note: Only supply exclusions OR inclusions
    include_values : list
        List of values to include
        Note: Only supply inclusions OR exclusions
    weight : float
        Weight of pixel to include

    Returns
    -------
    mask : ndarray
        Boolean mask of data
    """
    if any(i is not None for i in inclusion_range):
        min, max = inclusion_range
        mask = True
        if min is not None:
            mask = data >= min

        if max is not None:
            mask *= data <= max

    elif exclude_values is not None:
        mask = ~np.isin(data, exclude_values)

    elif include_values is not None:
        mask = np.isin(data, include_values)

    return mask.astype('float16') * weight


@pytest.mark.parametrize(('layer_name', 'inclusion_range', 'exclude_values',
                          'include_values', 'weight'), [
    ('ri_padus', (None, None), [1, ], None, 1),
    ('ri_padus', (None, None), [1, ], None, 0.5),
    ('ri_smod', (None, None), None, [1, ], 1),
    ('ri_smod', (None, None), None, [1, ], 0.5),
    ('ri_srtm_slope', (None, 5), None, None, 1),
    ('ri_srtm_slope', (0, 5), None, None, 1),
    ('ri_srtm_slope', (None, 5), None, None, 0.5)])
def test_layer_mask(layer_name, inclusion_range, exclude_values,
                    include_values, weight):
    """
    Test creation of layer masks

    Parameters
    ----------
    layer_name : str
        Layer name
    inclusion_range : tuple
        (min threshold, max threshold) for values to include
    exclude_values : list
        list of values to exclude
        Note: Only supply exclusions OR inclusions
    include_values : list
        List of values to include
        Note: Only supply inclusions OR exclusions
    """
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')
    with ExclusionLayers(excl_h5) as f:
        data = f[layer_name]

    truth = mask_data(data, inclusion_range, exclude_values,
                      include_values, weight)

    layer = LayerMask(layer_name, inclusion_range=inclusion_range,
                      exclude_values=exclude_values,
                      include_values=include_values, weight=weight)
    layer_test = layer._apply_mask(data)
    assert np.allclose(truth, layer_test)

    mask_test = ExclusionMask.run(excl_h5, layer)
    assert np.allclose(truth, mask_test)

    layer_dict = {layer_name: {"inclusion_range": inclusion_range,
                               "exclude_values": exclude_values,
                               "include_values": include_values,
                               "weight": weight}}
    dict_test = ExclusionMaskFromDict.run(excl_h5, layer_dict)
    assert np.allclose(truth, dict_test)


@pytest.mark.parametrize(('scenario'),
                         ['urban_pv', 'rural_pv', 'wind', 'weighted'])
def test_inclusion_mask(scenario):
    """
    Test creation of inclusion mask

    Parameters
    ----------
    scenario : str
        Standard reV exclusion scenario
    """
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')
    truth_path = os.path.join(TESTDATADIR, 'ri_exclusions',
                              '{}.npy'.format(scenario))
    truth = np.load(truth_path)

    layers_dict = CONFIGS[scenario]
    min_area = AREA.get(scenario, None)

    layers = []
    for layer, kwargs in layers_dict.items():
        layers.append(LayerMask(layer, **kwargs))

    mask_test = ExclusionMask.run(excl_h5, *layers,
                                  min_area=min_area)
    assert np.allclose(truth, mask_test)

    dict_test = ExclusionMaskFromDict.run(excl_h5, layers_dict,
                                          min_area=min_area)
    assert np.allclose(truth, dict_test)


def execute_pytest(capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
