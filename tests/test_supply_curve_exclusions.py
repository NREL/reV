# -*- coding: utf-8 -*-
"""
Exclusions unit test module
"""
import numpy as np
import os
import pytest
import warnings

from reV import TESTDATADIR
from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.exclusions import (LayerMask, ExclusionMask,
                                         ExclusionMaskFromDict)
from reV.utilities.exceptions import ExclusionLayerError


CONFIGS = {'urban_pv': {'ri_smod': {'exclude_values': [1, ],
                                    'exclude_nodata': True},
                        'ri_srtm_slope': {'inclusion_range': (0, 5),
                                          'exclude_nodata': True}},
           'rural_pv': {'ri_smod': {'include_values': [1, ],
                                    'exclude_nodata': True},
                        'ri_srtm_slope': {'inclusion_range': (0, 5),
                                          'exclude_nodata': True}},
           'wind': {'ri_smod': {'include_values': [1, ],
                                'exclude_nodata': True},
                    'ri_padus': {'exclude_values': [1, ],
                                 'exclude_nodata': True},
                    'ri_srtm_slope': {'inclusion_range': (0, 20),
                                      'exclude_nodata': True}},
           'weighted': {'ri_smod': {'include_values': [1, ],
                                    'exclude_nodata': True},
                        'ri_padus': {'exclude_values': [1, ], 'weight': 0.5,
                                     'exclude_nodata': True},
                        'ri_srtm_slope': {'inclusion_range': (0, 20),
                                          'exclude_nodata': True}},
           'bad': {'ri_smod': {'exclude_values': [1, 2, 3]}}}

AREA = {'urban_pv': 0.018, 'rural_pv': 1}


def mask_data(data, inclusion_range, exclude_values, include_values,
              weight, exclude_nodata, nodata_value):
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
    exclude_nodata : bool
        Flag to exclude the nodata parameter
    nodata_value : int | float
        Value signifying nodata (nan) field in data input.

    Returns
    -------
    mask : ndarray
        Numeric scalar float mask of inclusion values (1 is include, 0.5 is
        half include, 0 is exclude).
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

    mask = mask.astype('float16') * weight

    if exclude_nodata:
        mask[(data == nodata_value)] = 0.0

    return mask


@pytest.mark.parametrize(('layer_name', 'inclusion_range', 'exclude_values',
                          'include_values', 'weight', 'exclude_nodata'), [
    ('ri_padus', (None, None), [1, ], None, 1, False),
    ('ri_padus', (None, None), [1, ], None, 1, True),
    ('ri_padus', (None, None), [1, ], None, 0.5, False),
    ('ri_padus', (None, None), [1, ], None, 0.5, True),
    ('ri_smod', (None, None), None, [1, ], 1, False),
    ('ri_smod', (None, None), None, [1, ], 1, True),
    ('ri_smod', (None, None), None, [1, ], 0.5, False),
    ('ri_srtm_slope', (None, 5), None, None, 1, False),
    ('ri_srtm_slope', (0, 5), None, None, 1, False),
    ('ri_srtm_slope', (0, 5), None, None, 1, True),
    ('ri_srtm_slope', (None, 5), None, None, 0.5, False),
    ('ri_srtm_slope', (None, 5), None, None, 0.5, True)])
def test_layer_mask(layer_name, inclusion_range, exclude_values,
                    include_values, weight, exclude_nodata):
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
        nodata_value = f.get_nodata_value(layer_name)

    truth = mask_data(data, inclusion_range, exclude_values,
                      include_values, weight, exclude_nodata, nodata_value)

    layer = LayerMask(layer_name, inclusion_range=inclusion_range,
                      exclude_values=exclude_values,
                      include_values=include_values, weight=weight,
                      exclude_nodata=exclude_nodata,
                      nodata_value=nodata_value)
    layer_test = layer._apply_mask(data)
    assert np.allclose(truth, layer_test)

    mask_test = ExclusionMask.run(excl_h5, layers=layer)
    assert np.allclose(truth, mask_test)

    layer_dict = {layer_name: {"inclusion_range": inclusion_range,
                               "exclude_values": exclude_values,
                               "include_values": include_values,
                               "weight": weight,
                               "exclude_nodata": exclude_nodata}}
    dict_test = ExclusionMaskFromDict.run(excl_h5, layers_dict=layer_dict)
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
    with ExclusionLayers(excl_h5) as f:
        for layer, kwargs in layers_dict.items():
            nodata_value = f.get_nodata_value(layer)
            kwargs['nodata_value'] = nodata_value
            layers.append(LayerMask(layer, **kwargs))

    mask_test = ExclusionMask.run(excl_h5, layers=layers,
                                  min_area=min_area)
    assert np.allclose(truth, mask_test)

    dict_test = ExclusionMaskFromDict.run(excl_h5, layers_dict=layers_dict,
                                          min_area=min_area)
    assert np.allclose(truth, dict_test)


def test_bad_layer():
    """
    Test creation of inclusion mask
    """
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')
    excl_dict = CONFIGS['bad']
    with pytest.raises(ExclusionLayerError):
        with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict,
                                   check_layers=True) as f:
            # pylint: disable=pointless-statement
            f.mask

    with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict,
                               check_layers=False) as f:
        assert not f.mask.any()


@pytest.mark.parametrize(('ds_slice'),
                         [None,
                          (1, ),
                          (slice(None), ),
                          (slice(0, 10), ),
                          (slice(0, 100, 2), ),
                          ([5, 10, 2, 30, 50, 7], ),
                          (slice(None), 1),
                          (slice(None), slice(None)),
                          (slice(None), slice(0, 10)),
                          (slice(None), slice(0, 100, 2)),
                          (slice(None), [5, 10, 2, 30, 50, 7], ),
                          (10, 10),
                          (slice(0, 10), slice(0, 10)),
                          (slice(0, 100, 2), slice(0, 100, 2)),
                          ([5, 10, 2, 30, 50, 7], [5, 10, 2, 30, 50, 7])
                          ])
def test_no_excl(ds_slice):
    """
    Test ExclusionMask with no exclusions provided
    """
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')
    with ExclusionLayers(excl_h5) as f:
        shape = f.shape

    truth = np.ones(shape)
    with ExclusionMask(excl_h5) as f:
        if ds_slice is None:
            test = f.mask
        else:
            test = f[ds_slice]
            truth = truth[ds_slice]

        assert np.allclose(truth, test)

    truth = np.ones(shape)
    with ExclusionMaskFromDict(excl_h5) as f:
        if ds_slice is None:
            test = f.mask
        else:
            test = f[ds_slice]
            truth = truth[ds_slice]

        assert np.allclose(truth, test)


def test_multiple_excl_fractions():
    """
    Test that multiple fraction exclusions are handled properly
    """
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')

    excl_dict = {'ri_smod': {'include_values': [1, ], 'weight': 0.5,
                             'exclude_nodata': True}}
    with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
        truth = f.mask

    excl_dict = {'ri_padus': {'exclude_values': [1, ], 'weight': 0.25,
                              'exclude_nodata': True}}
    with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
        truth = np.minimum(truth, f.mask)

    excl_dict = {'ri_smod': {'include_values': [1, ], 'weight': 0.5,
                             'exclude_nodata': True},
                 'ri_padus': {'exclude_values': [1, ], 'weight': 0.25,
                              'exclude_nodata': True}}
    with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
        test = f.mask

    assert np.allclose(test, truth)
    assert np.all(test[test > 0] >= 0.25)


def test_inclusion_weights():
    """
    Test inclusion weights
    """
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')

    excl_dict = {'ri_smod': {'include_values': [1, ], 'weight': 1,
                             'exclude_nodata': True}}
    with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
        truth = f.mask

    excl_dict = {'ri_smod': {'include_values': [2, 3], 'weight': 0.5,
                             'exclude_nodata': True}}
    with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
        truth += f.mask

    excl_dict = {'ri_smod': {'inclusion_weights': {1: 1, 2: 0.5, 3: 0.5},
                             'exclude_nodata': True}}
    with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
        test = f.mask

    assert np.allclose(test, truth)
    assert np.all(test > 0)

    excl_dict = {'ri_smod': {'inclusion_weights': {1.0: 1, 2.0: 0.5, 3.0: 0.5},
                             'exclude_nodata': True}}
    with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
        test = f.mask

    assert np.allclose(test, truth)
    assert np.all(test > 0)


def test_exclusion_range():
    """
    Test a range-based exclusion value
    """
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')

    excl_dict = {'ri_padus': {'include_values': [1, ], 'weight': 0.25,
                              'exclude_nodata': False}}
    with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
        assert (f.mask).any()
        assert not (f.mask).all()
        assert (f.mask == 0.25).any()

    excl_dict = {'ri_padus': {'include_values': [1, ], 'weight': 0.25,
                              'exclude_nodata': False},
                 'ri_srtm_slope': {'exclude_range': [-1e6, 1e6],
                                   'exclude_nodata': True}}
    with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
        assert not (f.mask).any()


def test_force_include_values():
    """
    Test force inclusion
    """
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')

    excl_dict = {'ri_padus': {'exclude_values': [1, ], 'weight': 0.25,
                              'exclude_nodata': True}}
    with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
        truth = f.mask

    excl_dict = {'ri_smod': {'force_include_values': [1, ], 'weight': 0.5,
                             'exclude_nodata': True}}
    with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
        truth = np.maximum(truth, f.mask)

    excl_dict = {'ri_padus': {'exclude_values': [1, ], 'weight': 0.25,
                              'exclude_nodata': True},
                 'ri_smod': {'force_include_values': [1, ], 'weight': 0.5,
                             'exclude_nodata': True}}
    with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
        test = f.mask

    assert np.allclose(test, truth)


def test_force_include_range():
    """
    Test force inclusion of a whole range of float values
    """
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')

    excl_dict = {'ri_padus': {'exclude_values': [1, ], 'weight': 0.25,
                              'exclude_nodata': True}}
    with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
        assert not (f.mask).all()
        assert (f.mask == 0).any()

    excl_dict = {'ri_padus': {'exclude_values': [1, ], 'weight': 0.25,
                              'exclude_nodata': True},
                 'ri_srtm_slope': {'force_include_range': [-1e6, 1e6],
                                   'weight': 0.5,
                                   'exclude_nodata': True}}
    with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
        assert (f.mask == 0.5).all()


def test_legacy_kwargs():
    """We changed all inclusion_* kwargs to include_* kwargs with warning,
    test that the legacy kwargs still work but throw a warning
    """
    excl_h5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')

    excl_dict = {'ri_padus': {'inclusion_values': [1, ], 'weight': 0.25,
                              'exclude_nodata': False}}
    with pytest.warns() as record:
        with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
            assert (f.mask).any()
        assert len(record) == 1
        assert 'use "include_values"' in record[0].message.args[0]

    excl_dict = {'ri_padus': {'inclusion_range': (1, None), 'weight': 0.25,
                              'exclude_nodata': False}}
    with pytest.warns() as record:
        with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
            assert (f.mask).any()
        assert len(record) == 1
        assert 'use "include_range"' in record[0].message.args[0]

    # no warnings with current "include_range" kwarg
    excl_dict = {'ri_padus': {'include_range': (1, None), 'weight': 0.25,
                              'exclude_nodata': False}}
    with warnings.catch_warnings():
        with ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict) as f:
            assert (f.mask).any()
        warnings.simplefilter("error")


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
