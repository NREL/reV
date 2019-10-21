# -*- coding: utf-8 -*-
"""
Transmission Feature Tests
"""
import os
import pandas as pd
import pytest

from reV import TESTDATADIR
from reV.handlers.transmission import TransmissionFeatures as TF

TRANS_COSTS_1 = {'line_tie_in_cost': 200, 'line_cost': 1000,
                 'station_tie_in_cost': 50, 'center_tie_in_cost': 10,
                 'sink_tie_in_cost': 100, 'available_capacity': 0.1}


TRANS_COSTS_2 = {'line_tie_in_cost': 3000, 'line_cost': 2000,
                 'station_tie_in_cost': 500, 'center_tie_in_cost': 100,
                 'sink_tie_in_cost': 1e6, 'available_capacity': 0.9}

COSTS = {'1-0-43300': 200, '2-0-43300': 3000,
         '1-100-43300': 100200, '2-100-43300': 203000,
         '1-0-68867': 50, '2-0-68867': 500,
         '1-100-68867': 100050, '2-100-68867': 200500,
         '1-0-80844': 10, '2-0-80844': 100,
         '1-100-80844': 100010, '2-100-80844': 200100,
         '1-0-80843': 100, '2-0-80843': 1000000.0,
         '1-100-80843': 100100, '2-100-80843': 1200000.0}

LINE_CAPS = {1: {43430: 0.0, 43439: 0.0, 43440: 81.41666666666666,
                 43416: 81.41666666666666, 43420: 0.0,
                 43428: 81.41666666666666, 43432: 0.0,
                 43448: 81.41666666666666, 43451: 81.41666666666666,
                 43636: 81.41666666666666},
             2: {43430: 95.5, 43439: 95.5, 43440: 1099.0, 43416: 1099.0,
                 43420: 316.0, 43428: 1099.0, 43432: 95.5, 43448: 1099.0,
                 43451: 1099.0, 43636: 1099.0}}


@pytest.fixture
def trans_table():
    """Get the transmission mapping table"""
    path = os.path.join(TESTDATADIR, 'trans_tables/ri_transmission_table.csv')
    trans_table = pd.read_csv(path)
    return trans_table


@pytest.mark.parametrize(('i', 'trans_costs', 'distance', 'gid'),
                         ((1, TRANS_COSTS_1, 0, 43300),
                          (2, TRANS_COSTS_2, 0, 43300),
                          (1, TRANS_COSTS_1, 100, 43300),
                          (2, TRANS_COSTS_2, 100, 43300),
                          (1, TRANS_COSTS_1, 0, 68867),
                          (2, TRANS_COSTS_2, 0, 68867),
                          (1, TRANS_COSTS_1, 100, 68867),
                          (2, TRANS_COSTS_2, 100, 68867),
                          (1, TRANS_COSTS_1, 0, 80844),
                          (2, TRANS_COSTS_2, 0, 80844),
                          (1, TRANS_COSTS_1, 100, 80844),
                          (2, TRANS_COSTS_2, 100, 80844),
                          (1, TRANS_COSTS_1, 0, 80843),
                          (2, TRANS_COSTS_2, 0, 80843),
                          (1, TRANS_COSTS_1, 100, 80843),
                          (2, TRANS_COSTS_2, 100, 80843)))
def test_cost_calculation(i, trans_costs, distance, gid, trans_table):
    """
    Test tranmission capital cost calculation
    """
    tf = TF(trans_table, **trans_costs)
    true_cost = COSTS['{}-{}-{}'.format(i, distance, gid)]
    trans_cost = tf.cost(gid, distance)

    assert true_cost == trans_cost


@pytest.mark.parametrize(('trans_costs', 'capacity', 'gid'),
                         ((TRANS_COSTS_1, 350, 43300),
                          (TRANS_COSTS_2, 350, 43300),
                          (TRANS_COSTS_1, 100, 43300),
                          (TRANS_COSTS_2, 100, 43300),
                          (TRANS_COSTS_1, 350, 80844),
                          (TRANS_COSTS_2, 350, 80844),
                          (TRANS_COSTS_1, 100, 80844),
                          (TRANS_COSTS_2, 100, 80844)))
def test_connect(trans_costs, capacity, gid, trans_table):
    """
    Test connection to transmission lines and load centers
    """
    tf = TF(trans_table, **trans_costs)
    avail_cap = tf[gid].get('avail_cap', None)
    if avail_cap is not None:
        if avail_cap > capacity:
            assert tf.connect(gid, capacity, apply=False)


def test_substation_connect(trans_table):
    """
    Test connection to substation
    """
    capacity = 350
    gid = 68867
    tf = TF(trans_table, **TRANS_COSTS_1, line_limited=False)
    assert tf.connect(gid, capacity, apply=False)

    tf = TF(trans_table, **TRANS_COSTS_1, line_limited=True)
    assert not tf.connect(gid, capacity, apply=False)


@pytest.mark.parametrize(('i', 'trans_costs'), ((1, TRANS_COSTS_1),
                                                (2, TRANS_COSTS_2)))
def test_substation_load_spreading(i, trans_costs, trans_table):
    """
    Test load spreading on connection to substation
    """
    capacity = 350
    gid = 68867
    tf = TF(trans_table, **trans_costs)
    connect = tf.connect(gid, capacity, apply=True)
    assert connect

    line_gids = tf[gid]['lines']
    missing = [gid for gid in line_gids if gid not in LINE_CAPS[i]]

    assert not any(missing), 'New gids not in baseline: {}'.format(missing)
    for line_id in line_gids:
        msg = 'Bad line cap: {}'.format(line_id)
        assert LINE_CAPS[i][line_id] == tf[line_id]['avail_cap'], msg
