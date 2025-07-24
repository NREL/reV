# -*- coding: utf-8 -*-
"""
Transmission Feature Tests
"""

import os

import pandas as pd
import pytest

from reV import TESTDATADIR
from reV.utilities import SupplyCurveField
from reV.handlers.transmission import TransmissionFeatures as TF, POIFeatures

TRANS_COSTS_1 = {
    "line_tie_in_cost": 200,
    "line_cost": 1000,
    "station_tie_in_cost": 50,
    "center_tie_in_cost": 10,
    "sink_tie_in_cost": 100,
    "available_capacity": 0.1,
}


TRANS_COSTS_2 = {
    "line_tie_in_cost": 3000,
    "line_cost": 2000,
    "station_tie_in_cost": 500,
    "center_tie_in_cost": 100,
    "sink_tie_in_cost": 1e6,
    "available_capacity": 0.9,
}

COSTS = {
    "1-0-43300": 200,
    "2-0-43300": 3000,
    "1-100-43300": 100200,
    "2-100-43300": 203000,
    "1-0-68867": 50,
    "2-0-68867": 500,
    "1-100-68867": 100050,
    "2-100-68867": 200500,
    "1-0-80844": 10,
    "2-0-80844": 100,
    "1-100-80844": 100010,
    "2-100-80844": 200100,
    "1-0-80843": 100,
    "2-0-80843": 1000000.0,
    "1-100-80843": 100100,
    "2-100-80843": 1200000.0,
}

LINE_CAPS = {
    1: {
        43430: 0.0,
        43439: 0.0,
        43440: 81.41666666666666,
        43416: 81.41666666666666,
        43420: 0.0,
        43428: 81.41666666666666,
        43432: 0.0,
        43448: 81.41666666666666,
        43451: 81.41666666666666,
        43636: 81.41666666666666,
    },
    2: {
        43430: 95.5,
        43439: 95.5,
        43440: 1099.0,
        43416: 1099.0,
        43420: 316.0,
        43428: 1099.0,
        43432: 95.5,
        43448: 1099.0,
        43451: 1099.0,
        43636: 1099.0,
    },
}


@pytest.fixture
def trans_table():
    """Get the transmission mapping table"""
    path = os.path.join(TESTDATADIR, "trans_tables/ri_transmission_table.csv")
    trans_table = pd.read_csv(path)
    trans_table = trans_table.rename(
        columns=SupplyCurveField.map_from_legacy()
    )
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
    tcosts = trans_costs.copy()
    avail_cap_frac = tcosts.pop("available_capacity")
    tf = TF(trans_table, avail_cap_frac=avail_cap_frac, **tcosts)
    true_cost = COSTS["{}-{}-{}".format(i, distance, gid)]
    trans_cost = tf.cost(gid, distance)

    assert true_cost == trans_cost


@pytest.mark.parametrize(('trans_costs', "capacity", "gid"),
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
    tcosts = trans_costs.copy()
    avail_cap_frac = tcosts.pop("available_capacity")

    tf = TF(trans_table, avail_cap_frac=avail_cap_frac, **tcosts)
    avail_cap = tf[gid].get(SupplyCurveField.TRANS_CAPACITY, None)
    if avail_cap is not None:
        if avail_cap > capacity:
            assert tf.connect(gid, capacity, apply=False)


@pytest.mark.parametrize("line_limited", (False, True))
def test_substation_connect(line_limited, trans_table):
    """
    Test connection to substation
    """
    capacity = 350
    gid = 68867

    trans_costs = TRANS_COSTS_1.copy()
    avail_cap_frac = trans_costs.pop("available_capacity")

    tf = TF(
        trans_table,
        avail_cap_frac=avail_cap_frac,
        line_limited=line_limited,
        **trans_costs,
    )
    if line_limited:
        assert not tf.connect(gid, capacity, apply=False)
    else:
        assert tf.connect(gid, capacity, apply=False)


@pytest.mark.parametrize(
    ("i", "trans_costs"), ((1, TRANS_COSTS_1), (2, TRANS_COSTS_2))
)
def test_substation_load_spreading(i, trans_costs, trans_table):
    """
    Test load spreading on connection to substation
    """
    capacity = 350
    gid = 68867

    tcosts = trans_costs.copy()
    avail_cap_frac = tcosts.pop("available_capacity")

    tf = TF(trans_table, avail_cap_frac=avail_cap_frac, **tcosts)
    connect = tf.connect(gid, capacity, apply=True)
    assert connect

    line_gids = tf[gid]["lines"]
    missing = [gid for gid in line_gids if gid not in LINE_CAPS[i]]

    assert not any(missing), "New gids not in baseline: {}".format(missing)
    for line_id in line_gids:
        msg = "Bad line cap: {}".format(line_id)
        expected_match = (LINE_CAPS[i][line_id]
                          == tf[line_id][SupplyCurveField.TRANS_CAPACITY])
        assert expected_match, msg


def test_init_poi_features():
    """Test POIFeatures initialization."""
    poi_data = pd.DataFrame({SupplyCurveField.TRANS_GID: [0],
                             "ac_cap": [1000]})

    pf = POIFeatures(poi_data)

    assert pf[0][SupplyCurveField.TRANS_CAPACITY] == 1000


def test_poi_features_connect():
    """Test POIFeatures connection capacity."""
    poi_data = pd.DataFrame({SupplyCurveField.TRANS_GID: [0],
                             "ac_cap": [1000]})

    pf = POIFeatures(poi_data)

    assert pf.connect(0, 300) == 300
    assert pf.available_capacity(0) == 700
    assert pf.connect(0, 1000) == 700
    assert pf.available_capacity(0) == 0


def test_poi_features_missing_method():
    """Test POIFeatures for not supported methods."""
    poi_data = pd.DataFrame({SupplyCurveField.TRANS_GID: [0],
                             "ac_cap": [1000]})

    pf = POIFeatures(poi_data)

    with pytest.raises(NotImplementedError):
        pf.cost()

    with pytest.raises(NotImplementedError):
        pf._calc_cost()


def execute_pytest(capture="all", flags="-rapP"):
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
    pytest.main(["-q", "--show-capture={}".format(capture), fname, flags])


if __name__ == "__main__":
    execute_pytest()
