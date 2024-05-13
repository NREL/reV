# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:37:05 2019

@author: gbuster
"""

# pylint: disable=no-member
import os

import numpy as np
import pytest

from reV import TESTDATADIR
from reV.handlers.outputs import Outputs
from reV.supply_curve.extent import SupplyCurveExtent
from reV.supply_curve.points import (
    GenerationSupplyCurvePoint,
    SupplyCurvePoint,
)
from reV.supply_curve.sc_aggregation import SupplyCurveAggregation
from reV.utilities import MetaKeyName

F_EXCL = os.path.join(TESTDATADIR, "ri_exclusions/ri_exclusions.h5")
F_GEN = os.path.join(TESTDATADIR, "gen_out/gen_ri_pv_2012_x000.h5")
TM_DSET = "techmap_nsrdb"
EXCL_DICT = {
    "ri_srtm_slope": {"inclusion_range": (None, 5), "exclude_nodata": True},
    "ri_padus": {"exclude_values": [1], "exclude_nodata": True},
    "ri_reeds_regions": {
        "inclusion_range": (None, 400),
        "exclude_nodata": True,
    },
}

F_TECHMAP = os.path.join(TESTDATADIR, "sc_out/baseline_ri_tech_map.h5")
DSET_TM = "res_ri_pv"
RTOL = 0.001


@pytest.mark.parametrize("resolution", [7, 32, 50, 64, 163])
def test_points_calc(resolution):
    """Test the calculation of the SC points setup from exclusions tiff."""

    with SupplyCurveExtent(F_EXCL, resolution=resolution) as sc:
        assert sc.n_cols >= (sc.exclusions.shape[1] / resolution)
        assert sc.n_rows >= (sc.exclusions.shape[0] / resolution)
        assert len(sc) == (sc.n_rows * sc.n_cols)


@pytest.mark.parametrize(
    ("gids", "resolution"), [(range(361), 64), (range(12), 377)]
)
def test_slicer(gids, resolution):
    """Run tests on the different extent slicing algorithms."""

    with SupplyCurveExtent(F_EXCL, resolution=resolution) as sc:
        for gid in gids:
            row_slice0, col_slice0 = sc.get_excl_slices(gid)
            row_slice1, col_slice1 = SupplyCurvePoint.get_agg_slices(
                gid, sc.exclusions.shape, resolution
            )
            msg = "Slicing failed for gid {} and res {}".format(
                gid, resolution
            )
            assert row_slice0 == row_slice1, msg
            assert col_slice0 == col_slice1, msg


@pytest.mark.parametrize(
    (MetaKeyName.GID, "resolution", "excl_dict", "time_series"),
    [
        (37, 64, None, None),
        (37, 64, EXCL_DICT, None),
        (37, 64, None, 100),
        (37, 64, EXCL_DICT, 100),
        (37, 37, None, None),
        (37, 37, EXCL_DICT, None),
        (37, 37, None, 100),
        (37, 37, EXCL_DICT, 100),
    ],
)
def test_weighted_means(gid, resolution, excl_dict, time_series):
    """
    Test Supply Curve Point exclusions weighted mean calculation
    """
    with SupplyCurvePoint(
        gid, F_EXCL, TM_DSET, excl_dict=excl_dict, resolution=resolution
    ) as point:
        shape = (point._gids.max() + 1,)
        if time_series:
            shape = (time_series,) + shape

        arr = np.random.random(shape)
        means = point.exclusion_weighted_mean(arr.copy())
        excl = point.include_mask_flat[point.bool_mask]
        excl_sum = excl.sum()
        if len(arr.shape) == 2:
            assert means.shape[0] == shape[0]
            x = arr[:, point._gids[point.bool_mask]]
            x *= excl

            x = x[0]
            means = means[0]
        else:
            x = arr[point._gids[point.bool_mask]]
            x *= excl

        test = x.sum() / excl_sum
        assert np.allclose(test, means, rtol=RTOL)


@pytest.mark.parametrize(
    (MetaKeyName.GID, "resolution", "excl_dict", "time_series"),
    [
        (37, 64, None, None),
        (37, 64, EXCL_DICT, None),
        (37, 64, None, 100),
        (37, 64, EXCL_DICT, 100),
        (37, 37, None, None),
        (37, 37, EXCL_DICT, None),
        (37, 37, None, 100),
        (37, 37, EXCL_DICT, 100),
    ],
)
def test_aggregate(gid, resolution, excl_dict, time_series):
    """
    Test Supply Curve Point aggregate calculation
    """
    with SupplyCurvePoint(
        gid, F_EXCL, TM_DSET, excl_dict=excl_dict, resolution=resolution
    ) as point:
        shape = (point._gids.max() + 1,)
        if time_series:
            shape = (time_series,) + shape

        arr = np.random.random(shape)
        total = point.aggregate(arr.copy())
        excl = point.include_mask_flat[point.bool_mask]
        if len(arr.shape) == 2:
            assert total.shape[0] == shape[0]
            x = arr[:, point._gids[point.bool_mask]]
            x *= excl

            x = x[0]
            total = total[0]
        else:
            x = arr[point._gids[point.bool_mask]]
            x *= excl

        test = x.sum()
        assert np.allclose(test, total, rtol=RTOL)


def plot_all_sc_points(resolution=64):
    """Test the calculation of the SC points setup from exclusions tiff."""

    import matplotlib.pyplot as plt

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    _, axs = plt.subplots(1, 1)
    with SupplyCurveExtent(F_EXCL, resolution=resolution) as sc:
        colors *= len(sc)
        for gid in range(len(sc)):
            excl_meta = sc.get_excl_points("meta", gid)
            axs.scatter(
                excl_meta[MetaKeyName.LONGITUDE],
                excl_meta[MetaKeyName.LATITUDE],
                c=colors[gid],
                s=0.01,
            )

    with Outputs(F_GEN) as f:
        axs.scatter(
            f.meta[MetaKeyName.LONGITUDE],
            f.meta[MetaKeyName.LATITUDE],
            c="k",
            s=25,
        )

    axs.axis("equal")
    plt.show()


def plot_single_gen_sc_point(gid=2, resolution=64):
    """Test the calculation of the SC points setup from exclusions tiff."""
    import matplotlib.pyplot as plt

    colors = ["b", "g", "c", "y", "m"]
    colors *= 100

    _, axs = plt.subplots(1, 1)
    gen_index = SupplyCurveAggregation._parse_gen_index(F_GEN)
    with GenerationSupplyCurvePoint(
        gid,
        F_EXCL,
        F_GEN,
        F_TECHMAP,
        DSET_TM,
        gen_index,
        resolution=resolution,
    ) as sc:
        all_gen_gids = list(set(sc._gen_gids))

        excl_meta = sc.exclusions["meta", sc.rows, sc.cols]

        for i, gen_gid in enumerate(all_gen_gids):
            if gen_gid != -1:
                mask = sc._gen_gids == gen_gid
                axs.scatter(
                    excl_meta.loc[mask, MetaKeyName.LONGITUDE],
                    excl_meta.loc[mask, MetaKeyName.LATITUDE],
                    marker="s",
                    c=colors[i],
                    s=1,
                )

                axs.scatter(
                    sc.gen.meta.loc[gen_gid, MetaKeyName.LONGITUDE],
                    sc.gen.meta.loc[gen_gid, MetaKeyName.LATITUDE],
                    c="k",
                    s=100,
                )

        axs.scatter(sc.centroid[1], sc.centroid[0], marker="x", c="k", s=200)

    axs.axis("equal")
    plt.show()


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
