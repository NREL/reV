# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:37:05 2019

@author: gbuster
"""
import pytest
import os
from reV.supply_curve.aggregation import Aggregation
from reV.supply_curve.points import SupplyCurvePoint, SupplyCurveExtent
from reV.handlers.outputs import Outputs
from reV import TESTDATADIR


F_EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/exclusions.tif')
F_GEN = os.path.join(TESTDATADIR, 'gen_out/gen_ri_pv_2012_x000.h5')
F_TECHMAP = os.path.join(TESTDATADIR, 'sc_out/baseline_ri_tech_map.h5')
DSET_TM = 'res_ri_pv'


@pytest.mark.parametrize('resolution', [7, 32, 50, 64, 163])
def test_points_calc(resolution):
    """Test the calculation of the SC points setup from exclusions tiff."""

    with SupplyCurveExtent(F_EXCL, resolution=resolution) as sc:
        assert sc.n_cols >= (sc.exclusions.n_cols / resolution)
        assert sc.n_rows >= (sc.exclusions.n_rows / resolution)
        assert len(sc) == (sc.n_rows * sc.n_cols)


@pytest.mark.parametrize('gids, resolution',
                         [(range(361), 64), (range(12), 377)])
def test_slicer(gids, resolution):
    """Run tests on the different extent slicing algorithms."""

    with SupplyCurveExtent(F_EXCL, resolution=resolution) as sc:

        for gid in gids:
            row_slice0, col_slice0 = sc.get_excl_slices(gid)
            row_slice1, col_slice1 = SupplyCurvePoint.get_agg_slices(
                gid, sc.exclusions.shape, resolution)
            msg = ('Slicing failed for gid {} and res {}'
                   .format(gid, resolution))
            assert row_slice0 == row_slice1, msg
            assert col_slice0 == col_slice1, msg


def plot_all_sc_points(resolution=64):
    """Test the calculation of the SC points setup from exclusions tiff."""

    import matplotlib.pyplot as plt
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    _, axs = plt.subplots(1, 1)
    with SupplyCurveExtent(F_EXCL, resolution=resolution) as sc:
        colors *= len(sc)
        for gid in range(len(sc)):
            excl_meta = sc.get_excl_points('meta', gid)
            axs.scatter(excl_meta['longitude'], excl_meta['latitude'],
                        c=colors[gid], s=0.01)

    with Outputs(F_GEN) as f:
        axs.scatter(f.meta['longitude'], f.meta['latitude'], c='k', s=25)

    axs.axis('equal')
    plt.show()


def plot_single_sc_point(gid=2, resolution=64):
    """Test the calculation of the SC points setup from exclusions tiff."""
    import matplotlib.pyplot as plt

    colors = ['b', 'g', 'c', 'y', 'm']
    colors *= 100

    _, axs = plt.subplots(1, 1)
    gen_index = Aggregation._parse_gen_index(F_GEN)
    with SupplyCurvePoint(gid, F_EXCL, F_GEN, F_TECHMAP, DSET_TM, gen_index,
                          resolution=resolution) as sc:

        all_gen_gids = list(set(sc._gen_gids))

        excl_meta = sc.exclusions['meta', sc.rows, sc.cols]

        for i, gen_gid in enumerate(all_gen_gids):
            if gen_gid != -1:
                mask = (sc._gen_gids == gen_gid)
                axs.scatter(excl_meta.loc[mask, 'longitude'],
                            excl_meta.loc[mask, 'latitude'],
                            marker='s', c=colors[i], s=1)

                axs.scatter(sc.gen.meta.loc[gen_gid, 'longitude'],
                            sc.gen.meta.loc[gen_gid, 'latitude'],
                            c='k', s=100)

        axs.scatter(sc.centroid[1], sc.centroid[0], marker='x', c='k', s=200)

    axs.axis('equal')
    plt.show()


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
