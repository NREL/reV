# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:37:05 2019

@author: gbuster
"""
import pytest
import os
from scipy.spatial import cKDTree
from reV.supply_curve.aggregation import Aggregation
from reV.supply_curve.points import SupplyCurvePoint, SupplyCurveExtent
from reV.handlers.outputs import Outputs
from reV import TESTDATADIR


@pytest.mark.parametrize('resolution', [7, 32, 50, 64, 163])
def test_points_calc(resolution):
    """Test the calculation of the SC points setup from exclusions tiff."""
    fpath = os.path.join(TESTDATADIR, 'ri_exclusions/exclusions.tif')

    with SupplyCurveExtent(fpath, resolution=resolution) as sc:
        assert sc.n_cols >= (sc.exclusions.n_cols / resolution)
        assert sc.n_rows >= (sc.exclusions.n_rows / resolution)
        assert len(sc) == (sc.n_rows * sc.n_cols)


def test_single_point_kdtree(gid=2, resolution=64):
    """Test the creation of a kdtree within a SC point vs. passing in."""

    fpath_excl = os.path.join(TESTDATADIR, 'ri_exclusions/exclusions.tif')
    fpath_gen = os.path.join(TESTDATADIR, 'gen_out/gen_ri_pv_2012_x000.h5')

    with Outputs(fpath_gen) as o:
        gen_mask = (o.meta['latitude'] > 41.5)
        gen_tree = cKDTree(o.meta.loc[gen_mask, ['latitude', 'longitude']])

    with SupplyCurvePoint(fpath_excl, fpath_gen, gid=gid,
                          resolution=resolution) as sc1:
        meta1 = sc1.exclusion_meta

    with SupplyCurvePoint(fpath_excl, fpath_gen, gid=gid,
                          resolution=resolution, gen_tree=gen_tree,
                          gen_mask=gen_mask) as sc2:
        meta2 = sc2.exclusion_meta

    assert all((meta1 == meta2))


def test_sc_aggregation(resolution=64):
    """Get the SC points aggregation summary and test that there are expected
    columns and that all 100 resource gids were found"""

    fpath_excl = os.path.join(TESTDATADIR, 'ri_exclusions/exclusions.tif')
    fpath_gen = os.path.join(TESTDATADIR, 'gen_out/gen_ri_pv_2012_x000.h5')

    summary = Aggregation.summary(fpath_excl, fpath_gen, resolution=resolution)
    all_res_gids = []
    for gids in summary['resource_gids']:
        all_res_gids += gids

    assert 'col_ind' in summary
    assert 'row_ind' in summary
    assert 'gen_gids' in summary
    assert len(set(all_res_gids)) == 100


def plot_all_sc_points(resolution=64):
    """Test the calculation of the SC points setup from exclusions tiff."""

    import matplotlib.pyplot as plt
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fpath_excl = os.path.join(TESTDATADIR, 'ri_exclusions/exclusions.tif')
    fpath_gen = os.path.join(TESTDATADIR, 'gen_out/gen_ri_pv_2012_x000.h5')

    _, axs = plt.subplots(1, 1)
    with SupplyCurveExtent(fpath_excl, resolution=resolution) as sc:
        colors *= len(sc)
        for gid in range(len(sc)):
            excl_meta = sc.get_excl_points('meta', gid)
            axs.scatter(excl_meta['longitude'], excl_meta['latitude'],
                        c=colors[gid], s=0.01)
    with Outputs(fpath_gen) as f:
        axs.scatter(f.meta['longitude'], f.meta['latitude'], c='k', s=25)

    axs.axis('equal')
    plt.show()


def plot_single_sc_point(gid=2, resolution=64):
    """Test the calculation of the SC points setup from exclusions tiff."""
    import matplotlib.pyplot as plt

    colors = ['b', 'g', 'c', 'y', 'm']
    colors *= 100

    fpath_excl = os.path.join(TESTDATADIR, 'ri_exclusions/exclusions.tif')
    fpath_gen = os.path.join(TESTDATADIR, 'gen_out/gen_ri_pv_2012_x000.h5')

    _, axs = plt.subplots(1, 1)
    with SupplyCurvePoint(fpath_excl, fpath_gen, gid=gid,
                          resolution=resolution) as sc:

        base_meta = sc.get_base_excl_meta()

        axs.scatter(base_meta.loc[~sc._excl_mask, 'longitude'],
                    base_meta.loc[~sc._excl_mask, 'latitude'],
                    c='r', s=1)

        all_gen_gids = sc.exclusion_meta['global_gen_gid'].unique()

        for i, gen_gid in enumerate(all_gen_gids):
            mask = (sc.exclusion_meta['global_gen_gid'] == gen_gid)
            axs.scatter(sc.exclusion_meta.loc[mask, 'longitude'],
                        sc.exclusion_meta.loc[mask, 'latitude'],
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
