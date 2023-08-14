# -*- coding: utf-8 -*-
# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""
import numpy as np
import os
import pandas as pd
import pytest

from rex import Resource
from rex.utilities.exceptions import ResourceRuntimeError
from rex.utilities.utilities import safe_json_load

from reV.config.base_analysis_config import AnalysisConfig
from reV.config.project_points import ProjectPoints, PointsControl
from reV.generation.generation import Gen
from reV.SAM.SAM import RevPySam
from reV import TESTDATADIR
from reV.utilities.exceptions import ConfigError


def test_config_entries():
    """
    Test BaseConfig check_entry test
    """
    config_path = os.path.join(TESTDATADIR, 'config/collection.json')
    with pytest.raises(ConfigError):
        AnalysisConfig(config_path)


def test_clearsky():
    """
    Test Clearsky
    """
    res_file = os.path.join(TESTDATADIR, 'nsrdb/', 'ri_100_nsrdb_2012.h5')
    sam_files = os.path.join(TESTDATADIR, 'SAM/naris_pv_1axis_inv13_cs.json')
    pp = ProjectPoints(slice(0, 10), sam_files, 'pvwattsv5',
                       res_file=res_file)
    with pytest.raises(ResourceRuntimeError):
        # Get the SAM resource object
        RevPySam.get_sam_res(res_file, pp, pp.tech)


@pytest.mark.parametrize(('start', 'interval'),
                         [[0, 1], [13, 1], [10, 2], [13, 3]])
def test_proj_control_iter(start, interval):
    """Test the iteration of the points control."""
    n = 3
    res_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    sam_files = os.path.join(TESTDATADIR,
                             'SAM/wind_gen_standard_losses_0.json')
    pp = ProjectPoints(slice(start, 100, interval), sam_files, 'windpower',
                       res_file=res_file)
    pc = PointsControl(pp, sites_per_split=n)

    for i, pp_split in enumerate(pc):
        i0_nom = i * n
        i1_nom = i * n + n
        split = pp_split.project_points.df
        target = pp.df.iloc[i0_nom:i1_nom, :]
        msg = 'PointsControl iterator split did not function correctly!'
        assert all(split == target), msg


@pytest.mark.parametrize(('start', 'interval'),
                         [[0, 1], [13, 1], [10, 2], [13, 3]])
def test_proj_points_split(start, interval):
    """Test the split operation of project points."""
    res_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    sam_files = os.path.join(TESTDATADIR,
                             'SAM/wind_gen_standard_losses_0.json')
    pp = ProjectPoints(slice(start, 100, interval), sam_files, 'windpower',
                       res_file=res_file)

    iter_interval = 5
    for i0 in range(0, len(pp), iter_interval):
        i1 = i0 + iter_interval
        if i1 > len(pp):
            break

        pp_0 = ProjectPoints.split(i0, i1, pp)

        msg = 'ProjectPoints split did not function correctly!'
        assert pp_0.sites == pp.sites[i0:i1], msg
        assert all(pp_0.df == pp.df.iloc[i0:i1]), msg


def test_split_iter():
    """Test Points_Control on two slices of ProjectPoints"""
    res_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    sam_files = os.path.join(TESTDATADIR,
                             'SAM/wind_gen_standard_losses_0.json')
    pp = ProjectPoints(slice(0, 500, 5), sam_files, 'windpower',
                       res_file=res_file)

    n = 3
    for s, e in [(0, 50), (50, 100)]:
        pc = PointsControl.split(s, e, pp, sites_per_split=n)

        for i, pp_split in enumerate(pc):
            i0_nom = s + i * n
            i1_nom = s + i * n + n
            if i1_nom >= e:
                i1_nom = e

            split = pp_split.project_points.df
            target = pp.df.iloc[i0_nom:i1_nom]

            msg = 'PointsControl iterator split did not function correctly!'
            assert split.equals(target), msg


def test_config_mapping():
    """Test the mapping of multiple configs in the project points."""
    fpp = os.path.join(TESTDATADIR, 'project_points/pp_offshore.csv')
    sam_files = {'onshore': os.path.join(
                 TESTDATADIR, 'SAM/wind_gen_standard_losses_0.json'),
                 'offshore': os.path.join(
                 TESTDATADIR, 'SAM/wind_gen_standard_losses_1.json')}
    df = pd.read_csv(fpp, index_col=0)
    pp = ProjectPoints(fpp, sam_files, 'windpower')
    pc = PointsControl(pp, sites_per_split=100)
    for i, pc_split in enumerate(pc):
        for site in pc_split.sites:
            cid = pc_split.project_points[site][0]
            assert cid == df.loc[site].values[0]


def test_sam_config_kw_replace():
    """Test that the SAM config with old keys from pysam v1 gets updated on
    the fly and gets propogated to downstream splits."""

    fpp = os.path.join(TESTDATADIR, 'project_points/pp_offshore.csv')
    sam_files = {'onshore': os.path.join(
                 TESTDATADIR, 'SAM/wind_gen_standard_losses_0.json'),
                 'offshore': os.path.join(
                 TESTDATADIR, 'SAM/wind_gen_standard_losses_1.json')}
    res_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    pp = ProjectPoints(fpp, sam_files, 'windpower')

    gen = Gen('windpower', pp, sam_files, resource_file=res_file,
              sites_per_worker=100)
    config_on = gen.project_points.sam_inputs['onshore']
    config_of = gen.project_points.sam_inputs['offshore']
    assert 'turb_generic_loss' in config_on
    assert 'turb_generic_loss' in config_of

    pp_split = ProjectPoints.split(0, 10000, gen.project_points)
    config_on = pp_split.sam_inputs['onshore']
    config_of = pp_split.sam_inputs['offshore']
    assert 'turb_generic_loss' in config_on
    assert 'turb_generic_loss' in config_of

    pc_split = PointsControl.split(0, 10000, gen.project_points)
    config_on = pc_split.project_points.sam_inputs['onshore']
    config_of = pc_split.project_points.sam_inputs['offshore']
    assert 'turb_generic_loss' in config_on
    assert 'turb_generic_loss' in config_of

    for ipc in pc_split:
        if 'onshore' in ipc.project_points.sam_inputs:
            config = ipc.project_points.sam_inputs['onshore']
            assert 'turb_generic_loss' in config

        if 'offshore' in ipc.project_points.sam_inputs:
            config = ipc.project_points.sam_inputs['offshore']
            assert 'turb_generic_loss' in config


@pytest.mark.parametrize('counties', [['Washington'], ['Providence', 'Kent']])
def test_regions(counties):
    """
    Test ProjectPoint.regions class method
    """
    res_file = os.path.join(TESTDATADIR, 'nsrdb/', 'ri_100_nsrdb_2012.h5')
    sam_files = os.path.join(TESTDATADIR, 'SAM/naris_pv_1axis_inv13_cs.json')

    with Resource(res_file) as f:
        meta = f.meta

    baseline = meta.loc[meta['county'].isin(counties)].index.values.tolist()

    regions = {c: 'county' for c in counties}
    pp = ProjectPoints.regions(regions, res_file, sam_files)

    assert sorted(baseline) == pp.sites


@pytest.mark.parametrize('sites', [1, 2, 5, 10])
def test_coords(sites):
    """
    Test ProjectPoint.lat_lon_coords class method
    """
    res_file = os.path.join(TESTDATADIR, 'nsrdb/', 'ri_100_nsrdb_2012.h5')
    sam_files = os.path.join(TESTDATADIR, 'SAM/naris_pv_1axis_inv13_cs.json')

    with Resource(res_file) as f:
        meta = f.meta

    gids = np.random.choice(meta.index.values, sites, replace=False).tolist()
    if not isinstance(gids, list):
        gids = [gids]

    lat_lons = meta.loc[gids, ['latitude', 'longitude']].values
    pp = ProjectPoints.lat_lon_coords(lat_lons, res_file, sam_files)

    assert sorted(gids) == pp.sites


def test_duplicate_coords():
    """
    Test ProjectPoint.lat_lon_coords duplicate coords error
    """
    res_file = os.path.join(TESTDATADIR, 'nsrdb/', 'ri_100_nsrdb_2012.h5')
    sam_files = os.path.join(TESTDATADIR, 'SAM/naris_pv_1axis_inv13_cs.json')

    with Resource(res_file) as f:
        meta = f.meta

    duplicates = meta.loc[[2, 3, 3, 4], ['latitude', 'longitude']].values

    with pytest.raises(RuntimeError):
        ProjectPoints.lat_lon_coords(duplicates, res_file, sam_files)

    regions = {'Kent': 'county', 'Rhode Island': 'state'}
    with pytest.raises(RuntimeError):
        ProjectPoints.regions(regions, res_file, sam_files)


def test_sam_configs():
    """
    Test supplying SAM config as a JSON or a dict
    """
    fpp = os.path.join(TESTDATADIR, 'project_points/pp_offshore.csv')
    sam_files = {'onshore': os.path.join(
                 TESTDATADIR, 'SAM/wind_gen_standard_losses_0.json'),
                 'offshore': os.path.join(
                 TESTDATADIR, 'SAM/wind_gen_standard_losses_1.json')}
    pp_json = ProjectPoints(fpp, sam_files, 'windpower')

    sam_configs = {k: safe_json_load(v) for k, v in sam_files.items()}
    pp_dict = ProjectPoints(fpp, sam_configs, 'windpower')

    assert pp_json.sam_inputs == pp_dict.sam_inputs


def test_bad_sam_configs():
    """
    Test supplying SAM config as a JSON or a dict
    """
    fpp = os.path.join(TESTDATADIR, 'project_points/pp_offshore.csv')
    sam_files = {'onshore': os.path.join(
                 TESTDATADIR, 'SAM/wind_gen_standard_losses_0.json')}
    # Assert ProjecPoints fails with unequal config entries in points
    # vs sam_configs (as files)
    with pytest.raises(ConfigError):
        ProjectPoints(fpp, sam_files, 'windpower')

    sam_configs = {k: safe_json_load(v) for k, v in sam_files.items()}
    # Assert ProjecPoints fails with unequal config entries in points
    # vs sam_configs (as dicts)
    with pytest.raises(ConfigError):
        ProjectPoints(fpp, sam_configs, 'windpower')

    sites = slice(0, 100)
    sam_file = os.path.join(TESTDATADIR, 'SAM/wind_gen_standard_losses_0.csv')
    # Assert SAMConfig fails when the SAM config is not a json
    with pytest.raises(IOError):
        ProjectPoints(sites, sam_file, 'windpower')

    sites = slice(0, 100)
    sam_file = os.path.join(TESTDATADIR, 'SAM/wind_gen_standard_losses_0.json')
    sam_config = safe_json_load(sam_file)
    # Assert SAMConfig fails when supplying a raw SAM config dictionary.
    # The SAM config dict should be mapped to a config ID
    with pytest.raises(RuntimeError):
        ProjectPoints(sites, sam_config, 'windpower')

    fpp = os.path.join(TESTDATADIR, 'project_points/pp_offshore.csv')
    sam_files = [os.path.join(TESTDATADIR,
                              'SAM/wind_gen_standard_losses_0.json'),
                 os.path.join(TESTDATADIR,
                              'SAM/wind_gen_standard_losses_1.json')]
    # Assert ProjecPoints fails with a list of configs is provided instead
    # of a dictionary mapping the config files to config IDs
    with pytest.raises(ValueError):
        ProjectPoints(fpp, sam_files, 'windpower')


def test_nested_sites():
    """
    Test check for nested points list
    """
    res_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    with pytest.raises(RuntimeError):
        points = [[1, 2, 3, 5]]
        sam_file = os.path.join(TESTDATADIR,
                                'SAM/wind_gen_standard_losses_0.json')
        ProjectPoints(points, sam_file, 'windpower', res_file)


def test_project_points_h():
    """
    Test hub heights in project points
    """
    res_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    points = [1, 2, 3, 5]
    sam_file = os.path.join(TESTDATADIR,
                            'SAM/wind_gen_standard_losses_0.json')
    assert ProjectPoints(points, sam_file, 'pvwattsv8', res_file).h is None

    pp = ProjectPoints(points, sam_file, 'windpower', res_file)
    assert pp.h == [80] * 4


def test_project_points_d():
    """
    Test depth in project points
    """
    points = [1, 2, 3, 5]
    sam_file = os.path.join(TESTDATADIR, 'SAM/geothermal_default.json')
    assert ProjectPoints(points, sam_file, 'windpower').d is None

    pp = ProjectPoints(points, sam_file, 'geothermal')
    assert pp.d == [4500] * 4

    depths_in_data = list(range(len(pp.df)))
    pp = ProjectPoints(points, sam_file, 'geothermal')
    pp.df['resource_depth'] = depths_in_data

    assert pp.d == depths_in_data


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
