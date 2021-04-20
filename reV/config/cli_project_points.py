# -*- coding: utf-8 -*-
"""
Project Points CLI
"""
import click
import logging

from reV.config.project_points import ProjectPoints
from reV.utilities.exceptions import ProjectPointsValueError
from reV.utilities import log_versions
from reV import __version__

from rex.utilities.cli_dtypes import STR
from rex.utilities.loggers import init_logger
from rex.utilities.utilities import (dict_str_load, safe_json_load)

logger = logging.getLogger(__name__)


def _parse_lat_lons(lat_lon_fpath, lat_lon_coords):
    """
    Parse CLI latitude and longitude inputs for ProjectPoints

    Parameters
    ----------
    lat_lon_fpath : str
        File path to .csv or .json containing latitude, longitude coordinates
        of interest
    lat_lon_coords : tuple
        (lat, lon) coordinates of interest

    Returns
    -------
    lat_lons : str | tuple
        File path to .csv or .json containing latitude, longitude coordinates
        of single set of coordinates to convert to ProjectPoints
    """
    lat_lons = None
    msg = None
    if lat_lon_fpath is not None:
        lat_lons = lat_lon_fpath

    if lat_lon_coords:
        if lat_lons is None:
            lat_lons = lat_lon_coords
        else:
            msg = ("Both a 'lat-lon-fpath' and a pair of 'lat-lon-coords' "
                   "were supplied, but only one can be used to create "
                   "ProjectPoints!")

    if lat_lons is None:
        msg = ("A 'lat-lon-fpath' or a pair of 'lat-lon-coords' must be "
               "supplied in order to create ProjectPoints!")

    if msg is not None:
        logger.error(msg)
        raise ProjectPointsValueError(msg)

    return lat_lons


def _parse_regions(regions, region, region_col):
    """
    Parse CLI regions inputs for ProjectPoints

    Parameters
    ----------
    regions : str
        json string or file path to .json containing regions of
        interest in the form {'region': 'region_column'}
    region : str
        Region to extract
    region_col : str
        Meta column to search for region

    Returns
    -------
    regions : dict
        Dictionary of region_col: regions to generate project points
    """
    msg = None
    if regions is not None:
        if regions.endwtih('.json'):
            regions = safe_json_load(regions)
        else:
            regions = dict_str_load(regions)

    if region is not None:
        if regions is None:
            regions = {region: region_col}
        else:
            if region in regions:
                msg = ("Multiple values for {}: {} were provided!"
                       .format(region, region_col))
            else:
                regions.update({region: region_col})

    if regions is None:
        msg = ("At least a single 'region' and 'region-col' must be "
               "supplied in order to create ProjectPoints!")

    if msg is not None:
        logger.error(msg)
        raise ProjectPointsValueError(msg)

    return regions


@click.group()
@click.version_option(version=__version__)
@click.option('--fpath', '-f', type=click.Path(), required=True,
              help='.csv file path to save project points to (required)')
@click.option('--res_file', '-rf', required=True,
              help=('Filepath to single resource file, multi-h5 directory, '
                    'or /h5_dir/prefix*suffix (required)'))
@click.option('--sam_file', '-sf', required=True,
              type=click.Path(exists=True), help='SAM config file (required)')
@click.option('--verbose', '-v', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, fpath, res_file, sam_file, verbose):
    """reV ProjectPoints generator"""
    ctx.ensure_object(dict)
    ctx.obj['FPATH'] = fpath
    ctx.obj['RES_FILE'] = res_file
    ctx.obj['SAM_FILE'] = sam_file
    ctx.obj['VERBOSE'] = verbose

    if verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    init_logger('reV.config.project_points', log_level=log_level)
    log_versions(logger)


@main.command()
@click.option('--lat_lon_fpath', '-llf', type=click.Path(exists=True),
              default=None,
              help=('File path to .csv or .json containing latitude, '
                    'longitude coordinates of interest'))
@click.option('--lat_lon_coords', '--llc', nargs=2, type=float, default=None,
              help='(lat, lon) coordinates of interest')
@click.pass_context
def from_lat_lons(ctx, lat_lon_fpath, lat_lon_coords):
    """Convert latitude and longitude coordinates to ProjectPoints"""
    lat_lons = _parse_lat_lons(lat_lon_fpath, lat_lon_coords)
    logger.info('Creating ProjectPoints from {} and saving to {}'
                .format(lat_lons, ctx.obj['FPATH']))
    pp = ProjectPoints.lat_lon_coords(lat_lons, ctx.obj['RES_FILE'],
                                      ctx.obj['SAM_FILE'])
    pp.df.to_csv(ctx.obj['FPATH'])


@main.command()
@click.option('--regions', '-regs', type=STR, default=None,
              help=('json string or file path to .json containing regions of '
                    'interest containing regions of interest'))
@click.option('--region', '-r', type=STR, default=None,
              help='Region to extract')
@click.option('--region_col', '-col', type=STR, default='state',
              help='Meta column to search for region')
@click.pass_context
def from_regions(ctx, regions, region, region_col):
    """Extract ProjectPoints for given geographic regions"""
    regions = _parse_regions(regions, region, region_col)
    logger.info('Creating ProjectPoints from {} and saving to {}'
                .format(regions, ctx.obj['FPATH']))
    pp = ProjectPoints.regions(regions, ctx.obj['RES_FILE'],
                               ctx.obj['SAM_FILE'])
    pp.df.to_csv(ctx.obj['FPATH'])


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV ProjecPoints CLI')
        raise
