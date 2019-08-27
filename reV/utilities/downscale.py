# -*- coding: utf-8 -*-
"""Solar resource downscaling utility methods.

Created on April 8 2019

@author: gbuster
"""
import numpy as np
import pandas as pd
import logging

from reV.utilities.solar_position import SolarPosition

from nsrdb.all_sky import CLEAR_TYPES
from nsrdb.all_sky.all_sky import all_sky
from nsrdb.utilities.interpolation import temporal_lin, temporal_step


logger = logging.getLogger(__name__)


def make_time_index(year, frequency):
    """Make the NSRDB target time index.

    Parameters
    ----------
    year : int
        Year for time index.
    frequency : str
        String in the Pandas frequency format, e.g. '5min'.

    Returns
    -------
    ti : pd.DatetimeIndex
        Pandas datetime index for a full year at the requested resolution.
    """
    ti = pd.date_range('1-1-{y}'.format(y=year), '1-1-{y}'.format(y=year + 1),
                       freq=frequency)[:-1]
    return ti


def interp_cld_props(data, ti_native, ti_new,
                     var_list=('cld_reff_dcomp', 'cld_opd_dcomp')):
    """Interpolate missing cloud properties (NOT CLOUD TYPE).

    Parameters
    ----------
    data : dict
        Namespace of variables for input to all_sky. Must include the cloud
        variables in var_list and "cloud_type".
    ti_native : pd.DateTimeIndex
        Native time index of the original NSRDB data.
    ti_new : pd.DateTimeIndex
        Intended downscaled time index.
    var_list : list | tuple
        Cloud variables to downscale.

    Returns
    -------
    data : dict
        Namespace of variables with the cloud variables in var_list downscaled
        to the requested ti_new.
    """

    for var in var_list:

        # make sparse dataframe with new time_index
        data[var] = pd.DataFrame(data[var], index=ti_native).reindex(ti_new)

        # find location of bad data
        cld_fill_flag = ((data[var] < 0) | data[var].isnull())

        # replace to-fill values with nan
        data[var].values[cld_fill_flag] = np.nan

        # set clear timesteps cloud props to zero for better transitions
        data[var].values[np.isin(data['cloud_type'], CLEAR_TYPES)] = 0.0

        # interpolate empty values
        data[var] = data[var].interpolate(method='linear', axis=0).values

    return data


def downscale_nsrdb(SAM_res, res, project_points, frequency,
                    sam_vars=('dhi', 'dni', 'wind_speed', 'air_temperature'),
                    ghi_variability=0.05):
    """Downscale the NSRDB resource and return the preloaded SAM_res.

    Parameters
    ----------
    SAM_res : SAMResource
        reV SAM resource object.
    res : NSRDB
        reV NSRDB resource handler.
    project_points : ProjectPoints
        reV project points object.
    frequency : str
        String in the Pandas frequency format, e.g. '5min'.
    sam_vars : tuple | list
        Variables to save to SAM resource handler before returning.
    ghi_variability : float
        Maximum GHI synthetic variability fraction.

    Returns
    -------
    SAM_res : SAMResource
        reV SAM resource object with downscaled solar resource data loaded.
        Time index and shape are also updated.
    """

    logger.debug('Downscaling NSRDB resource data to "{}".'.format(frequency))

    # variables required for all-sky not including clouds, ti, sza
    var_list = ('aod',
                'surface_pressure',
                'surface_albedo',
                'ssa',
                'asymmetry',
                'alpha',
                'ozone',
                'total_precipitable_water',
                )

    # Indexing variable
    sites_slice = project_points.sites_as_slice

    # get downscaled time_index
    time_index = make_time_index(res.time_index.year[0], frequency)
    SAM_res._time_index = time_index
    SAM_res._shape = (len(time_index), len(project_points.sites))

    # downscale variables into an all-sky input variable namespace
    all_sky_ins = {'time_index': time_index}
    for var in var_list:
        all_sky_ins[var] = temporal_lin(res[var, :, sites_slice],
                                        res.time_index, time_index)

    # calculate downscaled solar zenith angle
    lat_lon = res.meta.loc[project_points.sites, ['latitude', 'longitude']]\
        .values.astype(np.float32)
    all_sky_ins['solar_zenith_angle'] = SolarPosition(time_index,
                                                      lat_lon).zenith

    # get downscaled cloud properties
    all_sky_ins['cloud_type'] = temporal_step(
        res['cloud_type', :, sites_slice], res.time_index, time_index)
    all_sky_ins['cld_opd_dcomp'] = res['cld_opd_dcomp', :, sites_slice]
    all_sky_ins['cld_reff_dcomp'] = res['cld_reff_dcomp', :, sites_slice]
    all_sky_ins = interp_cld_props(all_sky_ins, res.time_index, time_index)

    # add variability
    all_sky_ins['ghi_variability'] = ghi_variability

    # run all-sky
    logger.debug('Running all-sky for "{}".'.format(project_points))
    all_sky_outs = all_sky(**all_sky_ins)

    # set downscaled data to sam resource handler
    for k, v in all_sky_outs.items():
        if k in sam_vars:
            SAM_res[k] = v

    # downscale extra vars needed for SAM but not for all-sky
    for var in sam_vars:
        if var not in SAM_res._res_arrays:
            SAM_res[var] = temporal_lin(res[var, :, sites_slice],
                                        res.time_index, time_index)

    return SAM_res
