# -*- coding: utf-8 -*-
"""Curtailment utility methods.

Created on Fri Mar  1 13:47:30 2019

@author: gbuster
"""
import datetime
import logging
import numpy as np
import pandas as pd
from warnings import warn

from reV.utilities.exceptions import HandlerWarning

from rex.utilities.solar_position import SolarPosition
from rex.utilities.utilities import check_tz, get_lat_lon_cols

logger = logging.getLogger(__name__)


def curtail(resource, curtailment, sites, random_seed=0):
    """Curtail the SAM wind resource object based on project points.

    Parameters
    ----------
    resource : rex.sam_resource.SAMResource
        SAM resource object for WIND resource.
    curtailment : reV.config.curtailment.Curtailment
        Curtailment config object.
    sites : list
        List of GID's to apply this curtailment to.
    random_seed : int | NoneType
        Number to seed the numpy random number generator. Used to generate
        reproducable psuedo-random results if the probability of curtailment
        is not set to 1. Numpy random will be seeded with the system time if
        this is None.

    Returns
    -------
    resource : reV.handlers.sam_resource.SAMResource
        Same as the input argument but with the wind speed dataset set to zero
        where curtailment is in effect.
    """

    shape = (resource.shape[0], len(sites))
    site_pos = [resource.sites.index(id) for id in sites]

    # start with curtailment everywhere
    curtail_mult = np.zeros(shape)

    if curtailment.date_range is not None:
        year = resource.time_index.year[0]
        d0 = pd.to_datetime(datetime.datetime(
            month=int(curtailment.date_range[0][:2]),
            day=int(curtailment.date_range[0][2:]),
            year=year), utc=True)
        d1 = pd.to_datetime(datetime.datetime(
            month=int(curtailment.date_range[1][:2]),
            day=int(curtailment.date_range[1][2:]),
            year=year), utc=True)
        time_index = check_tz(resource.time_index)
        mask = (time_index >= d0) & (time_index < d1)
        mask = np.tile(np.expand_dims(mask, axis=1), shape[1])
        curtail_mult = np.where(mask, curtail_mult, 1)

    elif curtailment.months is not None:
        # Curtail resource when in curtailment months
        mask = np.isin(resource.time_index.month, curtailment.months)
        mask = np.tile(np.expand_dims(mask, axis=1), shape[1])
        curtail_mult = np.where(mask, curtail_mult, 1)

    else:
        msg = ('You must specify either months or date_range over '
               'which curtailment is possible!')
        logger.error(msg)
        raise KeyError(msg)

    # Curtail resource when curtailment is possible and is nighttime
    meta = resource["meta", sites]
    lat_lon_cols = get_lat_lon_cols(meta)
    solar_zenith_angle = SolarPosition(resource.time_index,
                                       meta[lat_lon_cols].values).zenith
    mask = (solar_zenith_angle > curtailment.dawn_dusk)
    curtail_mult = np.where(mask, curtail_mult, 1)

    # Curtail resource when curtailment is possible and not raining
    if curtailment.precipitation is not None:
        if 'precipitationrate' not in resource._res_arrays:
            warn('Curtailment has a precipitation threshold of "{}", but '
                 '"precipitationrate" was not found in the SAM resource '
                 'variables. The following resource variables were '
                 'available: {}.'
                 .format(curtailment.precipitation,
                         list(resource._res_arrays.keys())),
                 HandlerWarning)
        else:
            mask = (resource._res_arrays['precipitationrate'][:, site_pos]
                    < curtailment.precipitation)
            curtail_mult = np.where(mask, curtail_mult, 1)

    # Curtail resource when curtailment is possible and temperature is high
    if curtailment.temperature is not None:
        mask = (resource._res_arrays['temperature'][:, site_pos]
                > curtailment.temperature)
        curtail_mult = np.where(mask, curtail_mult, 1)

    # Curtail resource when curtailment is possible and not that windy
    if curtailment.wind_speed is not None:
        mask = (resource._res_arrays['windspeed'][:, site_pos]
                < curtailment.wind_speed)
        curtail_mult = np.where(mask, curtail_mult, 1)

    if curtailment.equation is not None:
        # pylint: disable=W0123,W0612
        wind_speed = resource._res_arrays['windspeed'][:, site_pos]
        temperature = resource._res_arrays['temperature'][:, site_pos]
        if 'precipitationrate' in resource._res_arrays:
            precipitation_rate = (
                resource._res_arrays['precipitationrate'][:, site_pos])
        mask = eval(curtailment.equation)
        curtail_mult = np.where(mask, curtail_mult, 1)

    # Apply probability mask when curtailment is possible.
    if curtailment.probability != 1:
        np.random.seed(seed=random_seed)
        mask = np.random.rand(shape[0], shape[1]) < curtailment.probability
        curtail_mult = np.where(mask, curtail_mult, 1)

    # Apply curtailment multiplier directly to resource
    resource.curtail_windspeed(sites, curtail_mult)

    return resource
