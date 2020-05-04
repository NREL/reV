# -*- coding: utf-8 -*-
"""Curtailment utility methods.

Created on Fri Mar  1 13:47:30 2019

@author: gbuster
"""
import numpy as np
from warnings import warn

from reV.utilities.exceptions import HandlerWarning

from rex.utilities.solar_position import SolarPosition


def curtail(resource, curtailment, random_seed=0):
    """Curtail the SAM wind resource object based on project points.

    Parameters
    ----------
    resource : rex.sam_resource.SAMResource
        SAM resource object for WIND resource.
    curtailment : reV.config.curtailment.Curtailment
        Curtailment config object.
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

    shape = resource.shape

    # start with curtailment everywhere
    curtail_mult = np.zeros(shape)

    # Curtail resource when in curtailment months
    mask = np.isin(resource.time_index.month, curtailment.months)
    mask = np.tile(np.expand_dims(mask, axis=1), shape[1])
    curtail_mult = np.where(mask, curtail_mult, 1)

    # Curtail resource when curtailment is possible and is nighttime
    sza = SolarPosition(
        resource.time_index,
        resource.meta[['latitude', 'longitude']].values).zenith
    mask = (sza > curtailment.dawn_dusk)
    curtail_mult = np.where(mask, curtail_mult, 1)

    # Curtail resource when curtailment is possible and not raining
    if curtailment.precipitation:
        if 'precipitationrate' not in resource._res_arrays:
            warn('Curtailment has a precipitation threshold of "{}", but '
                 '"precipitationrate" was not found in the SAM resource '
                 'variables. The following resource variables were '
                 'available: {}.'
                 .format(curtailment.precipitation,
                         list(resource._res_arrays.keys())),
                 HandlerWarning)
        else:
            mask = (resource._res_arrays['precipitationrate']
                    < curtailment.precipitation)
            curtail_mult = np.where(mask, curtail_mult, 1)

    # Curtail resource when curtailment is possible and temperature is high
    if curtailment.temperature:
        mask = (resource._res_arrays['temperature']
                > curtailment.temperature)
        curtail_mult = np.where(mask, curtail_mult, 1)

    # Curtail resource when curtailment is possible and not that windy
    mask = (resource._res_arrays['windspeed']
            < curtailment.wind_speed)
    curtail_mult = np.where(mask, curtail_mult, 1)

    # Apply probability mask when curtailment is possible.
    if curtailment.probability != 1:
        np.random.seed(seed=random_seed)
        mask = np.random.rand(shape[0], shape[1]) < curtailment.probability
        curtail_mult = np.where(mask, curtail_mult, 1)

    # Apply curtailment multiplier directly to resource
    resource.curtail_windspeed(resource.sites, curtail_mult)

    return resource
