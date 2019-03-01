# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:47:30 2019

@author: gbuster
"""
import numpy as np
from reV.utilities.solar_position import SolarPosition


def curtail(resource, curtailment, random_seed=123):
    """Curtail the SAM wind resource object based on project points.

    Parameters
    ----------
    resource : reV.handlers.sam_resource.SAMResource
        SAM resource object for WIND resource.
    curtailment : reV.config.curtailment.Curtailment
        Curtailment config object.
    random_seed : int
        Number to seed the numpy random number generator. Used to generate
        reproducable psuedo-random results if the probability of curtailment
        is not set to 1.

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
    for var in resource.var_list:
        # look for dataset with precipitation (hub height agnostic)
        if 'precip' in var:
            mask = (resource._res_arrays[var] <
                    curtailment.precipitation)
            curtail_mult = np.where(mask, curtail_mult, 1)
            break

    # Curtail resource when curtailment is possible and temperature is high
    for var in resource.var_list:
        # look for dataset with temperature (hub height agnostic)
        if 'temp' in var:
            mask = (resource._res_arrays[var] > curtailment.temperature)
            curtail_mult = np.where(mask, curtail_mult, 1)
            break

    # Curtail resource when curtailment is possible and not that windy
    for var in resource.var_list:
        # look for dataset with wind speed (hub height agnostic)
        if 'speed' in var:
            wind_speed_var = var
            mask = (resource._res_arrays[var] <
                    curtailment.wind_speed)
            curtail_mult = np.where(mask, curtail_mult, 1)
            break

    # Apply probability mask when curtailment is possible.
    if curtailment.probability != 1:
        np.random.seed(seed=random_seed)
        mask = np.random.rand(shape) < curtailment.probability
        curtail_mult = np.where(mask, curtail_mult, 1)

    # Apply curtailment multiplier directly to resource
    resource._res_arrays[wind_speed_var] *= curtail_mult

    return resource
