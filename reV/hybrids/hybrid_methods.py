# -*- coding: utf-8 -*-
"""Collection of functions used to hybridize columns in rep profiles meta.

@author: ppinchuk
"""


def aggregate_solar_capacity(h):
    """Compute the total solar capcity allowed in hybridization.

    Note
    ----
    No limiting is done on the ratio of wind to solar. This method
    checks for an existing 'hybrid_solar_capacity'. If one does not exist,
    it is assumed that there is no limit on the solar to wind capacity
    ratio and the solar capacity is copied into this new column.

    Parameters
    ----------
    h : `reV.hybrids.Hybridization`
        Instance of `reV.hybrids.Hybridization` class containing the
        attribute `hybrid_meta`, which is a DataFrame containing
        hybridized meta data.

    Returns
    -------
    data : Series | None
        A series of data containing the capacity allowed in the hybrid
        capacity sum, or `None` if 'hybrid_solar_capacity' already exists.

    Notes
    -----

    """
    if 'hybrid_solar_capacity' in h.hybrid_meta:
        return None
    return h.hybrid_meta['solar_capacity']


def aggregate_wind_capacity(h):
    """Compute the total wind capcity allowed in hybridization.

    Note
    ----
    No limiting is done on the ratio of wind to solar. This method
    checks for an existing 'hybrid_wind_capacity'. If one does not exist,
    it is assumed that there is no limit on the solar to wind capacity
    ratio and the wind capacity is copied into this new column.

    Parameters
    ----------
    h : `reV.hybrids.Hybridization`
        Instance of `reV.hybrids.Hybridization` class containing the
        attribute `hybrid_meta`, which is a DataFrame containing
        hybridized meta data.

    Returns
    -------
    data : Series | None
        A series of data containing the capacity allowed in the hybrid
        capacity sum, or `None` if 'hybrid_solar_capacity' already exists.

    Notes
    -----

    """
    if 'hybrid_wind_capacity' in h.hybrid_meta:
        return None
    return h.hybrid_meta['wind_capacity']


def aggregate_capacity(h):
    """Compute the total capcity by summing the individual capacities.

    Parameters
    ----------
    h : `reV.hybrids.Hybridization`
        Instance of `reV.hybrids.Hybridization` class containing the
        attribute `hybrid_meta`, which is a DataFrame containing
        hybridized meta data.

    Returns
    -------
    data : Series | None
        A series of data containing the aggregated capacity, or `None`
        if the capacity columns are missing.
    """

    sc, wc = 'hybrid_solar_capacity', 'hybrid_wind_capacity'
    missing_solar_cap = sc not in h.hybrid_meta.columns
    missing_wind_cap = wc not in h.hybrid_meta.columns
    if missing_solar_cap or missing_wind_cap:
        return None

    total_cap = h.hybrid_meta[sc] + h.hybrid_meta[wc]
    return total_cap


def aggregate_capacity_factor(h):
    """Compute the capacity-weighted mean capcity factor.

    Parameters
    ----------
    h : `reV.hybrids.Hybridization`
        Instance of `reV.hybrids.Hybridization` class containing the
        attribute `hybrid_meta`, which is a DataFrame containing
        hybridized meta data.

    Returns
    -------
    data : Series | None
        A series of data containing the aggregated capacity, or `None`
        if the capacity and/or mean_cf columns are missing.
    """

    sc, wc = 'hybrid_solar_capacity', 'hybrid_wind_capacity'
    scf, wcf = 'solar_mean_cf', 'wind_mean_cf'
    missing_solar_cap = sc not in h.hybrid_meta.columns
    missing_wind_cap = wc not in h.hybrid_meta.columns
    missing_solar_mean_cf = scf not in h.hybrid_meta.columns
    missing_wind_mean_cf = wcf not in h.hybrid_meta.columns
    missing_any = (missing_solar_cap or missing_wind_cap
                   or missing_solar_mean_cf or missing_wind_mean_cf)
    if missing_any:
        return None

    solar_cf_weighted = h.hybrid_meta[sc] * h.hybrid_meta[scf]
    wind_cf_weighted = h.hybrid_meta[wc] * h.hybrid_meta[wcf]
    total_capacity = aggregate_capacity(h)
    hybrid_cf = (solar_cf_weighted + wind_cf_weighted) / total_capacity
    return hybrid_cf


HYBRID_METHODS = {
    'hybrid_solar_capacity': aggregate_solar_capacity,
    'hybrid_wind_capacity': aggregate_wind_capacity,
    'hybrid_capacity': aggregate_capacity,
    'hybrid_mean_cf': aggregate_capacity_factor
}
