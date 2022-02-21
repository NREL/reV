# -*- coding: utf-8 -*-
"""Collection of functions used to hybridize columns in rep profiles meta.

@author: ppinchuk
"""
HYBRID_METHODS = {}


def hybrid_col(col_name):
    """A decorator factory that facilitates the registry of new hybrids.

    This decorator takes a column name as input and registers the decorated
    function as a method that computes a hybrid variable. During the
    hybridization step, the registered function (which takes an instance
    of the hybridization object as input) will be run and its output
    will be stored in the new hybrid meta DataFrame under the registered column
    name.

    Parameters
    ----------
    col_name : str
        Name of the new hybrid column. This should typically start with
        "hybrid_", though this is not a hard requirement.

    Examples
    --------
    Writing and registering a new hybridization:

    >>> @hybrid_col('scaled_elevation')
    >>> def some_new_hybrid_func(h):
    >>>     return h.hybrid_meta['elevation'] * 1000

    You can then verify the correct column was added:

    >>> from reV.hybrids import hybrid_col, Hybridization
    >>> SOLAR_FPATH = '/path/to/input/solar/file.h5
    >>> WIND_FPATH = '/path/to/input/wind/file.h5
    >>>
    >>> h = Hybridization(SOLAR_FPATH, WIND_FPATH).run()
    >>> assert 'scaled_elevation' in h.hybrid_meta.columns

    """
    def _register(func):
        HYBRID_METHODS[col_name] = func
        return func
    return _register


@hybrid_col('hybrid_solar_capacity')
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


@hybrid_col('hybrid_wind_capacity')
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


@hybrid_col('hybrid_capacity')
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


@hybrid_col('hybrid_mean_cf')
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
