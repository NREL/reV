"""
Module to compute solar zenith angle outside of SAM
"""
import numpy as np
import pandas as pd


def parse_time(time_index):
    """
    Convert UTC datetime index into:
    - Days since Greenwhich Noon
    - Zulu hour

    Parameters
    ----------
    time_index : pandas.DatetimeIndex

    Returns
    -------
    n : ndarray
        Days since Greenwich Noon
    zulu : ndarray
        Decimal hour in UTC (Zulu Hour)
    """
    if not isinstance(time_index, pd.DatetimeIndex):
        time_index = pd.to_datetime(time_index)

    n = (time_index.to_julian_date() - 2451545).values
    zulu = (time_index.hour + time_index.minute / 60).values

    return n, zulu


def calc_right_ascension(eclong, oblqec):
    """
    Compute Right Ascension angle in radians

    Parameters
    ----------
    eclong : ndarray
        Ecliptic longitude in radians
    oblqec : ndarray
        Obliquity of ecliptic in radians

    Returns
    -------
    ra : ndarray
        Right Ascension angle in radians
    """
    num = np.cos(oblqec) * np.sin(eclong)
    den = np.cos(eclong)
    ra = np.arctan(num / den)

    num[den < 0] = 0
    ra[den < 0] += np.pi
    ra[num < 0] += 2 * np.pi
    return ra


def calc_sun_pos(n):
    """
    Compute right ascension and declination angles of the sun in radians

    Parameters
    ----------
    n : ndarray
        Days since Grenwich Noon

    Returns
    -------
    ra : ndarray
        Right ascension angle of the sun in radians
    dec : ndarray
        Declination angle of the sun in radians
    """
    # Mean Longitude in degrees
    mnlong = np.remainder(280.460 + 0.9856474 * n, 360)
    # Mean anomaly in radians
    mnanom = np.radians(np.remainder(357.528 + 0.9856003 * n, 360))
    # Ecliptic longitude in radians
    eclong = mnlong + 1.915 * np.sin(mnanom) + 0.02 * np.sin(2 * mnanom)
    eclong = np.radians(np.remainder(eclong, 360))
    # Obliquity of ecliptic in radians
    oblqec = np.radians(23.439 - 0.0000004 * n)
    # Right ascension angle in radians
    ra = calc_right_ascension(eclong, oblqec)
    # Declination angle in radians
    dec = np.arcsin(np.sin(oblqec) * np.sin(eclong))

    return ra, dec


def calc_hour_angle(n, zulu, ra, lon):
    """
    Compute the hour angle of the sun

    Parameters
    ----------
    n : ndarray
        Days since Greenwich Noon
    zulu : ndarray
        Decimal hour in UTC (Zulu Hour)
    ra : ndarray
        Right Ascension angle in radians
    lon : float
        Longitude in degrees

    Returns
    -------
    ha : ndarray
        Hour angle in radians between -pi and pi
    """
    # Greenwich mean sidreal time in hours
    gmst = np.remainder(6.697375 + 0.657098242 * n + zulu, 24)
    # Local mean sidereal time in radians
    lmst = np.radians(np.remainder(gmst + lon / 15, 24) * 15)
    # Hour angle in radians
    ha = lmst - ra
    # Ensure hour angle falls between -pi and pi
    ha[ha < -np.pi] += 2 * np.pi
    ha[ha > np.pi] += -2 * np.pi

    return ha


def calc_elevation(dec, ha, lat):
    """
    Calculate the solar elevation

    Parameters
    ----------
    dec : ndarray
        Declination angle of the sun in radians
    ha : ndarray
        Hour angle in radians
    lat : float
        Latitude in degrees

    Returns
    -------
    elv : ndarray
        Solar elevation in radians
    """
    lat = np.radians(lat)
    arg = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha)
    elv = np.arcsin(arg)

    elv[arg > 1] = np.pi / 2
    elv[arg < -1] = -np.pi / 2

    return elv


def atm_correction(elv):
    """
    Apply atmospheric correction to elevation

    Parameters
    ----------
    elv : ndarray
        Solar elevation in radians

    Returns
    -------
    elv : ndarray
        Atmospheric corrected elevation in radians
    """
    elv = np.degrees(elv)
    refrac = (3.51561 * (0.1594 + 0.0196 * elv + 0.00002 * elv**2) /
                        (1 + 0.505 * elv + 0.0845 * elv**2))
    refrac[elv < -0.56] = 0.56

    elv = np.radians(elv + refrac)
    elv[elv > np.pi / 2] = np.pi / 2

    return elv


def calc_azimuth(dec, ha, lat):
    """
    Calculate the solar azimuth angle

    Parameters
    ----------
    dec : ndarray
        Declination angle of the sun in radians
    ha : ndarray
        Hour angle in radians
    lat : float
        Latitude in degrees

    Returns
    -------
    azm : ndarray
        Solar azimuth in radians
    """
    lat = np.radians(lat)
    elv = calc_elevation(dec, ha, lat)
    arg = ((np.sin(elv) * np.sin(lat) - np.sin(dec)) /
           (np.cos(elv) * np.cos(lat)))

    azm = np.arccos(arg)
    # Assign azzimuth = 180 deg if elv == 90 or -90
    azm[np.cos(elv) == 0] = np.pi
    azm[arg > 1] = 0
    azm[arg < -1] = np.pi

    return azm


def calc_zenith(dec, ha, lat):
    """
    Calculate the solar zenith angle

    Parameters
    ----------
    dec : ndarray
        Declination angle of the sun in radians
    ha : ndarray
        Hour angle in radians
    lat : float
        Latitude in degrees

    Returns
    -------
    zen : ndarray
        Solar azimuth in radians
    """
    lat = np.radians(lat)
    elv = calc_elevation(dec, ha, lat)
    # Atmospheric correct elevation
    elv = atm_correction(elv)

    zen = np.pi / 2 - elv

    return zen


def solar_zenith_angle(time_index, lat, lon):
    """
    Calculate solat zenith angle for given time(s) and position(s)

    Parameters
    ----------
    time_index : pandas.Datetime
        Datatime or DatetimeIndex of interest
    lat : float | ndarray
        Latitude or latitudes of interest
    lon : float | ndarray
        Longitude or longitudes of interest

    Returns
    -------
    sza : ndarray
        Solar zenith angle
    """
    n, zulu = parse_time(time_index)
    ra, dec = calc_sun_pos(n)
    ha = calc_hour_angle(n, zulu, ra, lon)
    sza = calc_zenith(dec, ha, lat)

    return sza
