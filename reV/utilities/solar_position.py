# -*- coding: utf-8 -*-
"""
Module to compute solar zenith angle outside of SAM
"""
import numpy as np
import pandas as pd


class SolarPosition:
    """
    Class to compute solar position for time(s) and site(s)
    Based off of SAM Solar Position Function:
    https://github.com/NREL/ssc/blob/develop/shared/lib_irradproc.cpp
    """
    def __init__(self, time_index, lat_lon):
        """
        Parameters
        ----------
        time_index : ndarray | pandas.DatetimeIndex | str
            Datetime stamps of interest
        lat_lon : ndarray
            (latitude, longitude) for site(s) of interest
        """
        if not isinstance(time_index, pd.DatetimeIndex):
            if isinstance(time_index, str):
                time_index = [time_index]

            time_index = pd.to_datetime(time_index)

        self._time_index = time_index

        if not isinstance(lat_lon, np.ndarray):
            lat_lon = np.array(lat_lon)

        self._lat_lon = np.expand_dims(lat_lon, axis=0).T

    @property
    def time_index(self):
        """
        Datetime stamp(s) of interest

        Returns
        -------
        time_index : pandas.DatetimeIndex
        """
        return self._time_index

    @property
    def latitude(self):
        """
        Latitudes of site(s)

        Returns
        -------
        lat : ndarray
        """
        lat = self._lat_lon[0]
        return lat

    @property
    def longitude(self):
        """
        longitude of site(s)

        Returns
        -------
        lon : ndarray
        """
        lon = self._lat_lon[1]
        return lon

    @staticmethod
    def _parse_time(time_index):
        """
        Convert UTC datetime index into:
        - Days since Greenwhich Noon
        - Zulu hour

        Parameters
        ----------
        time_index : pandas.DatetimeIndex
            Datetime stamps of interest

        Returns
        -------
        n : ndarray
            Days since Greenwich Noon
        zulu : ndarray
            Decimal hour in UTC (Zulu Hour)
        """
        n = (time_index.to_julian_date() - 2451545).values
        zulu = (time_index.hour + time_index.minute / 60).values

        return n, zulu

    @staticmethod
    def _calc_right_ascension(eclong, oblqec):
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
        ra = np.arctan2(num, den)
        return ra

    @staticmethod
    def _calc_sun_pos(n):
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
        ra = SolarPosition._calc_right_ascension(eclong, oblqec)
        # Declination angle in radians
        dec = np.arcsin(np.sin(oblqec) * np.sin(eclong))

        return ra, dec

    @staticmethod
    def _calc_hour_angle(n, zulu, ra, lon):
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
        # Greenwich mean sidereal time in degrees
        gmst = (6.697375 + 0.06570982441908 * n + 1.00273790935 * zulu) * 15
        # Local mean sidereal time in radians
        lmst = np.radians(np.remainder(gmst + lon, 360))
        # Hour angle in radians
        ha = lmst - ra
        # Ensure hour angle falls between -pi and pi
        ha[ha < -np.pi] += 2 * np.pi
        ha[ha > np.pi] += -2 * np.pi

        return ha

    @staticmethod
    def _calc_elevation(dec, ha, lat):
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
        arg = (np.sin(dec) * np.sin(lat)
               + np.cos(dec) * np.cos(lat) * np.cos(ha))
        elv = np.arcsin(arg)

        elv[arg > 1] = np.pi / 2
        elv[arg < -1] = -np.pi / 2

        return elv

    @staticmethod
    def _elevation(time_index, lat, lon):
        """
        Compute solar elevation angle from time_index and location

        Parameters
        ----------
        time_index : pandas.DatetimeIndex
            Datetime stamp(s) of interest
        lat : ndarray
            Latitude of site(s) of interest
        lon : ndarray
            Longitude of site(s) of interest

        Returns
        -------
        elevation : ndarray
            Solar elevation angle in radians
        """
        n, zulu = SolarPosition._parse_time(time_index)
        ra, dec = SolarPosition._calc_sun_pos(n)
        ha = SolarPosition._calc_hour_angle(n, zulu, ra, lon)
        elevation = SolarPosition._calc_elevation(dec, ha, lat)
        return elevation

    @staticmethod
    def _atm_correction(elv):
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
        refrac = (3.51561 * (0.1594 + 0.0196 * elv + 0.00002 * elv**2)
                  / (1 + 0.505 * elv + 0.0845 * elv**2))
        refrac[elv < -0.56] = 0.56

        elv = np.radians(elv + refrac)
        elv[elv > np.pi / 2] = np.pi / 2

        return elv

    @staticmethod
    def _calc_azimuth(dec, ha, lat):
        """
        Calculate the solar azimuth angle from solar position variables

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
        elv = SolarPosition._calc_elevation(dec, ha, lat)
        lat = np.radians(lat)
        arg = ((np.sin(elv) * np.sin(lat) - np.sin(dec))
               / (np.cos(elv) * np.cos(lat)))

        azm = np.arccos(arg)
        # Assign azzimuth = 180 deg if elv == 90 or -90
        azm[np.cos(elv) == 0] = np.pi
        azm[arg > 1] = 0
        azm[arg < -1] = np.pi

        return azm

    @staticmethod
    def _azimuth(time_index, lat, lon):
        """
        Compute solar azimuth angle from time_index and location

        Parameters
        ----------
        time_index : pandas.DatetimeIndex
            Datetime stamp(s) of interest
        lat : ndarray
            Latitude of site(s) of interest
        lon : ndarray
            Longitude of site(s) of interest

        Returns
        -------
        azimuth : ndarray
            Solar azimuth angle in radians
        """
        n, zulu = SolarPosition._parse_time(time_index)
        ra, dec = SolarPosition._calc_sun_pos(n)
        ha = SolarPosition._calc_hour_angle(n, zulu, ra, lon)
        azimuth = SolarPosition._calc_azimuth(dec, ha, lat)
        return azimuth

    @staticmethod
    def _calc_zenith(dec, ha, lat):
        """
        Calculate the solar zenith angle from solar position variables

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
        elv = SolarPosition._calc_elevation(dec, ha, lat)
        # Atmospheric correct elevation
        elv = SolarPosition._atm_correction(elv)

        zen = np.pi / 2 - elv

        return zen

    @staticmethod
    def _zenith(time_index, lat, lon):
        """
        Compute solar zenith angle from time_index and location

        Parameters
        ----------
        time_index : pandas.DatetimeIndex
            Datetime stamp(s) of interest
        lat : ndarray
            Latitude of site(s) of interest
        lon : ndarray
            Longitude of site(s) of interest

        Returns
        -------
        zenith : ndarray
            Solar zenith angle in radians
        """
        n, zulu = SolarPosition._parse_time(time_index)
        ra, dec = SolarPosition._calc_sun_pos(n)
        ha = SolarPosition._calc_hour_angle(n, zulu, ra, lon)
        zenith = SolarPosition._calc_zenith(dec, ha, lat)
        return zenith

    def _format_output(self, arr):
        """
        Format radians array for output:
        - Convert to degrees
        - Transpose if needed

        Parameters
        ----------
        arr : ndarray
            Data array in radians

        Returns
        -------
        arr : ndarray
            Data array in degrees and formatted as (time x sites)
        """
        arr = np.degrees(arr)

        if arr.shape[0] != len(self._time_index):
            arr = arr.T

        return arr

    @property
    def azimuth(self):
        """
        Solar azimuth angle

        Returns
        -------
        azimuth : ndarray
            Solar azimuth angle in degrees
        """
        azimuth = self._azimuth(self.time_index, self.latitude, self.longitude)

        return self._format_output(azimuth)

    @property
    def elevation(self):
        """
        Solar elevation angle

        Returns
        -------
        elevation : ndarray
            Solar elevation angle in degrees
        """
        elevation = self._elevation(self.time_index, self.latitude,
                                    self.longitude)

        return self._format_output(elevation)

    @property
    def apparent_elevation(self):
        """
        Refracted solar elevation angle

        Returns
        -------
        elevation : ndarray
            Solar elevation angle in degrees
        """
        elevation = self._elevation(self.time_index, self.latitude,
                                    self.longitude)
        elevation = self._atm_correction(elevation)

        return self._format_output(elevation)

    @property
    def zenith(self):
        """
        Solar zenith angle

        Returns
        -------
        zenith : ndarray
            Solar zenith angle in degrees
        """
        zenith = self._zenith(self.time_index, self.latitude, self.longitude)

        return self._format_output(zenith)
