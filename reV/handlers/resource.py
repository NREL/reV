"""
Classes to handle resource data
"""
import h5py
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Resource(object):
    """
    Base class to handle resource .h5 files
    """
    SCALE_ATTR = 'scale_factor'

    def __init__(self, h5_file, unscale=True):
        self._h5_file = h5_file
        self._h5 = h5py.File(self._h5_file, 'r')
        self._unscale = unscale

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self._h5_file)
        return msg

    def __enter__(self):
        """
        Enter method to allow use of with
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        Closes dataset on exiting with
        """
        self.close()

        if type is not None:
            raise

    def __getitem__(self, keys):
        ds = keys[0]
        ds_slice = keys[1:]
        if ds == 'time_index':
            out = self.time_index(*ds_slice)
        elif ds == 'meta':
            out = self.meta(*ds_slice)
        else:
            out = self.get_ds(ds, *ds_slice)

        return out

    def time_index(self, *slice):
        """
        Extract and convert time_index to pandas Datetime Index
        """
        time_index = self._h5['time_index'][slice]
        time_index: np.array
        return pd.to_datetime(time_index.astype(str))

    def meta(self, *slice):
        """
        Extract and convert meta to a pandas DataFrame
        """
        meta = self._h5['meta'][[slice[0]]]
        meta = pd.DataFrame(meta)
        if len(slice) == 2:
            meta = meta[slice[1]]

        return meta

    def get_ds(self, ds_name, *slice):
        """
        Extract data from given dataset
        """
        ds = self._h5[ds_name]
        if self._unscale:
            scale_factor = ds.attrs.get(self.SCALE_ATTR, 1)
        else:
            scale_factor = 1

        return ds[slice] / scale_factor

    def close(self):
        """
        Close h5 instance
        """
        self._h5.close()


class SolarResource(Resource):
    """
    Class to handle Solar Resource .h5 files
    """


class NSRDB(SolarResource):
    """
    Class to handle NSRDB .h5 files
    """
    SCALE_ATTR = 'psm_scale_factor'


class WindResource(Resource):
    """
    Class to handle Wind Resource .h5 files
    """

    @staticmethod
    def power_law_interp(ts_1, h_1, ts_2, h_2, h, mean=True):
        """
        Power-law interpolate/extrapolate time-series data to height h

        Parameters
        ----------
        ts_1 : 'ndarray'
            Time-series array at height h_1
        h_1 : 'int'
            Height corresponding to time-seris ts_1
        ts_2 : 'ndarray'
            Time-series array at height h_2
        h_2 : 'int'
            Height corresponding to time-seris ts_2
        h : 'float'
            Height of desired time-series
        mean : 'bool'
            Calculate average alpha versus point by point alpha

        Returns
        -------
        'ndarray'
            Time-series array at height h
        """
        if h == h_1:
            out = ts_1
        elif h == h_2:
            out = ts_2
        else:
            assert h_1 < h_2, 'Args not passed in ascending order!'
            if mean:
                alpha = (np.log(ts_2.mean() / ts_1.mean()) /
                         np.log(h_2 / h_1))

                if alpha < 0.06:
                    logger.warnings('Alpha is < 0.06', RuntimeWarning)
                elif alpha > 0.6:
                    logger.warnings('Alpha is > 0.6', RuntimeWarning)
            else:
                # Replace zero values for alpha calculation
                ts_1[ts_1 == 0] = 0.001
                ts_2[ts_2 == 0] = 0.001

                alpha = np.log(ts_2 / ts_1) / np.log(h_2 / h_1)
                # The Hellmann exponent varies from 0.06 to 0.6
                alpha[alpha < 0.06] = 0.06
                alpha[alpha > 0.6] = 0.6

            out = ts_1 * (h / h_1)**alpha

        return out

    @staticmethod
    def linear_interp(ts_1, h_1, ts_2, h_2, h):
        """
        Linear interpolate/extrapolate time-series data to height h

        Parameters
        ----------
        ts_1 : 'ndarray'
            Time-series array at height h_1
        h_1 : 'int'
            Height corresponding to time-seris ts_1
        ts_2 : 'ndarray'
            Time-series array at height h_2
        h_2 : 'int'
            Height corresponding to time-seris ts_2
        h : 'float'
            Height of desired time-series

        Returns
        -------
        'ndarray'
            Time-series array at height h
        """
        if h == h_1:
            out = ts_1
        elif h == h_2:
            out = ts_2
        else:
            assert h_1 < h_2, 'Args not passed in ascending order!'
            # Calculate slope for every posiiton in variable arrays
            m = (ts_2 - ts_1) / (h_2 - h_1)
            # Calculate intercept for every position in variable arrays
            b = ts_2 - m * h_2

            out = m * h + b

        return out

    @staticmethod
    def shortest_angle(a0, a1):
        """
        Calculate the shortest angle distance between a0 and a1

        Parameters
        ----------
        a0 : 'object'
            angle 0 in degrees
        a1 : 'object'
            angle 1 in degrees

        Returns
        -------
        'object'
            shortest angle distance between a0 and a1
        """
        da = (a1 - a0) % 360
        return 2 * da % 360 - da

    @staticmethod
    def circular_interp(ts_1, h_1, ts_2, h_2, h):
        """
        Circular interpolate/extrapolate time-series data to height h

        Parameters
        ----------
        ts_1 : 'ndarray'
            Time-series array at height h_1
        h_1 : 'int'
            Height corresponding to time-seris ts_1
        ts_2 : 'ndarray'
            Time-series array at height h_2
        h_2 : 'int'
            Height corresponding to time-seris ts_2
        h : 'float'
            Height of desired time-series

        Returns
        -------
        'ndarray'
            Time-series array at height h
        """
        if h == h_1:
            out = ts_1
        elif h == h_2:
            out = ts_2
        else:
            h_f = (h - h_1) / (h_2 - h_1)

            da = WindResource.shortest_angle(ts_1, ts_2) * h_f
            da = np.sign(da) * (np.abs(da) % 360)

            out = (ts_2 + da) % 360

        return out


class WTK(WindResource):
    """
    Class to handle WTK .h5 files
    """


class ECMWF(Resource):
    """
    Class to handle ECMWF weather forecast .h5 files
    TODO: How to Handle Wind Forecasts vs Solar Forecasts?
    """


class MERRA2(WindResource):
    """
    Class to handle MERRA2 .h5 files
    """


class ERA5(WindResource):
    """
    Class to handle ERA5 .h5 files
    """
