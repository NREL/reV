"""
Module to handle SAM Resource iterator to create site by site resource
DataFrames
"""
import numpy as np
import pandas as pd
from warnings import warn

from reV.utilities.exceptions import (HandlerKeyError, HandlerRuntimeError,
                                      HandlerValueError, SAMInputWarning)


def parse_keys(keys):
    """
    Parse keys for complex __getitem__ and __setitem__

    Parameters
    ----------
    keys : string | tuple
        key or key and slice to extract

    Returns
    -------
    key : string
        key to extract
    key_slice : slice | tuple
        Slice or tuple of slices of key to extract
    """
    if isinstance(keys, tuple):
        key = keys[0]
        key_slice = keys[1:]
    else:
        key = keys
        key_slice = (slice(None, None, None),)

    return key, key_slice


class SAMResource:
    """
    Resource Manager for SAM
    """

    # Resource variables to load for each res type
    RES_VARS = {'solar': ('dni', 'dhi', 'wind_speed', 'air_temperature'),
                'wind': ('pressure', 'temperature', 'winddirection',
                         'windspeed')}

    def __init__(self, project_points, time_index, require_wind_dir=False):
        """
        Parameters
        ----------
        project_points : reV.config.ProjectPoints
            Instance of ProjectPoints
        time_index : pandas.DatetimeIndex
            Time-series datetime index
        require_wind_dir : bool
            Boolean flag indicating that wind direction is required
        """
        self._i = 0
        self._project_points = project_points
        self._time_index = time_index
        self._shape = (len(time_index), len(project_points.sites))
        self._n = self._shape[1]
        self._var_list = None
        self._meta = None
        self._runnable = False
        self._res_arrays = {}
        h = project_points.h
        if h is None:
            # no hub height specified, get NSRDB solar data
            self._res_type = 'solar'
        else:
            # hub height specified, get WTK wind data.
            self._res_type = 'wind'
            if isinstance(h, (list, np.ndarray)):
                if len(h) != self._n:
                    msg = 'Must have a unique height for each site'
                    raise HandlerValueError(msg)
            if not require_wind_dir:
                self._res_arrays['winddirection'] = np.zeros(self._shape,
                                                             dtype='float32')

        self._h = h

    def __repr__(self):
        msg = "{} with {} {} sites".format(self.__class__.__name__,
                                           self._n, self._res_type)
        return msg

    def __len__(self):
        return self._n

    def __getitem__(self, keys):
        var, var_slice = parse_keys(keys)

        if var == 'time_index':
            out = self.time_index
            out = out[var_slice[0]]
        elif var == 'meta':
            out = self.meta
            out = out.loc[var_slice[0]]
        elif isinstance(var, str):
            out = self._get_var_ts(var, *var_slice)
        elif isinstance(var, int):
            site = var
            out, _ = self._get_res_df(site)
        else:
            raise HandlerKeyError('Cannot interpret {}'.format(var))

        return out

    def __setitem__(self, keys, arr):
        var, var_slice = parse_keys(keys)

        if var == 'meta':
            self.meta = arr
        else:
            self._set_var_array(var, arr, *var_slice)

    def __iter__(self):
        return self

    def __next__(self):
        if self._i < self._n:
            site = self.sites[self._i]
            res_df, site_meta = self._get_res_df(site)
            self._i += 1
            return res_df, site_meta
        else:
            raise StopIteration

    @property
    def sites(self):
        """
        Sites being pre-loaded for SAM

        Returns
        -------
        sites : list
            List of sites to be provided to SAM, defined by project_points
        """
        sites = self._project_points.sites
        return list(sites)

    @property
    def shape(self):
        """
        Shape of variable arrays

        Returns
        -------
        self._shape : tuple
            Shape (time_index, sites) of variable arrays
        """
        return self._shape

    @property
    def var_list(self):
        """
        Return variable list associated with SAMResource type

        Returns
        -------
        _var_list : list
            List of resource variables associated with resource type
            ('solar' or 'wind')
        """

        if self._var_list is None:
            if self._res_type in self.RES_VARS:
                self._var_list = list(self.RES_VARS[self._res_type])
            else:
                raise HandlerValueError("Resource type is invalid!")

        return self._var_list

    def append_var_list(self, var):
        """
        Append a new variable to the SAM resource protected var_list.

        Parameters
        ----------
        var : str
            New resource variable to be added to the protected var_list
            property.
        """

        self.var_list.append(var)

    @property
    def time_index(self):
        """
        Return time_index

        Returns
        -------
        self._time_index : pandas.DatetimeIndex
            Time-series datetime index
        """
        return self._time_index

    @property
    def meta(self):
        """
        Return sites meta

        Returns
        -------
        self._meta : pandas.DataFrame
            DataFrame of sites meta data
        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        """
        Set sites meta

        Parameters
        ----------
        meta : recarray | pandas.DataFrame
            Sites meta as records array or DataFrame
        """
        if len(meta) != self._n:
            raise HandlerValueError('Meta does not contain {} sites'
                                    .format(self._n))

        if not isinstance(meta, pd.DataFrame):
            meta = pd.DataFrame(meta, index=self.sites)
        else:
            if not np.array_equal(meta.index, self.sites):
                raise HandlerValueError('Meta does not match sites!')

        self._meta = meta

    @property
    def h(self):
        """
        Get heights for wind sites

        Returns
        -------
        self._h : int | float | list
            Hub height or height(s) for wind resource, None for solar resource
        """
        return self._h

    @staticmethod
    def check_units(var_name, var_array):
        """
        Check units of variable array and convert to SAM units if needed

        Parameters
        ----------
        var_name : str
            Variable name
        var_array : ndarray
            Variable data

        Returns
        -------
        var_array : ndarray
            Variable data with updated units if needed
        """
        if 'pressure' in var_name:
            # Check if pressure is in Pa, if so convert to atm
            if np.min(var_array) > 1e3:
                # convert pressure from Pa to ATM
                var_array *= 9.86923e-6
        elif 'temperature' in var_name:
            # Check if tempearture is in K, if so convert to C
            if np.min(var_array) > 273.15:
                var_array += -273.15

        return var_array

    def check_physical_ranges(self):
        """Check physical ranges and enforce usable SAM data.

        Current methodology sets windspeed=0 if any of the checks are violated.

        SAM ranges are from:
            https://github.com/NREL/ssc/blob/develop/shared/lib_windfile.cpp
            https://github.com/NREL/ssc/blob/develop/ssc/cmod_wfcheck.cpp
        """

        if self._res_type == 'wind':

            # units are in C
            check = ((self._res_arrays['temperature'] < -200) |
                     (self._res_arrays['temperature'] > 100))
            if check.any():
                ibad = check.any(axis=0)
                self._res_arrays['temperature'][:, ibad] = 0
                self._res_arrays['windspeed'][:, ibad] = 0

                warn('Wind resource temp. out of viable SAM range for sites '
                     '{}. Fixing and setting windspeed to zero.'
                     .format(list(np.array(self.sites)[ibad])),
                     SAMInputWarning)

            # units are in atm
            check = ((self._res_arrays['pressure'] < 0.5) |
                     (self._res_arrays['pressure'] > 1.1))
            if check.any():
                ibad = check.any(axis=0)
                self._res_arrays['pressure'][:, ibad] = 1
                self._res_arrays['windspeed'][:, ibad] = 0
                warn('Wind resource press. out of viable SAM range for sites '
                     '{}. Fixing and setting windspeed to zero.'
                     .format(list(np.array(self.sites)[ibad])),
                     SAMInputWarning)

            # units are in m/s
            check = ((self._res_arrays['windspeed'] < 0) |
                     (self._res_arrays['windspeed'] > 120))
            if check.any():
                ibad = check.any(axis=0)
                self._res_arrays['windspeed'][:, ibad] = 0
                warn('Wind resource speed out of viable SAM range for sites '
                     '{}. Setting windspeed to zero.'
                     .format(list(np.array(self.sites)[ibad])),
                     SAMInputWarning)

    def runnable(self):
        """
        Check to see if SAMResource iterator is runnable:
        - Meta must be loaded
        - Variables in var_list must be loaded

        Returns
        ------
        bool
            Returns True if runnable check passes
        """
        if self._meta is None:
            raise HandlerRuntimeError('meta has not been set!')
        else:
            for var in self.var_list:
                if var not in self._res_arrays.keys():
                    raise HandlerRuntimeError('{} has not been set!'
                                              .format(var))

        return True

    def _set_var_array(self, var, arr, *var_slice):
        """
        Set variable array (with unit checking)

        Parameters
        ----------
        var : str
            Resource variable name
        arr : ndarray
            Time series data of given variable for sites
        var_slice : tuple of int | list | slice
            Slice of variable array that corresponds to arr
        """
        if var in self.var_list:
            var_arr = self._res_arrays.get(var, np.zeros(self._shape,
                                                         dtype='float32'))
            if var_arr[var_slice].shape == arr.shape:
                var_arr[var_slice] = arr
                self._res_arrays[var] = self.check_units(var, var_arr)
            else:
                raise HandlerValueError('{} does not have proper shape: {}'
                                        .format(var, self._shape))
        else:
            raise HandlerKeyError('{} not in {}'
                                  .format(var, self.var_list))

    def _get_var_ts(self, var, *var_slice):
        """
        Get variable time-series

        Parameters
        ----------
        var : str
            Resource variable name
        var_slice : tuple of int | list | slice
            Slice of variable array to extract

        Returns
        -------
        ts : pandas.DataFrame
            Time-series for desired sites of variable var
        """
        if var in self.var_list:
            try:
                var_array = self._res_arrays[var]
            except KeyError:
                raise HandlerKeyError('{} has yet to be set!')

            sites = np.array(self.sites)
            ts = pd.DataFrame(var_array[var_slice],
                              index=self.time_index[var_slice[0]],
                              columns=sites[var_slice[1]])
        else:
            raise HandlerKeyError('{} not in {}'
                                  .format(var, self.var_list))

        return ts

    def _get_res_df(self, site):
        """
        Get resource time-series

        Parameters
        ----------
        site : int
            Site to extract

        Returns
        -------
        res_df : pandas.DataFrame
            Time-series of SAM resource variables for given site
        """
        self.runnable()
        try:
            idx = self.sites.index(site)
        except ValueError:
            raise HandlerValueError('{} is not in available sites'
                                    .format(site))
        site_meta = self.meta.loc[site].copy()
        if self._h is not None:
            try:
                h = self._h[idx]
            except TypeError:
                h = self._h

            site_meta['height'] = h

        res_df = pd.DataFrame(index=self.time_index)
        res_df.name = site
        for var_name, var_array in self._res_arrays.items():
            res_df[var_name] = var_array[:, idx]

        return res_df, site_meta

    def curtail_windspeed(self, gids, curtailment):
        """
        Apply temporal curtailment mask to windspeed resource at given sites

        Parameters
        ----------
        gids : int | list
            gids for site or list of sites to curtail
        curtailment : ndarray
            Temporal multiplier for curtailment
        """
        shape = (self.shape[0],)
        if isinstance(gids, int):
            site_pos = self.sites.index(gids)
        else:
            shape += (len(gids),)
            site_pos = [self.sites.index(id) for id in gids]

        if curtailment.shape != shape:
            raise HandlerValueError("curtailment must be of shape: {}"
                                    .format(shape))

        if 'windspeed' in self._res_arrays:
            self._res_arrays['windspeed'][:, site_pos] *= curtailment
        else:
            raise HandlerRuntimeError('windspeed has not be loaded!')
