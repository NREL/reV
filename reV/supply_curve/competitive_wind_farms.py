# -*- coding: utf-8 -*-
"""
Competitive Wind Farms exclusion handler
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CompetitiveWindFarms:
    """
    Handle competitive wind farm exclusion during supply curve sorting
    """

    def __init__(self, wind_dirs, sc_points, n_dirs=2, offshore=False):
        """
        Parameters
        ----------
        wind_dirs : pandas.DataFrame | str
            path to .csv or reVX.wind_dirs.wind_dirs.WindDirs output with
            the neighboring supply curve point gids and power-rose value at
            each cardinal direction
        sc_points : pandas.DataFrame | str
            Supply curve point summary table
        n_dirs : int, optional
            Number of prominent directions to use, by default 2
        offshore : bool
            Flag as to whether offshore farms should be included during
            CompetitiveWindFarms
        """
        self._wind_dirs = self._parse_wind_dirs(wind_dirs)

        self._sc_gids, self._sc_point_gids, self._mask = \
            self._parse_sc_points(sc_points, offshore=offshore)

        valid = np.isin(self.sc_point_gids, self._wind_dirs.index)
        if not np.all(valid):
            msg = ("'sc_points contains sc_point_gid values that do not "
                   "correspond to valid 'wind_dirs' sc_point_gids:\n{}"
                   .format(self.sc_point_gids[~valid]))
            logger.error(msg)
            raise RuntimeError(msg)

        mask = self._wind_dirs.index.isin(self._sc_point_gids.keys())
        self._wind_dirs = self._wind_dirs.loc[mask]
        self._upwind, self._downwind = self._get_neighbors(self._wind_dirs,
                                                           n_dirs=n_dirs)

    def __repr__(self):
        gids = len(self._upwind)
        neighbors = len(self._upwind.values[0])
        msg = ("{} with {} sc_point_gids and {} prominent directions"
               .format(self.__class__.__name__, gids, neighbors))

        return msg

    def __getitem__(self, keys):
        """
        Map gid for given mapping

        Parameters
        ----------
        keys : tuple
            (gid(s) to extract, gid) pair

        Returns
        -------
        gid(s) : int | list
            Mapped gid(s) for given mapping
        """
        if not isinstance(keys, tuple):
            msg = ("{} must be a tuple of form (source, gid) where source is: "
                   "'sc_gid', 'sc_point_gid',  or 'upwind', 'downwind'"
                   .format(keys))
            logger.error(msg)
            raise ValueError(msg)

        source, gid = keys
        if source == 'sc_point_gid':
            out = self.map_sc_gid_to_sc_point_gid(gid)
        elif source == 'sc_gid':
            out = self.map_sc_point_gid_to_sc_gid(gid)
        elif source == 'upwind':
            out = self.map_upwind(gid)
        elif source == 'downwind':
            out = self.map_downwind(gid)
        else:
            msg = ("{} must be: 'sc_gid', 'sc_point_gid',  or 'upwind', "
                   "'downwind'".format(source))
            logger.error(msg)
            raise ValueError(msg)

        return out

    @property
    def mask(self):
        """
        Supply curve point boolean mask, used for efficient exclusion
        False == excluded sc_point_gid

        Returns
        -------
        ndarray
        """
        return self._mask

    @property
    def sc_point_gids(self):
        """
        Un-masked sc_point_gids

        Returns
        -------
        ndarray
        """
        sc_point_gids = np.array(list(self._sc_point_gids.keys()), dtype=int)
        mask = self.mask[sc_point_gids]

        return sc_point_gids[mask]

    @property
    def sc_gids(self):
        """
        Un-masked sc_gids

        Returns
        -------
        ndarray
        """
        sc_gids = \
            np.concatenate([self._sc_point_gids[gid]
                            for gid in self.sc_point_gids])

        return sc_gids

    @staticmethod
    def _parse_table(table):
        """
        Extract features and their capacity from supply curve transmission
        mapping table

        Parameters
        ----------
        table : str | pd.DataFrame
            Path to .csv or .json or DataFrame to parse

        Returns
        -------
        table : pandas.DataFrame
            DataFrame extracted from file path
        """
        if isinstance(table, str):
            if table.endswith('.csv'):
                table = pd.read_csv(table)
            else:
                raise ValueError('Cannot parse {}'.format(table))

        elif not isinstance(table, pd.DataFrame):
            raise ValueError("Table must be a .csv, or "
                             "a pandas DataFrame")

        return table

    @staticmethod
    def _parse_wind_dirs(wind_dirs):
        """
        Parse prominent direction neighbors

        Parameters
        ----------
        wind_dirs : pandas.DataFrame | str
            Neighboring supply curve point gids and power-rose value at each
            cardinal direction

        Returns
        -------
        wind_dirs : pandas.DataFrame
            Neighboring supply curve point gids and power-rose value at each
            cardinal direction for each sc point gid
        """
        wind_dirs = CompetitiveWindFarms._parse_table(wind_dirs)

        wind_dirs = wind_dirs.set_index('sc_point_gid')
        columns = [c for c in wind_dirs if c.endswith(('_gid', '_pr'))]
        wind_dirs = wind_dirs[columns]

        return wind_dirs

    @staticmethod
    def _parse_sc_points(sc_points, offshore=False):
        """
        Parse supply curve point summary table into sc_gid to sc_point_gid
        mapping and vis-versa.

        Parameters
        ----------
        sc_points : pandas.DataFrame | str
            Supply curve point summary table
        offshore : bool
            Flag as to whether offshore farms should be included during
            CompetitiveWindFarms

        Returns
        -------
        sc_gids : pandas.DataFrame
            sc_gid to sc_point_gid mapping
        sc_point_gids : pandas.DataFrame
            sc_point_gid to sc_gid mapping
        mask : ndarray
            Mask array to mask excluded sc_point_gids
        """
        sc_points = CompetitiveWindFarms._parse_table(sc_points)
        if 'offshore' in sc_points and not offshore:
            logger.debug('Not including offshore supply curve points in'
                         'CompetitiveWindFarm')
            mask = sc_points['offshore'] == 0
            sc_points = sc_points.loc[mask]

        sc_points = sc_points[['sc_gid', 'sc_point_gid']]
        sc_gids = sc_points.set_index('sc_gid')
        sc_gids = {k: int(v[0]) for k, v in sc_gids.iterrows()}

        sc_point_gids = \
            sc_points.groupby('sc_point_gid')['sc_gid'].unique().to_frame()
        sc_point_gids = {int(k): v['sc_gid']
                         for k, v in sc_point_gids.iterrows()}

        mask = np.ones(int(1 + sc_points['sc_point_gid'].max()), dtype=bool)

        return sc_gids, sc_point_gids, mask

    @staticmethod
    def _get_neighbors(wind_dirs, n_dirs=2):
        """
        Parse prominent direction neighbors

        Parameters
        ----------
        wind_dirs : pandas.DataFrame | str
            Neighboring supply curve point gids and power-rose value at each
            cardinal direction for each available sc point gid
        n_dirs : int, optional
            Number of prominent directions to use, by default 2

        Returns
        -------
        upwind : pandas.DataFrame
            Upwind neighbor gids for n prominent wind directions
        downwind : pandas.DataFrame
            Downwind neighbor gids for n prominent wind directions
        """
        cols = [c for c in wind_dirs
                if (c.endswith('_gid') and not c.startswith('sc'))]
        directions = [c.split('_')[0] for c in cols]
        upwind_gids = wind_dirs[cols].values

        cols = ['{}_pr'.format(d) for d in directions]
        neighbor_pr = wind_dirs[cols].values

        neighbors = np.argsort(neighbor_pr)[:, :n_dirs]
        upwind_gids = np.take_along_axis(upwind_gids, neighbors, axis=1)

        downwind_map = {'N': 'S', 'NE': 'SW', 'E': 'W', 'SE': 'NW', 'S': 'N',
                        'SW': 'NE', 'W': 'E', 'NW': 'SE'}
        cols = ["{}_gid".format(downwind_map[d]) for d in directions]
        downwind_gids = wind_dirs[cols].values
        downwind_gids = np.take_along_axis(downwind_gids, neighbors, axis=1)

        downwind = {}
        upwind = {}
        for i, gid in enumerate(wind_dirs.index.values):
            downwind[gid] = downwind_gids[i]
            upwind[gid] = upwind_gids[i]

        return upwind, downwind

    def map_sc_point_gid_to_sc_gid(self, sc_point_gid):
        """
        Map given sc_point_gid to equivalent sc_gid(s)

        Parameters
        ----------
        sc_point_gid : int
            Supply curve point gid to map to equivalent supply curve gid(s)

        Returns
        -------
        int | list
            Equivalent supply curve gid(s)
        """
        return self._sc_point_gids[sc_point_gid]

    def map_sc_gid_to_sc_point_gid(self, sc_gid):
        """
        Map given sc_gid to equivalent sc_point_gid

        Parameters
        ----------
        sc_gid : int
            Supply curve gid to map to equivalent supply point curve gid

        Returns
        -------
        int
            Equivalent supply point curve gid
        """
        return self._sc_gids[sc_gid]

    def check_sc_gid(self, sc_gid):
        """
        Check to see if sc_gid is valid, if so return associated
        sc_point_gids

        Parameters
        ----------
        sc_gid : int
            Supply curve gid to map to equivalent supply point curve gid

        Returns
        -------
        int | None
            Equivalent supply point curve gid or None if sc_gid is invalid
            (offshore)
        """
        sc_point_gid = None
        if sc_gid in self._sc_gids:
            sc_point_gid = self._sc_gids[sc_gid]

        return sc_point_gid

    def map_upwind(self, sc_point_gid):
        """
        Map given sc_point_gid to upwind neighbors

        Parameters
        ----------
        sc_point_gid : int
            Supply point curve gid to get upwind neighbors
        Returns
        -------
        int | list
            upwind neighborings
        """
        return self._upwind[sc_point_gid]

    def map_downwind(self, sc_point_gid):
        """
        Map given sc_point_gid to downwind neighbors

        Parameters
        ----------
        sc_point_gid : int
            Supply point curve gid to get downwind neighbors
        Returns
        -------
        int | list
            downwind neighborings
        """
        return self._downwind[sc_point_gid]

    def exclude_sc_point_gid(self, sc_point_gid):
        """
        Exclude supply curve point gid, return False if gid is not present
        in list of available gids to avoid key errors elsewhere

        Parameters
        ----------
        sc_point_gid : int
            supply curve point gid to mask

        Returns
        -------
        bool
            Flag if gid is valid and was masked
        """
        if sc_point_gid in self._sc_point_gids:
            self._mask[sc_point_gid] = False
            out = True
        else:
            out = False

        return out

    def remove_noncompetitive_farm(self, sc_points, sort_on='total_lcoe',
                                   downwind=False):
        """
        Remove neighboring sc points for given number of prominent wind
        directions

        Parameters
        ----------
        sc_points : pandas.DataFrame | str
            Supply curve point summary table
        sort_on : str, optional
            column to sort on before excluding neighbors,
            by default 'total_lcoe'
        downwind : bool, optional
            Flag to remove downwind neighbors as well as upwind neighbors,
            by default False

        Returns
        -------
        sc_points : pandas.DataFrame
            Updated supply curve points after removing non-competative
            wind farms
        """
        sc_points = self._parse_table(sc_points)
        sc_points = sc_points.sort_values(sort_on)

        sc_point_gids = sc_points['sc_point_gid'].values.astype(int)

        for i in range(len(sc_points)):
            gid = sc_point_gids[i]
            if self.mask[gid]:
                upwind_gids = self['upwind', gid]
                for n in upwind_gids:
                    self.exclude_sc_point_gid(n)

                if downwind:
                    downwind_gids = self['downwind', gid]
                    for n in downwind_gids:
                        self.exclude_sc_point_gid(n)

        sc_gids = self.sc_gids
        mask = sc_points['sc_gid'].isin(sc_gids)

        return sc_points.loc[mask].reset_index(drop=True)

    @classmethod
    def run(cls, wind_dirs, sc_points, n_dirs=2, sort_on='total_lcoe',
            downwind=False, out_fpath=None):
        """
        Exclude given number of neighboring Supply Point gids based on most
        prominent wind directions

        Parameters
        ----------
        wind_dirs : pandas.DataFrame | str
            path to .csv or reVX.wind_dirs.wind_dirs.WindDirs output with
            the neighboring supply curve point gids and power-rose value at
            each cardinal direction
        sc_points : pandas.DataFrame | str
            Supply curve point summary table
        n_dirs : int, optional
            Number of prominent directions to use, by default 2
        sort_on : str, optional
            column to sort on before excluding neighbors,
            by default 'total_lcoe'
        downwind : bool, optional
            Flag to remove downwind neighbors as well as upwind neighbors,
            by default False
        out_fpath : str, optional
            Path to .csv file to save updated sc_points to,
            by default None

        Returns
        -------
        sc_points : pandas.DataFrame
            Updated supply curve points after removing non-competative
            wind farms
        """
        cwf = cls(wind_dirs, sc_points, n_dirs=n_dirs)
        sc_points = cwf.remove_noncompetitive_farm(sc_points, sort_on=sort_on,
                                                   downwind=downwind)

        if out_fpath is not None:
            sc_points.to_csv(out_fpath, index=False)

        return sc_points
