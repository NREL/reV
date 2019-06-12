"""
Cluster reV capacity factor profiles
"""
import sys
import pdb
import argparse
import geopandas as gpd
import h5py
import io
import logging
import numpy as np
import os
import pandas as pd
import psutil
import psycopg2
import pysal
import pywt
from datetime import datetime
from multiprocessing import Pool

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] - %(lineno)d - %(levelname)s - %(message)s')
_chandler = logging.StreamHandler()
_chandler.setLevel(logging.DEBUG)
_chandler.setFormatter(formatter)
logger.addHandler(_chandler)


class Clusters:
    """Base class for RPM clusters"""

    BASE_UTC_OFFSET = 0
    HDF_SCALE_FACTOR = 10000.

    def __init__(self):
        """
        Parameters
        ----------

        """


    @staticmethod
    def get_gids_by_region(region, column, limit=None, db_host='gispgdb.nrel.gov', db_name='dav-gis',
                           res_schema='nsrdb', res_table='grid', res_id_column='grid_gid',
                           res_geom_column='the_geom_4326',
                           res_lkup_schema='nsrdb', res_lkup_table='gid_lookup_join',
                           res_lkup_id_column='grid_gid', res_lkup_region_column='ba'):
        """
        Collect NSRDB grid_gid list for <region> from "nsrdb"."gid_lookup_join".

        :param region: [any] region identifier
        :param column: [string] column in "nsrdb"."gid_lookup_join" with matching <region> value
        :param limit: [int] limit result set
        :param db_host: [string] database host name
        :param db_name: [string] database name
        :param res_schema: [string] schema name with <res_table>
        :param res_table: [string] table name with <res_geom_column>
        :param res_id_column: [string] column name with resource polygon ID
        :param res_geom_column: [string] column name with resource polygon geometry
        :param res_lkup_schema: [string] schema name with <res_lkup_table>
        :param res_lkup_table: [string] table name with <res_lkup_column> and <res_lkup_id_column>
        :param res_lkup_id_column: [string] column name with resource polygon geometry ID
        :param res_lkup_region_column: [string] column name with region ID
        :return: [GeoDataFrame] grid_gid values in ascending order
        """

        with psycopg2.connect(host=db_host, dbname=db_name) as _conn:
            _kvals = {'res_lkup_id_column': res_lkup_id_column,
                      'res_geom_column': res_geom_column,
                      'res_lkup_schema': res_lkup_schema,
                      'res_lkup_table': res_lkup_table,
                      'res_schema': res_schema,
                      'res_table': res_table,
                      'res_id_column': res_id_column,
                      'res_lkup_region_column': res_lkup_region_column,
                      'r': region}

            _sql = """SELECT a."{res_lkup_id_column}" AS "gid"
                           , b."{res_geom_column}"    AS "geom"
                      FROM      "{res_lkup_schema}"."{res_lkup_table}" a
                      LEFT JOIN "{res_schema}"."{res_table}"           b ON a."{res_lkup_id_column}"
                                                                          = b."{res_id_column}"
                      WHERE a."{res_lkup_region_column}" = %(r)s
                        AND b."{res_geom_column}" IS NOT NULL
                      ORDER BY a."{res_lkup_id_column}" ASC
                      """.format(**_kvals)

            if limit:
                _sql += ' LIMIT %s' % limit

            _sql += ';'

            logger.debug('collecting grid values from {res_schema}.{res_table}'\
                    ' ({res_id_column}, {res_geom_column})' \
                    ' via {res_lkup_schema}.{res_lkup_table} for {res_lkup_region_column}'\
                    ' = {region}'.format(res_schema=res_schema,
                                        res_table=res_table,
                                        res_id_column=res_id_column,
                                        res_geom_column=res_geom_column,
                                        res_lkup_schema=res_lkup_schema,
                                        res_lkup_table=res_lkup_table,
                                        res_lkup_id_column=res_lkup_id_column,
                                        res_lkup_region_column=res_lkup_region_column,
                                        region=region))

            return gpd.GeoDataFrame.from_postgis(sql=_sql,
                                                 con=_conn,
                                                 geom_col='geom',
                                                 crs=None,
                                                 index_col=None,
                                                 coerce_float=True,
                                                 params={'r': region})


    @staticmethod
    def get_metadata(fpath, resource, datagroup='meta'):
        """
        Collect metadata for <resource> from <fpath>.

        :param fpath: [string] metadata HDF file path
        :param resource: [string] resource name
        :param datagroup: [string] datagroup name
        :return: [DataFrame] DataFrame with 'gid' and 'chunk' columns
                             describing the grid_gid and file chunk values
        """

        with h5py.File(fpath, 'r') as _hfile:
            _df = pd.DataFrame(data=_hfile[resource][datagroup].value,
                               columns=('resource_lkup', 'chunk', 'tz'),
                               dtype=(np.int, np.int))

        return _df.rename(columns={'resource_lkup': 'gid'})


    @staticmethod
    def get_chunks(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        """
       Collect chunk values for <df>['grid_gid'] from <meta>.

        :param df: [DataFrame] DataFrame containing area ID values in 'grid_gid' series
        :param meta: [DataFrame] DataFrame containing area ID values in 'grid_gid' series and HDF file chunk
                                 values in 'chunk' series
        :return: [ndarray] 1-dimensional array containing HDF file chunk values
        """

        _gid = np.in1d(meta['grid_gid'].values, df['grid_gid'].values)
        return meta[_gid]['chunk'].unique()


    @staticmethod
    def stabilize(x: np.ndarray, theta: int=2) -> np.ndarray:
        """
        Remove seasonality and trend from <x>.

        :param x: [ndarray] time series array
        :param theta: [float]
        :return: [ndarray]
        """

        return (1. / theta) * np.arcsinh(theta * np.diff(x, axis=1))


    @staticmethod
    def get_profiles(hdf_ws, resource, title, gids, meta, chunks, region, time_intervals, subset=None,
                     stabilize=stabilize, year=None, utc_offset=0):
        """
        Collect 'cf_profiles' from <hdf_ws>/<resource>_<year>_[chunk].h5 for each chunk in <chunks>
        and area ID values from <meta>.

        :param hdf_ws: [string] workspace file path containing capacity factor profiles
        :param resource: [string] resource name
        :param title: [string] project title
        :param gids: [np.array] area ID values
        :param meta: [DataFrame] area ID values as 'gid' Series, chunk values as 'chunk' Series, and
                                 timezone values as 'tz' Series
        :param chunks: [list] HDF file chunk values
        :param region: [string] region identifier
        :param time_intervals: [int] number of time steps per profile
        :param subset: [list] optional subset index
        :param stabilize: [function] postprocessing stabilizing function
        :param year: [int] year of interest
        :param utc_offset: [int] timezone adjustment in number of hours ahead of UTC
        :return: [(DataFrame, DataFrame)] original and stationary time series values with 'gid' Series
        """

        title = title or resource

        # initiate format values
        _kvals = {'resource': resource,
                  'title': title,
                  'year': year}

        # initiate empty arrays
        _profiles = np.empty(shape=(time_intervals, gids.shape[0]), dtype=np.int) * np.nan
        _gid_index = np.empty(shape=gids.shape[0], dtype=np.int) * np.nan

        logger.debug('[{g}] loading{y}{t} profiles from {w}'.format(y=' %s ' % year,
                                                                      t=title,
                                                                      w=hdf_ws,
                                                                      g=region))

        _i_start = 0
        for _chunk in chunks:
            # add chunk number to format values
            _kvals['chunk'] = _chunk

            # create mask for full meta dataframe
            _chunk_mask = meta['chunk'].values == _chunk

            # extract metadata just for chunk
            _chunk_meta = meta[_chunk_mask].reset_index(drop=True)[['gid', 'chunk', 'tz']]

            # get relevant index values
            _chunk_index = _chunk_meta[np.in1d(_chunk_meta['gid'].values, gids)].index

            # set end of slice index
            _i_end = _i_start + _chunk_index.shape[0]

            # load all profiles for chunk
            if year is not None:
                _hfile_fpath = os.path.join(hdf_ws, '{title}_{year:d}_{chunk:d}.h5'.format(**_kvals))
            else:
                _hfile_fpath = os.path.join(hdf_ws, '{title}_{chunk:d}.h5'.format(**_kvals))
            logger.debug('[{r}] loading {p:d} profiles from {f}'.format(p=_chunk_index.shape[0],
                                                                        f=_hfile_fpath,
                                                                        r=region))
            with h5py.File(_hfile_fpath, 'r') as _chunk_hfile:
                # select profiles from full chunk; convert to TZ == utc_offset
                _chunk_profiles = np.roll(_chunk_hfile['cf_profile'].value,
                                          shift=BASE_UTC_OFFSET + utc_offset,
                                          axis=0)

                # extract relevant profiles out of chunk's full set and save them in region's full set
                _profiles[:, _i_start:_i_end] = _chunk_profiles[:, _chunk_index] / HDF_SCALE_FACTOR

            # save relevant grid GIDs
            _gid_index[_i_start:_i_end] = _chunk_meta.iloc[_chunk_index, :]['gid'].values

            # move start of slice index forward
            _i_start = _i_end

        if subset:
            _profiles = _profiles[:, subset]
            _gid_index = _gid_index[subset]

        # check for missing data
        _gid_index_clean = _gid_index[~np.isnan(_gid_index)]

        _missing_profiles = _gid_index.shape[0] - _gid_index_clean.shape[0]
        if _missing_profiles:
            logger.warning('[{r}] {p:d} profiles were '
                           'not found in project set'.format(p=_missing_profiles, r=region))

            _profiles = _profiles[:, ~np.all(_profiles, axis=0)]

        logger.debug('[{r}] removing trend and seasonality'.format(r=region))
        if stabilize is not None:
            _profiles_stable = pd.DataFrame(stabilize(_profiles.transpose()))
            _profiles_stable['gid'] = _gid_index_clean
        else:
            _profiles_stable = None
        _profiles = pd.DataFrame(data=_profiles.transpose())
        _profiles['gid'] = _gid_index_clean.astype(np.int)

        return _profiles, _profiles_stable


    @staticmethod
    def get_dwt_coefficients(x, region, wavelet='Haar', level=None, indices=None):
        """
        Collect wavelet coefficients for time series <x> using mother wavelet <wavelet> at
        levels <level>.

        :param x: [ndarray] time series values
        :param region [string] region identifier
        :param wavelet: [string] mother wavelet type
        :param level: [int] optional wavelet computation level
        :param indices: [(int, ...)] coefficient array levels to keep
        :return: [list] stacked coefficients at <indices>
        """

        # set mother
        _wavelet = pywt.Wavelet(wavelet)

        # multi-level with default depth
        logger.info('[{r}] calculating wavelet'
                    ' coefficients with {w} wavelet'.format(r=region, w=_wavelet.family_name))

        _wavedec = pywt.wavedec(data=x, wavelet=_wavelet, axis=1, level=level)

        return subset_coefficients(x=_wavedec, gid_count=x.shape[0], indices=indices, region=region)


    @staticmethod
    def subset_coefficients(x, gid_count, region, indices=None):
        """
        Subset and stack wavelet coefficients
        :param x: [(ndarray, ...)] coefficients arrays
        :param gid_count: [int] number of area ID values
        :param region: [string] region indentifier
        :param indices: [(int, ...)]
        :return: [ndarray] stacked coefficients rounded to 3 decimal places and converted to integers
        """

        indices = indices or range(0, len(x))

        _coefficient_count = 0
        for _index in indices:
            _shape = x[_index].shape
            _coefficient_count += _shape[1]

        _combined_wc = np.empty(shape=(gid_count, _coefficient_count), dtype=np.int)

        logger.debug('[{r}] using {c:d} coefficients'.format(r=region, c=_coefficient_count))

        _i_start = 0
        for _index in indices:
            _i_end = _i_start + x[_index].shape[1]
            _combined_wc[:, _i_start:_i_end] = np.round(x[_index], 2) * 100
            _i_start = _i_end

        return _combined_wc


    @staticmethod
    def cluster_maxp(z, weight, region, floor=2, verbose=True, floor_variable=None,
                     initial=1, p=True):
        """
        Apply maxp clustering to <x> using <gids>['geom'] to create spatial weight matrix.

        :param z: [ndarray] observations by time series array
        :param weight [W] spatial weight matrix
        :param region [string] region identifier
        :param floor: [int] minimum sum of <floor_variable> across conjoined inputs
        :param verbose: [bool]
        :param floor_variable: [ndarray] array of floor values for each <gids> observation
        :param initial: [int] number of initial solutions to optimize over
        :param p: [bool] perform pseudo-p comparison
        :return: [Region] solution set
        """

        # back fill floor variable
        floor_variable = floor_variable or np.ones(shape=(z.shape[0], 1))

        logger.info('[{r}] regionalizing with maxp using floor {f:d} and '
                    '{i:d} initial solutions'.format(r=region, f=floor, i=initial))
        _r = pysal.Maxp(w=weight, z=z, floor=floor, verbose=verbose, floor_variable=floor_variable,
                        initial=initial)

        # calculate p
        if p:
            logger.info('[{r}] calculating p value'.format(r=region))
            _r.cinference()
            # @TODO: enable p-value usage
        return _r


    @staticmethod
    def _save_profile_to_db(db_host, db_name, db_schema, db_table, df, iteration, region,
                            res_lkup_id_column, res_schema, res_table, res_id_column):
        """
        Export profile to database table. Each post-process iteration becomes a new column.

        :param db_host: [string]
        :param db_name: [string]
        :param db_schema: [string]
        :param db_table: [string]
        :param df: [DataFrame]
        :param iteration: [int]
        :param region: [string]
        :param res_lkup_id_column: [string]
        :param res_schema: [string]
        :param res_table: [string]
        :param res_id_column: [string]
        :return: [bool] True on success
        """

        _kvals = {'s': db_schema,
                  't': db_table,
                  'c_jid': 'c{:04}g'.format(iteration - 1),
                  'c_id': 'c{:04}'.format(iteration),
                  'c_cf': 'ann_avg_cf',
                  'r': region,
                  'res_lkup_id_column': res_lkup_id_column,
                  'res_schema': res_schema,
                  'res_table': res_table,
                  'res_id_column': res_id_column}

        df['level'] = iteration

        with psycopg2.connect(host=db_host, dbname=db_name) as _conn:
            with _conn.cursor() as _cur:
                # prepare output table
                _sql = """SELECT EXISTS (
                            SELECT 1
                            FROM pg_catalog.pg_class c
                            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                            WHERE n.nspname = %(s)s
                            AND c.relname = %(t)s
                            AND c.relkind = 'r'
                            );"""
                _cur.execute(_sql, _kvals)
                _table_exists = _cur.fetchone()[0]

                if not _table_exists:
                    logger.debug('[{r}] creating output table "{s}"."{t}"'.format(**_kvals))
                    _sql = """CREATE TABLE "{s}"."{t}" ("level"      INT       NOT NULL
                                                      , "cluster_id" INT       NOT NULL
                                                      , "{res_lkup_id_column}"   BIGINT
                                                          NOT NULL REFERENCES
                                                          "{res_schema}".
                                                          "{res_table}" ("{res_id_column}")
                                                      , "rmse"       NUMERIC   NOT NULL
                                                      , "anncf"      NUMERIC[] NOT NULL
                                                      , CONSTRAINT "{t}_pkey" PRIMARY KEY
                                                       ("level", "cluster_id")
                                                       );""".format(**_kvals)
                    _cur.execute(_sql)
                else:
                    logger.debug('[{r}] output table "{s}"."{t}" exists'.format(**_kvals))

                # convert to CSV format
                _csv = io.StringIO()
                df.to_csv(_csv, index_label=False, index=True, header=False)
                _csv.seek(0)

                if iteration == 0:
                    if _table_exists:
                        logger.debug('[{r}] cleaning output table "{s}"."{t}"'.format(**_kvals))
                        _sql = """DELETE FROM "{s}"."{t}";""".format(**_kvals)
                        _cur.execute(_sql)

                _columns = ['cluster_id', res_lkup_id_column, 'rmse', 'anncf', 'level']
                logger.debug('[{r}] populating columns "{c}" of output table '
                             '"{s}"."{t}"'.format(c='", "'.join(_columns), **_kvals))
                _sql = 'COPY "{s}"."{t}" ("{c}") FROM STDIN WITH CSV DELIMITER \',\';'.\
                    format(c='", "'.join(_columns), **_kvals)
                _cur.copy_expert(_sql, _csv)
                _csv.close()

            _conn.commit()


    @staticmethod
    def _save_to_db(db_host, db_name, db_schema, db_table, df, iteration, region,
                    res_lkup_id_column, res_schema, res_table, res_id_column, res_geom_column):
        """
        Export results to database table. Each post-process iteration becomes a new column.

        :param db_host: [string]
        :param db_name: [string]
        :param db_schema: [string]
        :param db_table: [string]
        :param df: [DataFrame]
        :param iteration: [int]
        :param region: [string]
        :param res_lkup_id_column: [string]
        :param res_schema: [string]
        :param res_table: [string]
        :param res_id_column: [string]
        :param res_geom_column: [string]
        :return: [bool] True on success
        """

        _kvals = {'s': db_schema,
                  't': db_table,
                  'c_jid': 'c{:04}'.format(iteration - 1),
                  'c_id': 'c{:04}'.format(iteration),
                  'c_cf': 'ann_avg_cf',
                  'r': region,
                  'res_lkup_id_column': res_lkup_id_column,
                  'res_schema': res_schema,
                  'res_table': res_table,
                  'res_id_column': res_id_column,
                  'res_geom_column': res_geom_column}

        with psycopg2.connect(host=db_host, dbname=db_name) as _conn:
            with _conn.cursor() as _cur:
                # prepare output table
                _sql = """SELECT EXISTS (
                            SELECT 1
                            FROM pg_catalog.pg_class c
                            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                            WHERE n.nspname = %(s)s
                            AND c.relname = %(t)s
                            AND c.relkind = 'r'
                            );"""
                _cur.execute(_sql, _kvals)
                _table_exists = _cur.fetchone()[0]

                if not _table_exists:
                    logger.debug('[{r}] creating output table "{s}"."{t}"'.format(**_kvals))

                    _sql = 'CREATE TABLE "{s}"."{t}"' \
                           ' ("{res_lkup_id_column}" BIGINT NOT NULL' \
                           ' PRIMARY KEY REFERENCES "{res_schema}"."{res_table}"' \
                           ' ("{res_id_column}"),' \
                           ' "{c_cf}" NUMERIC(6, 2), "{c_id}" INT NULL);'.format(**_kvals)
                    _cur.execute(_sql)
                else:
                    logger.debug('[{r}] output table "{s}"."{t}" exists'.format(**_kvals))

                    _sql = """SELECT EXISTS (SELECT "column_name" FROM
                    "information_schema"."columns" WHERE "table_schema" = %(s)s AND "table_name"
                     = %(t)s AND "column_name" = %(c_id)s);"""
                    _cur.execute(_sql, _kvals)

                    if not _cur.fetchone()[0]:
                        logger.debug(
                            '[{r}] adding column "{c_id}" to'
                            ' output table "{s}"."{t}"'.format(**_kvals))
                        _sql = """ALTER TABLE "{s}"."{t}" ADD COLUMN "{c_id}" INT NULL;""".format(
                            **_kvals)
                        _cur.execute(_sql)
                    else:
                        if _table_exists:
                            logger.debug(
                                '[{r}] cleaning column "{c_id}" in output table "{s}"."{t}"'.format(
                                    **_kvals))
                            _sql = """UPDATE "{s}"."{t}" SET "{c_id}" = NULL;""".format(**_kvals)
                            _cur.execute(_sql)

                # create geom view
                _sql = """DROP VIEW IF EXISTS "{s}"."{t}_geom" CASCADE;""".format(**_kvals)
                _cur.execute(_sql)

                _sql = """CREATE VIEW "{s}"."{t}_geom" AS (
                          SELECT a.*, b."{res_geom_column}" FROM "{s}"."{t}" a
                          LEFT JOIN "{res_schema}"."{res_table}" b ON a."{res_lkup_id_column}"
                          = b."{res_id_column}");""".format(**_kvals)
                _cur.execute(_sql)

                # convert to CSV format
                _csv = io.StringIO()
                df.to_csv(_csv, index_label=False, index=False, header=False)
                _csv.seek(0)

                if iteration == 0:
                    if _table_exists:
                        logger.debug('[{r}] cleaning output table "{s}"."{t}"'.format(**_kvals))
                        _sql = """DELETE FROM "{s}"."{t}";""".format(**_kvals)
                        _cur.execute(_sql)
                    logger.debug(
                        '[{r}] populating column "{res_lkup_id_column}", "{c_cf}", and "{c_id}"'
                        ' in output table "{s}"."{t}"'.format(
                            **_kvals))
                    _sql = """COPY "{s}"."{t}" ("{res_lkup_id_column}", "{c_cf}", "{c_id}")
                     FROM STDIN WITH CSV DELIMITER ',';""".format(
                        **_kvals)
                    _cur.copy_expert(_sql, _csv)
                else:
                    logger.debug('[{r}] creating temp table "{c_id}"'.format(**_kvals))
                    _sql = """CREATE TEMP TABLE "{c_id}" ("gid" INT NOT NULL PRIMARY KEY,
                     "{c_id}" INT NOT NULL);""".format(
                        **_kvals)
                    _cur.execute(_sql)

                    logger.debug('[{r}] populating temp table "{c_id}"'.format(**_kvals))
                    _sql = """COPY "{c_id}" ("gid", "{c_id}") FROM STDIN
                     WITH CSV DELIMITER ',';""".format(
                        **_kvals)
                    _cur.copy_expert(_sql, _csv)

                    logger.debug(
                        '[{r}] populating column "{c_id}" in'
                        ' output table "{s}"."{t}"'.format(**_kvals))

                    _sql = """UPDATE "{s}"."{t}" a SET "{c_id}" = b."{c_id}" FROM "{c_id}" b
                     WHERE a."{c_jid}" = b."gid";""".format(
                        **_kvals)
                    _cur.execute(_sql)

                _csv.close()

            _conn.commit()


    @staticmethod
    def post_process_maxp(x, gids, orig_profiles, output_ws, region_name, weight_type,
                          coefficient_indices, floor, initial, output_name, title, resource_type,
                          iteration=0, out_db=None, in_db=None, save_to_disk=False, desc=None,
                          raw_profiles=None, rid_mapping=None):
        """
        Convert <x> to GeoDataFrame with hourly original profiles and annual average averaged by region;
        save to .pkl and .shp files.

        :param x: [Maxp] regionalization solution
        :param gids: [GeoDataFrame]
        :param orig_profiles: [DataFrame] time series with 'gid' Series
        :param output_ws: [string] folder path for saved files
        :param region_name: [string] region name
        :param weight_type: [string] spatial weight matrix type
        :param coefficient_indices: [list] index values used to subset wavelet coefficient levels
        :param floor: [int] Maxp floor value
        :param initial: [int] Maxp initial number of solutions value
        :param output_name: [string] output file or table name
        :param title: [string] project title
        :param resource_type: [string] resource description
        :param iteration: [int] recursive count
        :param out_db: [dict] {'db_host': string,
                               'db_name': string,
                               'db_schema': string}
        :param in_db: [dict] {'db_host': string,
                              'db_name': string,
                              'res_schema': string,
                              'res_table': string,
                              'res_id_column': string,
                              'res_geom_column': string,
                              'res_lkup_schema': string,
                              'res_lkup_table': string,
                              'res_lkup_id_column': string,
                              'res_lkup_region_column': string}
        :param save_to_disk: [bool] save pickled region definitions and dissolved regions
        :param desc: [text] project description
        :param raw_profiles: [DataFrame]
        :param rid_mapping: [DataFrame]
        :return: [GeoDataFrame] 'gid', 'rid', 'geom' and averaged original profile values
        """

        logger.debug('[{r}] processing regionalization iteration {i:d}'.format(r=region_name,
                                                                               i=iteration))

        # initiate formatting values
        kvals = {'profile_count': orig_profiles.shape[0],
                 'region_name': region_name,
                 'w': weight_type,
                 'indices': '_'.join(['c{i}'.format(i=i) for i in coefficient_indices or ['n']]),
                 'floor': floor,
                 'initial': initial,
                 'i': iteration,
                 'title': title,
                 'res': resource_type,
                 'desc': desc}

        # collect gid and regions  # @TODO: use x.area2region
        _r2a_gid = []
        _r2a_rid = []
        for _i, _gids in enumerate(x.regions):
            for _gid in _gids:
                _r2a_gid.append(_gid)
                _r2a_rid.append(_i)

        # set column names
        _old_rid_column = 'c{:04}'.format(iteration - 1)
        _rid_column = 'c{:04}'.format(iteration)
        _ts_columns = list(range(raw_profiles.shape[1]))

        _r2a = pd.DataFrame({'rid': _r2a_rid, 'gid': _r2a_gid})

        _r2a_results = gids.merge(_r2a, on='gid')

        # add original input profiles to result set
        _r2a_results = _r2a_results.merge(orig_profiles, on='gid')

        # add annual average capacity factor to result set
        _r2a_results['ann_cf_mean'] = _r2a_results[_ts_columns].mean(axis=1)

        _r2a.set_index('gid', inplace=True)

        _r2a.rename(columns={'rid': _rid_column}, inplace=True)

        if iteration > 0:
            _r2a.index.rename(_old_rid_column, inplace=True)
            rid_mapping.set_index(_old_rid_column, append=True, inplace=True)
            rid_mapping = rid_mapping.join(_r2a)
            _raw_profiles = raw_profiles.join(rid_mapping).set_index(_rid_column, append=True)\
                .reset_index(level=['c{:04d}'.format(x) for x in tuple(range(iteration))], drop=True)
        else:
            rid_mapping = _r2a
            _raw_profiles = raw_profiles.join(_r2a).set_index(_rid_column, append=True)

        # backfill islands
        _raw_profiles = _raw_profiles.reset_index()
        _raw_profiles[_rid_column] = _raw_profiles[_rid_column].fillna(-9).astype(int)
        _raw_profiles.set_index(['gid', _rid_column], inplace=True)

        _rmse_by_rid = ((((_raw_profiles - _raw_profiles.groupby(_rid_column, sort=False)
                           .mean())**2).mean(axis=1))**0.5).sort_values().reset_index()\
            .groupby(_rid_column, sort=False).nth(0).set_index('gid', append=True)
        _rmse_by_rid.columns=['rmse']

        # join DF with original time series
        _rmse_by_rid = _rmse_by_rid.join(raw_profiles, how='inner', lsuffix='rmse')

        # dissolve to hourly if needed
        if raw_profiles.shape[1] == 17520:
            logger.debug('[{r}]: converting to hourly data'.format(r=region_name))
            _hour = _rmse_by_rid.loc[:, _ts_columns].loc[:, ::2]
            _half_hour = _rmse_by_rid.loc[:, _ts_columns].loc[:, 1::2]
            _rmse_by_rid.drop(_ts_columns, axis=1, inplace=True)
            _ts_columns = list(range(8760))
            _hour.columns = _ts_columns
            _half_hour.columns = _ts_columns
            _8760 = (_hour + _half_hour) / 2.0
            _rmse_by_rid = _rmse_by_rid.join(_8760)
        elif raw_profiles.shape[1] == 8760:
            pass
        else:
            logger.error('[{r}] unexpected array size'.format(r=region_name))

        # put original time series into list for .to_sql() method
        _rmse_by_rid['anncf'] = _rmse_by_rid[_ts_columns].T\
            .apply(lambda x: str(list(x)).replace('[', '{').replace(']', '}'))

        # remove independent columns
        _rmse_by_rid.drop(_ts_columns, inplace=True, axis=1)

        if out_db is not None:
            if iteration == 0:
                # save original grid_gid ann cf
                _df = _r2a_results[['gid', 'ann_cf_mean', 'rid']]
            else:
                _df = _r2a_results[['gid', 'rid']]

            _db_table = output_name or 'maxp_{desc}_{res}_{region_name}_p{profile_count}' \
                                       '_{w}_{indices}_i{initial}'
            out_db['db_table'] = _db_table.format(**kvals)

            _save_to_db(df=_df, iteration=iteration, region=region_name,
                        res_lkup_id_column=in_db['res_lkup_id_column'], res_schema=in_db['res_schema'],
                        res_table=in_db['res_table'], res_geom_column=in_db['res_geom_column'],
                        res_id_column=in_db['res_id_column'], **out_db)

            out_db['db_table'] += '_profiles'

            _save_profile_to_db(df=_rmse_by_rid, iteration=iteration, region=region_name,
                                res_lkup_id_column=in_db['res_lkup_id_column'],
                                res_schema=in_db['res_schema'], res_table=in_db['res_table'],
                                res_id_column=in_db['res_id_column'], **out_db)

        if save_to_disk:
            # set filename
            _fname = output_name or 'maxp_{desc}_{title}_{region_name}' \
                                    '_{i:04d}_p{profile_count}_{w}_{indices}_i{initial}'
            _fname = _fname.format(**kvals)
            _fpath = os.path.join(output_ws, _fname)

            # save results to disk
            _r2a_results[['gid', 'rid', 'ann_cf_mean']].to_pickle('%s.pkl' % _fpath)

            logger.debug('[{r}] Saved to: {f}.{e}'.format(r=region_name, f=_fpath, e='pkl'))

        logger.debug('[{r}] dissolving solution'.format(r=region_name))
        try:
            _r2a_results = _r2a_results.drop('gid', axis=1).dissolve(by='rid', aggfunc='mean',
                                                                     as_index=False)
        except ValueError:

            logger.warning('[{r}] dissolve error'.format(r=region_name))
            _r2a_results['geom'] = _r2a_results.geom.buffer(0.00001)
            _r2a_results = _r2a_results.drop('gid', axis=1).dissolve(by='rid', aggfunc='mean',
                                                                     as_index=False)
        if save_to_disk:
            _r2a_results[['rid', 'ann_cf_mean', 'geom']].to_file(driver='ESRI Shapefile',
                                                                 filename='%s.shp' % _fpath)

        _r2a_results.rename(columns={'rid': 'gid'}, inplace=True)

        return _r2a_results, rid_mapping


    @staticmethod
    def get_weight_matrix(gdf, profiles, region, weight_type='rook', islands=set()):
        _weight_makers = {'rook': pysal.weights.Rook.from_dataframe,
                          'queen': pysal.weights.Queen.from_dataframe}

        def _remove_islands(gdf, w, profiles):
            gdf = gdf[~np.in1d(gdf['gid'].values, w.islands)]
            profiles = profiles[~np.in1d(profiles['gid'].values, w.islands)]

            return gdf, profiles

        # create weight matrix
        w = _weight_makers[weight_type](df=gdf, geom_col='geom', ids=gdf['gid'],
                                        silent_island_warning=True)

        logger.debug('[{r}] using neighborhood {w}'.format(r=region, w=weight_type))
        if w.islands:
            logger.warning('[{r}] {i:d} islands found and removed'.format(r=region, i=len(w.islands)))
            logger.debug('[{r}] removed grid_gid values: {i}'.format(r=region, i=w.islands))
            gdf, profiles = _remove_islands(gdf, w, profiles)
            islands.update(w.islands)
            # @TODO: needs a try/catch; sometimes islands will be all that remain and we should exit sucessfully
            return get_weight_matrix(gdf=gdf, profiles=profiles, weight_type=weight_type,
                                     islands=islands, region=region)
        else:
            return w, profiles, islands


    @staticmethod
    def get_regions(schema: str, table: str, column: str,
                    db_host: str, db_name: str) -> pd.DataFrame:
        """
        Get all resource regions

        :param schema:
        :param table:
        :param column:
        :param db_host:
        :param db_name:
        :return:
        """

        _sql = """SELECT DISTINCT "{column}" AS "region"
        FROM "{schema}"."{table}"
        WHERE "{column}" IS NOT NULL
        ORDER BY "{column}";""".format(column=column,
                                       schema=schema,
                                       table=table)

        with psycopg2.connect(host=db_host, dbname=db_name) as _conn:
            with _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as _cur:
                _cur.execute(_sql)

                return pd.DataFrame(_cur.fetchall())


    @staticmethod
    def get_cluster_level(region: str, project: str, region_resource: str, db_host: str, db_name: str) -> int:
        """
        Get current cluster level from rpm.resource_region_levels.

        :param project:
        :param resource:
        :param db_host:
        :param db_name:
        :return:
        """

        logger.info('[%s] getting cluster level' % (region, ))

        _sql = """SELECT "{resource}" AS "cluster_level"
        FROM "rpm"."resource_region_levels"
        WHERE "ba" = %(region)s
        AND "project" = %(project)s
        AND "{resource}" IS NOT NULL;""".format(resource=region_resource)

        with psycopg2.connect(host=db_host, dbname=db_name) as _conn:
            with _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as _cur:
                _cur.execute(_sql, {'project': project, 'region': region})

                _cluster_level = _cur.fetchone()

                if _cluster_level:
                    return _cluster_level['cluster_level']


    @staticmethod
    def get_cluster_ids(region: str, cluster_level: int, region_resource: str,
                        grid_id: str, schema: str, table: str,
                        db_host: str, db_name: str, trg: bool,
                        exc_schema: str, exc_table: str,
                        inc_pct: float) -> pd.DataFrame:
        """
        Get GIDs and cluster IDs for <region> and <resource>.
        :param region:
        :param cluster_level:
        :param resource:
        :param schema:
        :param table:
        :param db_host:
        :param db_name:
        :return:
        """

        logger.info('[%s] getting cluster IDs' % (region, ))

        _schema = schema or "rpm_{resource}_regions".format(resource=region_resource)
        _table = table or "{resource}_{region}".format(resource=region_resource, region=region)

        _sql =  """SELECT a."c{cluster_level:04}" AS "cluster_id"\n"""
        _sql += """     , a."{gid}" AS "grid_gid"\n"""
        if trg:
            _sql += """     , a."trg"\n"""

        _sql += """FROM "{schema}"."{table}" a\n"""
        if exc_schema and exc_table and inc_pct:
            _sql += """LEFT JOIN "{exc_schema}"."{exc_table}" b ON a."{gid}" = b."{gid}"\n"""
        _sql += """WHERE a."c{cluster_level:04}" IS NOT NULL\n"""
        if exc_schema and exc_table and inc_pct:
            _sql += """  AND b."{gid}" IS NOT NULL\n"""
            _sql += """  AND b."inc_pct" >= %(inc_pct)s\n"""
        _sql += """ORDER BY 1, 2;"""

        _sql = _sql.format(gid=grid_id, cluster_level=int(cluster_level), schema=_schema, table=_table, exc_schema=exc_schema, exc_table=exc_table)

        with psycopg2.connect(host=db_host, dbname=db_name) as _conn:
            with _conn.cursor() as _cur:
                _cur.execute(_sql, {'inc_pct': inc_pct})

                _results = _cur.fetchall()

                return _results


    @staticmethod
    def get_rep_profile(region: str, clusters: list, meta: pd.DataFrame, cluster_schema: str, cluster_table: str, year: int, rev_project_ws: str, profile_resource: str, rev_resource: str, utc_offset: int, db_host: str, db_name: str):
        """
        Determine representative profile for each cluster in each region.

        :param region:
        :param clusters:
        :param meta:
        :param ws:
        :param resource:
        :return:
        """

        logger.debug('[%s] extracting clusters' % (region, ))

        # extract GID/cluster ID map
        _rename = {0: 'cluster_id', 1: 'grid_gid', 2: 'trg'}
        if profile_resource == 'wind':
            _rename[2] = 'trg'
        _clusters = pd.DataFrame(clusters).rename(_rename, axis=1).set_index('grid_gid')

        # get chunks
        _chunks = get_chunks(df=_clusters.reset_index(), meta=meta)

        # subset metadata by chunk values relevant to gids
        _meta = meta[['grid_gid', 'chunk', 'tz']][np.in1d(meta['chunk'].values, _chunks)]

        # load profiles from <rev_project_ws>
        _time_intervals = 8760 if rev_resource in ('csp', 'wind') else 17520
        _profiles, _profiles_stable = get_profiles(resource=profile_resource, title=None, gids=_meta[['grid_gid']], meta=_meta, hdf_ws=rev_project_ws, chunks=_chunks, region=region, time_intervals=_time_intervals, subset=None, stabilize=None, utc_offset=utc_offset, year=year)

        _ts_columns = [str(x) for x in range(_time_intervals)]

        _profiles = _profiles.merge(_clusters.reset_index(), on='grid_gid').set_index('grid_gid')

        # calculate RMSE
        _join = ['cluster_id', ]
        if profile_resource == 'wind':
            _join.append('trg')
        _profiles_mean = _profiles.groupby(_join).mean()
        _profiles = _profiles.join(_profiles_mean, on=_join, rsuffix='_avg')

        for _i in _ts_columns:
            _profiles['%s_diff' % _i] = _profiles['%s' % _i] - _profiles['%s_avg' % _i]**2

        _avg_columns = ['%s_avg' % _ for _ in _ts_columns]
        _profiles.drop(columns=_avg_columns, inplace=True)

        _diff_columns = ['%s_diff' % _ for _ in _ts_columns]
        _profiles['rmse'] = _profiles[_diff_columns].mean(axis=1)**0.5
        _profiles.drop(columns=_diff_columns, inplace=True)

        #_profiles.sort_values(by='rmse', inplace=True)

        _slice = ['cluster_id', 'rmse']
        if profile_resource == 'wind':
            _slice.append('trg')

        _best_profiles = _profiles[_slice].groupby(_join).min().reset_index()

        # get profile
        _rmse_by_rid = _best_profiles.merge(_profiles.reset_index()[_join + ['grid_gid', 'rmse'] + _ts_columns], on=_slice + ['rmse', ])

        del _profiles
        del _profiles_stable

        # dissolve to hourly if needed
        if _time_intervals  == 17520:
            logger.debug('[{r}]: converting to hourly data'.format(r=region))
            _hour = _rmse_by_rid[_rmse_by_rid[_ts_columns].columns[::2]]
            _half_hour = _rmse_by_rid[_rmse_by_rid[_ts_columns].columns[1::2]]
            _rmse_by_rid.drop(_ts_columns, axis=1, inplace=True)
            _ts_columns = list(range(8760))
            _hour.columns = _ts_columns
            _half_hour.columns = _ts_columns
            _8760 = (_hour + _half_hour) / 2.0
            _rmse_by_rid = _rmse_by_rid.join(_8760)
        elif _time_intervals  == 8760:
            pass
        else:
            logger.error('[{r}] unexpected array size'.format(r=region))

        # put original time series into list for .to_sql() method
        _rmse_by_rid['anncf'] = _rmse_by_rid[_ts_columns].T.apply(lambda x: str(list(x)).replace('[', '{').replace(']', '}'))

        # remove independent columns
        _rmse_by_rid.drop(_ts_columns, inplace=True, axis=1)

        # add region
        _rmse_by_rid['region'] = region

        # convert to CSV format
        _csv = io.StringIO()
        _rmse_by_rid.to_csv(_csv, index_label=False, index=False, header=False)
        _csv.seek(0)

        # save to DB
        _kvals = {'s': cluster_schema,
                  't': cluster_table}

        _drop_sql = """DROP TABLE IF EXISTS "clusters";"""
        if profile_resource == 'wind':
            _create_sql = """CREATE TEMP TABLE "clusters" ("region" TEXT, "cluster_id" NUMERIC, "trg" NUMERIC, "grid_gid" BIGINT, "rmse" NUMERIC, "hrcf" double precision[]);"""
            _copy_sql = """COPY "clusters" ("cluster_id", "trg", "rmse", "grid_gid", "hrcf", "region") FROM STDIN WITH CSV DELIMITER ',';"""
            _update_sql = """UPDATE "{s}"."{t}" a SET "hrcf" = b."hrcf", "rmsvar" = ROUND(b."rmse"::NUMERIC, 3), "anncf" = ROUND(array_avg(b."hrcf")::NUMERIC, 3) FROM "clusters" b WHERE a."ba" = b."region" AND a."id" = b."cluster_id"::INT AND a."trg" = b."trg"::INT;""".format(**_kvals)
        else:
            _create_sql = """CREATE TEMP TABLE "clusters" ("region" TEXT, "cluster_id" NUMERIC, "grid_gid" BIGINT, "rmse" NUMERIC, "hrcf" double precision[]);"""
            _copy_sql = """COPY "clusters" ("cluster_id", "rmse", "grid_gid", "hrcf", "region") FROM STDIN WITH CSV DELIMITER ',';"""
            _update_sql = """UPDATE "{s}"."{t}" a SET "hrcf" = b."hrcf", "rmsvar" = ROUND(b."rmse"::NUMERIC, 3), "anncf" = ROUND(array_avg(b."hrcf")::NUMERIC, 3) FROM "clusters" b WHERE a."ba" = b."region" AND a."id" = b."cluster_id"::INT""".format(**_kvals)

        with psycopg2.connect(host=db_host, dbname=db_name) as _conn:
            with _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as _cur:
                logger.debug('[{r}] create temp table'.format(r=region))

                _cur.execute(_drop_sql)
                _cur.execute(_create_sql)

                logger.debug('[{r}] loading temp table'.format(r=region))
                _cur.copy_expert(_copy_sql, _csv)

                logger.debug('[{r}] updating {s}.{t}'.format(r=region, **_kvals))
                _cur.execute(_update_sql)
                _cur.execute(_drop_sql)
            _conn.commit()

        _csv.close()

        return (region, True)
