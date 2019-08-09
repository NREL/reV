# -*- coding: utf-8 -*-
"""
RPM output handler.
"""
import psutil
import os
import concurrent.futures as cf
import logging
import numpy as np
import pandas as pd

from reV.handlers.geotiff import Geotiff
from reV.handlers.outputs import Outputs
from reV.rpm.rpm_clusters import RPMClusters
from reV.utilities.exceptions import RPMRuntimeError, RPMTypeError

logger = logging.getLogger(__name__)


class RPMOutput:
    """Framework to format and process RPM clustering results."""

    def __init__(self, rpm_clusters, fpath_excl, fpath_techmap, dset_techmap,
                 fpath_gen, excl_area=0.0081, include_threshold=0.001,
                 n_profiles=1, rerank=True, cluster_kwargs=None,
                 parallel=True):
        """
        Parameters
        ----------
        rpm_clusters : pd.DataFrame | str
            Single DataFrame with (gid, gen_gid, cluster_id, rank),
            or str to file.
        fpath_excl : str | None
            Filepath to exclusions data (must match the techmap grid).
            None will not apply exclusions.
        fpath_techmap : str | None
            Filepath to tech mapping between exclusions and resource data.
            None will not apply exclusions.
        dset_techmap : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        fpath_gen : str
            reV generation output file.
        excl_area : float
            Area in km2 of one exclusion pixel.
        include_threshold : float
            Inclusion threshold. Resource pixels included more than this
            threshold will be considered in the representative profiles.
            Set to zero to find representative profile on all resource, not
            just included.
        n_profiles : int
            Number of representative profiles to output.
        rerank : bool
            Flag to rerank representative generation profiles after removing
            excluded generation pixels.
        cluster_kwargs : dict
            RPMClusters kwargs
        parallel : bool | int
            Flag to apply exclusions in parallel. Integer is interpreted as
            max number of workers. True uses all available.
        """

        logger.info('Initializing RPM output processing...')

        self._clusters = self._parse_cluster_arg(rpm_clusters)
        self._fpath_excl = fpath_excl
        self._fpath_techmap = fpath_techmap
        self._dset_techmap = dset_techmap
        self._fpath_gen = fpath_gen
        self.excl_area = excl_area
        self.include_threshold = include_threshold
        self.n_profiles = n_profiles
        self.rerank = rerank

        self.parallel = parallel
        if self.parallel is True:
            self.max_workers = os.cpu_count()
        elif self.parallel is False:
            self.max_workers = 1
        else:
            self.max_workers = self.parallel

        if cluster_kwargs is None:
            self.cluster_kwargs = {}
        else:
            self.cluster_kwargs = cluster_kwargs

        self._excl_lat = None
        self._excl_lon = None
        self._full_lat_slice = None
        self._full_lon_slice = None
        self._init_lat_lon()

    @staticmethod
    def _parse_cluster_arg(rpm_clusters):
        """Parse dataframe from cluster input arg.

        Parameters
        ----------
        rpm_clusters : pd.DataFrame | str
            Single DataFrame with (gid, gen_gid, cluster_id, rank),
            or str to file.

        Returns
        -------
        clusters : pd.DataFrame
            Single DataFrame with (gid, gen_gid, cluster_id, rank,
            latitude, longitude)
        """

        if isinstance(rpm_clusters, pd.DataFrame):
            clusters = rpm_clusters

        elif isinstance(rpm_clusters, str):
            if rpm_clusters.endswith('.csv'):
                clusters = pd.read_csv(rpm_clusters)
            elif rpm_clusters.endswith('.json'):
                clusters = pd.read_json(rpm_clusters)

        else:
            raise RPMTypeError('Expected a DataFrame or str but received {}'
                               .format(type(rpm_clusters)))

        RPMOutput._check_cluster_cols(clusters)

        return clusters

    @staticmethod
    def _check_cluster_cols(df, required=('gen_gid', 'gid', 'latitude',
                                          'longitude', 'cluster_id', 'rank')):
        """Check for required columns in the rpm cluster dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Single DataFrame with columns to check
        """

        missing = []
        for c in required:
            if c not in df:
                missing.append(c)
        if any(missing):
            raise RPMRuntimeError('Missing the following columns in RPM '
                                  'clusters input df: {}'.format(missing))

    def _init_lat_lon(self):
        """Initialize the lat/lon arrays and reduce their size."""

        if self._fpath_techmap is not None:

            self._full_lat_slice, self._full_lon_slice = \
                self._get_lat_lon_slices(cluster_id=None)

            logger.debug('Initial lat/lon shape is {} and {} and '
                         'range is {} - {} and {} - {}'
                         .format(self.excl_lat.shape, self.excl_lon.shape,
                                 self.excl_lat.min(), self._excl_lat.max(),
                                 self.excl_lon.min(), self._excl_lon.max()))
            self._excl_lat = self._excl_lat[self._full_lat_slice,
                                            self._full_lon_slice]
            self._excl_lon = self._excl_lon[self._full_lat_slice,
                                            self._full_lon_slice]
            logger.debug('Reduced lat/lon shape is {} and {} and '
                         'range is {} - {} and {} - {}'
                         .format(self.excl_lat.shape, self.excl_lon.shape,
                                 self.excl_lat.min(), self._excl_lat.max(),
                                 self.excl_lon.min(), self._excl_lon.max()))

    @staticmethod
    def _get_tm_data(fpath_techmap, dset_techmap, lat_slice, lon_slice):
        """Get the techmap data.

        Parameters
        ----------
        fpath_techmap : str
            Filepath to tech mapping between exclusions and resource data.
        dset_techmap : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        lat_slice : slice
            The latitude (row) slice to extract from the exclusions or
            techmap 2D datasets.
        lon_slice : slice
            The longitude (col) slice to extract from the exclusions or
            techmap 2D datasets.

        Returns
        -------
        techmap : np.ndarray
            Techmap data mapping exclusions grid to resource gid (flattened).
        """

        with Outputs(fpath_techmap) as tm:
            techmap = tm[dset_techmap, lat_slice, lon_slice].astype(np.int32)
        return techmap.flatten()

    @staticmethod
    def _get_excl_data(fpath_excl, lat_slice, lon_slice, band=0):
        """Get the exclusions data from a geotiff file.

        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions data (must match the techmap grid).
        lat_slice : slice
            The latitude (row) slice to extract from the exclusions or
            techmap 2D datasets.
        lon_slice : slice
            The longitude (col) slice to extract from the exclusions or
            techmap 2D datasets.
        band : int
            Band (dataset integer) of the geotiff containing the relevant data.

        Returns
        -------
        excl_data : np.ndarray
            Exclusions data flattened and normalized from 0 to 1 (1 is incld).
        """

        with Geotiff(fpath_excl) as excl:
            excl_data = excl[band, lat_slice, lon_slice]

        # infer exclusions that are scaled percentages from 0 to 100
        if excl_data.max() > 1:
            excl_data = excl_data.astype(np.float32)
            excl_data /= 100

        return excl_data

    def _get_lat_lon_slices(self, cluster_id=None, margin=0.1):
        """Get the slice args to locate exclusion/techmap data of interest.

        Parameters
        ----------
        cluster_id : str | None
            Single cluster ID of interest or None for full region.
        margin : float
            Extra margin around the cluster lat/lon box.

        Returns
        -------
        lat_slice : slice
            The latitude (row) slice to extract from the exclusions or
            techmap 2D datasets.
        lon_slice : slice
            The longitude (col) slice to extract from the exclusions or
            techmap 2D datasets.
        """

        box = self._get_coord_box(cluster_id)

        mask = ((self.excl_lat > np.min(box['latitude']) - margin)
                & (self.excl_lat < np.max(box['latitude']) + margin)
                & (self.excl_lon > np.min(box['longitude']) - margin)
                & (self.excl_lon < np.max(box['longitude']) + margin))

        lat_locs, lon_locs = np.where(mask)

        if self._full_lat_slice is None and self._full_lon_slice is None:
            lat_slice = slice(np.min(lat_locs), 1 + np.max(lat_locs))
            lon_slice = slice(np.min(lon_locs), 1 + np.max(lon_locs))
        else:
            lat_slice = slice(
                self._full_lat_slice.start + np.min(lat_locs),
                1 + self._full_lat_slice.start + np.max(lat_locs))
            lon_slice = slice(
                self._full_lon_slice.start + np.min(lon_locs),
                1 + self._full_lon_slice.start + np.max(lon_locs))

        return lat_slice, lon_slice

    def _get_all_lat_lon_slices(self, margin=0.1, free_mem=True):
        """Get the slice args for all clusters.

        Parameters
        ----------
        margin : float
            Extra margin around the cluster lat/lon box.
        free_mem : bool
            Flag to free lat/lon arrays from memory to clear space for later
            exclusion processing.

        Returns
        -------
        slices : dict
            Dictionary of tuples - (lat, lon) slices keyed by cluster id.
        """

        slices = {}
        for cid in self._clusters['cluster_id'].unique():
            slices[cid] = self._get_lat_lon_slices(cluster_id=cid,
                                                   margin=margin)

        if free_mem:
            # free up memory
            self._excl_lat = None
            self._excl_lon = None
            self._full_lat_slice = None
            self._full_lon_slice = None

        return slices

    def _get_coord_box(self, cluster_id=None):
        """Get the RPM cluster latitude/longitude range.

        Parameters
        ----------
        cluster_id : str | None
            Single cluster ID of interest or None for all clusters in
            self._clusters.

        Returns
        -------
        coord_box : dict
            Bounding box of the cluster or region:
                {'latitude': (lat_min, lat_max),
                 'longitude': (lon_min, lon_max)}
        """

        if cluster_id is not None:
            mask = (self._clusters['cluster_id'] == cluster_id)
        else:
            mask = len(self._clusters) * [True]

        lat_range = (self._clusters.loc[mask, 'latitude'].min(),
                     self._clusters.loc[mask, 'latitude'].max())
        lon_range = (self._clusters.loc[mask, 'longitude'].min(),
                     self._clusters.loc[mask, 'longitude'].max())
        box = {'latitude': lat_range, 'longitude': lon_range}
        return box

    @property
    def excl_lat(self):
        """Get the full 2D array of latitudes of the exclusion grid.

        Returns
        -------
        _excl_lat : np.ndarray
            2D array representing the latitudes at each exclusion grid cell
        """

        if self._excl_lat is None and self._fpath_techmap is not None:
            with Outputs(self._fpath_techmap) as f:
                logger.debug('Importing Latitude data from techmap...')
                self._excl_lat = f['latitude']
        return self._excl_lat

    @property
    def excl_lon(self):
        """Get the full 2D array of longitudes of the exclusion grid.

        Returns
        -------
        _excl_lon : np.ndarray
            2D array representing the latitudes at each exclusion grid cell
        """

        if self._excl_lon is None and self._fpath_techmap is not None:
            with Outputs(self._fpath_techmap) as f:
                logger.debug('Importing Longitude data from techmap...')
                self._excl_lon = f['longitude']
        return self._excl_lon

    @staticmethod
    def _single_excl(cluster_id, clusters, fpath_excl, fpath_techmap,
                     dset_techmap, lat_slice, lon_slice):
        """Calculate the exclusions for each resource GID in a cluster.

        Parameters
        ----------
        cluster_id : str
            Single cluster ID of interest.
        clusters : pandas.DataFrame
            Single DataFrame with (gid, gen_gid, cluster_id, rank)
        fpath_excl : str
            Filepath to exclusions data (must match the techmap grid).
        fpath_techmap : str
            Filepath to tech mapping between exclusions and resource data.
        dset_techmap : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        lat_slice : slice
            The latitude (row) slice to extract from the exclusions or
            techmap 2D datasets.
        lon_slice : slice
            The longitude (col) slice to extract from the exclusions or
            techmap 2D datasets.

        Returns
        -------
        inclusions : np.ndarray
            1D array of inclusions fraction corresponding to the indexed
            cluster provided by cluster_id.
        n_inclusions : np.ndarray
            1D array of number of included pixels corresponding to each
            gid in cluster_id.
        n_points : np.ndarray
            1D array of the total number of techmap pixels corresponding to
            each gid in cluster_id.
        """

        mask = (clusters['cluster_id'] == cluster_id)
        locs = np.where(mask)[0]
        inclusions = np.zeros((len(locs), ), dtype=np.float32)
        n_inclusions = np.zeros((len(locs), ), dtype=np.float32)
        n_points = np.zeros((len(locs), ), dtype=np.uint16)

        techmap = RPMOutput._get_tm_data(fpath_techmap, dset_techmap,
                                         lat_slice, lon_slice)
        exclusions = RPMOutput._get_excl_data(fpath_excl, lat_slice, lon_slice)

        for i, ind in enumerate(clusters.loc[mask, :].index.values):
            techmap_locs = np.where(
                techmap == int(clusters.loc[ind, 'gid']))[0]
            gid_excl_data = exclusions[techmap_locs]

            if gid_excl_data.size > 0:
                inclusions[i] = np.sum(gid_excl_data) / len(gid_excl_data)
                n_inclusions[i] = np.sum(gid_excl_data)
                n_points[i] = len(gid_excl_data)
            else:
                inclusions[i] = np.nan
                n_inclusions[i] = np.nan
                n_points[i] = 0

        return inclusions, n_inclusions, n_points

    def apply_exclusions(self):
        """Calculate exclusions for clusters, adding data to self._clusters.
        Returns
        -------
        self._clusters : pd.DataFrame
            self._clusters with new columns for exclusions data.
        """

        logger.info('Working on applying exclusions...')

        unique_clusters = self._clusters['cluster_id'].unique()
        static_clusters = self._clusters.copy()
        self._clusters['included_frac'] = 0.0
        self._clusters['included_area_km2'] = 0.0
        self._clusters['n_excl_pixels'] = 0
        futures = {}

        slices = self._get_all_lat_lon_slices()

        with cf.ProcessPoolExecutor(max_workers=self.max_workers) as exe:

            for i, cid in enumerate(unique_clusters):

                lat_s, lon_s = slices[cid]
                future = exe.submit(self._single_excl, cid,
                                    static_clusters, self._fpath_excl,
                                    self._fpath_techmap, self._dset_techmap,
                                    lat_s, lon_s)
                futures[future] = cid
                logger.debug('Kicked off exclusions for cluster "{}", {} out '
                             'of {}.'.format(cid, i + 1, len(unique_clusters)))

            for i, future in enumerate(cf.as_completed(futures)):
                cid = futures[future]
                mem = psutil.virtual_memory()
                logger.info('Finished exclusions for cluster "{}", {} out '
                            'of {} futures. '
                            'Memory usage is {:.2f} out of {:.2f} GB.'
                            .format(cid, i + 1, len(futures),
                                    mem.used / 1e9, mem.total / 1e9))
                incl, n_incl, n_pix = future.result()
                mask = (self._clusters['cluster_id'] == cid)

                self._clusters.loc[mask, 'included_frac'] = incl
                self._clusters.loc[mask, 'included_area_km2'] = \
                    n_incl * self.excl_area
                self._clusters.loc[mask, 'n_excl_pixels'] = n_pix

        logger.info('Finished applying exclusions.')

        if self.rerank:
            self.run_rerank()

        return self._clusters

    def run_rerank(self):
        """Re-rank rep profiles for just the included resource gids."""

        futures = {}

        with cf.ProcessPoolExecutor(max_workers=self.max_workers) as exe:

            for cid, df in self._clusters.groupby('cluster_id'):
                mask = (df['included_frac'] >= self.include_threshold)
                if any(mask) and not all(mask):
                    gen_gids = df.loc[mask, 'gen_gid']
                    self.cluster_kwargs['dist_rank_filter'] = False
                    self.cluster_kwargs['contiguous_filter'] = False
                    future = exe.submit(RPMClusters.cluster, self._fpath_gen,
                                        gen_gids, 1, **self.cluster_kwargs)
                    futures[future] = cid

            if futures:
                logger.info('Re-ranking representative profiles...')
                self._clusters['rank_included'] = np.nan

            for i, future in enumerate(cf.as_completed(futures)):
                cid = futures[future]
                mem = psutil.virtual_memory()
                logger.info('Finished re-ranking "{}", {} out of {}.'
                            'Memory usage is {:.2f} out of {:.2f} GB.'
                            .format(cid, i, len(futures),
                                    mem.used / 1e9, mem.total / 1e9))
                new = future.result()
                mask = ((self._clusters['cluster_id'] == cid)
                        & (self._clusters['included_frac']
                           > self.include_threshold))
                self._clusters.loc[mask, 'rank_included'] = new['rank'].values

    def _get_rep_profile(self, n=0):
        """Get a single representative profile timeseries dataframe.

        Parameters
        ----------
        n : int
            Rank of profile to get. Zero is the most representative profile.

        Returns
        -------
        profile_df : pd.DataFrame
            Dataframe of representative profiles. Index is timeseries,
            columns are cluster ids.
        """

        with Outputs(self._fpath_gen) as f:
            ti = f.time_index
        cols = self._clusters.cluster_id.unique()
        profile_df = pd.DataFrame(index=ti, columns=cols)
        profile_df.index.name = 'time_index'

        key = 'rank'
        if 'rank_included' in self._clusters:
            key = 'rank_included'

        for i, df in self._clusters.groupby('cluster_id'):
            mask = ~df[key].isnull()
            if any(mask):
                df_ranked = df[mask].sort_values(by=key)
                if n < len(df_ranked):
                    rep = df_ranked.iloc[n, :]
                    gen_gid = rep['gen_gid']
                    mask = (self._clusters['gen_gid'] == gen_gid)
                    self._clusters.loc[mask, 'representative'] = True

                    with Outputs(self._fpath_gen) as f:
                        profile_df.loc[:, i] = f['cf_profile', :, gen_gid]

        return profile_df

    @property
    def representative_profiles(self):
        """Representative profile timeseries dataframe.

        Returns
        -------
        profiles : dict
            Dictionary of dataframes of representative profiles. Keyed by
            rank.
        """

        if ('included_frac' not in self._clusters
                and self._fpath_excl is not None
                and self._fpath_techmap is not None):
            raise RPMRuntimeError('Exclusions must be applied before '
                                  'representative profiles can be '
                                  'determined.')

        if 'representative' not in self._clusters:
            self._clusters['representative'] = False

        profiles = {}
        for n in range(self.n_profiles):
            profiles[n] = self._get_rep_profile(n=n)

        return profiles

    @property
    def cluster_summary(self):
        """Summary dataframe with cluster_id primary key.

        Returns
        -------
        s : pd.DataFrame
            Summary dataframe with a row for each cluster id.
        """

        if ('included_frac' not in self._clusters
                and self._fpath_excl is not None
                and self._fpath_techmap is not None):
            raise RPMRuntimeError('Exclusions must be applied before '
                                  'representative profiles can be determined.')
        if 'representative' not in self._clusters:
            raise RPMRuntimeError('Representative profiles must be determined '
                                  'before summary table can be created.')

        ind = self._clusters.cluster_id.unique()
        cols = ['latitude',
                'longitude',
                'n_gen_gids',
                'included_frac',
                'included_area_km2',
                'representative_gid',
                'representative_gen_gid']
        s = pd.DataFrame(index=ind, columns=cols)
        s.index.name = 'cluster_id'

        for i, df in self._clusters.groupby('cluster_id'):
            s.loc[i, 'latitude'] = df['latitude'].mean()
            s.loc[i, 'longitude'] = df['longitude'].mean()
            s.loc[i, 'n_gen_gids'] = len(df)

            if 'included_frac' in df:
                s.loc[i, 'included_frac'] = df['included_frac'].mean()
                s.loc[i, 'included_area_km2'] = df['included_area_km2'].sum()

            key = 'representative'
            if df[key].any():
                s.loc[i, 'representative_gid'] = \
                    df.loc[df[key], 'gid'].sort_values(by=key).values[0]
                s.loc[i, 'representative_gen_gid'] = \
                    df.loc[df[key], 'gen_gid'].sort_values(by=key).values[0]

        return s

    def make_shape_file(self, fpath_shp):
        """Make shape file containing all clusters.

        Parameters
        ----------
        fpath_shp : str
            Filepath to write shape_file to.
        """

        # Geopandas doesnt like writing booleans
        if 'representative' in self._clusters:
            self._clusters['representative'] = \
                self._clusters['representative'].astype(int)

        RPMClusters._generate_shapefile(self._clusters, fpath_shp)

        if 'representative' in self._clusters:
            self._clusters['representative'] = \
                self._clusters['representative'].astype(bool)

    def export_all(self, out_dir, job_tag=None):
        """Run RPM output algorithms and write to CSV's.

        Parameters
        ----------
        out_dir : str
            Directory to dump output files.
        job_tag : str | None
            Optional name tag to add to the csvs being saved.
            Format is "rpm_cluster_output_{tag}.csv".
        """

        fn_out = 'rpm_cluster_output.csv'
        fn_pro = 'rpm_rep_profiles.csv'
        fn_sum = 'rpm_cluster_summary.csv'
        fn_shp = 'rpm_cluster_shapes.shp'

        if job_tag is not None:
            fn_out = fn_out.replace('.csv', '_{}.csv'.format(job_tag))
            fn_pro = fn_pro.replace('.csv', '_{}.csv'.format(job_tag))
            fn_sum = fn_sum.replace('.csv', '_{}.csv'.format(job_tag))
            fn_shp = fn_shp.replace('.shp', '_{}.shp'.format(job_tag))

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if ('included_frac' not in self._clusters
                and self._fpath_excl is not None
                and self._fpath_techmap is not None):
            self.apply_exclusions()

        for i, profile in self.representative_profiles.items():
            fn_pro_i = fn_pro.replace('.csv', '_{}.csv'.format(i))
            profile.to_csv(os.path.join(out_dir, fn_pro_i))
            logger.info('Saved {}'.format(fn_pro_i))

        self.cluster_summary.to_csv(os.path.join(out_dir, fn_sum))
        logger.info('Saved {}'.format(fn_sum))

        self._clusters.to_csv(os.path.join(out_dir, fn_out), index=False)
        logger.info('Saved {}'.format(fn_out))

        self.make_shape_file(os.path.join(out_dir, fn_shp))
        logger.info('Saved {}'.format(fn_shp))

    @classmethod
    def process_outputs(cls, rpm_clusters, fpath_excl, fpath_techmap,
                        dset_techmap, fpath_gen, out_dir, job_tag=None,
                        parallel=True, cluster_kwargs=None, **kwargs):
        """Perform output processing on clusters and write results to disk.

        Parameters
        ----------
        rpm_clusters : pd.DataFrame | str
            Single DataFrame with (gid, gen_gid, cluster_id, rank),
            or str to file.
        fpath_excl : str | None
            Filepath to exclusions data (must match the techmap grid).
            None will not apply exclusions.
        fpath_techmap : str | None
            Filepath to tech mapping between exclusions and resource data.
            None will not apply exclusions.
        dset_techmap : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        fpath_gen : str
            reV generation output file.
        out_dir : str
            Directory to dump output files.
        job_tag : str | None
            Optional name tag to add to the output files.
            Format is "rpm_cluster_output_{tag}.csv".
        parallel : bool | int
            Flag to apply exclusions in parallel. Integer is interpreted as
            max number of workers. True uses all available.
        cluster_kwargs : dict
            RPMClusters kwargs
        """

        rpmo = cls(rpm_clusters, fpath_excl, fpath_techmap, dset_techmap,
                   fpath_gen, cluster_kwargs=cluster_kwargs, parallel=parallel,
                   **kwargs)
        rpmo.export_all(out_dir, job_tag=job_tag)
