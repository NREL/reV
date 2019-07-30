"""
Pipeline between reV and RPM
"""
import os
import concurrent.futures as cf
import logging
import numpy as np
import pandas as pd

from reV.handlers.geotiff import Geotiff
from reV.handlers.outputs import Outputs
from reV.rpm.clusters import RPMClusters
from reV.utilities.exceptions import RPMValueError, RPMRuntimeError

logger = logging.getLogger(__name__)


class RPMClusterManager:
    """
    RPM Cluster Manager:
    - Extracts gids for all RPM regions
    - Runs RPMClusters in parallel for all regions
    - Save results to disk
    """
    def __init__(self, cf_profiles, rpm_meta, rpm_region_col=None):
        """
        Parameters
        ----------
        cf_profiles : str
            Path to reV .h5 files containing desired capacity factor profiles
        rpm_meta : pandas.DataFrame | str
            DataFrame or path to .csv or .json containing the RPM meta data:
            - Regions of interest
            - # of clusters per region
            - cf or resource GIDs if region is not in default meta data
        rpm_region_col : str | Nonetype
            If not None, the meta-data filed to map RPM regions to
        """
        self._cf_h5 = cf_profiles
        self._rpm_regions = self._map_rpm_regions(rpm_meta,
                                                  region_col=rpm_region_col)

    @staticmethod
    def _parse_rpm_meta(rpm_meta):
        """
        Extract rpm meta and map it to the cf profile data

        Parameters
        ----------
        rpm_meta : pandas.DataFrame | str
            DataFrame or path to .csv or .json containing the RPM meta data:
            - Regions of interest
            - # of clusters per region
            - cf or resource GIDs if region is not in default meta data

        Returns
        -------
        rpm_meta : pandas.DataFrame
            DataFrame of RPM regional meta data (clusters and cf/resource GIDs)
        """
        if isinstance(rpm_meta, str):
            if rpm_meta.endswith('.csv'):
                rpm_meta = pd.read_csv(rpm_meta)
            elif rpm_meta.endswith('.json'):
                rpm_meta = pd.read_json(rpm_meta)
            else:
                raise RPMValueError("Cannot read RPM meta, "
                                    "file must be a '.csv' or '.json'")
        elif not isinstance(rpm_meta, pd.DataFrame):
            raise RPMValueError("RPM meta must be supplied as a pandas "
                                "DataFrame or as a .csv, or .json file")

        return rpm_meta

    def _map_rpm_regions(self, rpm_meta, region_col=None):
        """
        Map RPM meta to cf_profile gids

        Parameters
        ----------
        rpm_meta : pandas.DataFrame | str
            DataFrame or path to .csv or .json containing the RPM meta data:
            - Regions of interest
            - # of clusters per region
            - cf or resource GIDs if region is not in default meta data
        region_col : str | Nonetype
            If not None, the meta-data filed to map RPM regions to

        Returns
        -------
        rpm_regions : dict
            Dictionary mapping rpm regions to cf GIDs and number of
            clusters
        """
        rpm_meta = self._parse_rpm_meta(rpm_meta)

        with Outputs(self._cf_h5, mode='r') as cfs:
            cf_meta = cfs.meta

        cf_meta.index.name = 'gen_gid'
        cf_meta = cf_meta.reset_index().set_index('gid')

        rpm_regions = {}
        for region, region_df in rpm_meta.groupby('region'):
            region_map = {}
            if 'gid' in region_df:
                region_meta = cf_meta.loc[region_df['gid'].values]
            elif region_col in cf_meta:
                pos = cf_meta[region_col] == region
                region_meta = cf_meta.loc[pos]
            else:
                raise RPMRuntimeError("Resource gids or a valid resource "
                                      "meta-data field must be supplied "
                                      "to map RPM regions")

            clusters = region_df['clusters'].unique()
            if len(clusters) > 1:
                raise RPMRuntimeError("Multiple values for 'clusters' "
                                      "were provided for region {}"
                                      .format(region))

            region_map['cluster_num'] = clusters[0]
            region_map['gen_gids'] = region_meta['gen_gid'].values
            rpm_regions[region] = region_map

        return rpm_regions

    def _cluster(self, parallel=True, **kwargs):
        """
        Cluster all RPM regions

        Parameters
        ----------
        parallel : bool
            Run clustering of each region in parallel
        kwargs : dict
            RPMCluster kwargs
        """
        if parallel:
            future_to_region = {}
            with cf.ProcessPoolExecutor() as executor:
                for region, region_map in self._rpm_regions.items():
                    clusters = region_map['cluster_num']
                    gen_gids = region_map['gen_gids']

                    future = executor.submit(RPMClusters.cluster, self._cf_h5,
                                             gen_gids, clusters, **kwargs)
                    future_to_region[future] = region

                for future in cf.as_completed(future_to_region):
                    region = future_to_region[future]
                    result = future.result()
                    self._rpm_regions[region].update({'clusters': result})

        else:
            for region, region_map in self._rpm_regions.items():
                clusters = region_map['clusters']
                gen_gids = region_map['gen_gids']
                result = RPMClusters.cluster(self._cf_h5, gen_gids, clusters,
                                             **kwargs)
                self._rpm_regions[region].update({'clusters': result})

    @staticmethod
    def _combine_region_clusters(rpm_regions):
        """
        Combine clusters for all rpm regions and create unique cluster ids

        Parameters
        ----------
        rpm_regions : dict
            Dictionary with RPM region info

        Returns
        -------
        rpm_clusters : pandas.DataFrame
            Single DataFrame with (region, gid, cluster_id, rank)
        """
        rpm_clusters = []
        for region, r_dict in rpm_regions.items():
            r_df = r_dict['clusters'].copy()
            ids = region + '-' + r_df.copy()['cluster_id'].astype(str).values
            r_df.loc[:, 'cluster_id'] = ids
            rpm_clusters.append(r_df)

        rpm_clusters = pd.concat(rpm_clusters)
        return rpm_clusters

    def save_clusters(self, out_file):
        """
        Save cluster results to disk

        Parameters
        ----------
        out_file : str
            Path to file to save clusters too, should be a .csv or .json
        """
        rpm_clusters = self._combine_region_clusters(self._rpm_regions)
        if out_file.endswith('.csv'):
            rpm_clusters.to_csv(out_file, index=False)
        elif out_file.endswith('.json'):
            rpm_clusters.to_json(out_file)
        else:
            raise RPMValueError('out_file must be a .csv or .json')

    @classmethod
    def run(cls, cf_profiles, rpm_meta, out_file,
            rpm_region_col=None, parallel=True, **kwargs):
        """
        RPM Cluster Manager:
        - Extracts gen_gids for all RPM regions
        - Runs RPMClusters in parallel for all regions
        - Save results to disk

        Parameters
        ----------
        cf_profiles : str
            Path to reV .h5 files containing desired capacity factor profiles
        rpm_meta : pandas.DataFrame | str
            DataFrame or path to .csv or .json containing the RPM meta data:
            - Regions of interest
            - # of clusters per region
            - cf or resource GIDs if region is not in default meta data
        out_file : str
            Path to file to save clusters too, should be a .csv or .json
        rpm_region_col : str | Nonetype
            If not None, the meta-data filed to map RPM regions to
        parallel : bool
            Run clustering of each region in parallel
        **kwargs : dict
            RPMClusters kwargs
        """
        rpm = cls(cf_profiles, rpm_meta, rpm_region_col=rpm_region_col)
        rpm._cluster(parallel=parallel, **kwargs)
        rpm.save_clusters(out_file)
        return rpm


class RPMOutput:
    """Framework to format and process RPM clustering results."""

    def __init__(self, rpm_clusters, fpath_excl, fpath_techmap, dset_techmap,
                 fpath_gen, excl_area=0.0081, include_threshold=0.001):
        """
        Parameters
        ----------
        rpm_clusters : pandas.DataFrame
            Single DataFrame with (region, gid, gen_gid, cluster_id, rank)
        fpath_excl : str
            Filepath to exclusions data (must match the techmap grid).
        fpath_techmap : str
            Filepath to tech mapping between exclusions and resource data.
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
        """

        self._clusters = rpm_clusters
        self._fpath_excl = fpath_excl
        self._fpath_techmap = fpath_techmap
        self._dset_techmap = dset_techmap
        self._fpath_gen = fpath_gen
        self.excl_area = excl_area
        self.include_threshold = include_threshold

        self._excl_lat = None
        self._excl_lon = None
        self._full_lat_slice = None
        self._full_lon_slice = None
        self._init_lat_lon()

    def _init_lat_lon(self):
        """Initialize the lat/lon arrays and reduce their size."""
        self._full_lat_slice, self._full_lon_slice = self._get_lat_lon_slices(
            cluster_id=None, margin=3)

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

    def _get_lat_lon_slices(self, cluster_id=None, margin=0.5):
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

        if self._excl_lat is None:
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

        if self._excl_lon is None:
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
            Single DataFrame with (region, gid, gen_gid, cluster_id, rank)
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

    def apply_exclusions(self, parallel=True):
        """Calculate exclusions for clusters, adding data to self._clusters.

        Parameters
        ----------
        Parallel : bool | int
            Flag to apply exclusions in parallel. Integer is interpreted as
            max number of workers. True uses all available.

        Returns
        -------
        clusters : pd.DataFrame
            Copy of self._clusters with new columns for exclusions data.
        """

        clusters = self._clusters.copy()
        static_clusters = self._clusters.copy()
        clusters['included_frac'] = 0.0
        clusters['included_area_km2'] = 0.0
        clusters['n_pixels'] = 0
        futures = {}

        if parallel is True:
            max_workers = os.cpu_count()
        elif parallel is False:
            max_workers = 1
        else:
            max_workers = parallel

        with cf.ProcessPoolExecutor(max_workers=max_workers) as exe:

            for i, cluster_id in enumerate(rpm_clusters.cluster_id.unique()):

                lat_s, lon_s = self._get_lat_lon_slices(cluster_id=cluster_id)
                future = exe.submit(self._single_excl, cluster_id,
                                    static_clusters, self._fpath_excl,
                                    self._fpath_techmap, self._dset_techmap,
                                    lat_s, lon_s)
                futures[future] = cluster_id
                logger.debug('Kicked off cluster "{}", {} out of {}.'
                             .format(cluster_id, i + 1,
                                     len(rpm_clusters.cluster_id.unique())))

            for i, future in enumerate(cf.as_completed(futures)):
                cluster_id = futures[future]
                logger.debug('Finished exclusions for cluster "{}", {} out '
                             'of {} futures.'
                             .format(cluster_id, i + 1, len(futures)))
                incl, n_incl, n_pix = future.result()
                mask = (clusters['cluster_id'] == cluster_id)

                clusters.loc[mask, 'included_frac'] = incl
                clusters.loc[mask, 'included_area_km2'] = \
                    n_incl * self.excl_area
                clusters.loc[mask, 'n_pixels'] = n_pix

        return clusters

    def make_profile_df(self):
        """Make the representative profile dataframe.

        Returns
        -------
        profile_df : pd.DataFrame
            Dataframe of representative profiles. Index is timeseries,
            columns are cluster ids.
        """

        if 'included_frac' not in self._clusters:
            raise KeyError('Exclusions must be applied before representative '
                           'profiles can be determined.')

        self._clusters['representative'] = False
        with Outputs(self._fpath_gen) as f:
            ti = f.time_index
        cols = self._clusters.cluster_id.unique()
        profile_df = pd.DataFrame(index=ti, columns=cols)

        for i, df in self._clusters.groupby('cluster_id'):
            mask = (df['included_frac'] > self.include_threshold)
            if any(mask):
                rep = df[mask].sort_values(by='rank').iloc[0, :]
                gen_gid = rep['gen_gid']
                mask = (self._clusters['gen_gid'] == gen_gid)
                self._clusters.loc[mask, 'representative'] = True

                with Outputs(self._fpath_gen) as f:
                    profile_df.loc[:, i] = f['cf_profile', :, gen_gid]

        return profile_df

    def make_cluster_summary(self):
        """Make a summary dataframe with cluster_id primary key.

        Returns
        -------
        s : pd.DataFrame
            Summary dataframe with a row for each cluster id.
        """

        if 'included_frac' not in self._clusters:
            raise KeyError('Exclusions must be applied before representative '
                           'profiles can be determined.')
        if 'representative' not in self._clusters:
            raise KeyError('Representative profiles must be determined before '
                           'summary table can be created.')

        ind = self._clusters.cluster_id.unique()
        cols = ['latitude',
                'longitude',
                'included_frac',
                'included_area_km2',
                'representative_gid',
                'representative_gen_gid']
        s = pd.DataFrame(index=ind, columns=cols)

        for i, df in self._clusters.groupby('cluster_id'):
            s.loc[i, 'latitude'] = df['latitude'].mean()
            s.loc[i, 'longitude'] = df['longitude'].mean()
            s.loc[i, 'included_frac'] = df['included_frac'].mean()
            s.loc[i, 'included_area_km2'] = df['included_area_km2'].sum()

            if df['representative'].any():
                s.loc[i, 'representative_gid'] = \
                    df.loc[df['representative'], 'gid'].values[0]
                s.loc[i, 'representative_gen_gid'] = \
                    df.loc[df['representative'], 'gen_gid'].values[0]

        return s

    def make_shape_files(self, out_dir):
        """Make shape files for all clusters.

        Parameters
        ----------
        out_dir : str
            Directory to dump output files. New shape_files subdir will be
            created in out_dir.
        """

        shp_dir = os.path.join(out_dir, 'shape_files/')
        if not os.path.exists(shp_dir):
            os.makedirs(shp_dir)

        # Geopandas doesnt like writing booleans
        if 'representative' in self._clusters:
            self._clusters['representative'] = \
                self._clusters['representative'].astype(int)

        for cluster_id, df in self._clusters.groupby('cluster_id'):
            fpath = os.path.join(shp_dir, '{}.shp'.format(cluster_id))
            RPMClusters._generate_shapefile(df, fpath)

        if 'representative' in self._clusters:
            self._clusters['representative'] = \
                self._clusters['representative'].astype(bool)

    def export_all(self, out_dir, tag=None):
        """Run RPM output algorithms and write to CSV's.

        Parameters
        ----------
        out_dir : str
            Directory to dump output files.
        tag : str | None
            Optional tag to add to the csvs being saved.
        """

        fn_out = 'rpm_cluster_output.csv'
        fn_pro = 'rpm_profiles.csv'
        fn_sum = 'rpm_cluster_summary.csv'

        if tag is not None:
            fn_out = fn_out.replace('.csv', '_{}.csv'.format(tag))
            fn_pro = fn_pro.replace('.csv', '_{}.csv'.format(tag))
            fn_sum = fn_sum.replace('.csv', '_{}.csv'.format(tag))

        if 'included_frac' not in self._clusters:
            self._clusters = self.apply_exclusions()

        self.make_profile_df().to_csv(os.path.join(out_dir, fn_pro))
        logger.info('Saved {}'.format(fn_pro))
        self.make_cluster_summary().to_csv(os.path.join(out_dir, fn_sum))
        logger.info('Saved {}'.format(fn_sum))
        self._clusters.to_csv(os.path.join(out_dir, fn_out), index=False)
        logger.info('Saved {}'.format(fn_out))
        self.make_shape_files(out_dir)
        logger.info('Saved shape files to {}'
                    .format(os.path.join(out_dir, 'shape_files/')))


if __name__ == '__main__':

    from reV.utilities.loggers import init_logger
    init_logger(__name__, log_level='DEBUG', log_file=None)

    fn_rpm_clusters = '/scratch/gbuster/rev/rpm/rpm_clusters.csv'
    fpath_excl = '/scratch/gbuster/rev/rpm/_windready_conus.tif'
    fpath_techmap = '/scratch/gbuster/rev/rpm/tech_map_conus.h5'
    dset_techmap = 'wtk'
    fpath_gen = '/projects/naris/extreme_events/generation/v90_full_ca_2012.h5'
    rpm_clusters = pd.read_csv(fn_rpm_clusters)

    rpme = RPMOutput(rpm_clusters, fpath_excl, fpath_techmap, dset_techmap,
                     fpath_gen)
    rpme.export_all('/scratch/gbuster/rev/rpm/')
