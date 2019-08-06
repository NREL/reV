# -*- coding: utf-8 -*-
"""
RPM Clustering Module

#### Sample usage:

import h5py

fname = '/projects/naris/extreme_events/generation/pv_ca_2012.h5'
fname = '/projects/naris/extreme_events/generation/v90_full_ca_2012.h5'
with h5py.File(fname, 'r') as f:
    wind_meta = pd.DataFrame(f['meta'][...])
    wind_gen_gids = wind_meta.iloc[::15].index.values

# Initiate
clusters = RPMClusters(fname, wind_gen_gids, n_clusters=6)

# Cluster on wavelet coefficients
labels = clusters._cluster_coefficients(**method_kwargs)

# Optimize clusters by minimizing distance and rmse to cluster centers
new_labels = clusters._dist_rank_optimization(**dist_rmse_kwargs)

# Run multistep clustering
clusters._cluster(**kwargs)

# classmethod
cluster_df = RPMClusters.cluster(fname, wind_gen_gids, n_clusters=6, **kwargs)

# Shapefiles
RPMClusters._generate_shapefile(clusters.meta, fpath='./test.shp')


#### Testing:

from reV.rpm.rpm import *
import h5py

fname = '/projects/naris/extreme_events/generation/pv_ca_2012.h5'
with h5py.File(fname, 'r') as f:
    wind_meta = pd.DataFrame(f['meta'][...])
    wind_gen_gids = wind_meta.index.values

cluster_df = RPMClusters.cluster(fname, wind_gen_gids, n_clusters=6)

"""
import logging
import pywt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.spatial import cKDTree
import geopandas as gpd
from shapely.geometry import Point

from reV.handlers.outputs import Outputs

logger = logging.getLogger(__name__)


class ClusteringMethods:
    """ Base class of clustering methods """

    @staticmethod
    def kmeans(data, **kwargs):
        """ Cluster based on kmeans methodology """

        kmeans = KMeans(**kwargs)
        results = kmeans.fit(data)
        return results.labels_


class RPMClusters:
    """ Base class for RPM clusters """
    def __init__(self, cf_h5_path, gen_gids, n_clusters):
        """
        Parameters
        ----------
        cf_h5_path : str
            Path to reV .h5 files containing desired capacity factor profiles
        gen_gids : list | ndarray
            List or vector of gen_gids to cluster on
        n_clusters : int
            Number of clusters to identify
        """
        self._meta, self._coefficients = self._parse_data(cf_h5_path,
                                                          gen_gids)
        self._n_clusters = n_clusters

    @property
    def coefficients(self):
        """
        Returns
        -------
        _coefficients : ndarray
            Array of wavelet coefficients for each gen_gid
        """
        return self._coefficients

    @property
    def meta(self):
        """
        Returns
        -------
        _meta : pandas.DataFrame
            DataFrame of meta data:
            - gen_gid
            - latitude
            - longitude
            - cluster_id
            - rank
        """
        return self._meta

    @property
    def n_clusters(self):
        """
        Returns
        -------
        _n_clusters : int
            Number of clusters
        """
        return self._n_clusters

    @property
    def cluster_coefficients(self):
        """
        Returns
        -------
        cluster_coeffs : ndarray
            Representative coefficients for each cluster
        """
        cluster_coeffs = None
        if 'cluster_id' in self._meta:
            cluster_coeffs = []
            for _, cdf in self._meta.groupby('cluster_id'):
                idx = cdf.index.values
                cluster_coeffs.append(self.coefficients[idx].mean(axis=0))

            cluster_coeffs = np.array(cluster_coeffs)

        return cluster_coeffs

    @property
    def cluster_ids(self):
        """
        Returns
        -------
        cluster_ids : ndarray
            Cluster cluster_id for each gen_gid
        """
        cluster_ids = None
        if 'cluster_id' in self._meta:
            cluster_ids = self._meta['cluster_id'].values
        return cluster_ids

    @property
    def cluster_coordinates(self):
        """
        Returns
        -------
        cluster_coords : ndarray
            lon, lat coordinates of the centroid of each cluster
        """
        cluster_coords = None
        if 'cluster_id' in self._meta:
            cluster_coords = self._meta.groupby('cluster_id')
            cluster_coords = cluster_coords[['longitude', 'latitude']].mean()
            cluster_coords = cluster_coords.values

        return cluster_coords

    @property
    def coordinates(self):
        """
        Returns
        -------
        coords : ndarray
            lon, lat coordinates for each gen_gid
        """
        coords = self._meta[['longitude', 'latitude']].values
        return coords

    @staticmethod
    def _parse_data(cf_h5_path, gen_gids):
        """
        Extract lat, lon coordinates for given gen_gids
        Extract and convert cf_profiles into wavelet coefficients

        Parameters
        ----------
        cf_h5_path : str
            Path to reV .h5 files containing desired capacity factor profiles
        gen_gids : list | ndarray
            List or vector of gen_gids to cluster on
        """

        with Outputs(cf_h5_path, mode='r', unscale=False) as cfs:
            meta = cfs.meta.loc[gen_gids, ['latitude', 'longitude']]
            gid_slice, gid_idx = RPMClusters._gid_pos(gen_gids)
            coeff = cfs['cf_profile', :, gid_slice][:, gid_idx]

        meta['gen_gid'] = gen_gids
        cols = ['gen_gid', 'latitude', 'longitude']
        meta = meta[cols].reset_index(drop=True)
        coeff = RPMClusters._calculate_wavelets(coeff.T)
        return meta, coeff

    @staticmethod
    def _gid_pos(gen_gids):
        """
        Parameters
        ----------
        gen_gids : list | ndarray
            List or vector of gen_gids to cluster on

        Returns
        -------
        gid_slice : slice
            Slice that encompasses the entire gen_gid range
        gid_idx : ndarray
            Adjusted list to extract gen_gids of interest from slice
        """
        if isinstance(gen_gids, list):
            gen_gids = np.array(gen_gids)

        s = gen_gids.min()
        e = gen_gids.max() + 1
        gid_slice = slice(s, e, None)
        gid_idx = gen_gids - s

        return gid_slice, gid_idx

    @staticmethod
    def _calculate_wavelets(ts_arrays):
        """ Calculates the wavelet coefficients of each
            timeseries within ndarray """
        coefficients = RPMWavelets.get_dwt_coefficients(ts_arrays)
        return coefficients

    def _cluster_coefficients(self, method="kmeans", **kwargs):
        """ Apply a clustering method to <self.ts_arrays> """
        logger.debug('Applying {} clustering '.format(method))

        c_func = getattr(ClusteringMethods, method)
        labels = c_func(self.coefficients, n_clusters=self.n_clusters,
                        **kwargs)
        return labels

    @staticmethod
    def _normalize_values(arr, norm=None, axis=None):
        """
        Normalize values in array by column
        Parameters
        ----------
        arr : ndarray
            ndarray of values extracted from meta
            shape (n samples, with m features)
        norm : str
            Normalization method to use, see sklearn.preprocessing.normalize
        Returns
        ---------
        arr : ndarray
            array with values normalized by column
            shape (n samples, with m features)
        """
        if norm:
            arr = normalize(arr, norm=norm, axis=axis)
        else:
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(float)

            min_all = arr.min(axis=axis)
            max_all = arr.max(axis=axis)
            range_all = max_all - min_all
            if axis is not None:
                pos = range_all == 0
                range_all[pos] = 1

            arr -= min_all
            arr /= range_all

        return arr

    def _dist_rank_optimization(self, **kwargs):
        """
        Re-cluster data by minimizing the sum of the:
        - distance between each point and each cluster centroid
        - distance between each point and each
        """
        cluster_coeffs = self.cluster_coefficients
        cluster_centroids = self.cluster_coordinates
        rmse = []
        dist = []
        for centroid, rep_coeffs in zip(cluster_centroids, cluster_coeffs):
            c_rmse = np.mean((self.coefficients - rep_coeffs) ** 2,
                             axis=1) ** 0.5
            rmse.append(c_rmse)
            c_dist = np.linalg.norm(self.coordinates - centroid, axis=1)
            dist.append(c_dist)

        rmse = self._normalize_values(np.array(rmse), **kwargs)
        dist = self._normalize_values(np.array(dist), **kwargs)
        err = (dist + rmse**2)
        new_labels = np.argmin(err, axis=0)
        return new_labels

    def _contiguous_filter(self):
        """
        Re-classify clusters by making contigous cluster polygons
        """
        meta = self._meta

        geometry = [Point(xy) for xy in zip(meta.longitude, meta.latitude)]
        gdf_points = gpd.GeoDataFrame(meta, geometry=geometry)

        clusters = self._get_cluster_geom(gdf_points,
                                          drop_islands=True, add_buffer=False)
        clusters.to_file('clusters.shp')
        gdf_points.to_file('gdf_points.shp')

        intersected = gpd.sjoin(gdf_points, clusters,
                                how="left", op='intersects')

        # drop duplicate rows
        gid_counts = intersected.groupby('gen_gid_left').size()
        duplicate_gids = gid_counts[gid_counts > 1].index
        mask = intersected['gen_gid_left'].isin(duplicate_gids)
        intersected.loc[mask, 'cluster_id_right'] = None
        intersected = intersected.drop_duplicates(subset=['gen_gid_left'])

        mask = np.isnan(intersected.cluster_id_right)
        assigned = intersected[~mask].reset_index()
        unassigned = intersected[mask]

        lookup = assigned[['latitude_left', 'longitude_left']]
        target = unassigned[['latitude_left', 'longitude_left']]
        tree = cKDTree(lookup)
        _, inds = tree.query(target, k=1)

        for i, ind in enumerate(list(unassigned.index)):
            nearest_cluster_id = assigned.loc[inds[i], 'cluster_id_left']
            intersected.loc[ind, 'cluster_id_left'] = nearest_cluster_id

        new_labels = intersected['cluster_id_left']

        return new_labels

    @staticmethod
    def _generate_shapefile(meta, fpath):
        """
        Generate cluster polygons and save to shapefile
        """
        geometry = [Point(xy) for xy in zip(meta.longitude, meta.latitude)]
        gdf_points = gpd.GeoDataFrame(meta, geometry=geometry)

        clusters = RPMClusters._get_cluster_geom(gdf_points,
                                                 drop_islands=False,
                                                 add_buffer=True)
        clusters.to_file(fpath)
        return fpath

    @staticmethod
    def _get_cluster_geom(gdf_points, drop_islands=False, add_buffer=False):
        """
        Generate cluster polygons as a geopandas dataframe
        """
        lookup = gdf_points[['latitude', 'longitude']]
        tree = cKDTree(lookup)
        dists, _ = tree.query(lookup, k=2)
        mean_dist = dists.T[1].mean()
        gdf_poly = gdf_points.copy()
        gdf_poly.geometry = gdf_poly.geometry.buffer(mean_dist)
        clusters = gdf_poly.dissolve(by='cluster_id').reset_index()
        clusters.geometry = clusters.geometry.buffer(-mean_dist / 2)

        # Drop Islands
        if drop_islands is True:
            for index, _ in clusters.iterrows():
                geom = clusters.loc[index, 'geometry']
                if geom.geom_type == 'MultiPolygon':
                    clusters.loc[index, 'geometry'] = max(geom,
                                                          key=lambda a: a.area)

        # Buffer
        if add_buffer is True:
            clusters.geometry = clusters.geometry.buffer(-mean_dist)
            clusters.geometry = clusters.geometry.buffer(mean_dist)

        return clusters

    def _calculate_ranks(self):
        """ Determine the rank of each location within all clusters
        based on the mean square errors """
        cluster_coeffs = self.cluster_coefficients
        for i, cdf in self.meta.groupby('cluster_id'):
            pos = cdf.index
            rep_coeffs = cluster_coeffs[i]
            coeffs = self.coefficients[pos]
            err = np.mean((coeffs - rep_coeffs) ** 2, axis=1) ** 0.5
            rank = np.argsort(err)
            self._meta.loc[pos, 'rank'] = rank

    def _cluster(self, method='kmeans', method_kwargs=None,
                 optimize_dist_rank=True, contiguous_filter=True,
                 dist_rmse_kwargs=None):
        """
        Run three step RPM clustering procedure:
        1) Cluster on wavelet coefficients
        2) Clean up clusters by optimizing rmse and distance
        3) Remove islands using polygon intersection

        Parameters
        ----------
        method : str
            Method to use to cluster coefficients
        method_kwargs : dict
            Kwargs for running _cluster_coefficients
        dist_rmse_kwargs : dict
            Kwargs for running _dist_rank_optimization
        intersect_kwargs : dict
            Kwargs for running Rob's new method
        """

        if self.n_clusters <= 1:
            optimize_dist_rank = False
            contiguous_filter = False

        if method_kwargs is None:
            method_kwargs = {}
        labels = self._cluster_coefficients(method=method, **method_kwargs)
        self._meta['cluster_id'] = labels

        # Optimize Distance & Rank
        if optimize_dist_rank is True:
            if dist_rmse_kwargs is None:
                dist_rmse_kwargs = {}
            new_labels = self._dist_rank_optimization(**dist_rmse_kwargs)
            self._meta['cluster_id'] = new_labels

        # Apply contiguous filter
        if contiguous_filter is True:
            new_labels = self._contiguous_filter()
            self._meta['cluster_id'] = new_labels

        self._calculate_ranks()

    @classmethod
    def cluster(cls, cf_h5_path, region_gen_gids, n_clusters, **kwargs):
        """
        Entry point for RPMCluster to get clusters for a given region
        defined as a list | array of gen_gids

        Parameters
        ----------
        cf_h5_path : str
            Path to reV .h5 files containing desired capacity factor profiles
        region_gen_gids : list | ndarray
            List or vector of gen_gids to cluster on
        n_clusters : int
            Number of clusters to identify
        kwargs : dict
            Internal kwargs for clustering

        Returns
        -------
        out : pandas.DataFrame
            Cluster results: (gen_gid, lon, lat, cluster_id, rank)
        """
        clusters = cls(cf_h5_path, region_gen_gids, n_clusters)
        try:
            clusters._cluster(**kwargs)
        except Exception as e:
            logger.exception('Clustering failed on gen_gids {} through {}: {}'
                             .format(np.min(region_gen_gids),
                                     np.max(region_gen_gids), e))
        return clusters.meta


class RPMWavelets:
    """Base class for RPM wavelets"""

    @classmethod
    def get_dwt_coefficients(cls, x, wavelet='Haar', level=None, indices=None):
        """
        Collect wavelet coefficients for time series <x> using
        mother wavelet <wavelet> at levels <level>.

        :param x: [ndarray] time series values
        :param wavelet: [string] mother wavelet type
        :param level: [int] optional wavelet computation level
        :param indices: [(int, ...)] coefficient array levels to keep
        :return: [list] stacked coefficients at <indices>
        """

        # set mother
        _wavelet = pywt.Wavelet(wavelet)

        # multi-level with default depth
        logger.debug('Calculating wavelet coefficients'
                     ' with {w} wavelet'.format(w=_wavelet.family_name))

        _wavedec = pywt.wavedec(data=x, wavelet=_wavelet, axis=1, level=level)

        return cls._subset_coefficients(x=_wavedec,
                                        gid_count=x.shape[0],
                                        indices=indices)

    @staticmethod
    def _subset_coefficients(x, gid_count, indices=None):
        """
        Subset and stack wavelet coefficients
        :param x: [(ndarray, ...)] coefficients arrays
        :param gid_count: [int] number of area ID values
        :param indices: [(int, ...)]
        :return: [ndarray] stacked coefficients: converted to integers
        """

        indices = indices or range(0, len(x))

        _coefficient_count = 0
        for _index in indices:
            _shape = x[_index].shape
            _coefficient_count += _shape[1]

        _combined_wc = np.empty(shape=(gid_count, _coefficient_count),
                                dtype=np.int32)

        logger.debug('{c:d} coefficients'.format(c=_coefficient_count))

        _i_start = 0
        for _index in indices:
            _i_end = _i_start + x[_index].shape[1]
            _combined_wc[:, _i_start:_i_end] = np.round(x[_index])
            _i_start = _i_end

        return _combined_wc
