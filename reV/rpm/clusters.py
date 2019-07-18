"""
RPM Clustering Module

#### Sample usage:

import h5py

fname = '/projects/naris/extreme_events/generation/pv_ca_2012.h5'
fname = '/projects/naris/extreme_events/generation/v90_full_ca_2012.h5'
data = h5py.File(fname, 'r')
meta = pd.DataFrame(data['meta'][...][::15])
cf_profile = data['cf_profile'][...][:,::15]

# Option: Extract county only
# mask = meta.county == b'San Diego'
# meta = meta[mask]
# cf_profile = cf_profile[:,mask]

# Initiate
clusters = RPMClusters("california_wind",
                       meta['latitude'], meta['longitude'],
                       cf_profile)

# Wavelets
clusters.calculate_wavelets(normalize=True)

# Cluster
clustering_args = {'k': 6}
clusters.apply_clustering(clustering_args, method="kmeans",
                          include_spatial=True)
clusters.recluster_by_centroid()

# Representative Profiles
clusters.get_representative_timeseries()

# Verification & Validation
clusters.pca_validation(plot=True)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pywt

import logging

logger = logging.getLogger(__name__)


class ClusteringMethods:
    """ Base class of clustering methods """

    @staticmethod
    def kmeans(data, args):
        """ Cluster based on kmeans methodology """

        n_clusters = args['k']
        kmeans = KMeans(n_clusters=n_clusters)
        results = kmeans.fit(data)
        return n_clusters, results.labels_, results.cluster_centers_


class RPMClusters:
    """ Base class for RPM clusters """

    def __init__(self, region_name, latitude, longitude, ts_arrays, gids=None):
        """
        Parameters
        ----------
        meta: Meta data with gid, latitude, longitude
        ts_arrays: Timeseries profiles to cluster with RPMWavelets
        """

        # Meta
        self.region_name = region_name
        self.gids = gids
        self.meta = pd.DataFrame(list(zip(latitude, longitude)),
                                 columns=['latitude', 'longitude'])
        if self.meta.shape[0] != ts_arrays.shape[0]:
            ts_arrays = ts_arrays.T

        # Data
        self.ts_arrays = ts_arrays
        self.n_locations = ts_arrays.shape[0]
        self.shape = ts_arrays.shape

        # Wavelets
        self.coefficients = None
        self.coefficients_normalized = None
        self.normalized = False

        # Clusters
        self.included_spatial = False
        self.clustering_data = None
        self.cluster_method = None
        self.n_clusters = None
        self.labels = None
        self.labels_centroid = None
        self.centers_data = None
        self.ranks = None
        self.representative_timeseries = None
        self.centroids_meta = None

    def calculate_wavelets(self, normalize=False):
        """ Calculates the wavelet coefficients of each
            timeseries within ndarray """

        logger.info('Applying wavelets to region: {}'.format(self.region_name))

        coefficients = RPMWavelets.get_dwt_coefficients(self.ts_arrays)
        self.coefficients = coefficients
        if normalize is True:
            coefficients = self.normalize(coefficients)
            self.normalized = True
            self.coefficients_normalized = coefficients
        return coefficients

    def apply_clustering(self, args, method="kmeans",
                         include_spatial=False, spatial_weight=50):
        """ Apply a clustering method to <self.ts_arrays> """

        coefficients = self._retrieve_coefficients()
        if not hasattr(ClusteringMethods, method):
            logger.warning('method does not exist')
            return None
        if coefficients is None:
            logger.warning('coefficients do not exist')
            return None
        if include_spatial is True:
            if self.normalized is False:
                logger.warning('Cannot include spatial clustering without'
                               ' having normalized the coefficients')
                return None

        logger.info('Applying {} clustering'
                    ' to region: {}'.format(method, self.region_name))

        if include_spatial is True:
            spatial = self.meta[['latitude', 'longitude']].to_numpy()
            spatial = spatial_weight * self.normalize(spatial)
            clustering_data = np.concatenate((spatial, coefficients), axis=1)
            self.included_spatial = True

        else:
            clustering_data = coefficients

        clustering_function = getattr(ClusteringMethods, method)
        results = clustering_function(clustering_data, args)
        self.n_clusters, self.labels, self.centers_data = results
        self.meta['cluster_id'] = self.labels
        self.cluster_method = method
        self.clustering_data = clustering_data
        return self.labels

    @staticmethod
    def normalize(data, axis=0, on_max_std=False):
        """ Normalize an ndarray """

        means = np.nanmean(data, axis=axis)
        stds = np.nanstd(data, axis=axis)
        if on_max_std is True:
            data = (data - means) / np.max(stds)
        else:
            data = (data - means) / stds
        data[np.isnan(data)] = 0
        return data

    def recluster_by_centroid(self):
        """ Recluster points with new labels based on centroid
        nearest neighbor """

        logger.info('Reclustering by centroid'
                    ' for region: {}'.format(self.region_name))

        if self.centroids_meta is None:
            self.get_centroids_meta()
        centroids = self.centroids_meta[['latitude', 'longitude']]
        meta = self.meta[['latitude', 'longitude']]
        tree = cKDTree(centroids)
        _, indices = tree.query(meta)
        self.labels_centroid = indices
        return self.labels_centroid

    @staticmethod
    def get_pca(data, n_components=2):
        """ Principal Component Analysis """

        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(data)
        columns = ['PC {}'.format(i + 1) for i in range(n_components)]
        principal_df = pd.DataFrame(data=principal_components,
                                    columns=columns)
        return principal_df

    def pca_validation(self, n_components=2, plot=False):
        """ Validate clustering assumptions with
        principal component analysis """

        coefficients = self._retrieve_coefficients()
        if coefficients is None:
            logger.warning('wavelet coefficients do not exist')
            return None

        if plot is False:
            return self.get_pca(coefficients, n_components=n_components)

        if plot is True:

            # PCA
            pca_df_2 = self.get_pca(coefficients, n_components=2)
            pca_df_2['PC 1'] = pca_df_2['PC 1'] / pca_df_2['PC 1'].max()
            pca_df_2['PC 2'] = pca_df_2['PC 2'] / pca_df_2['PC 2'].max()
            pca_df_3 = self.get_pca(coefficients, n_components=3)
            for c, dim in [('R', 'PC 1'), ('G', 'PC 2'), ('B', 'PC 3')]:
                col = pca_df_3[dim]
                pca_df_3[c] = (col - col.min()) / (col.max() - col.min())

            # Plotting
            _, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
            s = 8

            ax[0].scatter(self.meta['longitude'], self.meta['latitude'],
                          c=pca_df_3[['R', 'G', 'B']].to_numpy(), s=s)
            ax[0].set_xlabel('Longitude')
            ax[0].set_ylabel('Latitude')

            ax[1].scatter(pca_df_2['PC 1'], pca_df_2['PC 2'],
                          c=self.labels, cmap="rainbow", s=s, alpha=0.5)
            ax[1].set_xlabel('PC 1')
            ax[1].set_ylabel('PC 2')

            ax[2].scatter(self.meta['longitude'], self.meta['latitude'],
                          c=self.labels, cmap="rainbow", s=s)
            ax[2].set_title('Clustering ({})'.format(self.cluster_method))
            ax[2].set_xlabel('Longitude')
            ax[2].set_ylabel('Latitude')

            ax[3].scatter(self.meta['longitude'], self.meta['latitude'],
                          c=self.labels_centroid, cmap="rainbow", s=s)
            ax[3].set_title('Reclustered On Centroid')
            ax[3].set_xlabel('Longitude')
            ax[3].set_ylabel('Latitude')

            plt.tight_layout()
            plt.savefig('{}_{}.png'.format(self.region_name, self.n_clusters))
            return None

    def _retrieve_coefficients(self):
        """ Pull the coefficients that were used for clustering """
        if self.normalized is True:
            return self.coefficients_normalized
        else:
            return self.coefficients

    def _get_ranks(self):
        """ Determine the rank of each location within all clusters
        based on the mean square errors """

        logger.info('Getting ranks for region: {}'.format(self.region_name))

        if self.centers_data is None:
            logger.warning('no clustering has been applied')
            return None
        ranks = np.zeros(self.n_locations, dtype='int')
        for cluster_ind in range(self.n_clusters):
            cluster_mask = self.labels == cluster_ind
            cluster_data = self.clustering_data[cluster_mask]
            n_locations = len(cluster_data)
            centers_data = self.centers_data[cluster_ind]
            cc_rep = [centers_data for _ in range(n_locations)]
            centers_data = np.array(cc_rep)
            mse = (np.square(cluster_data,
                             centers_data)).mean(axis=1)
            ranks[cluster_mask] = ss.rankdata(mse)
        self.ranks = ranks
        return self.ranks

    def get_representative_timeseries(self):
        """ Determine the representative timeseries for each cluster
        based on the location with the lowest mean square error """

        if self.centers_data is None:
            logger.warning('no clustering has been applied')
            return None
        self._get_ranks()
        highest_ranking = self.ranks == 1
        self.representative_timeseries = self.ts_arrays[highest_ranking]
        return self.representative_timeseries

    def get_centroids_meta(self):
        """ Determine the representative spatial centroids of each cluster """

        if self.labels is None:
            logger.warning('no clustering has been applied')
            return None
        meta = self.meta
        centroids_meta = meta.groupby('cluster_id').mean()
        cols = ['latitude', 'longitude', 'cluster_id']
        centroids_meta = centroids_meta.reset_index().reindex(columns=cols)
        self.centroids_meta = centroids_meta


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
        logger.info('Calculating wavelet coefficients'
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
                                dtype=np.int)

        logger.debug('{c:d} coefficients'.format(c=_coefficient_count))

        _i_start = 0
        for _index in indices:
            _i_end = _i_start + x[_index].shape[1]
            _combined_wc[:, _i_start:_i_end] = np.round(x[_index], 2) * 100
            _i_start = _i_end

        return _combined_wc
