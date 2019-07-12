"""
RPM Clustering Module

#### Sample usage:

meta = pd.DataFrame(data=[(1,1,1),(2,1,2),(3,1,3),(4,1,4),
                          (5,2,1),(6,2,2),(7,2,3),(8,2,4)],
                    columns=['gids','latitude','longitude']).to_records()

ts_arrays = np.array([
    np.sin(np.linspace(1,40,200)),
    np.cos(np.linspace(1,40,200)),
    0.9 * np.sin(np.linspace(1,40,200)),
    1.1 * np.cos(np.linspace(1,40,200)),
    0.8 * np.sin(np.linspace(1,40,200)),
    1.2 * np.cos(np.linspace(1,40,200)),
    0.7 * np.sin(np.linspace(1,40,200)),
    1.3 * np.cos(np.linspace(1,40,200))
    ])

# Initiate
clusters = RPMClusters(meta['latitude'], meta['longitude'], ts_arrays)

# Wavelets
clusters.calculate_wavelets()

# Cluster
clustering_args = {'k': 2}
clusters.apply_clustering(clustering_args, method="kmeans")

# Representative Profiles
clusters.get_representative_timeseries()
clusters.get_centroids_meta()

# Verification & Validation
clusters.principal_component_analysis(plot=True)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
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

    def __init__(self, latitudes, longitudes, ts_arrays, gids=None):
        """
        Parameters
        ----------
        meta: Meta data with gid, latitude, longitude
        ts_arrays: Timeseries profiles to cluster with RPMWavelets
        """
        self.gids = gids
        self.meta = pd.DataFrame(list(zip(latitudes, longitudes)),
                                 columns=['latitude', 'longitude'])
        self.ts_arrays = ts_arrays
        self.n_locations = ts_arrays.shape[0]
        self.coefficients = None
        self.n_clusters = None
        self.labels = None
        self.centers_coefficients = None
        self.ranks = None
        self.representative_timeseries = None
        self.centroids_meta = None

    def calculate_wavelets(self):
        """ Calculates the wavelet coefficients of each
            timeseries within ndarray """

        self.coefficients = RPMWavelets.get_dwt_coefficients(self.ts_arrays)
        return self.coefficients

    def apply_clustering(self, args, method="kmeans"):
        """ Apply a clustering method to <self.ts_arrays> """

        if not hasattr(ClusteringMethods, method):
            logger.warning('method does not exist')
            return None
        if self.coefficients is None:
            logger.warning('coefficients do not exist')
            return None

        clustering_function = getattr(ClusteringMethods, method)
        results = clustering_function(self.coefficients, args)
        self.n_clusters, self.labels, self.centers_coefficients = results
        self.meta['cluster_id'] = self.labels
        return self.labels

    def principal_component_analysis(self, n_components=2, plot=False):
        """ Principal Component Analysis """
        if self.coefficients is None:
            logger.warning('wavelet coefficients do not exist')
            return None
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(self.coefficients)
        principal_df = pd.DataFrame(data=principal_components,
                                    columns=['PC 1', 'PC 2'])
        if plot is True:
            _, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].scatter(self.meta['latitude'], self.meta['longitude'],
                          c=self.labels, cmap="rainbow")
            ax[0].set_xlabel('Longitude')
            ax[0].set_ylabel('Latitude')
            ax[1].scatter(principal_df['PC 1'], principal_df['PC 2'],
                          c=self.labels, cmap="rainbow")
            ax[1].set_xlabel('PC 1')
            ax[1].set_ylabel('PC 2')
            plt.tight_layout()
            plt.show()
            return None
        else:
            return principal_df

    def _get_ranks(self):
        """ Determine the rank of each location within all clusters
        based on the mean square errors """
        if self.centers_coefficients is None:
            logger.warning('no clustering has been applied')
            return None
        ranks = np.zeros(self.n_locations, dtype='int')
        for cluster_ind in range(self.n_clusters):
            cluster_mask = self.labels == cluster_ind
            cluster_coefficients = self.coefficients[cluster_mask]
            n_locations = len(cluster_coefficients)
            centers_coefficients = self.centers_coefficients[cluster_ind]
            cc_rep = [centers_coefficients for _ in range(n_locations)]
            centers_coefficients = np.array(cc_rep)
            mse = (np.square(cluster_coefficients,
                             centers_coefficients)).mean(axis=1)
            ranks[cluster_mask] = ss.rankdata(mse)
        self.ranks = ranks
        return self.ranks

    def get_representative_timeseries(self):
        """ Determine the representative timeseries for each cluster
        based on the location with the lowest mean square error """
        if self.centers_coefficients is None:
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
