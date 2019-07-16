"""
RPM Clustering Module

#### Sample usage:

h5ls /projects/naris/extreme_events/generation/pv_ca_2012.h5
cf_profile               Dataset {8760, 26010}
meta                     Dataset {26010}
time_index               Dataset {8760}

import h5py

fname = '/projects/naris/extreme_events/generation/pv_ca_2012.h5'
data = h5py.File(fname, 'r')
meta = pd.DataFrame(data['meta'][...])
cf_profile = data['cf_profile'][...]

# Initiate
clusters = RPMClusters("california",
                       meta['latitude'], meta['longitude'],
                       cf_profile)

# Wavelets
clusters.calculate_wavelets()

# Cluster
clustering_args = {'k': 20}
clusters.apply_clustering(clustering_args, method="kmeans")

# Representative Profiles
clusters.get_representative_timeseries()
clusters.get_centroids_meta()

# Verification & Validation
clusters.pca_validation(plot=True)

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

    def __init__(self, region_name, latitude, longitude, ts_arrays, gids=None):
        """
        Parameters
        ----------
        meta: Meta data with gid, latitude, longitude
        ts_arrays: Timeseries profiles to cluster with RPMWavelets
        """

        self.region_name = region_name
        self.gids = gids
        self.meta = pd.DataFrame(list(zip(latitude, longitude)),
                                 columns=['latitude', 'longitude'])
        if self.meta.shape[0] != ts_arrays.shape[0]:
            ts_arrays = ts_arrays.T
        self.shape = ts_arrays.shape
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

        if self.coefficients is None:
            logger.warning('wavelet coefficients do not exist')
            return None

        if plot is False:
            return self.get_pca(self.coefficients, n_components=n_components)

        if plot is True:

            pca_df_2 = self.get_pca(self.coefficients, n_components=2)
            pca_df_3 = self.get_pca(self.coefficients, n_components=3)
            for c, dim in [('R', 'PC 1'), ('G', 'PC 2'), ('B', 'PC 3')]:
                col = pca_df_3[dim]
                pca_df_3[c] = (col - col.min()) / (col.max() - col.min())

            _, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

            ax[0].scatter(self.meta['longitude'], self.meta['latitude'],
                          c=pca_df_3[['R', 'G', 'B']].to_numpy(), s=8)
            ax[0].set_xlabel('Longitude')
            ax[0].set_ylabel('Latitude')

            ax[1].scatter(self.meta['longitude'], self.meta['latitude'],
                          c=self.labels, cmap="rainbow", s=8)
            ax[1].set_xlabel('Longitude')
            ax[1].set_ylabel('Latitude')

            ax[2].scatter(pca_df_2['PC 1'], pca_df_2['PC 2'],
                          c=self.labels, cmap="rainbow", s=5, alpha=0.5)
            ax[2].set_xlabel('PC 1')
            ax[2].set_ylabel('PC 2')

            plt.tight_layout()
            plt.savefig('{}_{}.png'.format(self.region_name, self.n_clusters))
            return None

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
