"""
RPM PCA Mapping
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import logging

logger = logging.getLogger(__name__)


class RPMPCA:
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
