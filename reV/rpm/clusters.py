"""

Sample usage:

    inputs = np.array([
        np.sin(np.linspace(40,1,200)),
        np.sin(np.linspace(1,40,200)),
        np.cos(np.linspace(40,1,200)),
        np.cos(np.linspace(1,40,200))
        ])

    clusters = RPM_Clusters(inputs)
    coefficients = clusters.calculate_wavelets()
    clustering_args = {'k': 2}
    results = clusters.apply_clustering(clustering_args, method="kmeans")

"""

# Essentials
import numpy as np
from sklearn.cluster import KMeans
import pywt

# Support
import logging

logger = logging.getLogger(__name__)


class Clustering_Methods:
    """ Base class of clustering methods """

    @staticmethod
    def kmeans(data, args):
        """ Cluster based on kmeans methodology """
        kmeans = KMeans(n_clusters=args['k'])
        results = kmeans.fit(data)
        return results.labels_


class RPM_Clusters:
    """Base class for RPM clusters"""

    def __init__(self, ts_profiles):
        """
        Parameters
        ----------
        ts_profiles:
        """
        self.ts_profiles = ts_profiles
        self.coefficients = None
        self.cluster_labels = None

    def calculate_wavelets(self):
        """ Calculates the wavelet coefficients of each
            timeseries within ndarray """

        self.coefficients = RPM_Wavelets.get_dwt_coefficients(self.ts_profiles)
        return self.coefficients

    def apply_clustering(self, args, method="kmeans"):
        """ Apply a clustering method to <self.ts_profiles> """

        if not hasattr(Clustering_Methods, method):
            logger.warning('method does not exist')
            return None
        if self.coefficients is None:
            logger.warning('coefficients do not exist')
            return None

        clustering_function = getattr(Clustering_Methods, method)
        self.cluster_labels = clustering_function(self.ts_profiles, args)
        return self.cluster_labels


class RPM_Wavelets:
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
