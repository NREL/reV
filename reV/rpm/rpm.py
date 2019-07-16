"""
Pipeline between reV and RPM
"""
import concurrent.futures as cf
import logging
import pandas as pd

from reV.handlers.outputs import Outputs
from reV.rpm.clusters import RPMClusters
from reV.utilities.exceptions import RPMValueError, RPMRuntimeError

logger = logging.getLogger(__name__)


class RPM:
    """
    Entry point for reV to RPM pipeline. Pipeline:
    - Creates 'regular' regional RPM 'clusters'
    - Ranks resource pixels against the cluster 'average'
    - Extracts representative (post-exclusion) profiles
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
            region_map['gen_gids'] = region_meta['gen_gid']
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
                    gids = region_map['gen_gids']

                    future = executor.submit(RPMClusters.cluster, self._cf_h5,
                                             gids, clusters, **kwargs)
                    future_to_region[future] = region

                for future in cf.as_completed(future_to_region):
                    region = future_to_region[future]
                    result = future.result()
                    self._rpm_regions[region].update({'clusters': result})

        else:
            for region, region_map in self._rpm_regions.items():
                clusters = region_map['clusters']
                gids = region_map['gen_gids']
                result = RPMClusters.cluster(self._cf_h5, gids, clusters,
                                             **kwargs)
                self._rpm_regions[region].update({'clusters': result})
