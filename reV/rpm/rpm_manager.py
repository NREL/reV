# -*- coding: utf-8 -*-
"""
Pipeline between reV and RPM
"""
import psutil
import os
import concurrent.futures as cf
import logging
import pandas as pd
from warnings import warn

from reV.handlers.outputs import Outputs
from reV.rpm.clusters import RPMClusters
from reV.rpm.rpm_output import RPMOutput
from reV.utilities.exceptions import RPMValueError, RPMRuntimeError

logger = logging.getLogger(__name__)


class RPMClusterManager:
    """
    RPM Cluster Manager:
    - Extracts gids for all RPM regions
    - Runs RPMClusters in parallel for all regions
    - Save results to disk
    """
    def __init__(self, cf_profiles, rpm_meta, rpm_region_col=None,
                 parallel=True):
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
        parallel : bool | int
            Flag to apply exclusions in parallel. Integer is interpreted as
            max number of workers. True uses all available.
        """
        if rpm_region_col is not None:
            logger.info('Initializing RPM clustering on regional column "{}".'
                        .format(rpm_region_col))

        self._cf_h5 = cf_profiles
        self._rpm_regions = self._map_rpm_regions(rpm_meta,
                                                  region_col=rpm_region_col)

        self.parallel = parallel
        if self.parallel is True:
            self.max_workers = os.cpu_count()
        elif self.parallel is False:
            self.max_workers = 1
        else:
            self.max_workers = self.parallel

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

            if region_meta['gen_gid'].empty:
                wmsg = ('Could not locate any generation in region "{}".'
                        .format(region))
                warn(wmsg)
                logger.warning(wmsg)
            else:
                region_map['cluster_num'] = clusters[0]
                region_map['gen_gids'] = region_meta['gen_gid'].values
                region_map['gids'] = region_meta.index.values
                rpm_regions[region] = region_map

        return rpm_regions

    def _cluster(self, **kwargs):
        """
        Cluster all RPM regions

        Parameters
        ----------
        kwargs : dict
            RPMCluster kwargs
        """
        if self.max_workers > 1:
            future_to_region = {}
            with cf.ProcessPoolExecutor(max_workers=self.max_workers) as exe:
                for region, region_map in self._rpm_regions.items():
                    logger.info('Kicking off clustering for "{}".'
                                .format(region))
                    clusters = region_map['cluster_num']
                    gen_gids = region_map['gen_gids']

                    future = exe.submit(RPMClusters.cluster, self._cf_h5,
                                        gen_gids, clusters, **kwargs)
                    future_to_region[future] = region

                for i, future in enumerate(cf.as_completed(future_to_region)):
                    mem = psutil.virtual_memory()
                    region = future_to_region[future]
                    logger.info('Finished clustering "{}", {} out of {}. '
                                'Memory usage is {:.2f} out of {:.2f} GB.'
                                .format(region, i + 1, len(future_to_region),
                                        mem.used / 1e9, mem.total / 1e9))
                    result = future.result()
                    self._rpm_regions[region].update({'clusters': result})

        else:
            for region, region_map in self._rpm_regions.items():
                logger.info('Kicking off clustering for "{}".'.format(region))
                clusters = region_map['cluster_num']
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
            r_df['gid'] = r_dict['gids']
            rpm_clusters.append(r_df)

        rpm_clusters = pd.concat(rpm_clusters, sort=False)
        rpm_clusters = rpm_clusters.reset_index(drop=True)

        if 'geometry' in rpm_clusters:
            rpm_clusters = rpm_clusters.drop('geometry', axis=1)

        return rpm_clusters

    @classmethod
    def run(cls, rpm_meta, fpath_gen, fpath_excl, fpath_techmap,
            dset_techmap, out_dir, include_threshold=0.001, job_tag=None,
            rpm_region_col=None, parallel=True, **cluster_kwargs):
        """
        RPM Cluster Manager:
        - Extracts gen_gids for all RPM regions
        - Runs RPMClusters in parallel for all regions
        - Save results to disk

        Parameters
        ----------
        rpm_meta : pandas.DataFrame | str
            DataFrame or path to .csv or .json containing the RPM meta data:
            - Regions of interest
            - # of clusters per region
            - cf or resource GIDs if region is not in default meta data
        fpath_gen : str
            Path to reV .h5 files containing desired capacity factor profiles
        fpath_excl : str | None
            Filepath to exclusions data (must match the techmap grid).
            None will not apply exclusions.
        fpath_techmap : str | None
            Filepath to tech mapping between exclusions and resource data.
            None will not apply exclusions.
        dset_techmap : str
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data.
        out_dir : str
            Directory to dump output files.
        include_threshold : float
            Inclusion threshold. Resource pixels included more than this
            threshold will be considered in the representative profiles.
            Set to zero to find representative profile on all resource, not
            just included.
        job_tag : str | None
            Optional name tag to add to the output files.
            Format is "rpm_cluster_output_{tag}.csv".
        rpm_region_col : str | Nonetype
            If not None, the meta-data filed to map RPM regions to
        parallel : bool | int
            Flag to apply exclusions in parallel. Integer is interpreted as
            max number of workers. True uses all available.
        **cluster_kwargs : dict
            RPMClusters kwargs
        """

        # intermediate job file
        f_int = os.path.join(out_dir, 'rpm_cluster_temp.csv')
        if job_tag is not None:
            f_int = f_int.replace('.csv', '_{}.csv'.format(job_tag))

        if not os.path.exists(f_int):
            rpm = cls(fpath_gen, rpm_meta, rpm_region_col=rpm_region_col,
                      parallel=parallel)
            rpm._cluster(**cluster_kwargs)
            rpm_clusters = rpm._combine_region_clusters(rpm._rpm_regions)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            rpm_clusters.to_csv(f_int, index=False)

        else:
            logger.info('Importing intermediate cluster results from: {}'
                        .format(f_int))
            rpm_clusters = f_int
            rpm = None

        RPMOutput.process_outputs(rpm_clusters, fpath_excl, fpath_techmap,
                                  dset_techmap, fpath_gen, out_dir,
                                  job_tag=job_tag, parallel=parallel,
                                  include_threshold=include_threshold,
                                  cluster_kwargs=cluster_kwargs)
        logger.info('reV-to-RPM processing is complete.')
        return rpm
