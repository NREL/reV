"""
Pipeline between reV and RPM
"""
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


class RPMExclusions:
    """Framework to apply exclusions to RPM clustering results."""

    def __init__(self, rpm_clusters, fpath_excl, fpath_techmap, dset_techmap):
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
        """

        self._clusters = rpm_clusters
        self._fpath_excl = fpath_excl
        self._fpath_techmap = fpath_techmap
        self._dset_techmap = dset_techmap

        self._excl_lat = None
        self._excl_lon = None

    def _get_tm_data(self, lat_slice, lon_slice):
        """Get the techmap data.

        Parameters
        ----------
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

        with Outputs(self._fpath_techmap) as tm:
            techmap = tm[self._dset_techmap, lat_slice, lon_slice]
        return techmap.flatten()

    def _get_excl_data(self, lat_slice, lon_slice, band=0):
        """Get the exclusions data from a geotiff file.

        Parameters
        ----------
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
            Exclusions data normalized from 0 to 1 (1 is included).
        """

        with Geotiff(self._fpath_excl) as excl:
            excl_data = excl[band, lat_slice, lon_slice]

        # infer exclusions that are scaled percentages from 0 to 100
        if excl_data.max() > 1:
            excl_data = excl_data.astype(np.float32)
            excl_data /= 100

        return excl_data

    def _get_lat_lon_slices(self, cluster_id, margin=5.0):
        """Get the slice args to locate exclusion/techmap data of interest.

        Parameters
        ----------
        cluster_id : str
            Single cluster ID of interest.
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

        cbox = self._get_coord_box(cluster_id)

        lat_mask = ((self.excl_lat > np.min(cbox['latitude']) - margin)
                    & (self.excl_lat < np.max(cbox['latitude']) + margin))
        lon_mask = ((self.excl_lon > np.min(cbox['longitude']) - margin)
                    & (self.excl_lon < np.max(cbox['longitude']) + margin))

        lat_locs = np.where(lat_mask)[0]
        lon_locs = np.where(lon_mask)[0]

        lat_slice = slice(np.min(lat_locs), 1 + np.max(lat_locs))
        lon_slice = slice(np.min(lon_locs), 1 + np.max(lon_locs))

        return lat_slice, lon_slice

    def _get_coord_box(self, cluster_id):
        """Get the RPM cluster latitude/longitude range.

        Parameters
        ----------
        cluster_id : str
            Single cluster ID of interest.

        Returns
        -------
        coord_box : dict
            Bounding box of the cluster:
                {'latitude': (lat_min, lat_max),
                 'longitude': (lon_min, lon_max)}
        """

        mask = (self._clusters['cluster_id'] == cluster_id)
        lat_range = (self._clusters.loc[mask, 'latitude'].min(),
                     self._clusters.loc[mask, 'latitude'].max())
        lon_range = (self._clusters.loc[mask, 'longitude'].min(),
                     self._clusters.loc[mask, 'longitude'].max())
        return {'latitude': lat_range, 'longitude': lon_range}

    @property
    def excl_lat(self):
        """Get a 1D array of latitudes of the exclusion grid.

        Returns
        -------
        _excl_lat : np.ndarray
            1D array representing the latitudes of each row in the
            exclusion grid.
        """

        if self._excl_lat is None:
            with Outputs(self._fpath_techmap) as f:
                shape, _, _ = f.get_dset_properties('longitude')
                self._excl_lat = f['latitude', :, int(shape[1] / 2)]
        return self._excl_lat

    @property
    def excl_lon(self):
        """Get a 1D array of longitudes of the exclusion grid.

        Returns
        -------
        _excl_lon : np.ndarray
            1D array representing the longitudes of each column in the
            exclusion grid.
        """

        if self._excl_lon is None:
            with Outputs(self._fpath_techmap) as f:
                shape, _, _ = f.get_dset_properties('longitude')
                self._excl_lon = f['longitude', int(shape[0] / 2), :]
        return self._excl_lon

    def calc_exclusions(self, cluster_id):
        """Calculate the exclusions for each resource GID in a cluster.

        Parameters
        ----------
        cluster_id : str
            Single cluster ID of interest.

        Returns
        -------
        single_cluster : pd.DataFrame
            DataFrame of resource gids in the cluster with added
            inclusions fraction field.
        """

        cluster_mask = (self._clusters['cluster_id'] == cluster_id)
        single_cluster = self._clusters[cluster_mask].copy()
        single_cluster['incl_frac'] = 0.0

        lat_slice, lon_slice = self._get_lat_lon_slices(cluster_id)
        excl_data = self._get_excl_data(lat_slice, lon_slice)
        techmap = self._get_tm_data(lat_slice, lon_slice)

        for i in single_cluster.index.values:
            techmap_locs = np.where(techmap == single_cluster.loc[i, 'gid'])[0]
            gid_excl_data = excl_data[techmap_locs]
            single_cluster.loc[i, 'incl_frac'] = (np.sum(gid_excl_data)
                                                  / len(gid_excl_data))

        return single_cluster


if __name__ == '__main__':
    fn_rpm_clusters = '/scratch/gbuster/rev/rpm/rpm_clusters.csv'
    fpath_excl = '/scratch/gbuster/rev/test_sc_agg/_windready_conus.tif'
    fpath_techmap = '/scratch/gbuster/rev/test_sc_agg/tech_map_conus.h5'
    dset_techmap = 'wtk'
    rpm_clusters = pd.read_csv(fn_rpm_clusters)

    rpme = RPMExclusions(rpm_clusters, fpath_excl, fpath_techmap, dset_techmap)
    c1 = rpme.calc_exclusions('Alameda-0')
