"""
Pipeline between reV and RPM
"""
import logging
import pandas as pd

# from reV.handlers.outputs import Outputs
# from reV.rpm.clusters import RPMClusters
from reV.utilities.exceptions import RPMValueError

logger = logging.getLogger(__name__)


class RPM:
    """
    Entry point for reV to RPM pipeline. Pipeline:
    - Creates 'regular' regional RPM 'clusters'
    - Ranks resource pixels against the cluster 'average'
    - Extracts representative (post-exclusion) profiles
    """
    def __init__(self, cf_profiles, rpm_meta):
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
        """
        self._cf_h5 = cf_profiles
        self._rpm_meta = self._parse_rpm_meta(rpm_meta)
        self._rpm_regions = self._preflight_check()

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

    def _preflight_check(self):
        """
        Ensure that the RPM meta data has everything needed for processing

        Returns
        -------
        rpm_regions : dict
            Dictionary mapping rpm regions to cf GIDs and number of
            clusters
        """
