# -*- coding: utf-8 -*-
"""
reV Project Points Configuration
"""
import logging
import pandas as pd
import numpy as np
import os
from warnings import warn
from math import ceil

from reV.utilities.exceptions import ConfigError, ConfigWarning
from reV.config.sam_config import SAMConfig
from reV.config.curtailment import Curtailment

from rex.resource import Resource, MultiFileResource
from rex.utilities import check_res_file

logger = logging.getLogger(__name__)


class PointsControl:
    """Class to manage and split ProjectPoints."""
    def __init__(self, project_points, sites_per_split=100):
        """
        Parameters
        ----------
        project_points : reV.config.ProjectPoints
            ProjectPoints instance to be split between execution workers.
        sites_per_split : int
            Sites per project points split instance returned in the __next__
            iterator function.
        """

        self._project_points = project_points
        self._sites_per_split = sites_per_split
        self._split_range = []
        self._i = 0
        self._iter_list = []

    def __iter__(self):
        """Initialize the iterator by pre-splitting into a list attribute."""
        last_site = 0
        ilim = len(self.project_points)

        logger.debug('PointsControl iterator initializing with sites '
                     '{} through {}'.format(self.project_points.sites[0],
                                            self.project_points.sites[-1]))

        # pre-initialize all iter objects
        while True:
            i0 = last_site
            i1 = np.min([i0 + self.sites_per_split, ilim])
            if i0 == i1:
                break

            last_site = i1

            new = PointsControl.split(i0, i1, self.project_points,
                                      sites_per_split=self.sites_per_split)
            new._split_range = [i0, i1]
            self._iter_list.append(new)

        logger.debug('PointsControl stopped iteration at attempted '
                     'index of {}. Length of iterator is: {}'
                     .format(i1, len(self)))
        return self

    def __next__(self):
        """Iterate through and return next site resource data.

        Returns
        -------
        next_pc : config.PointsControl
            Split instance of this class with a subset of project points based
            on the number of sites per split.
        """
        if self._i < self.N:
            # Get next PointsControl from the iter list
            next_pc = self._iter_list[self._i]
        else:
            # No more points controllers left in initialized list
            raise StopIteration

        logger.debug('PointsControl passing site project points '
                     'with indices {} to {} on iteration #{} '
                     .format(next_pc.split_range[0],
                             next_pc.split_range[1], self._i))
        self._i += 1
        return next_pc

    def __repr__(self):
        msg = ("{} for sites {} through {}"
               .format(self.__class__.__name__, self.sites[0], self.sites[-1]))
        return msg

    def __len__(self):
        """Len is the number of possible iterations aka splits."""
        return ceil(len(self.project_points) / self.sites_per_split)

    @property
    def N(self):
        """
        Length of current iterator list

        Returns
        -------
        N : int
            Number of iterators in list
        """
        return len(self._iter_list)

    @property
    def sites_per_split(self):
        """Get the iterator increment: number of sites per split.

        Returns
        -------
        _sites_per_split : int
            Sites per split iter object.
        """
        return self._sites_per_split

    @property
    def project_points(self):
        """Get the project points property.

        Returns
        -------
        _project_points : reV.config.project_points.ProjectPoints
            ProjectPoints instance corresponding to this PointsControl
            instance.
        """
        return self._project_points

    @property
    def sites(self):
        """Get the project points sites for this instance.

        Returns
        -------
        sites : list
            List of sites belonging to the _project_points attribute.
        """
        return self._project_points.sites

    @property
    def split_range(self):
        """Get the current split range property.

        Returns
        -------
        _split_range : list
            Two-entry list that indicates the starting and finishing
            (inclusive, exclusive, respectively) indices of a split instance
            of the PointsControl object. This is set in the iterator dunder
            methods of PointsControl.
        """
        return self._split_range

    @classmethod
    def split(cls, i0, i1, project_points, sites_per_split=100):
        """Split this execution by splitting the project points attribute.

        Parameters
        ----------
        i0/i1 : int
            Beginning/end (inclusive/exclusive, respectively) index split
            parameters for ProjectPoints.split.
        project_points : reV.config.ProjectPoints
            Project points instance that will be split.
        sites_per_split : int
            Sites per project points split instance returned in the __next__
            iterator function.

        Returns
        -------
        sub : PointsControl
            New instance of PointsControl with a subset of the original
            project points.
        """
        i0 = int(i0)
        i1 = int(i1)
        new_points = ProjectPoints.split(i0, i1, project_points)
        sub = cls(new_points, sites_per_split=sites_per_split)
        return sub


class ProjectPoints:
    """Class to manage site and SAM input configuration requests.

    Notes
    -----
    config_id@site0, SAM_config_dict@site0 = ProjectPoints[0]
    site_list_or_slice = ProjectPoints.sites
    site_list_or_slice = ProjectPoints.get_sites_from_config(config_id)
    ProjectPoints_sub = ProjectPoints.split(0, 10, project_points)
    h_list = ProjectPoints.h
    """

    def __init__(self, points, sam_config, tech, res_file=None,
                 curtailment=None):
        """
        Parameters
        ----------
        points : slice | str | pd.DataFrame | dict
            Slice specifying project points, string pointing to a project
            points csv, or a dataframe containing the effective csv contents.
        sam_config : dict | str | list | SAMConfig
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s), top level value is the SAM path. Can also be a single
            config file str. If it's a list, it is mapped to the sorted list
            of unique configs requested by points csv. Can also be a
            pre loaded SAMConfig object.
        tech : str
            SAM technology to analyze (pvwattsv7, windpower, tcsmoltensalt,
            solarwaterheat, troughphysicalheat, lineardirectsteam)
            The string should be lower-cased with spaces and _ removed.
        res_file : str | NoneType
            Optional resource file to find maximum length of project points if
            points slice stop is None.
        curtailment : NoneType | dict | str | config.curtailment.Curtailment
            Inputs for curtailment parameters. If not None, curtailment inputs
            are expected. Can be:
                - Explicit namespace of curtailment variables (dict)
                - Pointer to curtailment config json file with path (str)
                - Instance of curtailment config object
                  (config.curtailment.Curtailment)
        """

        # set protected attributes
        self._df = self._parse_points(points, res_file=res_file)
        self._sam_config_obj = self._parse_sam_config(sam_config)
        self._check_points_config_mapping()
        self._tech = tech
        self._h = None
        self._curtailment = self._parse_curtailment(curtailment)

    def __getitem__(self, site):
        """Get the SAM config ID and dictionary for the requested site.

        Parameters
        ----------
        site : int | str
            Site number of interest.

        Returns
        -------
        config_id : str
            Configuration ID (variable name) specified in the sam_generation
            config section.
        config : dict
            Actual SAM input values in a single level dictionary with variable
            names (keys) and values.
        """

        site_bool = (self.df['gid'] == site)
        try:
            config_id = self.df.loc[site_bool, 'config'].values[0]
        except KeyError:
            raise KeyError('Site {} not found in this instance of '
                           'ProjectPoints. Available sites include: {}'
                           .format(site, self.sites))
        return config_id, self.sam_configs[config_id]

    def __repr__(self):
        msg = ("{} for sites {} through {}"
               .format(self.__class__.__name__, self.sites[0], self.sites[-1]))
        return msg

    def __len__(self):
        """Length of this object is the number of sites."""
        return len(self.sites)

    @staticmethod
    def _parse_points(points, res_file=None):
        """Generate the project points df from inputs

        Parameters
        ----------
        points : str | pd.DataFrame | slice | list | dict
            Slice specifying project points, string pointing to a project
            points csv, or a dataframe containing the effective csv contents.
        res_file : str | NoneType
            Optional resource file to find maximum length of project points if
            points slice stop is None.

        Returns
        -------
        df : pd.DataFrame
            DataFrame mapping sites (gids) to SAM technology (config)
        """
        if isinstance(points, str):
            df = ProjectPoints._parse_csv(points)
        elif isinstance(points, dict):
            df = pd.DataFrame(points)
        elif isinstance(points, (slice, list, tuple)):
            df = ProjectPoints._parse_sites(points, res_file=res_file)
        elif isinstance(points, pd.DataFrame):
            df = points
        else:
            raise ValueError('Cannot parse Project points data from {}'
                             .format(type(points)))

        if ('gid' not in df.columns or 'config' not in df.columns):
            raise KeyError('Project points data must contain "gid" and '
                           '"config" column headers.')

        return df

    @staticmethod
    def _parse_csv(fname):
        """Import project points from .csv

        Parameters
        ----------
        fname : str
            Project points .csv file (with path). Must have 'gid' and 'config'
            column names.

        Returns
        -------
        df : pd.DataFrame
            DataFrame mapping sites (gids) to SAM technology (config)
        """
        if fname.endswith('.csv'):
            df = pd.read_csv(fname)
        else:
            raise ValueError('Config project points file must be '
                             '.csv, but received: {}'
                             .format(fname))
        return df

    @staticmethod
    def _parse_sites(points, res_file=None):
        """Parse project points from list or slice

        Parameters
        ----------
        points : str | pd.DataFrame | slice | list
            Slice specifying project points, string pointing to a project
            points csv, or a dataframe containing the effective csv contents.
        res_file : str | NoneType
            Optional resource file to find maximum length of project points if
            points slice stop is None.

        Returns
        -------
        df : pd.DataFrame
            DataFrame mapping sites (gids) to SAM technology (config)
        """
        df = pd.DataFrame(columns=['gid', 'config'])
        if isinstance(points, (list, tuple)):
            # explicit site list, set directly
            df['gid'] = points
        elif isinstance(points, slice):
            stop = points.stop
            if stop is None:
                if res_file is None:
                    raise ValueError('Must supply a resource file if '
                                     'points is a slice of type '
                                     ' slice(*, None, *)')

                multi_h5_res, _ = check_res_file(res_file)
                if multi_h5_res:
                    stop = MultiFileResource(res_file).shape[1]
                else:
                    stop = Resource(res_file).shape[1]

            df['gid'] = list(range(*points.indices(stop)))
        else:
            raise TypeError('Project Points sites needs to be set as a list, '
                            'tuple, or slice, but was set as: {}'
                            .format(type(points)))

        df['config'] = None
        return df

    @property
    def df(self):
        """Get the project points dataframe property.

        Returns
        -------
        _df : pd.DataFrame
            Table of sites and corresponding SAM configuration IDs.
            Has columns 'gid' and 'config'.
        """
        return self._df

    @staticmethod
    def _parse_sam_config(sam_config):
        """
        Create SAM files dictionary.

        Parameters
        ----------
        sam_config : dict | str | list | SAMConfig
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s), top level value is the SAM path. Can also be a single
            config file str. If it's a list, it is mapped to the sorted list
            of unique configs requested by points csv. Can also be a
            pre loaded SAMConfig object.

        Returns
        -------
        _sam_config_obj : reV.config.sam_config.SAMConfig
            SAM configuration object.
        """

        if isinstance(sam_config, SAMConfig):
            return sam_config

        else:
            if isinstance(sam_config, dict):
                config_dict = sam_config
            elif isinstance(sam_config, str):
                config_dict = {sam_config: sam_config}
            else:
                raise ValueError('Cannot parse SAM configs from {}'
                                 .format(type(sam_config)))

            for key, value in config_dict.items():
                if not os.path.isfile(value):
                    raise ConfigError('Invalid SAM config {}: {} does not '
                                      'exist'.format(key, value))

            return SAMConfig(config_dict)

    @property
    def sam_files(self):
        """Get the SAM files dictionary property.

        Returns
        -------
        _sam_files: dict
            Multi-level dictionary containing multiple SAM input config files.
            The top level key is the SAM config ID, top level value is the SAM
            config file path
        """
        return dict(self._sam_config_obj.items())

    @property
    def sam_config_obj(self):
        """Get the SAM config object.

        Returns
        -------
        _sam_config_obj : reV.config.sam_config.SAMConfig
            SAM configuration object.
        """
        return self._sam_config_obj

    @property
    def sam_configs(self):
        """Get the SAM configs dictionary property.

        Returns
        -------
        _sam_configs : dict
            Multi-level dictionary containing multiple SAM input
            configurations. The top level key is the SAM config ID, top level
            value is the SAM config. Each SAM config is a dictionary with keys
            equal to input names, values equal to the actual inputs.
        """

        return self.sam_config_obj.inputs

    def _check_points_config_mapping(self):
        """
        Check to ensure the project points (df) and SAM configs
        (sam_config_obj) are compatible. Update as necessary or break
        """
        # Extract unique config refences from project_points DataFrame
        df_configs = self.df['config'].unique()
        sam_configs = self.sam_files

        # Checks to make sure that the same number of SAM config .json files
        # as references in project_points DataFrame
        if len(df_configs) > len(sam_configs):
            raise ConfigError('Points references {} configs while only '
                              '{} SAM configs were provided!'
                              .format(len(df_configs), len(sam_configs)))

        # If project_points DataFrame was created from a list,
        # config will be None and needs to be added to _df from sam_configs
        if len(df_configs) == 1:
            if df_configs[0] is None:
                self._df['config'] = list(sam_configs.values())[0]

                df_configs = self.df['config'].unique()

        # Check to see if config references in project_points DataFrame
        # are valid file paths, if compare with SAM configs
        # and update as needed
        configs = {}
        for config in df_configs:
            if os.path.isfile(config):
                configs[config] = config
            elif config in sam_configs:
                configs[config] = sam_configs[config]
            else:
                raise ConfigError('{} does not map to a valid configuration '
                                  'file'.format(config))

        if configs != sam_configs:
            self._sam_config_obj = SAMConfig(configs)

    @property
    def sites(self):
        """Get the sites belonging to this instance of ProjectPoints.

        Returns
        -------
        _sites : list | slice
            List of sites belonging to this instance of ProjectPoints. The type
            is list if possible. Will be a slice only if slice stop is None.
        """
        return list(self.df['gid'].values)

    @property
    def sites_as_slice(self):
        """Get the sites in slice format.

        Returns
        -------
        sites_as_slice : list | slice
            Sites slice belonging to this instance of ProjectPoints.
            The type is slice if possible. Will be a list only if sites are
            non-sequential.
        """
        # try_slice is what the sites list would be if it is sequential
        if len(self.sites) > 1:
            try_step = self.sites[1] - self.sites[0]
        else:
            try_step = 1
        try_slice = slice(self.sites[0], self.sites[-1] + 1, try_step)
        try_list = list(range(*try_slice.indices(try_slice.stop)))

        if self.sites == try_list:
            # try_slice is equivelant to the site list
            sites_as_slice = try_slice
        else:
            # cannot be converted to a sequential slice, return list
            sites_as_slice = self.sites

        return sites_as_slice

    @property
    def tech(self):
        """Get the tech property from the config.

        Returns
        -------
        _tech : str
            SAM technology to analyze (pvwattsv7, windpower, tcsmoltensalt,
            solarwaterheat, troughphysicalheat, lineardirectsteam)
            The string should be lower-cased with spaces and _ removed.
        """
        tech = self._tech
        if 'wind' in tech.lower():
            tech = 'windpower'

        return self._tech

    @property
    def h(self):
        """Get the hub heights corresponding to the site list.

        Returns
        -------
        _h : list | NoneType
            Hub heights corresponding to each site, taken from the sam config
            for each site. This is None if the technology is not wind.
        """
        h_var = 'wind_turbine_hub_ht'
        if self._h is None:
            if 'wind' in self.tech:
                # wind technology, get a list of h values
                self._h = [self[site][1][h_var] for site in self.sites]

        return self._h

    @staticmethod
    def _parse_curtailment(curtailment_input):
        """Parse curtailment config object.

        Parameters
        ----------
        curtailment_input : None | dict | str | config.curtailment.Curtailment
            Inputs for curtailment parameters. If not None, curtailment inputs
            are expected. Can be:
                - Explicit namespace of curtailment variables (dict)
                - Pointer to curtailment config json file with path (str)
                - Instance of curtailment config object
                  (config.curtailment.Curtailment)

        Returns
        -------
        curtailments : NoneType | reV.config.curtailment.Curtailment
            None if no curtailment, reV curtailment config object if
            curtailment is being assessed.
        """
        if isinstance(curtailment_input, (str, dict)):
            # pointer to config file or explicit input namespace,
            # instantiate curtailment config object
            curtailment = Curtailment(curtailment_input)

        elif isinstance(curtailment_input, (Curtailment, type(None))):
            # pre-initialized curtailment object or no curtailment (None)
            curtailment = curtailment_input

        else:
            curtailment = None
            warn('Curtailment inputs not recognized. Received curtailment '
                 'input of type: "{}". Expected None, dict, str, or '
                 'Curtailment object. Defaulting to no curtailment.',
                 ConfigWarning)

        return curtailment

    @property
    def curtailment(self):
        """Get the curtailment config object.

        Returns
        -------
        _curtailment : NoneType | reV.config.curtailment.Curtailment
            None if no curtailment, reV curtailment config object if
            curtailment is being assessed.
        """
        return self._curtailment

    def join_df(self, df2, key='gid'):
        """Join new df2 to the _df attribute using the _df's gid as pkey.

        This can be used to add site-specific data to the project_points,
        taking advantage of the points_control iterator/split functions such
        that only the relevant site data is passed to the analysis functions.

        Parameters
        ----------
        df2 : pd.DataFrame
            Dataframe to be joined to the _df attribute. This likely contains
            site-specific inputs that are to be passed to parallel workers.
        key : str | pd.DataFrame.index
            Primary key of df2 to be joined to the _df attribute. Primary key
            of the _df attribute is fixed as the gid column.
        """
        # ensure df2 doesnt have any duplicate columns for suffix reasons.
        df2_cols = [c for c in df2.columns if c not in self._df or c == key]
        self._df = pd.merge(self._df, df2[df2_cols], how='left', left_on='gid',
                            right_on=key, copy=False, validate='1:1')

    def get_sites_from_config(self, config):
        """Get a site list that corresponds to a config key.

        Parameters
        ----------
        config : str
            SAM configuration ID associated with sites.

        Returns
        -------
        sites : list
            List of sites associated with the requested configuration ID. If
            the configuration ID is not recognized, an empty list is returned.
        """

        sites = self.df.loc[(self.df['config'] == config), 'gid'].values
        return list(sites)

    @classmethod
    def split(cls, i0, i1, project_points):
        """Return split instance of a ProjectPoints instance w/ site subset.

        Parameters
        ----------
        i0 : int
            Starting INDEX (not site number) (inclusive) of the site property
            attribute to include in the split instance. This is not necessarily
            the same as the starting site number, for instance if ProjectPoints
            is sites 20:100, i0=0 i1=10 will result in sites 20:30.
        i1 : int
            Ending INDEX (not site number) (exclusive) of the site property
            attribute to include in the split instance. This is not necessarily
            the same as the final site number, for instance if ProjectPoints is
            sites 20:100, i0=0 i1=10 will result in sites 20:30.
        project_points: ProjectPoints
            Instance of project points to split.

        Returns
        -------
        sub : ProjectPoints
            New instance of ProjectPoints with a subset of the following
            attributes: sites, project points df, and the self dictionary data
            struct.
        """
        # Extract DF subset with only index values between i0 and i1
        n = len(project_points)
        if i0 > n or i1 > n:
            raise ValueError('{} and {} must be within the range of '
                             'project_points (0 - {})'.format(i0, i1, n - 1))

        points_df = project_points.df.iloc[i0:i1]

        # make a new instance of ProjectPoints with subset DF
        sub = cls(points_df,
                  project_points.sam_config_obj,
                  project_points.tech,
                  curtailment=project_points.curtailment)

        return sub
