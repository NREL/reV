# -*- coding: utf-8 -*-
"""
Classes to collect reV outputs from multiple annual files.
"""
import glob
import logging
import os
import time
from warnings import warn

import numpy as np
import pandas as pd
from gaps.pipeline import parse_previous_status
from rex import Resource
from rex.utilities.utilities import (
    get_class_properties,
    get_lat_lon_cols,
    parse_year,
)

from reV.generation.base import LCOE_REQUIRED_OUTPUTS
from reV.config.output_request import SAMOutputRequest
from reV.handlers.outputs import Outputs
from reV.utilities import ModuleName, log_versions
from reV.utilities.exceptions import ConfigError, HandlerRuntimeError

logger = logging.getLogger(__name__)


class MultiYearGroup:
    """
    Handle group parameters
    """

    def __init__(self, name, out_dir, source_files=None,
                 source_dir=None, source_prefix=None,
                 source_pattern=None,
                 dsets=('cf_mean',), pass_through_dsets=None):
        """
        Parameters
        ----------
        name : str
            Group name. Can be ``"none"`` for no collection groups.
        out_dir : str
            Output directory - used for Pipeline handling.
        source_files : str | list, optional
            Explicit list of source files. Use either this input *OR*
            `source_dir` + `source_prefix`. If this input is
            ``"PIPELINE"``, the `source_files` input is determined from
            the status file of the previous pipeline step.
            If ``None``, use `source_dir` and `source_prefix`.
            By default, ``None``.
        source_dir : str, optional
            Directory to extract source files from (must be paired with
            `source_prefix`). By default, ``None``.
        source_prefix : str, optional
            File prefix to search for in source directory (must be
            paired with `source_dir`). By default, ``None``.
        source_pattern : str, optional
            Optional unix-style ``/filepath/pattern*.h5`` to specify the
            source files. This takes priority over `source_dir` and
            `source_prefix` but is not used if `source_files` are
            specified explicitly. By default, ``None``.
        dsets : str | list | tuple, optional
            List of datasets to collect. This can be set to
            ``"PIPELINE"`` if running from the command line as part of a
            reV pipeline. In this case, all the datasets from the
            previous pipeline step will be collected.
            By default, ``('cf_mean',)``.
        pass_through_dsets : list | tuple, optional
            Optional list of datasets that are identical in the
            multi-year files (e.g. input datasets that don't vary from
            year to year) that should be copied to the output multi-year
            file once without a year suffix or means/stdev calculation.
            By default, ``None``.
        """
        self._name = name
        self._dirout = out_dir
        self._source_files = source_files
        self._source_dir = source_dir
        self._source_prefix = source_prefix
        self._source_pattern = source_pattern
        self._pass_through_dsets = None
        self._dsets = None

        self._parse_pass_through_dsets(dsets, pass_through_dsets or [])
        self._parse_dsets(dsets)

    def _parse_pass_through_dsets(self, dsets, pass_through_dsets):
        """Parse a multi-year pass-through dataset collection request.

        Parameters
        ----------
        dsets : str | list
            One or more datasets to collect, or "PIPELINE"
        pass_through_dsets : list
            List of pass through datasets.
        """
        with Resource(self.source_files[0]) as res:
            all_dsets = res.datasets

        if isinstance(dsets, str) and dsets == 'PIPELINE':
            dsets = all_dsets

        if "lcoe_fcr" in dsets:
            for dset in LCOE_REQUIRED_OUTPUTS:
                if dset not in pass_through_dsets and dset in all_dsets:
                    pass_through_dsets.append(dset)

        if "dc_ac_ratio" in dsets:
            if "dc_ac_ratio" not in pass_through_dsets:
                pass_through_dsets.append("dc_ac_ratio")

        self._pass_through_dsets = SAMOutputRequest(pass_through_dsets)

    def _parse_dsets(self, dsets):
        """Parse a multi-year dataset collection request. Can handle PIPELINE
        argument which will find all datasets from one of the files being
        collected ignoring meta, time index, and pass_through_dsets

        Parameters
        ----------
        dsets : str | list
            One or more datasets to collect, or "PIPELINE"
        """
        if isinstance(dsets, str) and dsets == 'PIPELINE':
            files = parse_previous_status(self._dirout, ModuleName.MULTI_YEAR)
            with Resource(files[0]) as res:
                dsets = [d for d in res
                         if not d.startswith('time_index')
                         and d != 'meta'
                         and d not in self.pass_through_dsets]

        self._dsets = SAMOutputRequest(dsets)

    @property
    def name(self):
        """
        Returns
        -------
        name : str
            Group name
        """
        name = self._name if self._name.lower() != "none" else None
        return name

    @property
    def source_files(self):
        """
        Returns
        -------
        source_files : list
            list of source files to collect from
        """
        if self._source_files is not None:
            if isinstance(self._source_files, (list, tuple)):
                source_files = self._source_files
            elif self._source_files == "PIPELINE":
                source_files = parse_previous_status(self._dirout,
                                                     ModuleName.MULTI_YEAR)
            else:
                e = "source_files must be a list, tuple, or 'PIPELINE'"
                logger.error(e)
                raise ConfigError(e)

        elif self._source_pattern:
            source_files = glob.glob(self._source_pattern)
            if not all(fp.endswith('.h5') for fp in source_files):
                msg = ('Source pattern resulted in non-h5 files that cannot '
                       'be collected: {}, pattern: {}'
                       .format(source_files, self._source_pattern))
                logger.error(msg)
                raise RuntimeError(msg)

        elif self._source_dir and self._source_prefix:
            source_files = []
            for file in os.listdir(self._source_dir):
                if (file.startswith(self._source_prefix)
                        and file.endswith('.h5') and '_node' not in file):
                    source_files.append(os.path.join(self._source_dir,
                                                     file))
        else:
            e = ("source_files or both source_dir and "
                 "source_prefix must be provided")
            logger.error(e)
            raise ConfigError(e)

        if not any(source_files):
            e = ('Could not find any source files for '
                 'multi-year collection group: "{}" in "{}"'
                 .format(self.name, self._source_dir))
            logger.error(e)
            raise FileNotFoundError(e)

        return source_files

    @property
    def dsets(self):
        """
        Returns
        -------
        _dsets :list | tuple
            Datasets to collect
        """
        return self._dsets

    @property
    def pass_through_dsets(self):
        """Optional list of datasets that are identical in the multi-year
        files (e.g. input datasets that don't vary from year to year) that
        should be copied to the output multi-year file once without a
        year suffix or means/stdev calculation

        Returns
        -------
        list | tuple | None
        """
        return self._pass_through_dsets

    def _dict_rep(self):
        """Get a dictionary representation of this multi year collection group

        Returns
        -------
        dict
        """
        props = get_class_properties(self.__class__)
        out = {k: getattr(self, k) for k in props}
        out['group'] = self.name
        return out

    @classmethod
    def _factory(cls, out_dir, groups_dict):
        """
        Generate dictionary of MultiYearGroup objects for all groups in groups

        Parameters
        ----------
        out_dir : str
            Output directory, used for Pipeline handling
        groups_dict : dict
            Dictionary of group parameters, parsed from multi-year config file

        Returns
        -------
        groups : dict
            Dictionary of MultiYearGroup objects for each group in groups
        """
        groups = {}
        for name, kwargs in groups_dict.items():
            groups[name] = cls(name, out_dir, **kwargs)

        return groups


class MultiYear(Outputs):
    """
    Class to handle multiple years of data and:
    - collect datasets from multiple years
    - compute multi-year means
    - compute multi-year standard deviations
    - compute multi-year coefficient of variations

    """

    def __init__(self, h5_file, group=None, unscale=True, mode='r',
                 str_decode=True):
        """
        Parameters
        ----------
        h5_file : str
            Path to .h5 resource file
        group : str
            Group to collect datasets into
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        mode : str
            Mode to instantiate h5py.File instance
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        """
        log_versions(logger)
        super().__init__(h5_file, group=group, unscale=unscale, mode=mode,
                         str_decode=str_decode)

    @staticmethod
    def _create_dset_name(source_h5, dset):
        """
        Create output dataset name by parsing year from source_h5 and
        appending to source dataset name.

        Parameters
        ----------
        source_h5 : str
            Path to source .h5 file to copy data from
        dset : str
            Dataset to copy

        Returns
        -------
        dset_out : str
            Ouput dataset name
        """
        f_name = os.path.basename(source_h5)
        year = parse_year(f_name)
        dset_out = "{}-{}".format(dset, year)
        return dset_out

    def _copy_time_index(self, source_h5):
        """
        Copy time_index from source_h5 to time_index-{year} in multiyear .h5

        Parameters
        ----------
        source_h5 : str
            Path to source .h5 file to copy data from
        """
        dset_out = self._create_dset_name(source_h5, 'time_index')
        if dset_out not in self.datasets:
            logger.debug("- Collecting time_index from {}"
                         .format(os.path.basename(source_h5)))
            with Outputs(source_h5, mode='r') as f_in:
                time_index = f_in.h5['time_index'][...]

            self._create_dset(dset_out, time_index.shape, time_index.dtype,
                              data=time_index)

    def _copy_dset(self, source_h5, dset, meta=None, pass_through=False):
        """
        Copy dset_in from source_h5 to multiyear .h5

        Parameters
        ----------
        source_h5 : str
            Path to source .h5 file to copy data from
        dset : str
            Dataset to copy
        meta : pandas.DataFrame
            If provided confirm that source meta matches given meta
        pass_through : bool
            Flag to just pass through dataset without name modifications
            (no differences between years, no means or stdevs)
        """
        if pass_through:
            dset_out = dset
        else:
            dset_out = self._create_dset_name(source_h5, dset)

        if dset_out not in self.datasets:
            logger.debug("- Collecting {} from {}"
                         .format(dset, os.path.basename(source_h5)))
            with Outputs(source_h5, unscale=False, mode='r') as f_in:
                if meta is not None:
                    cols = get_lat_lon_cols(meta)
                    source_meta = f_in.meta

                    if len(meta) != len(source_meta):
                        msg = ('Meta data has different lengths between '
                               'collection files! Found {} and {}'
                               .format(len(meta), len(source_meta)))
                        logger.error(msg)
                        raise HandlerRuntimeError(msg)

                    if not np.allclose(meta[cols], source_meta[cols]):
                        msg = ('Coordinates do not match between '
                               'collection files!')
                        logger.warning(msg)
                        warn(msg)

                _, ds_dtype, ds_chunks = f_in.get_dset_properties(dset)
                ds_attrs = f_in.get_attrs(dset=dset)
                ds_data = f_in[dset]

            self._create_dset(dset_out, ds_data.shape, ds_dtype,
                              chunks=ds_chunks, attrs=ds_attrs, data=ds_data)

    @staticmethod
    def parse_source_files_pattern(source_files):
        """Parse a source_files pattern that can be either an explicit list of
        source files or a unix-style /filepath/pattern*.h5 and either way
        return a list of explicit filepaths.

        Parameters
        ----------
        source_files : list | str
            List of .h5 files to collect datasets from. This can also be a
            unix-style /filepath/pattern*.h5 to find .h5 files to collect,
            however all resulting files must be .h5 otherwise an exception will
            be raised. NOTE: .h5 file names must indicate the year the data
            pertains to

        Returns
        -------
        source_files : list
            List of .h5 filepaths.
        """

        if isinstance(source_files, str) and '*' in source_files:
            source_files = glob.glob(source_files)
        elif isinstance(source_files, str):
            source_files = [source_files]
        elif not isinstance(source_files, (list, tuple)):
            msg = ('Cannot recognize source_files type: {} {}'
                   .format(source_files, type(source_files)))
            logger.error(msg)
            raise TypeError(msg)

        if not all(fp.endswith('.h5') for fp in source_files):
            msg = ('Non-h5 files cannot be collected: {}'.format(source_files))
            logger.error(msg)
            raise RuntimeError(msg)

        return source_files

    def collect(self, source_files, dset, profiles=False, pass_through=False):
        """
        Collect dataset dset from given list of h5 files

        Parameters
        ----------
        source_files : list | str
            List of .h5 files to collect datasets from. This can also be a
            unix-style /filepath/pattern*.h5 to find .h5 files to collect,
            however all resulting files must be .h5 otherwise an exception will
            be raised. NOTE: .h5 file names must indicate the year the data
            pertains to
        dset : str
            Dataset to collect
        profiles : bool
            Boolean flag to indicate if profiles are being collected
            If True also collect time_index
        pass_through : bool
            Flag to just pass through dataset without name modifications
            (no differences between years, no means or stdevs)
        """
        source_files = self.parse_source_files_pattern(source_files)
        with Outputs(source_files[0], mode='r') as f_in:
            meta = f_in.h5['meta'][...]

        if 'meta' not in self.datasets:
            logger.debug("Copying meta")
            self._create_dset('meta', meta.shape, meta.dtype,
                              data=meta)

        meta = pd.DataFrame(meta)
        for year_h5 in source_files:
            if profiles:
                self._copy_time_index(year_h5)

            self._copy_dset(year_h5, dset, meta=meta,
                            pass_through=pass_through)

    def _get_source_dsets(self, dset_out):
        """
        Extract all available annual datasets associated with dset

        Parameters
        ----------
        dset_out : str
            Output dataset to find source datasets for

        Returns
        -------
        source_dsets : list
            List of annual datasets
        """
        dset = os.path.basename(dset_out).split("-")[0]
        logger.debug('-- source_dset root = {}'.format(dset))
        my_dset = ["{}-{}".format(dset, val) for val in ['means', 'stdev']]
        source_dsets = [ds for ds in self.datasets if dset in ds
                        and ds not in my_dset]
        if dset_out in source_dsets:
            source_dsets.remove(dset_out)

        return source_dsets

    def _update_dset(self, dset_out, dset_data):
        """
        Update dataset, create if needed

        Parameters
        ----------
        dset_out : str
            Dataset name
        dset_data : ndarray
            Dataset data to write to disc
        """
        if dset_out in self.datasets:
            logger.debug("- Updating {}".format(dset_out))
            self[dset_out] = dset_data
        else:
            logger.debug("- Creating {}".format(dset_out))
            source_dset = self._get_source_dsets(dset_out)[0]
            _, ds_dtype, ds_chunks = self.get_dset_properties(source_dset)
            ds_attrs = self.get_attrs(dset=source_dset)
            self._add_dset(dset_out, dset_data, ds_dtype,
                           chunks=ds_chunks, attrs=ds_attrs)

    def _compute_means(self, dset_out):
        """
        Compute multi-year means for given dataset

        Parameters
        ----------
        dset_out : str
            Multi-year means dataset name

        Returns
        -------
        my_means : ndarray
            Array of multi-year means
        """
        source_dsets = self._get_source_dsets(dset_out)
        logger.debug('\t- Computing {} from {}'.format(dset_out, source_dsets))

        my_means = np.zeros(len(self), dtype='float32')
        for ds in source_dsets:
            if self.h5[ds].shape == my_means.shape:
                my_means += self[ds]
            else:
                raise HandlerRuntimeError("{} shape {} should be {}"
                                          .format(ds, self.h5[ds].shape,
                                                  my_means.shape))
        my_means /= len(source_dsets)
        self._update_dset(dset_out, my_means)

        return my_means

    def means(self, dset):
        """
        Extract or compute multi-year means for given source dset

        Parameters
        ----------
        dset : str
            Dataset of interest

        Returns
        -------
        my_means : ndarray
            Array of multi-year means for dataset of interest
        """
        my_dset = "{}-means".format(dset)
        if my_dset in self.datasets:
            my_means = self[my_dset]
        else:
            my_means = self._compute_means(my_dset)

        return my_means

    def _compute_stdev(self, dset_out, means=None):
        """
        Compute multi-year standard deviation for given dataset

        Parameters
        ----------
        dset_out : str
            Multi-year stdev dataset name
        means : ndarray
            Array of pre-computed means

        Returns
        -------
        my_stdev : ndarray
            Array of multi-year standard deviations
        """
        if means is None:
            means = self._compute_means("{}-means".format(dset_out))

        source_dsets = self._get_source_dsets(dset_out)

        my_stdev = np.zeros(means.shape, dtype='float32')
        for ds in source_dsets:
            if self.h5[ds].shape == my_stdev.shape:
                my_stdev += (self[ds] - means)**2
            else:
                raise HandlerRuntimeError("{} shape {} should be {}"
                                          .format(ds, self.h5[ds].shape,
                                                  my_stdev.shape))

        my_stdev = np.sqrt(my_stdev / len(source_dsets))
        self._update_dset(dset_out, my_stdev)

        return my_stdev

    def stdev(self, dset):
        """
        Extract or compute multi-year standard deviation for given source dset

        Parameters
        ----------
        dset : str
            Dataset of interest

        Returns
        -------
        my_stdev : ndarray
            Array of multi-year standard deviation for dataset of interest
        """
        my_dset = "{}-stdev".format(dset)
        if my_dset in self.datasets:
            my_stdev = self[my_dset]
        else:
            my_means = self.means(dset)
            my_stdev = self._compute_stdev(my_dset, means=my_means)

        return my_stdev

    def CV(self, dset):
        """
        Extract or compute multi-year coefficient of variation for given
        source dset

        Parameters
        ----------
        dset : str
            Dataset of interest

        Returns
        -------
        my_cv : ndarray
            Array of multi-year coefficient of variation for
            dataset of interest
        """
        my_cv = self.stdev(dset) / self.means(dset)
        return my_cv

    @classmethod
    def is_profile(cls, source_files, dset):
        """
        Check dataset in source files to see if it is a profile.

        Parameters
        ----------
        source_files : list | str
            List of .h5 files to collect datasets from. This can also be a
            unix-style /filepath/pattern*.h5 to find .h5 files to collect,
            however all resulting files must be .h5 otherwise an exception will
            be raised. NOTE: .h5 file names must indicate the year the data
            pertains to
        dset : str
            Dataset to collect

        Returns
        -------
        is_profile : bool
            True if profile, False if not.
        """
        source_files = cls.parse_source_files_pattern(source_files)
        with Outputs(source_files[0]) as f:
            if dset not in f.datasets:
                raise KeyError('Dataset "{}" not found in source file: "{}"'
                               .format(dset, source_files[0]))

            shape, _, _ = f.get_dset_properties(dset)

        return len(shape) == 2

    @classmethod
    def pass_through(cls, my_file, source_files, dset, group=None):
        """
        Pass through a dataset that is identical in all source files to a
        dataset of the same name in the output multi-year file.

        Parameters
        ----------
        my_file : str
            Path to multi-year .h5 file
        source_files : list | str
            List of .h5 files to collect datasets from. This can also be a
            unix-style /filepath/pattern*.h5 to find .h5 files to collect,
            however all resulting files must be .h5 otherwise an exception will
            be raised. NOTE: .h5 file names must indicate the year the data
            pertains to
        dset : str
            Dataset to pass through (will also be the name of the output
            dataset in my_file)
        group : str
            Group to collect datasets into
        """
        source_files = cls.parse_source_files_pattern(source_files)
        logger.info('Passing through {} into {}.'
                    .format(dset, my_file))
        with cls(my_file, mode='a', group=group) as my:
            my.collect(source_files, dset, pass_through=True)

    @classmethod
    def collect_means(cls, my_file, source_files, dset, group=None):
        """
        Collect and compute multi-year means for given dataset

        Parameters
        ----------
        my_file : str
            Path to multi-year .h5 file
        source_files : list | str
            List of .h5 files to collect datasets from. This can also be a
            unix-style /filepath/pattern*.h5 to find .h5 files to collect,
            however all resulting files must be .h5 otherwise an exception will
            be raised. NOTE: .h5 file names must indicate the year the data
            pertains to
        dset : str
            Dataset to collect
        group : str
            Group to collect datasets into
        """
        logger.info('Collecting {} into {} '
                    'and computing multi-year means and standard deviations.'
                    .format(dset, my_file))
        source_files = cls.parse_source_files_pattern(source_files)
        with cls(my_file, mode='a', group=group) as my:
            my.collect(source_files, dset)
            means = my._compute_means("{}-means".format(dset))
            my._compute_stdev("{}-stdev".format(dset), means=means)

    @classmethod
    def collect_profiles(cls, my_file, source_files, dset, group=None):
        """
        Collect multi-year profiles associated with given dataset

        Parameters
        ----------
        my_file : str
            Path to multi-year .h5 file
        source_files : list | str
            List of .h5 files to collect datasets from. This can also be a
            unix-style /filepath/pattern*.h5 to find .h5 files to collect,
            however all resulting files must be .h5 otherwise an exception will
            be raised. NOTE: .h5 file names must indicate the year the data
            pertains to
        dset : str
            Profiles dataset to collect
        group : str
            Group to collect datasets into
        """
        logger.info('Collecting {} into {}'.format(dset, my_file))
        source_files = cls.parse_source_files_pattern(source_files)
        with cls(my_file, mode='a', group=group) as my:
            my.collect(source_files, dset, profiles=True)


def my_collect_groups(out_fpath, groups, clobber=True):
    """Collect all groups into a single multi-year HDF5 file.

    ``reV`` multi-year combines ``reV`` generation data from multiple
    years (typically stored in separate files) into a single multi-year
    file. Each dataset in the multi-year file is labeled with the
    corresponding years, and multi-year averages of the yearly datasets
    are also computed.

    Parameters
    ----------
    out_fpath : str
        Path to multi-year HDF5 file to use for multi-year
        collection.
    groups : dict
        Dictionary of collection groups and their parameters. This
        should be a dictionary mapping group names (keys) to a set
        of key word arguments (values) that can be used to initialize
        :class:`~reV.handlers.multi_year.MultiYearGroup` (excluding the
        required ``name`` and ``out_dir`` inputs, which are populated
        automatically). For example::

            groups = {
                "none": {
                    "dsets": [
                        "cf_profile",
                        "cf_mean",
                        "ghi_mean",
                        "lcoe_fcr",
                    ],
                    "source_dir": "./",
                    "source_prefix": "",
                    "pass_through_dsets": [
                        "capital_cost",
                        "fixed_operating_cost",
                        "system_capacity",
                        "fixed_charge_rate",
                        "variable_operating_cost",
                    ]
                },
                "solar_group": {
                    "source_files": "PIPELINE",
                    "dsets": [
                        "cf_profile_ac",
                        "cf_mean_ac",
                        "ac",
                        "dc",
                        "clipped_power"
                    ],
                    "pass_through_dsets": [
                        "system_capacity_ac",
                        "dc_ac_ratio"
                    ]
                },
                ...
            }

        The group names will be used as the HDF5 file group name under
        which the collected data will be stored. You can have exactly
        one group with the name ``"none"`` for a "no group" collection
        (this is typically what you want and all you need to specify).
    clobber : bool, optional
        Flag to purge the multi-year output file prior to running the
        multi-year collection step if the file already exists on disk.
        This ensures the data is always freshly collected from the
        single-year files. If ``False``, then datasets in the existing
        file will **not** be overwritten with (potentially new/updated)
        data from the single-year files. By default, ``True``.
    """
    if not out_fpath.endswith(".h5"):
        out_fpath = '{}.h5'.format(out_fpath)

    if clobber and os.path.exists(out_fpath):
        msg = ('Found existing multi-year file: "{}". Removing...'
               .format(str(out_fpath)))
        logger.warning(msg)
        warn(msg)
        os.remove(out_fpath)

    out_dir = os.path.dirname(out_fpath)
    groups = MultiYearGroup._factory(out_dir, groups)
    group_params = {name: group._dict_rep()
                    for name, group in groups.items()}

    logger.info('Multi-year collection is being run with output path: {}'
                .format(out_fpath))
    ts = time.time()
    for group_name, group in group_params.items():
        logger.info('- Collecting datasets "{}" from "{}" into "{}/"'
                    .format(group['dsets'], group['source_files'],
                            group_name))
        t0 = time.time()
        for dset in group['dsets']:
            if MultiYear.is_profile(group['source_files'], dset):
                MultiYear.collect_profiles(out_fpath, group['source_files'],
                                           dset, group=group['group'])
            else:
                MultiYear.collect_means(out_fpath, group['source_files'],
                                        dset, group=group['group'])

        pass_through_dsets = group.get('pass_through_dsets') or []
        for dset in pass_through_dsets:
            MultiYear.pass_through(out_fpath, group['source_files'],
                                   dset, group=group['group'])

        runtime = (time.time() - t0) / 60
        logger.info('- {} collection completed in: {:.2f} min.'
                    .format(group_name, runtime))

    runtime = (time.time() - ts) / 60
    logger.info('Multi-year collection completed in : {:.2f} min.'
                .format(runtime))

    return out_fpath
