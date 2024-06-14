# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
reV-NRWAL analysis module.

This module runs reV data through the NRWAL compute library. This code was
first developed to use a custom offshore wind LCOE equation library but has
since been refactored to analyze any equation library in NRWAL.

Everything in this module operates on the spatiotemporal resolution of the reV
generation output file. This is usually the wind or solar resource resolution
but could be the supply curve resolution after representative profiles is run.
"""
import logging
from warnings import warn

import numpy as np
import pandas as pd

from reV.generation.generation import Gen
from reV.handlers.outputs import Outputs
from reV.utilities import SiteDataField, ResourceMetaField, log_versions
from reV.utilities.exceptions import (
    DataShapeError,
    OffshoreWindInputError,
    OffshoreWindInputWarning,
)

logger = logging.getLogger(__name__)


class RevNrwal:
    """RevNrwal"""

    DEFAULT_META_COLS = (SiteDataField.CONFIG, )
    """Columns from the `site_data` table to join to the output meta data"""

    def __init__(self, gen_fpath, site_data, sam_files, nrwal_configs,
                 output_request, save_raw=True,
                 meta_gid_col=str(ResourceMetaField.GID),  # str() to fix docs
                 site_meta_cols=None):
        """Framework to handle reV-NRWAL analysis.

        ``reV`` NRWAL analysis runs ``reV`` data through the NRWAL
        compute library. Everything in this module operates on the
        spatiotemporal resolution of the ``reV`` generation output file
        (usually the wind or solar resource resolution but could also be
        the supply curve resolution after representative profiles is
        run).

        Parameters
        ----------
        gen_fpath : str
            Full filepath to HDF5 file with ``reV`` generation or
            rep_profiles output. Anything in the `output_request` input
            is added to and/or manipulated within this file.

            .. Note:: If executing ``reV`` from the command line, this
              input can also be ``"PIPELINE"`` to parse the output of
              one of the previous step and use it as input to this call.
              However, note that duplicate executions of ``reV``
              commands prior to this one within the pipeline may
              invalidate this parsing, meaning the `gen_fpath` input
              will have to be specified manually.

        site_data : str | pd.DataFrame
            Site-specific input data for NRWAL calculation.If this input
            is a string, it should be a path that points to a CSV file.
            Otherwise, this input should be a DataFrame with
            pre-extracted site data. Rows in this table should match
            the `meta_gid_col` in the `gen_fpath` meta data input
            sites via a ``gid`` column. A ``config`` column must also be
            provided that corresponds to the `nrwal_configs` input. Only
            sites with a gid in this file's ``gid`` column will be run
            through NRWAL.
        sam_files : dict | str
            A dictionary mapping SAM input configuration ID(s) to SAM
            configuration(s). Keys are the SAM config ID(s) which
            correspond to the keys in the `nrwal_configs` input. Values
            for each key are either a path to a corresponding SAM
            config file or a full dictionary of SAM config inputs. For
            example::

                sam_files = {
                    "default": "/path/to/default/sam.json",
                    "onshore": "/path/to/onshore/sam_config.yaml",
                    "offshore": {
                        "sam_key_1": "sam_value_1",
                        "sam_key_2": "sam_value_2",
                        ...
                    },
                    ...
                }

            This input can also be a string pointing to a single SAM
            config file. In this case, the ``config`` column of the
            CSV points input should be set to ``None`` or left out
            completely. See the documentation for the ``reV`` SAM class
            (e.g. :class:`reV.SAM.generation.WindPower`,
            :class:`reV.SAM.generation.PvWattsv8`,
            :class:`reV.SAM.generation.Geothermal`, etc.) for
            documentation on the allowed and/or required SAM config file
            inputs.
        nrwal_configs : dict
            A dictionary mapping SAM input configuration ID(s) to NRWAL
            configuration(s). Keys are the SAM config ID(s) which
            correspond to the keys in the `sam_files` input. Values
            for each key are either a path to a corresponding NRWAL YAML
            or JSON config file or a full dictionary of NRWAL config
            inputs. For example::

                nrwal_configs = {
                    "default": "/path/to/default/nrwal.json",
                    "onshore": "/path/to/onshore/nrwal_config.yaml",
                    "offshore": {
                        "nrwal_key_1": "nrwal_value_1",
                        "nrwal_key_2": "nrwal_value_2",
                        ...
                    },
                    ...
                }

        output_request : list | tuple
            List of output dataset names to be written to the
            `gen_fpath` file. Any key from the NRWAL configs or any of
            the inputs (site_data or sam_files) is available to be
            exported as an output dataset. If you want to manipulate a
            dset like ``cf_mean`` from `gen_fpath` and include it in the
            `output_request`, you should set ``save_raw=True`` and then
            use ``cf_mean_raw`` in the NRWAL equations as the input.
            This allows you to define an equation in the NRWAL configs
            for a manipulated ``cf_mean`` output that can be included in
            the `output_request` list.
        save_raw : bool, optional
            Flag to save an initial ("raw") copy of input datasets from
            `gen_fpath` that are also part of the `output_request`. For
            example, if you request ``cf_mean`` in output_request but
            also manipulate the ``cf_mean`` dataset in the NRWAL
            equations, the original ``cf_mean`` will be archived under
            the ``cf_mean_raw`` dataset in `gen_fpath`.
            By default, ``True``.
        meta_gid_col : str, optional
            Column label in the source meta data from `gen_fpath` that
            contains the unique gid identifier. This will be joined to
            the site_data ``gid`` column. By default, ``"gid"``.
        site_meta_cols : list | tuple, optional
            Column labels from `site_data` to be added to the meta data
            table in `gen_fpath`. If ``None``, only the columns in
            :attr:`DEFAULT_META_COLS` will be added. Any columns
            requested via this input will be considered *in addition to*
            the :attr:`DEFAULT_META_COLS`. By default, ``None``.
        """

        log_versions(logger)

        # delayed NRWAL import to cause less errors with old reV installs
        # if not running nrwal.
        from NRWAL import NrwalConfig

        self._meta_gid_col = meta_gid_col
        self._gen_fpath = gen_fpath
        self._site_data = site_data
        self._output_request = output_request
        self._meta_out = None
        self._time_index = None
        self._save_raw = save_raw
        self._nrwal_inputs = self._out = None

        self._nrwal_configs = {
            k: NrwalConfig(v) for k, v in nrwal_configs.items()
        }

        self._site_meta_cols = site_meta_cols
        if self._site_meta_cols is None:
            self._site_meta_cols = list(self.DEFAULT_META_COLS)
        else:
            self._site_meta_cols = list(self._site_meta_cols)
            self._site_meta_cols += list(self.DEFAULT_META_COLS)
            self._site_meta_cols = list(set(self._site_meta_cols))

        self._site_data = self._parse_site_data()
        self._meta_source = self._parse_gen_data()
        self._analysis_gids, self._site_data = self._parse_analysis_gids()

        pc = Gen.get_pc(
            self._site_data[[SiteDataField.GID, SiteDataField.CONFIG]],
            points_range=None, sam_configs=sam_files, tech='windpower')
        self._project_points = pc.project_points

        self._sam_sys_inputs = self._parse_sam_sys_inputs()
        meta_gids = self.meta_source[self._meta_gid_col].values
        logger.info(
            'Finished initializing NRWAL analysis module for "{}" '
            "{} through {} with {} total generation points and "
            "{} NRWAL analysis points.".format(
                self._meta_gid_col,
                meta_gids.min(),
                meta_gids.max(),
                len(self.meta_source),
                len(self.analysis_gids),
            )
        )

    def _parse_site_data(self, required_columns=(SiteDataField.GID,
                                                 SiteDataField.CONFIG)):
        """Parse the site-specific spatial input data file

        Parameters
        ----------
        required_columns : tuple | list
            List of column names that must be in the site_data in
            order to run the reV NRWAL module.

        Returns
        -------
        site_data : pd.DataFrame
            Dataframe of extracted site_data. Each row is an analysis point and
            columns are spatial data inputs.
        """

        if isinstance(self._site_data, str):
            self._site_data = pd.read_csv(self._site_data)

        if "dist_l_to_ts" in self._site_data:
            if self._site_data["dist_l_to_ts"].sum() > 0:
                w = (
                    'Possible incorrect Offshore data input! "dist_l_to_ts" '
                    "(distance land to transmission) input is non-zero. "
                    "Most reV runs set this to zero and input the cost "
                    "of transmission from landfall tie-in to "
                    "transmission feature in the supply curve module."
                )
                logger.warning(w)
                warn(w, OffshoreWindInputWarning)

        for c in required_columns:
            if c not in self._site_data:
                msg = 'Did not find required "{}" column in site_data!'.format(
                    c
                )
                logger.error(msg)
                raise KeyError(msg)

        self._site_data = self._site_data.sort_values(SiteDataField.GID)

        return self._site_data

    def _parse_gen_data(self):
        """Parse generation data and get meta data

        Returns
        -------
        meta : pd.DataFrame
            Full meta data from gen_fpath.
        """

        with Outputs(self._gen_fpath, mode="r") as out:
            meta = out.meta

        msg = (
            'Could not find "{}" column in source generation h5 file '
            "meta data! Available cols: {}".format(
                self._meta_gid_col, meta.columns.values.tolist()
            )
        )
        assert self._meta_gid_col in meta, msg

        # currently an assumption of sorted gids in the reV gen output
        msg = "Source capacity factor meta data is not ordered!"
        meta_gids = list(meta[self._meta_gid_col])
        assert meta_gids == sorted(meta_gids), msg

        return meta

    def _parse_analysis_gids(self):
        """Check the intersection of the generation gids and the site_data
        input gids.

        Returns
        -------
        analysis_gids : np.ndarray
            Array indicating which sites in the source meta data to process
            with NRWAL. This is the intersection of the gids in the generation
            meta data and the gids in the site_data input.
        site_data : pd.DataFrame
            The site_data table reduced to only those gids that are in the
            analysis_gids
        """

        meta_gids = self.meta_source[self._meta_gid_col].values

        missing = ~np.isin(meta_gids, self._site_data[SiteDataField.GID])
        if any(missing):
            msg = (
                "{} sites from the generation meta data input were "
                'missing from the "site_data" input and will not be '
                "run through NRWAL: {}".format(
                    missing.sum(), meta_gids[missing]
                )
            )
            logger.info(msg)

        missing = ~np.isin(self._site_data[SiteDataField.GID], meta_gids)
        if any(missing):
            missing = self._site_data[SiteDataField.GID].values[missing]
            msg = ('{} sites from the "site_data" input were missing from the '
                   'generation meta data and will not be run through NRWAL: {}'
                   .format(len(missing), missing))
            logger.info(msg)

        analysis_gids = (set(meta_gids)
                         & set(self._site_data[SiteDataField.GID]))
        analysis_gids = np.array(sorted(list(analysis_gids)))

        # reduce the site data table to only those sites being analyzed
        mask = np.isin(self._site_data[SiteDataField.GID], meta_gids)
        self._site_data = self._site_data[mask]

        return analysis_gids, self._site_data

    def _parse_sam_sys_inputs(self):
        """Get the SAM system inputs dict from project points.

        Returns
        -------
        system_inputs : pd.DataFrame
            DataFrame of SAM config inputs (columns) for every active nrwal
            analysis gid (row). Index is resource gids and there is also a
            column "gid" with the copied gids.
        """

        system_inputs = {}

        for gid in self.analysis_gids:
            system_inputs[gid] = self._project_points[gid][1]

        system_inputs = pd.DataFrame(system_inputs).T
        system_inputs = system_inputs.sort_index()
        system_inputs[SiteDataField.GID] = system_inputs.index.values
        system_inputs.index.name = SiteDataField.GID
        mask = system_inputs[SiteDataField.GID].isin(self.analysis_gids)
        system_inputs = system_inputs[mask]

        return system_inputs

    def _init_outputs(self):
        """Initialize a dictionary of outputs with dataset names as keys and
        numpy arrays as values. All datasets are initialized as 1D arrays and
        must be overwritten if found to be 2D. Only active analysis sites will
        have data in the output, sites that were not found in the site_data
        "gid" column will not have data in these output arrays

        Returns
        -------
        out : dict
            Dictionary of output data
        """
        out = {}

        for key in self._output_request:
            out[key] = np.full(
                len(self.analysis_gids), np.nan, dtype=np.float32
            )

            if key in self.gen_dsets and not self._save_raw:
                msg = (
                    'Output request "{0}" was also found in '
                    "the source gen file but save_raw=False! If "
                    "you are manipulating this "
                    "dset, make sure you set save_raw=False "
                    'and reference "{0}_raw" as the '
                    'input in the NRWAL equations and then define "{0}" '
                    "as the final manipulated dataset.".format(key)
                )
                logger.warning(msg)
                warn(msg)
            elif key in self.gen_dsets:
                msg = (
                    'Output request "{0}" was also found in '
                    "the source gen file. If you are manipulating this "
                    'dset, make sure you reference "{0}_raw" as the '
                    'input in the NRWAL equations and then define "{0}" '
                    "as the final manipulated dataset.".format(key)
                )
                logger.info(msg)

            if key in self._nrwal_inputs:
                out[key] = self._nrwal_inputs[key]

        return out

    def _preflight_checks(self):
        """Run some preflight checks on the offshore inputs"""
        sam_files = {
            k: v
            for k, v in self._project_points.sam_inputs.items()
            if k in self._nrwal_configs
        }

        for cid, sys_in in sam_files.items():
            loss1 = sys_in.get("wind_farm_losses_percent", 0)
            loss2 = sys_in.get("turb_generic_loss", 0)
            if loss1 != 0 or loss2 != 0:
                msg = (
                    'Wind farm loss for config "{}" is not 0. When using '
                    "NRWAL for offshore analysis, consider using gross "
                    "capacity factors from reV generation and applying "
                    "spatially dependent losses from the NRWAL equations"
                    .format(cid)
                )
                logger.info(msg)

        available_ids = list(self._nrwal_configs.keys())
        requested_ids = list(self._site_data[SiteDataField.CONFIG].values)
        missing = set(requested_ids) - set(available_ids)
        if any(missing):
            msg = (
                "The following config ids were requested in the offshore "
                "data input but were not available in the NRWAL config "
                "input dict: {}".format(missing)
            )
            logger.error(msg)
            raise OffshoreWindInputError(msg)

        check_gid_order = (self._site_data[SiteDataField.GID].values
                           == self._sam_sys_inputs[SiteDataField.GID].values)
        msg = 'NRWAL site_data and system input dataframe had bad order'
        assert (check_gid_order).all(), msg

        missing = [c for c in self._site_meta_cols if c not in self._site_data]
        if any(missing):
            msg = (
                "Could not find requested NRWAL site data pass through "
                "columns in offshore input data: {}".format(missing)
            )
            logger.error(msg)
            raise OffshoreWindInputError(msg)

    def _get_input_data(self):
        """Get all the input data from the site_data, SAM system configs, and
        generation h5 file, formatted together in one dictionary for NRWAL.

        Returns
        -------
        nrwal_inputs : dict
            Dictionary mapping required NRWAL input variable names (keys) to 1
            or 2D arrays of inputs for all the analysis_gids
        """

        logger.info("Setting up input data for NRWAL...")

        # preconditions for this to work properly
        assert len(self._site_data) == len(self.analysis_gids)
        assert len(self._sam_sys_inputs) == len(self.analysis_gids)

        all_required = []
        for config_id, nrwal_config in self._nrwal_configs.items():
            all_required += list(nrwal_config.required_inputs)
            all_required = list(set(all_required))

            missing_vars = [
                var
                for var in nrwal_config.required_inputs
                if var not in self._site_data
                and var not in self.meta_source
                and var not in self._sam_sys_inputs
                and var not in self.gen_dsets
            ]

            if any(missing_vars):
                msg = (
                    "Could not find required input variables {} "
                    'for NRWAL config "{}" in either the offshore '
                    "data or the SAM system data!".format(
                        missing_vars, config_id
                    )
                )
                logger.error(msg)
                raise OffshoreWindInputError(msg)

        meta_data_vars = [
            var for var in all_required if var in self.meta_source
        ]
        logger.info(
            "Pulling the following inputs from the gen meta data: {}".format(
                meta_data_vars
            )
        )
        nrwal_inputs = {
            var: self.meta_source[var].values[self.analysis_mask]
            for var in meta_data_vars
        }

        site_data_vars = [var for var in all_required
                          if var in self._site_data
                          and var not in nrwal_inputs]
        site_data_vars.append(SiteDataField.CONFIG)
        logger.info('Pulling the following inputs from the site_data input: {}'
                    .format(site_data_vars))
        for var in site_data_vars:
            nrwal_inputs[var] = self._site_data[var].values

        sam_sys_vars = [
            var
            for var in all_required
            if var in self._sam_sys_inputs and var not in nrwal_inputs
        ]
        logger.info(
            "Pulling the following inputs from the SAM system "
            "configs: {}".format(sam_sys_vars)
        )
        for var in sam_sys_vars:
            nrwal_inputs[var] = self._sam_sys_inputs[var].values

        gen_vars = [
            var
            for var in all_required
            if var in self.gen_dsets and var not in nrwal_inputs
        ]
        logger.info(
            "Pulling the following inputs from the generation "
            "h5 file: {}".format(gen_vars)
        )
        with Outputs(self._gen_fpath, mode="r") as f:
            source_gids = self.meta_source[self._meta_gid_col]
            gen_gids = np.where(source_gids.isin(self.analysis_gids))[0]
            for var in gen_vars:
                shape = f.shapes[var]
                if len(shape) == 1:
                    nrwal_inputs[var] = f[var, gen_gids]
                elif len(shape) == 2:
                    nrwal_inputs[var] = f[var, :, gen_gids]
                else:
                    msg = (
                        'Data shape for "{}" must be 1 or 2D but '
                        "received: {}".format(var, shape)
                    )
                    logger.error(msg)
                    raise DataShapeError(msg)

        logger.info("Finished setting up input data for NRWAL!")

        return nrwal_inputs

    @property
    def time_index(self):
        """Get the source time index."""
        if self._time_index is None:
            with Outputs(self._gen_fpath, mode="r") as out:
                self._time_index = out.time_index

        return self._time_index

    @property
    def gen_dsets(self):
        """Get the available datasets from the gen source file"""
        with Outputs(self._gen_fpath, mode="r") as out:
            dsets = out.dsets

        return dsets

    @property
    def meta_source(self):
        """Get the full meta data (onshore + offshore)"""
        return self._meta_source

    @property
    def meta_out(self):
        """Get the combined onshore and offshore meta data."""
        if self._meta_out is None:
            self._meta_out = self._meta_source.copy()
            for col in self._site_meta_cols:
                data = self._nrwal_inputs[col]
                self._meta_out.loc[self.analysis_mask, col] = data

        return self._meta_out

    @property
    def analysis_mask(self):
        """Get a boolean array to mask the source generation meta data where
        True is sites that are to be analyzed by NRWAL.

        Returns
        -------
        np.ndarray
        """
        mask = np.isin(
            self.meta_source[self._meta_gid_col], self.analysis_gids
        )
        return mask

    @property
    def analysis_gids(self):
        """Get an array of gids from the source generation meta data that are
        to-be analyzed by nrwal.

        Returns
        -------
        np.ndarray
        """
        return self._analysis_gids

    @property
    def outputs(self):
        """Get a dict of NRWAL outputs. Only active analysis sites will have
        data in the output, sites that were not found in the site_data "gid"
        column will not have data in these output arrays"""
        return self._out

    def _save_nrwal_out(self, name, nrwal_out, output_mask):
        """Save a dataset from the nrwal_out dictionary to the self._out
        attribute

        Parameters
        ----------
        name : str
            Dataset name of the nrwal output to be saved.
        nrwal_out : dict
            Output dictionary from a successfully evaluated NrwalConfig object
            containing the dataset with the input name
        output_mask : np.ndarray
            Boolean array showing which gids in self.analysis_gids should be
            assigned data from this NRWAL output. If not all true, there are
            probably multiple NrwalConfig objects that map to different sets of
            gids.
        """
        value = nrwal_out[name]
        value = self._value_to_array(value, name)

        if len(value.shape) == 1:
            self._out[name][output_mask] = value[output_mask]

        elif len(value.shape) == 2:
            if len(self._out[name].shape) == 1:
                if not all(np.isnan(self._out[name])):
                    msg = (
                        'Output dataset "{}" was initialized as 1D but was '
                        "later found to be 2D but was not all NaN!".format(
                            name
                        )
                    )
                    logger.error(msg)
                    raise DataShapeError(msg)

                # re-initialize the dataset as 2D now that we
                # know what the output looks like
                out_shape = (len(self.time_index), len(self.analysis_gids))
                self._out[name] = np.full(out_shape, np.nan, dtype=np.float32)

            self._out[name][:, output_mask] = value[:, output_mask]

        else:
            msg = (
                'Could not make sense of NRWAL output "{}" '
                "with shape {}".format(name, value.shape)
            )
            logger.error(msg)
            raise DataShapeError(msg)

    def _save_nrwal_misc(self, name, nrwal_config, output_mask):
        """Save miscellaneous output requests from a NRWAL config object (not
        NRWAL output dictionary) to the self._out attribute.

        Parameters
        ----------
        name : str
            Dataset name of the nrwal output to be saved.
        nrwal_config : NrwalConfig
            NrwalConfig object containing NRWAL Equation objects or something
            else that is to be exported to the outputs
        output_mask : np.ndarray
            Boolean array showing which gids in self.analysis_gids should be
            assigned data from this NRWAL output. If not all true, there are
            probably multiple NrwalConfig objects that map to different sets of
            gids.
        """

        from NRWAL import Equation

        value = nrwal_config[name]

        if isinstance(value, Equation):
            msg = (
                'Cannot retrieve Equation "{}" from NRWAL. '
                "Must be a number!".format(name)
            )
            assert not any(value.variables), msg
            value = value.eval()

        value = self._value_to_array(value, name)
        self._out[name][output_mask] = value[output_mask]

    def _value_to_array(self, value, name):
        """Turn the input into numpy array if it isn't already."""
        if np.issubdtype(type(value), np.number):
            value *= np.ones(len(self.analysis_gids), dtype=np.float32)

        if not isinstance(value, np.ndarray):
            msg = (
                'NRWAL key "{}" returned bad type of "{}", needs to be '
                "numeric or an output array.".format(name, type(value))
            )
            logger.error(msg)
            raise TypeError(msg)
        return value

    def run_nrwal(self):
        """Run analysis via the NRWAL analysis library"""

        self._preflight_checks()
        self.save_raw_dsets()
        self._nrwal_inputs = self._get_input_data()
        self._out = self._init_outputs()

        for i, (cid, nrwal_config) in enumerate(self._nrwal_configs.items()):
            output_mask = self._site_data[SiteDataField.CONFIG].values == cid
            logger.info('Running NRWAL config {} of {}: "{}" and applying '
                        'to {} out of {} total sites'
                        .format(i + 1, len(self._nrwal_configs), cid,
                                output_mask.sum(), len(output_mask)))

            nrwal_out = nrwal_config.eval(inputs=self._nrwal_inputs)

            # pylint: disable=C0201
            for name in self._out.keys():
                if name in nrwal_out:
                    self._save_nrwal_out(name, nrwal_out, output_mask)

                elif name in nrwal_config.keys():
                    self._save_nrwal_misc(name, nrwal_config, output_mask)

                elif name not in self._nrwal_inputs:
                    msg = (
                        'Could not find "{}" in the output dict of NRWAL '
                        "config {}".format(name, cid)
                    )
                    logger.warning(msg)
                    warn(msg)

    def check_outputs(self):
        """Check the nrwal outputs for nan values and raise errors if found."""
        for name, arr in self._out.items():
            if np.isnan(arr).all():
                msg = (
                    'Output array "{}" is all NaN! Probably was not '
                    "found in the available NRWAL keys.".format(name)
                )
                logger.warning(msg)
                warn(msg)
            elif np.isnan(arr).any():
                mask = np.isnan(arr)
                nan_meta = self.meta_source[self.analysis_mask][mask]
                nan_gids = nan_meta[self._meta_gid_col].values
                msg = (
                    "NaN values ({} out of {}) persist in NRWAL "
                    'output "{}"!'.format(np.isnan(arr).sum(), len(arr), name)
                )
                logger.warning(msg)
                logger.warning(
                    "This is the NRWAL meta that is causing NaN "
                    "outputs: {}".format(nan_meta)
                )
                logger.warning(
                    "These are the resource gids causing NaN "
                    "outputs: {}".format(nan_gids)
                )
                warn(msg)

    def save_raw_dsets(self):
        """If requested by save_raw=True, archive raw datasets that exist in
        the gen_fpath file and are also requested in the output_request"""
        if self._save_raw:
            with Outputs(self._gen_fpath, "a") as f:
                for dset in self._output_request:
                    dset_raw = "{}_raw".format(dset)
                    if dset in f and dset_raw not in f:
                        logger.info(
                            'Saving raw data from "{}" to "{}"'.format(
                                dset, dset_raw
                            )
                        )
                        f._add_dset(
                            dset_raw,
                            f[dset],
                            f.dtypes[dset],
                            attrs=f.attrs[dset],
                        )

    def write_to_gen_fpath(self):
        """Save NRWAL outputs to input generation fpath file.

        Returns
        -------
        str
            Path to output file.
        """

        logger.info("Writing NRWAL outputs to: {}".format(self._gen_fpath))
        write_all = self.analysis_mask.all()

        with Outputs(self._gen_fpath, "a") as f:
            meta_attrs = f.attrs["meta"]
            del f._h5["meta"]
            f._set_meta("meta", self.meta_out, attrs=meta_attrs)

            for dset, arr in self._out.items():
                if len(arr.shape) == 1:
                    data = np.full(
                        len(self.meta_source), np.nan, dtype=np.float32
                    )
                else:
                    full_shape = (len(self.time_index), len(self.meta_source))
                    data = np.full(full_shape, np.nan, dtype=np.float32)

                dset_attrs = {"scale_factor": 1}
                dset_dtype = np.float32
                if dset in f.dsets:
                    logger.info(
                        'Found "{}" in file, loading data and '
                        "overwriting data for {} out of {} sites.".format(
                            dset,
                            self.analysis_mask.sum(),
                            len(self.analysis_mask),
                        )
                    )
                    dset_attrs = f.attrs[dset]
                    dset_dtype = f.dtypes[dset]
                    if not write_all:
                        data = f[dset]

                if len(arr.shape) == 1:
                    data[self.analysis_mask] = arr
                else:
                    data[:, self.analysis_mask] = arr

                logger.info(
                    'Writing final "{}" to: {}'.format(dset, self._gen_fpath)
                )
                f._add_dset(dset, data, dset_dtype, attrs=dset_attrs)

        logger.info(
            "Finished writing NRWAL outputs to: {}".format(self._gen_fpath)
        )
        return self._gen_fpath

    def write_meta_to_csv(self, out_fpath=None):
        """Combine NRWAL outputs with meta and write to output csv.

        Parameters
        ----------
        out_fpath : str, optional
            Full path to output NRWAL CSV file. The file path does not
            need to include file ending - it will be added automatically
            if missing. If ``None``, the generation HDF5 filepath will
            be converted to a CSV out path by replacing the ".h5" file
            ending with ".csv". By default, ``None``.

        Returns
        -------
        str
            Path to output file.
        """
        if out_fpath is None:
            out_fpath = self._gen_fpath.replace(".h5", ".csv")
        elif not out_fpath.endswith(".csv"):
            out_fpath = "{}.csv".format(out_fpath)

        logger.info("Writing NRWAL outputs to: {}".format(out_fpath))
        meta_out = self.meta_out[self.analysis_mask].copy()

        for dset, arr in self._out.items():
            if len(arr.shape) != 1 or arr.shape[0] != meta_out.shape[0]:
                msg = (
                    "Skipping output {!r}: shape {} cannot be combined "
                    "with meta of shape {}!".format(
                        dset, arr.shape, meta_out.shape
                    )
                )
                logger.warning(msg)
                warn(msg)
                continue
            meta_out[dset] = arr

        meta_out.to_csv(out_fpath, index=False)
        logger.info("Finished writing NRWAL outputs to: {}".format(out_fpath))
        return out_fpath

    def run(self, csv_output=False, out_fpath=None):
        """Run NRWAL analysis.

        Parameters
        ----------
        csv_output : bool, optional
            Option to write H5 file meta + all requested outputs to
            CSV file instead of storing in the HDF5 file directly. This
            can be useful if the same HDF5 file is used for multiple
            sets of NRWAL runs. Note that all requested output datasets
            must be 1-dimensional in order to fir within the CSV output.

            .. Important:: This option is not compatible with
              ``save_raw=True``. If you set ``csv_output=True``, then
              the `save_raw` option is forced to be ``False``.
              Therefore, make sure that you do not have any references
              to "input_dataset_name_raw" in your NRWAL config. If you
              need to manipulate an input dataset, save it to a
              different output name in the NRWAL config or manually add
              an "input_dataset_name_raw" dataset to your generation
              HDF5 file before running NRWAL.

            By default, ``False``.
        out_fpath : str, optional
            This option has no effect if ``csv_output=False``.
            Otherwise, this should be the full path to output NRWAL CSV
            file. The file path does not need to include file ending -
            it will be added automatically if missing. If ``None``, the
            generation HDF5 filepath will be converted to a CSV out path
            by replacing the ".h5" file ending with ".csv".
            By default, ``None``.

        Returns
        -------
        str
            Path to output file.
        """
        if csv_output and self._save_raw:
            msg = (
                "`save_raw` option not allowed with `csv_output`. Setting"
                "`save_raw=False`"
            )
            logger.warning(msg)
            warn(msg)
            self._save_raw = False

        if any(self.analysis_gids):
            self.run_nrwal()
            self.check_outputs()
            if csv_output:
                out_fp = self.write_meta_to_csv(out_fpath)
            else:
                out_fp = self.write_to_gen_fpath()

        logger.info("NRWAL module complete!")

        return out_fp
