# -*- coding: utf-8 -*-
"""
reV generation module.
"""

import copy
import json
import logging
import os
import pprint

import numpy as np
import pandas as pd
from rex.multi_file_resource import MultiFileResource
from rex.multi_res_resource import MultiResolutionResource
from rex.resource import Resource
from rex.utilities.utilities import check_res_file

from reV.generation.base import BaseGen
from reV.SAM.generation import (
    Geothermal,
    LinearDirectSteam,
    MhkWave,
    PvSamv1,
    PvWattsv5,
    PvWattsv7,
    PvWattsv8,
    SolarWaterHeat,
    TcsMoltenSalt,
    WindPower,
)
from reV.utilities import ModuleName, ResourceMetaField, SupplyCurveField
from reV.utilities.exceptions import (
    ConfigError,
    InputError,
    ProjectPointsValueError,
)

logger = logging.getLogger(__name__)


ATTR_DIR = os.path.dirname(os.path.realpath(__file__))
ATTR_DIR = os.path.join(ATTR_DIR, "output_attributes")
with open(os.path.join(ATTR_DIR, "other.json")) as f:
    OTHER_ATTRS = json.load(f)
with open(os.path.join(ATTR_DIR, "generation.json")) as f:
    GEN_ATTRS = json.load(f)
with open(os.path.join(ATTR_DIR, "linear_fresnel.json")) as f:
    LIN_ATTRS = json.load(f)
with open(os.path.join(ATTR_DIR, "solar_water_heat.json")) as f:
    SWH_ATTRS = json.load(f)
with open(os.path.join(ATTR_DIR, "trough_heat.json")) as f:
    TPPH_ATTRS = json.load(f)


class Gen(BaseGen):
    """Gen"""

    # Mapping of reV technology strings to SAM generation objects
    OPTIONS = {
        "geothermal": Geothermal,
        "lineardirectsteam": LinearDirectSteam,
        "mhkwave": MhkWave,
        "pvsamv1": PvSamv1,
        "pvwattsv5": PvWattsv5,
        "pvwattsv7": PvWattsv7,
        "pvwattsv8": PvWattsv8,
        "solarwaterheat": SolarWaterHeat,
        "tcsmoltensalt": TcsMoltenSalt,
        "windpower": WindPower,
    }

    """reV technology options."""

    # Mapping of reV generation outputs to scale factors and units.
    # Type is scalar or array and corresponds to the SAM single-site output
    OUT_ATTRS = copy.deepcopy(OTHER_ATTRS)
    OUT_ATTRS.update(GEN_ATTRS)
    OUT_ATTRS.update(LIN_ATTRS)
    OUT_ATTRS.update(SWH_ATTRS)
    OUT_ATTRS.update(TPPH_ATTRS)
    OUT_ATTRS.update(BaseGen.ECON_ATTRS)

    def __init__(
        self,
        technology,
        project_points,
        sam_files,
        resource_file,
        low_res_resource_file=None,
        output_request=("cf_mean",),
        site_data=None,
        curtailment=None,
        gid_map=None,
        drop_leap=False,
        sites_per_worker=None,
        memory_utilization_limit=0.4,
        scale_outputs=True,
        write_mapped_gids=False,
        bias_correct=None,
    ):
        """ReV generation analysis class.

        ``reV`` generation analysis runs SAM simulations by piping in
        renewable energy resource data (usually from the NSRDB or WTK),
        loading the SAM config, and then executing the PySAM compute
        module for a given technology. See the documentation for the
        ``reV`` SAM class (e.g. :class:`reV.SAM.generation.WindPower`,
        :class:`reV.SAM.generation.PvWattsv8`,
        :class:`reV.SAM.generation.Geothermal`, etc.) for info on the
        allowed and/or required SAM config file inputs. If economic
        parameters are supplied in the SAM config, then you can bundle a
        "follow-on" econ calculation by just adding the desired econ
        output keys to the `output_request`. You can request ``reV`` to
        run the analysis for one or more "sites", which correspond to
        the meta indices in the resource data (also commonly called the
        ``gid's``).

        Examples
        --------
        The following is an example of the most simple way to run reV
        generation. Note that the ``TESTDATADIR`` refers to the local cloned
        repository and will need to be replaced with a valid path if you
        installed ``reV`` via a simple pip install.

        >>> import os
        >>> from reV import Gen, TESTDATADIR
        >>>
        >>> sam_tech = 'pvwattsv8'
        >>> sites = 0
        >>> fp_sam = os.path.join(TESTDATADIR, 'SAM/naris_pv_1axis_inv13.json')
        >>> fp_res = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_2013.h5')
        >>>
        >>> gen = Gen(sam_tech, sites, fp_sam, fp_res)
        >>> gen.run()
        >>>
        >>> gen.out
        {'cf_mean': array([0.16966143], dtype=float32)}
        >>>
        >>> sites = [3, 4, 7, 9]
        >>> req = ('cf_mean', 'lcoe_fcr')
        >>> gen = Gen(sam_tech, sites, fp_sam, fp_res, output_request=req)
        >>> gen.run()
        >>>
        >>> gen.out
        {'fixed_charge_rate': array([0.096, 0.096, 0.096, 0.096],
         'base_capital_cost': array([39767200, 39767200, 39767200, 39767200],
         'base_variable_operating_cost': array([0, 0, 0, 0],
         'base_fixed_operating_cost': array([260000, 260000, 260000, 260000],
         'capital_cost': array([39767200, 39767200, 39767200, 39767200],
         'fixed_operating_cost': array([260000, 260000, 260000, 260000],
         'variable_operating_cost': array([0, 0, 0, 0],
         'capital_cost_multiplier': array([1, 1, 1, 1],
         'cf_mean': array([0.17859147, 0.17869979, 0.1834818 , 0.18646291],
         'lcoe_fcr': array([130.32126, 130.24226, 126.84782, 124.81981]}

        Parameters
        ----------
        technology : str
            String indicating which SAM technology to analyze. Must be
            one of the keys of
            :attr:`~reV.generation.generation.Gen.OPTIONS`. The string
            should be lower-cased with spaces and underscores removed.
        project_points : int | list | tuple | str | dict | pd.DataFrame | slice
            Input specifying which sites to process. A single integer
            representing the generation GID of a site may be specified
            to evaluate reV at a single location. A list or tuple of
            integers (or slice) representing the generation GIDs of
            multiple sites can be specified to evaluate reV at multiple
            specific locations. A string pointing to a project points
            CSV file may also be specified. Typically, the CSV contains
            the following columns:

                - ``gid``: Integer specifying the generation GID of each
                  site.
                - ``config``: This is an *optional* column that contains
                  a key from the `sam_files` input dictionary
                  (see below) corresponding to the SAM configuration to
                  use for each particular site. This value can also be
                  ``None``, ``"default"``, or left out completely if you
                  specify only a single SAM configuration file as the
                  `sam_files` input.
                - ``curtailment``: This is an *optional* column that
                  contains a key from the `curtailment` input dictionary
                  (see below) corresponding to the curtailment to apply
                  at that particular site. This value can also be
                  ``None``, ``"default"``, or left out completely if you
                  specify only a single curtailment configuration file
                  as the `curtailment` input.
                - ``capital_cost_multiplier``: This is an *optional*
                  multiplier input that, if included, will be used to
                  regionally scale the ``capital_cost`` input in the SAM
                  config. If you include this column in your CSV, you
                  *do not* need to specify ``capital_cost``, unless you
                  would like that value to vary regionally and
                  independently of the multiplier (i.e. the multiplier
                  will still be applied on top of the ``capital_cost``
                  input).

            The CSV file may also contain other site-specific inputs by
            including a column named after a config keyword (e.g. a
            column called ``wind_turbine_rotor_diameter`` may be
            included to specify a site-specific turbine diameter for
            each location). Columns that do not correspond to a config
            key may also be included, but they will be ignored. A
            DataFrame following the same guidelines as the CSV input
            (or a dictionary that can be used to initialize such a
            DataFrame) may be used for this input as well.

            .. Note:: By default, the generation GID of each site is
              assumed to match the resource GID to be evaluated for that
              site. However, unique generation GID's can be mapped to
              non-unique resource GID's via the `gid_map` input (see the
              documentation for `gid_map` for more details).

        sam_files : dict | str
            A dictionary mapping SAM input configuration ID(s) to SAM
            configuration(s). Keys are the SAM config ID(s) which
            correspond to the ``config`` column in the project points
            CSV. Values for each key are either a path to a
            corresponding SAM config file or a full dictionary
            of SAM config inputs. For example::

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
            info on the allowed and/or required SAM config file inputs.
        resource_file : str
            Filepath to resource data. This input can be path to a
            single resource HDF5 file or a path including a wildcard
            input like ``/h5_dir/prefix*suffix`` (i.e. if your datasets
            like wind speed, wind direction, pressure, and so on are
            spread out over multiple files). In all cases, the resource
            data must be readable by :py:class:`rex.resource.Resource`
            or :py:class:`rex.multi_file_resource.MultiFileResource`.
            (i.e. the resource data conform to the
            `rex data format <https://tinyurl.com/3fy7v5kx>`_). This
            means the data file(s) must contain a 1D ``time_index``
            dataset indicating the UTC time of observation, a 1D
            ``meta`` dataset represented by a DataFrame with
            site-specific columns, and 2D resource datasets that match
            the dimensions of (``time_index``, ``meta``). The time index
            must start at 00:00 of January 1st of the year under
            consideration, and its shape must be a multiple of 8760.

            .. Note:: If executing ``reV`` from the command line, this
              input string can contain brackets ``{}`` that will be
              filled in by the `analysis_years` input. If your datasets
              span multiple files (e.g. "wtk_wind_speed_2012.h5",
              "wtk_pressure_2012.h5", "wtk_wind_direction_2012.h5"), you
              may use a wildcard input along with brackets, like so:
              ``"wtk_*_{}.h5"``. Alternatively, this input can be a list
              of explicit files to process. In this case, the length of
              the list must match the length of the `analysis_years`
              input exactly, and the paths are assumed to align with the
              `analysis_years` (i.e. the first path corresponds to the
              first analysis year, the second path corresponds to the
              second analysis year, and so on). Wild cards are allowed,
              even if you list out the years explicitly (i.e.
              ``["wtk_*_2012.h5", "wtk_*_2013.h5", ...]``)

            .. Important:: If you are using custom resource data (i.e.
              not NSRDB/WTK/Sup3rCC, etc.), ensure the following:

                  - The data conforms to the
                    `rex data format <https://tinyurl.com/3fy7v5kx>`_.
                  - The ``meta`` DataFrame is organized such that every
                    row is a pixel and at least the columns
                    ``latitude``, ``longitude``, ``timezone``, and
                    ``elevation`` are given for each location.
                  - The time index and associated temporal data is in
                    UTC.
                  - The latitude is between -90 and 90 and longitude is
                    between -180 and 180.
                  - For solar data, ensure the DNI/DHI are not zero. You
                    can calculate one of these these inputs from the
                    other using the relationship

                    .. math:: GHI = DNI * cos(SZA) + DHI

        low_res_resource_file : str, optional
            Optional low resolution resource file that will be
            dynamically mapped+interpolated to the nominal-resolution
            `resource_file`. This needs to be of the same format as
            `resource_file` - both files need to be handled by the
            same ``rex Resource`` handler (e.g. ``WindResource``). All
            of the requirements from the `resource_file` apply to this
            input as well. If ``None``, no dynamic mapping to higher
            resolutions is performed. By default, ``None``.
        output_request : list | tuple, optional
            List of output variables requested from SAM. Can be any
            of the parameters in the "Outputs" group of the PySAM module
            (e.g. :py:class:`PySAM.Windpower.Windpower.Outputs`,
            :py:class:`PySAM.Pvwattsv8.Pvwattsv8.Outputs`,
            :py:class:`PySAM.Geothermal.Geothermal.Outputs`, etc.) being
            executed. This list can also include a select number of SAM
            config/resource parameters to include in the output:
            any key in any of the
            `output attribute JSON files <https://tinyurl.com/4bmrpe3j/>`_
            may be requested. If ``cf_mean`` is not included in this
            list, it will automatically be added. Time-series profiles
            requested via this input are output in UTC.

            .. Note:: If you are performing ``reV`` solar runs using
              ``PVWatts`` and would like ``reV`` to include AC capacity
              values in your aggregation/supply curves, then you must
              include the ``"dc_ac_ratio"`` time series as an output in
              `output_request` when running ``reV`` generation. The AC
              capacity outputs will automatically be added during the
              aggregation/supply curve step if the ``"dc_ac_ratio"``
              dataset is detected in the generation file.

            By default, ``('cf_mean',)``.
        site_data : str | pd.DataFrame, optional
            Site-specific input data for SAM calculation. If this input
            is a string, it should be a path that points to a CSV file.
            Otherwise, this input should be a DataFrame with
            pre-extracted site data. Rows in this table should match
            the input sites via a ``gid`` column. The rest of the
            columns should match configuration input keys that will take
            site-specific values. Note that some or all site-specific
            inputs can be specified via the `project_points` input
            table instead. If ``None``, no site-specific data is
            considered.

            .. Note:: This input is often used to provide site-based
               regional capital cost multipliers. ``reV`` does not
               ingest multipliers directly; instead, this file is
               expected to have a ``capital_cost`` column that gives the
               multiplier-adjusted capital cost value for each location.
               Therefore, you *must* re-create this input file every
               time you change your base capital cost assumption.

            By default, ``None``.
        curtailment : dict | str, optional
            Input for curtailment parameters, which can be one of:

                - Single string representing path to curtailment config
                  file. In this case, the curtailment config is given
                  the name "default" and applied everywhere (if the
                  project points "curtailment" column is missing or all
                  ``None``) or only where the project points
                  "curtailment" column contains a value of "default"
                - Dictionary mapping user-defined curtailment "names" to
                  either A) strings (paths) or B) explicit namespaces of
                  curtailment configurations (dicts). Mixing these two
                  _is_ allowed.

            The allowed key-value input pairs in the curtailment
            configuration are documented as properties of the
            :class:`reV.config.curtailment.Curtailment` class. If
            ``None``, no curtailment is modeled. You can select which
            curtailment gets applied to which site using the
            "curtailment" column key in the project points input.
            By default, ``None``.
        gid_map : dict | str, optional
            Mapping of unique integer generation gids (keys) to single
            integer resource gids (values). This enables unique
            generation gids in the project points to map to non-unique
            resource gids, which can be useful when evaluating multiple
            resource datasets in ``reV`` (e.g., forecasted ECMWF
            resource data to complement historical WTK meteorology).
            This input can be a pre-extracted dictionary or a path to a
            JSON or CSV file. If this input points to a CSV file, the
            file must have the columns ``gid`` (which matches the
            project points) and ``gid_map`` (gids to extract from the
            resource input). If ``None``, the GID values in the project
            points are assumed to match the resource GID values.
            By default, ``None``.
        drop_leap : bool, optional
            Drop leap day instead of final day of year when handling
            leap years. By default, ``False``.
        sites_per_worker : int, optional
            Number of sites to run in series on a worker. ``None``
            defaults to the resource file chunk size.
            By default, ``None``.
        memory_utilization_limit : float, optional
            Memory utilization limit (fractional). Must be a value
            between 0 and 1. This input sets how many site results will
            be stored in-memory at any given time before flushing to
            disk. By default, ``0.4``.
        scale_outputs : bool, optional
            Flag to scale outputs in-place immediately upon ``Gen``
            returning data. By default, ``True``.
        write_mapped_gids : bool, optional
            Option to write mapped gids to output meta instead of
            resource gids. By default, ``False``.
        bias_correct : str | pd.DataFrame, optional
            Optional DataFrame or CSV filepath to a wind or solar
            resource bias correction table. This has columns:

                - ``gid``: GID of site (can be index name of dataframe)
                - ``method``: function name from ``rex.bias_correction`` module

            The ``gid`` field should match the true resource ``gid`` regardless
            of the optional ``gid_map`` input. Only ``windspeed`` **or**
            ``GHI`` + ``DNI`` + ``DHI`` are corrected, depending on the
            technology (wind for the former, PV or CSP for the latter). See the
            functions in the ``rex.bias_correction`` module for available
            inputs for ``method``. Any additional kwargs required for the
            requested ``method`` can be input as additional columns in the
            ``bias_correct`` table e.g., for linear bias correction functions
            you can include ``scalar`` and ``adder`` inputs as columns in the
            ``bias_correct`` table on a site-by-site basis. If ``None``, no
            corrections are applied. By default, ``None``.
        """
        pc = self.get_pc(
            points=project_points,
            points_range=None,
            sam_configs=sam_files,
            tech=technology,
            sites_per_worker=sites_per_worker,
            res_file=resource_file,
            curtailment=curtailment,
        )

        super().__init__(
            pc,
            output_request,
            site_data=site_data,
            drop_leap=drop_leap,
            memory_utilization_limit=memory_utilization_limit,
            scale_outputs=scale_outputs,
        )

        if self.tech not in self.OPTIONS:
            msg = (
                'Requested technology "{}" is not available. '
                "reV generation can analyze the following "
                "SAM technologies: {}".format(
                    self.tech, list(self.OPTIONS.keys())
                )
            )
            logger.error(msg)
            raise KeyError(msg)

        self.write_mapped_gids = write_mapped_gids
        self._res_file = resource_file
        self._lr_res_file = low_res_resource_file
        self._sam_module = self.OPTIONS[self.tech]
        self._run_attrs["sam_module"] = self._sam_module.MODULE
        self._run_attrs["res_file"] = resource_file

        self._multi_h5_res, self._hsds = check_res_file(resource_file)
        self._gid_map = self._parse_gid_map(gid_map)
        self._nn_map = self._parse_nn_map()
        self._bc = self._parse_bc(bias_correct)

    @property
    def res_file(self):
        """Get the resource filename and path.

        Returns
        -------
        res_file : str
            Filepath to single resource file, multi-h5 directory,
            or /h5_dir/prefix*suffix
        """
        return self._res_file

    @property
    def lr_res_file(self):
        """Get the (optional) low-resolution resource filename and path.

        Returns
        -------
        str | None
        """
        return self._lr_res_file

    @property
    def meta(self):
        """Get resource meta for all sites in project points.

        Returns
        -------
        meta : pd.DataFrame
            Meta data df for sites in project points. Column names are meta
            data variables, rows are different sites. The row index
            does not indicate the site number if the project points are
            non-sequential or do not start from 0, so a `SiteDataField.GID`
            column is added.
        """
        if self._meta is None:
            res_cls = Resource
            kwargs = {"hsds": self._hsds}
            if self._multi_h5_res:
                res_cls = MultiFileResource
                kwargs = {}

            res_gids = self.project_points.sites
            if self._gid_map is not None:
                res_gids = [self._gid_map[i] for i in res_gids]

            with res_cls(self.res_file, **kwargs) as res:
                meta_len = res.shapes["meta"][0]

                if np.max(res_gids) > meta_len:
                    msg = (
                        "ProjectPoints has a max site gid of {} which is "
                        "out of bounds for the meta data of len {} from "
                        "resource file: {}".format(
                            np.max(res_gids), meta_len, self.res_file
                        )
                    )
                    logger.error(msg)
                    raise ProjectPointsValueError(msg)

                self._meta = res["meta", res_gids]

            self._meta.loc[:, ResourceMetaField.GID] = res_gids
            if self.write_mapped_gids:
                sites = self.project_points.sites
                self._meta.loc[:, ResourceMetaField.GID] = sites
            self._meta.index = self.project_points.sites
            self._meta.index.name = ResourceMetaField.GID
            self._meta.loc[:, "reV_tech"] = self.project_points.tech

        return self._meta

    @property
    def time_index(self):
        """Get the generation resource time index data.

        Returns
        -------
        _time_index : pandas.DatetimeIndex
            Time-series datetime index
        """
        if self._time_index is None:
            if not self._multi_h5_res:
                res_cls = Resource
                kwargs = {"hsds": self._hsds}
            else:
                res_cls = MultiFileResource
                kwargs = {}

            with res_cls(self.res_file, **kwargs) as res:
                time_index = res.time_index

            downscale = self.project_points.sam_config_obj.downscale
            step = self.project_points.sam_config_obj.time_index_step
            if downscale is not None:
                from rex.utilities.downscale import make_time_index

                year = time_index.year[0]
                ds_freq = downscale["frequency"]
                time_index = make_time_index(year, ds_freq)
                logger.info(
                    "reV solar generation running with temporal "
                    'downscaling frequency "{}" with final '
                    "time_index length {}".format(ds_freq, len(time_index))
                )
            elif step is not None:
                time_index = time_index[::step]

            time_index = self.handle_lifetime_index(time_index)
            time_index = self.handle_leap_ti(
                time_index, drop_leap=self._drop_leap
            )

            self._time_index = time_index

        return self._time_index

    def handle_lifetime_index(self, ti):
        """Adjust the time index if modeling full system lifetime.

        Parameters
        ----------
        ti : pandas.DatetimeIndex
            Time-series datetime index with leap days.

        Returns
        -------
        ti : pandas.DatetimeIndex
            Time-series datetime index.
        """
        life_var = "system_use_lifetime_output"
        lifetime_periods = []
        for sam_meta in self.sam_metas.values():
            if life_var in sam_meta and sam_meta[life_var] == 1:
                lifetime_period = sam_meta["analysis_period"]
                lifetime_periods.append(lifetime_period)
            else:
                lifetime_periods.append(1)

        if not any(ltp > 1 for ltp in lifetime_periods):
            return ti

        # Only one time index may be passed, check that lifetime periods match
        n_unique_periods = len(np.unique(lifetime_periods))
        if n_unique_periods != 1:
            msg = (
                "reV cannot handle multiple analysis_periods when "
                "modeling with `system_use_lifetime_output` set "
                "to 1. Found {} different analysis_periods in the SAM "
                "configs".format(n_unique_periods)
            )
            logger.error(msg)
            raise ConfigError(msg)

        # Collect requested variables to check for lifetime compatibility
        array_vars = [
            var for var, attrs in GEN_ATTRS.items() if attrs["type"] == "array"
        ]
        valid_vars = ["gen_profile", "cf_profile", "cf_profile_ac"]
        invalid_vars = set(array_vars) - set(valid_vars)
        invalid_requests = [
            var for var in self.output_request if var in invalid_vars
        ]

        if invalid_requests:
            # SAM does not output full lifetime for all array variables
            msg = (
                "reV can only handle the following output arrays "
                "when modeling with `system_use_lifetime_output` set "
                "to 1: {}. Try running without {}.".format(
                    ", ".join(valid_vars), ", ".join(invalid_requests)
                )
            )
            logger.error(msg)
            raise ConfigError(msg)

        sam_meta = self.sam_metas[next(iter(self.sam_metas))]
        analysis_period = sam_meta["analysis_period"]
        logger.info(
            "reV generation running with a full system "
            "life of {} years.".format(analysis_period)
        )

        old_end = ti[-1]
        new_end = old_end + pd.DateOffset(years=analysis_period - 1)
        step = old_end - ti[-2]
        time_extension = pd.date_range(old_end, new_end, freq=step)
        ti = time_extension.union(ti)

        return ti

    @classmethod
    def _run_single_worker(
        cls,
        points_control,
        tech=None,
        res_file=None,
        lr_res_file=None,
        output_request=None,
        scale_outputs=True,
        gid_map=None,
        nn_map=None,
        bias_correct=None,
    ):
        """Run a SAM generation analysis based on the points_control iterator.

        Parameters
        ----------
        points_control : reV.config.PointsControl
            A PointsControl instance dictating what sites and configs are run.
        tech : str
            SAM technology to analyze (pvwattsv7, windpower, tcsmoltensalt,
            solarwaterheat, lineardirectsteam, geothermal)
            The string should be lower-cased with spaces and _ removed.
        res_file : str
            Filepath to single resource file, multi-h5 directory,
            or /h5_dir/prefix*suffix
        lr_res_file : str | None
            Optional low resolution resource file that will be dynamically
            mapped+interpolated to the nominal-resolution res_file. This
            needs to be of the same format as resource_file, e.g. they both
            need to be handled by the same rex Resource handler such as
            WindResource
        output_request : list | tuple
            Output variables requested from SAM.
        scale_outputs : bool
            Flag to scale outputs in-place immediately upon Gen returning data.
        gid_map : None | dict
            Mapping of unique integer generation gids (keys) to single integer
            resource gids (values). This enables the user to input unique
            generation gids in the project points that map to non-unique
            resource gids. This can be None or a pre-extracted dict.
        nn_map : np.ndarray
            Optional 1D array of nearest neighbor mappings associated with the
            res_file to lr_res_file spatial mapping. For details on this
            argument, see the rex.MultiResolutionResource docstring.
        bias_correct : None | pd.DataFrame
            Optional DataFrame or CSV filepath to a wind or solar
            resource bias correction table. This has columns:

                - ``gid``: GID of site (can be index name of dataframe)
                - ``method``: function name from ``rex.bias_correction`` module

            The ``gid`` field should match the true resource ``gid`` regardless
            of the optional ``gid_map`` input. Only ``windspeed`` **or**
            ``GHI`` + ``DNI`` + ``DHI`` are corrected, depending on the
            technology (wind for the former, PV or CSP for the latter). See the
            functions in the ``rex.bias_correction`` module for available
            inputs for ``method``. Any additional kwargs required for the
            requested ``method`` can be input as additional columns in the
            ``bias_correct`` table e.g., for linear bias correction functions
            you can include ``scalar`` and ``adder`` inputs as columns in the
            ``bias_correct`` table on a site-by-site basis. If ``None``, no
            corrections are applied. By default, ``None``.

        Returns
        -------
        out : dict
            Output dictionary from the SAM reV_run function. Data is scaled
            within this function to the datatype specified in Gen.OUT_ATTRS.
        """

        # Extract the site df from the project points df.
        site_df = points_control.project_points.df
        site_df = site_df.set_index(ResourceMetaField.GID, drop=True)

        # run generation method for specified technology
        try:
            out = cls.OPTIONS[tech].reV_run(
                points_control,
                res_file,
                site_df,
                lr_res_file=lr_res_file,
                output_request=output_request,
                gid_map=gid_map,
                nn_map=nn_map,
                bias_correct=bias_correct,
            )

        except Exception as e:
            out = {}
            logger.exception("Worker failed for PC: {}".format(points_control))
            raise e

        if scale_outputs:
            # dtype convert in-place so no float data is stored unnecessarily
            for site, site_output in out.items():
                for k in site_output.keys():
                    # iterate through variable names in each site's output dict
                    if k in cls.OUT_ATTRS:
                        if out[site][k] is None:
                            continue
                        # get dtype and scale for output variable name
                        dtype = cls.OUT_ATTRS[k].get("dtype", "float32")
                        scale_factor = cls.OUT_ATTRS[k].get("scale_factor", 1)

                        # apply scale factor and dtype
                        out[site][k] *= scale_factor

                        if np.issubdtype(dtype, np.integer):
                            # round after scaling if integer dtype
                            out[site][k] = np.round(out[site][k])

                        if isinstance(out[site][k], np.ndarray):
                            # simple astype for arrays
                            out[site][k] = out[site][k].astype(dtype)
                        else:
                            # use numpy array conversion for scalar values
                            out[site][k] = np.array(
                                [out[site][k]], dtype=dtype
                            )[0]

        return out

    def _parse_gid_map(self, gid_map):
        """
        Parameters
        ----------
        gid_map : None | dict | str
            This can be None, a pre-extracted dict, or a filepath to json or
            csv. If this is a csv, it must have the columns "gid" (which
            matches the project points) and "gid_map" (gids to extract from the
            resource input)

        Returns
        -------
        gid_map : None | dict
            Mapping of unique integer generation gids (keys) to single integer
            resource gids (values). This enables the user to input unique
            generation gids in the project points that map to non-unique
            resource gids.
        """

        if isinstance(gid_map, str):
            if gid_map.endswith(".csv"):
                gid_map = pd.read_csv(gid_map).to_dict()
                msg = f"Need {ResourceMetaField.GID} in gid_map column"
                assert ResourceMetaField.GID in gid_map, msg
                assert "gid_map" in gid_map, 'Need "gid_map" in gid_map column'
                gid_map = {
                    gid_map[ResourceMetaField.GID][i]: gid_map["gid_map"][i]
                    for i in gid_map[ResourceMetaField.GID].keys()
                }

            elif gid_map.endswith(".json"):
                with open(gid_map) as f:
                    gid_map = json.load(f)

        if isinstance(gid_map, dict):
            if not self._multi_h5_res:
                res_cls = Resource
                kwargs = {"hsds": self._hsds}
            else:
                res_cls = MultiFileResource
                kwargs = {}

            with res_cls(self.res_file, **kwargs) as res:
                for gen_gid, res_gid in gid_map.items():
                    msg1 = (
                        "gid_map values must all be int but received "
                        "{}: {}".format(gen_gid, res_gid)
                    )
                    msg2 = (
                        "Could not find the gen_gid to res_gid mapping "
                        "{}: {} in the resource meta data.".format(
                            gen_gid, res_gid
                        )
                    )
                    assert isinstance(gen_gid, int), msg1
                    assert isinstance(res_gid, int), msg1
                    assert res_gid in res.meta.index.values, msg2

                for gen_gid in self.project_points.sites:
                    msg3 = (
                        "Could not find the project points gid {} in the "
                        "gen_gid input of the gid_map.".format(gen_gid)
                    )
                    assert gen_gid in gid_map, msg3

        elif gid_map is not None:
            msg = (
                "Could not parse gid_map, must be None, dict, or path to "
                "csv or json, but received: {}".format(gid_map)
            )
            logger.error(msg)
            raise InputError(msg)

        return gid_map

    def _parse_nn_map(self):
        """Parse a nearest-neighbor spatial mapping array if lr_res_file is
        provided (resource data is at two resolutions and the low-resolution
        data must be mapped to the nominal-resolution data)

        Returns
        -------
        nn_map : np.ndarray
            Optional 1D array of nearest neighbor mappings associated with the
            res_file to lr_res_file spatial mapping. For details on this
            argument, see the rex.MultiResolutionResource docstring.
        """
        nn_map = None
        if self.lr_res_file is not None:
            handler_class = Resource
            if "*" in self.res_file or "*" in self.lr_res_file:
                handler_class = MultiFileResource

            with handler_class(self.res_file) as hr_res, handler_class(
                self.lr_res_file
            ) as lr_res:
                logger.info(
                    "Making nearest neighbor map for multi "
                    "resolution resource data..."
                )
                nn_d, nn_map = MultiResolutionResource.make_nn_map(
                    hr_res, lr_res
                )
                logger.info(
                    "Done making nearest neighbor map for multi "
                    "resolution resource data!"
                )

            logger.info(
                "Made nearest neighbor mapping between nominal-"
                "resolution and low-resolution resource files. "
                "Min / mean / max dist: {:.3f} / {:.3f} / {:.3f}".format(
                    nn_d.min(), nn_d.mean(), nn_d.max()
                )
            )

        return nn_map

    @staticmethod
    def _parse_bc(bias_correct):
        """Parse the bias correction data.

        Parameters
        ----------
        bias_correct : str | pd.DataFrame, optional
            Optional DataFrame or CSV filepath to a wind or solar
            resource bias correction table. This has columns:

                - ``gid``: GID of site (can be index name of dataframe)
                - ``method``: function name from ``rex.bias_correction`` module

            The ``gid`` field should match the true resource ``gid`` regardless
            of the optional ``gid_map`` input. Only ``windspeed`` **or**
            ``GHI`` + ``DNI`` + ``DHI`` are corrected, depending on the
            technology (wind for the former, PV or CSP for the latter). See the
            functions in the ``rex.bias_correction`` module for available
            inputs for ``method``. Any additional kwargs required for the
            requested ``method`` can be input as additional columns in the
            ``bias_correct`` table e.g., for linear bias correction functions
            you can include ``scalar`` and ``adder`` inputs as columns in the
            ``bias_correct`` table on a site-by-site basis. If ``None``, no
            corrections are applied. By default, ``None``.

        Returns
        -------
        bias_correct : None | pd.DataFrame
            Optional DataFrame or CSV filepath to a wind or solar
            resource bias correction table. This has columns:

                - ``gid``: GID of site (can be index name of dataframe)
                - ``method``: function name from ``rex.bias_correction`` module

            The ``gid`` field should match the true resource ``gid`` regardless
            of the optional ``gid_map`` input. Only ``windspeed`` **or**
            ``GHI`` + ``DNI`` + ``DHI`` are corrected, depending on the
            technology (wind for the former, PV or CSP for the latter). See the
            functions in the ``rex.bias_correction`` module for available
            inputs for ``method``. Any additional kwargs required for the
            requested ``method`` can be input as additional columns in the
            ``bias_correct`` table e.g., for linear bias correction functions
            you can include ``scalar`` and ``adder`` inputs as columns in the
            ``bias_correct`` table on a site-by-site basis. If ``None``, no
            corrections are applied. By default, ``None``.
        """

        if isinstance(bias_correct, type(None)):
            return bias_correct

        if isinstance(bias_correct, str):
            bias_correct = pd.read_csv(bias_correct).rename(
                SupplyCurveField.map_to(ResourceMetaField), axis=1
            )

        msg = (
            "Bias correction data must be a filepath to csv or a dataframe "
            "but received: {}".format(type(bias_correct))
        )
        assert isinstance(bias_correct, pd.DataFrame), msg

        msg = (
            "Bias correction table must have {!r} column but only found: "
            "{}".format(ResourceMetaField.GID, list(bias_correct.columns))
        )
        assert (
            ResourceMetaField.GID in bias_correct
            or bias_correct.index.name == ResourceMetaField.GID
        ), msg

        if bias_correct.index.name != ResourceMetaField.GID:
            bias_correct = bias_correct.set_index(ResourceMetaField.GID)

        msg = (
            'Bias correction table must have "method" column but only '
            "found: {}".format(list(bias_correct.columns))
        )
        assert "method" in bias_correct, msg

        return bias_correct

    def _parse_output_request(self, req):
        """Set the output variables requested from generation.

        Parameters
        ----------
        req : list | tuple
            Output variables requested from SAM.

        Returns
        -------
        output_request : list
            Output variables requested from SAM.
        """

        output_request = super()._parse_output_request(req)

        # ensure that cf_mean is requested from output
        if "cf_mean" not in output_request:
            output_request.append("cf_mean")

        if _is_solar_run_with_ac_outputs(self.tech):
            if "dc_ac_ratio" not in output_request:
                output_request.append("dc_ac_ratio")
            for dset in ["cf_mean", "cf_profile"]:
                ac_dset = f"{dset}_ac"
                if dset in output_request and ac_dset not in output_request:
                    output_request.append(ac_dset)

        for request in output_request:
            if request not in self.OUT_ATTRS:
                msg = (
                    'User output request "{}" not recognized. '
                    "Will attempt to extract from PySAM.".format(request)
                )
                logger.debug(msg)

        return list(set(output_request))

    def _reduce_kwargs(self, pc, **kwargs):
        """Reduce the global kwargs on a per-worker basis to reduce memory
        footprint

        Parameters
        ----------
        pc : PointsControl
            PointsControl object for a single worker chunk
        kwargs : dict
            reV generation kwargs for all gids that needs to be reduced before
            being sent to ``_run_single_worker()``

        Returns
        -------
        kwargs : dict
            Same as input but reduced just for the gids in pc
        """

        gids = pc.project_points.gids
        gid_map = kwargs.get("gid_map", None)
        bias_correct = kwargs.get("bias_correct", None)

        if bias_correct is not None:
            if gid_map is not None:
                gids = [gid_map[gid] for gid in gids]

            mask = bias_correct.index.isin(gids)
            kwargs["bias_correct"] = bias_correct[mask]

        return kwargs

    def run(self, out_fpath=None, max_workers=1, timeout=1800, pool_size=None):
        """Execute a parallel reV generation run with smart data flushing.

        Parameters
        ----------
        out_fpath : str, optional
            Path to output file. If ``None``, no output file will
            be written. If the filepath is specified but the module name
            (generation) and/or resource data year is not included, the
            module name and/or resource data year will get added to the
            output file name. By default, ``None``.
        max_workers : int, optional
            Number of local workers to run on. If ``None``, or if
            running from the command line and omitting this argument
            from your config file completely, this input is set to
            ``os.cpu_count()``. Otherwise, the default is ``1``.
        timeout : int, optional
            Number of seconds to wait for parallel run iteration to
            complete before returning zeros. By default, ``1800``
            seconds.
        pool_size : int, optional
            Number of futures to submit to a single process pool for
            parallel futures. If ``None``, the pool size is set to
            ``os.cpu_count() * 2``. By default, ``None``.

        Returns
        -------
        str | None
            Path to output HDF5 file, or ``None`` if results were not
            written to disk.
        """
        # initialize output file
        self._init_fpath(out_fpath, module=ModuleName.GENERATION)
        self._init_h5()
        self._init_out_arrays()
        if pool_size is None:
            pool_size = os.cpu_count() * 2

        kwargs = {
            "tech": self.tech,
            "res_file": self.res_file,
            "lr_res_file": self.lr_res_file,
            "output_request": self.output_request,
            "scale_outputs": self.scale_outputs,
            "gid_map": self._gid_map,
            "nn_map": self._nn_map,
            "bias_correct": self._bc,
        }

        logger.info(
            "Running reV generation for: {}".format(self.points_control)
        )
        logger.debug(
            'The following project points were specified: "{}"'.format(
                self.project_points
            )
        )
        logger.debug(
            "The following SAM configs are available to this run:\n{}".format(
                pprint.pformat(self.sam_configs, indent=4)
            )
        )
        logger.debug(
            "The SAM output variables have been requested:\n{}".format(
                self.output_request
            )
        )

        # use serial or parallel execution control based on max_workers
        try:
            if max_workers == 1:
                logger.debug(
                    "Running serial generation for: {}".format(
                        self.points_control
                    )
                )
                for i, pc_sub in enumerate(self.points_control):
                    self.out = self._run_single_worker(pc_sub, **kwargs)
                    logger.info(
                        "Finished reV gen serial compute for: {} "
                        "(iteration {} out of {})".format(
                            pc_sub, i + 1, len(self.points_control)
                        )
                    )
                self.flush()
            else:
                logger.debug(
                    "Running parallel generation for: {}".format(
                        self.points_control
                    )
                )
                self._parallel_run(
                    max_workers=max_workers,
                    pool_size=pool_size,
                    timeout=timeout,
                    **kwargs,
                )

        except Exception as e:
            logger.exception("reV generation failed!")
            raise e

        return self._out_fpath


def _is_solar_run_with_ac_outputs(tech):
    """True if tech is pvwattsv8+"""
    if "pvwatts" not in tech.casefold():
        return False
    return tech.casefold() not in {f"pvwattsv{i}" for i in range(8)}
