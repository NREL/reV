# -*- coding: utf-8 -*-
# pylint: disable=anomalous-backslash-in-string
"""reV supply curve aggregation framework.

Created on Fri Jun 21 13:24:31 2019

@author: gbuster
"""
import logging
import os
from concurrent.futures import as_completed
from warnings import warn

import numpy as np
import pandas as pd
import psutil
from rex.multi_file_resource import MultiFileResource
from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool

from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.aggregation import (
    AbstractAggFileHandler,
    Aggregation,
    BaseAggregation,
)
from reV.supply_curve.exclusions import FrictionMask
from reV.supply_curve.extent import SupplyCurveExtent
from reV.supply_curve.points import GenerationSupplyCurvePoint
from reV.utilities import ResourceMetaField, SupplyCurveField, log_versions
from reV.utilities.exceptions import (
    EmptySupplyCurvePointError,
    FileInputError,
    InputWarning,
    OutputWarning,
)

logger = logging.getLogger(__name__)


class SupplyCurveAggFileHandler(AbstractAggFileHandler):
    """
    Framework to handle aggregation summary context managers:
    - exclusions .h5 file
    - generation .h5 file
    - econ .h5 file (optional)
    - friction surface .h5 file (optional)
    - variable power density .csv (optional)
    """

    def __init__(
        self,
        excl_fpath,
        gen_fpath,
        econ_fpath=None,
        data_layers=None,
        power_density=None,
        excl_dict=None,
        friction_fpath=None,
        friction_dset=None,
        area_filter_kernel="queen",
        min_area=None,
    ):
        """
        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        econ_fpath : str | None
            Filepath to .h5 reV econ output results. This is optional and only
            used if the lcoe_dset is not present in the gen_fpath file.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        power_density : float | str | None
            Power density in MW/km2 or filepath to variable power
            density file. None will attempt to infer a constant
            power density from the generation meta data technology.
            Variable power density csvs must have "gid" and "power_density"
            columns where gid is the resource gid (typically wtk or nsrdb gid)
            and the power_density column is in MW/km2.
        excl_dict : dict | None
            Dictionary of exclusion keyword arugments of the format
            {layer_dset_name: {kwarg: value}} where layer_dset_name is a
            dataset in the exclusion h5 file and kwarg is a keyword argument to
            the reV.supply_curve.exclusions.LayerMask class.
        friction_fpath : str | None
            Filepath to friction surface data (cost based exclusions).
            Must be paired with friction_dset. The friction data must be the
            same shape as the exclusions. Friction input creates a new output
            "mean_lcoe_friction" which is the nominal LCOE multiplied by the
            friction data.
        friction_dset : str | None
            Dataset name in friction_fpath for the friction surface data.
            Must be paired with friction_fpath. Must be same shape as
            exclusions.
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | None
            Minimum required contiguous area filter in sq-km
        """
        super().__init__(
            excl_fpath,
            excl_dict=excl_dict,
            area_filter_kernel=area_filter_kernel,
            min_area=min_area,
        )

        self._gen = self._open_gen_econ_resource(gen_fpath, econ_fpath)
        # pre-initialize the resource meta data
        _ = self._gen.meta

        self._data_layers = data_layers
        self._power_density = power_density
        self._parse_power_density()

        self._friction_layer = None
        if friction_fpath is not None and friction_dset is not None:
            self._friction_layer = FrictionMask(friction_fpath, friction_dset)

            if not np.all(self._friction_layer.shape == self._excl.shape):
                e = ("Friction layer shape {} must match exclusions shape {}!"
                     .format(self._friction_layer.shape, self._excl.shape))
                logger.error(e)
                raise FileInputError(e)

    @staticmethod
    def _open_gen_econ_resource(gen_fpath, econ_fpath):
        """Open a rex resource file handler for the reV generation and
        (optionally) the reV econ output(s).

        Parameters
        ----------
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        econ_fpath : str | None
            Filepath to .h5 reV econ output results. This is optional and only
            used if the lcoe_dset is not present in the gen_fpath file.

        Returns
        -------
        handler : Resource | MultiFileResource
            Open resource handler initialized with gen_fpath and (optionally)
            econ_fpath.
        """

        handler = None
        is_gen_h5 = isinstance(gen_fpath, str) and gen_fpath.endswith(".h5")
        is_econ_h5 = isinstance(econ_fpath, str) and econ_fpath.endswith(".h5")

        if is_gen_h5 and not is_econ_h5:
            handler = Resource(gen_fpath)
        elif is_gen_h5 and is_econ_h5:
            handler = MultiFileResource(
                [gen_fpath, econ_fpath], check_files=True
            )

        return handler

    def _parse_power_density(self):
        """Parse the power density input. If file, open file handler."""

        if isinstance(self._power_density, str):
            self._pdf = self._power_density

            if self._pdf.endswith(".csv"):
                self._power_density = pd.read_csv(self._pdf)
                if (ResourceMetaField.GID in self._power_density
                        and 'power_density' in self._power_density):
                    self._power_density = \
                        self._power_density.set_index(ResourceMetaField.GID)
                else:
                    msg = ('Variable power density file must include "{}" '
                           'and "power_density" columns, but received: {}'
                           .format(ResourceMetaField.GID,
                                   self._power_density.columns.values))
                    logger.error(msg)
                    raise FileInputError(msg)
            else:
                msg = (
                    "Variable power density file must be csv but received: "
                    "{}".format(self._pdf)
                )
                logger.error(msg)
                raise FileInputError(msg)

    def close(self):
        """Close all file handlers."""
        self._excl.close()
        self._gen.close()
        if self._friction_layer is not None:
            self._friction_layer.close()

    @property
    def gen(self):
        """Get the gen file handler object.

        Returns
        -------
        _gen : Outputs
            reV gen outputs handler object.
        """
        return self._gen

    @property
    def data_layers(self):
        """Get the data layers object.

        Returns
        -------
        _data_layers : dict
            Data layers namespace.
        """
        return self._data_layers

    @property
    def power_density(self):
        """Get the power density object.

        Returns
        -------
        _power_density : float | None | pd.DataFrame
            Constant power density float, None, or opened dataframe with
            (resource) "gid" and "power_density columns".
        """
        return self._power_density

    @property
    def friction_layer(self):
        """Get the friction layer (cost based exclusions).

        Returns
        -------
        friction_layer : None | FrictionMask
            Friction layer with scalar friction values if valid friction inputs
            were entered. Otherwise, None to not apply friction layer.
        """
        return self._friction_layer


class SupplyCurveAggregation(BaseAggregation):
    """SupplyCurveAggregation"""

    def __init__(self, excl_fpath, tm_dset, econ_fpath=None,
                 excl_dict=None, area_filter_kernel='queen', min_area=None,
                 resolution=64, excl_area=None, res_fpath=None, gids=None,
                 pre_extract_inclusions=False, res_class_dset=None,
                 res_class_bins=None, cf_dset='cf_mean-means',
                 lcoe_dset='lcoe_fcr-means', h5_dsets=None, data_layers=None,
                 power_density=None, friction_fpath=None, friction_dset=None,
                 cap_cost_scale=None, recalc_lcoe=True, zones_dset=None):
        r"""ReV supply curve points aggregation framework.

        ``reV`` supply curve aggregation combines a high-resolution
        (e.g. 90m) exclusion dataset with a (typically) lower resolution
        (e.g. 2km) generation dataset by mapping all data onto the high-
        resolution grid and aggregating it by a large factor (e.g. 64 or
        128). The result is coarsely-gridded data that summarizes
        capacity and generation potential as well as associated
        economics under a particular land access scenario. This module
        can also summarize extra data layers during the aggregation
        process, allowing for complementary land characterization
        analysis.

        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions data HDF5 file. The exclusions HDF5
            file should contain the layers specified in `excl_dict`
            and `data_layers`. These layers may also be spread out
            across multiple HDF5 files, in which case this input should
            be a list or tuple of filepaths pointing to the files
            containing the layers. Note that each data layer must be
            uniquely defined (i.e.only appear once and in a single
            input file).
        tm_dset : str
            Dataset name in the `excl_fpath` file containing the
            techmap (exclusions-to-resource mapping data). This data
            layer links the supply curve GID's to the generation GID's
            that are used to evaluate performance metrics such as
            ``mean_cf``.

            .. Important:: This dataset uniquely couples the (typically
              high-resolution) exclusion layers to the (typically
              lower-resolution) resource data. Therefore, a separate
              techmap must be used for every unique combination of
              resource and exclusion coordinates.

            .. Note:: If executing ``reV`` from the command line, you
              can specify a name that is not in the exclusions HDF5
              file, and ``reV`` will calculate the techmap for you. Note
              however that computing the techmap and writing it to the
              exclusion HDF5 file is a blocking operation, so you may
              only run a single ``reV`` aggregation step at a time this
              way.
        econ_fpath : str, optional
            Filepath to HDF5 file with ``reV`` econ output results
            containing an `lcoe_dset` dataset. If ``None``, `lcoe_dset`
            should be a dataset in the `gen_fpath` HDF5 file that
            aggregation is executed on.

            .. Note:: If executing ``reV`` from the command line, this
              input can be set to ``"PIPELINE"`` to parse the output
              from one of these preceding pipeline steps:
              ``multi-year``, ``collect``, or ``generation``. However,
              note that duplicate executions of any of these commands
              within the pipeline may invalidate this parsing, meaning
              the `econ_fpath` input will have to be specified manually.

            By default, ``None``.
        excl_dict : dict | None
            Dictionary of exclusion keyword arguments of the format
            ``{layer_dset_name: {kwarg: value}}``, where
            ``layer_dset_name`` is a dataset in the exclusion h5 file
            and the ``kwarg: value`` pair is a keyword argument to
            the :class:`reV.supply_curve.exclusions.LayerMask` class.
            For example::

                excl_dict = {
                    "typical_exclusion": {
                        "exclude_values": 255,
                    },
                    "another_exclusion": {
                        "exclude_values": [2, 3],
                        "weight": 0.5
                    },
                    "exclusion_with_nodata": {
                        "exclude_range": [10, 100],
                        "exclude_nodata": True,
                        "nodata_value": -1
                    },
                    "partial_setback": {
                        "use_as_weights": True
                    },
                    "height_limit": {
                        "exclude_range": [0, 200]
                    },
                    "slope": {
                        "include_range": [0, 20]
                    },
                    "developable_land": {
                        "force_include_values": 42
                    },
                    "more_developable_land": {
                        "force_include_range": [5, 10]
                    },
                    "viewsheds": {
                        "exclude_values": 1,
                        "extent": {
                            "layer": "federal_parks",
                            "include_range": [1, 5]
                        }
                    }
                    ...
                }

            Note that all the keys given in this dictionary should be
            datasets of the `excl_fpath` file. If ``None`` or empty
            dictionary, no exclusions are applied. By default, ``None``.
        area_filter_kernel : {"queen", "rook"}, optional
            Contiguous area filter method to use on final exclusions
            mask. The filters are defined as::

                # Queen:     # Rook:
                [[1,1,1],    [[0,1,0],
                 [1,1,1],     [1,1,1],
                 [1,1,1]]     [0,1,0]]

            These filters define how neighboring pixels are "connected".
            Once pixels in the final exclusion layer are connected, the
            area of each resulting cluster is computed and compared
            against the `min_area` input. Any cluster with an area
            less than `min_area` is excluded from the final mask.
            This argument has no effect if `min_area` is ``None``.
            By default, ``"queen"``.
        min_area : float, optional
            Minimum area (in km\ :sup:`2`) required to keep an isolated
            cluster of (included) land within the resulting exclusions
            mask. Any clusters of land with areas less than this value
            will be marked as exclusions. See the documentation for
            `area_filter_kernel` for an explanation of how the area of
            each land cluster is computed. If ``None``, no area
            filtering is performed. By default, ``None``.
        resolution : int, optional
            Supply Curve resolution. This value defines how many pixels
            are in a single side of a supply curve cell. For example,
            a value of ``64`` would generate a supply curve where the
            side of each supply curve cell is ``64x64`` exclusion
            pixels. By default, ``64``.
        excl_area : float, optional
            Area of a single exclusion mask pixel (in km\ :sup:`2`).
            If ``None``, this value will be inferred from the profile
            transform attribute in `excl_fpath`. By default, ``None``.
        res_fpath : str, optional
            Filepath to HDF5 resource file (e.g. WTK or NSRDB). This
            input is required if techmap dset is to be created or if the
            ``gen_fpath`` input to the ``summarize`` or ``run`` methods
            is ``None``. By default, ``None``.
        gids : list, optional
            List of supply curve point gids to get summary for. If you
            would like to obtain all available ``reV`` supply curve
            points to run, you can use the
            :class:`reV.supply_curve.extent.SupplyCurveExtent` class
            like so::

                import pandas as pd
                from reV.supply_curve.extent import SupplyCurveExtent

                excl_fpath = "..."
                resolution = ...
                tm_dset = "..."
                with SupplyCurveExtent(excl_fpath, resolution) as sc:
                    gids = sc.valid_sc_points(tm_dset).tolist()
                ...

            If ``None``, supply curve aggregation is computed for all
            gids in the supply curve extent. By default, ``None``.
        pre_extract_inclusions : bool, optional
            Optional flag to pre-extract/compute the inclusion mask from
            the `excl_dict` input. It is typically faster to compute
            the inclusion mask on the fly with parallel workers.
            By default, ``False``.
        res_class_dset : str, optional
            Name of dataset in the ``reV`` generation HDF5 output file
            containing resource data. If ``None``, no aggregated
            resource classification is performed (i.e. no ``mean_res``
            output), and the `res_class_bins` is ignored.
            By default, ``None``.
        res_class_bins : list, optional
            Optional input to perform separate aggregations for various
            resource data ranges. If ``None``, only a single aggregation
            per supply curve point is performed. Otherwise, this input
            should be a list of floats or ints representing the resource
            bin boundaries. One aggregation per resource value range is
            computed, and only pixels within the given resource range
            are aggregated. By default, ``None``.
        cf_dset : str, optional
            Dataset name from the ``reV`` generation HDF5 output file
            containing a 1D dataset of mean capacity factor values. This
            dataset will be mapped onto the high-resolution grid and
            used to compute the mean capacity factor for non-excluded
            area. By default, ``"cf_mean-means"``.
        lcoe_dset : str, optional
            Dataset name from the ``reV`` generation HDF5 output file
            containing a 1D dataset of mean LCOE values. This
            dataset will be mapped onto the high-resolution grid and
            used to compute the mean LCOE for non-excluded area, but
            only if the LCOE is not re-computed during processing (see
            the `recalc_lcoe` input for more info).
            By default, ``"lcoe_fcr-means"``.
        h5_dsets : list, optional
            Optional list of additional datasets from the ``reV``
            generation/econ HDF5 output file to aggregate. If ``None``,
            no extra datasets are aggregated.

            .. WARNING:: This input is meant for passing through 1D
               datasets. If you specify a 2D or higher-dimensional
               dataset, you may run into memory errors. If you wish to
               aggregate 2D datasets, see the rep-profiles module.

            By default, ``None``.
        data_layers : dict, optional
            Dictionary of aggregation data layers of the format::

                data_layers = {
                    "output_layer_name": {
                        "dset": "layer_name",
                        "method": "mean",
                        "fpath": "/path/to/data.h5"
                    },
                    "another_output_layer_name": {
                        "dset": "input_layer_name",
                        "method": "mode",
                        # optional "fpath" key omitted
                    },
                    ...
                }

            The ``"output_layer_name"`` is the column name under which
            the aggregated data will appear in the output CSV file. The
            ``"output_layer_name"`` does not have to match the ``dset``
            input value. The latter should match the layer name in the
            HDF5 from which the data to aggregate should be pulled. The
            ``method`` should be one of
            ``{"mode", "mean", "min", "max", "sum", "category"}``,
            describing how the high-resolution data should be aggregated
            for each supply curve point. ``fpath`` is an optional key
            that can point to an HDF5 file containing the layer data. If
            left out, the data is assumed to exist in the file(s)
            specified by the `excl_fpath` input. If ``None``, no data
            layer aggregation is performed. By default, ``None``
        power_density : float | str, optional
            Power density value (in MW/km\ :sup:`2`) or filepath to
            variable power density CSV file containing the following
            columns:

                - ``gid`` : resource gid (typically wtk or nsrdb gid)
                - ``power_density`` : power density value (in
                  MW/km\ :sup:`2`)

            If ``None``, a constant power density is inferred from the
            generation meta data technology. By default, ``None``.
        friction_fpath : str, optional
            Filepath to friction surface data (cost based exclusions).
            Must be paired with the `friction_dset` input below. The
            friction data must be the same shape as the exclusions.
            Friction input creates a new output column
            ``"mean_lcoe_friction"`` which is the nominal LCOE
            multiplied by the friction data. If ``None``, no friction
            data is aggregated. By default, ``None``.
        friction_dset : str, optional
            Dataset name in friction_fpath for the friction surface
            data. Must be paired with the `friction_fpath` above. If
            ``None``, no friction data is aggregated.
            By default, ``None``.
        cap_cost_scale : str, optional
            Optional LCOE scaling equation to implement "economies of
            scale". Equations must be in python string format and must
            return a scalar value to multiply the capital cost by.
            Independent variables in the equation should match the names
            of the columns in the ``reV`` supply curve aggregation
            output table (see the documentation of
            :class:`~reV.supply_curve.sc_aggregation.SupplyCurveAggregation`
            for details on available outputs). If ``None``, no economies
            of scale are applied. By default, ``None``.
        recalc_lcoe : bool, optional
            Flag to re-calculate the LCOE from the multi-year mean
            capacity factor and annual energy production data. This
            requires several datasets to be aggregated in the h5_dsets
            input:

                - ``system_capacity``
                - ``fixed_charge_rate``
                - ``capital_cost``
                - ``fixed_operating_cost``
                - ``variable_operating_cost``

            If any of these datasets are missing from the ``reV``
            generation HDF5 output, or if `recalc_lcoe` is set to
            ``False``, the mean LCOE will be computed from the data
            stored under the `lcoe_dset` instead. By default, ``True``.
        zones_dset: str, optional
            Dataset name in `excl_fpath` containing the zones to be applied.
            If specified, supply curve aggregation will be performed separately
            for each discrete zone within each supply curve site. This option
            can be used for uses cases such as subdividing sites by parcel,
            such that each parcel within each site is output to a separate
            sc_gid. The input data layer should consist of unique integer
            values for each zone. Values of zero will be treated as excluded
            areas.

        Examples
        --------
        Standard outputs:

        sc_gid : int
            Unique supply curve gid. This is the enumerated supply curve
            points, which can have overlapping geographic locations due
            to different resource bins at the same geographic SC point.
        res_gids : list
            Stringified list of resource gids (e.g. original WTK or
            NSRDB resource GIDs) corresponding to each SC point.
        gen_gids : list
            Stringified list of generation gids (e.g. GID in the reV
            generation output, which corresponds to the reV project
            points and not necessarily the resource GIDs).
        gid_counts : list
            Stringified list of the sum of inclusion scalar values
            corresponding to each `gen_gid` and `res_gid`, where 1 is
            included, 0 is excluded, and 0.7 is included with 70 percent
            of available land. Each entry in this list is associated
            with the corresponding entry in the `gen_gids` and
            `res_gids` lists.
        n_gids : int
            Total number of included pixels. This is a boolean sum and
            considers partial inclusions to be included (e.g. 1).
        mean_cf : float
            Mean capacity factor of each supply curve point (the
            arithmetic mean is weighted by the inclusion layer)
            (unitless).
        mean_lcoe : float
            Mean LCOE of each supply curve point (the arithmetic mean is
            weighted by the inclusion layer). Units match the reV econ
            output ($/MWh). By default, the LCOE is re-calculated using
            the multi-year mean capacity factor and annual energy
            production. This requires several datasets to be aggregated
            in the h5_dsets input: ``fixed_charge_rate``,
            ``capital_cost``,
            ``fixed_operating_cost``, ``annual_energy_production``, and
            ``variable_operating_cost``. This recalc behavior can be
            disabled by setting ``recalc_lcoe=False``.
        mean_res : float
            Mean resource, the resource dataset to average is provided
            by the user in `res_class_dset`. The arithmetic mean is
            weighted by the inclusion layer.
        capacity : float
            Total capacity of each supply curve point (MW). Units are
            contingent on the `power_density` input units of MW/km2.
        area_sq_km : float
            Total included area for each supply curve point in km2. This
            is based on the nominal area of each exclusion pixel which
            by default is calculated from the exclusion profile
            attributes. The NREL reV default is 0.0081 km2 pixels
            (90m x 90m). The area sum considers partial inclusions.
        latitude : float
            Supply curve point centroid latitude coordinate, in degrees
            (does not consider exclusions).
        longitude : float
            Supply curve point centroid longitude coordinate, in degrees
            (does not consider exclusions).
        country : str
            Country of the supply curve point based on the most common
            country of the associated resource meta data. Does not
            consider exclusions.
        state : str
            State of the supply curve point based on the most common
            state of the associated resource meta data. Does not
            consider exclusions.
        county : str
            County of the supply curve point based on the most common
            county of the associated resource meta data. Does not
            consider exclusions.
        elevation : float
            Mean elevation of the supply curve point based on the mean
            elevation of the associated resource meta data. Does not
            consider exclusions.
        timezone : int
            UTC offset of local timezone based on the most common
            timezone of the associated resource meta data. Does not
            consider exclusions.
        sc_point_gid : int
            Spatially deterministic supply curve point gid. Duplicate
            `sc_point_gid` values can exist due to resource binning.
        sc_row_ind : int
            Row index of the supply curve point in the aggregated
            exclusion grid.
        sc_col_ind : int
            Column index of the supply curve point in the aggregated
            exclusion grid
        res_class : int
            Resource class for the supply curve gid. Each geographic
            supply curve point (`sc_point_gid`) can have multiple
            resource classes associated with it, resulting in multiple
            supply curve gids (`sc_gid`) associated with the same
            spatially deterministic supply curve point.


        Optional outputs:

        mean_friction : float
            Mean of the friction data provided in 'friction_fpath' and
            'friction_dset'. The arithmetic mean is weighted by boolean
            inclusions and considers partial inclusions to be included.
        mean_lcoe_friction : float
            Mean of the nominal LCOE multiplied by mean_friction value.
        mean_{dset} : float
            Mean input h5 dataset(s) provided by the user in 'h5_dsets'.
            These mean calculations are weighted by the partial
            inclusion layer.
        data_layers : float | int | str | dict
            Requested data layer aggregations, each data layer must be
            the same shape as the exclusion layers.

                - mode: int | str
                    Most common value of a given data layer after
                    applying the boolean inclusion mask.
                - mean : float
                    Arithmetic mean value of a given data layer weighted
                    by the scalar inclusion mask (considers partial
                    inclusions).
                - min : float | int
                    Minimum value of a given data layer after applying
                    the boolean inclusion mask.
                - max : float | int
                    Maximum value of a given data layer after applying
                    the boolean inclusion mask.
                - sum : float
                    Sum of a given data layer weighted by the scalar
                    inclusion mask (considers partial inclusions).
                - category : dict
                    Dictionary mapping the unique values in the
                    `data_layer` to the sum of inclusion scalar values
                    associated with all pixels with that unique value.
        """
        log_versions(logger)
        logger.info("Initializing SupplyCurveAggregation...")
        logger.debug("Exclusion filepath: {}".format(excl_fpath))
        logger.debug("Exclusion dict: {}".format(excl_dict))

        super().__init__(
            excl_fpath,
            tm_dset,
            excl_dict=excl_dict,
            area_filter_kernel=area_filter_kernel,
            min_area=min_area,
            resolution=resolution,
            excl_area=excl_area,
            res_fpath=res_fpath,
            gids=gids,
            pre_extract_inclusions=pre_extract_inclusions,
        )

        self._econ_fpath = econ_fpath
        self._res_class_dset = res_class_dset
        self._res_class_bins = self._convert_bins(res_class_bins)
        self._cf_dset = cf_dset
        self._lcoe_dset = lcoe_dset
        self._h5_dsets = h5_dsets
        self._cap_cost_scale = cap_cost_scale
        self._power_density = power_density
        self._friction_fpath = friction_fpath
        self._friction_dset = friction_dset
        self._data_layers = data_layers
        self._recalc_lcoe = recalc_lcoe
        self._zones_dset = zones_dset

        logger.debug("Resource class bins: {}".format(self._res_class_bins))

        if self._power_density is None:
            msg = (
                "Supply curve aggregation power density not specified. "
                "Will try to infer based on lookup table: {}".format(
                    GenerationSupplyCurvePoint.POWER_DENSITY
                )
            )
            logger.warning(msg)
            warn(msg, InputWarning)

        self._check_data_layers()

    def _check_data_layers(
        self, methods=("mean", "max", "min", "mode", "sum", "category")
    ):
        """Run pre-flight checks on requested aggregation data layers.

        Parameters
        ----------
        methods : list | tuple
            Data layer aggregation methods that are available to the user.
        """

        if self._data_layers is not None:
            logger.debug("Checking data layers...")

            with ExclusionLayers(self._excl_fpath) as f:
                shape_base = f.shape

            for k, v in self._data_layers.items():
                if "dset" not in v:
                    raise KeyError(
                        'Data aggregation "dset" data layer "{}" '
                        "must be specified.".format(k)
                    )
                if "method" not in v:
                    raise KeyError(
                        'Data aggregation "method" data layer "{}" '
                        "must be specified.".format(k)
                    )
                if v["method"].lower() not in methods:
                    raise ValueError(
                        "Cannot recognize data layer agg method: "
                        '"{}". Can only do: {}.'.format(v["method"], methods)
                    )
                if "fpath" in v:
                    with ExclusionLayers(v["fpath"]) as f:
                        try:
                            mismatched_shapes = any(f.shape != shape_base)
                        except TypeError:
                            mismatched_shapes = f.shape != shape_base
                        if mismatched_shapes:
                            msg = (
                                'Data shape of data layer "{}" is {}, '
                                "which does not match the baseline "
                                "exclusions shape {}.".format(
                                    k, f.shape, shape_base
                                )
                            )
                            raise FileInputError(msg)

        logger.debug("Finished checking data layers.")

    @staticmethod
    def _get_res_gen_lcoe_data(
        gen, res_class_dset, res_class_bins, lcoe_dset
    ):
        """Extract the basic resource / generation / lcoe data to be used in
        the aggregation process.

        Parameters
        ----------
        gen : Resource | MultiFileResource
            Open rex resource handler initialized from gen_fpath and
            (optionally) econ_fpath.
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of two-entry lists dictating the resource class bins.
            None if no resource classes.
        lcoe_dset : str
            Dataset name from f_gen containing LCOE mean values.

        Returns
        -------
        res_data : np.ndarray | None
            Extracted resource data from res_class_dset
        res_class_bins : list
            List of resouce class bin ranges.
        lcoe_data : np.ndarray | None
            LCOE data extracted from lcoe_dset in gen
        """

        dset_list = (res_class_dset, lcoe_dset)
        gen_dsets = [] if gen is None else gen.datasets
        labels = ("res_class_dset", "lcoe_dset")
        temp = [None, None]

        if isinstance(gen, Resource):
            source_fps = [gen.h5_file]
        elif isinstance(gen, MultiFileResource):
            source_fps = gen._h5_files
        else:
            msg = 'Did not recognize gen object input of type "{}": {}'.format(
                type(gen), gen
            )
            logger.error(msg)
            raise TypeError(msg)

        for i, dset in enumerate(dset_list):
            if dset in gen_dsets:
                _warn_about_large_datasets(gen, dset)
                temp[i] = gen[dset]
            elif dset not in gen_dsets and dset is not None:
                w = (
                    'Could not find "{}" input as "{}" in source files: {}. '
                    "Available datasets: {}".format(
                        labels[i], dset, source_fps, gen_dsets
                    )
                )
                logger.warning(w)
                warn(w, OutputWarning)

        res_data, lcoe_data = temp

        if res_class_dset is None or res_class_bins is None:
            res_class_bins = [None]

        return res_data, res_class_bins, lcoe_data

    @staticmethod
    def _get_extra_dsets(gen, h5_dsets):
        """Extract extra ancillary datasets to be used in the aggregation
        process

        Parameters
        ----------
        gen : Resource | MultiFileResource
            Open rex resource handler initialized from gen_fpath and
            (optionally) econ_fpath.
        h5_dsets : list | None
            Optional list of additional datasets from the source h5 gen/econ
            files to aggregate.

        Returns
        -------
        h5_dsets_data : dict | None
            If additional h5_dsets are requested, this will be a dictionary
            keyed by the h5 dataset names. The corresponding values will be
            the extracted arrays from the h5 files.
        """

        # look for the datasets required by the LCOE re-calculation and make
        # lists of the missing datasets
        gen_dsets = [] if gen is None else gen.datasets
        lcoe_recalc_req = ('fixed_charge_rate',
                           'capital_cost',
                           'fixed_operating_cost',
                           'variable_operating_cost',
                           'system_capacity')
        missing_lcoe_source = [k for k in lcoe_recalc_req
                               if k not in gen_dsets]

        if isinstance(gen, Resource):
            source_fps = [gen.h5_file]
        elif isinstance(gen, MultiFileResource):
            source_fps = gen._h5_files
        else:
            msg = 'Did not recognize gen object input of type "{}": {}'.format(
                type(gen), gen
            )
            logger.error(msg)
            raise TypeError(msg)

        h5_dsets_data = None
        if h5_dsets is not None:

            if not isinstance(h5_dsets, (list, tuple)):
                e = (
                    "Additional h5_dsets argument must be a list or tuple "
                    "but received: {} {}".format(type(h5_dsets), h5_dsets)
                )
                logger.error(e)
                raise TypeError(e)

            missing_h5_dsets = [k for k in h5_dsets if k not in gen_dsets]
            if any(missing_h5_dsets):
                msg = (
                    'Could not find requested h5_dsets "{}" in '
                    "source files: {}. Available datasets: {}".format(
                        missing_h5_dsets, source_fps, gen_dsets
                    )
                )
                logger.error(msg)
                raise FileInputError(msg)

            h5_dsets_data = {dset: gen[dset] for dset in h5_dsets}

        if any(missing_lcoe_source):
            msg = (
                "Could not find the datasets in the gen source file that "
                "are required to re-calculate the multi-year LCOE. If you "
                "are running a multi-year job, it is strongly suggested "
                "you pass through these datasets to re-calculate the LCOE "
                "from the multi-year mean CF: {}".format(missing_lcoe_source)
            )
            logger.warning(msg)
            warn(msg, InputWarning)

        return h5_dsets_data

    @classmethod
    def run_serial(
        cls,
        excl_fpath,
        gen_fpath,
        tm_dset,
        gen_index,
        econ_fpath=None,
        excl_dict=None,
        inclusion_mask=None,
        area_filter_kernel="queen",
        min_area=None,
        resolution=64,
        gids=None,
        args=None,
        res_class_dset=None,
        res_class_bins=None,
        cf_dset="cf_mean-means",
        lcoe_dset="lcoe_fcr-means",
        h5_dsets=None,
        data_layers=None,
        power_density=None,
        friction_fpath=None,
        friction_dset=None,
        excl_area=None,
        cap_cost_scale=None,
        recalc_lcoe=True,
        zones_dset=None,
    ):
        """Standalone method to create agg summary - can be parallelized.

        Parameters
        ----------
        excl_fpath : str | list | tuple
            Filepath to exclusions h5 with techmap dataset
            (can be one or more filepaths).
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        tm_dset : str
            Dataset name in the exclusions file containing the
            exclusions-to-resource mapping data.
        gen_index : np.ndarray
            Array of generation gids with array index equal to resource gid.
            Array value is -1 if the resource index was not used in the
            generation run.
        econ_fpath : str | None
            Filepath to .h5 reV econ output results. This is optional and only
            used if the lcoe_dset is not present in the gen_fpath file.
        excl_dict : dict | None
            Dictionary of exclusion keyword arugments of the format
            {layer_dset_name: {kwarg: value}} where layer_dset_name is a
            dataset in the exclusion h5 file and kwarg is a keyword argument to
            the reV.supply_curve.exclusions.LayerMask class.
        inclusion_mask : np.ndarray | dict | optional
            2D array pre-extracted inclusion mask where 1 is included and 0 is
            excluded. This must be either match the full exclusion shape or
            be a dict lookup of single-sc-point exclusion masks corresponding
            to the gids input and keyed by gids, by default None which will
            calculate exclusions on the fly for each sc point.
        area_filter_kernel : str
            Contiguous area filter method to use on final exclusions mask
        min_area : float | None
            Minimum required contiguous area filter in sq-km
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gids : list | None
            List of supply curve point gids to get summary for (can use to
            subset if running in parallel), or None for all gids in the SC
            extent, by default None
        args : list | None
            List of positional args for sc_point_method
        res_class_dset : str | None
            Dataset in the generation file dictating resource classes.
            None if no resource classes.
        res_class_bins : list | None
            List of two-entry lists dictating the resource class bins.
            None if no resource classes.
        cf_dset : str
            Dataset name from f_gen containing capacity factor mean values.
        lcoe_dset : str
            Dataset name from f_gen containing LCOE mean values.
        h5_dsets : list | None
            Optional list of additional datasets from the source h5 gen/econ
            files to aggregate.
        data_layers : None | dict
            Aggregation data layers. Must be a dictionary keyed by data label
            name. Each value must be another dictionary with "dset", "method",
            and "fpath".
        power_density : float | str | None
            Power density in MW/km2 or filepath to variable power
            density file. None will attempt to infer a constant
            power density from the generation meta data technology.
            Variable power density csvs must have "gid" and "power_density"
            columns where gid is the resource gid (typically wtk or nsrdb gid)
            and the power_density column is in MW/km2.
        friction_fpath : str | None
            Filepath to friction surface data (cost based exclusions).
            Must be paired with friction_dset. The friction data must be the
            same shape as the exclusions. Friction input creates a new output
            "mean_lcoe_friction" which is the nominal LCOE multiplied by the
            friction data.
        friction_dset : str | None
            Dataset name in friction_fpath for the friction surface data.
            Must be paired with friction_fpath. Must be same shape as
            exclusions.
        excl_area : float | None, optional
            Area of an exclusion pixel in km2. None will try to infer the area
            from the profile transform attribute in excl_fpath, by default None
        cap_cost_scale : str | None
            Optional LCOE scaling equation to implement "economies of scale".
            Equations must be in python string format and return a scalar
            value to multiply the capital cost by. Independent variables in
            the equation should match the names of the columns in the reV
            supply curve aggregation table.
        recalc_lcoe : bool
            Flag to re-calculate the LCOE from the multi-year mean capacity
            factor and annual energy production data. This requires several
            datasets to be aggregated in the h5_dsets input: system_capacity,
            fixed_charge_rate, capital_cost, fixed_operating_cost,
            and variable_operating_cost.
        zones_dset: str, optional
            Dataset name in `excl_fpath` containing the zones to be applied.
            If specified, supply curve aggregation will be performed separately
            for each discrete zone within each supply curve site. This option
            can be used for uses cases such as subdividing sites by parcel,
            such that each parcel within each site is output to a separate
            sc_gid. The input data layer should consist of unique integer
            values for each zone. Values of zero will be treated as excluded
            areas.

        Returns
        -------
        summary : list
            List of dictionaries, each being an SC point summary.
        """
        summary = []

        with SupplyCurveExtent(excl_fpath, resolution=resolution) as sc:
            exclusion_shape = sc.exclusions.shape
            if gids is None:
                gids = sc.valid_sc_points(tm_dset)
            elif np.issubdtype(type(gids), np.number):
                gids = [gids]

            slice_lookup = sc.get_slice_lookup(gids)

        logger.debug(
            "Starting SupplyCurveAggregation serial with "
            "supply curve {} gids".format(len(gids))
        )

        cls._check_inclusion_mask(inclusion_mask, gids, exclusion_shape)

        # pre-extract handlers so they are not repeatedly initialized
        file_kwargs = {
            "econ_fpath": econ_fpath,
            "data_layers": data_layers,
            "power_density": power_density,
            "excl_dict": excl_dict,
            "area_filter_kernel": area_filter_kernel,
            "min_area": min_area,
            "friction_fpath": friction_fpath,
            "friction_dset": friction_dset,
        }
        with SupplyCurveAggFileHandler(
            excl_fpath, gen_fpath, **file_kwargs
        ) as fh:
            temp = cls._get_res_gen_lcoe_data(
                fh.gen, res_class_dset, res_class_bins, lcoe_dset
            )
            res_data, res_class_bins, lcoe_data = temp
            h5_dsets_data = cls._get_extra_dsets(fh.gen, h5_dsets)

            n_finished = 0
            for gid in gids:
                gid_inclusions = cls._get_gid_inclusion_mask(
                    inclusion_mask, gid, slice_lookup, resolution=resolution
                )

                zones = cls._get_gid_zones(
                    excl_fpath, zones_dset, gid, slice_lookup
                )
                zone_ids = np.unique(zones[zones != 0]).tolist()

                for ri, res_bin in enumerate(res_class_bins):
                    for zi, zone_id in enumerate(zone_ids, start=1):
                        zone_mask = zones == zone_id
                        try:
                            pointsum = GenerationSupplyCurvePoint.summarize(
                                gid,
                                fh.exclusions,
                                fh.gen,
                                tm_dset,
                                gen_index,
                                res_class_dset=res_data,
                                res_class_bin=res_bin,
                                cf_dset=cf_dset,
                                lcoe_dset=lcoe_data,
                                h5_dsets=h5_dsets_data,
                                data_layers=fh.data_layers,
                                resolution=resolution,
                                exclusion_shape=exclusion_shape,
                                power_density=fh.power_density,
                                args=args,
                                excl_dict=excl_dict,
                                inclusion_mask=gid_inclusions,
                                excl_area=excl_area,
                                close=False,
                                friction_layer=fh.friction_layer,
                                cap_cost_scale=cap_cost_scale,
                                recalc_lcoe=recalc_lcoe,
                                zone_mask=zone_mask,
                            )

                        except EmptySupplyCurvePointError:
                            logger.debug("SC point {}, zone ID {} is empty"
                                         .format(gid, zone_id))
                        else:
                            pointsum['res_class'] = ri
                            pointsum['zone_id'] = zone_id

                            summary.append(pointsum)
                            logger.debug(
                                "Serial aggregation completed for"
                                "resource class {}, zone ID {}: "
                                "{:,d} out of {:,d} zones complete".format(
                                    ri, zone_id, zi, len(zone_ids)
                                )
                            )

                n_finished += 1
                logger.debug(
                    "Serial aggregation completed for gid {}: "
                    "{:,d} out of {:,d} points complete".format(
                        gid, n_finished, len(gids)
                    )
                )

        return summary

    def run_parallel(
        self, gen_fpath, args=None, max_workers=None, sites_per_worker=100
    ):
        """Get the supply curve points aggregation summary using futures.

        Parameters
        ----------
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        args : tuple | list | None
            List of summary arguments to include. None defaults to all
            available args defined in the class attr.
        max_workers : int | None, optional
            Number of cores to run summary on. None is all
            available cpus, by default None
        sites_per_worker : int
            Number of sc_points to summarize on each worker, by default 100

        Returns
        -------
        summary : list
            List of dictionaries, each being an SC point summary.
        """

        gen_index = self._parse_gen_index(gen_fpath)
        chunks = int(np.ceil(len(self.gids) / sites_per_worker))
        chunks = np.array_split(self.gids, chunks)

        logger.info(
            "Running supply curve point aggregation for "
            "points {} through {} at a resolution of {} "
            "on {} cores in {} chunks.".format(
                self.gids[0],
                self.gids[-1],
                self._resolution,
                max_workers,
                len(chunks),
            )
        )

        slice_lookup = None
        if self._inclusion_mask is not None:
            with SupplyCurveExtent(
                self._excl_fpath, resolution=self._resolution
            ) as sc:
                assert sc.exclusions.shape == self._inclusion_mask.shape
                slice_lookup = sc.get_slice_lookup(self.gids)

        futures = []
        summary = []
        n_finished = 0
        loggers = [__name__, "reV.supply_curve.point_summary", "reV"]
        with SpawnProcessPool(max_workers=max_workers, loggers=loggers) as exe:
            # iterate through split executions, submitting each to worker
            for gid_set in chunks:
                # submit executions and append to futures list
                chunk_incl_masks = None
                if self._inclusion_mask is not None:
                    chunk_incl_masks = {}
                    for gid in gid_set:
                        rs, cs = slice_lookup[gid]
                        chunk_incl_masks[gid] = self._inclusion_mask[rs, cs]

                futures.append(
                    exe.submit(
                        self.run_serial,
                        self._excl_fpath,
                        gen_fpath,
                        self._tm_dset,
                        gen_index,
                        econ_fpath=self._econ_fpath,
                        excl_dict=self._excl_dict,
                        inclusion_mask=chunk_incl_masks,
                        res_class_dset=self._res_class_dset,
                        res_class_bins=self._res_class_bins,
                        cf_dset=self._cf_dset,
                        lcoe_dset=self._lcoe_dset,
                        h5_dsets=self._h5_dsets,
                        data_layers=self._data_layers,
                        resolution=self._resolution,
                        power_density=self._power_density,
                        friction_fpath=self._friction_fpath,
                        friction_dset=self._friction_dset,
                        area_filter_kernel=self._area_filter_kernel,
                        min_area=self._min_area,
                        gids=gid_set,
                        args=args,
                        excl_area=self._excl_area,
                        cap_cost_scale=self._cap_cost_scale,
                        recalc_lcoe=self._recalc_lcoe,
                        zones_dset=self._zones_dset,
                    )
                )

            # gather results
            for future in as_completed(futures):
                n_finished += 1
                summary += future.result()
                if n_finished % 10 == 0:
                    mem = psutil.virtual_memory()
                    logger.info(
                        "Parallel aggregation futures collected: "
                        "{} out of {}. Memory usage is {:.3f} GB out "
                        "of {:.3f} GB ({:.2f}% utilized).".format(
                            n_finished,
                            len(chunks),
                            mem.used / 1e9,
                            mem.total / 1e9,
                            100 * mem.used / mem.total,
                        )
                    )

        return summary

    @staticmethod
    def _convert_bins(bins):
        """Convert a list of floats or ints to a list of two-entry bin bounds.

        Parameters
        ----------
        bins : list | None
            List of floats or ints (bin edges) to convert to list of two-entry
            bin boundaries or list of two-entry bind boundaries in final format

        Returns
        -------
        bins : list
            List of two-entry bin boundaries
        """

        if bins is None:
            return None

        type_check = [isinstance(x, (list, tuple)) for x in bins]

        if all(type_check):
            return bins

        if any(type_check):
            raise TypeError(
                "Resource class bins has inconsistent "
                "entry type: {}".format(bins)
            )

        bbins = []
        for i, b in enumerate(sorted(bins)):
            if i < len(bins) - 1:
                bbins.append([b, bins[i + 1]])

        return bbins

    @staticmethod
    def _summary_to_df(summary):
        """Convert the agg summary list to a DataFrame.

        Parameters
        ----------
        summary : list
            List of dictionaries, each being an SC point summary.

        Returns
        -------
        summary : DataFrame
            Summary of the SC points.
        """
        summary = pd.DataFrame(summary)
        sort_by = [x for x in (SupplyCurveField.SC_POINT_GID, 'res_class')
                   if x in summary]
        summary = summary.sort_values(sort_by)
        summary = summary.reset_index(drop=True)
        summary.index.name = SupplyCurveField.SC_GID

        return summary

    def summarize(
        self, gen_fpath, args=None, max_workers=None, sites_per_worker=100
    ):
        """
        Get the supply curve points aggregation summary

        Parameters
        ----------
        gen_fpath : str
            Filepath to .h5 reV generation output results.
        args : tuple | list | None
            List of summary arguments to include. None defaults to all
            available args defined in the class attr.
        max_workers : int | None, optional
            Number of cores to run summary on. None is all
            available cpus, by default None
        sites_per_worker : int
            Number of sc_points to summarize on each worker, by default 100

        Returns
        -------
        summary : list
            List of dictionaries, each being an SC point summary.
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        if max_workers == 1:
            gen_index = self._parse_gen_index(gen_fpath)
            afk = self._area_filter_kernel
            summary = self.run_serial(
                self._excl_fpath,
                gen_fpath,
                self._tm_dset,
                gen_index,
                econ_fpath=self._econ_fpath,
                excl_dict=self._excl_dict,
                inclusion_mask=self._inclusion_mask,
                res_class_dset=self._res_class_dset,
                res_class_bins=self._res_class_bins,
                cf_dset=self._cf_dset,
                lcoe_dset=self._lcoe_dset,
                h5_dsets=self._h5_dsets,
                data_layers=self._data_layers,
                resolution=self._resolution,
                power_density=self._power_density,
                friction_fpath=self._friction_fpath,
                friction_dset=self._friction_dset,
                area_filter_kernel=afk,
                min_area=self._min_area,
                gids=self.gids,
                args=args,
                excl_area=self._excl_area,
                cap_cost_scale=self._cap_cost_scale,
                recalc_lcoe=self._recalc_lcoe,
                zones_dset=self._zones_dset,
            )
        else:
            summary = self.run_parallel(
                gen_fpath=gen_fpath,
                args=args,
                max_workers=max_workers,
                sites_per_worker=sites_per_worker,
            )

        if not any(summary):
            e = (
                "Supply curve aggregation found no non-excluded SC points. "
                "Please check your exclusions or subset SC GID selection."
            )
            logger.error(e)
            raise EmptySupplyCurvePointError(e)

        summary = self._summary_to_df(summary)

        return summary

    def run(
        self,
        out_fpath,
        gen_fpath=None,
        args=None,
        max_workers=None,
        sites_per_worker=100,
    ):
        """Run a supply curve aggregation.

        Parameters
        ----------
        gen_fpath : str, optional
            Filepath to HDF5 file with ``reV`` generation output
            results. If ``None``, a simple aggregation without any
            generation, resource, or cost data is performed.

            .. Note:: If executing ``reV`` from the command line, this
              input can be set to ``"PIPELINE"`` to parse the output
              from one of these preceding pipeline steps:
              ``multi-year``, ``collect``, or ``econ``. However, note
              that duplicate executions of any of these commands within
              the pipeline may invalidate this parsing, meaning the
              `gen_fpath` input will have to be specified manually.

            By default, ``None``.
        args : tuple | list, optional
            List of columns to include in summary output table. ``None``
            defaults to all available args defined in the
            :class:`~reV.supply_curve.sc_aggregation.SupplyCurveAggregation`
            documentation. By default, ``None``.
        max_workers : int, optional
            Number of cores to run summary on. ``None`` is all available
            CPUs. By default, ``None``.
        sites_per_worker : int, optional
            Number of sc_points to summarize on each worker.
            By default, ``100``.

        Returns
        -------
        str
            Path to output CSV file containing supply curve aggregation.
        """

        if gen_fpath is None:
            out = Aggregation.run(
                self._excl_fpath,
                self._res_fpath,
                self._tm_dset,
                excl_dict=self._excl_dict,
                resolution=self._resolution,
                excl_area=self._excl_area,
                area_filter_kernel=self._area_filter_kernel,
                min_area=self._min_area,
                pre_extract_inclusions=self._pre_extract_inclusions,
                max_workers=max_workers,
                sites_per_worker=sites_per_worker,
            )
            summary = out["meta"]
        else:
            summary = self.summarize(
                gen_fpath=gen_fpath,
                args=args,
                max_workers=max_workers,
                sites_per_worker=sites_per_worker,
            )

        out_fpath = _format_sc_agg_out_fpath(out_fpath)
        summary.to_csv(out_fpath)

        return out_fpath


def _format_sc_agg_out_fpath(out_fpath):
    """Add CSV file ending and replace underscore, if necessary."""
    if not out_fpath.endswith(".csv"):
        out_fpath = "{}.csv".format(out_fpath)

    project_dir, out_fn = os.path.split(out_fpath)
    out_fn = out_fn.replace(
        "supply_curve_aggregation", "supply-curve-aggregation"
    )
    return os.path.join(project_dir, out_fn)


def _warn_about_large_datasets(gen, dset):
    """Warn user about multi-dimensional datasets in passthrough datasets"""
    dset_shape = gen.shapes.get(dset, (1,))
    if len(dset_shape) > 1:
        msg = (
            "Generation dataset {!r} is not 1-dimensional (shape: {})."
            "You may run into memory errors during aggregation - use "
            "rep-profiles for aggregating higher-order datasets instead!"
            .format(dset, dset_shape)
        )
        logger.warning(msg)
        warn(msg, UserWarning)
