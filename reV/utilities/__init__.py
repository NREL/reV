# -*- coding: utf-8 -*-
"""reV utilities."""
import ast
import inspect
from enum import Enum, EnumMeta

import PySAM
from rex.utilities.loggers import log_versions as rex_log_versions

from reV.version import __version__


class _DocstringEnumMeta(EnumMeta):
    """Metaclass to assign docstrings to Enum members"""

    def __new__(metacls, clsname, bases, clsdict):
        cls = super().__new__(metacls, clsname, bases, clsdict)

        try:
            source = inspect.getsource(cls)
        except TypeError:
            return cls  # source not available (e.g., in interactive shell)

        module = ast.parse(source)

        for node in ast.iter_child_nodes(module):
            if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
                prev = None
                for body_item in node.body:
                    if isinstance(body_item, ast.Assign):
                        target = body_item.targets[0]
                        if isinstance(target, ast.Name):
                            name = target.id
                            prev = body_item
                    elif (isinstance(body_item, ast.Expr)
                          and isinstance(body_item.value, ast.Constant)):
                        if prev:
                            doc = body_item.value.s
                            member = cls.__members__.get(name)
                            if member:
                                member._description = (doc.strip()
                                                       .replace("\n    ", " "))
                            prev = None
        return cls


class DocEnum(Enum, metaclass=_DocstringEnumMeta):
    """Base Enum class with docstring support"""

    @property
    def description(self):
        """Description of enum member pulled from docstring"""
        return getattr(self, '_description', None)


class FieldEnum(str, DocEnum):
    """Base Field enum with some mapping methods."""

    @classmethod
    def map_to(cls, other):
        """Return a rename map from this enum to another.

        Mapping is performed on matching enum names. In other words, if
        both enums have a `ONE` attribute, this will be mapped from one
        enum to another.

        Parameters
        ----------
        other : :class:`Enum`
            ``Enum`` subclass with ``__members__`` attribute.

        Returns
        -------
        dict
            Dictionary mapping matching values from one enum to another.

        Examples
        --------
        >>> class Test1(FieldEnum):
        >>>     ONE = "one_x"
        >>>     TWO = "two"
        >>>
        >>> class Test2(Enum):
        >>>     ONE = "one_y"
        >>>     THREE = "three"
        >>>
        >>> Test1.map_to(Test2)
        {<Test1.ONE: 'one_x'>: <Test2.ONE: 'one_y'>}
        """
        return {
            cls[mem]: other[mem]
            for mem in cls.__members__
            if mem in other.__members__
        }

    @classmethod
    def map_from(cls, other):
        """Map from a dictionary of name / member pairs to this enum.

        Parameters
        ----------
        other : dict
            Dictionary mapping key values (typically old aliases) to
            enum values. For example, ``{'sc_gid': 'SC_GID'}`` would
            return a dictionary that maps ``'sc_gid'`` to the ``SC_GID``
            member of this enum.

        Returns
        -------
        dict
            Mapping of input dictionary keys to member values of this
            enum.

        Examples
        --------
        >>> class Test(FieldEnum):
        >>>     ONE = "one_x"
        >>>     TWO = "two_y"
        >>>
        >>> Test.map_from({1: "ONE", 2: "TWO"})
        {1: <Test.ONE: 'one_x'>, 2: <Test.TWO: 'two_y'>}
        """
        return {name: cls[mem] for name, mem in other.items()}

    def __str__(self):
        return self.value

    def __format__(self, format_spec):
        return str.__format__(self.value, format_spec)


class SiteDataField(FieldEnum):
    """An enumerated map to site data column names."""

    GID = "gid"
    CONFIG = "config"
    CURTAILMENT = "curtailment"


class ResourceMetaField(FieldEnum):
    """An enumerated map to resource meta column names.

    Each output name should match the name of a key the resource file
    meta table.
    """

    GID = "gid"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    ELEVATION = "elevation"
    TIMEZONE = "timezone"
    COUNTY = "county"
    STATE = "state"
    COUNTRY = "country"
    OFFSHORE = "offshore"


class SupplyCurveField(FieldEnum):
    """An enumerated map to supply curve summary/meta keys.

    This is a collection of known supply curve fields that reV outputs
    across aggregation, supply curve, and bespoke outputs.

    Not all of these columns are guaranteed in every supply-curve like
    output (e.g. "convex_hull_area" is a bespoke-only output).

    The docstrings for each field are used as a description when
    exporting metadata information about supply curve columns. See
    TBA for details.
    """

    # ############## #
    # Shared outputs #
    # ############## #

    SC_GID = "sc_gid"
    """Supply curve GID (Specific to this particular supply curve output)"""

    LATITUDE = "latitude"
    """Centroid latitude of the supply curve grid-cell"""

    LONGITUDE = "longitude"
    """Centroid longitude of the supply curve grid-cell"""

    COUNTRY = "country"
    """Country of the supply curve grid-cell"""

    STATE = "state"
    """State of the supply curve grid-cell"""

    COUNTY = "county"
    """County of the supply curve grid-cell"""

    ELEVATION = "elevation_m"
    """Mean elevation of the supply curve grid-cell"""

    TIMEZONE = "timezone"
    """
    Timezone of supply curve grid-cell, expressed as an hourly offset from UTC
    """

    SC_POINT_GID = "sc_point_gid"
    """
    Unique ID that can be used to match supply curve grid-cells across reV
    supply curves at the same resolution
    """

    SC_ROW_IND = "sc_row_ind"
    """Supply curve grid-cell row ID (Invariant across supply curves)"""

    SC_COL_IND = "sc_col_ind"
    """Supply curve grid-cell column ID (Invariant across supply curves)"""

    SOURCE_GIDS = "source_gids"

    RES_GIDS = "res_gids"
    """List of resource GID's mapped to this supply curve grid-cells"""

    GEN_GIDS = "gen_gids"
    """List of generation GID's mapped to this supply curve point"""

    GID_COUNTS = "gid_counts"
    """
    Number of high-resolution cells corresponding to each generation GID
    for this supply curve point
    """

    N_GIDS = "n_gids"
    """
    Total number of not fully excluded pixels associated with the available
    resource/generation gids
    """

    ZONE_ID = "zone_id"
    """Zone ID of the supply curve grid-cell, if applicable. Defaults to 1."""

    MEAN_RES = "resource"
    """
    Mean resource (e.g. wind speed, gha, temperature, etc.) across the supply
    curve grid-cell
    """

    MEAN_CF_AC = "capacity_factor_ac"
    """Mean capacity factor (AC) across supply curve grid-cell"""

    MEAN_CF_DC = "capacity_factor_dc"
    """Mean capacity factor (DC) across supply curve grid-cell"""

    WAKE_LOSSES = "losses_wakes_pct"
    """Mean wake losses across supply curve grid-cell"""

    MEAN_LCOE = "lcoe_site_usd_per_mwh"
    """
    Mean power plant levelized cost of energy across supply curve grid-cell
    """

    CAPACITY_AC_MW = "capacity_ac_mw"
    """
    Capacity of system based on area_sq_km * AC capacity density assumption
    """

    CAPACITY_DC_MW = "capacity_dc_mw"
    """
    Capacity of system based on area_sq_km * DC capacity density assumption
    """

    OFFSHORE = "offshore"
    """
    Flag value indicating if the supply curve grid-cell is offshore (1)
    or not (0)
    """

    AREA_SQ_KM = "area_developable_sq_km"
    """Developable area after spatial exclusions applied"""

    MEAN_FRICTION = "friction_site"

    MEAN_LCOE_FRICTION = "lcoe_friction_usd_per_mwh"

    RAW_LCOE = "lcoe_raw_usd_per_mwh"
    """
    Mean power plant levelized cost of energy across supply curve grid-cell
    without any multipliers or economies of scale applied
    """

    EOS_MULT = "multiplier_cc_eos"
    """
    Capital cost economies of Scale (EOS) multiplier value (defaults to `1`
    if no EOS curve was specified)
    """

    FIXED_EOS_MULT = "multiplier_foc_eos"
    """
    Fixed operating cost economies of Scale (EOS) multiplier value (defaults
    to `1` if no EOS curve was specified)
    """

    VAR_EOS_MULT = "multiplier_voc_eos"
    """
    Variable operating cost economies of Scale (EOS) multiplier value
    (defaults to `1` if no EOS curve was specified)
    """

    REG_MULT = "multiplier_cc_regional"
    """
    Regional capital cost multiplier to capture taxes, labor, land lease
    regional differences
    """

    SC_POINT_ANNUAL_ENERGY_MWH = "annual_energy_site_mwh"
    """
    Total annual energy for supply curve grid-cell (computed using
    "capacity_ac_mw" and "capacity_factor_ac")
    """

    COST_BASE_CC_USD_PER_AC_MW = "cost_base_cc_usd_per_ac_mw"
    """
    Included-area weighted capital cost for supply curve grid-cell with no
    multipliers or economies of scale applied (defaults to `None` for
    non-LCOE runs)
    """

    COST_SITE_CC_USD_PER_AC_MW = "cost_site_cc_usd_per_ac_mw"
    """
    Included-area weighted capital cost for supply curve grid-cell
    (defaults to `None` for non-LCOE runs)
    """

    COST_BASE_FOC_USD_PER_AC_MW = "cost_base_foc_usd_per_ac_mw"
    """
    Included-area weighted fixed operating cost for supply curve grid-cell
    with no multipliers or economies of scale applied (defaults to `None` for
    non-LCOE runs)
    """

    COST_SITE_FOC_USD_PER_AC_MW = "cost_site_foc_usd_per_ac_mw"
    """
    Included-area weighted fixed operating cost for supply curve grid-cell
    (defaults to `None` for non-LCOE runs)
    """

    COST_BASE_VOC_USD_PER_AC_MWH = "cost_base_voc_usd_per_ac_mwh"
    """
    Included-area weighted variable operating cost for supply curve grid-cell
    with no multipliers or economies of scale applied (defaults to `None` for
    non-LCOE runs)
    """

    COST_SITE_VOC_USD_PER_AC_MWH = "cost_site_voc_usd_per_ac_mwh"
    """
    Included-area weighted variable operating cost for supply curve grid-cell
    (defaults to `None` for non-LCOE runs)
    """

    FIXED_CHARGE_RATE = "fixed_charge_rate"
    """
    Fixed charge rate used for LCOE computation
    (defaults to `None` for non-LCOE runs)
    """

    # ############### #
    # Bespoke outputs #
    # ############### #

    POSSIBLE_X_COORDS = "possible_x_coords"
    """
    List of turbine x coordinates considered during layout optimization
    (in meters relative to grid-cell)
    """

    POSSIBLE_Y_COORDS = "possible_y_coords"
    """
    List of turbine y coordinates considered during layout optimization
    (in meters relative to grid-cell)
    """

    TURBINE_X_COORDS = "turbine_x_coords"
    """
    List of optimized layout turbine x coordinates
    (in meters relative to grid-cell)
    """

    TURBINE_Y_COORDS = "turbine_y_coords"
    """
    List of optimized layout turbine y coordinates
    (in meters relative to grid-cell)
    """

    N_TURBINES = "n_turbines"
    """
    Number of turbines in the optimized layout for this supply curve
    grid-cell
    """

    INCLUDED_AREA = "area_included_sq_km"
    """Area available for wind turbine layout optimization"""

    INCLUDED_AREA_CAPACITY_DENSITY = (
        "capacity_density_included_area_mw_per_km2"
    )
    """
    Capacity density of the optimized wind plant layout defined using the
    area available after removing the exclusions
    """

    CONVEX_HULL_AREA = "area_convex_hull_sq_km"
    """Area of the convex hull of the optimized wind plant layout"""

    CONVEX_HULL_CAPACITY_DENSITY = "capacity_density_convex_hull_mw_per_km2"
    """
    Capacity density of the optimized wind plant layout defined using the
    convex hull area of the layout
    """

    FULL_CELL_CAPACITY_DENSITY = "capacity_density_full_cell_mw_per_km2"
    """
    Capacity density of the optimized wind plant layout defined using the full
    non-excluded area of the supply curve grid-cell
    """

    BESPOKE_AEP = "optimized_plant_aep"
    """
    Annual energy production of the optimized wind plant layout computed using
    wind speed/direction joint probability distribution (as opposed to
    historical weather data)
    """

    BESPOKE_OBJECTIVE = "optimized_plant_objective"
    """
    Objective function value of the optimized wind plant layout. This is
    typically the LCOE computed using wind speed/direction joint probability
    distribution (as opposed to historical weather data)
    """

    BESPOKE_CAPITAL_COST = "optimized_plant_capital_cost"
    """Capital cost of the optimized wind plant layout"""

    BESPOKE_FIXED_OPERATING_COST = "optimized_plant_fixed_operating_cost"
    """Annual fixed operating cost of the optimized wind plant layout"""

    BESPOKE_VARIABLE_OPERATING_COST = "optimized_plant_variable_operating_cost"
    """Variable operating cost of the optimized wind plant layout"""

    BESPOKE_BALANCE_OF_SYSTEM_COST = "optimized_plant_balance_of_system_cost"
    """Balance of system cost of the optimized wind plant layout"""

    # #################### #
    # Transmission outputs #
    # #################### #

    TRANS_GID = "trans_gid"
    """Transmission connection feature GID"""

    TRANS_TYPE = "trans_type"
    """Transmission connection feature type"""

    TOTAL_LCOE_FRICTION = "lcoe_total_friction_usd_per_mwh"
    TRANS_CAPACITY = "trans_capacity"

    DIST_SPUR_KM = "dist_spur_km"
    """
    Distance between the grid-cell centroid and cheapest available electrical
    substation. Used in lcot calculations.
    """

    DIST_EXPORT_KM = "dist_export_km"
    """Length of the offshore export cable"""

    REINFORCEMENT_DIST_KM = "dist_reinforcement_km"
    """
    Distance between the connected substation and nearest regional load
    center. Used in lcot calculations.
    """

    TIE_LINE_COST_PER_MW = "cost_spur_usd_per_mw_ac"
    """
    Cost of the spur line used to connect the grid-cell centroid with the
    cheapest available electrical substation
    """

    CONNECTION_COST_PER_MW = "cost_poi_usd_per_mw_ac"
    """Substation connection/upgrade/installation cost"""

    EXPORT_COST_PER_MW = "cost_export_usd_per_mw_ac"
    """Cost of the offshore export cable """

    REINFORCEMENT_COST_PER_MW = "cost_reinforcement_usd_per_mw_ac"
    """Non-levelized reinforcement transmission capital costs"""

    TOTAL_TRANS_CAP_COST_PER_MW = "cost_total_trans_usd_per_mw_ac"
    """
    Non-levelized spur and point-of-interconnection transmission capital costs
    """

    LCOT = "lcot_usd_per_mwh"
    """
    Levelized cost of transmission. Includes spur-transmission,
    point-of-interconnection, and reinforcement costs.
    """

    TOTAL_LCOE = "lcoe_all_in_usd_per_mwh"
    """All-in LCOE. Includes site-lcoe + lcot"""

    N_PARALLEL_TRANS = "count_num_parallel_trans"
    """
    Number of parallel transmission lines connecting the grid-cell centroid
    with the cheapest available electrical substation
    """

    POI_LAT = "latitude_poi"
    """
    Latitude of the cheapest available electrical substation for the supply
    curve grid-cell
    """

    POI_LON = "longitude_poi"
    """
    Longitude of the cheapest available electrical substation for the supply
    curve grid-cell
    """

    REINFORCEMENT_POI_LAT = "latitude_reinforcement_poi"
    """
    Latitude of the nearest regional load center for the supply curve
    grid-cell
    """

    REINFORCEMENT_POI_LON = "longitude_reinforcement_poi"
    """
    Longitude of the nearest regional load center for the supply curve
    grid-cell
    """

    @classmethod
    def map_from_legacy(cls):
        """Map of legacy names to current values.

        Returns
        -------
        dict
            Dictionary that maps legacy supply curve column names to
            members of this enum.
        """
        legacy_map = {}
        for current_field, old_field in cls.map_to(_LegacySCAliases).items():
            aliases = old_field.value
            if isinstance(aliases, str):
                aliases = [aliases]
            legacy_map.update({alias: current_field for alias in aliases})

        return legacy_map

    @property
    def units(self):
        """Units of the supply curve column, or ``"N/A"`` if not applicable"""
        return _SC_UNITS.get(self, "N/A")


class _LegacySCAliases(Enum):
    """Legacy supply curve column names.

    Enum values can be either a single string or an iterable of string
    values where each string value represents a previously known alias.
    """

    ELEVATION = "elevation"
    MEAN_RES = "mean_res"
    MEAN_CF_AC = "mean_cf"
    MEAN_LCOE = "mean_lcoe"
    CAPACITY_AC_MW = "capacity"
    AREA_SQ_KM = "area_sq_km"
    MEAN_FRICTION = "mean_friction"
    MEAN_LCOE_FRICTION = "mean_lcoe_friction"
    RAW_LCOE = "raw_lcoe"
    TRANS_TYPE = "category"
    TRANS_CAPACITY = "avail_cap"
    DIST_SPUR_KM = "dist_km"
    REINFORCEMENT_DIST_KM = "reinforcement_dist_km"
    TIE_LINE_COST_PER_MW = "tie_line_cost_per_mw", "cost_spur_usd_per_mw"
    CONNECTION_COST_PER_MW = "connection_cost_per_mw", "cost_poi_usd_per_mw"
    EXPORT_COST_PER_MW = "cost_export_usd_per_mw"
    REINFORCEMENT_COST_PER_MW = ("reinforcement_cost_per_mw",
                                 "cost_reinforcement_usd_per_mw")
    TOTAL_TRANS_CAP_COST_PER_MW = ("trans_cap_cost_per_mw",
                                   "cost_total_trans_usd_per_mw")
    LCOT = "lcot"
    TOTAL_LCOE = "total_lcoe"
    TOTAL_LCOE_FRICTION = "total_lcoe_friction"
    N_PARALLEL_TRANS = "n_parallel_trans"
    EOS_MULT = "eos_mult", "capital_cost_multiplier"
    REG_MULT = "reg_mult"
    SC_POINT_ANNUAL_ENERGY_MWH = "sc_point_annual_energy"
    POI_LAT = "poi_lat"
    POI_LON = "poi_lon"
    REINFORCEMENT_POI_LAT = "reinforcement_poi_lat"
    REINFORCEMENT_POI_LON = "reinforcement_poi_lon"
    BESPOKE_AEP = "bespoke_aep"
    BESPOKE_OBJECTIVE = "bespoke_objective"
    BESPOKE_CAPITAL_COST = "bespoke_capital_cost"
    BESPOKE_FIXED_OPERATING_COST = "bespoke_fixed_operating_cost"
    BESPOKE_VARIABLE_OPERATING_COST = "bespoke_variable_operating_cost"
    BESPOKE_BALANCE_OF_SYSTEM_COST = "bespoke_balance_of_system_cost"
    INCLUDED_AREA = "included_area"
    INCLUDED_AREA_CAPACITY_DENSITY = "included_area_capacity_density"
    CONVEX_HULL_AREA = "convex_hull_area"
    CONVEX_HULL_CAPACITY_DENSITY = "convex_hull_capacity_density"
    FULL_CELL_CAPACITY_DENSITY = "full_cell_capacity_density"
    COST_BASE_CC_USD_PER_AC_MW = "cost_base_occ_usd_per_ac_mw"
    COST_SITE_CC_USD_PER_AC_MW = "cost_site_occ_usd_per_ac_mw"


class ModuleName(str, Enum):
    """A collection of the module names available in reV.

    Each module name should match the name of the click command
    that will be used to invoke its respective cli. As of 3/1/2022,
    this means that all commands are lowercase with underscores
    replaced by dashes.

    Reference
    ---------
    See this line in the click source code to get the most up-to-date
    click name conversions: https://tinyurl.com/4rehbsvf
    """

    BESPOKE = "bespoke"
    COLLECT = "collect"
    ECON = "econ"
    GENERATION = "generation"
    HYBRIDS = "hybrids"
    MULTI_YEAR = "multi-year"
    NRWAL = "nrwal"
    QA_QC = "qa-qc"
    REP_PROFILES = "rep-profiles"
    SUPPLY_CURVE = "supply-curve"
    SUPPLY_CURVE_AGGREGATION = "supply-curve-aggregation"
    TECH_MAPPING = "tech-mapping"

    def __str__(self):
        return self.value

    def __format__(self, format_spec):
        return str.__format__(self.value, format_spec)

    @classmethod
    def all_names(cls):
        """All module names.

        Returns
        -------
        set
            The set of all module name strings.
        """
        # pylint: disable=no-member
        return {v.value for v in cls.__members__.values()}


def log_versions(logger):
    """Log package versions:
    - rex and reV to info
    - h5py, numpy, pandas, scipy, and PySAM to debug

    Parameters
    ----------
    logger : logging.Logger
        Logger object to log memory message to.
    """
    logger.info("Running with reV version {}".format(__version__))
    rex_log_versions(logger)
    logger.debug("- PySAM version {}".format(PySAM.__version__))


_SC_UNITS = {
    SupplyCurveField.ELEVATION: "m",
    SupplyCurveField.LATITUDE: "degrees",
    SupplyCurveField.LONGITUDE: "degrees",

    SupplyCurveField.AREA_SQ_KM: "km2",
    SupplyCurveField.CAPACITY_AC_MW: "MWac",
    SupplyCurveField.CAPACITY_DC_MW: "MWdc",
    SupplyCurveField.MEAN_CF_AC: "ratio",
    SupplyCurveField.MEAN_CF_DC: "ratio",
    SupplyCurveField.WAKE_LOSSES: "%",
    SupplyCurveField.MEAN_LCOE: "$/MWh",
    SupplyCurveField.RAW_LCOE: "$/MWh",
    SupplyCurveField.SC_POINT_ANNUAL_ENERGY_MWH: "MWh",
    SupplyCurveField.COST_BASE_CC_USD_PER_AC_MW: "$/MWac",
    SupplyCurveField.COST_SITE_CC_USD_PER_AC_MW: "$/MWac",
    SupplyCurveField.COST_BASE_FOC_USD_PER_AC_MW: "$/MWac",
    SupplyCurveField.COST_SITE_FOC_USD_PER_AC_MW: "$/MWac",
    SupplyCurveField.COST_BASE_VOC_USD_PER_AC_MWH: "$/MWh",
    SupplyCurveField.COST_SITE_VOC_USD_PER_AC_MWH: "$/MWh",

    SupplyCurveField.BESPOKE_AEP: "MWh",
    SupplyCurveField.BESPOKE_CAPITAL_COST: "$",
    SupplyCurveField.BESPOKE_FIXED_OPERATING_COST: "$/year",
    SupplyCurveField.BESPOKE_VARIABLE_OPERATING_COST: "$/kWh",
    SupplyCurveField.BESPOKE_BALANCE_OF_SYSTEM_COST: "$",
    SupplyCurveField.INCLUDED_AREA: "km2",
    SupplyCurveField.INCLUDED_AREA_CAPACITY_DENSITY: "MW/km2",
    SupplyCurveField.CONVEX_HULL_AREA: "km2",
    SupplyCurveField.CONVEX_HULL_CAPACITY_DENSITY: "MW/km2",
    SupplyCurveField.FULL_CELL_CAPACITY_DENSITY: "MW/km2",

    SupplyCurveField.LCOT: "$/MWh",
    SupplyCurveField.MEAN_RES: "varies",
    SupplyCurveField.REINFORCEMENT_COST_PER_MW: "$/MWac",
    SupplyCurveField.REINFORCEMENT_DIST_KM: "km",
    SupplyCurveField.TOTAL_LCOE: "$/MWh",
    SupplyCurveField.TOTAL_TRANS_CAP_COST_PER_MW: "$/MWac",
    SupplyCurveField.DIST_SPUR_KM: "km",
    SupplyCurveField.DIST_EXPORT_KM: "km",
    SupplyCurveField.TIE_LINE_COST_PER_MW: "$/MWac",
    SupplyCurveField.CONNECTION_COST_PER_MW: "$/MWac",
    SupplyCurveField.EXPORT_COST_PER_MW: "$/MWac",

    SupplyCurveField.POI_LAT: "degrees",
    SupplyCurveField.POI_LON: "degrees",
    SupplyCurveField.REINFORCEMENT_POI_LAT: "degrees",
    SupplyCurveField.REINFORCEMENT_POI_LON: "degrees",

}
