# -*- coding: utf-8 -*-
"""reV utilities."""
from enum import Enum

import PySAM
from rex.utilities.loggers import log_versions as rex_log_versions

from reV.version import __version__


class FieldEnum(str, Enum):
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
    """

    SC_GID = "sc_gid"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    COUNTRY = "country"
    STATE = "state"
    COUNTY = "county"
    ELEVATION = "elevation_m"
    TIMEZONE = "timezone"
    SC_POINT_GID = "sc_point_gid"
    SC_ROW_IND = "sc_row_ind"
    SC_COL_IND = "sc_col_ind"
    SOURCE_GIDS = "source_gids"
    RES_GIDS = "res_gids"
    GEN_GIDS = "gen_gids"
    GID_COUNTS = "gid_counts"
    N_GIDS = "n_gids"
    ZONE_ID = "zone_id"
    MEAN_RES = "resource"
    MEAN_CF_AC = "capacity_factor_ac"
    MEAN_CF_DC = "capacity_factor_dc"
    MEAN_LCOE = "lcoe_site_usd_per_mwh"
    CAPACITY_AC_MW = "capacity_ac_mw"
    CAPACITY_DC_MW = "capacity_dc_mw"
    OFFSHORE = "offshore"
    AREA_SQ_KM = "area_developable_sq_km"
    MEAN_FRICTION = "friction_site"
    MEAN_LCOE_FRICTION = "lcoe_friction_usd_per_mwh"
    RAW_LCOE = "lcoe_raw_usd_per_mwh"
    EOS_MULT = "multiplier_cc_eos"
    REG_MULT = "multiplier_cc_regional"
    SC_POINT_ANNUAL_ENERGY_MWH = "annual_energy_site_mwh"
    COST_BASE_OCC_USD_PER_AC_MW = "cost_base_occ_usd_per_ac_mw"
    COST_SITE_OCC_USD_PER_AC_MW = "cost_site_occ_usd_per_ac_mw"
    COST_BASE_FOC_USD_PER_AC_MW = "cost_base_foc_usd_per_ac_mw"
    COST_SITE_FOC_USD_PER_AC_MW = "cost_site_foc_usd_per_ac_mw"
    COST_BASE_VOC_USD_PER_AC_MW = "cost_base_voc_usd_per_ac_mw"
    COST_SITE_VOC_USD_PER_AC_MW = "cost_site_voc_usd_per_ac_mw"
    FIXED_CHARGE_RATE = "fixed_charge_rate"

    # Bespoke outputs
    POSSIBLE_X_COORDS = "possible_x_coords"
    POSSIBLE_Y_COORDS = "possible_y_coords"
    TURBINE_X_COORDS = "turbine_x_coords"
    TURBINE_Y_COORDS = "turbine_y_coords"
    N_TURBINES = "n_turbines"
    INCLUDED_AREA = "area_included_sq_km"
    INCLUDED_AREA_CAPACITY_DENSITY = (
        "capacity_density_included_area_mw_per_km2"
    )
    CONVEX_HULL_AREA = "area_convex_hull_sq_km"
    CONVEX_HULL_CAPACITY_DENSITY = "capacity_density_convex_hull_mw_per_km2"
    FULL_CELL_CAPACITY_DENSITY = "capacity_density_full_cell_mw_per_km2"
    BESPOKE_AEP = "optimized_plant_aep"
    BESPOKE_OBJECTIVE = "optimized_plant_objective"
    BESPOKE_CAPITAL_COST = "optimized_plant_capital_cost"
    BESPOKE_FIXED_OPERATING_COST = "optimized_plant_fixed_operating_cost"
    BESPOKE_VARIABLE_OPERATING_COST = "optimized_plant_variable_operating_cost"
    BESPOKE_BALANCE_OF_SYSTEM_COST = "optimized_plant_balance_of_system_cost"

    # Transmission outputs
    TRANS_GID = "trans_gid"
    TRANS_TYPE = "trans_type"
    TOTAL_LCOE_FRICTION = "lcoe_total_friction_usd_per_mwh"
    TRANS_CAPACITY = "trans_capacity"
    DIST_SPUR_KM = "dist_spur_km"
    DIST_EXPORT_KM = "dist_export_km"
    REINFORCEMENT_DIST_KM = "dist_reinforcement_km"
    TIE_LINE_COST_PER_MW = "cost_spur_usd_per_mw"
    CONNECTION_COST_PER_MW = "cost_poi_usd_per_mw"
    EXPORT_COST_PER_MW = "cost_export_usd_per_mw"
    REINFORCEMENT_COST_PER_MW = "cost_reinforcement_usd_per_mw"
    TOTAL_TRANS_CAP_COST_PER_MW = "cost_total_trans_usd_per_mw"
    LCOT = "lcot_usd_per_mwh"
    TOTAL_LCOE = "lcoe_all_in_usd_per_mwh"
    N_PARALLEL_TRANS = "count_num_parallel_trans"
    POI_LAT = "latitude_poi"
    POI_LON = "longitude_poi"
    REINFORCEMENT_POI_LAT = "latitude_reinforcement_poi"
    REINFORCEMENT_POI_LON = "longitude_reinforcement_poi"

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
    TIE_LINE_COST_PER_MW = "tie_line_cost_per_mw"
    CONNECTION_COST_PER_MW = "connection_cost_per_mw"
    REINFORCEMENT_COST_PER_MW = "reinforcement_cost_per_mw"
    TOTAL_TRANS_CAP_COST_PER_MW = "trans_cap_cost_per_mw"
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
