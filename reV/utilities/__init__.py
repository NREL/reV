# -*- coding: utf-8 -*-
"""reV utilities."""

from enum import Enum, EnumMeta

import PySAM
from rex.utilities.loggers import log_versions as rex_log_versions

from reV.version import __version__

OldSupplyCurveField = {
    "sc_point_gid": "SC_POINT_GID",
    "source_gids": "SOURCE_GIDS",
    "sc_gid": "SC_GID",
    "gid_counts": "GID_COUNTS",
    "gid": "GID",
    "n_gids": "N_GIDS",
    "res_gids": "RES_GIDS",
    "gen_gids": "GEN_GIDS",
    "area_sq_km": "AREA_SQ_KM",
    "latitude": "LATITUDE",
    "longitude": "LONGITUDE",
    "elevation": "ELEVATION",
    "timezone": "TIMEZONE",
    "county": "COUNTY",
    "state": "STATE",
    "country": "COUNTRY",
    "mean_lcoe": "MEAN_LCOE",
    "mean_res": "MEAN_RES",
    "capacity": "CAPACITY",
    "sc_row_ind": "SC_ROW_IND",
    "sc_col_ind": "SC_COL_IND",
    "mean_cf": "MEAN_CF",
    "capital_cost": "CAPITAL_COST",
    "mean_lcoe_friction": "MEAN_LCOE_FRICTION",
    "total_lcoe_friction": "TOTAL_LCOE_FRICTION",
}


class FieldEnum(str, Enum):
    """Base Field enum with some mapping methods."""

    @classmethod
    def map_to(cls, other):
        """Return a rename map from this enum to another."""
        return {
            cls[mem]: other[mem]
            for mem in cls.__members__
            if mem in other.__members__
        }

    @classmethod
    def map_from(cls, other):
        """Return a rename map from a dictionary of name / member pairs (e.g.
        'sc_gid': 'SC_GID') to this enum."""
        return {name: cls[mem] for name, mem in other.items()}

    @classmethod
    def map_from_legacy(cls):
        """Return a dictionary -> this enum map using the dictionary of legacy
        names"""
        return cls.map_from(OldSupplyCurveField)

    def __str__(self):
        return self.value

    def __format__(self, format_spec):
        return str.__format__(self.value, format_spec)


class SiteDataField(FieldEnum):
    """An enumerated map to site data column names."""

    GID = "gid"
    CONFIG = "config"


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


# Dictionary of "old" supply curve field names. Used to rename legacy data to
# match current naming conventions


class OriginalSupplyCurveField(FieldEnum):
    """An enumerated map to supply curve summary/meta keys.

    Each output name should match the name of a key in
    meth:`AggregationSupplyCurvePoint.summary` or
    meth:`GenerationSupplyCurvePoint.point_summary` or
    meth:`BespokeSinglePlant.meta`
    """

    SC_POINT_GID = "sc_point_gid"
    SOURCE_GIDS = "source_gids"
    SC_GID = "sc_gid"
    GID_COUNTS = "gid_counts"
    GID = "gid"
    N_GIDS = "n_gids"
    RES_GIDS = "res_gids"
    GEN_GIDS = "gen_gids"
    AREA_SQ_KM = "area_sq_km"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    ELEVATION = "elevation"
    TIMEZONE = "timezone"
    COUNTY = "county"
    STATE = "state"
    COUNTRY = "country"
    MEAN_CF = "mean_cf"
    MEAN_LCOE = "mean_lcoe"
    MEAN_RES = "mean_res"
    CAPACITY = "capacity"
    OFFSHORE = "offshore"
    SC_ROW_IND = "sc_row_ind"
    SC_COL_IND = "sc_col_ind"
    CAPACITY_AC = "capacity_ac"
    CAPITAL_COST = "capital_cost"
    FIXED_OPERATING_COST = "fixed_operating_cost"
    VARIABLE_OPERATING_COST = "variable_operating_cost"
    FIXED_CHARGE_RATE = "fixed_charge_rate"
    SC_POINT_CAPITAL_COST = "sc_point_capital_cost"
    SC_POINT_FIXED_OPERATING_COST = "sc_point_fixed_operating_cost"
    SC_POINT_ANNUAL_ENERGY = "sc_point_annual_energy"
    SC_POINT_ANNUAL_ENERGY_AC = "sc_point_annual_energy_ac"
    MEAN_FRICTION = "mean_friction"
    MEAN_LCOE_FRICTION = "mean_lcoe_friction"
    TOTAL_LCOE_FRICTION = "total_lcoe_friction"
    RAW_LCOE = "raw_lcoe"
    CAPITAL_COST_SCALAR = "capital_cost_scalar"
    SCALED_CAPITAL_COST = "scaled_capital_cost"
    SCALED_SC_POINT_CAPITAL_COST = "scaled_sc_point_capital_cost"
    TURBINE_X_COORDS = "turbine_x_coords"
    TURBINE_Y_COORDS = "turbine_y_coords"
    EOS_MULT = "eos_mult"
    REG_MULT = "reg_mult"


class SupplyCurveField(FieldEnum):
    """An enumerated map to supply curve summary/meta keys.

    Each output name should match the name of a key in
    meth:`AggregationSupplyCurvePoint.summary` or
    meth:`GenerationSupplyCurvePoint.point_summary` or
    meth:`BespokeSinglePlant.meta`
    """

    SC_POINT_GID = "sc_point_gid_m"
    SOURCE_GIDS = "source_gids_m"
    SC_GID = "sc_gid_m"
    GID_COUNTS = "gid_counts_m"
    GID = "gid_m"
    N_GIDS = "n_gids_m"
    RES_GIDS = "res_gids_m"
    GEN_GIDS = "gen_gids_m"
    AREA_SQ_KM = "area_sq_km_m"
    LATITUDE = "latitude_m"
    LONGITUDE = "longitude_m"
    ELEVATION = "elevation_m"
    TIMEZONE = "timezone_m"
    COUNTY = "county_m"
    STATE = "state_m"
    COUNTRY = "country_m"
    MEAN_CF = "mean_cf_m"
    MEAN_LCOE = "mean_lcoe_m"
    MEAN_RES = "mean_res_m"
    CAPACITY = "capacity_m"
    OFFSHORE = "offshore_m"
    SC_ROW_IND = "sc_row_ind_m"
    SC_COL_IND = "sc_col_ind_m"
    CAPACITY_AC = "capacity_ac_m"
    CAPITAL_COST = "capital_cost_m"
    FIXED_OPERATING_COST = "fixed_operating_cost_m"
    VARIABLE_OPERATING_COST = "variable_operating_cost_m"
    FIXED_CHARGE_RATE = "fixed_charge_rate_m"
    SC_POINT_CAPITAL_COST = "sc_point_capital_cost_m"
    SC_POINT_FIXED_OPERATING_COST = "sc_point_fixed_operating_cost_m"
    SC_POINT_ANNUAL_ENERGY = "sc_point_annual_energy_m"
    SC_POINT_ANNUAL_ENERGY_AC = "sc_point_annual_energy_ac_m"
    MEAN_FRICTION = "mean_friction_m"
    MEAN_LCOE_FRICTION = "mean_lcoe_friction_m"
    TOTAL_LCOE_FRICTION = "total_lcoe_friction_m"
    RAW_LCOE = "raw_lcoe_m"
    CAPITAL_COST_SCALAR = "capital_cost_scalar_m"
    SCALED_CAPITAL_COST = "scaled_capital_cost_m"
    SCALED_SC_POINT_CAPITAL_COST = "scaled_sc_point_capital_cost_m"
    TURBINE_X_COORDS = "turbine_x_coords_m"
    TURBINE_Y_COORDS = "turbine_y_coords_m"
    EOS_MULT = "eos_mult_m"
    REG_MULT = "reg_mult_m"


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
