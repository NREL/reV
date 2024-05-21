# -*- coding: utf-8 -*-
"""reV utilities."""
from enum import Enum

import PySAM
from rex.utilities.loggers import log_versions as rex_log_versions

from reV.version import __version__


class SiteDataField(str, Enum):
    """An enumerated map to site data column names."""
    GID = "gid"
    CONFIG = "config"


class ResourceMetaField(str, Enum):
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
    OFFSHORE = 'offshore'


class MetaKeyName(str, Enum):
    """An enumerated map to summary/meta keys.

    Each output name should match the name of a key in
    meth:`AggregationSupplyCurvePoint.summary` or
    meth:`GenerationSupplyCurvePoint.point_summary` or
    meth:`BespokeSinglePlant.meta`
    """
    SC_POINT_GID = 'sc_point_gid'
    SOURCE_GIDS = 'source_gids'
    SC_GID = 'sc_gid'
    GID_COUNTS = 'gid_counts'
    GID = 'gid'
    N_GIDS = 'n_gids'
    RES_GIDS = 'res_gids'
    GEN_GIDS = 'gen_gids'
    AREA_SQ_KM = 'area_sq_km'
    LATITUDE = 'latitude'
    LONGITUDE = 'longitude'
    ELEVATION = 'elevation'
    TIMEZONE = 'timezone'
    COUNTY = "county"
    STATE = "state"
    COUNTRY = "country"
    MEAN_CF = 'mean_cf'
    MEAN_LCOE = 'mean_lcoe'
    MEAN_RES = 'mean_res'
    CAPACITY = 'capacity'
    OFFSHORE = 'offshore'
    SC_ROW_IND = 'sc_row_ind'
    SC_COL_IND = 'sc_col_ind'
    CAPACITY_AC = 'capacity_ac'
    CAPITAL_COST = 'capital_cost'
    FIXED_OPERATING_COST = 'fixed_operating_cost'
    FIXED_CHARGE_RATE = 'fixed_charge_rate'
    SC_POINT_CAPITAL_COST = 'sc_point_capital_cost'
    SC_POINT_FIXED_OPERATING_COST = 'sc_point_fixed_operating_cost'
    SC_POINT_ANNUAL_ENERGY = 'sc_point_annual_energy'
    SC_POINT_ANNUAL_ENERGY_AC = 'sc_point_annual_energy_ac'
    MEAN_FRICTION = 'mean_friction'
    MEAN_LCOE_FRICTION = 'mean_lcoe_friction'
    TOTAL_LCOE_FRICTION = 'total_lcoe_friction'
    RAW_LCOE = 'raw_lcoe'
    CAPITAL_COST_SCALAR = 'capital_cost_scalar'
    SCALED_CAPITAL_COST = 'scaled_capital_cost'
    SCALED_SC_POINT_CAPITAL_COST = 'scaled_sc_point_capital_cost'
    TURBINE_X_COORDS = 'turbine_x_coords'
    TURBINE_Y_COORDS = 'turbine_y_coords'
    EOS_MULT = "eos_mult"
    REG_MULT = "reg_mult"

    def __str__(self):
        return self.value

    def __format__(self, format_spec):
        return str.__format__(self.value, format_spec)


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

    BESPOKE = 'bespoke'
    COLLECT = 'collect'
    ECON = 'econ'
    GENERATION = 'generation'
    HYBRIDS = 'hybrids'
    MULTI_YEAR = 'multi-year'
    NRWAL = 'nrwal'
    QA_QC = 'qa-qc'
    REP_PROFILES = 'rep-profiles'
    SUPPLY_CURVE = 'supply-curve'
    SUPPLY_CURVE_AGGREGATION = 'supply-curve-aggregation'

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
    logger.info('Running with reV version {}'.format(__version__))
    rex_log_versions(logger)
    logger.debug('- PySAM version {}'.format(PySAM.__version__))
