# -*- coding: utf-8 -*-
"""
reV Econ utilities
"""


def lcoe_fcr(fixed_charge_rate, capital_cost, fixed_operating_cost,
             annual_energy_production, variable_operating_cost):
    """Calculate the Levelized Cost of Electricity (LCOE) using the
    fixed-charge-rate method:

    LCOE = ((fixed_charge_rate * capital_cost + fixed_operating_cost)
            / annual_energy_production + variable_operating_cost)

    Parameters
    ----------
    fixed_charge_rate : float | np.ndarray
        Fixed charge rage (unitless)
    capital_cost : float | np.ndarray
        Capital cost (aka Capital Expenditures) ($)
    fixed_operating_cost : float | np.ndarray
        Fixed annual operating cost ($/year)
    annual_energy_production : float | np.ndarray
        Annual energy production (kWh/year)
    variable_operating_cost : float | np.ndarray
        Variable operating cost ($/kWh)

    Returns
    -------
    lcoe : float | np.ndarray
        LCOE in $/MWh
    """
    lcoe = ((fixed_charge_rate * capital_cost + fixed_operating_cost)
            / annual_energy_production + variable_operating_cost)
    lcoe *= 1000  # convert $/kWh to $/MWh
    return lcoe
