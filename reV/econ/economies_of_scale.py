# -*- coding: utf-8 -*-
"""
reV module for calculating economies of scale where larger power plants will
have reduced capital cost.
"""

import copy
import logging
import re

# pylint: disable=unused-import
import numpy as np
import pandas as pd
from rex.utilities.utilities import check_eval_str

from reV.econ.utilities import lcoe_fcr
from reV.utilities import SupplyCurveField

logger = logging.getLogger(__name__)


class EconomiesOfScale:
    """Class to calculate economies of scale where power plant capital cost is
    reduced for larger power plants.

    Units
    -----
    capacity_factor : unitless
    capacity : kW
    annual_energy_production : kWh
    fixed_charge_rate : unitless
    fixed_operating_cost : $ (per year)
    variable_operating_cost : $/kWh
    lcoe : $/MWh
    """

    def __init__(self, data, cap_eqn=None, fixed_eqn=None, var_eqn=None):
        """

        Parameters
        ----------
        data : dict | pd.DataFrame
            Namespace of econ data to use to calculate economies of scale. Keys
            in dict or column labels in dataframe should match the Independent
            variables in the eqn input. Should also include variables required
            to calculate LCOE.
        cap_eqn : str, optional
            LCOE scaling equation to implement "economies of scale".
            Equation must be in python string format and return a scalar
            value to multiply the capital cost by. Independent variables in
            the equation should match the keys in the data input arg. This
            equation may use numpy functions with the package prefix "np". If
            ``None``, no economies of scale are applied to the capital cost.
            By default, ``None``.
        fixed_eqn : str, optional
            LCOE scaling equation to implement "economies of scale".
            Equation must be in python string format and return a scalar
            value to multiply the fixed operating cost by. Independent
            variables in the equation should match the keys in the data input
            arg. This equation may use numpy functions with the package prefix
            "np". If ``None``, no economies of scale are applied to the
            fixed operating cost. By default, ``None``.
        var_eqn : str, optional
            LCOE scaling equation to implement "economies of scale".
            Equation must be in python string format and return a scalar
            value to multiply the variable operating cost by. Independent
            variables in the equation should match the keys in the data input
            arg. This equation may use numpy functions with the package prefix
            "np". If ``None``, no economies of scale are applied to the
            variable operating cost. By default, ``None``.
        """
        self._data = data
        self._cap_eqn = cap_eqn
        self._fixed_eqn = fixed_eqn
        self._var_eqn = var_eqn
        self._vars = None
        self._preflight()

    def _preflight(self):
        """Run checks to validate EconomiesOfScale equation and input data."""

        for eq in self._all_equations:
            if eq is not None:
                check_eval_str(str(eq))

        if isinstance(self._data, pd.DataFrame):
            self._data = {
                k: self._data[k].values.flatten() for k in self._data.columns
            }

        if not isinstance(self._data, dict):
            e = (
                "Cannot evaluate EconomiesOfScale with data input of type: "
                "{}".format(type(self._data))
            )

            logger.error(e)
            raise TypeError(e)

        missing = [name for name in self.vars if name not in self._data]

        if any(missing):
            e = (
                "Cannot evaluate EconomiesOfScale, missing data for variables"
                ": {} for equation: {}".format(missing, self._cap_eqn)
            )
            logger.error(e)
            raise KeyError(e)

    @property
    def _all_equations(self):
        """gen: All EOS equations"""
        yield from (self._cap_eqn, self._fixed_eqn, self._var_eqn)

    @staticmethod
    def is_num(s):
        """Check if a string is a number"""
        try:
            float(s)
        except ValueError:
            return False
        else:
            return True

    @staticmethod
    def is_method(s):
        """Check if a string is a numpy/pandas or python builtin method"""
        return bool(s.startswith(("np.", "pd.")) or s in dir(__builtins__))

    @property
    def vars(self):
        """Get a list of variable names that the EconomiesOfScale equation
        uses as input.

        Returns
        -------
        vars : list
            List of strings representing variable names that were parsed from
            the equation string. This will return an empty list if the equation
            has no variables.
        """
        if self._vars is not None:
            return self._vars

        self._vars = []
        for eq in self._all_equations:
            if eq is None:
                continue

            delimiters = (">", "<", ">=", "<=", "==", ",", "*", "/", "+",
                          "-", " ", "(", ")", "[", "]")
            regex_pattern = "|".join(map(re.escape, delimiters))
            for sub_str in re.split(regex_pattern, str(eq)):
                is_valid_var_name = (sub_str and not self.is_num(sub_str)
                                     and not self.is_method(sub_str))
                if is_valid_var_name:
                    self._vars.append(sub_str)

        self._vars = sorted(set(self._vars))
        return self._vars

    def _evaluate(self, eqn):
        """Evaluate the EconomiesOfScale equation with Independent variables
        parsed into a kwargs dictionary input.

        Parameters
        ----------
        eqn : str
            LCOE scaling equation to implement "economies of scale".
            Equation must be in python string format and return a scalar
            multiplier. Independent variables in the equation should match the
            keys in the data input arg. This equation may use numpy functions
            with the package prefix "np". If ``None``, this function returns
            ``1``.

        Returns
        -------
        out : float | np.ndarray
            Evaluated output of the EconomiesOfScale equation.
        """
        if eqn is None:
            return 1

        kwargs = {k: self._data[k] for k in self.vars}
        # pylint: disable=eval-used
        return eval(str(eqn), globals(), kwargs)

    @staticmethod
    def _get_prioritized_keys(input_dict, key_list):
        """Get data from an input dictionary based on an ordered (prioritized)
        list of retrieval keys. If no keys are found in the input_dict, an
        error will be raised.

        Parameters
        ----------
        input_dict : dict
            Dictionary of data
        key_list : list | tuple
            Ordered (prioritized) list of retrieval keys.

        Returns
        -------
        out : object
            Data retrieved from input_dict using the first key in key_list
            found in the input_dict.
        """

        out = None
        for key in key_list:
            if key in input_dict:
                out = input_dict[key]
                break

        if out is None:
            e = (
                "Could not find requested key list ({}) in the input "
                "dictionary keys: {}".format(key_list, list(input_dict.keys()))
            )
            logger.error(e)
            raise KeyError(e)

        return out

    @property
    def capital_cost_scalar(self):
        """Evaluated output of the EconomiesOfScale capital cost equation.
        Should be numeric scalars to apply directly to the capital cost.

        Returns
        -------
        out : float | np.ndarray
            Evaluated output of the EconomiesOfScale equation. Should be
            numeric scalars to apply directly to the capital cost.
        """
        return self._evaluate(self._cap_eqn)

    @property
    def fixed_operating_cost_scalar(self):
        """Evaluated output of the EconomiesOfScale fixed operating cost
        equation. Should be numeric scalars to apply directly to the fixed
        operating cost.

        Returns
        -------
        out : float | np.ndarray
            Evaluated output of the EconomiesOfScale equation. Should be
            numeric scalars to apply directly to the fixed operating cost.
        """
        return self._evaluate(self._fixed_eqn)

    @property
    def variable_operating_cost_scalar(self):
        """Evaluated output of the EconomiesOfScale equation variable
        operating cost. Should be numeric scalars to apply directly to the
        variable operating cost.

        Returns
        -------
        out : float | np.ndarray
            Evaluated output of the EconomiesOfScale equation. Should be
            numeric scalars to apply directly to the variable operating cost.
        """
        return self._evaluate(self._var_eqn)

    def _cost_from_cap(self, col_name):
        """Get full cost value from cost per mw in data.

        Parameters
        ----------
        col_name : str
            Name of column containing the cost per mw value.

        Returns
        -------
        float | None
            Cost value if it was found in data, ``None`` otherwise.
        """
        cap = self._data.get(SupplyCurveField.CAPACITY_AC_MW)
        if cap is None:
            return None

        cost_per_mw = self._data.get(col_name)
        if cost_per_mw is None:
            return None

        return cap * cost_per_mw

    @property
    def raw_capital_cost(self):
        """Unscaled (raw) capital cost found in the data input arg.

        Returns
        -------
        out : float | np.ndarray
            Unscaled (raw) capital_cost ($) found in the data input arg.
        """
        raw_capital_cost_from_cap = self._cost_from_cap(
            SupplyCurveField.COST_SITE_CC_USD_PER_AC_MW
        )
        if raw_capital_cost_from_cap is not None:
            return raw_capital_cost_from_cap

        key_list = ["capital_cost", "mean_capital_cost"]
        return self._get_prioritized_keys(self._data, key_list)

    @property
    def scaled_capital_cost(self):
        """Capital cost found in the data input arg scaled by the evaluated
        EconomiesOfScale input equation.

        Returns
        -------
        out : float | np.ndarray
            Capital cost ($) found in the data input arg scaled by the
            evaluated EconomiesOfScale equation.
        """
        cc = copy.deepcopy(self.raw_capital_cost)
        cc *= self.capital_cost_scalar
        return cc

    @property
    def fcr(self):
        """Fixed charge rate from input data arg

        Returns
        -------
        out : float | np.ndarray
            Fixed charge rate from input data arg
        """
        fcr = self._data.get(SupplyCurveField.FIXED_CHARGE_RATE)
        if fcr is not None and fcr > 0:
            return fcr

        key_list = ["fixed_charge_rate", "mean_fixed_charge_rate", "fcr",
                    "mean_fcr"]
        return self._get_prioritized_keys(self._data, key_list)

    @property
    def raw_fixed_operating_cost(self):
        """Unscaled (raw) fixed operating cost from input data arg

        Returns
        -------
        out : float | np.ndarray
            Unscaled (raw) fixed operating cost ($/year) from input data arg
        """
        foc_from_cap = self._cost_from_cap(
            SupplyCurveField.COST_SITE_FOC_USD_PER_AC_MW
        )
        if foc_from_cap is not None:
            return foc_from_cap

        key_list = ["fixed_operating_cost", "mean_fixed_operating_cost",
                    "foc", "mean_foc"]
        return self._get_prioritized_keys(self._data, key_list)

    @property
    def scaled_fixed_operating_cost(self):
        """Fixed operating cost found in the data input arg scaled by the
        evaluated EconomiesOfScale input equation.

        Returns
        -------
        out : float | np.ndarray
            Fixed operating cost ($/year) found in the data input arg scaled
            by the evaluated EconomiesOfScale equation.
        """
        foc = copy.deepcopy(self.raw_fixed_operating_cost)
        foc *= self.fixed_operating_cost_scalar
        return foc

    @property
    def raw_variable_operating_cost(self):
        """Unscaled (raw) variable operating cost from input data arg

        Returns
        -------
        out : float | np.ndarray
            Unscaled (raw) variable operating cost ($/kWh) from input
            data arg
        """
        voc_mwh = self._data.get(SupplyCurveField.COST_SITE_VOC_USD_PER_AC_MWH)
        if voc_mwh is not None:
            return voc_mwh / 1000  # convert to $/kWh

        key_list = ["variable_operating_cost", "mean_variable_operating_cost",
                    "voc", "mean_voc"]
        return self._get_prioritized_keys(self._data, key_list)

    @property
    def scaled_variable_operating_cost(self):
        """Variable operating cost found in the data input arg scaled by the
        evaluated EconomiesOfScale input equation.

        Returns
        -------
        out : float | np.ndarray
            Variable operating cost ($/kWh) found in the data input arg
            scaled by the evaluated EconomiesOfScale equation.
        """
        voc = copy.deepcopy(self.raw_variable_operating_cost)
        voc *= self.variable_operating_cost_scalar
        return voc

    @property
    def aep(self):
        """Annual energy production (kWh) back-calculated from the raw LCOE:

        AEP = (fcr * raw_cap_cost + raw_foc) / (raw_lcoe - raw_voc)

        Returns
        -------
        out : float | np.ndarray
        """
        num = self.fcr * self.raw_capital_cost + self.raw_fixed_operating_cost
        denom = self.raw_lcoe - (self.raw_variable_operating_cost * 1000)
        return num / denom * 1000  # convert MWh to KWh

    @property
    def raw_lcoe(self):
        """Raw LCOE ($/MWh) taken from the input data

        Returns
        -------
        lcoe : float | np.ndarray
        """
        key_list = [SupplyCurveField.RAW_LCOE, SupplyCurveField.MEAN_LCOE]
        return copy.deepcopy(self._get_prioritized_keys(self._data, key_list))

    @property
    def scaled_lcoe(self):
        """LCOE ($/MWh) calculated with the scaled costs based on the
        EconomiesOfScale input equation.

        LCOE = (FCR * scaled_capital_cost + scaled_FOC) / AEP + scaled_VOC

        Returns
        -------
        lcoe : float | np.ndarray
            LCOE calculated with the scaled costs based on the
            EconomiesOfScale input equation.
        """
        return lcoe_fcr(self.fcr, self.scaled_capital_cost,
                        self.scaled_fixed_operating_cost, self.aep,
                        self.scaled_variable_operating_cost)
