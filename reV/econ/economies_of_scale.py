# -*- coding: utf-8 -*-
"""
reV module for calculating economies of scale where larger power plants will
have reduced capital cost.
"""
import logging
import copy
import re
import numpy as np  # pylint: disable=unused-import
import pandas as pd

from reV.econ.utilities import lcoe_fcr
from rex.utilities.utilities import check_eval_str

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

    def __init__(self, eqn, data):
        """
        Parameters
        ----------
        eqn : str
            LCOE scaling equation to implement "economies of scale".
            Equation must be in python string format and return a scalar
            value to multiply the capital cost by. Independent variables in
            the equation should match the keys in the data input arg. This
            equation may use numpy functions with the package prefix "np".
        data : dict | pd.DataFrame
            Namespace of econ data to use to calculate economies of scale. Keys
            in dict or column labels in dataframe should match the Independent
            variables in the eqn input. Should also include variables required
            to calculate LCOE.
        """
        self._eqn = eqn
        self._data = data
        self._preflight()

    def _preflight(self):
        """Run checks to validate EconomiesOfScale equation and input data."""

        if self._eqn is not None:
            check_eval_str(str(self._eqn))

        if isinstance(self._data, pd.DataFrame):
            self._data = {k: self._data[k].values.flatten()
                          for k in self._data.columns}

        if not isinstance(self._data, dict):
            e = ('Cannot evaluate EconomiesOfScale with data input of type: {}'
                 .format(type(self._data)))
            logger.error(e)
            raise TypeError(e)

        missing = []
        for name in self.vars:
            if name not in self._data:
                missing.append(name)

        if any(missing):
            e = ('Cannot evaluate EconomiesOfScale, missing data for variables'
                 ': {} for equation: {}'.format(missing, self._eqn))
            logger.error(e)
            raise KeyError(e)

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
        return bool(s.startswith(('np.', 'pd.')) or s in dir(__builtins__))

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
        var_names = []
        if self._eqn is not None:
            delimiters = ('*', '/', '+', '-', ' ', '(', ')', '[', ']', ',')
            regex_pattern = '|'.join(map(re.escape, delimiters))
            var_names = []
            for sub in re.split(regex_pattern, str(self._eqn)):
                if sub:
                    if not self.is_num(sub) and not self.is_method(sub):
                        var_names.append(sub)
            var_names = sorted(list(set(var_names)))

        return var_names

    def _evaluate(self):
        """Evaluate the EconomiesOfScale equation with Independent variables
        parsed into a kwargs dictionary input.

        Returns
        -------
        out : float | np.ndarray
            Evaluated output of the EconomiesOfScale equation. Should be
            numeric scalars to apply directly to the capital cost.
        """
        out = 1
        if self._eqn is not None:
            kwargs = {k: self._data[k] for k in self.vars}
            # pylint: disable=eval-used
            out = eval(str(self._eqn), globals(), kwargs)

        return out

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
            e = ('Could not find requested key list ({}) in the input '
                 'dictionary keys: {}'
                 .format(key_list, list(input_dict.keys())))
            logger.error(e)
            raise KeyError(e)

        return out

    @property
    def capital_cost_scalar(self):
        """Evaluated output of the EconomiesOfScale equation. Should be
        numeric scalars to apply directly to the capital cost.

        Returns
        -------
        out : float | np.ndarray
            Evaluated output of the EconomiesOfScale equation. Should be
            numeric scalars to apply directly to the capital cost.
        """
        return self._evaluate()

    @property
    def raw_capital_cost(self):
        """Unscaled (raw) capital cost found in the data input arg.

        Returns
        -------
        out : float | np.ndarray
            Unscaled (raw) capital_cost found in the data input arg.
        """
        key_list = ['capital_cost', 'mean_capital_cost']
        return self._get_prioritized_keys(self._data, key_list)

    @property
    def scaled_capital_cost(self):
        """Capital cost found in the data input arg scaled by the evaluated
        EconomiesOfScale input equation.

        Returns
        -------
        out : float | np.ndarray
            Capital cost found in the data input arg scaled by the evaluated
            EconomiesOfScale equation.
        """
        cc = copy.deepcopy(self.raw_capital_cost)
        cc *= self.capital_cost_scalar
        return cc

    @property
    def system_capacity(self):
        """Get the system capacity in kW (SAM input, not the reV supply
        curve capacity).

        Returns
        -------
        out : float | np.ndarray
        """
        key_list = ['system_capacity', 'mean_system_capacity']
        return self._get_prioritized_keys(self._data, key_list)

    @property
    def fcr(self):
        """Fixed charge rate from input data arg

        Returns
        -------
        out : float | np.ndarray
            Fixed charge rate from input data arg
        """
        key_list = ['fixed_charge_rate', 'mean_fixed_charge_rate',
                    'fcr', 'mean_fcr']
        return self._get_prioritized_keys(self._data, key_list)

    @property
    def foc(self):
        """Fixed operating cost from input data arg

        Returns
        -------
        out : float | np.ndarray
            Fixed operating cost from input data arg
        """
        key_list = ['fixed_operating_cost', 'mean_fixed_operating_cost',
                    'foc', 'mean_foc']
        return self._get_prioritized_keys(self._data, key_list)

    @property
    def voc(self):
        """Variable operating cost from input data arg

        Returns
        -------
        out : float | np.ndarray
            Variable operating cost from input data arg
        """
        key_list = ['variable_operating_cost', 'mean_variable_operating_cost',
                    'voc', 'mean_voc']
        return self._get_prioritized_keys(self._data, key_list)

    @property
    def aep(self):
        """Annual energy production back-calculated from the raw LCOE:

        AEP = (fcr * raw_cap_cost + foc) / raw_lcoe

        Returns
        -------
        out : float | np.ndarray
        """

        aep = (self.fcr * self.raw_capital_cost + self.foc) / self.raw_lcoe
        aep *= 1000  # convert MWh to KWh
        return aep

    @property
    def raw_lcoe(self):
        """Raw LCOE taken from the input data

        Returns
        -------
        lcoe : float | np.ndarray
        """
        key_list = ['raw_lcoe', 'mean_lcoe']
        return copy.deepcopy(self._get_prioritized_keys(self._data, key_list))

    @property
    def scaled_lcoe(self):
        """LCOE calculated with the scaled capital cost based on the
        EconomiesOfScale input equation.

        LCOE = (FCR * scaled_capital_cost + FOC) / AEP + VOC

        Returns
        -------
        lcoe : float | np.ndarray
            LCOE calculated with the scaled capital cost based on the
            EconomiesOfScale input equation.
        """
        return lcoe_fcr(self.fcr, self.scaled_capital_cost, self.foc,
                        self.aep, self.voc)
