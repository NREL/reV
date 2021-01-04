# -*- coding: utf-8 -*-
"""
reV module for calculating economies of scale where larger power plants will
have reduced capital cost.
"""
import logging
import copy
import re
import pandas as pd

from rex.utilities.utilities import check_eval_str

logger = logging.getLogger(__name__)


class EconomiesOfScale:
    """Class to calculate economies of scale where power plant capital cost is
    reduced for larger power plants."""

    def __init__(self, eqn, data):
        """
        Parameters
        ----------
        eqn : str
            LCOE scaling equation to implement "economies of scale".
            Equation must be in python string format and return a scalar
            value to multiply the capital cost by. Independent variables in
            the equation should match the keys in the data input arg.
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
        check_eval_str(self._eqn)

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
        delimiters = ('*', '/', '+', '-', ' ', '(', ')', '[', ']')
        regex_pattern = '|'.join(map(re.escape, delimiters))
        var_names = [sub for sub in re.split(regex_pattern, str(self._eqn))
                     if sub
                     and not self.is_num(sub)
                     and not self.is_method(sub)]
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
        kwargs = {k: self._data[k] for k in self.vars}
        # pylint: disable=eval-used
        out = eval(str(self._eqn), globals(), kwargs)
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
        return self._data['capital_cost']

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
        cc *= self.capital_cost_scalar()
        return cc

    @property
    def fcr(self):
        """Fixed charge rate from input data arg

        Returns
        -------
        out : float | np.ndarray
            Fixed charge rate from input data arg
        """
        return self._data['fcr']

    @property
    def foc(self):
        """Fixed operating cost from input data arg

        Returns
        -------
        out : float | np.ndarray
            Fixed operating cost from input data arg
        """
        return self._data['foc']

    @property
    def voc(self):
        """Variable operating cost from input data arg

        Returns
        -------
        out : float | np.ndarray
            Variable operating cost from input data arg
        """
        return self._data['voc']

    @property
    def aep(self):
        """Annual energy production from input data arg

        Returns
        -------
        out : float | np.ndarray
            Annual energy production from input data arg
        """
        return self._data['aep']

    @property
    def raw_lcoe(self):
        """LCOE calculated with the unscaled (raw) capital cost

        Returns
        -------
        lcoe : float | np.ndarray
            LCOE calculated with the unscaled (raw) capital cost
        """
        lcoe = ((self.fcr * self.raw_capital_cost + self.foc) / self.aep
                + self.voc)
        return lcoe

    @property
    def scaled_lcoe(self):
        """LCOE calculated with the scaled capital cost based on the
        EconomiesOfScale input equation.

        Returns
        -------
        lcoe : float | np.ndarray
            LCOE calculated with the scaled capital cost based on the
            EconomiesOfScale input equation.
        """
        lcoe = ((self.fcr * self.scaled_capital_cost + self.foc) / self.aep
                + self.voc)
        return lcoe
