# -*- coding: utf-8 -*-
"""reV powercurve losses module.

"""
import logging

import numpy as np
from scipy.optimize import minimize_scalar

from reV.utilities.exceptions import reVLossesValueError

logger = logging.getLogger(__name__)


class PowercurveLosses:
    """A converter between annual losses and powercurve shifts.

    Given a target annual loss value, this class facilitates the
    calculation of a powercurve shift that attempts to match the target
    loss as close as possible.

    Attributes
    ----------
    wind_speed : :obj:`np.array`
        An array containing the wind speeds corresponding to the values
        in the :attr:`powercurve` array.
    powercurve : :obj:`np.array`
        An array containing the generated power at the corresponding
        wind speed in the :attr:`wind_speed` array. These values are
        set during initialization and are treated as the "original
        powercurve". This input must have at least one positive value,
        and if a cutoff speed is detected (see `Warnings` section
        below), then all values above that wind speed must be set to 0.
    wind_resource : :obj:`np.array`
        An array containing the wind speeds (i.e. wind speed
        distribution) for the site at which the powercurve will be used.
        This distribution is used to calculate the annual generation
        of the original powercurve as well as any additional calcaulted
        powercurves. The generation values are then compared in order to
        calculate the loss resulting from a shifted powercurve.

    Warnings
    --------
    This class will attempt to infer a cutoff speed from the powercurve
    input. Specifically, it will look for a transition from the highest
    rated power down to zero in a single ``wind_speed`` step of the
    powercurve. If such a transition is detected, the wind speed
    corresponding to the zero value will be set as the cutoff speed, and
    all calculated powercurves will be clipped at this speed. If your
    input powercurve contains a cutoff speed, ensure that it adheres to
    the expected pattern of dropping from max rated power to zero power
    in a single wind speed step.
    """

    def __init__(self, wind_speed, powercurve, wind_resource):
        """
        Parameters
        ----------
        wind_speed : iter
            An iterable containing the wind speeds corresponding to the
            generated power values in ``powercurve`` input. The input
            values should all be non-zero, and the units should match
            the units of the ``wind_resource`` input (typically, m/s).
        powercurve : iter
            An iterable containing the generated power at the
            corresponding wind speed in the ``wind_speed`` input. These
            values are treated as the "original powercurve".
        wind_resource : iter
            An iterable containing the wind speeds measured at the site
            where this powercurve will be applied to caulcate
            generation. These values are used to calculate the loss
            resulting from a shifted powercurve compared to the
            generation of the original powercurve. The input
            values should all be non-zero, and the units of
            should match the units of the  ``wind_speed`` input
            (typically, m/s).
        """
        self.wind_speed = np.array(wind_speed)
        self.powercurve = np.array(powercurve)
        self.wind_resource = np.array(wind_resource)

        self._power_gen = None
        self._cutoff_wind_speed = None

        self._validate_arrays_not_empty()
        self._validate_wind_speed()
        self._validate_powercurve()
        self._validate_wind_resource()

    def _validate_arrays_not_empty(self):
        """Validate that the input data arrays are not empty. """
        attr_names = ['wind_speed', 'powercurve', 'wind_resource']
        for name in attr_names:
            arr = getattr(self, name)
            if not arr.size:
                msg = "Invalid {} input: Array is empty! - {}"
                msg = msg.format(name.replace('_', ' '), arr)
                logger.error(msg)
                raise reVLossesValueError(msg)

    def _validate_wind_speed(self):
        """Validate that the input wind speed is non-negative. """
        if not (self.wind_speed >= 0).all():
            msg = "Invalid wind speed input: Contains negative values! - {}"
            msg = msg.format(self.wind_speed)
            logger.error(msg)
            raise reVLossesValueError(msg)

    def _validate_powercurve(self):
        """Validate the input powercurve. """
        if not (self.powercurve > 0).any():
            msg = "Invalid powercurve input: Found no positive values! - {}"
            msg = msg.format(self.powercurve)
            logger.error(msg)
            raise reVLossesValueError(msg)

        if 0 < self.cutoff_wind_speed < np.inf:
            cutoff_windspeed_ind = np.where(
                self.wind_speed >= self.cutoff_wind_speed
            )[0].min()
            if (self.powercurve[cutoff_windspeed_ind:]).any():
                msg = ("Invalid powercurve input: Found non-zero values above "
                       "cutoff! - {}")
                msg = msg.format(self.powercurve)
                logger.error(msg)
                raise reVLossesValueError(msg)

    def _validate_wind_resource(self):
        """Validate that the input wind resource is non-negative. """
        if not (self.wind_resource >= 0).all():
            msg = "Invalid wind resource input: Contains negative values! - {}"
            msg = msg.format(self.wind_resource)
            logger.error(msg)
            raise reVLossesValueError(msg)

    def apply_shift(self, shift):
        """Calculate a new powercurve by shifting the original.

        This function shifts the original powercurve horizontally
        by the given amount and truncates any power above the cutoff
        speed (if one was detected).

        Parameters
        ----------
        shift : float
            The amount to shift the original powercurve by, in wind
            speed units (typically, m/s).

        Returns
        -------
        :obj:`np.array`
            An array containing the shifted powercurve.
        """
        new_pc = np.interp(
            self.wind_speed - shift, self.wind_speed, self.powercurve
        )
        if self.cutoff_wind_speed:
            new_pc[self.wind_speed >= self.cutoff_wind_speed] = 0
        return new_pc

    def annual_losses_with_shifted_powercurve(self, shift):
        """Calculate the annual losses from a shifted powercurve.

        This function uses the wind resource data that the object was
        initialized with to calculate the total annual power generation
        with a shifted powercurve. This generation is compared with the
        generation of the original (un-shifted) powercurve to compute
        the total annual losses as a result of the shift.

        Parameters
        ----------
        shift : float
            The amount to shift the original powercurve by, in wind
            speed units (typically, m/s).

        Returns
        -------
        float
            Total losses (%) as a result of a powercurve shifted by the
            input ``shift`` amount.
        """
        new_curve = self.apply_shift(shift)
        power_gen_with_losses = np.interp(
            self.wind_resource, self.wind_speed, new_curve
        ).sum()
        return (1 - power_gen_with_losses / self.power_gen_no_losses) * 100

    def _obj(self, shift, target):
        """Objective function: |output - target|."""
        losses = self.annual_losses_with_shifted_powercurve(shift)
        return np.abs(losses - target)

    def _find_shift(self, target):
        """Run a minimization of the objective function. """
        return minimize_scalar(
            self._obj,
            args=(target),
            bounds=self.shift_bounds,
            method='bounded'
        ).x

    def calculate(self, target):
        """Shift the powercurve to yield annual losses closest to target.

        This function shifts the input powercurve (the one used to
        initialize the object) to generate an annual loss percentage
        closest to the ``target``. The losses are computed w.r.t the
        generation of the original (un-shifted) powercurve.

        Parameters
        ----------
        target : float
            Target value for annual generation losses (%).

        Returns
        -------
        :obj:`np.array`
            An array containing a shifted powercurve that most closely
            yields the ``target`` annual generation losses.

        Warnings
        --------
        This function attempts to find an optimal shift value for the
        powercurve such that the annual generation losses match the
        ``target`` value, but there is no guarantee that a close match
        can be found, if it even exists. Therefore, it is possible that
        the losses resulting from the shifted powercurve will not match
        the ``target``. This is especially likely if the ``target`` is
        large or if the input powercurve and/or wind resource is
        abnormal.
        """
        shift = self._find_shift(target)
        new_curve = self.apply_shift(shift)
        self._validate_shifted_powercurve(new_curve)
        return new_curve

    def _validate_shifted_powercurve(self, new_curve):
        """Ensure new powercurve has some non-zero generation. """
        mask = [self.wind_speed <= self.cutoff_wind_speed]
        min_expected_power_gen = self.powercurve[self.powercurve > 0].min()
        if not (new_curve[mask] > min_expected_power_gen).any():
            msg = ("Calculated powercurve is invalid. No power generation "
                   "below the cutoff wind speed ({} m/s) detected. Target "
                   "loss percentage  may be too large! Please try again with "
                   "a lower target value.")
            msg = msg.format(self.cutoff_wind_speed)
            logger.error(msg)
            raise reVLossesValueError(msg)

    @property
    def cutoff_wind_speed(self):
        """float or :obj:`np.inf`: The detected cutoff wind speed."""
        if self._cutoff_wind_speed is None:
            ind = np.argmax(self.powercurve[::-1])
            # pylint: disable=chained-comparison
            if ind > 0 and self.powercurve[-ind] <= 0:
                self._cutoff_wind_speed = self.wind_speed[-ind]
            else:
                self._cutoff_wind_speed = np.inf
        return self._cutoff_wind_speed

    @property
    def shift_bounds(self):
        """tuple: Bounds on the powercurve shift for the fitting procedure."""
        min_ind = np.where(self.powercurve)[0][0]
        max_ind = np.where(self.powercurve[::-1])[0][0]
        max_shift = self.wind_speed[-max_ind] - self.wind_speed[min_ind]
        return (0, max_shift)

    @property
    def power_gen_no_losses(self):
        """float: Total power generation from original powercurve."""
        if self._power_gen is None:
            self._power_gen = np.interp(
                self.wind_resource, self.wind_speed, self.powercurve
            ).sum()
        return self._power_gen


class PowercurveLossesMixin:
    """Mixin class for :class:`reV.SAM.generation.AbstractSamWind`.

    Warning
    -------
    Using this class for anything excpet as a mixin for
    :class:`~reV.SAM.generation.AbstractSamWind` may result in unexpected
    results and/or errors.
    """

    POWERCURVE_CONFIG_KEY = 'reV-powercurve_losses'
    """Specify powercurve loss target in the config file using this key."""

    def add_powercurve_losses(self):
        """Adjust powercurve in SAM config file to account for losses.

        This function reads the information in the
        ``reV-powercurve_losses`` key of the ``sam_sys_inputs``
        dictionary and computes a new powercurve that accounts for the
        loss percentage specified from that input. If no powercurve loss
        info is specified in ``sam_sys_inputs``, the powercurve will not
        be adjusted.

        See Also
        --------
        :class:`PowercurveLosses` : Powercurve re-calculation.

        """
        target_losses = self.sam_sys_inputs.pop(
            self.POWERCURVE_CONFIG_KEY, None
        )
        if target_losses is None:
            return

        wind_speed = self.sam_sys_inputs['wind_turbine_powercurve_windspeeds']
        powercurve = self.sam_sys_inputs['wind_turbine_powercurve_powerout']
        wind_resource = [d[-2] for d in self['wind_resource_data']['data']]

        pc_losses = PowercurveLosses(wind_speed, powercurve, wind_resource)
        new_curve = pc_losses.calculate(target_losses)
        self.sam_sys_inputs['wind_turbine_powercurve_powerout'] = new_curve
