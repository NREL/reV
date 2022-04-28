# -*- coding: utf-8 -*-
"""reV power curve losses module.

"""
import json
import logging
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize_scalar

from reV.utilities.exceptions import reVLossesValueError
from reV.losses.utils import _validate_arrays_not_empty

logger = logging.getLogger(__name__)


class PowerCurve:
    """A turbine power curve.

    Attributes
    ----------
    wind_speed : :obj:`np.array`
        An array containing the wind speeds corresponding to the values
        in the :attr:`power_curve` array.
    generation : :obj:`np.array`
        An array containing the generated power at the corresponding
        wind speed in the :attr:`wind_speed` array. This input must have
        at least one positive value, and if a cutoff speed is detected
        (see `Warnings` section below), then all values above that wind
        speed must be set to 0.

    Warnings
    --------
    This class will attempt to infer a cutoff speed from the
    ``generation`` input. Specifically, it will look for a transition
    from the highest rated power down to zero in a single ``wind_speed``
    step of the power curve. If such a transition is detected, the wind
    speed corresponding to the zero value will be set as the cutoff
    speed, and all calculated power curves will be clipped at this
    speed. If your input power curve contains a cutoff speed, ensure
    that it adheres to the expected pattern of dropping from max rated
    power to zero power in a single wind speed step.
    """
    def __init__(self, wind_speed, generation):
        """
        Parameters
        ----------
        wind_speed : iter
            An iterable containing the wind speeds corresponding to the
            generated power values in ``generation`` input. The input
            values should all be non-zero.
        generation : iter
            An iterable containing the generated power at the
            corresponding wind speed in the ``wind_speed`` input. This
            input must have at least one positive value, and if a cutoff
            speed is detected (see `Warnings` section below), then all
            values above that wind speed must be set to 0.
        """
        self.wind_speed = np.array(wind_speed)
        self.generation = np.array(generation)
        self._cutoff_wind_speed = None

        _validate_arrays_not_empty(
            self, array_names=['wind_speed', 'generation']
        )
        self._validate_wind_speed()
        self._validate_generation()

    def _validate_wind_speed(self):
        """Validate that the input wind speed is non-negative. """
        if not (self.wind_speed >= 0).all():
            msg = "Invalid wind speed input: Contains negative values! - {}"
            msg = msg.format(self.wind_speed)
            logger.error(msg)
            raise reVLossesValueError(msg)

    def _validate_generation(self):
        """Validate the input generation. """
        if not (self.generation > 0).any():
            msg = "Invalid generation input: Found no positive values! - {}"
            msg = msg.format(self.generation)
            logger.error(msg)
            raise reVLossesValueError(msg)

        if 0 < self.cutoff_wind_speed < np.inf:
            cutoff_windspeed_ind = np.where(
                self.wind_speed >= self.cutoff_wind_speed
            )[0].min()
            if (self.generation[cutoff_windspeed_ind:]).any():
                msg = ("Invalid generation input: Found non-zero values above "
                       "cutoff! - {}")
                msg = msg.format(self.generation)
                logger.error(msg)
                raise reVLossesValueError(msg)

    @property
    def cutoff_wind_speed(self):
        """float or :obj:`np.inf`: The detected cutoff wind speed."""
        if self._cutoff_wind_speed is None:
            ind = np.argmax(self.generation[::-1])
            # pylint: disable=chained-comparison
            if ind > 0 and self.generation[-ind] <= 0:
                self._cutoff_wind_speed = self.wind_speed[-ind]
            else:
                self._cutoff_wind_speed = np.inf
        return self._cutoff_wind_speed

    def __eq__(self, other):
        return np.isclose(self.generation, other).all()

    def __ne__(self, other):
        return not np.isclose(self.generation, other).all()

    def __lt__(self, other):
        return self.generation < other

    def __le__(self, other):
        return self.generation <= other

    def __gt__(self, other):
        return self.generation > other

    def __ge__(self, other):
        return self.generation >= other

    def __len__(self):
        return len(self.generation)

    def __getitem__(self, key):
        return self.generation[key]

    def __call__(self, wind_speed):
        """Calculate the power curve value for the given ``wind_speed``.

        Parameters
        ----------
        wind_speed : :obj:`int` | :obj:`float` | :obj:`list` | :obj:`np.array`
            Wind speed value corresponding to the desired powerrcurve
            value.

        Returns
        -------
        float | :obj:`np.array`
            The power curve value(s) for the input wind speed(s).
        """
        if isinstance(wind_speed, (int, float)):
            wind_speed = [wind_speed]
        new_pc = np.interp(wind_speed, self.wind_speed, self.generation)
        if self.cutoff_wind_speed:
            new_pc[wind_speed >= self.cutoff_wind_speed] = 0
        return new_pc


class PowerCurveLosses:
    """A converter between annual losses and power curve transformation.

    Given a target annual loss value, this class facilitates the
    calculation of a power curve transformation such that the annual
    generation losses incurred by using the transformed power curve when
    compared to the original (non-transformed) power curve match the
    target loss as close as possible.

    The underlying assumption for this approach is that some types of
    losses can be realized by a transformation of the power curve (see
    the values of :obj:`TRANSFORMATIONS` for details on all of the
    power curve transformations that have been implemented).

    The advantage of this approach is that, unlike haircut losses (where
    a single loss value is applied across the board to all generation),
    the losses are distributed non-uniformly across the power curve. For
    example, even in the overly simplified case of a horizontal
    translation of the power curve (which is only physically realistic
    for certain types of losses like blade degradation), the losses are
    distributed primarily across region 2 of the power curve (the steep,
    almost linear, portion where the generation rapidly increases). This
    means that, unlike with haircut losses, generation is able to reach
    max rated power (albeit at a greater wind speed).

    Attributes
    ----------
    power_curve : :obj:`PowerCurve`
        A :obj:`PowerCurve` object representing the "original
        powerCurve".
    wind_resource : :obj:`np.array`
        An array containing the wind speeds (i.e. wind speed
        distribution) for the site at which the power curve will be
        used. This distribution is used to calculate the annual
        generation of the original power curve as well as any additional
        calcaulted power curves. The generation values are then compared
        in order to calculate the loss resulting from a transformed
        power curve.
    """

    def __init__(self, power_curve, wind_resource):
        """
        Parameters
        ----------
        power_curve : PowerCurve
            A :obj:`PowerCurve` object representing the turbine
            power curve. This input is treated as the
            "original" power curve.
        wind_resource : iter
            An iterable containing the wind speeds measured at the site
            where this power curve will be applied to caulcate
            generation. These values are used to calculate the loss
            resulting from a transformed power curve compared to the
            generation of the original power curve. The input
            values should all be non-zero, and the units of
            should match the units of the ``power_curve`` input
            (typically, m/s).
        """
        self.power_curve = power_curve
        self.wind_resource = np.array(wind_resource)
        self._power_gen = None

        _validate_arrays_not_empty(self, array_names=['wind_resource'])
        self._validate_wind_resource()

    def _validate_wind_resource(self):
        """Validate that the input wind resource is non-negative. """
        if not (self.wind_resource >= 0).all():
            msg = "Invalid wind resource input: Contains negative values! - {}"
            msg = msg.format(self.wind_resource)
            logger.error(msg)
            raise reVLossesValueError(msg)

    def annual_losses_with_transformed_power_curve(
        self, transformed_power_curve
    ):
        """Calculate the annual losses from a transformed power curve.

        This function uses the wind resource data that the object was
        initialized with to calculate the total annual power generation
        with a transformed power curve. This generation is compared with
        the generation of the original (non-transformed) power curve to
        compute the total annual losses as a result of the
        transformation.

        Parameters
        ----------
        transformed_power_curve : PowerCurve
            A :obj:`PowerCurve` object representing the transformed
            power curve. The power generated with this power curve will
            be compared with the power generated by the "original" power
            curve to calculate annual losses.

        Returns
        -------
        float
            Total losses (%) as a result of a the power curve
            transformation.
        """
        power_gen_with_losses = transformed_power_curve(self.wind_resource)
        power_gen_with_losses = power_gen_with_losses.sum()
        return (1 - power_gen_with_losses / self.power_gen_no_losses) * 100

    def _obj(self, transformation_variable, target, transformation):
        """Objective function: |output - target|."""
        new_power_curve = transformation.apply(transformation_variable)
        losses = self.annual_losses_with_transformed_power_curve(
            new_power_curve
        )
        return np.abs(losses - target)

    def fit(self, target, transformation):
        """Fit a power curve transformation.

        This function fits a transformation to the input power curve
        (the one used to initialize the object) to generate an annual
        loss percentage closest to the ``target``. The losses are
        computed w.r.t the generation of the original (non-transformed)
        power curve.

        Parameters
        ----------
        target : float
            Target value for annual generation losses (%).
        transformation : PowerCurveTransformation
            A :class:`PowerCurveTransformation` class representing the
            power curve transformation to use.

        Returns
        -------
        :obj:`np.array`
            An array containing a transformed power curve that most
            closely yields the ``target`` annual generation losses.

        Warnings
        --------
        This function attempts to find an optimal transformation for the
        power curve such that the annual generation losses match the
        ``target`` value, but there is no guarantee that a close match
        can be found, if it even exists. Therefore, it is possible that
        the losses resulting from the transformed power curve will not
        match the ``target``. This is especially likely if the
        ``target`` is large or if the input power curve and/or wind
        resource is abnormal.
        """
        transformation = transformation(self.power_curve)
        fit_var = minimize_scalar(
            self._obj,
            args=(target, transformation),
            bounds=transformation.bounds,
            method='bounded'
        ).x
        return transformation.apply(fit_var)

    @property
    def power_gen_no_losses(self):
        """float: Total power generation from original power curve."""
        if self._power_gen is None:
            self._power_gen = self.power_curve(self.wind_resource).sum()
        return self._power_gen


class PowerCurveLossesInput:
    """Power curve losses specification.

    This class stores and validates information about the desired losses
    from a given type of power curve transfromation. In particular, the
    target loss percentage must be provided. This input is then
    validated to be used powercurve transformation fitting.

    """

    REQUIRED_KEYS = {'target_losses_percent'}
    """Required keys in the input specification dictionary."""

    def __init__(self, specs):
        """

        Parameters
        ----------
        specs : dict
            A dictionary containing specifications for the power curve
            losses. This dictionary must contain the following keys:
                - `target_losses_percent`
                    An integer or float value representing the
                    total percentage of annual energy production that
                    should be lost due to the powercurve transformation.
                    This value must be in the range [0, 100].
            The input dictionary can also provide the following optional
            keys:
                - `transformation` - by default, ``horizontal_translation``
                    A string representing the type of transformation to
                    apply to the power curve. This sting must be one of
                    the keys of :obj:`TRANSFORMATIONS`.

        """
        self._specs = specs
        self._transformation_name = self._specs.get(
            'transformation', 'horizontal_translation'
        )
        self._validate()

    def _validate(self):
        """Validate the input specs."""
        self._validate_required_keys_exist()
        self._validate_transformation()
        self._validate_percentage()

    def _validate_required_keys_exist(self):
        """Raise an error if any required keys are missing."""
        missing_keys = [n not in self._specs for n in self.REQUIRED_KEYS]
        if any(missing_keys):
            msg = (
                "The following required keys are missing from the power curve "
                "losses specification: {}".format(sorted(missing_keys))
            )
            logger.error(msg)
            raise reVLossesValueError(msg)

    def _validate_transformation(self):
        """Validate that the transformation exists in TRANSFORMATIONS. """
        if self._transformation_name not in TRANSFORMATIONS:
            msg = (
                "Transformation {!r} not understood! Input must be one of: {} "
            ).format(self._transformation_name, list(TRANSFORMATIONS.keys()))
            logger.error(msg)
            raise reVLossesValueError(msg)

    def _validate_percentage(self):
        """Validate that the percentage is in the range [0, 100]. """
        if not 0 <= self.target <= 100:
            msg = (
                "Percentage of annual energy production loss to be attributed "
                "to the powercurve transformation must be in the range "
                "[0, 100], but got {} for transformation {!r}"
            ).format(self.target, self._transformation_name)
            logger.error(msg)
            raise reVLossesValueError(msg)

    def __repr__(self):
        specs = self._specs.copy()
        specs.update({'transformation': self._transformation_name})
        specs_as_str = ", ".join(
            ["{}={!r}".format(k, v) for k, v in specs.items()]
        )
        return "PowerCurveLossesInput({})".format(specs_as_str)

    @property
    def target(self):
        """int or float: Target loss percentage due to transformation."""
        return self._specs['target_losses_percent']

    @property
    def transformation(self):
        """PowerCurveTransformation: Requested power curve transformation."""
        return TRANSFORMATIONS[self._transformation_name]


class PowerCurveLossesMixin:
    """Mixin class for :class:`reV.SAM.generation.AbstractSamWind`.

    Warning
    -------
    Using this class for anything excpet as a mixin for
    :class:`~reV.SAM.generation.AbstractSamWind` may result in unexpected
    results and/or errors.
    """

    POWERCURVE_CONFIG_KEY = 'reV-power_curve_losses'
    """Specify power curve loss target in the config file using this key."""

    def add_power_curve_losses(self):
        """Adjust power curve in SAM config file to account for losses.

        This function reads the information in the
        ``reV-power_curve_losses`` key of the ``sam_sys_inputs``
        dictionary and computes a new power curve that accounts for the
        loss percentage specified from that input. If no power curve
        loss info is specified in ``sam_sys_inputs``, the power curve
        will not be adjusted.

        See Also
        --------
        :class:`PowerCurveLosses` : Power curve re-calculation.

        """
        power_curve_losses_info = self.sam_sys_inputs.pop(
            self.POWERCURVE_CONFIG_KEY, None
        )
        if power_curve_losses_info is None:
            return

        if isinstance(power_curve_losses_info, str):
            power_curve_losses_info = json.loads(power_curve_losses_info)

        loss_input = PowerCurveLossesInput(power_curve_losses_info)
        if loss_input.target <= 0:
            return

        wind_speed = self.sam_sys_inputs['wind_turbine_powercurve_windspeeds']
        generation = self.sam_sys_inputs['wind_turbine_powercurve_powerout']
        power_curve = PowerCurve(wind_speed, generation)

        wind_resource = [d[-2] for d in self['wind_resource_data']['data']]
        pc_losses = PowerCurveLosses(power_curve, wind_resource)

        new_curve = pc_losses.fit(loss_input.target, loss_input.transformation)
        self.sam_sys_inputs['wind_turbine_powercurve_powerout'] = new_curve


class PowerCurveTransformation(ABC):
    """Abscrtact base class for power curve transformations.

    **This class is not meant to be instantiated**.

    This class provides an interface for power curve transformations,
    which are meant to more realistically represent certain types of
    losses when compared to simple haircut losses (i.e. constant loss
    value applied at all points on the power curve).

    Attributes
    ----------
    power_curve : :obj:`PowerCurve`
        A :obj:`PowerCurve` object representing the "original" power
        curve.
    """
    def __init__(self, power_curve):
        """
        Parameters
        ----------
        power_curve : PowerCurve
            A :obj:`PowerCurve` object representing the turbine
            power curve. This input is treated as the "original" power
            curve.
        """
        self.power_curve = power_curve

    @abstractmethod
    def apply(self, transformation_var):
        """Apply a transformation to the original power curve."""

    @property
    @abstractmethod
    def bounds(self):
        """tuple: Bounds on the transformation_var."""


class HorizontalPowerCurveTranslation(PowerCurveTransformation):
    """Utility for applying horizontal power curve translations.

    This kind of power curve transformation is simplistic, and should
    only be used for a small handful of applicable turbine losses
    (i.e. blade degradation). See ``Warnings`` for more details.

    The losses in this type of transformation are distributed primarily
    across region 2 of the power curve (the steep, almost linear,
    portion where the generation rapidly increases).

    Attributes
    ----------
    power_curve : :obj:`PowerCurve`
        A :obj:`PowerCurve` object representing the "original" power
        curve.

    Warnings
    --------
    This kind of power curve translation is not genrally realistic.
    Using this transformation as a primary source of losses (i.e. many
    different kinds of losses bundled together) is extremely likely to
    yield unrealistic results!
    """

    def apply(self, transformation_var):
        """Apply a horizontal translation to the original power curve.

        This function shifts the original power curve horizontally
        by the given amount and truncates any power above the cutoff
        speed (if one was detected).

        Parameters
        ----------
        transformation_var : float
            The amount to shift the original power curve by, in wind
            speed units (typically, m/s).

        Returns
        -------
        :obj:`PowerCurve`
            An new power curve containing the generation values from the
            shifted power curve.
        """
        new_gen = self.power_curve(
            self.power_curve.wind_speed - transformation_var
        )
        mask = (
            self.power_curve.wind_speed >= self.power_curve.cutoff_wind_speed
        )
        new_gen[mask] = 0

        new_curve = PowerCurve(self.power_curve.wind_speed, new_gen)
        self._validate_shifted_power_curve(new_curve)
        return new_curve

    def _validate_shifted_power_curve(self, new_curve):
        """Ensure new power curve has some non-zero generation. """
        mask = [
            self.power_curve.wind_speed <= self.power_curve.cutoff_wind_speed
        ]
        min_expected_power_gen = self.power_curve[self.power_curve > 0].min()
        if not (new_curve[mask] > min_expected_power_gen).any():
            msg = ("Calculated power curve is invalid. No power generation "
                   "below the cutoff wind speed ({} m/s) detected. Target "
                   "loss percentage  may be too large! Please try again with "
                   "a lower target value.")
            msg = msg.format(self.power_curve.cutoff_wind_speed)
            logger.error(msg)
            raise reVLossesValueError(msg)

    @property
    def bounds(self):
        """tuple: Bounds on the power curve shift for the fitting procedure."""
        min_ind = np.where(self.power_curve)[0][0]
        max_ind = np.where(self.power_curve[::-1])[0][0]
        max_shift = (
            self.power_curve.wind_speed[-max_ind]
            - self.power_curve.wind_speed[min_ind]
        )
        return (0, max_shift)


TRANSFORMATIONS = {
    'horizontal_translation': HorizontalPowerCurveTranslation
}
"""Implemented power curve transformations."""
