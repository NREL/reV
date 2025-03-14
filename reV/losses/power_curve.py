# -*- coding: utf-8 -*-
"""reV power curve losses module.

"""
import json
import logging
import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize_scalar

from reV.utilities.exceptions import reVLossesValueError, reVLossesWarning
from reV.losses.utils import _validate_arrays_not_empty

logger = logging.getLogger(__name__)


class PowerCurve:
    """A turbine power curve.

    Attributes
    ----------
    wind_speed : :obj:`numpy.array`
        An array containing the wind speeds corresponding to the values
        in the :attr:`generation` array.
    generation : :obj:`numpy.array`
        An array containing the generated power in kW at the corresponding
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
        wind_speed : array_like
            An iterable containing the wind speeds corresponding to the
            generated power values in ``generation`` input. The input
            values should all be non-zero.
        generation : array_like
            An iterable containing the generated power in kW at the
            corresponding wind speed in the ``wind_speed`` input. This
            input must have at least one positive value, and if a cutoff
            speed is detected (see `Warnings` section below), then all
            values above that wind speed must be set to 0.
        """
        self.wind_speed = np.array(wind_speed)
        self.generation = np.array(generation)
        self._cutoff_wind_speed = None
        self._cutin_wind_speed = None
        self.i_cutoff = None

        _validate_arrays_not_empty(self,
                                   array_names=['wind_speed', 'generation'])
        self._validate_wind_speed()
        self._validate_generation()

    def _validate_wind_speed(self):
        """Validate that the input wind speed is non-negative. """
        if not (self.wind_speed >= 0).all():
            msg = ("Invalid wind speed input: Contains negative values! - {}"
                   .format(self.wind_speed))
            logger.error(msg)
            raise reVLossesValueError(msg)

    def _validate_generation(self):
        """Validate the input generation. """
        if not (self.generation > 0).any():
            msg = ("Invalid generation input: Found no positive values! - {}"
                   .format(self.generation))
            logger.error(msg)
            raise reVLossesValueError(msg)

        if 0 < self.cutoff_wind_speed < np.inf:
            wind_speeds_above_cutoff = np.where(self.wind_speed
                                                >= self.cutoff_wind_speed)
            cutoff_wind_speed_ind = wind_speeds_above_cutoff[0].min()
            if (self.generation[cutoff_wind_speed_ind:]).any():
                msg = ("Invalid generation input: Found non-zero values above "
                       "cutoff! - {}".format(self.generation))
                logger.error(msg)
                raise reVLossesValueError(msg)

    @property
    def cutin_wind_speed(self):
        """The detected cut-in wind speed at which power generation begins

        Returns
        --------
        float
        """
        if self._cutin_wind_speed is None:
            ind = np.where(self.generation > 0)[0][0]
            if ind > 0:
                self._cutin_wind_speed = self.wind_speed[ind - 1]
            else:
                self._cutin_wind_speed = 0
        return self._cutin_wind_speed

    @property
    def cutoff_wind_speed(self):
        """The detected cutoff wind speed at which the power generation is zero

        Returns
        --------
        float | np.inf
        """
        if self._cutoff_wind_speed is None:
            ind = np.argmax(self.generation[::-1])
            # pylint: disable=chained-comparison
            if ind > 0 and self.generation[-ind] <= 0:
                self.i_cutoff = len(self.generation) - ind
                self._cutoff_wind_speed = self.wind_speed[-ind]
            else:
                self._cutoff_wind_speed = np.inf
        return self._cutoff_wind_speed

    @property
    def rated_power(self):
        """Get the rated power (max power) of the turbine power curve. The
        units are dependent on the input power curve but this is typically in
        units of kW.

        Returns
        -------
        float
        """
        return np.max(self.generation)

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

    def __getitem__(self, index):
        return self.generation[index]

    def __call__(self, wind_speed):
        """Calculate the power curve value for the given ``wind_speed``.

        Parameters
        ----------
        wind_speed : int | float | list | array_like
            Wind speed value corresponding to the desired power curve
            value.

        Returns
        -------
        float | :obj:`numpy.array`
            The power curve value(s) for the input wind speed(s).
        """
        if isinstance(wind_speed, (int, float)):
            wind_speed = np.array([wind_speed])
        power_generated = np.interp(wind_speed, self.wind_speed,
                                    self.generation)
        if self.cutoff_wind_speed:
            power_generated[wind_speed >= self.cutoff_wind_speed] = 0
        return power_generated


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
        The original Power Curve.
    wind_resource : :obj:`numpy.array`
        An array containing the wind speeds (i.e. wind speed
        distribution) for the site at which the power curve will be
        used. This distribution is used to calculate the annual
        generation of the original power curve as well as any additional
        calculated power curves. The generation values are then compared
        in order to calculate the loss resulting from a transformed
        power curve.
    weights : :obj:`numpy.array`
        An array of the same length as ``wind_resource`` containing
        weights to apply to each generation value calculated for the
        corresponding wind speed.
    """

    def __init__(self, power_curve, wind_resource, weights=None, site=None):
        """
        Parameters
        ----------
        power_curve : :obj:`PowerCurve`
            The "original" power curve to be adjusted.
        wind_resource : array_like
            An iterable containing the wind speeds measured at the site
            where this power curve will be applied to calculate
            generation. These values are used to calculate the loss
            resulting from a transformed power curve compared to the
            generation of the original power curve. The input
            values should all be non-zero, and the units of
            should match the units of the ``power_curve`` input
            (typically, m/s).
        weights : array_like, optional
            An iterable of the same length as ``wind_resource``
            containing weights to apply to each generation value
            calculated for the corresponding wind speed.
        site : int | str, optional
            Site number (gid) for debugging and logging.
            By default, ``None``.
        """

        self.power_curve = power_curve
        self.wind_resource = np.array(wind_resource)
        if weights is None:
            self.weights = np.ones_like(self.wind_resource)
        else:
            self.weights = np.array(weights)
        self._power_gen = None
        self.site = "[unknown]" if site is None else site

        _validate_arrays_not_empty(self,
                                   array_names=['wind_resource', 'weights'])
        self._validate_wind_resource()
        self._validate_weights()

    def _validate_wind_resource(self):
        """Validate that the input wind resource is non-negative. """
        if not (self.wind_resource >= 0).all():
            msg = ("Invalid wind resource input for site {}: Contains "
                   "negative values! - {}"
                   .format(self.site, self.wind_resource))
            msg = msg.format(self.wind_resource)
            logger.error(msg)
            raise reVLossesValueError(msg)

    def _validate_weights(self):
        """Validate that the input weights size matches the wind resource. """
        if self.wind_resource.size != self.weights.size:
            msg = ("Invalid weights input: Does not match size of wind "
                   "resource for site {}! - {} vs {}"
                   .format(self.site, self.weights.size,
                           self.wind_resource.size))
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
        transformed_power_curve : :obj:`PowerCurve`
            A transformed power curve. The power generated with this
            power curve will be compared with the power generated by the
            "original" power curve to calculate annual losses.

        Returns
        -------
        float
            Total losses (%) as a result of a the power curve
            transformation.
        """
        power_gen_with_losses = transformed_power_curve(self.wind_resource)
        power_gen_with_losses *= self.weights
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
            A PowerCurveTransformation class representing the power
            curve transformation to use.

        Returns
        -------
        :obj:`numpy.array`
            An array containing a transformed power curve that most
            closely yields the ``target`` annual generation losses.

        Warns
        -----
        reVLossesWarning
            If the fit did not meet the target annual losses to within
            1%.

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
        fit_var = minimize_scalar(self._obj,
                                  args=(target, transformation),
                                  bounds=transformation.optm_bounds,
                                  method='bounded').x

        if fit_var > np.max(transformation.bounds):
            msg = ('Transformation "{}" for site {} resulted in fit parameter '
                   '{} greater than the max bound of {}. Limiting to the max '
                   'bound, but the applied losses may not be correct.'
                   .format(transformation, self.site, fit_var,
                           np.max(transformation.bounds)))
            logger.warning(msg)
            warnings.warn(msg, reVLossesWarning)
            fit_var = np.max(transformation.bounds)

        if fit_var < np.min(transformation.bounds):
            msg = ('Transformation "{}" for site {} resulted in fit parameter '
                   '{} less than the min bound of {}. Limiting to the min '
                   'bound, but the applied losses may not be correct.'
                   .format(transformation, self.site, fit_var,
                           np.min(transformation.bounds)))
            logger.warning(msg)
            warnings.warn(msg, reVLossesWarning)
            fit_var = np.min(transformation.bounds)

        error = self._obj(fit_var, target, transformation)

        if error > 1:
            new_power_curve = transformation.apply(fit_var)
            losses = self.annual_losses_with_transformed_power_curve(
                new_power_curve)
            msg = ("Unable to find a transformation for site {} such that the "
                   "losses meet the target within 1%! Obtained fit with loss "
                   "percentage {}% when target was {}%. Consider using a "
                   "different transformation or reducing the target losses!"
                   .format(self.site, losses, target))
            logger.warning(msg)
            warnings.warn(msg, reVLossesWarning)

        return transformation.apply(fit_var)

    @property
    def power_gen_no_losses(self):
        """float: Total power generation from original power curve."""
        if self._power_gen is None:
            self._power_gen = self.power_curve(self.wind_resource)
            self._power_gen *= self.weights
            self._power_gen = self._power_gen.sum()
        return self._power_gen


class PowerCurveLossesInput:
    """Power curve losses specification.

    This class stores and validates information about the desired losses
    from a given type of power curve transformation. In particular, the
    target loss percentage must be provided. This input is then
    validated to be used power curve transformation fitting.

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

                - ``target_losses_percent``
                   An integer or float value representing the
                   total percentage of annual energy production that
                   should be lost due to the power curve transformation.
                   This value must be in the range [0, 100].

            The input dictionary can also provide the following optional
            keys:

                - ``transformation`` - by default, ``horizontal_translation``
                  A string representing the type of transformation to
                  apply to the power curve. This sting must be one of
                  the keys of :obj:`TRANSFORMATIONS`. See the relevant
                  transformation class documentation for detailed
                  information on that type of power curve
                  transformation.


        """
        self._specs = specs
        self._transformation_name = self._specs.get('transformation',
                                                    'exponential_stretching')
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
            msg = ("The following required keys are missing from the power "
                   "curve losses specification: {}"
                   .format(sorted(missing_keys)))
            logger.error(msg)
            raise reVLossesValueError(msg)

    def _validate_transformation(self):
        """Validate that the transformation exists in TRANSFORMATIONS. """
        if self._transformation_name not in TRANSFORMATIONS:
            msg = ("Transformation {!r} not understood! "
                   "Input must be one of: {} "
                   .format(self._transformation_name,
                           list(TRANSFORMATIONS.keys())))
            logger.error(msg)
            raise reVLossesValueError(msg)

    def _validate_percentage(self):
        """Validate that the percentage is in the range [0, 100]. """
        if not 0 <= self.target <= 100:
            msg = ("Percentage of annual energy production loss to be "
                   "attributed to the power curve transformation must be in "
                   "the range [0, 100], but got {} for transformation {!r}"
                   .format(self.target, self._transformation_name))
            logger.error(msg)
            raise reVLossesValueError(msg)

    def __repr__(self):
        specs = self._specs.copy()
        specs.update({'transformation': self._transformation_name})
        specs_as_str = ", ".join(["{}={!r}".format(k, v)
                                  for k, v in specs.items()])
        return "PowerCurveLossesInput({})".format(specs_as_str)

    @property
    def target(self):
        """int or float: Target loss percentage due to transformation."""
        return self._specs['target_losses_percent']

    @property
    def transformation(self):
        """PowerCurveTransformation: Power curve transformation."""
        return TRANSFORMATIONS[self._transformation_name]


class PowerCurveWindResource:
    """Wind resource data for calculating power curve shift."""

    def __init__(self, temperature, pressure, wind_speed):
        """Power Curve Wind Resource.

        Parameters
        ----------
        temperature : array_like
            An iterable representing the temperatures at a single site
            (in C). Must be the same length as the `pressure` and
            `wind_speed` inputs.
        pressure : array_like
            An iterable representing the pressures at a single site
            (in PA or ATM). Must be the same length as the `temperature`
            and `wind_speed` inputs.
        wind_speed : array_like
            An iterable representing the wind speeds at a single site
            (in m/s). Must be the same length as the `temperature` and
            `pressure` inputs.
        """
        self._temperatures = np.array(temperature)
        self._pressures = np.array(pressure)
        self._wind_speeds = np.array(wind_speed)
        self.wind_speed_weights = None

    def wind_resource_for_site(self):
        """Extract scaled wind speeds at the resource site.

        Get the wind speeds for this site, accounting for the scaling
        done in SAM [1]_ based on air pressure [2]_. These wind speeds
        can then be used to sample the power curve and obtain generation
        values.

        Returns
        -------
        array-like
            Array of scaled wind speeds.

        References
        ----------
        .. [1] Scaling done in SAM: https://tinyurl.com/2uzjawpe
        .. [2] SAM Wind Power Reference Manual for explanations on
           generation and air density calculations (pp. 18):
           https://tinyurl.com/2p8fjba6

        """
        if self._pressures.max() < 2:  # units are ATM
            pressures_pascal = self._pressures * 101325.027383
        elif self._pressures.min() > 1e4:  # units are PA
            pressures_pascal = self._pressures
        else:
            msg = ("Unable to determine pressure units: pressure values "
                   "found in the range {:.2f} to {:.2f}. Please make "
                   "sure input pressures are in units of PA or ATM"
                   .format(self._pressures.min(), self._pressures.max()))
            logger.error(msg)
            raise reVLossesValueError(msg)

        temperatures_K = self._temperatures + 273.15  # originally in celsius
        specific_gas_constant_dry_air = 287.058  # units: J / kg / K
        sea_level_air_density = 1.225  # units: kg/m**3 at 15 degrees celsius

        site_air_densities = pressures_pascal / (specific_gas_constant_dry_air
                                                 * temperatures_K)
        weights = (sea_level_air_density / site_air_densities) ** (1 / 3)
        return self._wind_speeds / weights

    @property
    def wind_speeds(self):
        """:obj:`numpy.array`: Array of adjusted wind speeds. """
        return self.wind_resource_for_site()


class _PowerCurveWindDistribution:
    """`PowerCurveWindResource` interface mocker for wind distributions. """

    def __init__(self, speeds, weights):
        """Power Curve Wind Resource for Wind Distributions.

        Parameters
        ----------
        speeds : array_like
            An iterable representing the wind speeds at a single site
            (in m/s). Must be the same length as the `weights` input.
        weights : array_like
            An iterable representing the wind speed weights at a single
            site. Must be the same length as the `speeds` input.
        """
        self.wind_speeds = np.array(speeds)
        self.wind_speed_weights = np.array(weights)


def adjust_power_curve(power_curve, resource_data, target_losses, site=None):
    """Adjust power curve to account for losses.

    This function computes a new power curve that accounts for the
    loss percentage specified from the target loss.

    Parameters
    ----------
    power_curve : :obj:`PowerCurve`
        Power curve to be adjusted to match target losses.
    resource_data : :obj:`PowerCurveWindResource`
        Resource data for the site being investigated.
    target_losses : :obj:`PowerCurveLossesInput`
        Target loss and power curve shift info.
    site : int | str, optional
        Site number (gid) for debugging and logging.
        By default, ``None``.

    Returns
    -------
    :obj:`PowerCurve`
        Power Curve shifted to meet the target losses. Power Curve is
        not adjusted if all wind speeds are above the cutout or below
        the cutin speed.

    See Also
    --------
    :obj:`PowerCurveLosses` : Power curve re-calculation.
    """
    site = "[unknown]" if site is None else site

    if (resource_data.wind_speeds <= power_curve.cutin_wind_speed).all():
        msg = ("All wind speeds for site {} are below the wind speed "
               "cutin ({} m/s). No power curve adjustments made!"
               .format(site, power_curve.cutin_wind_speed))
        logger.warning(msg)
        warnings.warn(msg, reVLossesWarning)
        return power_curve

    if (resource_data.wind_speeds >= power_curve.cutoff_wind_speed).all():
        msg = ("All wind speeds for site {} are above the wind speed "
               "cutoff ({} m/s). No power curve adjustments made!"
               .format(site, power_curve.cutoff_wind_speed))
        logger.warning(msg)
        warnings.warn(msg, reVLossesWarning)
        return power_curve

    pc_losses = PowerCurveLosses(power_curve, resource_data.wind_speeds,
                                 resource_data.wind_speed_weights, site=site)

    logger.debug("Transforming power curve using the {} transformation to "
                 "meet {}% loss target..."
                 .format(target_losses.transformation, target_losses.target))

    new_curve = pc_losses.fit(target_losses.target,
                              target_losses.transformation)
    logger.debug("Transformed power curve: {}".format(list(new_curve)))
    return new_curve


class PowerCurveLossesMixin:
    """Mixin class for :class:`reV.SAM.generation.AbstractSamWind`.

    Warnings
    --------
    Using this class for anything except as a mixin for
    :class:`~reV.SAM.generation.AbstractSamWind` may result in
    unexpected results and/or errors.
    """

    POWER_CURVE_CONFIG_KEY = 'reV_power_curve_losses'
    """Specify power curve loss target in the config file using this key."""

    def add_power_curve_losses(self):
        """Adjust power curve in SAM config file to account for losses.

        This function reads the information in the
        ``reV_power_curve_losses`` key of the ``sam_sys_inputs``
        dictionary and computes a new power curve that accounts for the
        loss percentage specified from that input. If no power curve
        loss info is specified in ``sam_sys_inputs``, the power curve
        will not be adjusted.

        See Also
        --------
        :func:`adjust_power_curve` : Power curve shift calculation.
        """
        loss_input = self._user_power_curve_input()
        if not loss_input:
            return

        resource = self.wind_resource_from_input()
        site = getattr(self, 'site', "[unknown]")
        new_curve = adjust_power_curve(self.input_power_curve, resource,
                                       loss_input, site=site)
        self.sam_sys_inputs['wind_turbine_powercurve_powerout'] = new_curve

    def _user_power_curve_input(self):
        """Get power curve loss info from config. """
        power_curve_losses_info = self.sam_sys_inputs.pop(
            self.POWER_CURVE_CONFIG_KEY, None
        )
        if power_curve_losses_info is None:
            return

        # site-specific info is input as str
        if isinstance(power_curve_losses_info, str):
            power_curve_losses_info = json.loads(power_curve_losses_info)

        logger.info("Applying power curve losses using the following input:"
                    "\n{}".format(power_curve_losses_info))

        loss_input = PowerCurveLossesInput(power_curve_losses_info)
        if loss_input.target <= 0:
            logger.debug("Power curve target loss is 0. Skipping power curve "
                         "transformation.")
            return

        return loss_input

    @property
    def input_power_curve(self):
        """:obj:`PowerCurve`: Original power curve for site. """
        wind_speed = self.sam_sys_inputs['wind_turbine_powercurve_windspeeds']
        generation = self.sam_sys_inputs['wind_turbine_powercurve_powerout']
        return PowerCurve(wind_speed, generation)

    def wind_resource_from_input(self):
        """Collect wind resource and weights from inputs.

        Returns
        -------
        :obj:`PowerCurveWindResource`
            Wind resource used to compute power curve shift.

        Raises
        ------
        reVLossesValueError
            If power curve losses are not compatible with the
            'wind_resource_model_choice'.
        """
        if self['wind_resource_model_choice'] == 0:
            temperatures, pressures, wind_speeds, __, = map(
                np.array, zip(*self['wind_resource_data']['data'])
            )
            return PowerCurveWindResource(temperatures, pressures, wind_speeds)
        elif self['wind_resource_model_choice'] == 2:
            wrd = np.array(self['wind_resource_distribution'])
            return _PowerCurveWindDistribution(wrd[:, 0], wrd[:, -1])
        else:
            msg = ("reV power curve losses cannot be used with "
                   "'wind_resource_model_choice' = {}"
                   .format(self['wind_resource_model_choice']))
            logger.error(msg)
            raise reVLossesValueError(msg)


class AbstractPowerCurveTransformation(ABC):
    """Abstract base class for power curve transformations.

    **This class is not meant to be instantiated**.

    This class provides an interface for power curve transformations,
    which are meant to more realistically represent certain types of
    losses when compared to simple haircut losses (i.e. constant loss
    value applied at all points on the power curve).

    If you would like to implement your own power curve transformation,
    you should subclass this class and implement the :meth:`apply`
    method and the :attr:`bounds` property. See the documentation for
    each of these below for more details.

    Attributes
    ----------
    power_curve : :obj:`PowerCurve`
        The "original" input power curve.
    """

    def __init__(self, power_curve):
        """Abstract Power Curve Transformation class.

        Parameters
        ----------
        power_curve : :obj:`PowerCurve`
            The turbine power curve. This input is treated as the
            "original" power curve.
        """
        self.power_curve = power_curve
        self._transformed_generation = None

    def _validate_non_zero_generation(self, new_curve):
        """Ensure new power curve has some non-zero generation."""
        mask = (self.power_curve.wind_speed
                <= self.power_curve.cutoff_wind_speed)
        min_expected_power_gen = self.power_curve[self.power_curve > 0].min()
        if not (new_curve[mask] > min_expected_power_gen).any():
            msg = ("Calculated power curve is invalid. No power generation "
                   "below the cutoff wind speed ({} m/s) detected. Target "
                   "loss percentage  may be too large! Please try again with "
                   "a lower target value."
                   .format(self.power_curve.cutoff_wind_speed))
            logger.error(msg)
            raise reVLossesValueError(msg)

    def _validate_same_cutoff(self, new_curve):
        """Validate that the new power curve has the same high-wind cutout as
        the original curve."""
        old_cut = self.power_curve.cutoff_wind_speed
        new_cut = new_curve.cutoff_wind_speed
        if old_cut != new_cut:
            msg = ('Original power curve windspeed cutout is {}m/s and new '
                   'curve cutout is {}m/s. Something went wrong!'
                   .format(old_cut, new_cut))
            logger.error(msg)
            raise reVLossesValueError(msg)

    @abstractmethod
    def apply(self, transformation_var):
        """Apply a transformation to the original power curve.

        Parameters
        ----------
        transformation_var : : float
            A single variable controlling the "strength" of the
            transformation. The :obj:`PowerCurveLosses` object will
            run an optimization using this variable to fit the target
            annual losses incurred with the transformed power curve
            compared to the original power curve using the given wind
            resource distribution.

        Returns
        -------
        :obj:`PowerCurve`
            An new power curve containing the generation values from the
            transformed power curve.

        Raises
        ------
        NotImplementedError
            If the transformation implementation did not set the
            ``_transformed_generation`` attribute.

        Notes
        -----
        When implementing a new transformation, override this method and
        set the ``_transformed_generation`` protected attribute to be
        the generation corresponding to the transformed power curve.
        Then, call ``super().apply(transformation_var)`` in order to
        apply cutout speed curtailment and validation for the
        transformed power curve. For example, here is the implementation
        for a transformation that shifts the power curve horizontally::

            self._transformed_generation = self.power_curve(
                self.power_curve.wind_speed - transformation_var
            )
            return super().apply(transformation_var)

        """
        if self._transformed_generation is None:
            msg = ("Transformation implementation for {}.apply did not set "
                   "the `_transformed_generation` attribute."
                   .format(type(self).__name__))
            logger.error(msg)
            raise NotImplementedError(msg)

        if not np.isinf(self.power_curve.cutoff_wind_speed):
            mask = (self.power_curve.wind_speed
                    >= self.power_curve.cutoff_wind_speed)
            self._transformed_generation[mask] = 0
            new_curve = PowerCurve(self.power_curve.wind_speed,
                                   self._transformed_generation)
            self._validate_non_zero_generation(new_curve)

            i_max = np.argmax(self._transformed_generation)
            i_cutoff = self.power_curve.i_cutoff
            rated_power = self.power_curve.rated_power
            self._transformed_generation[i_max:i_cutoff] = rated_power

        new_curve = PowerCurve(self.power_curve.wind_speed,
                               self._transformed_generation)
        self._validate_non_zero_generation(new_curve)
        self._validate_same_cutoff(new_curve)
        return new_curve

    @property
    @abstractmethod
    def bounds(self):
        """tuple: true Bounds on the ``transformation_var``."""

    @property
    def optm_bounds(self):
        """Bounds for scipy optimization, sometimes more generous than the
        actual fit parameter bounds which are enforced after the
        optimization."""
        return self.bounds


class HorizontalTranslation(AbstractPowerCurveTransformation):
    """Utility for applying horizontal power curve translations.

    The mathematical representation of this transformation is:

    .. math:: P_{transformed}(u) = P_{original}(u - t),

    where :math:`P_{transformed}` is the transformed power curve,
    :math:`P_{original}` is the original power curve, :math:`u` is
    the wind speed, and :math:`t` is the transformation variable
    (horizontal translation amount).

    This kind of power curve transformation is simplistic, and should
    only be used for a small handful of applicable turbine losses
    (i.e. blade degradation). See ``Warnings`` for more details.

    The losses in this type of transformation are distributed primarily
    across region 2 of the power curve (the steep, almost linear,
    portion where the generation rapidly increases):

    .. image:: ../../../examples/rev_losses/horizontal_translation.png
       :align: center

    Attributes
    ----------
    power_curve : :obj:`PowerCurve`
        The "original" input power curve.

    Warnings
    --------
    This kind of power curve translation is not generally realistic.
    Using this transformation as a primary source of losses (i.e. many
    different kinds of losses bundled together) is extremely likely to
    yield unrealistic results!
    """

    def apply(self, transformation_var):
        """Apply a horizontal translation to the original power curve.

        This function shifts the original power curve horizontally,
        along the "wind speed" (x) axis, by the given amount. Any power
        above the cutoff speed (if one was detected) is truncated after
        the transformation.

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
        self._transformed_generation = self.power_curve(
            self.power_curve.wind_speed - transformation_var
        )
        return super().apply(transformation_var)

    @property
    def bounds(self):
        """tuple: true Bounds on the power curve shift (different from the
        optimization boundaries)"""
        min_ind = np.where(self.power_curve)[0][0]
        max_ind = np.where(self.power_curve[::-1])[0][0]
        max_shift = (self.power_curve.wind_speed[-max_ind - 1]
                     - self.power_curve.wind_speed[min_ind])
        return (0, max_shift)


class LinearStretching(AbstractPowerCurveTransformation):
    """Utility for applying a linear stretch to the power curve.

    The mathematical representation of this transformation is:

    .. math:: P_{transformed}(u) = P_{original}(u/t),

    where :math:`P_{transformed}` is the transformed power curve,
    :math:`P_{original}` is the original power curve, :math:`u` is
    the wind speed, and :math:`t` is the transformation variable
    (wind speed multiplier).

    The losses in this type of transformation are distributed primarily
    across regions 2 and 3 of the power curve. In particular, losses are
    smaller for wind speeds closer to the cut-in speed, and larger for
    speeds close to rated power:

    .. image:: ../../../examples/rev_losses/linear_stretching.png
       :align: center

    Attributes
    ----------
    power_curve : :obj:`PowerCurve`
        The "original" input power curve.
    """

    def apply(self, transformation_var):
        """Apply a linear stretch to the original power curve.

        This function stretches the original power curve along the
        "wind speed" (x) axis. Any power above the cutoff speed (if one
        was detected) is truncated after the transformation.

        Parameters
        ----------
        transformation_var : float
            The linear multiplier of the wind speed scaling.

        Returns
        -------
        :obj:`PowerCurve`
            An new power curve containing the generation values from the
            shifted power curve.
        """
        self._transformed_generation = self.power_curve(
            self.power_curve.wind_speed / transformation_var
        )
        return super().apply(transformation_var)

    @property
    def bounds(self):
        """tuple: true Bounds on the wind speed multiplier (different from the
        optimization boundaries)"""
        min_ind_pc = np.where(self.power_curve)[0][0]
        min_ind_ws = np.where(self.power_curve.wind_speed > 1)[0][0]
        min_ws = self.power_curve.wind_speed[max(min_ind_pc, min_ind_ws)]
        max_ws = min(self.power_curve.wind_speed.max(),
                     self.power_curve.cutoff_wind_speed)
        max_multiplier = np.ceil(max_ws / min_ws)
        return (1, max_multiplier)


class ExponentialStretching(AbstractPowerCurveTransformation):
    """Utility for applying an exponential stretch to the power curve.

    The mathematical representation of this transformation is:

    .. math:: P_{transformed}(u) = P_{original}(u^{1/t}),

    where :math:`P_{transformed}` is the transformed power curve,
    :math:`P_{original}` is the original power curve, :math:`u` is
    the wind speed, and :math:`t` is the transformation variable
    (wind speed exponent).

    The losses in this type of transformation are distributed primarily
    across regions 2 and 3 of the power curve. In particular, losses are
    smaller for wind speeds closer to the cut-in speed, and larger for
    speeds close to rated power:

    .. image:: ../../../examples/rev_losses/exponential_stretching.png
       :align: center

    Attributes
    ----------
    power_curve : :obj:`PowerCurve`
        The "original" input power curve.
    """

    def apply(self, transformation_var):
        """Apply an exponential stretch to the original power curve.

        This function stretches the original power curve along the
        "wind speed" (x) axis. Any power above the cutoff speed (if one
        was detected) is truncated after the transformation.

        Parameters
        ----------
        transformation_var : float
            The exponent of the wind speed scaling.

        Returns
        -------
        :obj:`PowerCurve`
            An new power curve containing the generation values from the
            shifted power curve.
        """
        self._transformed_generation = self.power_curve(
            self.power_curve.wind_speed ** (1 / transformation_var)
        )
        return super().apply(transformation_var)

    @property
    def bounds(self):
        """tuple: Bounds on the wind speed exponent."""
        min_ind_pc = np.where(self.power_curve)[0][0]
        min_ind_ws = np.where(self.power_curve.wind_speed > 1)[0][0]
        min_ws = self.power_curve.wind_speed[max(min_ind_pc, min_ind_ws)]
        max_ws = min(self.power_curve.wind_speed.max(),
                     self.power_curve.cutoff_wind_speed)
        max_exponent = np.ceil(np.log(max_ws) / np.log(min_ws))
        return (1, max_exponent)

    @property
    def optm_bounds(self):
        """Bounds for scipy optimization, sometimes more generous than the
        actual fit parameter bounds which are enforced after the
        optimization."""
        return (0.5, self.bounds[1])


TRANSFORMATIONS = {
    'horizontal_translation': HorizontalTranslation,
    'linear_stretching': LinearStretching,
    'exponential_stretching': ExponentialStretching
}
"""Implemented power curve transformations."""
