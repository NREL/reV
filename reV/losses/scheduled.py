# -*- coding: utf-8 -*-
"""reV scheduled losses module.

"""
import logging
import warnings
import json

import numpy as np

from reV.losses.utils import (convert_to_full_month_names,
                              filter_unknown_month_names,
                              hourly_indices_for_months)
from reV.utilities.exceptions import reVLossesValueError, reVLossesWarning


logger = logging.getLogger(__name__)


class Outage:
    """A specific type of outage.

    This class stores and validates information about a single type of
    outage. In particular, the number of outages, duration, percentage
    of farm down, and the allowed months for scheduling the outage
    must all be provided. These inputs are then validated so that they
    can be used in instances of scheduling objects.

    """

    REQUIRED_KEYS = {'count',
                     'duration',
                     'percentage_of_farm_down',
                     'allowed_months'}
    """Required keys in the input specification dictionary."""

    def __init__(self, specs):
        """

        Parameters
        ----------
        specs : dict
            A dictionary containing specifications for this outage. This
            dictionary must contain the following keys:
                - `count`
                    An integer value representing the total number of
                    times this outage should be scheduled. This number
                    should be larger than 0.
                - `duration`
                    An integer value representing the total number of
                    consecutive hours that this outage should take. This
                    value must be larger than 0 and less than the number
                    of hours in the allowed months.
                - `percentage_of_farm_down`
                    An integer or float value representing the total
                    percentage of the farm that will be take offline for
                    the duration of the outage. This value must be
                    in the range (0, 100].
                - `allowed_months`
                    A list of month names corresponding to the allowed
                    months for the scheduled outages. Month names can be
                    unformatted and can be specified using 3-letter
                    month abbreviations.
            The input dictionary can also provide the following optional
            keys:
                - `allow_outage_overlap` - by default, ``True``
                    A bool flag indicating wether or not this outage is
                    allowed to overlap with other outages, including
                    itself. It is recommended to set this value to
                    ``True`` whenever possible, as it allows for more
                    flexible scheduling.
                - `name` - by default, string containing init parameters
                    A unique name for the outage, used for more
                    descriptive error messages.

        """
        self._specs = specs
        self._full_month_names = None
        self._total_available_hours = None
        self._name = None
        self._validate()

    def _validate(self):
        """Validate the input specs."""
        self._validate_required_keys_exist()
        self._validate_count()
        self._validate_and_convert_to_full_name_months()
        self._validate_duration()
        self._validate_percentage()

    def _validate_required_keys_exist(self):
        """Raise an error if any required keys are missing."""
        missing_keys = [n not in self._specs for n in self.REQUIRED_KEYS]
        if any(missing_keys):
            msg = (
                "The following required keys are missing from the Outage "
                "specification: {}".format(sorted(missing_keys))
            )
            logger.error(msg)
            raise reVLossesValueError(msg)

    def _validate_count(self):
        """Validate that the total number of outages is an integer. """
        if not isinstance(self.count, int):
            msg = "Number of outages must be an integer, but got {} for {}"
            msg = msg.format(self.count, self.name)
            logger.error(msg)
            raise reVLossesValueError(msg)

        if self.count < 1:
            msg = "Number of outages must be greater than 0, but got {} for {}"
            msg = msg.format(self.count, self.name)
            logger.error(msg)
            raise reVLossesValueError(msg)

    def _validate_and_convert_to_full_name_months(self):
        """Validate month input and convert to full month names. """
        months = convert_to_full_month_names(self._specs['allowed_months'])
        known_months, unknown_months = filter_unknown_month_names(months)

        if unknown_months:
            msg = ("The following month names were not understood: {}. Please "
                   "use either the full month name or the standard 3-letter "
                   "month abbreviation. For more info, see the month name "
                   "documentation for the python standard package `calendar`."
                   ).format(unknown_months)
            logger.warning(msg)
            warnings.warn(msg, reVLossesWarning)

        if not known_months:
            msg = ("No known month names were provided! Please "
                   "use either the full month name or the standard 3-letter "
                   "month abbreviation. For more info, see the month name "
                   "documentation for the python standard package `calendar`. "
                   "Received input: {!r}"
                   ).format(self._specs['allowed_months'])
            logger.error(msg)
            raise reVLossesValueError(msg)

        self._full_month_names = list(set(known_months))

    def _validate_duration(self):
        """Validate that the duration is between 0 and the max total. """
        if not isinstance(self.duration, int):
            msg = ("Duration must be an integer number of hours, "
                   "but got {} for {}").format(self.duration, self.name)
            logger.error(msg)
            raise reVLossesValueError(msg)

        if not 1 <= self.duration <= self.total_available_hours:
            msg = (
                "Duration of outage must be between 1 and the total available "
                "hours based on allowed month input ({} for a total hour "
                "count of {}), but got {} for {}"
            ).format(self.allowed_months, self.total_available_hours,
                     self.percentage_of_farm_down, self.name)
            logger.error(msg)
            raise reVLossesValueError(msg)

    def _validate_percentage(self):
        """Validate that the percentage is in the range (0, 100]. """
        if not 0 < self.percentage_of_farm_down <= 100:
            msg = (
                "Percentage of farm down during outage must be in the range "
                "(0, 100], but got {} for {}"
            ).format(self.percentage_of_farm_down, self.name)
            logger.error(msg)
            raise reVLossesValueError(msg)

    def __repr__(self):
        return "Outage({!r})".format(self._specs)

    def __str__(self):
        if self._name is None:
            self._name = self._specs.get('name') or self._default_name()
        return self._name

    def _default_name(self):
        """Generate a default name for the outage."""
        specs = self._specs.copy()
        specs.update(
            {'allowed_months': self.allowed_months,
             'allow_outage_overlap': self.allow_outage_overlap}
        )
        specs_as_str = ", ".join(
            ["{}={}".format(k, v) for k, v in specs.items()]
        )
        return "Outage({})".format(specs_as_str)

    @property
    def count(self):
        """int: Total number of times outage should be scheduled."""
        return self._specs['count']

    @property
    def duration(self):
        """int: Total number of consecutive hours per outage."""
        return self._specs['duration']

    @property
    def percentage_of_farm_down(self):
        """int or float: Percent of farm taken down per outage."""
        return self._specs['percentage_of_farm_down']

    @property
    def allowed_months(self):
        """list: Months during which outage can be scheduled."""
        return self._full_month_names

    @property
    def allow_outage_overlap(self):
        """bool: Indicator for overlap with other outages."""
        return self._specs.get('allow_outage_overlap', True)

    @property
    def name(self):
        """str: Name of the outage."""
        return self._specs.get('name', str(self))

    @property
    def total_available_hours(self):
        """int: Total number of hours avialbale based on allowed months."""
        if self._total_available_hours is None:
            self._total_available_hours = len(
                hourly_indices_for_months(self.allowed_months)
            )
        return self._total_available_hours


class OutageScheduler:
    """A scheduler for multiple input outages.

    Given a list of information about different types of desired
    outages, this class leverages the stochastic scheduling routines of
    :class:`SingleOutageScheduler` to calculate the total losses due to
    the input outages on an hourly basis.

    Attributes
    ----------
    outages : :obj:`list` of :obj:`Outages <Outage>`
        The user-provided list of :obj:`Outages <Outage>` containing
        info about all types of outages to be scheduled.
    seed : :obj:`int`
        The seed value used to seed the random generator in order
        to produce random but reproducible losses. This is useful
        for ensuring that stochastically scheduled losses vary
        between different sites (i.e. that randomly scheduled
        outages in two different location do not match perfectly on
        an hourly basis).
    total_losses : :obj:`np.array`
        An array (of length 8760) containing the per-hour total loss
        percentage resulting from the stochastically scheduled outages.
        This array contains only zero values before the
        :meth:`~OutageScheduler.calculate` method is run.
    can_schedule_more : :obj:`np.array`
        A boolean array (of length 8760) indicating wether or not more
        losses can be scheduled for a given hour. This array keeps track
        of all the scheduling conflicts between input outages.

    Warnings
    --------
    It is possible that not all outages input by the user will be
    scheduled. This can happen when there is not enough time allowed
    for all of the input outages. To avoid this issue, always be sure to
    allow a large enough month range for long outages that take up a big
    portion of the farm and try to allow outage overlap whenever
    possible.

    See Also
    --------
    :class:`SingleOutageScheduler` : Single outage scheduler.
    :class:`Outage` : Specifications for a single outage.
    """

    def __init__(self, outages, seed=None):
        """
        Parameters
        ----------
        outages : list of :obj:`Outages <Outage>`
            A list of :obj:`Outages <Outage>`, where each :obj:`Outage`
            contains info about a single type of outage. See the
            documentation of :class:`Outage` for a description of the
            required keys of each outage dictionary.
        seed : int or `None`
            An integer value used to seed the random generator in order
            to produce random but reproducible losses. This is useful
            for ensuring that stochastically scheduled losses vary
            between different sites (i.e. that randomly scheduled
            outages in two different location do not match perfectly on
            an hourly basis). If `None`, the seed is set to 0.
        """
        self.outages = outages
        self.seed = seed or 0
        self.total_losses = np.zeros(8760)
        self.can_schedule_more = np.full(8760, True)

    def calculate(self):
        """Calculate total losses from stochastically scheduled outages.

        This function calls :meth:`SingleOutageScheduler.calculate`
        on every outage input (sorted by largest duration and then
        largest number of outages) and returns the aggregate the losses
        from the result.

        Returns
        -------
        :obj:`np.array`
            An array (of length 8760) containing the per-hour total loss
            percentage resulting from the stochastically scheduled
            outages.
        """
        sorted_outages = sorted(
            self.outages, key=lambda outage: (outage.duration, outage.count)
        )
        for outage in sorted_outages[::-1]:
            SingleOutageScheduler(outage, self).calculate()
        return self.total_losses


class SingleOutageScheduler:
    """A scheduler for a single outage.

    Given information about a single type of outage, this class
    facilitates the (randomized) scheduling of all requested instances
    of the outage. See :meth:`SingleOutageScheduler.calculate` for
    specific details about the scheduling process.

    Attributes
    ----------
    outage : :obj:`Outage`
        The user-provided :obj:`Outage` containing info about the outage
        to be scheduled.
    scheduler : :obj:`OutageScheduler`
        A scheduler object that keeps track of the total hourly losses
        from the input outage as well as any other outages it has
        already scheduled.
    can_schedule_more : :obj:`np.array`
        A boolean array (of length 8760) indicating wether or not more
        losses can be scheduled for a given hour. This is specific
        to the input outage only.

    Warnings
    --------
    It is possible that not all outages input by the user can be
    scheduled. This can happen when there is not enough time allowed
    for all of the input outages. To avoid this issue, always be sure to
    allow a large enough month range for long outages that take up a big
    portion of the farm and try to allow outage overlap whenever
    possible.

    See Also
    --------
    :class:`OutageScheduler` : Scheduler for multiple outages.
    :class:`Outage` : Specifications for a single outage.
    """

    MAX_ITER = 10_000
    """Max number of extra attempts to schedule outages."""

    def __init__(self, outage, scheduler):
        """

        Parameters
        ----------
        outage : Outage
            An outage object contianing info about the outage to be
            scheduled.
        scheduler : OutageScheduler
            A scheduler object that keeps track of the total hourly
            losses from the input outage as well as any other outages
            it has already scheduled.
        """
        self.outage = outage
        self.scheduler = scheduler
        self.can_schedule_more = np.full(8760, False)
        self._scheduled_outage_inds = []

    def calculate(self):
        """Calculate losses from stochastically scheduled outages.

        This function attempts to schedule outages according to the
        specification provided in the :obj:`Outage` input. Specifically,
        it checks the available hours based on the main
        :obj:`Scheduler <OutageScheduler>` (which may have other outages
        already scheduled) and attempts to randomly add new outages with
        the specified duration and percent of losses. The function
        terminates when the desired number of outages (specified by
        :attr:`Outage.count`) have been successfully scheduled, or when
        the number of attempts exceeds
        :attr:`~SingleOutageScheduler.MAX_ITER` + :attr:`Outage.count`.

        Warns
        -----
        reVLossesWarning
            If the number of requested outages could not be scheduled.
        """
        self.update_when_can_schedule_from_months()

        for iter_ind in range(self.outage.count + self.MAX_ITER):
            self.update_when_can_schedule()
            if not self.can_schedule_more.any():
                break
            outage_slice = self.find_random_outage_slice(
                seed=self.scheduler.seed + iter_ind
            )
            if self.can_schedule_more[outage_slice].all():
                self.schedule_losses(outage_slice)
            if len(self._scheduled_outage_inds) == self.outage.count:
                break

        if len(self._scheduled_outage_inds) < self.outage.count:
            if len(self._scheduled_outage_inds) == 0:
                msg_start = "Could not schedule any requested outages"
            else:
                msg_start = ("Could only schedule {} out of {} requested "
                             "outages")
                msg_start = msg_start.format(len(self._scheduled_outage_inds),
                                             self.outage.count)
            msg = ("{} after a max of {:,} iterations. You are likely "
                   "attempting to schedule a lot of long outages or a lot "
                   "of short outages with a large percentage of the farm at "
                   "a time. Please adjust the outage specifications and try "
                   "again")
            msg = msg.format(msg_start, self.outage.count + self.MAX_ITER)
            logger.warning(msg)
            warnings.warn(msg, reVLossesWarning)

    def update_when_can_schedule_from_months(self):
        """
        Update :attr:`can_schedule_more` using :attr:`Outage.allowed_months`.

        This function sets the :attr:`can_schedule_more` bool array to
        `True` for all of the months in :attr:`Outage.allowed_months`.
        """
        inds = hourly_indices_for_months(self.outage.allowed_months)
        self.can_schedule_more[inds] = True

    def update_when_can_schedule(self):
        """Update :attr:`can_schedule_more` using :obj:`OutageScheduler`.

        This function sets the :attr:`can_schedule_more` bool array to
        `True` wherever :attr:`OutageScheduler.can_schedule_more` is
        also `True` and wherever the losses from this outage would not
        cause the :attr:`OutageScheduler.total_losses` to exceed 100%.
        """
        self.can_schedule_more &= self.scheduler.can_schedule_more
        if self.outage.allow_outage_overlap:
            self.can_schedule_more &= (
                self.scheduler.total_losses
                + self.outage.percentage_of_farm_down
            ) <= 100
        else:
            self.can_schedule_more &= self.scheduler.total_losses == 0

    def find_random_outage_slice(self, seed=None):
        """Find a random slot of time for this type of outage.

        This function randomly selects a starting time for this outage
        given the allowed times in :attr:`can_schedule_more`. It does
        **not** verify that the outage can be scheduled for the entire
        requested duration.

        Parameters
        ----------
        seed : int, optional
            Integer used to seed the :func:`np.random.choice` call.
            If :obj:`None`, seed is not used.

        Returns
        -------
        :obj:`slice`
            A slice corresponding to the random slot of time for this
            type of outage.
        """
        if seed is not None:
            np.random.seed(seed)
        outage_ind = np.random.choice(np.where(self.can_schedule_more)[0])
        return slice(outage_ind, outage_ind + self.outage.duration)

    def schedule_losses(self, outage_slice):
        """Schedule the input outage during the given slice of time.

        Given a slice in the hourly loss array, add the losses from this
        outage (which is equivalent to scheduling them).

        Parameters
        ----------
        outage_slice : slice
            A slice corresponding to the slot of time to schedule this
            outage.
        """
        self._scheduled_outage_inds.append(outage_slice.start)
        self.scheduler.total_losses[outage_slice] += (
            self.outage.percentage_of_farm_down
        )
        if not self.outage.allow_outage_overlap:
            self.scheduler.can_schedule_more[outage_slice] = False


class ScheduledLossesMixin:
    """Mixin class for :class:`reV.SAM.generation.AbstractSamGeneration`.

    Warning
    -------
    Using this class for anything excpet as a mixin for
    :class:`~reV.SAM.generation.AbstractSamGeneration` may result in
    unexpected results and/or errors.
    """

    OUTAGE_CONFIG_KEY = 'reV-outages'
    """Specify outage information in the config file using this key."""

    def add_scheduled_losses(self):
        """Add stochastically scheduled losses to SAM config file.

        This function reads the information in the ``reV-outages`` key
        of the ``sam_sys_inputs`` dictionary and computes stochastically
        scheduled losses from that input. If the value for
        ``reV-outages`` is a string, it must have been generated by
        calling :func:`json.dumps` on the list of dictionaries
        containing outage specifications. Otherwise, the outage
        information is expected to be a list of dictionaries containing
        outage specifications. See :class:`Outage` for a description of
        the specifications allowed for each outage. The scheduled losses
        are passed to SAM via the ``hourly`` key to signify which hourly
        capacity factors should be adjusted with outage losses. If no
        outage info is specified in ``sam_sys_inputs``, no scheduled
        losses are added.

        See Also
        --------
        :class:`Outage` : Single outage specification.

        """

        outage_specs = self.sam_sys_inputs.pop(self.OUTAGE_CONFIG_KEY, None)
        if outage_specs is None:
            return

        if isinstance(outage_specs, str):
            outage_specs = json.loads(outage_specs)

        outages = [Outage(spec) for spec in outage_specs]
        logger.debug("Adding the following stochastically scheduled outages: "
                     "{}".format(outages))
        logger.debug("Scheduled outages seed: {}".format(self.outage_seed))

        scheduler = OutageScheduler(outages, seed=self.outage_seed)
        self.sam_sys_inputs['hourly'] = scheduler.calculate()
        logger.debug("Hourly adjustment factors as a result of scheduled "
                     "outages: {}".fromat(scheduler.total_losses.tolist()))

    @property
    def outage_seed(self):
        """int: A value to use as the seed for the outage losses. """
        try:
            return int(self.meta.name)
        except (AttributeError, TypeError, ValueError):
            pass

        try:
            return hash(tuple(self.meta))
        except (AttributeError, TypeError):
            pass

        return 0
