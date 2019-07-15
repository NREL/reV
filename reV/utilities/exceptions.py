# -*- coding: utf-8 -*-
"""
Custom Exceptions and Errors for reV
"""


class reVError(Exception):
    """
    Generic Error for reV
    """


class ConfigError(Exception):
    """
    Error for bad configuration inputs
    """


class FileInputError(Exception):
    """
    Error during input file checks.
    """


class ExecutionError(Exception):
    """
    Error for execution failure
    """


class PipelineError(Exception):
    """
    Error for pipeline execution failure
    """


class HandlerKeyError(Exception):
    """
    KeyError for Handlers
    """


class HandlerRuntimeError(Exception):
    """
    RuntimeError for Handlers
    """


class HandlerValueError(Exception):
    """
    ValueError for Handlers
    """


class ResourceError(Exception):
    """
    Error for poorly formatted resource.
    """


class SAMExecutionError(Exception):
    """
    Execution error for SAM simulations
    """


class RPMValueError(Exception):
    """
    ValueError for RPM Pipeline
    """


class SupplyCurveError(Exception):
    """
    Execution error for SAM simulations
    """


class EmptySupplyCurvePointError(SupplyCurveError):
    """
    Execution error for SAM simulations
    """
    pass


class SupplyCurveInputError(SupplyCurveError):
    """
    Execution error for SAM simulations
    """


class OutputWarning(Warning):
    """
    Warning for suspect output files or data
    """
    pass


class ExtrapolationWarning(Warning):
    """
    Warning for when value will be extrapolated
    """


class ConfigWarning(Warning):
    """
    Warning for unclear or default configuration inputs
    """


class SAMInputWarning(Warning):
    """
    Warning for bad SAM inputs
    """


class HandlerWarning(Warning):
    """
    Warning during .h5 handling
    """


class FileInputWarning(Warning):
    """
    Warning during input file checks.
    """
