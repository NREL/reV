"""
Custom Exceptions and Errors for reV
"""


class reVError(Exception):
    """
    Generic Error for reV
    """
    pass


class ConfigError(Exception):
    """
    Error for bad configuration inputs
    """
    pass


class ExecutionError(Exception):
    """
    Error for execution failure
    """
    pass


class HandlerKeyError(Exception):
    """
    KeyError for Handlers
    """
    pass


class HandlerRuntimeError(Exception):
    """
    RuntimeError for Handlers
    """
    pass


class HandlerValueError(Exception):
    """
    ValueError for Handlers
    """
    pass


class ResourceError(Exception):
    """
    Error for poorly formatted resource.
    """
    pass


class SAMExecutionError(Exception):
    """
    Execution error for SAM simulations
    """
    pass


class ExtrapolationWarning(Warning):
    """
    Warning for when value will be extrapolated
    """
    pass


class ConfigWarning(Warning):
    """
    Warning for unclear or default configuration inputs
    """
    pass


class SAMInputWarning(Warning):
    """
    Warning for bad SAM inputs
    """
    pass


class HandlerWarning(Warning):
    """
    Warning during .h5 handling
    """
    pass
