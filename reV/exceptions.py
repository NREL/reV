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


class ResourceKeyError(Exception):
    """
    KeyError for Resource Handlers
    """
    pass


class ResourceRuntimeError(Exception):
    """
    RuntimeError for Resource Handlers
    """
    pass


class ResourceValueError(Exception):
    """
    ValueError for Resource Handlers
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
