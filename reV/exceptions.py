"""
Custom Exceptions and Errors for reV
"""


class reVError(Exception):
    """
    Generic Error for reV
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


class ExtrapolationWarning(Warning):
    """
    Warning for when value will be extrapolated
    """
    pass
