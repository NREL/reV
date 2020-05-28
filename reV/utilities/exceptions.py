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


class JSONError(Exception):
    """
    Error reading json file.
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


class CollectionValueError(HandlerValueError):
    """
    ValueError for collection handler.
    """


class CollectionRuntimeError(HandlerRuntimeError):
    """
    RuntimeError for collection handler.
    """


class ResourceError(Exception):
    """
    Error for poorly formatted resource.
    """


class PySAMVersionError(Exception):
    """
    Version error for SAM installation
    """


class SAMExecutionError(Exception):
    """
    Execution error for SAM simulations
    """


class SAMInputError(Exception):
    """
    Input error for SAM simulations
    """


class SupplyCurveError(Exception):
    """
    Execution error for SAM simulations
    """


class EmptySupplyCurvePointError(SupplyCurveError):
    """
    Execution error for SAM simulations
    """


class SupplyCurveInputError(SupplyCurveError):
    """
    Execution error for SAM simulations
    """


class NearestNeighborError(Exception):
    """
    Execution error for bad nearest neighbor mapping results.
    """


class DataShapeError(Exception):
    """
    Error with mismatched data shapes.
    """


class ExclusionLayerError(Exception):
    """
    Error with bad exclusion data
    """


class OutputWarning(Warning):
    """
    Warning for suspect output files or data
    """


class ExtrapolationWarning(Warning):
    """
    Warning for when value will be extrapolated
    """


class InputWarning(Warning):
    """
    Warning for unclear or default configuration inputs
    """


class OffshoreWindInputWarning(Warning):
    """
    Warning for potentially dangerous offshore wind inputs
    """


class OrcaOutputWarning(Warning):
    """
    Warning for bad ORCA outputs.
    """


class ConfigWarning(Warning):
    """
    Warning for unclear or default configuration inputs
    """


class SAMInputWarning(Warning):
    """
    Warning for bad SAM inputs
    """


class SAMExecutionWarning(Warning):
    """
    Warning for problematic SAM execution
    """


class PySAMVersionWarning(Warning):
    """
    Version warning for SAM installation
    """


class ParallelExecutionWarning(Warning):
    """
    Warning for parallel job execution.
    """


class SlurmWarning(Warning):
    """
    Warning for SLURM errors/warnings
    """


class HandlerWarning(Warning):
    """
    Warning during .h5 handling
    """


class CollectionWarning(Warning):
    """
    Warning during .h5 collection
    """


class FileInputWarning(Warning):
    """
    Warning during input file checks.
    """


class reVDeprecationWarning(Warning):
    """
    Warning of deprecated feature.
    """
