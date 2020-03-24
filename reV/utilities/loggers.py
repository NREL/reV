# -*- coding: utf-8 -*-
"""
Logging for reV
"""
import logging
import sys
import os
import psutil

__all__ = ['setup_logger', 'LoggingAttributes', 'init_logger']

FORMAT = '%(levelname)s - %(asctime)s [%(filename)s:%(lineno)d] : %(message)s'
LOG_LEVEL = {'INFO': logging.INFO,
             'DEBUG': logging.DEBUG,
             'WARNING': logging.WARNING,
             'ERROR': logging.ERROR,
             'CRITICAL': logging.CRITICAL}


def get_handler(log_level="INFO", log_file=None, log_format=FORMAT):
    """
    get logger handler

    Parameters
    ----------
    log_level : str
        handler-specific logging level, must be key in LOG_LEVEL.
    log_file : str
        path to the log file
    log_format : str
        format string to use with the logging package

    Returns
    -------
    handler : logging.FileHandler | logging.StreamHandler
        handler to add to logger
    """
    if log_file:
        # file handler with mode "a"
        handler = logging.FileHandler(log_file, mode='a')
    else:
        # stream handler to system stdout
        handler = logging.StreamHandler(sys.stdout)

    if log_format:
        logformat = logging.Formatter(log_format)
        handler.setFormatter(logformat)

    # Set a handler-specific logging level (root logger should be at debug)
    handler.setLevel(LOG_LEVEL[log_level.upper()])

    return handler


def setup_logger(logger_name, log_level="INFO", log_file=None,
                 log_format=FORMAT):
    """
    Setup logging instance with given name and attributes

    Parameters
    ----------
    logger_name : str
        Name of logger
    log_level : str
        Level of logging to capture, must be key in LOG_LEVEL. If multiple
        handlers/log_files are requested in a single call of this function,
        the specified logging level will be applied to all requested handlers.
    log_file : str | list
        Path to file to use for logging, if None use a StreamHandler
        list of multiple handlers is permitted
    log_format : str
        Format for loggings, default is FORMAT

    Returns
    -------
    logger : logging.logger
        instance of logger for given name, with given level and added handler
    handler : logging.FileHandler | logging.StreamHandler | list
        handler(s) added to logger
    """
    logger = logging.getLogger(logger_name)
    current_handlers = [str(h) for h in logger.handlers]

    # Set root logger to debug, handlers will control levels above debug
    logger.setLevel(LOG_LEVEL["DEBUG"])

    handlers = []
    if isinstance(log_file, list):
        for h in log_file:
            handlers.append(get_handler(log_level=log_level, log_file=h,
                                        log_format=log_format))
    else:
        handlers.append(get_handler(log_level=log_level, log_file=log_file,
                                    log_format=log_format))

    for handler in handlers:
        if str(handler) not in current_handlers:
            logger.addHandler(handler)

    return logger


class LoggingAttributes:
    """
    Class to store and pass logging attributes to modules
    """
    def __init__(self):
        self._loggers = {}

    def __setitem__(self, logger_name, attributes):
        log_attrs = self[logger_name]
        for attr, value in attributes.items():
            if attr == 'log_file':
                handlers = list(log_attrs.get('log_file', []))
                if not isinstance(value, (list, tuple)):
                    # make the log_file request into a iterable list
                    value = [value]
                for v in value:
                    if v not in handlers:
                        # check if each handler has been previously set
                        handlers.append(v)
                log_attrs[attr] = handlers
            else:
                log_attrs[attr] = value

        self._loggers[logger_name] = log_attrs

    def __getitem__(self, logger_name):
        return self._loggers.get(logger_name, {})

    def init_logger(self, loggers):
        """
        Extract logger attributes and initialize logger

        Parameters
        ----------
        loggers : str | list
            Logger names to initialize
        """
        if not isinstance(loggers, list):
            loggers = [loggers]

        for logger_name in loggers:
            try:
                attrs = self[logger_name]
                setup_logger(logger_name, **attrs)
            except KeyError:
                pass


REV_LOGGERS = LoggingAttributes()


def init_logger(logger_name, log_level="INFO", log_file=None,
                log_format=FORMAT):
    """
    Starts logging instance and adds logging attributes to REV_LOGGERS

    Parameters
    ----------
    logger_name : str
        Name of logger to initialize
    log_level : str
        Level of logging to capture, must be key in LOG_LEVEL. If multiple
        handlers/log_files are requested in a single call of this function,
        the specified logging level will be applied to all requested handlers.
    log_file : str | list
        Path to file to use for logging, if None use a StreamHandler
        list of multiple handlers is permitted
    log_format : str
        Format for loggings, default is FORMAT

    Returns
    -------
    logger : logging.logger
        logging instance that was initialized
    """
    kwargs = {"log_level": log_level, "log_file": log_file,
              "log_format": log_format}
    logger = setup_logger(logger_name, **kwargs)

    REV_LOGGERS[logger_name] = kwargs

    return logger


def init_mult(name, logdir, modules, verbose=False, node=False):
    """Init multiple loggers to a single file or stdout.

    Parameters
    ----------
    name : str
        Job name; name of log file.
    logdir : str
        Target directory to save .log files.
    modules : list | tuple
        List of reV modules to initialize loggers for.
    verbose : bool
        Option to turn on debug logging.
    node : bool
        Flag for whether this is a node-level logger. If this is a node logger,
        and the log level is info, the log_file will be None (sent to stdout).

    Returns
    -------
    loggers : list
        List of logging instances that were initialized.
    """

    if verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    loggers = []
    for module in modules:
        log_file = os.path.join(logdir, '{}.log'.format(name))

        # check for redundant loggers in the REV_LOGGERS singleton
        logger = REV_LOGGERS[module]

        if ((not node or (node and log_level == 'DEBUG'))
                and 'log_file' not in logger):
            # No log file belongs to this logger, init a logger file
            logger = init_logger(module, log_level=log_level,
                                 log_file=log_file)
        elif node and log_level == 'INFO':
            # Node level info loggers only go to STDOUT/STDERR files
            logger = init_logger(module, log_level=log_level, log_file=None)
        loggers.append(logger)
    return loggers


def log_mem(logger, log_level='DEBUG'):
    """Log the memory usage to the input logger object.

    Parameters
    ----------
    logger : logging.Logger
        Logger object to log memory message to.
    log_level : str
        DEBUG or INFO for different log levels for this log message.

    Returns
    -------
    msg : str
        Memory utilization log message string.
    """
    mem = psutil.virtual_memory()
    msg = ('Memory utilization is {0:.3f} GB out of '
           '{1:.3f} GB total ({2:.1f}% used)'
           .format(mem.used / 1e9, mem.total / 1e9,
                   100 * mem.used / mem.total))
    if log_level.upper() == 'DEBUG':
        logger.debug('\t- {}'.format(msg))
    else:
        logger.info(msg)
    return msg
