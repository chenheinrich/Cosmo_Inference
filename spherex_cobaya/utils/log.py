import os
import logging

def set_log_level(logger):
    if "LOG_LEVEL" in os.environ:
        level = os.environ["LOG_LEVEL"].upper()
        exec("logger.setLevel(logging.{})".format(level))
    else:
        logger.setLevel(logging.INFO)


def setup_logger(name):
    logger = logging.getLogger(name)
    set_log_level(logger)
    logging.basicConfig()
    return logger


logger = setup_logger('spherex_cobaya')


def class_logger(obj):
    return setup_logger(type(obj).__name__)


def file_logger(file):
    this_file = os.path.splitext(os.path.basename(file))[0]
    logger = setup_logger(' ' + this_file + ' ')
    return logger


class LoggedError(Exception):
    """
    Dummy exception, to be raised when the originating exception
    has been cleanly handled and logged.
    """

    def __init__(self, logger, *args, **kwargs):
        if args:
            logger.error(*args, **kwargs)
        msg = args[0] if len(args) else ""
        if msg and len(args) > 1:
            msg = msg % args[1:]
        super().__init__(msg)
