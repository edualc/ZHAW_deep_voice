"""
    Provides a uniform way of logging in the application.
    Based on previous work of Gygax and Egly.
"""
import logging

formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y%m%d %I:%M:%S')


def get_logger(name, level):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    return logger


def add_file_handler(logger, log_file_path):
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

