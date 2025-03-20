"""Logging module."""

import logging
lib_name = "ebs"
logger = logging.getLogger(lib_name)

shell_handler = logging.StreamHandler()
file_handler = logging.FileHandler("debug.log")

logger.setLevel(logging.DEBUG)
shell_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)

shell_fmt = "%(levelname)s [%(asctime)s] \033[0m%(message)s"
file_fmt = (
    f"[{lib_name}] %(levelname)s %(asctime)s [%(filename)s:"
    "%(funcName)s:%(lineno)d] %(message)s"
)

file_formatter = logging.Formatter(file_fmt)

file_handler.setFormatter(file_formatter)

logger.addHandler(shell_handler)
logger.addHandler(file_handler)

logger.propagate = True
