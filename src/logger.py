import logging
from colorama import Fore, Style, init

init(autoreset=True)

LEVEL_COLORS = {
    "DEBUG": Fore.CYAN,
    "INFO": Fore.GREEN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.MAGENTA,
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        color = LEVEL_COLORS.get(levelname, "")
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"


def get_logger(name: str = "main") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = ColorFormatter(
        "[%(levelname)s] %(asctime)s - %(name)s.%(funcName)s - %(message)s", "%H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
