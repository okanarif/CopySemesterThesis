"""
Logging utility for the SwarmPlanning pipeline.

Provides a factory function `get_logger(name)` that returns a Python standard
`logging.Logger` with a colored, timestamped formatter suitable for both
terminal and Jupyter notebook output.

Usage
-----
    from logger import get_logger
    log = get_logger(__name__)

    log.info("Environment loaded successfully.")
    log.warning("Resolution is very coarse.")
    log.error("Missing required field: 'world'")
"""

import logging
import sys
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# ANSI color codes
# ─────────────────────────────────────────────────────────────────────────────

class _Ansi:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    CYAN    = "\033[0;36m"
    GREEN   = "\033[0;32m"
    YELLOW  = "\033[0;33m"
    RED     = "\033[0;31m"
    RED_B   = "\033[1;31m"
    GREY    = "\033[0;90m"
    WHITE_B = "\033[1;37m"


# ─────────────────────────────────────────────────────────────────────────────
# Custom formatter with per-level colors
# ─────────────────────────────────────────────────────────────────────────────

_LEVEL_COLORS = {
    "DEBUG":    _Ansi.CYAN,
    "INFO":     _Ansi.GREEN,
    "WARNING":  _Ansi.YELLOW,
    "ERROR":    _Ansi.RED,
    "CRITICAL": _Ansi.RED_B,
}

_LOG_FMT  = "%(asctime)s  %(levelname)s  %(name)-22s  %(message)s"
_DATE_FMT = "%H:%M:%S"


class _ColorFormatter(logging.Formatter):
    """Formatter that wraps the level name in ANSI color codes."""

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelname, _Ansi.RESET)
        padded_level = f"{record.levelname:<8}"
        record.levelname = f"{color}{padded_level}{_Ansi.RESET}"
        record.name = f"{_Ansi.GREY}{record.name}{_Ansi.RESET}"
        return super().format(record)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(
    name: str,
    level: int = logging.DEBUG,
    stream: Optional[object] = None,
) -> logging.Logger:
    """
    Return a colored, timestamped logger.

    Calling this function multiple times with the same `name` is idempotent —
    the handler is added only once, so log messages are never duplicated.

    Parameters
    ----------
    name : str
        Logger name. Use ``__name__`` inside modules, or a descriptive label
        such as ``"swarm_notebook"`` at the notebook level.
    level : int
        Minimum logging level (default: ``logging.DEBUG``).
    stream : file-like, optional
        Output stream (default: ``sys.stdout``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Idempotency guard — skip if already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        _ColorFormatter(fmt=_LOG_FMT, datefmt=_DATE_FMT)
    )
    logger.addHandler(handler)

    return logger
