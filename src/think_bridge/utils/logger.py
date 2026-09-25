"""Rank-aware colored logging and compact boxed summaries."""

from __future__ import annotations

import logging
import os
import sys
import textwrap


# ANSI color codes
class _Colors:
    RESET = "\033[0m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"


_LEVEL_COLORS = {
    logging.DEBUG: _Colors.GRAY,
    logging.INFO: _Colors.BLUE,
    logging.WARNING: _Colors.YELLOW,
    logging.ERROR: _Colors.RED,
    logging.CRITICAL: _Colors.RED + _Colors.BOLD,
}


class _ColorFormatter(logging.Formatter):
    """Format one log record with optional ANSI level coloring."""

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        name = record.name
        msg = record.getMessage()
        if self.use_color:
            color = _LEVEL_COLORS.get(record.levelno, "")
            return f"{color}[{level}:{name}]{_Colors.RESET} {msg}"
        return f"[{level}:{name}] {msg}"


def get_logger(name: str = "think_bridge") -> logging.Logger:
    """Return a rank-aware stdout logger without duplicate handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = local_rank == 0

    handler = logging.StreamHandler(sys.stdout)
    use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    handler.setFormatter(_ColorFormatter(use_color=use_color))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if is_main else logging.ERROR)
    logger.propagate = False

    return logger


logger = get_logger()


def print_box(title: str, content: dict | str, *, width: int = 60) -> None:
    """Print a wrapped rank-zero summary inside a Unicode border."""
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    if isinstance(width, bool) or not isinstance(width, int) or width < 4:
        raise ValueError("box width must be an integer of at least four")
    inner_width = width - 2

    def wrapped_lines(value: object) -> list[str]:
        logical_lines = str(value).splitlines() or [""]
        rendered: list[str] = []
        for logical_line in logical_lines:
            rendered.extend(
                textwrap.wrap(
                    logical_line,
                    width=inner_width,
                    expand_tabs=True,
                    replace_whitespace=True,
                    drop_whitespace=True,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                or [""]
            )
        return rendered

    def print_line(line: str) -> None:
        print(f"│ {line:<{inner_width}} │")

    border = "─" * width
    print(f"┌{border}┐")
    for line in wrapped_lines(title):
        print_line(line)
    print(f"├{border}┤")
    if isinstance(content, dict):
        for k, v in content.items():
            for line in wrapped_lines(f"{k}: {v}"):
                print_line(line)
    else:
        for line in wrapped_lines(content):
            print_line(line)
    print(f"└{border}┘")
    sys.stdout.flush()
