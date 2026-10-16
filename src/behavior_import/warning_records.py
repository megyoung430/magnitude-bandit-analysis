"""Keep a copy of the Python warnings raised while a block runs, without changing what is shown.

``warnings.catch_warnings(record=True)`` would swallow the warnings until the block ends. Here
the warning hook is wrapped instead: each warning is recorded and passed on to the hook that was
in place, so the console shows the same warnings at the same moment as without the log. What is
recorded is what Python's warning filters let through, which is what the console shows.
"""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple


class WarningRecord(NamedTuple):
    """One warning that was raised.

    Attributes:
        category (str): The warning class name, e.g. ``"ReversalWarning"``.
        message (str): The warning text.
        filename (str): The source file that raised it.
        lineno (int): The line in *filename*.
    """

    category: str
    message: str
    filename: str
    lineno: int


@contextmanager
def record_warnings() -> Iterator[list[WarningRecord]]:
    """Record every warning shown while the block runs, and still show it.

    Yields:
        list[WarningRecord]: Filled as warnings arrive, in order; stops growing when the block
        ends. Nested uses each record their own block.
    """
    records: list[WarningRecord] = []
    show_warning = warnings.showwarning

    def record_and_show(message, category, filename, lineno, *args, **kwargs):
        records.append(WarningRecord(category.__name__, str(message), filename, lineno))
        show_warning(message, category, filename, lineno, *args, **kwargs)

    with warnings.catch_warnings():
        warnings.showwarning = record_and_show
        yield records
