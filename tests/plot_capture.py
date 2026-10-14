"""Helpers for characterization tests of figure functions.

``capture_figures`` replaces ``plt.show`` and ``Figure.savefig`` with recorders, so
a test can call a public plotting function and then inspect the exact figures it
produced. The recorded figure objects stay inspectable after the function closes
them.
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


@dataclass
class SavedFigure:
    figure: Figure
    path: str
    kwargs: dict


@dataclass
class Capture:
    shown: list[list[Figure]] = field(default_factory=list)
    saved: list[SavedFigure] = field(default_factory=list)

    def saved_names(self) -> list[str]:
        return [Path(s.path).name for s in self.saved]


def capture_figures(monkeypatch, *, write_files: bool = False) -> Capture:
    """Record ``plt.show`` and ``Figure.savefig`` calls; skip writing unless *write_files*."""
    capture = Capture()
    real_savefig = Figure.savefig

    def show(*args, **kwargs):
        capture.shown.append([plt.figure(n) for n in plt.get_fignums()])

    def savefig(self, fname, *args, **kwargs):
        capture.saved.append(SavedFigure(self, str(fname), kwargs))
        if write_files:
            real_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(plt, "show", show)
    monkeypatch.setattr(Figure, "savefig", savefig)
    plt.close("all")
    return capture


def lines_by_label(ax, label):
    return [ln for ln in ax.lines if ln.get_label() == label]


def line_data(line):
    return list(line.get_xdata()), list(line.get_ydata())
