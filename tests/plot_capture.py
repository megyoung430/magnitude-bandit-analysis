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
from matplotlib.figure import Figure, SubplotParams  # noqa: E402


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


def title_pad_points(ax):
    """Distance in points between the axes' top edge and the title's baseline anchor."""
    position = ax.title.get_position()
    shifted = ax.title.get_transform().transform(position)
    base = ax.transAxes.transform(position)
    return (shifted[1] - base[1]) * 72 / ax.figure.dpi


def was_tight_laid_out(fig):
    """True if ``fig.tight_layout()`` moved the subplot margins off their defaults."""
    default = SubplotParams()
    pars = fig.subplotpars
    return (pars.left, pars.right, pars.bottom, pars.top) != (
        default.left,
        default.right,
        default.bottom,
        default.top,
    )


def cap_size(container):
    """Cap size in points of an error-bar container (bar-with-yerr or errorbar)."""
    errorbar = getattr(container, "errorbar", container)
    return errorbar.lines[1][0].get_markersize() / 2
