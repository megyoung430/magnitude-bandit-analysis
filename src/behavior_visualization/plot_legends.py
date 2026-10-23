"""Legend tidying applied to every figure when it is saved, so no plot function needs to know it.

:func:`tidy_legends` runs in the save step of :mod:`plot_helpers` on the finished figure:

- a legend repeated identically on several panels is kept on the first panel only;
- a lone mouse id is taken out of a legend and returned as a caption line (``line: MY_175_L``);
- a legend that sits inside its axes is moved out of the data: below the axes when it is short,
  beside them when it is long (a list of mice), and as one figure legend for the panels when no
  place next to its own panel is free of the other panels;
- every legend is drawn without a frame, with short handles and title-case labels.

The ideas follow the legend rules of the removed paper style; sizes, spines, line widths and
colours are left as each plot drew them.
"""

import math
import re

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.transforms import Bbox

HANDLE_LENGTH = 1.5
# Legends with more entries than this go beside the axes instead of below them.
MAX_ENTRIES_BELOW = 4
# Entries per column of a legend beside the axes.
ENTRIES_PER_COLUMN = 20
# Gap in points between the axes (with their labels) and a moved legend.
GAP_POINTS = 6
MOUSE_ID = re.compile(r"[A-Z]{1,4}_\d+(_[A-Z]+)?")
# Lower case in a title-case label: articles, conjunctions, prepositions, and the statistics.
SMALL_WORDS = frozenset(
    "a an and or the of from to in on at by for per vs via with across after before over under "
    "into between than se std".split()
)
_WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def _title_case_word(word: str, first: bool) -> str:
    """*word* (letters only) capitalised, unless it is a small word that is not first."""
    if not word.islower() or (word in SMALL_WORDS and not first):
        return word
    return word[0].upper() + word[1:]


def title_case(label: str) -> str:
    """*label* in title case: each lower-case word capitalised except small words.

    Words with a capital or a digit already (``P(best)``, ``MY_175_L``, ``t=0``) are kept as they
    are, as are articles, short prepositions and the statistics ``se`` and ``std`` after the
    first word. Hyphenated parts are capitalised one by one.

    Args:
        label (str): A legend label.

    Returns:
        str: The label in title case.
    """
    out, first = [], True
    for token in label.split(" "):
        if not token or re.search(r"[A-Z0-9_=]", token):
            out.append(token)
            first = first and not token
            continue
        out.append(_WORD.sub(lambda m, f=first: _title_case_word(m.group(), f), token))
        first = False
    return " ".join(out)


# --- reading and rebuilding legends ------------------------------------------------------------


def _panels(fig: plt.Figure) -> list[plt.Axes]:
    return [ax for ax in fig.axes if ax.get_visible() and ax.get_label() != "<colorbar>"]


def _labels(legend: Legend) -> list[str]:
    return [t.get_text() for t in legend.get_texts()]


def _style(legend: Legend) -> dict:
    """Font sizes and title of *legend*, to carry over to its replacement."""
    texts = legend.get_texts()
    style = {"title": legend.get_title().get_text() or None}
    if texts:
        style["fontsize"] = texts[0].get_fontsize()
    return style


def _new_legend(owner, handles, labels, style: dict, **placement) -> Legend:
    """A frameless legend with short handles and title-case labels on *owner* (axes or figure)."""
    return owner.legend(
        handles,
        [title_case(label) for label in labels],
        frameon=False,
        handlelength=HANDLE_LENGTH,
        **style,
        **placement,
    )


def _placement(legend: Legend) -> dict:
    """``loc``, ``ncols`` and, if set, the anchor of *legend* in its parent's coordinates."""
    # private attributes: matplotlib offers no public way to read these back
    placement = {"loc": legend._loc, "ncols": legend._ncols}
    if legend._bbox_to_anchor is not None:
        parent = legend.parent
        transform = parent.transAxes if isinstance(parent, plt.Axes) else parent.transFigure
        anchor = legend.get_bbox_to_anchor().transformed(transform.inverted())
        placement.update(bbox_to_anchor=anchor.bounds, bbox_transform=transform)
    return placement


def _restyle(legend: Legend, keep: set[str] | None = None) -> Legend | None:
    """Rebuild *legend* in the same place, frameless; with *keep*, only those labels."""
    owner, style, placement = legend.parent, _style(legend), _placement(legend)
    pairs = [
        (h, t)
        for h, t in zip(legend.legend_handles, _labels(legend), strict=True)
        if keep is None or t in keep
    ]
    legend.remove()
    if not pairs:
        return None
    handles, labels = zip(*pairs, strict=True)
    return _new_legend(owner, handles, labels, style, **placement)


# --- the rules -----------------------------------------------------------------------------------


def _drop_repeated(fig: plt.Figure) -> None:
    """Keep the first panel's legend and remove later legends with the same entries."""
    seen = set()
    for ax in _panels(fig):
        legend = ax.get_legend()
        if legend is None:
            continue
        labels = tuple(_labels(legend))
        if labels in seen:
            legend.remove()
        seen.add(labels)


def _mouse_of(label: str) -> str | None:
    head = label.split(" ")[0]
    return head if MOUSE_ID.fullmatch(head) else None


def _drop_single_mice(fig: plt.Figure) -> list[str]:
    """Take a lone mouse out of each legend; return ``line: <label>`` for the caption."""
    notes = []
    for legend in [*fig.legends, *(ax.get_legend() for ax in _panels(fig))]:
        if legend is None:
            continue
        mice = [label for label in _labels(legend) if _mouse_of(label)]
        if len(mice) == 1:
            _restyle(legend, keep=set(_labels(legend)) - {mice[0]})
            notes.append(f"line: {mice[0]}")
    return notes


def _is_inside(ax: plt.Axes, legend: Legend) -> bool:
    """Whether *legend* is drawn within the axes area, over the data."""
    if legend._bbox_to_anchor is None:
        return True
    anchor = legend.get_bbox_to_anchor()
    box = ax.bbox
    return (
        box.x0 - 1 <= anchor.x0
        and anchor.x1 <= box.x1 + 1
        and (box.y0 - 1 <= anchor.y0 and anchor.y1 <= box.y1 + 1)
    )


def _group(fig: plt.Figure, ax: plt.Axes) -> list[plt.Axes]:
    """*ax* and its twins: the panels drawn in the same place."""
    position = ax.get_position().bounds
    return [other for other in fig.axes if other.get_position().bounds == position]


def _union(boxes: list[Bbox]) -> Bbox:
    return Bbox.union(boxes)


def _candidates(n_entries: int, area: Bbox, gap: float) -> list[dict]:
    """Placements (in display coordinates) beside and below *area*, the preferred one first."""
    below = {
        "loc": "upper center",
        "anchor": (area.x0 + area.width / 2, area.y0 - gap),
        "ncols": min(n_entries, 4),
    }
    beside = {
        "loc": "upper left",
        "anchor": (area.x1 + gap, area.y1),
        "ncols": max(1, math.ceil(n_entries / ENTRIES_PER_COLUMN)),
    }
    return [below, beside] if n_entries <= MAX_ENTRIES_BELOW else [beside, below]


def _place(fig, handles, labels, style, candidate) -> Legend:
    x, y = fig.transFigure.inverted().transform(candidate["anchor"])
    return _new_legend(
        fig,
        handles,
        labels,
        style,
        loc=candidate["loc"],
        ncols=candidate["ncols"],
        bbox_to_anchor=(x, y),
        bbox_transform=fig.transFigure,
    )


def _clear_of(box: Bbox, others: list[Bbox]) -> bool:
    return not any(box.overlaps(other) for other in others)


def _move_out(fig: plt.Figure, renderer, gap: float) -> None:
    """Move every legend that covers its axes to a free place beside or below them."""
    leftover: list[tuple] = []
    placed_boxes: list[Bbox] = []
    for ax in _panels(fig):
        legend = ax.get_legend()
        if legend is None or not _is_inside(ax, legend):
            continue
        group = _group(fig, ax)
        others = [p.get_tightbbox(renderer) for p in _panels(fig) if p not in group]
        area = _union([p.get_tightbbox(renderer) for p in group])
        handles, labels, style = legend.legend_handles, _labels(legend), _style(legend)
        legend.remove()
        others += placed_boxes
        for candidate in _candidates(len(labels), area, gap):
            placed = _place(fig, handles, labels, style, candidate)
            box = placed.get_window_extent(renderer)
            if _clear_of(box, others):
                placed_boxes.append(box)
                break
            placed.remove()
        else:
            leftover.append((handles, labels, style))
    if leftover:
        _merge_beside_figure(fig, renderer, leftover, gap)


def _merge_beside_figure(fig: plt.Figure, renderer, leftover: list[tuple], gap: float) -> None:
    """One figure legend of the entries of *leftover* (each label once), beside all panels."""
    entries = {}
    for handles, labels, _ in leftover:
        for handle, label in zip(handles, labels, strict=True):
            entries.setdefault(label, handle)
    area = _union([ax.get_tightbbox(renderer) for ax in fig.axes if ax.get_visible()])
    area = _union([area, *(leg.get_window_extent(renderer) for leg in fig.legends)])
    beside = _candidates(MAX_ENTRIES_BELOW + 1, area, gap)[0]
    beside["ncols"] = max(1, math.ceil(len(entries) / ENTRIES_PER_COLUMN))
    _place(fig, list(entries.values()), list(entries), leftover[0][2], beside)


def tidy_legends(fig: plt.Figure) -> list[str]:
    """Apply the legend rules of this module to *fig* in place; return its caption lines.

    Args:
        fig (plt.Figure): A finished figure, laid out as it will be saved.

    Returns:
        list[str]: ``line: <mouse>`` for each legend that named a single mouse.
    """
    _drop_repeated(fig)
    notes = _drop_single_mice(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap = GAP_POINTS * fig.dpi / 72
    for legend in [*fig.legends, *(ax.get_legend() for ax in _panels(fig))]:
        if legend is not None:
            _restyle(legend)
    _move_out(fig, renderer, gap)
    return notes
