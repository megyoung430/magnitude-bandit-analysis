"""Entry point for the drifting-task port-preference figures.

Drifting problems only: the drifting task has block changes, not good or bad reversals.
"""

import logging
from pathlib import Path

import pandas as pd

from src.behavior_analysis.port_preference import (
    COMPARED_CATEGORIES,
    add_ratio,
    category_summary,
    count_nan_ratios,
    count_port_preference,
    mask_single_category_mice,
    pool_problems,
)
from src.behavior_analysis.reversal_format import check_blocks_only_sessions
from src.behavior_visualization.plot_port_preference import (
    plot_category_ratios,
    plot_port_ratio_heatmaps,
)

logger = logging.getLogger(__name__)

CATEGORY_FIGURE = "Port Preference X vs Plus"
HEATMAP_FIGURE = "Port Preference Per Mouse"


def _initiation_ports(by_port: pd.DataFrame) -> dict[str, set[str]]:
    """``{mouse: initiation ports over its problems}``."""
    return {m: set(g["initiation"]) for m, g in by_port.groupby("mouse")}


def run_port_preference_analysis(
    subjects_trials_by_problem: dict, out_dir: Path, *, scope_label: str
) -> dict:
    """How much each port, and X versus plus ports, is chosen relative to how often it is best.

    The measure is ``P(chosen) / P(best)`` over the trials where the port (or any port of the
    category) was offered; see ``behavior_analysis.port_preference`` for the definitions of X
    (on the initiation port's diagonals) and plus (in its row or column), relative to each
    problem's initiation port, and of the counting. Counts are pooled over the problems given,
    per mouse, before any ratio is formed; mice are the replicates for the mean ± SE of the X
    and plus ratios. Ports or categories that were never best have no ratio; how many there are
    is logged and returned. A mouse offered ports of one category only in the pooled scope has
    that category's ratio equal to 1 by construction, so its category ratios are set to NaN;
    these mice are listed in the log, in the result, and in the figure title.

    Writes, as PNG and PDF:

    - ``Port Preference X vs Plus``: bars of mean ± SE across mice, one line per mouse;
    - ``Port Preference Per Mouse``: one 3 x 3 maze per mouse coloured by the port ratios
      (log2 scale centred on 1, grey where undefined).

    Pass all drifting problems for the across-problem figures, or one problem for that
    problem's own figures.

    Args:
        subjects_trials_by_problem (dict): ``{problem: {mouse: {session_key: session}}}``.
        out_dir (Path): Folder for the figures; created if missing.
        scope_label (str): Names the data in the figure titles, e.g. ``"problem-03"``.
            Keyword-only.

    Returns:
        dict: ``"counts"`` (the :class:`PortCounts` per mouse and problem), ``"port_ratios"``
        and ``"category_ratios"`` (per-mouse counts pooled over the problems, with ``ratio``),
        ``"summary"`` (mean, se and n_mice of the X and plus ratios), ``"nan_cases"``
        (``{"ports": n, "categories": n}`` offered but never best) and ``"single_category_mice"``
        (sorted names of the mice left out of the category figure and summary).

    Raises:
        ValueError: If a session is counter format (the drifting task has only block changes),
            or the counting rules of ``count_port_preference`` are broken.
    """
    check_blocks_only_sessions(subjects_trials_by_problem, "run_port_preference_analysis")
    counts = count_port_preference(subjects_trials_by_problem)
    port_ratios = add_ratio(pool_problems(counts.by_port, "port"))
    category_ratios = add_ratio(pool_problems(counts.by_category, "category"))
    # Counted before the masking, so a masked mouse is not also reported as "never best".
    nan_cases = count_nan_ratios(port_ratios, category_ratios)
    category_ratios, single_category_mice = mask_single_category_mice(category_ratios)
    summary = category_summary(category_ratios)
    log = logger.warning if any(nan_cases.values()) else logger.info
    log(
        "port preference [%s]: %d mouse-port and %d mouse-category ratios undefined "
        "(offered but never best)",
        scope_label,
        nan_cases["ports"],
        nan_cases["categories"],
    )
    if single_category_mice:
        logger.warning(
            "port preference [%s]: %d %s excluded: only one category offered (%s)",
            scope_label,
            len(single_category_mice),
            "mouse" if len(single_category_mice) == 1 else "mice",
            ", ".join(single_category_mice),
        )
    out_dir = Path(out_dir)
    plot_category_ratios(
        category_ratios[category_ratios["category"].isin(COMPARED_CATEGORIES)],
        summary,
        scope_label=scope_label,
        n_excluded=len(single_category_mice),
        save_path=out_dir / CATEGORY_FIGURE,
    )
    plot_port_ratio_heatmaps(
        port_ratios,
        scope_label=scope_label,
        save_path=out_dir / HEATMAP_FIGURE,
        initiation_ports=_initiation_ports(counts.by_port),
    )
    return {
        "counts": counts,
        "port_ratios": port_ratios,
        "category_ratios": category_ratios,
        "summary": summary,
        "nan_cases": nan_cases,
        "single_category_mice": single_category_mice,
    }
