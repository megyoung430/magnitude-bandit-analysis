"""Poke analysis for one problem: long pokes per relative session and the two maze heatmaps.

Entry points:

- :func:`run_poke_analysis` runs everything for one problem, writes the figures to one folder and
  returns the numbers (including a per-session summary table) without writing any table file.
- :func:`previous_problem_choice_towers` and :func:`previous_choice_towers_for_problem`
  (re-exported from :mod:`src.behavior_analysis.poke_counts`) give each mouse's previous-problem
  choice ports from the full ordered problem dict; the runner passes the result of the second as
  ``previous_choice_towers``.

Figures written to ``out_dir`` (PNG and PDF):

- ``Long Pokes by Relative Session``: mean non-choice long pokes per trial (``num_long_pokes``)
  for each relative session of the problem, bars = mean +/- SE across mice, lines = mice.
- ``Maze Heatmaps - Pokes After Choice`` and ``Maze Heatmaps - Pokes After Initiation``: one map
  per mouse, all sessions of the problem together.
"""

from __future__ import annotations

import warnings
from collections.abc import Collection, Mapping
from pathlib import Path

import numpy as np

from src.behavior_analysis.long_pokes_by_session import (
    problem_mean_long_pokes,
    session_mean_long_pokes,
    summarize_pokes_by_session,
    to_relative_sessions,
)
from src.behavior_analysis.poke_counts import (  # noqa: F401  (re-exported)
    SessionPokeCounts,
    mouse_poke_counts,
    previous_choice_towers_for_problem,
    previous_problem_choice_towers,
    problem_choice_ports,
    total_counts,
)
from src.behavior_analysis.poke_tables import cross_check_table
from src.behavior_analysis.stats import mean_se_n
from src.behavior_import.problem_numbering import DEFAULT_INITIATION_TOWER
from src.behavior_visualization.plot_maze_heatmaps import plot_maze_heatmaps_across_mice
from src.behavior_visualization.plot_num_long_pokes import (
    plot_num_pokes_across_mice,
    relative_session_label,
)

LONG_POKES_STEM = "Long Pokes by Relative Session"
AFTER_CHOICE_STEM = "Maze Heatmaps - Pokes After Choice"
AFTER_INITIATION_STEM = "Maze Heatmaps - Pokes After Initiation"
_SCALE_NOTE = "colour scale: 0 to each mouse's own maximum count"
_JITTER_SEED = 0


def _validate_problem(subjects_trials: Mapping) -> None:
    """Check that *subjects_trials* is ``{mouse: {session_key: session_dict with "df"}}``."""
    if not subjects_trials:
        raise ValueError("subjects_trials_by_problem has no mice.")
    for mouse, sessions in subjects_trials.items():
        if not isinstance(sessions, Mapping) or not sessions:
            raise ValueError(f"{mouse}: expected a non-empty dict of sessions")
        for key, session in sessions.items():
            if isinstance(session, Mapping) and "df" in session:
                continue
            if isinstance(session, Mapping) and any(
                isinstance(v, Mapping) and "df" in v for v in session.values()
            ):
                raise ValueError(
                    "expected {mouse: {session: session_dict}} for one problem, but "
                    f"{mouse} -> {key} holds another level of sessions; pass the "
                    "sessions of a single problem"
                )
            raise ValueError(f"{mouse} {key}: session has no raw 'df' recordings")


def _previous_ports(
    mice: list[str], previous: Mapping[str, Collection[str] | None] | None
) -> dict[str, Collection[str] | None] | None:
    if previous is None:
        return None
    missing = [m for m in mice if m not in previous]
    if missing:
        raise ValueError(
            f"previous_choice_towers has no entry for mice {missing}; use None for a mouse "
            "without a previous problem"
        )
    return {m: previous[m] for m in mice}


def _check_parsed(parsed: Mapping, subjects_trials: Mapping) -> dict:
    """*parsed* restricted to the problem's mice, after checking it covers every session."""
    for mouse, sessions in subjects_trials.items():
        missing = [key for key in sessions if key not in parsed.get(mouse, {})]
        if missing:
            raise ValueError(f"poke_counts has no entry for {mouse} sessions {missing}")
    return {mouse: dict(parsed[mouse]) for mouse in sorted(subjects_trials)}


def _initiation_ports(subjects_trials: Mapping) -> dict[str, str]:
    """Each mouse's initiation tower (default ``"B2"`` when unlogged).

    Raises:
        ValueError: If one mouse's sessions of the problem use several initiation towers.
    """
    ports = {}
    for mouse, sessions in subjects_trials.items():
        towers = {s.get("initiation_tower") or DEFAULT_INITIATION_TOWER for s in sessions.values()}
        if len(towers) != 1:
            raise ValueError(
                f"{mouse}: sessions of one problem use several initiation towers: {sorted(towers)}"
            )
        ports[mouse] = next(iter(towers))
    return ports


def _warn_on_disagreement(per_session: Mapping[str, Mapping[str, SessionPokeCounts]]) -> None:
    for mouse, by_session in per_session.items():
        for session, counts in by_session.items():
            check = counts.long_poke_check
            if check.mismatches:
                first_trial, raw, printed = check.mismatches[0]
                warnings.warn(
                    f"{mouse} {session}: raw long pokes disagree with the printed long_pokes in "
                    f"{len(check.mismatches)} of {check.n_trials} trials (first: trial "
                    f"{first_trial}, raw {raw}, printed {printed})",
                    UserWarning,
                    stacklevel=2,
                )


def _long_poke_ymax(summary) -> float:
    """Upper y-limit for the long-poke plot: room above every bar, error bar and mouse line."""
    se = np.where(np.isfinite(summary.se), summary.se, 0.0)
    values = np.concatenate([summary.mean + se, *summary.by_mouse.values()])
    finite = values[np.isfinite(values)]
    return max(1.0, 1.25 * float(finite.max())) if finite.size else 1.0


def _panel_titles(per_session, kind: str) -> dict[str, str]:
    def n_trials(counts: SessionPokeCounts) -> int:
        return counts.n_valid - (counts.n_open_windows if kind == "after_choice" else 0)

    return {
        mouse: f"{mouse} (n={sum(n_trials(c) for c in by_session.values())} trials)"
        for mouse, by_session in per_session.items()
    }


def run_poke_analysis(
    subjects_trials_by_problem: dict,
    out_dir: Path,
    *,
    previous_choice_towers: dict | None = None,
    poke_counts: dict | None = None,
) -> dict:
    """Long pokes per relative session and the two maze heatmaps for one problem.

    Args:
        subjects_trials_by_problem (dict): The sessions of ONE problem, ``{mouse: {session_key:
            session_dict}}``, as in a problem of ``load_trials_by_problem`` (the argument is named
            as in the other entry points). Each session needs the importer's raw ``"df"``
            recordings (one DataFrame or a list), ``"num_long_pokes"`` and ``"choice_towers"``.
        out_dir (Path): Folder for the figures and tables (created if missing); the runner gives
            the problem's ``long-pokes`` folder.
        previous_choice_towers (dict | None): ``{mouse: ports of the mouse's previous problem, or
            None}`` for the red outlines, as returned by
            :func:`previous_choice_towers_for_problem`. ``None`` (default) draws no red outlines;
            a dict must have an entry for every mouse. Keyword-only.
        poke_counts (dict | None): ``{mouse: {session_key: SessionPokeCounts}}`` already parsed
            (for example ``per_session`` of an earlier call, or ``mouse_poke_counts`` per
            mouse), so the raw events are parsed once. It must cover every mouse and session;
            ``None`` (default) parses the sessions' ``"df"`` here. Keyword-only.

    Returns:
        dict: A dict with ``"long_pokes"`` (``"by_session"`` = ``{mouse: {session: mean}}``,
        ``"relative"`` = the same keyed ``rel_ses-NN``, ``"problem_mean_by_mouse"`` and
        ``"across_mice"`` = ``(mean, se, n)`` of those means), ``"counts"`` (``{kind: {mouse:
        {port: count}}}`` for ``"after_initiation"`` and ``"after_choice"``), ``"per_session"``
        (``{mouse: {session: SessionPokeCounts}}``), ``"cross_check"`` (the raw-versus-printed
        long-poke comparison per session), ``"summary"`` (a DataFrame with one row per mouse and
        session: ``mouse, session, n_trials, n_port_finding_trials, n_trials_counted,
        n_trials_dropped_no_print, n_trials_dropped_inconsistent, n_open_after_choice_windows,
        n_long_poke_checked, n_long_poke_agree, parse_error``) and ``"files"`` (the figure paths
        written, PNG then PDF per figure). No table is written to disk.

    Raises:
        ValueError: If the input is not one problem's ``{mouse: {session: session}}`` with raw
            recordings, or if *previous_choice_towers* or *poke_counts* miss a mouse or session.
        PokeAlignmentError: If a session's parsed trials do not match its imported trials; this
            is likely a code bug and stops the run.

    Notes:
        A session whose raw events raise ``PokeParseError`` is skipped with a warning: it has
        zero counts and a row in ``"summary"`` whose ``parse_error`` column holds
        the reason (the long-poke means, which come from the imported trials, still include it).
        Disagreements between the raw long pokes and the printed ``long_pokes`` lists raise a
        ``UserWarning`` and are listed in the cross-check table. The last trial of each recording
        has no following initiation, so it is left out of the after-choice map.
    """
    _validate_problem(subjects_trials_by_problem)
    out_dir = Path(out_dir)
    mice = sorted(subjects_trials_by_problem)
    previous = _previous_ports(mice, previous_choice_towers)
    initiation_ports = _initiation_ports(subjects_trials_by_problem)

    per_session = (
        _check_parsed(poke_counts, subjects_trials_by_problem)
        if poke_counts is not None
        else {mouse: mouse_poke_counts(mouse, subjects_trials_by_problem[mouse]) for mouse in mice}
    )
    _warn_on_disagreement(per_session)
    counts = {
        kind: {mouse: total_counts(per_session[mouse], kind) for mouse in mice}
        for kind in ("after_initiation", "after_choice")
    }
    means = session_mean_long_pokes(subjects_trials_by_problem)
    relative = to_relative_sessions(means)
    problem_means = problem_mean_long_pokes(subjects_trials_by_problem)

    out_dir.mkdir(parents=True, exist_ok=True)

    plot_num_pokes_across_mice(
        relative,
        _long_poke_ymax(summarize_pokes_by_session(relative)),
        out_dir / LONG_POKES_STEM,
        rng=np.random.default_rng(_JITTER_SEED),
        xlabel="Relative Session",
        tick_label=relative_session_label,
    )
    current = {m: problem_choice_ports(subjects_trials_by_problem[m]) for m in mice}
    plot_maze_heatmaps_across_mice(
        counts["after_choice"],
        colorbar_label="Pokes after choice (count per port)",
        current_ports=current,
        previous_ports=previous,
        titles=_panel_titles(per_session, "after_choice"),
        suptitle=f"Pokes after choice (ITI until next initiation)\n{_SCALE_NOTE}",
        save_path=out_dir / AFTER_CHOICE_STEM,
    )
    plot_maze_heatmaps_across_mice(
        counts["after_initiation"],
        colorbar_label="Long pokes after initiation (count per port)",
        current_ports=current,
        previous_ports=previous,
        initiation_port=initiation_ports,
        titles=_panel_titles(per_session, "after_initiation"),
        suptitle=f"Long pokes after initiation (up to and including the choice)\n{_SCALE_NOTE}",
        save_path=out_dir / AFTER_INITIATION_STEM,
    )
    stems = (LONG_POKES_STEM, AFTER_CHOICE_STEM, AFTER_INITIATION_STEM)
    return {
        "long_pokes": {
            "by_session": means,
            "relative": relative,
            "problem_mean_by_mouse": problem_means,
            "across_mice": mean_se_n(list(problem_means.values())),
        },
        "counts": counts,
        "per_session": per_session,
        "cross_check": {
            mouse: {s: c.long_poke_check for s, c in by_session.items()}
            for mouse, by_session in per_session.items()
        },
        "summary": cross_check_table(per_session),
        "files": [out_dir / f"{s}.{e}" for s in stems for e in ("png", "pdf")],
    }
