"""Poke analysis for one problem: long pokes per relative session and the two maze heatmaps.

Entry points:

- :func:`run_poke_analysis` runs everything for one problem, writes the figures to one folder and
  returns the numbers (including a per-session summary table) without writing any table file.
- :func:`previous_problem_choice_towers` and :func:`previous_choice_towers_for_problem`
  (re-exported from :mod:`src.behavior_analysis.poke_counts`) give each mouse's previous-problem
  choice ports from the full ordered problem dict; the runner passes the result of the second as
  ``previous_choice_towers``.

Figures written to ``out_dir`` (PNG, PDF and SVG, each with a caption file):

- ``Long Pokes by Relative Session``: mean non-choice long pokes per trial (``num_long_pokes``)
  for each relative session of the problem, bars = mean +/- SE across mice, lines = mice.
- ``Maze Heatmaps - Pokes After Choice`` and ``Maze Heatmaps - Pokes After Initiation``: one map
  per mouse, all sessions of the problem together.
- The same two with `` (Port Finding)`` added to the name: the pokes of the port-finding trials
  of every session of the problem, summed per mouse. A mouse without port-finding trials in the
  problem has no panel, and a figure with no panel at all is not drawn.

Port finding can span sessions: a mouse that does not finish starts the next session in port
finding again. The earlier session has no bandit trials, so it is not a session of the problem;
the caller passes it as ``port_finding_sessions`` (see
:mod:`src.behavior_analysis.port_finding_sessions`) and its trials are counted in the
port-finding maps, their titles and the summary table. The task maps and the long-poke means
never see those sessions.
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
    total_port_finding_counts,
)
from src.behavior_analysis.poke_tables import cross_check_table
from src.behavior_analysis.port_finding_check import port_finding_check_table
from src.behavior_analysis.port_finding_summary import problem_port_finding_summaries
from src.behavior_analysis.stats import mean_se_n
from src.behavior_import.problem_numbering import DEFAULT_INITIATION_TOWER
from src.behavior_import.session_order import ordered_session_keys
from src.behavior_visualization.plot_helpers import OUTPUT_FORMATS
from src.behavior_visualization.plot_maze_heatmaps import plot_maze_heatmaps_across_mice
from src.behavior_visualization.plot_num_long_pokes import (
    plot_num_pokes_across_mice,
    relative_session_label,
)
from src.pipelines.port_finding_summary import write_problem_port_finding_heatmaps

LONG_POKES_STEM = "Long Pokes by Relative Session"
AFTER_CHOICE_STEM = "Maze Heatmaps - Pokes After Choice"
AFTER_INITIATION_STEM = "Maze Heatmaps - Pokes After Initiation"
PORT_FINDING_AFTER_CHOICE_STEM = f"{AFTER_CHOICE_STEM} (Port Finding)"
PORT_FINDING_AFTER_INITIATION_STEM = f"{AFTER_INITIATION_STEM} (Port Finding)"
_SCALE_NOTE = "Colour scale: 0 to each mouse's own maximum count."
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


def _validate_port_finding_sessions(found: Mapping | None, subjects_trials: Mapping) -> dict:
    """*found* as a dict (``{}`` for ``None``), after checking it against the problem.

    Raises:
        ValueError: If it names a mouse that is not in the problem, a session that is also a task
            session of the problem, or a record without raw ``"df"`` recordings.
    """
    found = {} if found is None else found
    for mouse, records in found.items():
        if mouse not in subjects_trials:
            raise ValueError(f"port_finding_sessions names mouse {mouse!r}, not in the problem")
        clash = sorted(set(records) & set(subjects_trials[mouse]))
        if clash:
            raise ValueError(f"{mouse}: port-finding-only sessions {clash} are task sessions too")
        for key, record in records.items():
            if not (isinstance(record, Mapping) and "df" in record):
                raise ValueError(f"{mouse} {key}: port-finding-only session has no raw 'df'")
    return found


def _with_port_finding_sessions(per_session: Mapping, found: Mapping) -> dict:
    """*per_session* with the parsed port-finding-only sessions added, in session order."""
    merged = {}
    for mouse, by_session in per_session.items():
        extra = found.get(mouse, {})
        both = {**by_session, **extra}
        merged[mouse] = (
            {key: both[key] for key in ordered_session_keys(mouse, both)} if extra else by_session
        )
    return merged


def _warn_on_flagged_port_finding(check) -> None:
    for row in check[check["flagged"]].itertuples():
        warnings.warn(
            f"{row.mouse}: port-finding check flagged, {row.n_port_finding_trials} port-finding "
            f"trials (expected minimum {row.expected_min_port_finding_trials:g}): "
            f"{row.flag_reason}",
            UserWarning,
            stacklevel=3,
        )


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


def _port_finding_trial_counts(per_session, kind: str) -> dict[str, int]:
    """Port-finding trials per mouse in the *kind* map (after choice: closed windows only)."""

    def n_trials(counts: SessionPokeCounts) -> int:
        found = counts.port_finding
        return found.n_valid - (found.n_open_windows if kind == "after_choice" else 0)

    return {
        mouse: sum(n_trials(c) for c in by_session.values())
        for mouse, by_session in per_session.items()
    }


def _port_finding_totals(per_session, kind: str) -> tuple[dict, dict[str, str]]:
    """``({mouse: per-port totals}, {mouse: panel title})`` of the mice with port-finding trials."""
    n_trials = _port_finding_trial_counts(per_session, kind)
    mice = [mouse for mouse, n in n_trials.items() if n > 0]
    totals = {mouse: total_port_finding_counts(per_session[mouse], kind) for mouse in mice}
    titles = {mouse: f"{mouse} (n={n_trials[mouse]} port-finding trials)" for mouse in mice}
    return totals, titles


def _draw_port_finding_maps(
    per_session, out_dir: Path, current: dict, previous: dict | None, initiation: dict
) -> tuple[dict, list[str]]:
    """Draw the two port-finding maps that have at least one mouse.

    Returns:
        ``({kind: {mouse: per-port totals}}, [stems drawn])``; a kind no mouse has port-finding
        trials for is an empty dict and has no figure.
    """
    maps = (
        ("after_choice", PORT_FINDING_AFTER_CHOICE_STEM, "Pokes after choice (count per port)",
         "Port-finding pokes after choice (ITI until next initiation)"),
        ("after_initiation", PORT_FINDING_AFTER_INITIATION_STEM,
         "Long pokes after initiation (count per port)",
         "Port-finding long pokes after initiation (up to and including the choice)"),
    )  # fmt: skip
    totals_by_kind, drawn = {}, []
    for kind, stem, colorbar_label, heading in maps:
        totals, titles = _port_finding_totals(per_session, kind)
        totals_by_kind[kind] = totals
        if not totals:
            continue
        plot_maze_heatmaps_across_mice(
            totals,
            colorbar_label=colorbar_label,
            current_ports={m: current[m] for m in totals},
            previous_ports=None if previous is None else {m: previous[m] for m in totals},
            initiation_port={m: initiation[m] for m in totals},
            titles=titles,
            suptitle=f"{heading}\n{_SCALE_NOTE}",
            save_path=out_dir / stem,
        )
        drawn.append(stem)
    return totals_by_kind, drawn


def run_poke_analysis(
    subjects_trials_by_problem: dict,
    out_dir: Path,
    *,
    previous_choice_towers: dict | None = None,
    poke_counts: dict | None = None,
    port_finding_sessions: dict | None = None,
    summary_scope: str | None = None,
) -> dict:
    """Long pokes per relative session and the maze heatmaps for one problem.

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
        port_finding_sessions (dict | None): ``{mouse: {session_key: record}}`` of the
            port-finding-only sessions that belong to this problem (see
            :func:`~src.behavior_analysis.port_finding_sessions.port_finding_sessions_for_problem`);
            each record needs ``"df"``, and ``"initiation_tower"`` if it is not ``"B2"``. Their
            trials are added to the port-finding maps, titles and ``"summary"`` only. ``None``
            (default) adds none. Keyword-only.
        summary_scope (str | None): If given (e.g. ``"cohort-03 drifting problem-02"``), the
            port-finding summary heatmaps are written too, with this caption scope (see
            :mod:`src.pipelines.port_finding_summary`; they parse the raw events again).
            Default ``None``: not written. Keyword-only.

    Returns:
        dict: A dict with ``"long_pokes"`` (``"by_session"`` = ``{mouse: {session: mean}}``,
        ``"relative"`` = the same keyed ``rel_ses-NN``, ``"problem_mean_by_mouse"`` and
        ``"across_mice"`` = ``(mean, se, n)`` of those means), ``"counts"`` (``{kind: {mouse:
        {port: count}}}`` for ``"after_initiation"`` and ``"after_choice"``, task trials only),
        ``"port_finding_counts"`` (the same for the port-finding trials, with only the mice that
        have some; ``{}`` for a kind with none), ``"per_session"``
        (``{mouse: {session: SessionPokeCounts}}``, the port-finding-only sessions included),
        ``"cross_check"`` (the raw-versus-printed long-poke comparison per session),
        ``"summary"`` (a DataFrame with one row per mouse and session, the port-finding-only
        ones included with ``n_trials`` 0 and ``port_finding_only`` true: ``mouse, session,
        port_finding_only, n_trials, n_port_finding_trials, n_port_finding_counted,
        n_port_finding_dropped, n_port_finding_open_after_choice_windows, n_trials_counted,
        n_trials_dropped_no_print, n_trials_dropped_inconsistent, n_open_after_choice_windows,
        n_long_poke_checked, n_long_poke_agree, parse_error``), ``"port_finding_check"`` (one
        row per mouse with port-finding trials against the minimum ``n_ports x found_t_thresh``,
        see :func:`~src.behavior_analysis.port_finding_check.port_finding_check_table`) and
        ``"files"`` (the figure paths written, PNG, PDF then SVG per figure; the port-finding
        figures only if drawn) and ``"port_finding_summary"`` (``{mouse: MousePortFinding}``
        with *summary_scope*, else ``{}``). No table is written to disk.

    Raises:
        ValueError: If the input is not one problem's ``{mouse: {session: session}}`` with raw
            recordings, if *previous_choice_towers* or *poke_counts* miss a mouse or session, or
            if *port_finding_sessions* names a mouse outside the problem, a session of the
            problem, or a record without ``"df"``.
        PokeAlignmentError: If a session's parsed trials do not match its imported trials; this
            is likely a code bug and stops the run.

    Notes:
        A session whose raw events raise ``PokeParseError`` is skipped with a warning: it has
        zero counts and a row in ``"summary"`` whose ``parse_error`` column holds
        the reason (the long-poke means, which come from the imported trials, still include it).
        Disagreements between the raw long pokes and the printed ``long_pokes`` lists raise a
        ``UserWarning`` and are listed in the cross-check table. The last trial of each recording
        has no following initiation, so it is left out of the after-choice map. A flagged
        port-finding check (fewer trials than the task needs, counts carried in from an
        earlier session, or variables set by hand) raises a ``UserWarning`` and is flagged in
        ``"port_finding_check"``; the run goes on.
    """
    _validate_problem(subjects_trials_by_problem)
    found = _validate_port_finding_sessions(port_finding_sessions, subjects_trials_by_problem)
    out_dir = Path(out_dir)
    mice = sorted(subjects_trials_by_problem)
    previous = _previous_ports(mice, previous_choice_towers)
    initiation_ports = _initiation_ports(subjects_trials_by_problem)

    task_counts = (
        _check_parsed(poke_counts, subjects_trials_by_problem)
        if poke_counts is not None
        else {mouse: mouse_poke_counts(mouse, subjects_trials_by_problem[mouse]) for mouse in mice}
    )
    found_counts = {mouse: mouse_poke_counts(mouse, records) for mouse, records in found.items()}
    per_session = _with_port_finding_sessions(task_counts, found_counts)
    _warn_on_disagreement(per_session)
    check = port_finding_check_table(
        {mouse: {**subjects_trials_by_problem[mouse], **found.get(mouse, {})} for mouse in mice}
    )
    _warn_on_flagged_port_finding(check)
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
        initiation_port=initiation_ports,
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
    port_finding_counts, port_finding_stems = _draw_port_finding_maps(
        per_session, out_dir, current, previous, initiation_ports
    )
    stems = (LONG_POKES_STEM, AFTER_CHOICE_STEM, AFTER_INITIATION_STEM, *port_finding_stems)
    summaries = {}
    if summary_scope is not None:
        summaries = problem_port_finding_summaries(
            subjects_trials_by_problem,
            previous_choice_towers=previous_choice_towers,
            port_finding_sessions=found,
        )
        summary_stems = write_problem_port_finding_heatmaps(summaries, out_dir, scope=summary_scope)
        stems = (*stems, *(base.name for base in summary_stems))
    return {
        "long_pokes": {
            "by_session": means,
            "relative": relative,
            "problem_mean_by_mouse": problem_means,
            "across_mice": mean_se_n(list(problem_means.values())),
        },
        "counts": counts,
        "port_finding_counts": port_finding_counts,
        "per_session": per_session,
        "cross_check": {
            mouse: {s: c.long_poke_check for s, c in by_session.items()}
            for mouse, by_session in per_session.items()
        },
        "summary": cross_check_table(
            per_session, port_finding_only={mouse: set(records) for mouse, records in found.items()}
        ),
        "port_finding_check": check,
        "files": [out_dir / f"{s}.{e}" for s in stems for e in OUTPUT_FORMATS],
        "port_finding_summary": summaries,
    }
