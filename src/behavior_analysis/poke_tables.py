"""Tables of the poke analysis results (pure: they build DataFrames and write nothing)."""

from __future__ import annotations

from collections.abc import Collection, Mapping

import pandas as pd

from src.behavior_analysis.poke_counts import KINDS, SessionPokeCounts
from src.behavior_analysis.poke_events import ARMS9

PerMouseSession = Mapping[str, Mapping[str, SessionPokeCounts]]


def long_pokes_by_session_table(
    means: Mapping[str, Mapping[str, float]], per_session: PerMouseSession
) -> pd.DataFrame:
    """One row per mouse and session: relative session number, trial count and mean long pokes.

    Args:
        means (Mapping[str, Mapping[str, float]]): ``{mouse: {session: mean non-choice long
            pokes per trial}}`` in session order.
        per_session (PerMouseSession): ``{mouse: {session: SessionPokeCounts}}`` for the trial
            counts; must hold every mouse and session of *means*.

    Returns:
        pd.DataFrame: One row per mouse and session; columns ``mouse``, ``session``,
        ``relative_session`` (``rel_ses-01``, ... in session order), ``n_trials``,
        ``mean_long_pokes``.
    """
    rows = [
        {
            "mouse": mouse,
            "session": session,
            "relative_session": f"rel_ses-{i:02d}",
            "n_trials": per_session[mouse][session].n_trials,
            "mean_long_pokes": mean,
        }
        for mouse, by_session in means.items()
        for i, (session, mean) in enumerate(by_session.items(), start=1)
    ]
    return pd.DataFrame(
        rows, columns=["mouse", "session", "relative_session", "n_trials", "mean_long_pokes"]
    )


def counts_by_port_table(counts: Mapping[str, Mapping[str, Mapping[str, int]]]) -> pd.DataFrame:
    """One row per mouse, poke type and port from ``{kind: {mouse: {port: count}}}``.

    Args:
        counts (Mapping[str, Mapping[str, Mapping[str, int]]]): ``{kind: {mouse: {port: count}}}``
            with *kind* ``"after_initiation"`` and ``"after_choice"`` and all nine ports per mouse.

    Returns:
        pd.DataFrame: Columns ``mouse``, ``kind``, ``port``, ``count``.
    """
    rows = [
        {"mouse": mouse, "kind": kind, "port": port, "count": by_port[port]}
        for kind in KINDS
        for mouse, by_port in counts[kind].items()
        for port in ARMS9
    ]
    return pd.DataFrame(rows, columns=["mouse", "kind", "port", "count"])


def counts_by_session_table(per_session: PerMouseSession) -> pd.DataFrame:
    """One row per mouse, session, poke type and port.

    Args:
        per_session (PerMouseSession): ``{mouse: {session: SessionPokeCounts}}``.

    Returns:
        pd.DataFrame: Columns ``mouse``, ``session``, ``kind`` (``"after_initiation"`` or
        ``"after_choice"``), ``port``, ``count``.
    """
    rows = [
        {
            "mouse": mouse,
            "session": session,
            "kind": kind,
            "port": port,
            "count": getattr(counts, kind)[port],
        }
        for mouse, by_session in per_session.items()
        for session, counts in by_session.items()
        for kind in KINDS
        for port in ARMS9
    ]
    return pd.DataFrame(rows, columns=["mouse", "session", "kind", "port", "count"])


def cross_check_table(
    per_session: PerMouseSession,
    port_finding_only: Mapping[str, Collection[str]] | None = None,
) -> pd.DataFrame:
    """Trials counted, dropped and port finding, and raw-versus-printed long pokes, per session.

    Port-finding trials are not task trials: they are in the ``n_port_finding_*`` columns only,
    not in ``n_trials`` or either drop count. ``n_port_finding_trials`` is all of them,
    ``n_port_finding_counted`` those whose pokes are in the port-finding maps,
    ``n_port_finding_dropped`` those without usable pokes (timeouts, untrustworthy logs) and
    ``n_port_finding_open_after_choice_windows`` the counted ones missing from the after-choice
    map because the recording ended before the next initiation.

    ``parse_error`` is empty unless the session's raw events could not be read.

    ``port_finding_only`` marks the sessions that hold only port finding and are not sessions of
    the problem (their trials count in the port-finding maps of the problem that follows): their
    ``n_trials`` and every task count is 0.

    Args:
        per_session (PerMouseSession): ``{mouse: {session: SessionPokeCounts}}``.
        port_finding_only (Mapping[str, Collection[str]] | None): ``{mouse: sessions}`` that are
            port-finding-only. Default ``None``: none are.

    Returns:
        pd.DataFrame: One row per mouse and session; columns ``mouse``, ``session``,
        ``port_finding_only``, ``n_trials``, ``n_port_finding_trials``, ``n_port_finding_counted``,
        ``n_port_finding_dropped``, ``n_port_finding_open_after_choice_windows``,
        ``n_trials_counted``,
        ``n_trials_dropped_no_print``, ``n_trials_dropped_inconsistent``,
        ``n_open_after_choice_windows``,
        ``n_long_poke_checked``, ``n_long_poke_agree``, ``parse_error``.
    """
    port_finding_only = port_finding_only or {}
    rows = [
        {
            "mouse": mouse,
            "session": session,
            "port_finding_only": session in port_finding_only.get(mouse, ()),
            "n_trials": counts.n_trials,
            "n_port_finding_trials": counts.n_port_finding_trials,
            "n_port_finding_counted": counts.port_finding.n_valid,
            "n_port_finding_dropped": counts.port_finding.n_dropped,
            "n_port_finding_open_after_choice_windows": counts.port_finding.n_open_windows,
            "n_trials_counted": counts.n_valid,
            "n_trials_dropped_no_print": counts.n_dropped_no_print,
            "n_trials_dropped_inconsistent": counts.n_dropped_inconsistent,
            "n_open_after_choice_windows": counts.n_open_windows,
            "n_long_poke_checked": counts.long_poke_check.n_trials,
            "n_long_poke_agree": counts.long_poke_check.n_agree,
            "parse_error": counts.parse_error or "",
        }
        for mouse, by_session in per_session.items()
        for session, counts in by_session.items()
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "mouse",
            "session",
            "port_finding_only",
            "n_trials",
            "n_port_finding_trials",
            "n_port_finding_counted",
            "n_port_finding_dropped",
            "n_port_finding_open_after_choice_windows",
            "n_trials_counted",
            "n_trials_dropped_no_print",
            "n_trials_dropped_inconsistent",
            "n_open_after_choice_windows",
            "n_long_poke_checked",
            "n_long_poke_agree",
            "parse_error",
        ],
    )
