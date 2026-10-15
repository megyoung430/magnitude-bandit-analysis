"""Tables of the poke analysis results (pure; the pipeline writes them to CSV)."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.behavior_analysis.poke_counts import KINDS, SessionPokeCounts
from src.behavior_analysis.poke_events import ARMS9

PerMouseSession = Mapping[str, Mapping[str, SessionPokeCounts]]


def long_pokes_by_session_table(
    means: Mapping[str, Mapping[str, float]], per_session: PerMouseSession
) -> pd.DataFrame:
    """One row per mouse and session: relative session number, trial count and mean long pokes.

    Args:
        means: ``{mouse: {session: mean non-choice long pokes per trial}}`` in session order.
        per_session: ``{mouse: {session: SessionPokeCounts}}`` for the trial counts.
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
    """One row per mouse, poke type and port from ``{kind: {mouse: {port: count}}}``."""
    rows = [
        {"mouse": mouse, "kind": kind, "port": port, "count": by_port[port]}
        for kind in KINDS
        for mouse, by_port in counts[kind].items()
        for port in ARMS9
    ]
    return pd.DataFrame(rows, columns=["mouse", "kind", "port", "count"])


def counts_by_session_table(per_session: PerMouseSession) -> pd.DataFrame:
    """One row per mouse, session, poke type and port."""
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


def cross_check_table(per_session: PerMouseSession) -> pd.DataFrame:
    """Raw-event long pokes against the printed lists, one row per mouse and session."""
    rows = [
        {
            "mouse": mouse,
            "session": session,
            "n_trials": counts.n_trials,
            "n_trials_counted": counts.n_valid,
            "n_trials_dropped_no_print": counts.n_dropped_no_print,
            "n_trials_dropped_inconsistent": counts.n_dropped_inconsistent,
            "n_open_after_choice_windows": counts.n_open_windows,
            "n_long_poke_checked": counts.long_poke_check.n_trials,
            "n_long_poke_agree": counts.long_poke_check.n_agree,
        }
        for mouse, by_session in per_session.items()
        for session, counts in by_session.items()
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "mouse",
            "session",
            "n_trials",
            "n_trials_counted",
            "n_trials_dropped_no_print",
            "n_trials_dropped_inconsistent",
            "n_open_after_choice_windows",
            "n_long_poke_checked",
            "n_long_poke_agree",
        ],
    )
