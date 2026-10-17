"""Tests for the value-change diagnostic: window indices, arm labels, figure and output tree.

Every session is scripted from three arms A, B, C (C always the lowest), so the best-decrease
rank-swap events, their windows and the arm behind each label are known by hand:

====== ===================== ===========================================
trials magnitudes (A, B, C)  what happens
====== ===================== ===========================================
0-16   (2, 4, 1)             B is the best arm
17-19  (4, 2, 1)             A becomes best (best value unchanged: no event)
20-21  (1.5, 2, 1)           trial 20 is event 1 (A 4 -> 1.5, B and A exchange)
22-25  (1.5, 1.2, 1)         trial 22 is event 2 (B 2 -> 1.2, A passes B again)
26-29  (4, 2, 1)             A rises (an increase, no event)
30-35  (2.5, 3, 1)           trial 30 is event 3 (A 4 -> 2.5, B passes A)
36-39  (4, 2, 1)             A passes B again, ending event 3's post window
====== ===================== ===========================================

The animal picks A except on trial 18 (C), trial 21 (B) and trial 27 (B).
"""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from src.behavior_analysis.value_change_diagnostic import (  # noqa: E402
    contiguous_runs,
    diagnose_session,
    nearest_event_labels,
)
from src.behavior_analysis.value_change_windows import get_value_change_windows  # noqa: E402
from src.behavior_import.load_trials import ImportedTrials  # noqa: E402
from src.behavior_visualization.plot_value_change_diagnostic import (  # noqa: E402
    POST_COLOR,
    PRE_COLOR,
    event_text,
    plot_value_change_diagnostic,
)
from src.pipelines import value_change_diagnostic as pipeline  # noqa: E402
from tests.cohort_fixtures import BLOCKED_NAME, DRIFTING_NAME  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "individual_analyses"
KEY = "ses-1_date-20260101"


def _script(segments):
    return [
        dict(zip("ABC", magnitudes, strict=True))
        for first, last, magnitudes in segments
        for _ in range(last - first + 1)
    ]


EVENT_SCRIPT = _script(
    [
        (0, 16, (2, 4, 1)),
        (17, 19, (4, 2, 1)),
        (20, 21, (1.5, 2, 1)),
        (22, 25, (1.5, 1.2, 1)),
        (26, 29, (4, 2, 1)),
        (30, 35, (2.5, 3, 1)),
        (36, 39, (4, 2, 1)),
    ]
)
EVENT_CHOICES = ["A"] * 40
EVENT_CHOICES[18], EVENT_CHOICES[21], EVENT_CHOICES[27] = "C", "B", "B"
NO_EVENT_SCRIPT = [{"A": 4, "B": 2, "C": 1}] * 30


def make_session(rms, choices=None, task_name=DRIFTING_NAME):
    """A session dict with trial variables and the per-tower series the session plot reads."""
    n = len(rms)
    choices = (choices or ["A", "B", "C"] * (n // 3 + 1))[:n]
    return {
        "task_name": task_name,
        "trial_variables": [{"reward_magnitudes": rms, "choice": choices, "blocks": [1] * n}],
        "reward_magnitudes_by_tower": {arm: [m[arm] for m in rms] for arm in "ABC"},
        "choices_by_tower": {arm: [int(c == arm) for c in choices] for arm in "ABC"},
        "blocks": [1] * n,
    }


@pytest.fixture
def event_session():
    return make_session(EVENT_SCRIPT, EVENT_CHOICES)


# ---------------------------------------------------------------- the pure helper


def test_events_are_found_in_trial_order_with_known_t0(event_session):
    diagnostics = diagnose_session("m", KEY, event_session)
    assert [d.trial_idx for d in diagnostics] == [20, 22, 30]
    assert [d.number for d in diagnostics] == [1, 2, 3]


def test_arms_follow_the_ranks_at_t_minus_one(event_session):
    first, second, third = diagnose_session("m", KEY, event_session)
    assert first.arms == {"prev_best": "A", "prev_second": "B", "third": "C"}
    # At trial 21 B is the best arm, so the names swap relative to event 1.
    assert second.arms == {"prev_best": "B", "prev_second": "A", "third": "C"}
    assert third.arms == {"prev_best": "A", "prev_second": "B", "third": "C"}


def test_pre_window_is_cut_where_the_best_arm_last_changed(event_session):
    first, second, third = diagnose_session("m", KEY, event_session)
    # B was best until trial 16, so event 1 reaches back only to trials 17-19.
    assert first.pre_idx == (17, 18, 19)
    # Event 2: B has been best only since trial 20, so just trials 20-21.
    assert second.pre_idx == (20, 21)
    # Event 3: A has been best for 8 trials, so the full 5 requested.
    assert third.pre_idx == (25, 26, 27, 28, 29)


def test_post_window_is_truncated_at_the_first_best_second_exchange(event_session):
    first, second, third = diagnose_session("m", KEY, event_session)
    # Event 1: A passes B at trial 22, so the window ends at trial 21.
    assert first.post_idx == (20, 21)
    assert first.truncated and first.swap_offset == 2
    # Event 2: B passes A at trial 30 (offset 8).
    assert second.post_idx == tuple(range(22, 30))
    # Event 3: A passes B at trial 36 (offset 6).
    assert third.post_idx == tuple(range(30, 36))


def test_labels_name_the_arm_chosen_on_each_trial(event_session):
    first, second, _ = diagnose_session("m", KEY, event_session)
    assert first.labels_by_trial == {
        17: "prev_best",
        18: "third",
        19: "prev_best",
        20: "prev_best",
        21: "prev_second",
    }
    # Trials 20-21 seen from event 2: A is now the prev second arm.
    assert (second.labels_by_trial[20], second.labels_by_trial[21]) == ("prev_second", "prev_best")


def test_shaded_indices_are_exactly_the_non_none_bins(event_session):
    windows = get_value_change_windows(
        {"m": {KEY: event_session}}, "best", "decrease", "rank_swap",
        pre=5, post=15, truncate_at_rank_swap=True,
    )  # fmt: skip
    diagnostics = diagnose_session("m", KEY, event_session)
    for event, diagnostic in zip(windows.events["m"], diagnostics, strict=True):
        t0 = event.trial_idx
        assert diagnostic.pre_idx == tuple(
            t0 - 5 + p for p, label in enumerate(event.pre) if label is not None
        )
        assert diagnostic.post_idx == tuple(
            t0 + p for p, label in enumerate(event.post) if label is not None
        )


def test_lengths_used_count_shaded_trials_around_t0(event_session):
    first, _, third = diagnose_session("m", KEY, event_session)
    assert (first.pre_used, first.post_used) == (3, 1)
    assert (third.pre_used, third.post_used) == (5, 5)
    assert (first.pre_requested, first.post_requested) == (5, 15)


def test_a_session_without_events_gives_no_diagnostics():
    assert diagnose_session("m", KEY, make_session(NO_EVENT_SCRIPT)) == []


def test_contiguous_runs_merge_neighbouring_indices():
    assert contiguous_runs((3, 4, 5, 9, 11, 12)) == [(3, 3), (9, 1), (11, 2)]
    assert contiguous_runs(()) == []


def test_nearest_event_labels_prefer_the_event_whose_t0_is_closest(event_session):
    labels = nearest_event_labels(diagnose_session("m", KEY, event_session))
    # Trial 21 is one step after event 1 and one step before event 2: the earlier event wins.
    assert labels[21] == "prev_second"
    # Trial 20 is event 1's t=0 and two steps before event 2's.
    assert labels[20] == "prev_best"
    # Trial 22 is event 2's t=0; event 1's window no longer reaches it.
    assert labels[22] == "prev_second"


def test_mismatched_series_lengths_raise(event_session):
    series = event_session["reward_magnitudes_by_tower"]
    series["A"] = series["A"][:-1]
    with pytest.raises(ValueError, match="trials"):
        diagnose_session("m", KEY, event_session)


# ---------------------------------------------------------------- the figure


def test_event_text_gives_trial_arms_and_lengths(event_session):
    first = diagnose_session("m", KEY, event_session)[0]
    assert event_text(first) == (
        "#1 t=0 at trial 20: prev best = A, prev second = B, third = C | "
        "pre 3 of 5 trials, post 1 of 15 trials, truncated at swap (t=+2)"
    )


def test_event_text_says_when_the_window_was_not_truncated(event_session):
    third = diagnose_session("m", KEY, event_session)[2]
    assert event_text(third).endswith("truncated at swap (t=+6)")
    session = make_session(
        EVENT_SCRIPT[:36], EVENT_CHOICES[:36]
    )  # the swap that cut event 3 is never reached
    last = diagnose_session("m", KEY, session)[2]
    assert event_text(last).endswith("not truncated")


@pytest.fixture
def event_figure(event_session):
    diagnostics = diagnose_session("m", KEY, event_session)
    fig = plot_value_change_diagnostic(event_session, diagnostics, title="m | " + KEY)
    yield fig
    plt.close(fig)


def test_figure_has_main_strip_and_text_axes_with_one_band_per_event(event_figure):
    main, strip, text = event_figure.axes
    # bottom to top: event 1 is the top band
    assert [t.get_text() for t in strip.get_yticklabels()] == ["#3", "#2", "#1"]
    assert main.get_xlim() == strip.get_xlim()
    assert any("#3 t=0 at trial 30" in t.get_text() for t in text.texts)


def test_strip_shades_every_event_in_pre_and_post_colours(event_figure):
    strip = event_figure.axes[1]
    bars = [c for c in strip.collections if isinstance(c, PolyCollection)]
    colours = [tuple(c.get_facecolor()[0]) for c in bars]
    expected = matplotlib.colors.to_rgba
    assert colours.count(expected(PRE_COLOR)) == 3
    assert colours.count(expected(POST_COLOR)) == 3
    assert PRE_COLOR != POST_COLOR


def test_strip_bars_cover_the_window_indices(event_figure):
    strip = event_figure.axes[1]
    pre_bars = [
        c for c in strip.collections
        if isinstance(c, PolyCollection)
        and tuple(c.get_facecolor()[0]) == matplotlib.colors.to_rgba(PRE_COLOR)
    ]  # fmt: skip
    # Event 1 (drawn first, top band) has pre bins 17-19, which span x = 16.5 .. 19.5.
    x = np.concatenate([path.vertices[:, 0] for path in pre_bars[0].get_paths()])
    assert (x.min(), x.max()) == (16.5, 19.5)


def test_choice_markers_in_windows_are_coloured_by_label(event_figure):
    main = event_figure.axes[0]
    overlay = next(c for c in main.collections if c.get_label() == "prev best")
    xs = {round(x) for x, _ in overlay.get_offsets()}
    assert {17, 19, 20}.issubset(xs)
    assert 18 not in xs  # trial 18 is a "third" choice


def test_session_without_events_still_gets_a_labelled_figure():
    session = make_session(NO_EVENT_SCRIPT)
    fig = plot_value_change_diagnostic(session, [], title="m | none")
    assert any("no best-decrease rank-swap events" in t.get_text() for t in fig.axes[-1].texts)
    plt.close(fig)


# ---------------------------------------------------------------- the output tree


def two_problem_cohort():
    """Problem 1 has events (m1 ses-1) and none; problem 2 is blocked and must be skipped."""
    return {
        1: {
            "m1": {
                KEY: make_session(EVENT_SCRIPT, EVENT_CHOICES),
                "ses-2_date-20260102": make_session(NO_EVENT_SCRIPT),
            },
            "m2": {KEY: make_session(NO_EVENT_SCRIPT)},
        },
        2: {"m1": {"ses-3_date-20260103": make_session(NO_EVENT_SCRIPT, task_name=BLOCKED_NAME)}},
    }


@pytest.fixture
def written_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "load_trials_with_log",
        lambda *args, **kwargs: ImportedTrials(two_problem_cohort(), log=None),
    )
    pipeline.run_cohort_diagnostic(
        3, data_root=tmp_path / "data", results_root=tmp_path / "results"
    )
    return tmp_path / "results" / "figures" / "value-change-diagnostic" / "cohort-03"


def test_tree_has_one_png_per_drifting_session_and_an_index(written_tree):
    found = sorted(str(p.relative_to(written_tree)) for p in written_tree.rglob("*") if p.is_file())
    assert found == [
        "index.txt",
        "problem-01/m1/ses-1_date-20260101.png",
        "problem-01/m1/ses-2_date-20260102.png",
        "problem-01/m2/ses-1_date-20260101.png",
    ]


def test_pngs_are_real_images(written_tree):
    png = written_tree / "problem-01/m2/ses-1_date-20260101.png"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_index_lists_sessions_events_and_t0_trials(written_tree):
    lines = (written_tree / "index.txt").read_text().splitlines()
    text = "\n".join(lines)
    assert "ses-1_date-20260101: 3 events, t=0 at trials 20, 22, 30" in text
    assert "ses-2_date-20260102: no best-decrease rank-swap events" in text
    assert "problem-02" not in text


def test_cohort_without_drifting_problems_raises(tmp_path, monkeypatch):
    blocked_only = {1: {"m1": {KEY: make_session(NO_EVENT_SCRIPT, task_name=BLOCKED_NAME)}}}
    monkeypatch.setattr(
        pipeline, "load_trials_with_log", lambda *a, **k: ImportedTrials(blocked_only, log=None)
    )
    with pytest.raises(ValueError, match="no drifting"):
        pipeline.run_cohort_diagnostic(3, data_root=tmp_path, results_root=tmp_path / "r")


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "plot_value_change_diagnostic_script", SCRIPT_DIR / "plot_value_change_diagnostic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_runs_each_cohort_with_the_cache_flag(monkeypatch):
    module = _load_script()
    calls = []
    monkeypatch.setattr(
        module, "run_cohort_diagnostic", lambda cohort, **kw: calls.append((cohort, kw))
    )
    module.main(["--cohorts", "1", "3", "--no-cache"])
    assert [cohort for cohort, _ in calls] == [1, 3]
    assert all(kw["use_cache"] is False for _, kw in calls)
    module.main(["--cohorts", "2"])
    assert calls[-1][1]["use_cache"] is True


def test_script_requires_cohorts():
    with pytest.raises(SystemExit):
        _load_script().main([])


def test_index_entry_uses_the_singular_for_one_event(event_session):
    one = diagnose_session("m", KEY, event_session)[:1]
    assert pipeline.index_entry(KEY, one) == "ses-1_date-20260101: 1 event, t=0 at trial 20"
