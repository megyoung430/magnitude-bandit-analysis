"""The drifting-only entry points refuse counter-format data and speak of block changes."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.behavior_analysis.reversal_aligned_arms import (
    build_windows_with_magnitudes,
    compute_reversal_aligned_arms,
)
from src.behavior_analysis.reversal_format import (
    check_block_change_windows,
    check_blocks_only_sessions,
)
from src.behavior_visualization.plot_reversal_aligned_arms import plot_reversal_aligned_arms
from src.pipelines.reversal_aligned_arms import run_reversal_aligned_arm_analysis
from src.pipelines.value_change_analysis import run_value_change_analysis
from src.pipelines.value_psychometrics import run_value_psychometric_analysis
from tests.plot_capture import capture_figures
from tests.plot_fixtures import make_mouse
from tests.reversal_arm_fixtures import make_reversal, stamp_windows
from tests.test_value_change_windows import BASE, key, mag, recording, script, session


@pytest.fixture(autouse=True)
def no_rendering(monkeypatch):
    capture_figures(monkeypatch)
    yield
    plt.close("all")


def counter_problems():
    """One drifting-looking problem whose mouse "bad" logs the good/bad reversal counters."""
    return {3: {"ok": make_mouse(1, counters=False), "bad": make_mouse(2, counters=True)}}


def blocks_only_problems():
    return {3: {"ok": make_mouse(1, counters=False), "other": make_mouse(2, counters=False)}}


def with_trials_and_counters(sessions):
    """Mark trial-variable sessions (which carry no ``trial`` series) as counter format."""
    return {k: {**s, "trial": [1, 2, 3], "has_good": True} for k, s in sessions.items()}


def value_sessions():
    rms = script(60, BASE, {10: mag(5, 2, 1)})
    return {key(1): session(recording(rms))}


# ---------------------------------------------------------------- the shared helper


def test_helper_accepts_blocks_only_sessions():
    check_blocks_only_sessions(blocks_only_problems(), "some analysis")


def test_helper_names_analysis_mouse_session_and_the_task_fact():
    with pytest.raises(ValueError) as error:
        check_blocks_only_sessions(counter_problems(), "some analysis")
    message = str(error.value)
    assert "some analysis" in message
    assert "'bad'" in message and "ses-1_date-20260901" in message
    assert "drifting task has only block changes" in message


def test_helper_ignores_sessions_without_trials():
    problems = {1: {"m": {"ses-1": {"has_good": True, "trial": []}}}}
    check_blocks_only_sessions(problems, "some analysis")


def test_window_helper_requires_the_blocks_boundary_source():
    check_block_change_windows({"m": [{"boundary_source": "blocks"}]}, "some analysis")
    with pytest.raises(ValueError, match="'m'.*good_reversals.*block changes"):
        check_block_change_windows({"m": [{"boundary_source": "good_reversals"}]}, "x")
    with pytest.raises(ValueError, match="no boundary_source"):
        check_block_change_windows({"m": [{}]}, "x")


# ---------------------------------------------------------------- the three entry points


def test_psychometrics_raises_on_a_counter_session(tmp_path):
    problems = {1: {"m": with_trials_and_counters(value_sessions())}}
    with pytest.raises(ValueError, match="'m'.*drifting task has only block changes"):
        run_value_psychometric_analysis(problems, tmp_path)


def test_value_change_raises_on_a_counter_session(tmp_path):
    problems = {1: {"m": with_trials_and_counters(value_sessions())}}
    with pytest.raises(ValueError, match="'m'.*drifting task has only block changes"):
        run_value_change_analysis(problems, tmp_path)


def test_reversal_aligned_arms_raises_on_counter_trials(tmp_path):
    with pytest.raises(ValueError, match="'bad'.*drifting task has only block changes"):
        run_reversal_aligned_arm_analysis(counter_problems(), tmp_path, pre=10, post=6)


def test_reversal_aligned_arms_raises_on_windows_from_counters(tmp_path):
    windows = {
        3: build_windows_with_magnitudes(make_mouse_dict(counters=True), pre=10, post=6),
    }
    with pytest.raises(ValueError, match="good_reversals"):
        run_reversal_aligned_arm_analysis(windows, tmp_path, pre=10, post=6)


def make_mouse_dict(counters):
    return {"m": make_mouse(1, counters=counters)}


def test_blocks_only_data_runs_through_every_entry_point(tmp_path):
    value_problems = {1: {"m": value_sessions()}}
    run_value_change_analysis(value_problems, tmp_path / "vc", truncate_modes=(True,))
    run_value_psychometric_analysis(
        {
            1: {
                "m": {
                    "ses-1": {
                        "trial_variables": [
                            {
                                "reward_magnitudes": [BASE] * 4,
                                "chosen_rank": ["best"] * 4,
                                "choice": ["A"] * 4,
                            }
                        ]
                    }
                }
            }
        },
        tmp_path / "vp",
        value_bases=("true",),
        pooling_modes=(True,),
        tie_modes=("exclude",),
    )
    result = run_reversal_aligned_arm_analysis(
        blocks_only_problems(), tmp_path / "ra", pre=10, post=6, reversal_types=("all",)
    )
    assert result["results"]["pooled"]["prev"]["all"].num_reversals > 0


# ---------------------------------------------------------------- windows and wording


def test_windows_of_blocks_only_data_come_from_the_blocks_series():
    windows = build_windows_with_magnitudes(make_mouse_dict(counters=False), pre=10, post=6)
    sources = {r["boundary_source"] for r in windows["m"]}
    assert sources == {"blocks"}


def test_figure_title_and_axis_say_block_change():
    windows = {"m0": [make_reversal(["A"] * 4, ["C"] * 3)]}
    windows = stamp_windows(windows, pre=4, post=3)
    result = compute_reversal_aligned_arms(windows, scheme="prev", pre=4, post=3)
    ax, _ = plot_reversal_aligned_arms(result)
    assert "Around Block Changes (All)" in ax.get_title()
    assert "n=1 block changes" in ax.get_title()
    assert "before the block change" in ax.get_title()
    assert ax.get_xlabel() == "Trials from Block Change"
    assert "eversal" not in ax.get_title()
