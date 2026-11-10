"""Regression-extension figures: file names, short titles with the statistics line, captions."""

import re

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.behavior_analysis import choice_logit_extensions as ext  # noqa: E402
from src.behavior_visualization import plot_helpers  # noqa: E402
from src.behavior_visualization import plot_regression_extensions as plots  # noqa: E402
from src.pipelines import regression_extensions as rx  # noqa: E402
from tests.test_plot_choice_logit import check_files  # noqa: E402
from tests.test_plot_null_agents import legends_over_panels  # noqa: E402

STATS_LINE = re.compile(r"mean ± se across mice, n=\d+ mice")
PER_MOUSE_LINE = re.compile(r"per mouse, n=\d+ mice")


def mice_of(n, prefix="m"):
    return [f"{prefix}{i}" for i in range(1, n + 1)]


def believed_table(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "mouse": mice_of(n),
            "believed_weight": rng.normal(0.2, 0.05, n),
            "unvisited_weight": rng.normal(-0.3, 0.1, n),
            "gain_believed": rng.normal(0.01, 0.005, n),
            "gain_no_reward_believed": rng.normal(-0.05, 0.01, n),
        }
    )


def binary_table(n, shift=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "mouse": mice_of(n),
            "gain_binary": rng.normal(shift, 0.01, n),
            "gain_binary_pulse": rng.normal(0.003, 0.002, n),
        }
    )


def partition_table(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "mouse": mice_of(n),
            "unique_tower": rng.normal(0.01, 0.005, n),
            "unique_reward": rng.normal(0.03, 0.01, n),
            "unique_choice": rng.normal(0.06, 0.01, n),
            "shared_reward_choice": rng.normal(0.02, 0.01, n),
        }
    )


def calibration_tables(n, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for model, sharpness in (("full", 1.0), ("choice only", 0.5)):
        probabilities = rng.dirichlet([sharpness] * 3, size=60 * n)
        chosen = (rng.random(60 * n)[:, None] > probabilities.cumsum(axis=1)).sum(axis=1)
        mice = np.repeat(mice_of(n), 60).astype(object)
        out[model] = ext.calibration_table(probabilities, chosen, mice, n_bins=5)
    return out


@pytest.fixture
def saved_figures(monkeypatch):
    """``{figure name: (suptitle, panel titles, legend texts, legends over data)}``.

    Recorded after the save step, which moves legends out of the data and drops repeats.
    """
    seen = {}
    real = plot_helpers._write_figure_files

    def spy(fig, base, formats, dpi):
        result = real(fig, base, formats, dpi)
        titles = [ax.get_title() for ax in fig.axes if ax.get_visible() and ax.get_title()]
        legends = [*fig.legends, *(ax.get_legend() for ax in fig.axes if ax.get_legend())]
        texts = [t.get_text() for legend in legends for t in legend.get_texts()]
        seen[str(base).rsplit("/", 1)[-1]] = (
            fig.get_suptitle(), titles, texts, legends_over_panels(fig),
        )  # fmt: skip
        return result

    monkeypatch.setattr(plot_helpers, "_write_figure_files", spy)
    return seen


def check_one_title(seen, name):
    suptitle, panels, legend, over = seen[name]
    assert "\n" not in suptitle
    assert over == []
    return panels, legend


def test_believed_value_figure(tmp_path, saved_figures):
    sources = {"Mice": believed_table(5), "Q-Learner": believed_table(6, 1)}
    base = tmp_path / "Believed Value"
    plots.plot_believed_value(sources, base)
    check_files(base)
    panels, _ = check_one_title(saved_figures, "Believed Value")
    assert len(panels) == 4
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels), panels
    caption = base.with_name("Believed Value.caption.txt").read_text()
    assert "never-visited" in caption.lower()
    assert "positive" in caption.lower()
    assert "Mice: n=5 mice" in caption and "Q-Learner: n=6 mice" in caption


def test_binary_vs_pulse_with_enough_mice_shows_mean_se_and_tests(tmp_path, saved_figures):
    sources = {"Mice": binary_table(6), "WSLS": binary_table(6, 0.03, 1)}
    base = tmp_path / "Binary vs Pulse"
    plots.plot_binary_vs_pulse(sources, base)
    check_files(base)
    panels, _ = check_one_title(saved_figures, "Binary vs Pulse")
    assert len(panels) == 2
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels), panels
    caption = base.with_name("Binary vs Pulse.caption.txt").read_text()
    assert "Wilcoxon" in caption and "sign test" in caption.lower()


def test_binary_vs_pulse_with_few_mice_shows_mice_individually_and_says_so(tmp_path, saved_figures):
    sources = {"Mice": binary_table(3)}
    base = tmp_path / "Binary vs Pulse"
    plots.plot_binary_vs_pulse(sources, base)
    panels, _ = check_one_title(saved_figures, "Binary vs Pulse")
    assert all(PER_MOUSE_LINE.search(p.split("\n")[-1]) for p in panels), panels
    assert not any("±" in p for p in panels)
    caption = base.with_name("Binary vs Pulse.caption.txt").read_text()
    assert "fewer than 5 mice" in caption


def test_binary_vs_pulse_draws_no_mean_for_a_source_with_few_mice(tmp_path):
    sources = {"Mice": binary_table(3), "WSLS": binary_table(6, 0.03, 1)}
    plots.plot_binary_vs_pulse(sources, tmp_path / "x")
    caption = (tmp_path / "x.caption.txt").read_text()
    assert "Mice: n=3 mice, per-mouse values only" in caption


def test_variance_partition_figure_has_one_panel_per_source_on_a_common_axis(
    tmp_path, saved_figures
):
    sources = {
        "Mice": partition_table(4),
        "Q-Learner": partition_table(5, 1),
        "WSLS": partition_table(5, 2),
    }
    base = tmp_path / "Unique and Shared Variance"
    plots.plot_variance_partition(sources, base)
    check_files(base)
    panels, legend = check_one_title(saved_figures, "Unique and Shared Variance")
    assert len(panels) == 3
    assert [p.split("\n")[0] for p in panels] == ["Mice", "Q-Learner", "WSLS"]
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels), panels
    caption = base.with_name("Unique and Shared Variance.caption.txt").read_text()
    assert "shared" in caption.lower() and "tower only" in caption.lower()
    assert "m1" in legend  # per-mouse colors are explained once


def test_calibration_figure_has_a_full_and_a_choice_history_panel(tmp_path, saved_figures):
    base = tmp_path / "Calibration (Mice)"
    plots.plot_calibration("Mice", calibration_tables(4), base)
    check_files(base)
    panels, _ = check_one_title(saved_figures, "Calibration (Mice)")
    assert [p.split("\n")[0] for p in panels] == ["Full Model", "Choice History Only"]
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels), panels
    caption = base.with_name("Calibration (Mice).caption.txt").read_text()
    assert "identity" in caption.lower() and "mean |observed - predicted|" in caption


def test_calibration_y_axis_is_the_observed_fraction_chosen(tmp_path, monkeypatch):
    seen = {}
    real = plots.save_or_show

    def spy(fig, path):
        seen["axes"] = list(fig.axes)
        return real(fig, path)

    monkeypatch.setattr(plots, "save_or_show", spy)
    tables = calibration_tables(3)
    plots.plot_calibration("Mice", tables, tmp_path / "c")
    left = seen["axes"][0]
    assert left.get_ylabel() == "Observed Fraction Chosen"
    assert left.get_xlabel() == "Mean Predicted P(Choice)"
    identity = [ln for ln in left.get_lines() if list(ln.get_xdata()) == [0, 1]]
    assert identity and list(identity[0].get_ydata()) == [0, 1]
    # the mean curve's y values are the mean over mice of the observed fractions of each bin
    expected = tables["full"].groupby("bin")["observed"].mean().to_numpy()
    drawn = [ln.get_ydata() for ln in left.get_lines() if len(ln.get_ydata()) == len(expected)]
    assert any(np.allclose(y, expected) for y in drawn)


def test_every_figure_needs_data(tmp_path):
    with pytest.raises(ValueError, match="no sources"):
        plots.plot_believed_value({}, tmp_path / "x")
    with pytest.raises(ValueError, match="no sources"):
        plots.plot_binary_vs_pulse({}, tmp_path / "x")
    with pytest.raises(ValueError, match="no sources"):
        plots.plot_variance_partition({}, tmp_path / "x")
    with pytest.raises(ValueError, match="no mice"):
        plots.plot_variance_partition({"Mice": partition_table(3).iloc[:0]}, tmp_path / "x")
    with pytest.raises(ValueError, match="no calibration"):
        plots.plot_calibration("Mice", {}, tmp_path / "x")


def test_plot_extensions_writes_every_figure_into_the_folder(tmp_path):
    tables = {
        label: rx.ExtensionTables(
            believed=believed_table(5, k),
            binary_pulse=binary_table(5, seed=k),
            partition=partition_table(5, k),
            calibration=calibration_tables(5, k),
            calibration_summary={},
        )
        for k, label in enumerate(("Mice", "Q-Learner"))
    }
    plots.plot_extensions(tables, tmp_path / "regression-extensions")
    folder = tmp_path / "regression-extensions"
    expected = [
        "Believed Value", "Binary vs Pulse", "Unique and Shared Variance",
        "Calibration (Mice)", "Calibration (Q-Learner)",
    ]  # fmt: skip
    for name in expected:
        for suffix in ("pdf", "png", "svg", "caption.txt"):
            assert (folder / f"{name}.{suffix}").exists(), (name, suffix)
