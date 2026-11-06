"""Choice-logit figures: file names, short titles with the statistics line, and captions."""

import re

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.behavior_analysis import model_value_psychometrics as mvp  # noqa: E402
from src.behavior_visualization import plot_choice_logit as plots  # noqa: E402
from src.behavior_visualization import plot_helpers  # noqa: E402

STATS_LINE = re.compile(r"(mean|median)( ± (se|std))? across mice, n=\d+ mice")
EXTENSIONS = ("pdf", "png", "svg", "caption.txt")


@pytest.fixture
def titles(monkeypatch):
    """``{figure base name: (suptitle, [panel titles])}`` of every figure saved in the test."""
    seen = {}
    real = plot_helpers._write_figure_files

    def spy(fig, base, formats, dpi):
        panel_titles = [ax.get_title() for ax in fig.axes if ax.get_visible() and ax.get_title()]
        seen[str(base).rsplit("/", 1)[-1]] = (fig.get_suptitle(), panel_titles)
        return real(fig, base, formats, dpi)

    monkeypatch.setattr(plot_helpers, "_write_figure_files", spy)
    return seen


def weights_table(n_mice=4, n_lags=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_mice):
        for group in ("reward", "choice"):
            for lag in range(1, n_lags + 1):
                rows.append((f"m{m}", f"{group}_lag{lag}", group, lag, rng.normal(0.5 / lag, 0.05)))
        for tower in ("B3", "C2"):
            rows.append((f"m{m}", f"tower_{tower}", "tower", np.nan, rng.normal(0, 0.2)))
    return pd.DataFrame(rows, columns=["mouse", "feature", "group", "lag", "weight"])


def check_files(base):
    for ext in EXTENSIONS:
        assert base.with_name(f"{base.name}.{ext}").exists(), ext


def check_titles(titles, name, n_panels, n_without_stats=0):
    """Panels show a heading and the statistics line, except *n_without_stats* scatter panels."""
    suptitle, panels = titles[name]
    assert "\n" not in suptitle
    assert len(panels) == n_panels
    with_stats = [t for t in panels if STATS_LINE.search(t.split("\n")[-1])]
    assert len(with_stats) == n_panels - n_without_stats, panels


def test_kernel_figure(tmp_path, titles):
    base = tmp_path / "kernels"
    plots.plot_kernels(weights_table(), base)
    check_files(base)
    check_titles(titles, "kernels", 2)
    caption = base.with_name("kernels.caption.txt").read_text()
    assert "thin lines" in caption.lower()


def test_kernel_figure_with_reference_adds_the_tower_recovery_panel(tmp_path, titles):
    base = tmp_path / "kernels-ref"
    weights = weights_table()
    reference = weights.assign(weight=weights["weight"] + 0.05)
    plots.plot_kernels(weights, base, reference=reference, reference_label="Generating Rule")
    check_files(base)
    check_titles(titles, "kernels-ref", 3, n_without_stats=1)
    assert "generating rule" in base.with_name("kernels-ref.caption.txt").read_text().lower()


def test_kernel_figure_needs_weights(tmp_path):
    with pytest.raises(ValueError, match="no weights"):
        plots.plot_kernels(weights_table().iloc[:0], tmp_path / "x")


def comparison_table(n_mice=4):
    rows = []
    for m in range(n_mice):
        for model, nats in (("reward", 1.05), ("reward+choice", 1.0), ("full", 0.98)):
            rows.append((f"m{m}", model, 1000, nats + 0.01 * m, np.log(3) - nats - 0.01 * m))
    return pd.DataFrame(
        rows, columns=["mouse", "model", "n_trials", "nats_per_choice", "gain_over_chance"]
    )


def test_model_comparison_figure(tmp_path, titles):
    base = tmp_path / "comparison"
    plots.plot_model_comparison(comparison_table(), base)
    check_files(base)
    check_titles(titles, "comparison", 1)
    caption = base.with_name("comparison.caption.txt").read_text()
    assert "1.0986" in caption  # chance level, log(3) nats per choice


def curves_for(n_mice=3, seed=0):
    rng = np.random.default_rng(seed)
    n = 300 * n_mice
    values = rng.normal(size=(n, 3))
    chosen = np.where(rng.random(n) < 0.6, values.argmax(axis=1), rng.integers(0, 3, n))
    table = mvp.rank_table(np.repeat([f"m{i}" for i in range(n_mice)], 300), values, chosen)
    return mvp.binned_psychometric_curves(table, n_bins=4), table


def test_psychometric_rows_figure(tmp_path, titles):
    curves, _ = curves_for()
    rows = [
        plots.PsychometricRow("Model Value", curves, plots.value_labels("Model Value")),
        plots.PsychometricRow("Agent Q-Value", curves, plots.value_labels("Agent Q-Value")),
    ]
    base = tmp_path / "psychometrics"
    plots.plot_psychometric_rows(rows, "Psychometric Curves", ("A caption line.",), base)
    check_files(base)
    check_titles(titles, "psychometrics", 6)
    assert "A caption line." in base.with_name("psychometrics.caption.txt").read_text()


def test_psychometric_rows_need_a_row(tmp_path):
    with pytest.raises(ValueError, match="no rows"):
        plots.plot_psychometric_rows([], "x", (), tmp_path / "x")


def pairs_frame(seed=0):
    rng = np.random.default_rng(seed)
    n = 600
    meta = pd.DataFrame(
        dict(
            mouse=np.repeat(["a", "b", "c"], n // 3),
            problem=1,
            trial=np.tile(np.arange(n // 3), 3),
            trial_in_session=np.tile(np.arange(n // 3) % 40, 3),
            after_gap=False,
            reward=rng.integers(0, 5, n),
        )
    )
    return mvp.stay_switch_pairs_from_values(
        meta, rng.normal(size=(n, 3)), rng.integers(0, 3, n), n_bins=4
    )


@pytest.mark.parametrize(
    "name, function",
    [("stay", plots.plot_stay_switch_by_reward), ("target", plots.plot_switch_targets_by_reward)],
)
def test_stay_switch_figures(tmp_path, titles, name, function):
    pairs = pairs_frame()
    by_reward = {
        "stay": mvp.stay_switch_curves_by_reward,
        "target": mvp.switch_target_curves_by_reward,
    }[name]
    curves = by_reward(pairs, pool_trials=True)
    base = tmp_path / name
    function(curves, "Model Value", True, base)
    check_files(base)
    check_titles(titles, name, 6)
    caption = base.with_name(f"{name}.caption.txt").read_text()
    assert "rewarded" in caption.lower()


def test_stay_switch_figure_keeps_an_empty_row(tmp_path, titles):
    pairs = pairs_frame()
    curves = mvp.stay_switch_curves_by_reward(pairs[pairs["rewarded"]], pool_trials=True)
    base = tmp_path / "empty-row"
    plots.plot_stay_switch_by_reward(curves, "Model Value", True, base)
    check_files(base)
    assert sum(t.startswith("Previous Choice Unrewarded") for t in titles["empty-row"][1]) == 3
