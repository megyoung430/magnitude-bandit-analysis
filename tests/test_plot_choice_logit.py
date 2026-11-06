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
    probabilities = np.full((n, 3), 1 / 3)
    table = mvp.rank_table(
        np.repeat([f"m{i}" for i in range(n_mice)], 300), values, chosen, probabilities
    )
    return mvp.psychometric_curves_from_table(table, n_bins=4)


def test_psychometric_rows_figure_draws_the_baseline(tmp_path, titles):
    curves, baselines = curves_for()
    rows = [
        plots.PsychometricRow(
            "Model Value", curves, plots.value_labels("Model Value", "Logit"), baselines
        ),
        plots.PsychometricRow("Agent Q-Value", curves, plots.value_labels("Agent Q-Value")),
    ]
    base = tmp_path / "psychometrics"
    plots.plot_psychometric_rows(rows, "Psychometric Curves", ("A caption line.",), base)
    check_files(base)
    check_titles(titles, "psychometrics", 6)
    caption = base.with_name("psychometrics.caption.txt").read_text()
    assert "A caption line." in caption
    assert "Dashed" in caption and "fewer than 3 mice" in caption


def test_value_labels_name_the_unit_and_the_bin_mean():
    assert plots.value_labels("Model Value", "Logit")[0] == "Model Value of Best (Bin Mean, Logit)"
    assert plots.value_labels("Task Magnitude", "Pulses", binned=False)[2] == (
        "Task Magnitude of Best - Second (Pulses)"
    )
    assert plots.value_labels("Agent Q-Value")[1] == "Agent Q-Value of Second (Bin Mean)"


def test_a_baseline_adds_dashed_lines_to_a_panel():
    curves, baselines = curves_for()
    panel = lambda **kw: plots.plot_psychometric_panel(  # noqa: E731
        curves["best"]["best_val"], curves["second"]["best_val"], "x", **kw
    )
    plain = panel()
    with_baseline = panel(
        baseline_best=baselines["best"]["best_val"], baseline_second=baselines["second"]["best_val"]
    )
    dashed = [line for line in with_baseline.get_lines() if line.get_linestyle() == "--"]
    assert len(dashed) == 2
    assert not [line for line in plain.get_lines() if line.get_linestyle() == "--"]


def test_a_baseline_on_top_of_the_observed_curve_stays_visible():
    curves, baselines = curves_for()
    ax = plots.plot_psychometric_panel(
        curves["best"]["best_val"], curves["second"]["best_val"], "x",
        baseline_best=baselines["best"]["best_val"],
    )  # fmt: skip
    dashed = next(line for line in ax.get_lines() if line.get_linestyle() == "--")
    solid = max(
        (ln for ln in ax.get_lines() if ln.get_linestyle() == "-"), key=lambda ln: ln.zorder
    )
    assert dashed.zorder > solid.zorder  # drawn above, so it cannot hide under the observed line
    assert dashed.get_marker() not in ("", "None", None)  # hollow markers show where it coincides


def test_bin_means_do_not_set_the_ticks_of_a_continuous_axis():
    curves, _ = curves_for()
    curve = curves["best"]["best_val"]
    ax = plots.plot_psychometric_panel(curve, curves["second"]["best_val"], "x")
    assert len(ax.get_xticks()) <= 8
    assert not np.array_equal(np.round(ax.get_xticks(), 6), np.round(curve.x, 6))


def test_exact_values_keep_one_tick_each():
    table = mvp.rank_table(
        np.repeat(["a", "b", "c"], 4),
        np.tile([[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 2, 4]], (3, 1)).astype(float),
        np.zeros(12, dtype=int),
    )
    curves, _ = mvp.psychometric_curves_from_table(table, n_bins=None)
    ax = plots.plot_psychometric_panel(
        curves["best"]["best_val"], curves["second"]["best_val"], "x"
    )
    assert list(ax.get_xticks()) == list(curves["best"]["best_val"].x)


def test_rows_name_the_row_in_every_panel_title(tmp_path, titles):
    curves, baselines = curves_for()
    rows = [
        plots.PsychometricRow(
            "Value-Blind Sticky Agent", curves, plots.value_labels("V"), baselines
        ),
        plots.PsychometricRow("Value-Driven Agent", curves, plots.value_labels("V"), baselines),
    ]
    plots.plot_psychometric_rows(rows, "Control", ("c",), tmp_path / "rows")
    panel_titles = titles["rows"][1]
    assert sum(t.startswith("Value-Blind Sticky Agent") for t in panel_titles) == 3
    assert sum(t.startswith("Value-Driven Agent") for t in panel_titles) == 3
    caption = (tmp_path / "rows.caption.txt").read_text()
    assert "coincide" in caption


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
        meta, rng.normal(size=(n, 3)), rng.integers(0, 3, n), np.full((n, 3), 1 / 3)
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
    curves, baselines = by_reward(pairs, pool_trials=True, n_bins=4)
    base = tmp_path / name
    function(curves, "Model Value", True, base, baselines_by_row=baselines, unit="Logit")
    check_files(base)
    check_titles(titles, name, 6)
    caption = base.with_name(f"{name}.caption.txt").read_text()
    assert "rewarded" in caption.lower()
    assert "Dashed" in caption and "Model Value" in caption  # value label not lowercased


def test_stay_switch_figure_keeps_an_empty_row(tmp_path, titles):
    pairs = pairs_frame()
    curves, _ = mvp.stay_switch_curves_by_reward(
        pairs[pairs["rewarded"]], pool_trials=True, n_bins=4
    )
    base = tmp_path / "empty-row"
    plots.plot_stay_switch_by_reward(curves, "Model Value", True, base)
    check_files(base)
    assert sum(t.startswith("Previous Choice Unrewarded") for t in titles["empty-row"][1]) == 3


def test_a_curve_figure_with_a_single_row_is_drawn(tmp_path):
    from src.behavior_visualization.plot_helpers import Title
    from src.behavior_visualization.plot_stay_switch import STAY_SWITCH_SPEC, plot_outcome_curves

    curves, _ = mvp.stay_switch_curves_by_reward(pairs_frame(), pool_trials=True, n_bins=4)
    base = tmp_path / "one-row"
    plot_outcome_curves(
        {"rewarded": curves["rewarded"]}, STAY_SWITCH_SPEC, Title("One Row", None, ("c",)), base,
        row_headings={"rewarded": "Previous Choice Rewarded"},
        panels=(("diff", "x"), ("chosen", "y"), ("alt", "z")),
    )  # fmt: skip
    check_files(base)


def test_the_model_comparison_returns_its_axes(tmp_path):
    fresh = plots.plot_model_comparison(comparison_table(), tmp_path / "returned")
    assert fresh.get_ylabel().startswith("Gain over Chance")
    _, ax = plots.plt.subplots()
    assert plots.plot_model_comparison(comparison_table(), ax=ax) is ax


def test_the_reward_tail_is_not_a_point_of_the_kernel(tmp_path):
    weights = pd.concat(
        [
            weights_table(),
            pd.DataFrame(
                [(f"m{m}", "reward_tail", "reward", np.nan, 0.1) for m in range(4)],
                columns=["mouse", "feature", "group", "lag", "weight"],
            ),
        ]
    )
    plots.plot_kernels(weights, tmp_path / "with-tail")
    assert "reward tail" in tmp_path.joinpath("with-tail.caption.txt").read_text()
