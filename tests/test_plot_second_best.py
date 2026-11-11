"""Second-best figures: file names, short titles with the statistics line, captions, bands."""

import re

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.behavior_analysis import second_best as sb  # noqa: E402
from src.behavior_analysis import switch_regression as sr  # noqa: E402
from src.behavior_visualization import plot_second_best as plots  # noqa: E402
from src.pipelines.second_best import SecondBestCalibration, SourceSecondBest  # noqa: E402
from tests.test_plot_choice_logit import check_files  # noqa: E402
from tests.test_plot_regression_extensions import saved_figures  # noqa: E402, F401

STATS_LINE = re.compile(r"mean ± se across mice, n=\d+ mice$")
PER_MOUSE_LINE = re.compile(r"per mouse, n=\d+ mice$")


def fake_source(n, seed=0, skipped=None, min_used=30):
    """Mice with every cell but leaving the best after zero (empty in the task), as measured."""
    rng = np.random.default_rng(seed)
    mice = [f"m{i}" for i in range(1, n + 1)]
    rows, dropped, graded = [], [], []
    for m in mice:
        for measure in sb.MEASURES:
            for condition in sb.ROW_CONDITIONS:
                cell = dict(
                    mouse=m,
                    measure=measure,
                    condition=condition,
                    n_switches=300,
                    n_used=0 if (measure == sb.SECOND_BEST and condition == sr.AFTER_ZERO) else 120,
                    n_success=70,
                    n_tied=3,
                    n_unvisited=0,
                    p=rng.uniform(0.5, 0.8),
                )
                if cell["n_used"] == 0:
                    cell.update(n_success=0, p=np.nan)
                    dropped.append(cell)
                else:
                    rows.append(cell)
        for measure in sb.GRADED_MEASURES:
            for condition in sb.ROW_CONDITIONS:
                for d in sb.DIFFERENCES:
                    graded.append(dict(mouse=m, measure=measure, condition=condition,
                                       difference=d, n_used=40 if d < 4 else 4, n_success=30,
                                       p=0.5 + 0.1 * d + rng.normal(0, 0.03)))  # fmt: skip
    return SourceSecondBest(
        proportions=pd.DataFrame(rows),
        graded=pd.DataFrame(graded),
        dropped=pd.DataFrame(dropped),
        skipped=skipped or {},
        min_used=min_used,
        n_lags=5,
    )


def fake_calibration(source, agents=("wsls_magnitude", "wsls_recency")):
    rows = [
        dict(agent=a, mouse=r.mouse, measure=r.measure, condition=r.condition, observed=r.p,
             n_null=100, null_mean=0.5, null_sd=0.03, null_low=0.45, null_high=0.55,
             z=(r.p - 0.5) / 0.03, percentile=0.99, p_empirical=0.0099)
        for a in agents
        for r in source.proportions.itertuples()
    ]  # fmt: skip
    table = pd.DataFrame(rows)
    summary = pd.concat(
        [sb.calibrated_summary(g).assign(agent=a) for a, g in table.groupby("agent", sort=False)]
    )
    return SecondBestCalibration(table, pd.DataFrame(), summary, pd.DataFrame(), 100, 0)


def _panels(seen, name):
    suptitle, panels, legend, over = seen[name]
    assert "\n" not in suptitle
    assert over == []  # legends sit outside the data
    return suptitle, panels, legend


# ---------------------------------------------------------------------------
# Second Best Choices
# ---------------------------------------------------------------------------


def test_choices_figure_has_a_panel_per_measure_and_condition(tmp_path, saved_figures):  # noqa: F811
    sources = {"Mice": fake_source(6), "WSLS": fake_source(6, 1)}
    base = tmp_path / "Second Best Choices"
    plots.plot_second_best_choices(sources, base)
    check_files(base)
    suptitle, panels, legend = _panels(saved_figures, base.name)
    assert suptitle == "Second Best Choices"
    assert [p.split("\n")[0] for p in panels] == [
        "Second Best After Leaving the Best, After Zero",
        "Second Best After Leaving the Best, After Non-Zero",
        "Believed-Better Alternative, After Zero",
        "Believed-Better Alternative, After Non-Zero",
        "Truly Better Alternative, After Zero",
        "Truly Better Alternative, After Non-Zero",
    ]
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels[1:]), panels
    assert "m1" in legend


def test_the_empty_leave_best_after_zero_panel_says_so(tmp_path, monkeypatch):
    seen = {}
    real = plots.save_or_show

    def spy(fig, path):
        seen["texts"] = [t.get_text() for ax in fig.axes for t in ax.texts]
        return real(fig, path)

    monkeypatch.setattr(plots, "save_or_show", spy)
    plots.plot_second_best_choices({"Mice": fake_source(6)}, tmp_path / "c")
    assert seen["texts"].count("none") == 1


def test_few_mice_are_shown_per_mouse_without_mean_or_error(tmp_path, saved_figures):  # noqa: F811
    base = tmp_path / "Second Best Choices"
    plots.plot_second_best_choices({"Mice": fake_source(5)}, base)
    _, panels, _ = _panels(saved_figures, base.name)
    assert all(PER_MOUSE_LINE.search(p.split("\n")[-1]) for p in panels[1:]), panels
    assert not any("±" in p for p in panels)
    caption = base.with_name(f"{base.name}.caption.txt").read_text()
    assert "fewer than 6 mice" in caption


def test_caption_states_definitions_exclusions_and_the_minimum(tmp_path):
    base = tmp_path / "Second Best Choices"
    plots.plot_second_best_choices({"Mice": fake_source(6)}, base)
    caption = base.with_name(f"{base.name}.caption.txt").read_text()
    assert "strict maximum" in caption and "previous trial" in caption
    assert "never follows a zero payout" in caption
    assert "0.5 line is for orientation only" in caption
    assert "at least 30 used switches" in caption
    assert "Mice, m1, Truly Better Alternative, After Non-Zero: 70/120" in caption
    assert "3 tied" in caption
    assert "Dropped cells, Mice" in caption and "m1 (0)" in caption


def test_calibrated_mice_get_a_null_band_and_calibrated_numbers_in_the_caption(
    tmp_path, monkeypatch
):
    seen = {}
    real = plots.save_or_show

    def spy(fig, path):
        seen["bands"] = sum(
            1 for ax in fig.axes for line in ax.get_lines() if line.get_linewidth() == 5
        )
        return real(fig, path)

    monkeypatch.setattr(plots, "save_or_show", spy)
    mice = fake_source(6)
    plots.plot_second_best_choices(
        {"Mice": mice, "WSLS": fake_source(6, 1)}, tmp_path / "c", {"Mice": fake_calibration(mice)}
    )
    # six mice in five drawn panels (leaving the best has no after-zero cell)
    assert seen["bands"] == 6 * 5
    caption = (tmp_path / "c.caption.txt").read_text()
    assert "Null calibration, Mice" in caption and "100 replicates" in caption
    assert "vs WSLS (Magnitude)" in caption and "vs WSLS (Recency)" in caption
    assert "calibrated z vs WSLS (Magnitude)" in caption and "Wilcoxon" in caption


def test_skipped_mice_are_named_and_counted_on_the_axis(tmp_path, monkeypatch):
    seen = {}
    real = plots.save_or_show

    def spy(fig, path):
        seen["ticks"] = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        return real(fig, path)

    monkeypatch.setattr(plots, "save_or_show", spy)
    sources = {"Mice": fake_source(2, skipped={"m9": "never switches"}), "WSLS": fake_source(6, 1)}
    plots.plot_second_best_choices(sources, tmp_path / "s")
    assert seen["ticks"] == ["Mice\nn=2 (1 skipped)", "WSLS\nn=6"]
    caption = (tmp_path / "s.caption.txt").read_text()
    assert "Skipped, Mice: m9 (never switches)" in caption


def test_every_figure_needs_sources(tmp_path):
    for draw in (plots.plot_second_best_choices, plots.plot_better_alternative_curves):
        with pytest.raises(ValueError, match="no sources"):
            draw({}, tmp_path / "x")


# ---------------------------------------------------------------------------
# Better Alternative Curves
# ---------------------------------------------------------------------------


def test_curves_figure_has_a_panel_per_value_and_condition(tmp_path, saved_figures):  # noqa: F811
    sources = {"Mice": fake_source(6), "Q-Learner": fake_source(6, 1)}
    base = tmp_path / "Better Alternative Curves"
    plots.plot_better_alternative_curves(sources, base)
    check_files(base)
    suptitle, panels, legend = _panels(saved_figures, base.name)
    assert suptitle == "Better Alternative Curves"
    assert [p.split("\n")[0] for p in panels] == [
        "True Value, After Zero", "True Value, After Non-Zero",
        "Believed Value, After Zero", "Believed Value, After Non-Zero",
    ]  # fmt: skip
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels), panels
    assert "Q-Learner" in legend
    caption = base.with_name(f"{base.name}.caption.txt").read_text()
    assert "fewer than 10 used switches" in caption


def test_curve_points_with_few_switches_are_not_drawn(tmp_path, monkeypatch):
    seen = {}
    real = plots.save_or_show

    def spy(fig, path):
        ax = fig.axes[0]
        seen["x"] = [  # x of the points actually drawn (finite y) of the data lines
            np.asarray(line.get_xdata())[np.isfinite(np.asarray(line.get_ydata(), dtype=float))]
            for line in ax.get_lines()
            if not line.get_label().startswith("_")
        ]
        return real(fig, path)

    monkeypatch.setattr(plots, "save_or_show", spy)
    plots.plot_better_alternative_curves({"Mice": fake_source(6)}, tmp_path / "g")
    # the 4-pulse bin has 4 used switches per mouse in the fake: only 1-3 are drawn
    drawn = np.concatenate(seen["x"])
    assert set(np.unique(drawn)) <= {1.0, 2.0, 3.0}


def test_simulated_mice_are_gray_and_not_in_the_legend(tmp_path, saved_figures):  # noqa: F811
    simulated = fake_source(6, 1)
    for table in (simulated.proportions, simulated.graded, simulated.dropped):
        table["mouse"] = "sim" + table["mouse"]
    base = tmp_path / "Second Best Choices"
    plots.plot_second_best_choices({"Mice": fake_source(6), "Q-Learner": simulated}, base)
    _, _, legend = _panels(saved_figures, base.name)
    assert "m1" in legend and not any(name.startswith("sim") for name in legend)
