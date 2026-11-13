"""Switch-regression figures: file names, short titles with the statistics line, captions."""

import re

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.behavior_analysis.switching import switch_regression as sr  # noqa: E402
from src.behavior_visualization.switching import plot_switch_regression as plots  # noqa: E402
from src.pipelines.switching.switch_regression import NullCalibration, SourceSwitchRegression  # noqa: E402
from tests.test_plot_choice_logit import check_files  # noqa: E402
from tests.test_plot_regression_extensions import saved_figures  # noqa: E402, F401

STATS_LINE = re.compile(r"mean ± se across mice, n=\d+ mice$")
PER_MOUSE_LINE = re.compile(r"per mouse, n=\d+ mice$")
FEATURES = ("choice_lag2", "choice_lag3", "trace", "recency", "never_chosen")


def fake_source(n, seed=0, values=sr.VALUES, decays=None, skipped=None):
    rng = np.random.default_rng(seed)
    mice = [f"m{i}" for i in range(1, n + 1)]
    rows = [
        dict(mouse=m, condition=c, value=v, beta=rng.normal(0.3, 0.1), se=0.05,
             nats_history=0.6, nats_value=0.59, gain=rng.normal(0.01, 0.005))
        for m in mice for c in sr.CONDITIONS for v in values
    ]  # fmt: skip
    history = [
        dict(mouse=m, condition=c, feature=f, weight=rng.normal(0.5, 0.2), se=0.1)
        for m in mice for c in sr.CONDITIONS for f in FEATURES
    ]  # fmt: skip
    counts = [
        dict(mouse=m, condition=c, n_switches=300 + i, n_problems=4)
        for i, m in enumerate(mice) for c in sr.CONDITIONS
    ]  # fmt: skip
    return SourceSwitchRegression(
        values=pd.DataFrame(rows),
        history=pd.DataFrame(history),
        counts=pd.DataFrame(counts),
        decays=pd.DataFrame({"mouse": mice, "trace_decay": 0.9 if decays is None else decays}),
        decay_table=pd.DataFrame(),
        skipped=skipped or {},
        n_lags=3,
        l2=1.0,
        trace_grid=(0.5, 0.9),
        recency_cap=100,
        min_switches=100,
    )


def _panels(seen, name):
    suptitle, panels, legend, over = seen[name]
    assert "\n" not in suptitle
    assert over == []  # legends sit outside the data
    return suptitle, panels, legend


def test_value_weight_figure_has_a_panel_per_value_and_condition(tmp_path, saved_figures):  # noqa: F811
    sources = {"Mice": fake_source(6), "WSLS": fake_source(6, 1)}
    base = tmp_path / "Switch Regression Value Weights"
    plots.plot_value_weights(sources, base)
    check_files(base)
    suptitle, panels, legend = _panels(saved_figures, base.name)
    assert suptitle == "Switch Regression Value Weights"
    assert [p.split("\n")[0] for p in panels] == [
        "True Value, After Zero", "True Value, After Non-Zero",
        "Believed Value, After Zero", "Believed Value, After Non-Zero",
        "Model Value, After Zero", "Model Value, After Non-Zero",
    ]  # fmt: skip
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels), panels
    assert "m1" in legend
    caption = base.with_name(f"{base.name}.caption.txt").read_text()
    assert "consistency check" in caption and "not an independent test" in caption
    assert "leave-one-problem-out" in caption.lower()
    assert "no intercept" in caption.lower()
    assert "m1: 300 after zero, 300 after non-zero" in caption
    assert "Wilcoxon" in caption


def test_value_weight_figure_with_few_mice_is_per_mouse(tmp_path, saved_figures):  # noqa: F811
    base = tmp_path / "Switch Regression Value Weights"
    plots.plot_value_weights({"Mice": fake_source(5)}, base)
    _, panels, _ = _panels(saved_figures, base.name)
    assert all(PER_MOUSE_LINE.search(p.split("\n")[-1]) for p in panels), panels
    assert not any("±" in p for p in panels)
    caption = base.with_name(f"{base.name}.caption.txt").read_text()
    assert "fewer than 6 mice" in caption


def test_model_comparison_figure_shows_the_gain_of_each_value(tmp_path, saved_figures):  # noqa: F811
    sources = {"Mice": fake_source(6), "Q-Learner": fake_source(6, 2)}
    base = tmp_path / "Switch Regression Model Comparison"
    plots.plot_model_comparison(sources, base)
    check_files(base)
    suptitle, panels, _ = _panels(saved_figures, base.name)
    assert suptitle == "Switch Regression Model Comparison"
    assert len(panels) == 6
    caption = base.with_name(f"{base.name}.caption.txt").read_text()
    assert "history-only" in caption and "nats per switch" in caption.lower()


def test_history_weight_figure_has_a_panel_per_condition_and_source(tmp_path, saved_figures):  # noqa: F811
    sources = {"Mice": fake_source(4), "WSLS": fake_source(6, 1)}
    base = tmp_path / "Switch Regression History Weights"
    plots.plot_history_weights(sources, base)
    check_files(base)
    suptitle, panels, _ = _panels(saved_figures, base.name)
    assert suptitle == "Switch Regression History Weights"
    assert [p.split("\n")[0] for p in panels] == [
        "Mice, After Zero", "WSLS, After Zero", "Mice, After Non-Zero", "WSLS, After Non-Zero",
    ]  # fmt: skip
    last = [p.split("\n")[-1] for p in panels]
    assert PER_MOUSE_LINE.search(last[0]) and STATS_LINE.search(last[1]), last
    caption = base.with_name(f"{base.name}.caption.txt").read_text()
    assert "trace decay" in caption.lower() and "Mice: 0.9 (4 mice)" in caption


def test_a_source_without_model_values_leaves_its_column_empty(tmp_path, saved_figures):  # noqa: F811
    sources = {"Mice": fake_source(6), "WSLS": fake_source(6, 1, values=("true", "believed"))}
    plots.plot_value_weights(sources, tmp_path / "v")
    _, panels, _ = _panels(saved_figures, "v")
    assert len(panels) == 6
    # the empty source does not count, so the panel still states the one n of the others
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels), panels


def test_every_figure_needs_sources(tmp_path):
    for draw in (plots.plot_value_weights, plots.plot_model_comparison, plots.plot_history_weights):
        with pytest.raises(ValueError, match="no sources"):
            draw({}, tmp_path / "x")


def test_plot_switch_regression_writes_every_figure(tmp_path):
    out = tmp_path / "switch-regression"
    plots.plot_switch_regression({"Mice": fake_source(6), "WSLS": fake_source(6, 1)}, out)
    for name in plots.FIGURE_NAMES:
        for suffix in ("pdf", "png", "svg", "caption.txt"):
            assert (out / f"{name}.{suffix}").exists(), (name, suffix)


def fake_calibration(source, agents=("wsls_magnitude", "wsls_recency")):
    rows = [
        dict(agent=a, mouse=r.mouse, condition=r.condition, value=r.value, term=term,
             observed=getattr(r, term), n_null=100, null_mean=0.0, null_sd=0.05,
             null_low=-0.1, null_high=0.1, z=getattr(r, term) / 0.05, percentile=0.99,
             p_empirical=0.0099)
        for a in agents
        for r in source.values.query("value != 'model'").itertuples()
        for term in ("beta", "gain")
    ]  # fmt: skip
    table = pd.DataFrame(rows)
    summary = pd.concat(
        [sr.calibrated_summary(g).assign(agent=a) for a, g in table.groupby("agent", sort=False)]
    )
    return NullCalibration(table, pd.DataFrame(), summary, pd.DataFrame(), 100, 0)


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
    plots.plot_value_weights(
        {"Mice": mice, "WSLS": fake_source(6, 1)}, tmp_path / "v", {"Mice": fake_calibration(mice)}
    )
    assert seen["bands"] == 6 * 2 * 2  # six mice, two calibrated values, two conditions
    caption = (tmp_path / "v.caption.txt").read_text()
    assert "Null calibration, Mice" in caption and "100 replicates" in caption
    assert "vs WSLS (Magnitude)" in caption and "vs WSLS (Recency)" in caption
    assert "calibrated z vs WSLS (Magnitude)" in caption  # six mice: across-mice test
    assert "cannot separate it from the null agents" in caption
    assert "not clustered by problem" in caption


def test_skipped_mice_are_named_with_their_reason_and_counted_on_the_axis(tmp_path, monkeypatch):
    seen = {}
    real = plots.save_or_show

    def spy(fig, path):
        seen["ticks"] = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        seen["texts"] = [t.get_text() for ax in fig.axes for t in ax.texts]
        return real(fig, path)

    monkeypatch.setattr(plots, "save_or_show", spy)
    reason = "40 switches after zero, 90 after non-zero; at least 100 per condition are needed"
    sources = {
        "Mice": fake_source(2, skipped={"m9": reason}),
        "WSLS": fake_source(2, 1, values=("true", "believed")),
    }
    plots.plot_value_weights(sources, tmp_path / "v")
    assert seen["ticks"][0] == "Mice\nn=2 (1 skipped)"
    assert seen["texts"].count("not fitted") == 2  # WSLS has no model value in either condition
    caption = (tmp_path / "v.caption.txt").read_text()
    assert f"Skipped, Mice: m9 ({reason})." in caption


def test_trace_weights_of_mixed_decays_get_no_mean(tmp_path, monkeypatch):
    seen = {}
    real = plots.save_or_show

    def spy(fig, path):
        seen["bars"] = [len(ax.containers) for ax in fig.axes]
        return real(fig, path)

    monkeypatch.setattr(plots, "save_or_show", spy)
    mixed = fake_source(6, decays=[0.5, 0.9, 0.9, 0.95, 0.9, 0.7])
    plots.plot_history_weights({"Mixed": mixed, "Same": fake_source(6, 1)}, tmp_path / "h")
    # five features; the mixed-decay source loses the trace's mean bar in both conditions
    assert seen["bars"] == [4, 5, 4, 5]
    assert "not on one scale" in (tmp_path / "h.caption.txt").read_text()


def _renamed(source, prefix):
    """*source* with its mice renamed ``<prefix>1`` ... (e.g. simulated mice next to real ones)."""
    rename = {m: f"{prefix}{m[1:]}" for m in source.decays["mouse"]}
    return source._replace(
        **{
            field: getattr(source, field).replace({"mouse": rename})
            for field in ("values", "history", "counts", "decays")
        }
    )


@pytest.mark.parametrize(
    "plot", [plots.plot_value_weights, plots.plot_model_comparison, plots.plot_history_weights]
)
def test_mice_of_other_sources_are_gray_and_not_in_the_legend(tmp_path, saved_figures, plot):  # noqa: F811
    sim = _renamed(fake_source(6, 2), "sim")
    sources = {"Mice": fake_source(6), "Q-Learner": sim, "WSLS": fake_source(6, 1)}
    base = tmp_path / "figure"
    plot(sources, base)
    _, _, legend = _panels(saved_figures, base.name)
    assert "m1" in legend and not any(t.startswith("sim") for t in legend)


def test_mouse_colors_keep_the_palette_for_the_first_source_only():
    sim = _renamed(fake_source(3, 2), "sim")
    colors, unnamed = plots._mouse_colors({"Mice": fake_source(3), "Q-Learner": sim})
    assert unnamed == {"sim1", "sim2", "sim3"}
    assert {colors[m] for m in unnamed} == {plots.OTHER_MICE_COLOR}
    assert len({colors[m] for m in ("m1", "m2", "m3")}) == 3


def test_each_bar_has_its_mean_written_above_it(tmp_path, monkeypatch):
    seen = []
    real_text = matplotlib.axes.Axes.text

    def spy(self, x, y, s, *args, **kwargs):
        seen.append(s)
        return real_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", spy)
    mice, few = fake_source(6), fake_source(5, 1)
    plots.plot_value_weights({"Mice": mice, "WSLS": few}, tmp_path / "v")
    first = mice.values.query("value == 'true' and condition == @sr.AFTER_ZERO")["beta"].mean()
    assert f"{first:.2g}" in seen
    # 6 panels with one bar each: the source with 5 mice has no bar and no number
    assert sum(_is_number(t) for t in seen) == 6


def _is_number(text):
    try:
        float(text)
    except ValueError:
        return False
    return True
