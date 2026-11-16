"""Persistence figures: file names, short titles with the statistics line, captions, rules."""

import re

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.behavior_analysis.value import stay_by_run_length as rl  # noqa: E402
from src.behavior_visualization.choice_logit import plot_choice_kernel as plots  # noqa: E402
from src.pipelines.choice_logit.choice_kernel_long import LongKernel  # noqa: E402
from tests.test_plot_choice_logit import check_files  # noqa: E402
from tests.test_plot_regression_extensions import saved_figures  # noqa: E402, F401

STATS_LINE = re.compile(r"mean ± se across mice, n=\d+ mice$")
PER_MOUSE_LINE = re.compile(r"mean across mice, n=\d+ mice$|per mouse, n=\d+ mice$")
LAGS = list(range(1, 16))


def caption_of(base):
    return base.with_name(f"{base.name}.caption.txt").read_text()


def fake_long(n, decay=0.8, amplitude=0.9, seed=0, prefix="m"):
    rng = np.random.default_rng(seed)
    mice = [f"{prefix}{i}" for i in range(1, n + 1)]
    rows = [
        dict(mouse=m, feature=f"choice_lag{k}", group="choice", lag=float(k),
             weight=amplitude * decay ** (k - 1) + rng.normal(0, 0.02))
        for m in mice for k in LAGS
    ]  # fmt: skip
    nats = [
        dict(mouse=m, model=model, n_trials=1000, nats_per_choice=0.95 - 0.004 * (tail))
        for m in mice for model, tail in (("full", 0), ("full + choice tail", 1))
    ]  # fmt: skip
    weights = pd.DataFrame(rows)
    return LongKernel(weights, weights, pd.DataFrame(nats), {}, 15, 0.9, 0.9, 1.0)


def fake_recovery():
    rows = [
        dict(alpha_k=a, lag=k, true=2.0 * a * (1 - a) ** (k - 1),
             fitted_mean=2.0 * a * (1 - a) ** (k - 1) * 0.95, fitted_se=0.02, n_mice=6)
        for a in (0.1, 0.3) for k in LAGS
    ]  # fmt: skip
    return pd.DataFrame(rows)


def fake_stay(n, slope, seed=0, prefix="m"):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, n + 1):
        for payout, base in (("zero", 0.2), ("non-zero", 0.6)):
            for k, name in enumerate(rl.BIN_NAMES):
                p_stay = min(0.99, base + slope * k + rng.normal(0, 0.02))
                rows.append(
                    dict(mouse=f"{prefix}{i}", payout=payout, bin=name, n=100, p_stay=p_stay)
                )
    return pd.DataFrame(rows)


def test_long_kernel_figure(tmp_path, saved_figures):  # noqa: F811
    base = tmp_path / "Choice Kernel Long"
    sources = {"Mice": fake_long(6, 0.9), "WSLS (Magnitude)": fake_long(6, 0.2, seed=1)}
    plots.plot_choice_kernel_long(sources, base, fake_recovery())
    check_files(base)
    suptitle, panels, legend, over = saved_figures[base.name]
    assert suptitle == "Choice Kernel Long" and "\n" not in suptitle
    assert [p.split("\n")[0] for p in panels] == [
        "Choice Kernel to Lag 15",
        "Known-Kernel Recovery",
    ]
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels), panels
    assert "Mice (n=6)" in legend and "WSLS (Magnitude) (n=6)" in legend and over == []
    caption = caption_of(base)
    assert "held-out nats per choice without the tail" in caption
    assert "Mice (n=6 mice): mean choice weight lag 1" in caption and "timescale" in caption


def test_few_mice_have_no_mean_or_error(tmp_path, saved_figures):  # noqa: F811
    base = tmp_path / "Choice Kernel Long"
    plots.plot_choice_kernel_long({"Mice": fake_long(5)}, base)
    _, panels, _, _ = saved_figures[base.name]
    assert "±" not in panels[0] and PER_MOUSE_LINE.search(panels[0].split("\n")[-1])


def test_simulated_mice_are_gray_and_unnamed(tmp_path, monkeypatch):
    seen = {}
    real = plots.save_or_show

    def spy(fig, path):
        colors = {line.get_color() for line in fig.axes[0].lines if line.get_alpha() == 0.4}
        seen["thin"] = colors
        return real(fig, path)

    monkeypatch.setattr(plots, "save_or_show", spy)
    plots.plot_choice_kernel_long({"Synthetic": fake_long(6)}, tmp_path / "c", mice_named=False)
    assert seen["thin"] == {plots.OTHER_MICE_COLOR}


def test_stay_by_run_length_figure(tmp_path, saved_figures):  # noqa: F811
    base = tmp_path / "Stay by Run Length"
    sources = {"Mice": fake_stay(6, 0.05), "WSLS": fake_stay(6, 0.0, seed=1)}
    plots.plot_stay_by_run_length(sources, base)
    check_files(base)
    suptitle, panels, legend, over = saved_figures[base.name]
    assert suptitle == "Stay by Run Length"
    assert [p.split("\n")[0] for p in panels] == ["After Zero Payout", "After Non-Zero Payout"]
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels), panels
    assert "Mice (n=6)" in legend and over == []
    caption = caption_of(base)
    assert "Mice, after zero payout: slope of P(stay) per bin 0.0" in caption
    assert "fewer than 20 decisions" in caption


def test_stay_figure_with_few_mice_shows_mice_only(tmp_path, saved_figures):  # noqa: F811
    base = tmp_path / "Stay by Run Length"
    plots.plot_stay_by_run_length({"Mice": fake_stay(4, 0.05)}, base)
    _, panels, _, _ = saved_figures[base.name]
    assert not any("±" in p for p in panels)


def test_the_figures_need_data(tmp_path):
    with pytest.raises(ValueError, match="no sources"):
        plots.plot_choice_kernel_long({}, tmp_path / "x")
    with pytest.raises(ValueError, match="no data"):
        plots.plot_stay_by_run_length({"Mice": pd.DataFrame()}, tmp_path / "x")


def test_both_figures_are_saved_into_a_folder(tmp_path):
    plots.plot_choice_kernel_figures(
        {"Mice": fake_long(6)}, {"Mice": fake_stay(6, 0.05)}, tmp_path / "out", fake_recovery()
    )
    for name in plots.FIGURE_NAMES:
        check_files(tmp_path / "out" / name)
