"""Kernel Q figures: file names, short titles with the statistics line, captions, rules."""

import re

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.agents import kernel_q_agent as kq  # noqa: E402
from src.agents import null_agents as na  # noqa: E402
from src.behavior_analysis import strategy_models as sm  # noqa: E402
from src.behavior_visualization import plot_kernel_q as plots  # noqa: E402
from src.pipelines import strategy_comparison as pipe  # noqa: E402
from tests.test_plot_choice_logit import check_files  # noqa: E402
from tests.test_plot_regression_extensions import saved_figures  # noqa: E402, F401
from tests.test_plot_strategy_models import fake_source  # noqa: E402

STATS_LINE = re.compile(r"mean ± se across mice, n=\d+ mice$")
PER_MOUSE_LINE = re.compile(r"per mouse, n=\d+ mice$")
ON_AXIS = re.compile(r"n per (source|estimate) on the axis$")


def caption_of(base):
    return base.with_name(f"{base.name}.caption.txt").read_text()


def kernel_source(n, seed=0, prefix="m"):
    """A fake comparison whose mice also have kernel Q and habit fits."""
    source = fake_source(n, seed, prefix)
    rng = np.random.default_rng(seed)
    fits = {}
    for mouse in sorted(source.nats["mouse"].unique()):
        params = tuple(rng.uniform(low, high) for low, high in (
            (0.3, 0.8), (0.1, 0.4), (0.6, 1.0), (0.05, 0.4), (0.5, 2.0), (0.5, 2.0)))  # fmt: skip
        params = (*params, 0.2)
        habit = na.WslsHabitParams(0.4, -0.3, 0.2, 0.7, 1.0, float(rng.choice(na.HABIT_RATES)))
        fits[mouse] = {sm.KERNEL_Q: params, sm.WSLS_HABIT: habit}
    return source._replace(fits=fits)


def fake_profile(mice):
    rng = np.random.default_rng(1)
    rows = [
        dict(mouse=m, decay=d, n_choices=900, nats_per_choice=1.0 + 0.3 * (d - 0.8) ** 2
             + rng.normal(0, 0.003))
        for m in mice for d in (sm.NO_TRACE, *sm.DECAY_GRID)
    ]  # fmt: skip
    return pd.DataFrame(rows)


def test_ablation_figure(tmp_path, saved_figures):  # noqa: F811
    base = tmp_path / "Kernel Q Ablations"
    plots.plot_kernel_q_ablations({"Mice": kernel_source(6)}, base)
    check_files(base)
    suptitle, panels, legend, over = saved_figures[base.name]
    assert suptitle == "Kernel Q Ablations" and "\n" not in suptitle
    assert [p.split("\n")[0] for p in panels] == [
        "No Asymmetry Minus Full", "No Forgetting Minus Full", "No Kernel Minus Full",
        "Kernel Only Minus Full",
    ]  # fmt: skip
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels), panels
    assert over == []
    caption = caption_of(base)
    assert "No Kernel: beta_k held at 0" in caption and "Wilcoxon p=" in caption


def test_ablation_figure_with_few_mice_has_no_mean_or_test(tmp_path, saved_figures):  # noqa: F811
    base = tmp_path / "Kernel Q Ablations"
    plots.plot_kernel_q_ablations({"Mice": kernel_source(5)}, base)
    _, panels, _, _ = saved_figures[base.name]
    assert not any("±" in p for p in panels)
    assert all(PER_MOUSE_LINE.search(p.split("\n")[-1]) for p in panels)
    assert "Wilcoxon" not in caption_of(base)


def test_parameter_figure_lists_every_mouse_in_the_caption(tmp_path, saved_figures):  # noqa: F811
    base = tmp_path / "Kernel Q Parameters"
    plots.plot_kernel_q_parameters({"Mice": kernel_source(6)}, base)
    check_files(base)
    _, panels, _, _ = saved_figures[base.name]
    assert [p.split("\n")[0] for p in panels] == list(plots.PARAMETER_LABELS.values())
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels)
    caption = caption_of(base)
    assert all(f"Mice, m{i}: alpha_pos" in caption for i in range(1, 7))


def test_simulated_sources_are_gray_and_unnamed(tmp_path, monkeypatch):
    seen = {}
    real = plots.save_or_show

    def spy(fig, path):
        seen["legend"] = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        return real(fig, path)

    monkeypatch.setattr(plots, "save_or_show", spy)
    sources = {"Mice": kernel_source(6), "Synthetic": kernel_source(6, 1, prefix="sim")}
    plots.plot_kernel_q_parameters(sources, tmp_path / "p")
    assert "m1" in seen["legend"] and not any(n.startswith("sim") for n in seen["legend"])


def test_decay_figure_marks_the_fitted_decay_and_compares_three_estimates(
    tmp_path,
    saved_figures,  # noqa: F811
):
    source = kernel_source(6)
    mice = sorted(source.fits)
    profile = fake_profile(mice)
    long_weights = pd.DataFrame(
        [dict(mouse=m, feature=f"choice_lag{k}", group="choice", lag=float(k),
              weight=0.5 * 0.85 ** (k - 1)) for m in mice for k in range(1, 16)]
    )  # fmt: skip
    timescales = pipe.timescale_table(source, long_weights)
    assert timescales["long_decay"].between(0.84, 0.86).all()
    base = tmp_path / "Choice Kernel Decay"
    plots.plot_choice_kernel_decay(profile, timescales, base)
    check_files(base)
    suptitle, panels, legend, over = saved_figures[base.name]
    assert suptitle == "Choice Kernel Decay"
    assert [p.split("\n")[0] for p in panels] == [
        "Held-Out Fit by Trace Decay",
        "Kernel Decay by Estimate",
    ]
    assert PER_MOUSE_LINE.search(panels[0].split("\n")[-1])
    assert STATS_LINE.search(panels[1].split("\n")[-1]) or ON_AXIS.search(panels[1].split("\n")[-1])
    assert "m1" in legend and over == []
    assert "Best decay of the grid per mouse" in caption_of(base)


def test_the_timescale_table_is_nan_without_long_weights():
    table = pipe.timescale_table(kernel_source(3))
    assert table["long_decay"].isna().all()
    assert np.allclose(table["kq_decay"], 1 - table["kq_alpha_k"])
    assert set(table["habit_alpha_k"]) <= set(na.HABIT_RATES)


def test_the_figures_need_data(tmp_path):
    with pytest.raises(ValueError, match="no sources"):
        plots.plot_kernel_q_ablations({}, tmp_path / "x")
    with pytest.raises(ValueError, match="no decay profile"):
        plots.plot_choice_kernel_decay(pd.DataFrame(), pd.DataFrame(), tmp_path / "x")


def test_all_three_figures_are_saved_into_a_folder(tmp_path):
    source = kernel_source(6)
    profile = fake_profile(sorted(source.fits))
    plots.plot_kernel_q_figures({"Mice": source}, profile, pipe.timescale_table(source),
                                tmp_path / "out")  # fmt: skip
    for name in plots.FIGURE_NAMES:
        check_files(tmp_path / "out" / name)


def test_kernel_parameter_names_match_the_agent():
    assert set(plots.PARAMETER_LABELS) == set(kq.PARAMETER_NAMES) - {"plus_bias"}


@pytest.mark.parametrize("draw", ["plot_kernel_q_parameters", "plot_kernel_q_ablations"])
def test_unnamed_simulated_mice_have_no_legend(tmp_path, monkeypatch, draw):
    seen = {}
    real = plots.save_or_show

    def spy(fig, path):
        seen["legends"] = [ax.get_legend() for ax in fig.axes]
        return real(fig, path)

    monkeypatch.setattr(plots, "save_or_show", spy)
    sources = {"Synthetic": kernel_source(6, 1, prefix="sim")}
    getattr(plots, draw)(sources, tmp_path / "p", mice_named=False)
    assert all(legend is None for legend in seen["legends"])
