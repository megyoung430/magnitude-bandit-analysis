"""Strategy-model figures: file names, titles with the statistics line, captions, verdicts."""

import re

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.agents import kernel_q_agent as kq  # noqa: E402
from src.behavior_analysis import strategy_models as sm  # noqa: E402
from src.behavior_visualization import plot_strategy_models as plots  # noqa: E402
from src.pipelines.strategy_comparison import Recovery, SourceComparison  # noqa: E402
from tests.test_plot_choice_logit import check_files  # noqa: E402
from tests.test_plot_regression_extensions import saved_figures  # noqa: E402, F401

STATS_LINE = re.compile(r"mean ± se across mice, n=\d+ mice$")
PER_MOUSE_LINE = re.compile(r"per mouse, n=\d+ mice$")


def fake_source(n, seed=0, prefix="m", skipped=None):
    rng = np.random.default_rng(seed)
    mice = [f"{prefix}{i}" for i in range(1, n + 1)]
    rows, parameters = [], []
    for mouse in mice:
        base = rng.uniform(0.9, 1.1)
        for k, model in enumerate(sm.COMPARED):
            rows.append(dict(mouse=mouse, model=model, n_choices=900,
                             nats_per_choice=base + 0.02 * k * rng.uniform(0.5, 1.5)
                             - 0.03 * (model == sm.HEURISTIC_PLUS)))  # fmt: skip
        for model, parameter, value in (
            (sm.HEURISTIC_PLUS, "v", rng.normal(0.3, 0.1)),
            (sm.HEURISTIC_PLUS, "u", rng.normal(2.0, 0.3)),
            (sm.Q_LEARNER, "learning_rate", rng.uniform(0.1, 0.5)),
            (sm.Q_LEARNER, "beta", rng.uniform(1, 3)),
            (sm.WSLS_GRADED, "w_m", rng.normal(0.5, 0.1)),
            (sm.WSLS_GRADED, "p_recent", rng.uniform(0.5, 0.9)),
            *[(sm.KERNEL_Q, name, rng.uniform(0.1, 0.9)) for name in kq.PARAMETER_NAMES],
        ):
            parameters.append(dict(mouse=mouse, model=model, parameter=parameter, value=value,
                                   at_bound=False))  # fmt: skip
    nats = pd.DataFrame(rows)
    return SourceComparison(
        scores=pd.DataFrame(),
        parameters=pd.DataFrame(parameters),
        nats=nats,
        differences=sm.difference_table(nats),
        winners=sm.winners(nats),
        logit=pd.DataFrame({"mouse": mice, "nats_per_choice": 0.95}),
        fits={},
        skipped=skipped or {},
        n_starts=1,
        seed=0,
    )


def fake_recovery(diagonal=1.0, n_mice=4, correlation=0.95, seed=0):
    rng = np.random.default_rng(seed)
    won, parameters = [], []
    for mouse in (f"m{i}" for i in range(1, n_mice + 1)):
        for generator in sm.MODELS:
            for replicate in range(2):
                hit = rng.uniform() < diagonal
                winner = (
                    generator
                    if hit
                    else sm.MODELS[(sm.MODELS.index(generator) + 1) % len(sm.MODELS)]
                )
                won.append(dict(mouse=mouse, generator=generator, replicate=replicate,
                                winner=winner))  # fmt: skip
                for model, name in plots.RECOVERED:
                    if model == generator:
                        true = rng.normal(1, 0.5)
                        estimate = true * correlation + rng.normal(0, 1 - correlation)
                        parameters.append(
                            dict(
                                mouse=mouse,
                                generator=generator,
                                replicate=replicate,
                                parameter=name,
                                true=true,
                                estimate=estimate,
                                error=estimate - true,
                            )
                        )
    won, parameters = pd.DataFrame(won), pd.DataFrame(parameters)
    counts, fractions = sm.recovery_matrix(won)
    summary = pd.concat(
        [sm.parameter_recovery_summary(g).assign(generator=m) for m, g in
         parameters.groupby("generator", sort=False)]
    )  # fmt: skip
    return Recovery(won, pd.DataFrame(), counts, fractions, parameters, summary, pd.DataFrame(),
                    2, 0, 1)  # fmt: skip


def caption_of(base):
    return base.with_name(f"{base.name}.caption.txt").read_text()


# ---------------------------------------------------------------------------
# What recovery says
# ---------------------------------------------------------------------------


def test_good_recovery_is_called_good():
    lines = plots.recovery_statement(fake_recovery(1.0))
    assert lines[0].startswith("Model recovery is good")
    assert all("poorly recovered" not in line for line in lines)


def test_poor_recovery_is_said_plainly():
    lines = plots.recovery_statement(fake_recovery(0.4, correlation=0.1))
    assert lines[0].startswith("Model recovery is poor for")
    assert "Q-Learner Learning Rate is poorly recovered" in " ".join(lines)


# ---------------------------------------------------------------------------
# Strategy Model Comparison
# ---------------------------------------------------------------------------


def test_comparison_figure_panels_and_statistics_lines(tmp_path, saved_figures):  # noqa: F811
    base = tmp_path / "Strategy Model Comparison"
    plots.plot_strategy_model_comparison({"Mice": fake_source(6)}, base, fake_recovery())
    check_files(base)
    suptitle, panels, legend, over = saved_figures[base.name]
    assert suptitle == "Strategy Model Comparison" and "\n" not in suptitle
    assert [p.split("\n")[0] for p in panels[:3]] == [
        "Heuristic+ Minus Q-Learner", "WSLS (Magnitude) Minus Q-Learner",
        "Heuristic+ Minus WSLS (Magnitude)",
    ]  # fmt: skip
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels[:3] + panels[-4:]), panels
    assert "m1" in legend and over == []


def test_few_mice_have_no_mean_error_or_test(tmp_path, saved_figures):  # noqa: F811
    base = tmp_path / "Strategy Model Comparison"
    plots.plot_strategy_model_comparison({"Mice": fake_source(5)}, base)
    _, panels, _, _ = saved_figures[base.name]
    assert not any("±" in p for p in panels)
    assert any(PER_MOUSE_LINE.search(p.split("\n")[-1]) for p in panels)
    assert "Wilcoxon" not in caption_of(base)
    assert "fewer than 6 mice" in caption_of(base)


def test_caption_leads_with_the_recovery_verdict_and_states_the_decisions(tmp_path):
    base = tmp_path / "c"
    plots.plot_strategy_model_comparison(
        {"Mice": fake_source(6)}, base, fake_recovery(0.3, correlation=0.1)
    )
    caption = caption_of(base)
    assert caption.splitlines()[0].startswith("Model recovery is poor for")
    assert "fixed at 0.9" in caption and "descriptive reference" in caption
    assert "Wilcoxon p=" in caption and "Mice, m1: held-out nats per choice" in caption
    assert "winner" in caption and "choice logit" in caption


def test_without_recovery_the_caption_says_so(tmp_path):
    base = tmp_path / "c"
    plots.plot_strategy_model_comparison({"Mice": fake_source(6)}, base)
    assert "recovery was not run" in caption_of(base)


def test_simulated_sources_are_gray_and_skipped_mice_are_named(tmp_path, monkeypatch):
    seen = {}
    real = plots.save_or_show

    def spy(fig, path):
        seen["ticks"] = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        seen["legend"] = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        return real(fig, path)

    monkeypatch.setattr(plots, "save_or_show", spy)
    sources = {
        "Mice": fake_source(6, skipped={"m9": "1 problem(s)"}),
        "Q-Learner": fake_source(6, 1, prefix="sim"),
    }
    plots.plot_strategy_model_comparison(sources, tmp_path / "c")
    assert seen["ticks"] == ["Mice\nn=6 (1 skipped)", "Q-Learner\nn=6"]
    assert "m1" in seen["legend"] and not any(n.startswith("sim") for n in seen["legend"])
    assert "Skipped, Mice: m9 (1 problem(s))" in caption_of(tmp_path / "c")


# ---------------------------------------------------------------------------
# Strategy Model Recovery
# ---------------------------------------------------------------------------


def test_recovery_figure(tmp_path, saved_figures):  # noqa: F811
    base = tmp_path / "Strategy Model Recovery"
    plots.plot_strategy_model_recovery(fake_recovery(), base)
    check_files(base)
    suptitle, panels, legend, over = saved_figures[base.name]
    assert suptitle == "Strategy Model Recovery"
    assert [p.split("\n")[0] for p in panels] == [
        "Model Recovery", "Heuristic+ v", "Heuristic+ u", "Q-Learner Learning Rate",
        "WSLS (Graded) p_recent", "Kernel Q-Learner Alpha Pos",
        "Kernel Q-Learner Alpha Neg", "Kernel Q-Learner Zeta",
        "Kernel Q-Learner Alpha K", "Kernel Q-Learner Beta",
        "Kernel Q-Learner Beta K", "WSLS (Habit) w_h",
    ]  # fmt: skip
    assert "m1" in legend and over == []
    assert "Model recovery is good" in caption_of(base)


def test_the_figures_need_data(tmp_path):
    with pytest.raises(ValueError, match="no sources"):
        plots.plot_strategy_model_comparison({}, tmp_path / "x")
    empty = fake_recovery()._replace(won=pd.DataFrame())
    with pytest.raises(ValueError, match="no simulated mice"):
        plots.plot_strategy_model_recovery(empty, tmp_path / "x")


def test_both_figures_are_saved_into_a_folder(tmp_path):
    plots.plot_strategy_models({"Mice": fake_source(6)}, tmp_path / "out", fake_recovery())
    for name in plots.FIGURE_NAMES:
        check_files(tmp_path / "out" / name)


def test_a_correlation_that_cannot_be_computed_is_not_called_poor():
    lines = plots.recovery_statement(fake_recovery(1.0, n_mice=1))
    text = " ".join(lines)
    assert "cannot be judged by correlation" in text and "poorly recovered" not in text
