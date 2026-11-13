"""The GLM-HMM figures: files, short titles, captions and what the mice count allows."""

import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.agents import glm_hmm_agent as gha
from src.behavior_analysis.strategy_models import strategy_models as sm
from src.behavior_visualization.common.plot_helpers import FIGURE_FORMATS, caption_path
from src.behavior_visualization.glm_hmm import plot_glm_hmm as plots
from src.pipelines.glm_hmm import glm_hmm as pipe

STATS = re.compile(
    r"^(mean ± se across mice, n=\d+ mice|per mouse, n=\d+ mice|"
    r"mean ± se across mice, n per source on the axis|per mouse, n per source on the axis)$"
)
KS = (1, 2, 3, 4)


def fake_result(mouse: str, rng, selected: int) -> pipe.MouseResult:
    gains = {1: 0.0, 2: 0.01, 3: 0.015, 4: 0.012}
    table = pd.DataFrame(
        dict(n_states=KS, nats=[0.6 - g for g in gains.values()], gain=list(gains.values()),
             gain_se=[0.0, 0.003, 0.004, 0.004])
    )  # fmt: skip
    weights = pd.DataFrame(
        [
            dict(n_states=k, state=z, feature=f, weight=float(rng.normal()))
            for k in (1, 2)
            for z in range(k)
            for f in gha.FEATURES
        ]
    )
    trial = pd.DataFrame(
        [
            dict(state=z, trial_in_session=b, probability=float(0.5 + 0.1 * rng.normal()), n=20)
            for z in (0, 1)
            for b in range(0, 100, 10)
        ]
    )
    by_problem = trial.rename(columns={"trial_in_session": "problem_number"})
    by_problem["problem_number"] = by_problem["problem_number"] // 10 + 1
    block = trial.rename(columns={"trial_in_session": "offset"})
    block["offset"] = block["offset"] // 3 - 10
    kernel = pd.DataFrame(
        [
            dict(state=s, lag=lag, weight=float(1.0 / lag + 0.1 * rng.normal()), n_trials=300)
            for s in ("all", 0, 1)
            for lag in range(1, 6)
        ]
    )
    return pipe.MouseResult(
        mouse=mouse, n_trials=3000, n_sessions=50,
        scores=pd.DataFrame(), table=table, selected_k=selected, summary_k=2,
        readouts=dict(gain_k2=gains[2] + rng.normal(scale=0.005), gain_k3=gains[3],
                      gain_k4=gains[4], best_gain=gains[3]),
        static_levels=pd.DataFrame(
            dict(model=[sm.KERNEL_Q, sm.WSLS_HABIT], nats=[0.58, 0.62], n_scored=[2800, 2800])
        ),
        weights=weights,
        occupancy=pd.DataFrame(dict(state=[0, 1], occupancy=[0.6, 0.4], dwell=[40.0, 90.0],
                                    initial=[0.5, 0.5])),
        transition=np.array([[0.97, 0.03], [0.02, 0.98]]),
        by_trial_in_session=trial, by_problem=by_problem, block_aligned=block, kernel=kernel,
        n_converged=(8, 10),
    )  # fmt: skip


def fake_recovery(mice) -> list:
    rng = np.random.default_rng(1)
    out = []
    for mouse in mice:
        for scenario in pipe.SCENARIOS:
            switching = scenario in pipe.SWITCHING
            weights = pd.DataFrame(
                [
                    dict(state=z, fitted_state=z, feature=f, true=float(rng.normal()),
                         recovered=float(rng.normal()))
                    for z in range(3)
                    for f in gha.FEATURES
                ]
            ) if switching else pd.DataFrame()  # fmt: skip
            dwell = pd.DataFrame(
                dict(state=[0, 1, 2], true=[50.0] * 3, recovered=[40.0, 60.0, 80.0])
            ) if switching else pd.DataFrame()  # fmt: skip
            out.append(
                pipe.Recovery(
                    scenario, mouse,
                    dict(selected_k=3 if switching else 1, best_gain=0.02 if switching else 0.0),
                    pd.DataFrame(), pd.DataFrame(), 0.8 if switching else np.nan, weights, dwell,
                )
            )  # fmt: skip
    return out


def fake_data(n_mice: int) -> plots.FigureData:
    rng = np.random.default_rng(0)
    mice = [f"m{i}" for i in range(n_mice)]
    results = {m: fake_result(m, rng, 2 if i % 2 else 1) for i, m in enumerate(mice)}
    kernels = [
        pd.DataFrame(
            [
                dict(state=s, lag=lag, weight=float(1.0 / lag), n_trials=200)
                for s in ("all", 0, 1)
                for lag in range(1, 6)
            ]
        )
    ]
    null = pd.DataFrame(
        [
            dict(mouse=m, replicate=k, selected_k=1, gain_k2=float(rng.normal(0.0, 0.004)),
                 gain_k3=float(rng.normal(0.0, 0.005)), best_gain=float(rng.normal(0.004, 0.004)),
                 kernel=kernels[0] if k < 2 else np.nan)
            for m in mice
            for k in range(6)
        ]
    )  # fmt: skip
    observed = pd.DataFrame(
        {
            "mouse": mice,
            **{
                c: [r.readouts[c] for r in results.values()]
                for c in ("gain_k2", "gain_k3", "best_gain")
            },
        }
    )
    return plots.FigureData(
        mice=results,
        null=null,
        calibration=pipe.calibrate_gains(observed, null),
        recovery=fake_recovery(mice[:3]),
        replicates=6,
    )


@pytest.fixture(scope="module", params=[3, 8], ids=["three-mice", "eight-mice"])
def saved(request, tmp_path_factory):
    out = tmp_path_factory.mktemp(f"glm-hmm-{request.param}")
    titles = {}
    original = plots.save_or_show

    def spy(fig, save_path, **kwargs):
        axes = [ax.get_title() for ax in fig.axes if ax.get_visible() and ax.get_title()]
        titles[str(save_path).split("/")[-1]] = (fig.get_suptitle(), axes)
        return original(fig, save_path, **kwargs)

    plots.save_or_show = spy
    try:
        plots.plot_all(fake_data(request.param), out)
    finally:
        plots.save_or_show = original
        plt.close("all")
    return request.param, out, titles


def test_every_figure_is_saved_as_pdf_png_svg_with_a_caption(saved):
    _, out, _ = saved
    for name in plots.FIGURE_NAMES:
        for fmt in FIGURE_FORMATS:
            assert (out / f"{name}.{fmt}").stat().st_size > 0
        assert caption_path(out / name).read_text().strip()


def test_figure_titles_are_the_names_without_statistics(saved):
    _, _, titles = saved
    assert set(titles) == set(plots.FIGURE_NAMES)
    for name, (suptitle, _) in titles.items():
        assert suptitle == name


def test_panel_titles_are_a_heading_and_one_statistics_line(saved):
    _, _, titles = saved
    panel_titles = [t for _, axes in titles.values() for t in axes]
    assert panel_titles
    for title in panel_titles:
        heading, *rest = title.split("\n")
        assert len(rest) == 1 and STATS.match(rest[0]), title
        assert heading == heading.strip() and heading


def test_the_statistics_line_claims_mean_and_se_only_from_six_mice(saved):
    n, _, titles = saved
    lines = {t.split("\n")[1] for name, (_, axes) in titles.items() for t in axes
             if name != plots.SIMULATED_RECOVERY}  # fmt: skip
    claims_mean = any("mean ± se" in line for line in lines)
    assert claims_mean == (n >= 6)
    assert all("n=" in line or "per source" in line for line in lines)


def test_the_selection_caption_has_the_null_and_every_mouse(saved):
    n, out, _ = saved
    text = caption_path(out / plots.MODEL_SELECTION).read_text()
    assert "6 replays per mouse" in text
    assert ("Wilcoxon p=" in text) == (n >= 6)
    assert ("fewer than 6 mice" in text) == (n < 6)
    for i in range(n):
        assert f"m{i}: 3000 trials in 50 sessions; selected" in text


def test_the_weights_caption_marks_mice_with_one_selected_state(saved):
    _, out, _ = saved
    text = caption_path(out / plots.STATE_WEIGHTS).read_text()
    assert "hollow" in text and "State 1 the largest" in text


def test_the_recovery_caption_lists_every_simulated_agent(saved):
    _, out, _ = saved
    text = caption_path(out / plots.SIMULATED_RECOVERY).read_text()
    assert "m0, Switching, Fast: selected 3" in text
    assert "m0, Static Blend: selected 1" in text


def test_recovery_needs_simulations():
    data = fake_data(3)._replace(recovery=())
    with pytest.raises(ValueError, match="no simulated"):
        plots.plot_simulated_recovery(data, None)
    plt.close("all")


def test_the_kernel_figure_needs_no_replay_kernels(tmp_path):
    data = fake_data(3)
    data = data._replace(null=data.null.assign(kernel=np.nan))
    plots.plot_choice_kernel_by_state(data, tmp_path / "kernel")
    assert (tmp_path / "kernel.png").stat().st_size > 0
    plt.close("all")
