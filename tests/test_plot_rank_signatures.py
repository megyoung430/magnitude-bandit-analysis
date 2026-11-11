"""The rank-signature figures: files, short titles, captions, and what the mice count allows."""

import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import rank_calibration as rcal
from src.behavior_analysis import rank_signatures as sg
from src.behavior_analysis import strategy_models as sm
from src.behavior_visualization import plot_rank_signatures as plots
from src.behavior_visualization.plot_helpers import FIGURE_FORMATS, caption_path
from src.pipelines import rank_signatures as pipe

STATS = re.compile(r"^(mean ± se across mice|per mouse), n=\d+ mice$")
READOUTS = sorted({name for names in sg.TESTS.values() for name in names})
MODELS = (sm.WSLS_MAGNITUDE, sm.WSLS_HABIT, sm.KERNEL_Q)
CURVES = {
    "crossing_step": ("",), "third_option": ("",), "transfer_p100": ("",),
    "transfer_threshold": ("",), "cross_believed": ("new_best", "own", "other"),
    "cross_true_best": ("next", "prev", "other"), "cross_true_second": ("next", "prev", "other"),
}  # fmt: skip


def curve_table(rng, mouse=None) -> pd.DataFrame:
    rows = [
        dict(curve=curve, series=series, x=x, y=float(rng.random()), n=10)
        for curve, all_series in CURVES.items()
        for series in all_series
        for x in range(-5, 11)
    ]
    table = pd.DataFrame(rows)
    return table if mouse is None else table.assign(mouse=mouse)


def fake_data(n_mice: int) -> plots.FigureData:
    rng = np.random.default_rng(0)
    mice = [f"m{i}" for i in range(n_mice)]
    readouts = pd.DataFrame({"mouse": mice, **{r: rng.normal(size=n_mice) for r in READOUTS}})
    curves = pd.concat([curve_table(rng, m) for m in mice], ignore_index=True)
    replay_rows, replay_curves = {}, {}
    for model in MODELS:
        replay_rows[model] = pd.DataFrame(
            [
                {"mouse": m, "replicate": k, **{r: rng.normal() for r in READOUTS}}
                for m in mice
                for k in range(8)
            ]
        )
        replay_curves[model] = pd.concat([curve_table(rng, m) for m in mice], ignore_index=True)
    replays = pipe.Replays(
        readouts=replay_rows,
        curves=replay_curves,
        failed=pd.DataFrame(
            [(m, model, 1 if m == "m0" else 0) for model in MODELS for m in mice],
            columns=["mouse", "agent", "n_failed"],
        ),
        replicates=8,
        seed=0,
    )
    tables, summaries = [], []
    for model, null in replay_rows.items():
        table = rcal.calibrate(readouts, null)
        tables.append(rcal.passes(table, sg.PREDICTED).assign(agent=model))
        summaries.append(rcal.summarize(table, sg.PREDICTED).assign(agent=model))
    calibration = pipe.Calibration(
        pd.concat(tables, ignore_index=True), pd.concat(summaries, ignore_index=True)
    )
    return plots.FigureData(
        pipe.SourceAnalysis(readouts, curves, {"m_skipped": "too few decisions"}),
        replays,
        calibration,
    )


@pytest.fixture(scope="module", params=[3, 8], ids=["three-mice", "eight-mice"])
def saved(request, tmp_path_factory):
    out = tmp_path_factory.mktemp(f"rank-{request.param}")
    titles = {}
    original = plots.save_or_show

    def spy(fig, save_path, **kwargs):
        axes = [ax.get_title() for ax in fig.axes if ax.get_visible() and ax.get_title()]
        titles[str(save_path)] = (fig.get_suptitle(), axes)
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


def test_panel_titles_are_a_heading_and_one_statistics_line(saved):
    _, _, titles = saved
    panel_titles = [t for _, axes in titles.values() for t in axes]
    assert panel_titles
    for title in panel_titles:
        heading, *rest = title.split("\n")
        assert len(rest) == 1 and STATS.match(rest[0]), title
        assert heading == heading.strip()


def test_the_statistics_line_claims_mean_and_se_only_from_six_mice(saved):
    n, _, titles = saved
    lines = {t.split("\n")[1] for _, axes in titles.values() for t in axes}
    expected = f"mean ± se across mice, n={n} mice" if n >= 6 else f"per mouse, n={n} mice"
    assert lines == {expected}


def test_figure_titles_are_the_names_without_statistics(saved):
    _, _, titles = saved
    assert {Path_name.split("/")[-1] for Path_name in titles} == set(plots.FIGURE_NAMES)
    for key, (suptitle, _) in titles.items():
        assert suptitle == key.split("/")[-1]


def test_the_caption_has_the_calibration_and_who_passes(saved):
    _, out, _ = saved
    text = caption_path(out / plots.STAY_WEIGHTS).read_text()
    assert "replays of each agent" in text
    assert "passing" in text and "Kernel Q-Learner" in text
    assert "Skipped, m_skipped: too few decisions" in text
    assert "left out for too little data" in text


def test_a_caption_gives_the_z_summary_only_from_six_mice(saved):
    n, out, _ = saved
    text = caption_path(out / plots.STAY_WEIGHTS).read_text()
    assert ("Wilcoxon p=" in text) == (n >= 6)
    assert ("fewer than 6 mice" in text) == (n < 6)


def test_a_missing_curve_raises():
    data = fake_data(3)
    broken = data._replace(source=data.source._replace(curves=data.source.curves.iloc[:0]))
    with pytest.raises(ValueError, match="no curve"):
        plots.plot_third_option(broken, None)
    plt.close("all")
