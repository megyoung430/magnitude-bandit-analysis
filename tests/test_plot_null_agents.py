"""Null-agent figures: file names, short titles with the statistics line, captions, legends."""

import re

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.behavior_visualization import plot_helpers  # noqa: E402
from src.behavior_visualization import plot_null_agents as plots  # noqa: E402
from tests.test_plot_choice_logit import check_files, weights_table  # noqa: E402

STATS_LINE = re.compile(r"mean ± se across mice, n=\d+ mice")
MODELS = ("wsls", "wsls_recency", "wsls_magnitude", "choice_logit")


def legends_over_panels(fig) -> list[str]:
    """Labels of each legend whose box overlaps the data area of a panel."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legends = [*fig.legends, *(ax.get_legend() for ax in fig.axes if ax.get_legend())]
    panels = [ax for ax in fig.axes if ax.get_visible() and ax.axison]
    return [
        " / ".join(t.get_text() for t in legend.get_texts())
        for legend in legends
        if any(legend.get_window_extent(renderer).overlaps(ax.bbox) for ax in panels)
    ]


@pytest.fixture
def figures(monkeypatch):
    """``{figure base name: (suptitle, panel titles, legend texts, legends over data)}``.

    Recorded after the save step, which moves legends out of the data and drops repeats.
    """
    seen = {}
    real = plot_helpers._write_figure_files

    def spy(fig, base, formats, dpi):
        result = real(fig, base, formats, dpi)
        panel_titles = [ax.get_title() for ax in fig.axes if ax.get_visible() and ax.get_title()]
        legends = [*fig.legends, *(ax.get_legend() for ax in fig.axes if ax.get_legend())]
        texts = [t.get_text() for leg in legends for t in leg.get_texts()]
        seen[str(base).rsplit("/", 1)[-1]] = (
            fig.get_suptitle(), panel_titles, texts, legends_over_panels(fig),
        )  # fmt: skip
        return result

    monkeypatch.setattr(plot_helpers, "_write_figure_files", spy)
    return seen


def scores(n_mice=4):
    rows = [
        (f"m{m}", model, 1000, 1.0 - 0.02 * k + 0.01 * m)
        for m in range(n_mice)
        for k, model in enumerate(MODELS)
    ]
    return pd.DataFrame(rows, columns=["mouse", "model", "n_choices", "nats_per_choice"])


def stay_tables(n_mice=4):
    rows = [(f"m{m}", k, 100, 0.4 + 0.1 * k) for m in range(n_mice) for k in range(5)]
    observed = pd.DataFrame(rows, columns=["mouse", "magnitude", "n_pairs", "p_stay"])
    implied = pd.DataFrame(
        [
            (f"m{m}", model, k, 0.4 + 0.1 * k * (model == "wsls_magnitude"))
            for m in range(n_mice)
            for model in MODELS[:3]
            for k in range(5)
        ],
        columns=["mouse", "model", "magnitude", "p_stay"],
    )
    return observed, implied


def test_fit_figure_has_two_panels_with_statistics_and_a_caption(tmp_path, figures):
    observed, implied = stay_tables()
    base = tmp_path / "Null Agent Fits"
    plots.plot_null_agent_fits(scores(), observed, implied, base)
    check_files(base)
    suptitle, panels, legend, over = figures["Null Agent Fits"]
    assert over == []
    assert "\n" not in suptitle
    assert len(panels) == 2
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels), panels
    caption = base.with_name("Null Agent Fits.caption.txt").read_text()
    assert "in-sample" in caption.lower() and "held out" in caption.lower()
    assert "WSLS (Magnitude)" in legend and "Observed" in legend


def test_fit_figure_labels_are_title_case(tmp_path):
    assert plots.MODEL_LABELS["wsls"] == "WSLS"
    assert plots.MODEL_LABELS["wsls_recency"] == "WSLS (Recency)"
    assert plots.MODEL_LABELS["wsls_magnitude"] == "WSLS (Magnitude)"
    assert plots.MODEL_LABELS["choice_logit"] == "Choice Logit (Held-Out)"


def test_fit_figure_needs_scores(tmp_path):
    observed, implied = stay_tables()
    with pytest.raises(ValueError, match="no scores"):
        plots.plot_null_agent_fits(scores().iloc[:0], observed, implied, tmp_path / "x")


def test_kernel_overlay_has_one_line_per_source_and_a_legend_outside_the_data(tmp_path, figures):
    sources = {
        "Mice": weights_table(4),
        "WSLS": weights_table(4, seed=1),
        "WSLS (Magnitude)": weights_table(4, seed=2),
    }
    base = tmp_path / "Kernels by Source"
    plots.plot_kernels_by_source(sources, base)
    check_files(base)
    suptitle, panels, legend, over = figures["Kernels by Source"]
    assert over == []
    assert len(panels) == 2
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels), panels
    assert legend == ["Mice", "WSLS", "WSLS (Magnitude)"]
    caption = base.with_name("Kernels by Source.caption.txt").read_text()
    assert "Mice: n=4 mice" in caption


def test_kernel_overlay_needs_sources(tmp_path):
    with pytest.raises(ValueError, match="no sources"):
        plots.plot_kernels_by_source({}, tmp_path / "x")
    with pytest.raises(ValueError, match="no weights"):
        plots.plot_kernels_by_source({"A": weights_table().iloc[:0]}, tmp_path / "x")


def test_source_colors_are_one_per_source_and_distinct():
    colors = plots.source_colors(["Mice", "WSLS", "WSLS (Recency)", "Q-Learner"])
    assert len(set(colors.values())) == 4
    assert np.all([c.startswith("#") for c in colors.values()])


def test_kernel_overlay_gives_each_sources_n_when_they_differ(tmp_path, figures):
    sources = {"Mice": weights_table(4), "WSLS": weights_table(2, seed=1)}
    plots.plot_kernels_by_source(sources, tmp_path / "Kernels by Source")
    _, panels, legend, over = figures["Kernels by Source"]
    assert legend == ["Mice (n=4)", "WSLS (n=2)"]
    assert over == []
    assert all("n=4 mice" in p for p in panels)  # the title states the largest
