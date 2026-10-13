import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.behavior_visualization.plot_num_reversals import (
    plot_reversal_curves_across_mice,
    plot_reversal_curves_for_mouse,
)


def test_panels_draw_into_given_axes():
    fig, (left, right) = plt.subplots(1, 2)
    curves = {"good": {"a": np.array([0.0, 1, 2]), "b": np.array([1.0])}}
    assert plot_reversal_curves_across_mice(curves, ax=left, threshold=2) is left
    assert (
        plot_reversal_curves_for_mouse(
            {"total": np.array([1.0, 2])}, ["ses-01", "ses-02"], ax=right
        )
        is right
    )
    assert len(fig.axes) == 2
    mean_line = [ln for ln in left.lines if ln.get_label() == "Good Rev"][0]
    np.testing.assert_allclose(mean_line.get_ydata(), [0.5, 1.0, 1.5])
    plt.close(fig)
