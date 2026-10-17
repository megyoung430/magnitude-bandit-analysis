"""Tests for the rsync alignment diagnostic figure."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_visualization.plot_rsync_alignment import plot_rsync_alignment
from src.sync.rsync import RsyncAligner
from tests import plot_capture  # noqa: F401  (selects the Agg backend)
from tests.sync_fixtures import simulate_pulses


@pytest.fixture
def aligner():
    pulses_A, pulses_B = simulate_pulses(np.random.default_rng(3), n_pulse=100)
    return RsyncAligner(pulses_A, pulses_B)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_draws_into_the_given_axes_without_making_a_figure(aligner):
    fig, axes = plt.subplots(1, 3)
    returned = plot_rsync_alignment(aligner, axes=axes)
    assert list(returned) == list(axes)
    assert plt.get_fignums() == [fig.number]
    assert len(axes[0].patches) > 0  # log-mse histograms
    assert len(axes[1].patches) > 0  # interval discrepancy histogram
    assert len(axes[2].lines) == 1  # matched pulse times


def test_makes_its_own_figure_when_no_axes_are_given(aligner):
    axes = plot_rsync_alignment(aligner)
    assert len(axes) == 3
    assert len(plt.get_fignums()) == 1


def test_wrong_number_of_axes_raises(aligner):
    _, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match="three axes"):
        plot_rsync_alignment(aligner, axes=axes)
