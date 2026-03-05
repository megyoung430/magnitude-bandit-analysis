"""Comparison plots between GP and Markov bandit simulations."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_run_mean_vs_std(
    gp_means: np.ndarray,
    gp_stds: np.ndarray,
    markov_means: np.ndarray,
    markov_stds: np.ndarray,
    sig_values: np.ndarray,
    pud_values: np.ndarray,
) -> None:
    """Scatter-plot mean vs std dev of best-arm run length for both bandit types.

    Points are colour-coded by their respective parameter value (sig for GP,
    p_ud for Markov).

    Args:
        gp_means: Mean run lengths for each GP sig value.
        gp_stds: Std dev of run lengths for each GP sig value.
        markov_means: Mean run lengths for each Markov pud value.
        markov_stds: Std dev of run lengths for each Markov pud value.
        sig_values: GP length-scale values (used for colour mapping).
        pud_values: Markov transition probabilities (used for colour mapping).
    """
    plt.figure(figsize=(7, 6))

    plt.scatter(
        gp_means, gp_stds,
        c=sig_values, cmap="Blues",
        label="GP (colored by sig)",
        s=70, edgecolor="k",
    )
    plt.scatter(
        markov_means, markov_stds,
        c=pud_values, cmap="Reds",
        label="Markov (colored by p_ud)",
        s=70, marker="s", edgecolor="k",
    )

    plt.xlabel("Average Run Length")
    plt.ylabel("Std Dev of Run Length")
    plt.title("Run Length Statistics (max_block_len = 80)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
