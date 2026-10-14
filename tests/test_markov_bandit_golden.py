"""Golden regression tests for the seeded Markov bandit simulations.

The expected values were produced by the original (pre-refactor) implementation.
They pin the exact outputs *and* the generator state afterwards, so any change
in the order or number of random draws fails here.  Inputs are tiny and built
inline, so nothing depends on lab data.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.tasks.markov_bandit import (
    plot_example_session_markov,
    sample_two_markov_bandits,
    simulate_3armed_blockswaps_markov,
    sweep_markov_metrics,
)

# Each state is (PCG64 state, increment, has_uint32, uinteger).
GOLDEN = {
    "sample": {
        0: (
            [
                [3, 4],
                [3, 4],
                [4, 2],
                [4, 2],
                [4, 2],
                [4, 3],
                [2, 3],
                [2, 4],
                [2, 4],
                [1, 3],
                [1, 3],
                [1, 3],
                [1, 2],
                [1, 2],
                [1, 3],
                [2, 3],
                [2, 3],
                [2, 3],
                [2, 3],
                [2, 3],
                [4, 3],
                [4, 3],
                [4, 3],
                [2, 3],
            ],
            (
                59567054273565164299735393952512885957,
                87136372517582989555478159403783844777,
                0,
                1382418508,
            ),
        ),
        1: (
            [
                [2, 3],
                [2, 3],
                [4, 3],
                [4, 3],
                [4, 3],
                [2, 4],
                [2, 4],
                [3, 4],
                [3, 4],
                [3, 2],
                [3, 2],
                [3, 4],
                [2, 4],
                [2, 3],
                [2, 3],
                [1, 3],
                [1, 3],
                [2, 3],
                [2, 3],
                [4, 2],
                [4, 2],
                [3, 2],
                [3, 4],
                [3, 4],
            ],
            (
                144154990036762603511740819254607262611,
                194290289479364712180083596243593368443,
                0,
                2754483512,
            ),
        ),
        2: (
            [
                [2, 3],
                [2, 3],
                [2, 1],
                [2, 1],
                [3, 1],
                [3, 1],
                [2, 1],
                [2, 1],
                [2, 3],
                [2, 3],
                [2, 1],
                [2, 1],
                [3, 1],
                [3, 2],
                [3, 2],
                [1, 3],
                [1, 3],
                [2, 3],
                [2, 3],
                [2, 3],
                [4, 3],
                [4, 2],
                [3, 2],
                [3, 2],
            ],
            (
                319702083360979218796620462376237003096,
                121863417007658695389390353187995180015,
                0,
                2635894179,
            ),
        ),
    },
    "sample_nodistinct": (
        [
            [3, 1],
            [2, 1],
            [1, 1],
            [2, 2],
            [2, 1],
            [2, 1],
            [3, 2],
            [3, 2],
            [2, 1],
            [1, 1],
            [1, 2],
            [2, 1],
            [2, 2],
            [3, 2],
            [2, 2],
            [3, 3],
            [2, 2],
            [2, 3],
            [2, 2],
            [1, 3],
        ],
        (
            100087997025860722635188448295910536066,
            222003063171874261427395693950637096479,
            1,
            3773043663,
        ),
    ),
    "sim": {
        0: (
            [1.575, 0.475, 3.6363636363636362, 2.568081253059188],
            {
                "vals3": [
                    [3.0, 4.0, 0.0],
                    [3.0, 4.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [0.0, 2.0, 4.0],
                    [0.0, 3.0, 4.0],
                    [0.0, 3.0, 2.0],
                    [0.0, 4.0, 2.0],
                    [0.0, 4.0, 2.0],
                    [0.0, 3.0, 1.0],
                    [0.0, 3.0, 1.0],
                    [3.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [3.0, 0.0, 1.0],
                    [3.0, 0.0, 2.0],
                    [3.0, 0.0, 2.0],
                    [3.0, 0.0, 2.0],
                    [3.0, 0.0, 2.0],
                    [3.0, 0.0, 2.0],
                    [3.0, 0.0, 4.0],
                    [3.0, 0.0, 4.0],
                    [3.0, 0.0, 4.0],
                    [3.0, 2.0, 0.0],
                    [3.0, 2.0, 0.0],
                    [4.0, 1.0, 0.0],
                    [4.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [1.0, 0.0, 3.0],
                    [1.0, 0.0, 3.0],
                    [1.0, 0.0, 2.0],
                    [3.0, 0.0, 2.0],
                    [3.0, 0.0, 1.0],
                    [3.0, 0.0, 1.0],
                    [4.0, 0.0, 1.0],
                    [0.0, 4.0, 1.0],
                ],
                "boundaries": [4, 11, 23, 32, 39],
                "best_arm_series": [
                    1,
                    1,
                    0,
                    0,
                    2,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    2,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    2,
                    2,
                    2,
                    0,
                    0,
                    0,
                    0,
                    1,
                ],
                "best_minus_second": [
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    3.0,
                    3.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                ],
                "best_second_switches_per_block": [1, 1, 1, 1, 1, 0],
            },
            (
                257735685572621886780759442549454666500,
                87136372517582989555478159403783844777,
                0,
                1780933312,
            ),
        ),
        1: (
            [1.525, 0.475, 5.0, 3.3166247903554],
            {
                "vals3": [
                    [2.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [3.0, 0.0, 1.0],
                    [3.0, 0.0, 1.0],
                    [3.0, 0.0, 1.0],
                    [4.0, 0.0, 1.0],
                    [4.0, 0.0, 2.0],
                    [4.0, 0.0, 2.0],
                    [4.0, 0.0, 3.0],
                    [4.0, 0.0, 3.0],
                    [2.0, 0.0, 3.0],
                    [2.0, 4.0, 0.0],
                    [2.0, 4.0, 0.0],
                    [3.0, 4.0, 0.0],
                    [3.0, 4.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [4.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [3.0, 2.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [0.0, 2.0, 4.0],
                    [0.0, 2.0, 4.0],
                    [0.0, 2.0, 4.0],
                    [0.0, 2.0, 3.0],
                    [0.0, 1.0, 3.0],
                    [0.0, 1.0, 3.0],
                    [0.0, 1.0, 3.0],
                    [0.0, 1.0, 2.0],
                ],
                "boundaries": [8, 20, 32],
                "best_arm_series": [
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                ],
                "best_minus_second": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    3.0,
                    2.0,
                    1.0,
                    2.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                ],
                "best_second_switches_per_block": [2, 1, 1, 0],
            },
            (
                266472066238805656059682977835659635775,
                194290289479364712180083596243593368443,
                0,
                3545443161,
            ),
        ),
    },
    "sweep4": [
        [[1.5166666666666666, 1.5416666666666667], [1.6166666666666665, 1.3]],
        [[0.43333333333333335, 0.42500000000000004], [0.43333333333333335, 0.26666666666666666]],
        [[5.238095238095238, 4.0256410256410255], [7.037037037037037, 5.238095238095238]],
        [[3.1742258746387786, 2.8291208866617343], [4.190316816634871, 3.100132610067964]],
    ],
    "sweep5": [
        [[1.5166666666666666, 1.5416666666666667], [1.6166666666666665, 1.3]],
        [[0.43333333333333335, 0.42500000000000004], [0.43333333333333335, 0.26666666666666666]],
        [[5.238095238095238, 4.0256410256410255], [7.037037037037037, 5.238095238095238]],
        [[3.1742258746387786, 2.8291208866617343], [4.190316816634871, 3.100132610067964]],
        [[0.15, 0.24], [0.5833333333333334, 0.7692307692307693]],
    ],
}

# Captured from the pre-refactor implementation, like GOLDEN.
GOLDEN_EXTRA = {
    "plot": {
        "title": "Example session | pud=0.200, max_block_len=20 | gap_mean=1.53, "
        "frac≥2=0.50, run_mean=6.67, run_std=2.62",
        "vals3": [
            [3.0, 4.0, 0.0],
            [3.0, 4.0, 0.0],
            [3.0, 4.0, 0.0],
            [3.0, 4.0, 0.0],
            [3.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 4.0, 3.0],
            [0.0, 4.0, 3.0],
            [0.0, 4.0, 3.0],
            [0.0, 4.0, 3.0],
            [0.0, 4.0, 3.0],
            [0.0, 4.0, 2.0],
            [0.0, 3.0, 2.0],
            [0.0, 3.0, 2.0],
            [0.0, 3.0, 2.0],
            [0.0, 3.0, 2.0],
            [0.0, 3.0, 2.0],
            [0.0, 1.0, 3.0],
            [0.0, 1.0, 3.0],
            [0.0, 1.0, 3.0],
            [3.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [3.0, 4.0, 0.0],
            [3.0, 4.0, 0.0],
            [3.0, 4.0, 0.0],
            [3.0, 4.0, 0.0],
            [3.0, 4.0, 0.0],
            [2.0, 4.0, 0.0],
            [2.0, 0.0, 4.0],
            [2.0, 0.0, 3.0],
            [2.0, 0.0, 3.0],
            [2.0, 0.0, 3.0],
            [2.0, 0.0, 3.0],
            [4.0, 0.0, 3.0],
            [4.0, 0.0, 2.0],
            [4.0, 0.0, 2.0],
            [4.0, 0.0, 2.0],
            [4.0, 0.0, 2.0],
            [4.0, 0.0, 2.0],
            [0.0, 4.0, 1.0],
            [0.0, 4.0, 1.0],
            [0.0, 3.0, 1.0],
            [0.0, 3.0, 1.0],
            [0.0, 3.0, 1.0],
            [0.0, 3.0, 1.0],
            [0.0, 3.0, 2.0],
            [0.0, 4.0, 2.0],
        ],
        "boundaries": [10, 24, 41, 52],
    },
    "sweep11": [
        [[1.513636363636364, 1.481818181818182], [1.5818181818181818, 1.468181818181818]],
        [[0.41590909090909095, 0.4068181818181818], [0.4772727272727273, 0.4068181818181818]],
        [[5.222353404171586, 4.239669421487604], [6.562770562770563, 5.548340548340549]],
        [[2.878278510551962, 2.5918813733750543], [2.908410408101901, 3.4709454457191855]],
        [[0.4126984126984127, 0.41333333333333333], [0.25925925925925924, 0.6875]],
    ],
    "boundary_at_T": {
        0: (
            [1.575, 0.475, 5.714285714285714, 3.806546455564065],
            {
                "vals3": [
                    [3.0, 4.0, 0.0],
                    [3.0, 4.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 4.0, 0.0],
                    [2.0, 4.0, 0.0],
                    [1.0, 3.0, 0.0],
                    [1.0, 3.0, 0.0],
                    [1.0, 3.0, 0.0],
                    [1.0, 2.0, 0.0],
                    [1.0, 2.0, 0.0],
                    [1.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [4.0, 0.0, 3.0],
                    [4.0, 0.0, 3.0],
                    [4.0, 0.0, 3.0],
                    [2.0, 0.0, 3.0],
                    [2.0, 0.0, 3.0],
                    [1.0, 0.0, 4.0],
                    [1.0, 0.0, 4.0],
                    [1.0, 0.0, 3.0],
                    [1.0, 0.0, 3.0],
                    [1.0, 0.0, 2.0],
                    [1.0, 0.0, 2.0],
                    [3.0, 0.0, 2.0],
                    [3.0, 0.0, 1.0],
                    [3.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [2.0, 0.0, 3.0],
                    [1.0, 0.0, 3.0],
                    [1.0, 0.0, 3.0],
                    [1.0, 0.0, 4.0],
                    [1.0, 0.0, 4.0],
                ],
                "boundaries": [20],
                "best_arm_series": [
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    0,
                    0,
                    0,
                    0,
                    2,
                    2,
                    2,
                    2,
                    2,
                ],
                "best_minus_second": [
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    3.0,
                    3.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                ],
                "best_second_switches_per_block": [2, 3],
            },
            (
                2701282021809918926030211926021887878,
                87136372517582989555478159403783844777,
                0,
                3982098483,
            ),
        ),
        1: (
            [1.525, 0.475, 5.714285714285714, 5.174505793221381],
            {
                "vals3": [
                    [2.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [4.0, 1.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 0.0, 4.0],
                    [2.0, 0.0, 4.0],
                    [3.0, 0.0, 4.0],
                    [3.0, 0.0, 4.0],
                    [4.0, 0.0, 2.0],
                    [4.0, 0.0, 2.0],
                    [4.0, 0.0, 1.0],
                    [3.0, 0.0, 1.0],
                    [3.0, 0.0, 2.0],
                    [4.0, 0.0, 2.0],
                    [4.0, 0.0, 3.0],
                    [4.0, 0.0, 3.0],
                    [4.0, 0.0, 2.0],
                    [4.0, 0.0, 2.0],
                    [4.0, 0.0, 2.0],
                    [3.0, 0.0, 2.0],
                    [3.0, 0.0, 1.0],
                    [3.0, 0.0, 1.0],
                    [3.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                ],
                "boundaries": [20],
                "best_arm_series": [
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    2,
                    2,
                    2,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "best_minus_second": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    3.0,
                    2.0,
                    1.0,
                    2.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                ],
                "best_second_switches_per_block": [4, 1],
            },
            (
                141057997973854463245841633089599386781,
                194290289479364712180083596243593368443,
                0,
                3842775367,
            ),
        ),
    },
}

SWEEP_PRINTS = "".join(
    f"Done pud={pud:.4f}, max_block_len={mbl}\n" for mbl in (12, 15) for pud in (0.1, 0.3)
)


def generator_state(rng: np.random.Generator) -> tuple:
    state = rng.bit_generator.state
    return (state["state"]["state"], state["state"]["inc"], state["has_uint32"], state["uinteger"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sample_two_markov_bandits_matches_golden(seed):
    rng = np.random.default_rng(seed)
    out = sample_two_markov_bandits(24, 0.25, (1, 2, 3, 4), rng, min_hold=2)
    expected, state = GOLDEN["sample"][seed]
    np.testing.assert_array_equal(out, np.array(expected))
    assert generator_state(rng) == state


def test_sample_two_markov_bandits_without_distinctness_matches_golden():
    rng = np.random.default_rng(3)
    out = sample_two_markov_bandits(20, 0.3, (1, 2, 3), rng, min_hold=1, enforce_distinct=False)
    expected, state = GOLDEN["sample_nodistinct"]
    np.testing.assert_array_equal(out, np.array(expected))
    assert generator_state(rng) == state


@pytest.mark.parametrize("seed", [0, 1])
def test_simulate_3armed_blockswaps_markov_matches_golden(seed):
    rng = np.random.default_rng(seed)
    *metrics, session = simulate_3armed_blockswaps_markov(
        T=40,
        pud=0.2,
        max_block_len=12,
        min_block_len=4,
        n_blocks_draw=20,
        rng=rng,
        min_hold=2,
        return_session=True,
    )
    expected_metrics, expected_session, state = GOLDEN["sim"][seed]
    assert metrics == expected_metrics
    assert set(session) == set(expected_session)
    for key, expected in expected_session.items():
        np.testing.assert_array_equal(session[key], np.array(expected), err_msg=key)
    assert generator_state(rng) == state


SWEEP_KWARGS = dict(
    pud_values=np.array([0.1, 0.3]), max_block_lens=np.array([12, 15]), T=40, n_sims=3, seed=0
)


def test_sweep_markov_metrics_matches_golden(capsys):
    heatmaps = sweep_markov_metrics(**SWEEP_KWARGS)
    assert len(heatmaps) == 4
    for got, expected in zip(heatmaps, GOLDEN["sweep4"], strict=True):
        np.testing.assert_array_equal(got, np.array(expected))
    assert capsys.readouterr().out == SWEEP_PRINTS


def test_sweep_markov_metrics_with_switches_matches_golden(capsys):
    heatmaps = sweep_markov_metrics(include_switches=True, **SWEEP_KWARGS)
    assert len(heatmaps) == 5
    for got, expected in zip(heatmaps, GOLDEN["sweep5"], strict=True):
        np.testing.assert_array_equal(got, np.array(expected))
    assert capsys.readouterr().out == SWEEP_PRINTS


def test_sweep_returns_independent_arrays():
    heatmaps = sweep_markov_metrics(**SWEEP_KWARGS)
    assert all(h.base is None for h in heatmaps)


# Non-positive reward values can make the inactive arm win the argmax at a block
# boundary; the role swap then looks up a stream that does not exist.
def test_non_positive_values_raise_key_error():
    with pytest.raises(KeyError) as excinfo:
        simulate_3armed_blockswaps_markov(
            T=60,
            pud=0.3,
            max_block_len=8,
            min_block_len=2,
            val_list=(-1, 0, 1),
            min_hold=1,
            rng=np.random.default_rng(0),
        )
    assert excinfo.value.args == (2,)


def test_plot_example_session_markov_draws_expected_figure(monkeypatch):
    plt.close("all")
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    session = plot_example_session_markov(pud=0.2, max_block_len=20, T=60, seed=0)

    assert len(shown) == 1
    (ax,) = shown[0].axes
    n_boundaries = len(session["boundaries"])
    assert len(ax.lines) == 3 + n_boundaries
    assert ax.get_xlim() == (0, 59)
    assert ax.get_ylim() == (-0.5, 4.5)
    assert ax.get_xlabel() == "Trial"
    expected = GOLDEN_EXTRA["plot"]
    assert ax.get_title() == expected["title"]
    np.testing.assert_array_equal(session["vals3"], np.array(expected["vals3"]))
    np.testing.assert_array_equal(session["boundaries"], np.array(expected["boundaries"]))
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels[:3] == ["Arm 0", "Arm 1", "Arm 2"]
    assert ("Block" in labels) == (n_boundaries > 0)
    plt.close("all")


def test_plot_example_session_markov_bad_pud_fails_before_the_legend(monkeypatch):
    # The title is formatted from pud after the traces are drawn and before the legend.
    plt.close("all")
    monkeypatch.setattr(plt, "show", lambda: None)
    with pytest.raises(ValueError, match="Unknown format code 'f'"):
        plot_example_session_markov(pud="0.2", max_block_len=20, T=60, seed=0)
    (ax,) = plt.gcf().axes
    assert len(ax.lines) >= 3
    assert ax.get_legend() is None
    plt.close("all")


# With n_sims > 8 the cell mean differs in the last bit between plain left-to-right
# addition and np.mean's pairwise summation.
def test_sweep_markov_metrics_with_many_simulations_matches_golden():
    heatmaps = sweep_markov_metrics(include_switches=True, **{**SWEEP_KWARGS, "n_sims": 11})
    for got, expected in zip(heatmaps, GOLDEN_EXTRA["sweep11"], strict=True):
        np.testing.assert_array_equal(got, np.array(expected))


# Block lengths of exactly 20 put a boundary at trial 40 == T, which is dropped:
# a boundary must fall strictly inside the session.
@pytest.mark.parametrize("seed", [0, 1])
def test_block_boundary_at_session_end_is_dropped(seed):
    rng = np.random.default_rng(seed)
    *metrics, session = simulate_3armed_blockswaps_markov(
        T=40,
        pud=0.2,
        max_block_len=20,
        min_block_len=20,
        n_blocks_draw=5,
        rng=rng,
        min_hold=2,
        return_session=True,
    )
    expected_metrics, expected_session, state = GOLDEN_EXTRA["boundary_at_T"][seed]
    assert metrics == expected_metrics
    assert set(session) == set(expected_session)
    for key, expected in expected_session.items():
        np.testing.assert_array_equal(session[key], np.array(expected), err_msg=key)
    assert list(session["boundaries"]) == [20]
    assert generator_state(rng) == state
