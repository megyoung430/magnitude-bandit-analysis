"""Synthetic sync-pulse trains for the rsync and pyControl-time tests."""

import numpy as np


def kept_pulse_indices(n_pulse: int) -> tuple[np.ndarray, np.ndarray]:
    """Original pulse numbers that survive in A and in B when pulses go missing.

    A loses its first 5 % and the 21-33 % stretch; B loses the 74-85 % stretch and its last
    5 %. Both lists stay sorted, so a pulse kept by both appears in the same order in each.

    Args:
        n_pulse (int): Number of pulses before any are removed.

    Returns:
        tuple[np.ndarray, np.ndarray]: Surviving pulse numbers for A and for B.
    """
    keep_A = np.r_[int(n_pulse * 0.05) : int(n_pulse * 0.21), int(n_pulse * 0.33) : n_pulse]
    keep_B = np.r_[0 : int(n_pulse * 0.74), int(n_pulse * 0.85) : int(n_pulse * 0.95)]
    return keep_A, keep_B


def simulate_pulses(
    rng: np.random.Generator,
    n_pulse: int = 1000,
    interval: tuple[int, int] = (100, 1900),
    units_B: float = 2.0,
    noise_SD: float = 2.0,
    missing_pulses: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Pulse trains seen by two systems whose clocks drift apart.

    System A sees pulses at random integer intervals (ms). System B sees the same pulses in
    units of ``1 / units_B`` ms, plus a random-walk clock error of *noise_SD* ms per pulse.
    With *missing_pulses*, each train loses the stretches listed by
    :func:`kept_pulse_indices` and A's later pulses are shifted by 2e5 ms, as if A had
    restarted its clock.

    Args:
        rng (np.random.Generator): Source of randomness; seed it for a reproducible train.
        n_pulse (int): Number of pulses before any are removed.
        interval (tuple[int, int]): Range ``[low, high)`` of inter-pulse intervals on A (ms).
        units_B (float): B's timestamps per A millisecond.
        noise_SD (float): Standard deviation of B's per-pulse clock error (A ms).
        missing_pulses (bool): Remove stretches of pulses from each train.

    Returns:
        tuple[np.ndarray, np.ndarray]: Pulse times on A and on B.
    """
    pulse_times_A = np.cumsum(rng.integers(*interval, size=n_pulse)).astype(float)
    clock_error = np.cumsum(rng.normal(scale=noise_SD, size=n_pulse))
    pulse_times_B = units_B * (pulse_times_A + clock_error)
    if missing_pulses:
        keep_A, keep_B = kept_pulse_indices(n_pulse)
        restarted = keep_A >= int(n_pulse * 0.33)
        pulse_times_A = pulse_times_A[keep_A] + 2e5 * restarted
        pulse_times_B = pulse_times_B[keep_B]
    return pulse_times_A, pulse_times_B
