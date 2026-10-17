# Align the timestamps of two recording systems using sync pulses with random inter-pulse
# intervals. https://pycontrol.readthedocs.io/en/latest/user-guide/synchronisation
# Dependencies: Python 3, NumPy, scikit-learn.
# (c) Thomas Akam 2018-2022. Released under the GPL-3 open source licence.
# Ported to a plotting-free class with input validation by the magnitude-bandit-analysis lab.

"""Convert timestamps between two recording systems from shared random-interval sync pulses."""

import warnings

import numpy as np
from sklearn.mixture import GaussianMixture


class RsyncError(Exception):
    """The two pulse trains could not be matched to each other."""


class RsyncWarning(UserWarning):
    """The two pulse trains could not be matched, and the caller asked not to raise."""


_NO_MATCH_MESSAGE = "No match found between inter-pulse interval sequences."


def min_pulses_required(chunk_size: int = 5) -> int:
    """Fewest pulses a train may hold for the aligner to form two chunks from it."""
    return 2 * chunk_size + 1


def _as_pulse_times(pulse_times, name: str, min_pulses: int) -> np.ndarray:
    """Return *pulse_times* as a float array after checking it can be aligned.

    Args:
        pulse_times (array-like): Sync pulse times of one system.
        name (str): Names the system ("A" or "B") in error messages.
        min_pulses (int): Fewest pulses the train may hold.

    Returns:
        np.ndarray: The pulse times as floats.

    Raises:
        ValueError: If the train is not 1-D, not finite, not strictly increasing, or shorter
            than *min_pulses*.
    """
    times = np.asarray(pulse_times, dtype=float)
    if times.ndim != 1:
        raise ValueError(f"pulse_times_{name} must be 1-D, got shape {times.shape}.")
    if len(times) < min_pulses:
        raise ValueError(
            f"pulse_times_{name} has {len(times)} pulses; at least {min_pulses} are needed "
            "to form two chunks."
        )
    if not np.isfinite(times).all():
        raise ValueError(f"pulse_times_{name} must be finite (no NaN or inf).")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"pulse_times_{name} must be strictly increasing.")
    return times


def _estimate_units(pulse_times_A: np.ndarray, pulse_times_B: np.ndarray) -> tuple[float, float]:
    """Units of B relative to A (A is 1), from the mean of each train's ordinary intervals."""
    raw_intervals_A = np.diff(pulse_times_A)
    raw_intervals_B = np.diff(pulse_times_B)
    # Intervals over 3x the median are taken to span missing pulses and are left out.
    good_intervals_A = raw_intervals_A[raw_intervals_A < 3 * np.median(raw_intervals_A)]
    good_intervals_B = raw_intervals_B[raw_intervals_B < 3 * np.median(raw_intervals_B)]
    return 1, np.mean(good_intervals_A) / np.mean(good_intervals_B)


def _align_chunks(
    intervals_A: np.ndarray, intervals_B: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Best-matching position in B for each chunk of A, with the two lowest errors per chunk.

    Args:
        intervals_A (np.ndarray): Inter-pulse intervals of A (ms).
        intervals_B (np.ndarray): Inter-pulse intervals of B (ms).
        chunk_size (int): Intervals per chunk.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Start index of each A chunk, start
        index of its best match in B, and the mean squared error of the best and of the second
        best alignment.
    """
    n_pulses_A = len(intervals_A) + 1
    starts_A = np.arange(0, n_pulses_A - chunk_size, chunk_size)
    starts_B = np.zeros(starts_A.shape, int)
    min_mse = np.zeros(starts_A.shape)
    second_mse = np.zeros(starts_A.shape)
    ones_chunk = np.ones(chunk_size)
    intervals_B2 = intervals_B**2
    for i, start_A in enumerate(starts_A):
        chunk_A = intervals_A[start_A : start_A + chunk_size]
        # Expanding (B - A)^2 lets one correlation score every offset in B at once.
        mse = (
            np.correlate(intervals_B2, ones_chunk, mode="valid")
            + np.sum(chunk_A**2)
            - 2 * np.correlate(intervals_B, chunk_A, mode="valid")
        ) / chunk_size
        starts_B[i] = np.argmin(mse)
        sorted_mse = np.sort(mse)
        min_mse[i] = sorted_mse[0]
        second_mse[i] = sorted_mse[1]
    return starts_A, starts_B, min_mse, second_mse


def _classify_chunks(
    min_mse: np.ndarray, second_mse: np.ndarray, random_state: int
) -> tuple[np.ndarray, np.ndarray, GaussianMixture]:
    """Label chunks as matched or not with a 2-component mixture on log mean squared error.

    Args:
        min_mse (np.ndarray): Mean squared error of each chunk's best alignment.
        second_mse (np.ndarray): Mean squared error of each chunk's second best alignment.
        random_state (int): Seed for the mixture's initialisation.

    Returns:
        tuple[np.ndarray, np.ndarray, GaussianMixture]: Log mean squared error of every best
        alignment followed by every second best, whether each of those is in the low-error
        (matched) component, and the fitted mixture.

    Raises:
        ValueError: If a log error is not finite, which would misalign the labels.
    """
    chunk_mse = np.hstack([min_mse, second_mse])
    # An exact match has zero error, which has no logarithm; use the smallest nonzero error.
    chunk_mse[chunk_mse == 0] = np.min(chunk_mse[chunk_mse != 0])
    log_mse = np.log(chunk_mse)
    if not np.isfinite(log_mse).all():
        raise ValueError("Mean squared error of a chunk alignment is negative or not finite.")
    log_mse = log_mse.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type="spherical", random_state=random_state)
    gmm.fit(log_mse)
    valid_matches = gmm.predict(log_mse) == np.argmin(gmm.means_)
    return log_mse.ravel(), valid_matches, gmm


def _corresponding_times(
    pulse_times_A: np.ndarray,
    pulse_times_B: np.ndarray,
    starts_A: np.ndarray,
    starts_B: np.ndarray,
    is_match: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partner time on the other system for every pulse in matched chunks, NaN elsewhere.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A pulse time at each B pulse, B pulse time
        at each A pulse, and the index of the B pulse matched to each A pulse (-1 if none).
    """
    cor_times_A = np.full(pulse_times_B.shape, np.nan)
    cor_times_B = np.full(pulse_times_A.shape, np.nan)
    matched_index_B = np.full(pulse_times_A.shape, -1)
    for start_A, start_B, valid in zip(starts_A, starts_B, is_match, strict=True):
        if valid:
            cor_times_A[start_B : start_B + chunk_size] = pulse_times_A[
                start_A : start_A + chunk_size
            ]
            cor_times_B[start_A : start_A + chunk_size] = pulse_times_B[
                start_B : start_B + chunk_size
            ]
            matched_index_B[start_A : start_A + chunk_size] = np.arange(
                start_B, start_B + chunk_size
            )
    return cor_times_A, cor_times_B, matched_index_B


def _is_aligned(gmm: GaussianMixture, cor_times_A: np.ndarray, cor_times_B: np.ndarray) -> bool:
    """Whether matched and unmatched chunks separate cleanly and matches keep their order."""
    # Difference in mixture means must exceed 3x the sum of the standard deviations.
    separation_ok = np.abs(gmm.means_[0] - gmm.means_[1])[0] > 3 * np.sum(np.sqrt(gmm.covariances_))
    # Corresponding times must increase on both systems.
    order_ok = (np.nanmin(np.diff(cor_times_A)) > 0) and (np.nanmin(np.diff(cor_times_B)) > 0)
    return bool(separation_ok and order_ok)


class RsyncAligner:
    """Convert timestamps between two recording systems using shared random-interval pulses.

    Typically the pulses are generated by pyControl's Rsync hardware object and recorded by
    both pyControl and a second system. Give the aligner the pulse times each system recorded;
    it works out which pulses correspond by matching short chunks of the inter-pulse interval
    sequence of A against B (minimum mean squared error), then converts timestamps in either
    direction with :meth:`A_to_B` and :meth:`B_to_A`. If the systems count time in different
    units, give them with *units_A* and *units_B*, or set either to ``'auto'`` to estimate the
    units of B relative to A from the mean inter-pulse intervals (the explicit values of the
    other argument are then ignored).

    Differences from the upstream ``Rsync_aligner`` (Thomas Akam, GPL-3): plotting lives in
    ``src.behavior_visualization.plot_rsync_alignment``, the inputs are validated, the mixture
    model is seeded, and the order check tests both ``cor_times_A`` and ``cor_times_B``
    (upstream tests ``cor_times_A`` twice, so a mis-ordered B was never caught).

    Attributes:
        pulse_times_A (np.ndarray): Sync pulse times recorded by A.
        pulse_times_B (np.ndarray): Sync pulse times recorded by B.
        cor_times_A (np.ndarray): A pulse time matched to each B pulse, NaN if unmatched.
        cor_times_B (np.ndarray): B pulse time matched to each A pulse, NaN if unmatched.
        matched_index_B (np.ndarray): Index in B of the pulse matched to each A pulse, -1 if
            unmatched.
        units_A (float): Time unit of A in ms.
        units_B (float): Time unit of B in ms (relative to A when estimated).
        log_mse (np.ndarray): Log mean squared error of each chunk's best alignment followed
            by that of its second best.
        valid_matches (np.ndarray): Whether each entry of *log_mse* is in the matched component.
        dAdB (float): Empirical ratio of A to B time units from matched intervals.
        first_matched_time_A, last_matched_time_A, first_matched_time_B, last_matched_time_B
            (float): Times of the first and last matched pulse on each system (NaN if none).
    """

    def __init__(
        self,
        pulse_times_A,
        pulse_times_B,
        units_A="auto",
        units_B="auto",
        chunk_size: int = 5,
        raise_exception: bool = True,
        random_state: int = 0,
    ):
        """Match the pulses of A to those of B.

        Args:
            pulse_times_A (array-like): Times when sync pulses occurred, recorded by A.
            pulse_times_B (array-like): Times when sync pulses occurred, recorded by B.
            units_A (float | str): Time unit of A in milliseconds, e.g. 1000 if A counts
                seconds. ``'auto'`` estimates the units of B relative to A.
            units_B (float | str): Time unit of B in milliseconds, or ``'auto'``.
            chunk_size (int): Inter-pulse intervals per chunk that are matched together.
            raise_exception (bool): Raise :class:`RsyncError` if no match is found; otherwise
                emit an :class:`RsyncWarning`.
            random_state (int): Seed for the mixture model, so a given pair of trains always
                gives the same alignment.

        Raises:
            ValueError: If either train is not 1-D, finite and strictly increasing, holds fewer
                than ``2 * chunk_size + 1`` pulses, or *chunk_size* is not a positive integer.
            RsyncError: If no match is found and *raise_exception* is True.
        """
        if int(chunk_size) != chunk_size or chunk_size < 1:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}.")
        min_pulses = min_pulses_required(chunk_size)
        pulse_times_A = _as_pulse_times(pulse_times_A, "A", min_pulses)
        pulse_times_B = _as_pulse_times(pulse_times_B, "B", min_pulses)
        if units_A == "auto" or units_B == "auto":
            units_A, units_B = _estimate_units(pulse_times_A, pulse_times_B)

        # Find alignments of chunks which minimise the sum of squared errors.
        starts_A, starts_B, min_mse, second_mse = _align_chunks(
            np.diff(pulse_times_A) * units_A, np.diff(pulse_times_B) * units_B, chunk_size
        )
        self.log_mse, self.valid_matches, gmm = _classify_chunks(min_mse, second_mse, random_state)
        is_match = self.valid_matches[: len(starts_A)]  # first half: best alignments
        self.cor_times_A, self.cor_times_B, self.matched_index_B = _corresponding_times(
            pulse_times_A, pulse_times_B, starts_A, starts_B, is_match, chunk_size
        )
        self.pulse_times_A = pulse_times_A
        self.pulse_times_B = pulse_times_B
        self.units_A = units_A
        self.units_B = units_B
        self._set_extrapolation_terms()

        if not is_match.any() or not _is_aligned(gmm, self.cor_times_A, self.cor_times_B):
            if raise_exception:
                raise RsyncError(_NO_MATCH_MESSAGE)
            warnings.warn(f"Rsync warning: {_NO_MATCH_MESSAGE}", RsyncWarning, stacklevel=2)

    def _set_extrapolation_terms(self) -> None:
        """Set the matched-pulse ends and clock ratio used to extrapolate past them."""
        matched_A = self.cor_times_A[~np.isnan(self.cor_times_A)]
        matched_B = self.cor_times_B[~np.isnan(self.cor_times_B)]
        if len(matched_A) == 0:
            self.dAdB = self.first_matched_time_A = self.last_matched_time_A = np.nan
            self.first_matched_time_B = self.last_matched_time_B = np.nan
            return
        diff_cor_times_B = np.diff(self.cor_times_B)
        has_interval = ~np.isnan(diff_cor_times_B)
        # Empirical units_A / units_B from matched intervals.
        self.dAdB = np.sum(np.diff(self.pulse_times_A)[has_interval]) / np.sum(
            diff_cor_times_B[has_interval]
        )
        self.first_matched_time_A = matched_A[0]
        self.last_matched_time_A = matched_A[-1]
        self.first_matched_time_B = matched_B[0]
        self.last_matched_time_B = matched_B[-1]

    def interval_discrepancy_ms(self) -> np.ndarray:
        """Difference between A and B inter-pulse intervals across matched consecutive pulses.

        Only pairs of pulses that are neighbours on both systems count, so a gap of missing
        pulses is never mistaken for a timing error. The values are in milliseconds when the
        units were given explicitly, and in A's units when they were estimated.

        Returns:
            np.ndarray: A interval minus B interval for every such pair.
        """
        index_B = self.matched_index_B
        neighbours = (index_B[:-1] >= 0) & (index_B[1:] == index_B[:-1] + 1)
        intervals_A = np.diff(self.pulse_times_A)[neighbours] * self.units_A
        intervals_B = np.diff(self.pulse_times_B)[index_B[:-1][neighbours]] * self.units_B
        return intervals_A - intervals_B

    def A_to_B(self, times_A, extrapolate: bool = False) -> np.ndarray:
        """Convert times in the A reference frame to the B reference frame.

        Args:
            times_A (array-like): Times on system A.
            extrapolate (bool): Continue the clock beyond the first and last matched pulse;
                otherwise those times are NaN.

        Returns:
            np.ndarray: The times on system B (NaN where unmatched and not extrapolating).
        """
        times_A = np.asarray(times_A, dtype=float)
        times_B = np.interp(
            times_A, self.pulse_times_A, self.cor_times_B, left=np.nan, right=np.nan
        )
        if extrapolate:
            pre = times_A < self.first_matched_time_A
            times_B[pre] = (times_A[pre] - self.first_matched_time_A) / self.dAdB + (
                self.first_matched_time_B
            )
            post = times_A > self.last_matched_time_A
            times_B[post] = (times_A[post] - self.last_matched_time_A) / self.dAdB + (
                self.last_matched_time_B
            )
        return times_B

    def B_to_A(self, times_B, extrapolate: bool = False) -> np.ndarray:
        """Convert times in the B reference frame to the A reference frame.

        Args:
            times_B (array-like): Times on system B.
            extrapolate (bool): Continue the clock beyond the first and last matched pulse;
                otherwise those times are NaN.

        Returns:
            np.ndarray: The times on system A (NaN where unmatched and not extrapolating).
        """
        times_B = np.asarray(times_B, dtype=float)
        times_A = np.interp(
            times_B, self.pulse_times_B, self.cor_times_A, left=np.nan, right=np.nan
        )
        if extrapolate:
            pre = times_B < self.first_matched_time_B
            times_A[pre] = (times_B[pre] - self.first_matched_time_B) * self.dAdB + (
                self.first_matched_time_A
            )
            post = times_B > self.last_matched_time_B
            times_A[post] = (times_B[post] - self.last_matched_time_B) * self.dAdB + (
                self.last_matched_time_A
            )
        return times_A


# Name used by the lab's existing ephys code (GridMaze) and by the upstream pyControl tools.
Rsync_aligner = RsyncAligner
