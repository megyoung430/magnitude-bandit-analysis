"""Run the motion check on every run of section 2d: load each plane's ``ops`` and call the check."""

from pathlib import Path

import numpy as np

from src.funcimg_analysis.event_frames import frame_period_s
from src.funcimg_analysis.motion_check import motion_check


def load_ops(plane_dir) -> dict:
    """Suite2p's ``ops.npy`` of a plane, as a dict."""
    return np.load(Path(plane_dir) / "ops.npy", allow_pickle=True).item()


def collect_motion_check(
    residual_runs: list[dict], planes: dict[tuple[str, str], Path], **settings
) -> list[dict]:
    """:func:`motion_check` of each ``collect_residual_dff`` run, on its background-corrected ΔF/F0.

    Args:
        residual_runs (list[dict]): ``collect_residual_dff`` results (``session``, ``run``,
            ``time_s`` and ``channels[channel]["dff"]``).
        planes (dict): ``(session, run)`` to the plane folder holding ``ops.npy``.
        **settings: Passed on to :func:`motion_check`.

    Returns:
        list[dict]: Each run's result with ``session``, ``run``, ``plane_dir`` and ``period_s``.

    Raises:
        KeyError: If a run has no plane folder in *planes*.
    """

    def check(run: dict) -> dict:
        where = (run["session"], run["run"])
        if where not in planes:
            raise KeyError(f"No plane folder for {where[0]} {where[1]}.")
        period = frame_period_s(run["time_s"])
        dff = {channel: run["channels"][channel]["dff"] for channel in ("green", "red")}
        result = motion_check(dff, load_ops(planes[where]), period, **settings)
        return {
            "session": where[0],
            "run": where[1],
            "plane_dir": planes[where],
            "period_s": period,
            **result,
        }

    return [check(run) for run in residual_runs]
