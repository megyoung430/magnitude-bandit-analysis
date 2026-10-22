"""Running the motion check over section 2d's runs: ops are read from each run's plane folder."""

import numpy as np
import pytest

from src.funcimg_analysis.collect_motion import collect_motion_check, load_ops
from tests.test_funcimg_motion_check import N_FRAMES, PERIOD, SETTINGS, _artefact_run


def _residual_run(dff, session="ses-1", run="run-01"):
    return {
        "session": session,
        "run": run,
        "time_s": 5.0 + np.arange(N_FRAMES) * PERIOD,
        "channels": {channel: {"dff": traces} for channel, traces in dff.items()},
    }


def test_each_run_is_checked_with_its_own_ops(tmp_path):
    ops, dff = _artefact_run(np.random.default_rng(0))
    np.save(tmp_path / "ops.npy", ops, allow_pickle=True)
    planes = {("ses-1", "run-01"): tmp_path}
    [result] = collect_motion_check([_residual_run(dff)], planes, **SETTINGS)
    assert (result["session"], result["run"], result["plane_dir"]) == ("ses-1", "run-01", tmp_path)
    assert result["period_s"] == pytest.approx(PERIOD)
    assert result["motion"]["xoff"] == pytest.approx(ops["xoff"])
    assert result["cut_s"] > 0


def test_a_run_without_a_plane_folder_fails_loudly(tmp_path):
    _, dff = _artefact_run(np.random.default_rng(0))
    with pytest.raises(KeyError, match="ses-1 run-01"):
        collect_motion_check([_residual_run(dff)], {}, **SETTINGS)


def test_load_ops_returns_the_saved_dict(tmp_path):
    np.save(tmp_path / "ops.npy", {"Ly": 4}, allow_pickle=True)
    assert load_ops(tmp_path) == {"Ly": 4}
