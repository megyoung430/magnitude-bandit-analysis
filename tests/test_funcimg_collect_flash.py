"""Reading a run's registered movies for the flash check and clip, cached for reruns."""

import numpy as np
import pytest

import src.funcimg_analysis.collect_flash as collect_flash
from src.funcimg_analysis.collect_flash import collect_clip, collect_flash_check, valid_box

N_FRAMES, LY, LX, PERIOD = 300, 30, 40, 0.1
CELL = {"ypix": np.repeat(np.arange(13, 17), 4), "xpix": np.tile(np.arange(18, 22), 4)}
SETTINGS = {
    "n_cells": 1,
    "n_peaks": 2,
    "min_peak_z": 3.0,
    "min_peak_gap_s": 5.0,
    "baseline_s": (-2.0, -0.5),
    "response_s": (0.0, 1.0),
    "crop_px": 6,
    "smooth_px": 0.0,
}


@pytest.fixture
def plane(tmp_path):
    """A plane folder with int16 binaries in which the cell doubles at frames 100 and 200."""
    green = np.full((N_FRAMES, LY, LX), 100, dtype=np.int16)
    for peak in (100, 200):
        green[peak : peak + 10, CELL["ypix"], CELL["xpix"]] = 200
    red = np.full((N_FRAMES, LY, LX), 100, dtype=np.int16)
    green.tofile(tmp_path / "data.bin")
    red.tofile(tmp_path / "data_chan2.bin")
    ops = {"nframes": N_FRAMES, "Ly": LY, "Lx": LX, "yrange": [2, 28], "xrange": [4, 36]}
    np.save(tmp_path / "ops.npy", ops, allow_pickle=True)
    np.save(tmp_path / "stat.npy", np.array([CELL], dtype=object), allow_pickle=True)
    return tmp_path


@pytest.fixture
def motion_run(plane):
    z = np.zeros((1, N_FRAMES))
    z[0, [100, 200]] = 8.0
    return {
        "plane_dir": plane,
        "period_s": PERIOD,
        "z_kept": {"green": z, "red": np.zeros((1, N_FRAMES))},
        "stats": {"kept frames": {"green": {"skew": np.array([2.0])}}},
    }


def test_the_flash_check_reads_the_movies_and_finds_the_cell(motion_run, tmp_path):
    [cell] = collect_flash_check(motion_run, cache_dir=tmp_path / "cache", **SETTINGS)
    assert cell["peaks"].tolist() == [100, 200]
    assert cell["image"]["green"][cell["mask"]] == pytest.approx(1.0)
    assert cell["image"]["red"][cell["mask"]] == pytest.approx(0.0)


def test_a_rerun_comes_from_the_cache_without_computing(motion_run, tmp_path, monkeypatch):
    first = collect_flash_check(motion_run, cache_dir=tmp_path / "cache", **SETTINGS)

    def fail(*args, **kwargs):
        raise AssertionError("recomputed instead of reading the cache")

    monkeypatch.setattr(collect_flash, "flash_check", fail)
    again = collect_flash_check(motion_run, cache_dir=tmp_path / "cache", **SETTINGS)
    assert again[0]["peaks"].tolist() == first[0]["peaks"].tolist()


def test_a_changed_setting_is_not_served_from_the_cache(motion_run, tmp_path):
    collect_flash_check(motion_run, cache_dir=tmp_path / "cache", **SETTINGS)
    collect_flash_check(motion_run, cache_dir=tmp_path / "cache", **{**SETTINGS, "crop_px": 4})
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 2


def test_the_clip_is_the_valid_crop_binned(plane, tmp_path):
    clip = collect_clip(plane, 95, 20, 2, 2, cache_dir=tmp_path / "cache")
    assert clip["green"].shape == clip["red"].shape == (10, 13, 16)
    assert clip["origin"] == (2, 4)
    assert clip["green"].max() > 100 and clip["red"].max() == 100


def test_valid_box_is_the_ops_crop():
    assert valid_box({"yrange": [2, 28], "xrange": [4, 36]}) == (slice(2, 28), slice(4, 36))


def test_a_missing_binary_fails_loudly(motion_run, tmp_path):
    (motion_run["plane_dir"] / "data_chan2.bin").unlink()
    with pytest.raises(FileNotFoundError, match="data_chan2.bin"):
        collect_flash_check(motion_run, cache_dir=tmp_path / "cache", **SETTINGS)
