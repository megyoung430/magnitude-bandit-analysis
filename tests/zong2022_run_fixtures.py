"""A synthetic derivatives tree for the Zong et al. 2022 run-level tests and the notebook check.

One session, one run, ``without_destriping`` registered plane (binaries + ``ops.npy``), the
run's Cellpose segmentation and the image it was drawn on, ``frame_times.npz`` and a
``source_tiff.json``. Three ROIs carry planted transients: ROI 0 fires at every rewarded
outcome, ROI 2 at every unrewarded one, ROI 1 at unrelated times.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

from tests.zong2022_fixtures import PERIOD_S, planted_cell

LY = LX = 128
SESSION = "ses-224_date-20260809"
NEUROPIL_LEVEL = 100.0
ROI_SLICES = (
    (slice(20, 28), slice(20, 28)),
    (slice(60, 68), slice(70, 78)),
    (slice(100, 108), slice(30, 38)),
)
REWARDED_FRAMES = np.array([500, 950, 1400, 1850])
UNREWARDED_FRAMES = np.array([725, 1175])
UNRELATED_FRAMES = np.array([600, 1050, 1500, 1950])
FRAME_T0 = 20.0
PULSES = (3, 1, 4, 2, 0, 0)  # rewarded outcomes first, then the unrewarded ones


def texture(seed=0):
    image = gaussian_filter(np.random.default_rng(seed).random((LY, LX)), 2.0)
    return (image - image.min()) / np.ptp(image)


def cell_labels():
    labels = np.zeros((LY, LX), int)
    for k, (ys, xs) in enumerate(ROI_SLICES, start=1):
        labels[ys, xs] = k
    return labels


def planted_traces(n_frames, seed=0):
    """Fcorr of the three ROIs, shifted so Fcell - 0.7 * Fnp gives it back (Fnp is 100)."""
    rng = np.random.default_rng(seed)
    spikes = (REWARDED_FRAMES, UNRELATED_FRAMES, UNREWARDED_FRAMES)
    inside = [s[s < n_frames - 100] for s in spikes]  # short runs keep the early transients
    return np.array([planted_cell(rng, n_frames, s).fcorr for s in inside])


def write_movie(path, traces, scale):
    movie = np.full((traces.shape[1], LY, LX), NEUROPIL_LEVEL, dtype=np.float32)
    for k, (ys, xs) in enumerate(ROI_SLICES):
        movie[:, ys, xs] = (traces[k] + 0.7 * NEUROPIL_LEVEL)[:, None, None]
    np.round(movie * scale).astype(np.int16).tofile(path)


def outcome_table():
    frames = np.r_[REWARDED_FRAMES, UNREWARDED_FRAMES]
    return pd.DataFrame(
        {
            "outcome_time": FRAME_T0 + PERIOD_S * frames + 0.01,
            "num_rew_pulses": PULSES,
            "phase": "task",
        }
    )


def write_synthetic_run(derivatives: Path, n_frames: int = 2400, run: str = "run-01") -> dict:
    """Write the tree; return the paths a test needs."""
    run_dir = derivatives / SESSION / "funcimg" / run
    plane = run_dir / "without_destriping" / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    traces = planted_traces(n_frames)
    write_movie(plane / "data.bin", traces, 1.0)
    write_movie(plane / "data_chan2.bin", traces, 0.5)
    ops = {
        "Ly": LY, "Lx": LX, "fs": 1 / PERIOD_S, "nframes": n_frames, "nchannels": 2,
        "yrange": [2, LY - 2], "xrange": [2, LX - 2], "meanImg": texture().astype(np.float32),
        "xoff": np.zeros(n_frames), "yoff": np.zeros(n_frames),
    }  # fmt: skip
    np.save(plane / "ops.npy", ops, allow_pickle=True)
    seg_dir = run_dir / "suite2p" / "plane0"
    seg_dir.mkdir(parents=True)
    np.save(seg_dir / "meanImg_green_functional_contrast_seg.npy", {"masks": cell_labels()},
            allow_pickle=True)  # fmt: skip
    Image.fromarray((255 * texture()).astype(np.uint8)).save(
        seg_dir / "meanImg_green_functional_contrast.png"
    )
    time_s = FRAME_T0 + PERIOD_S * np.arange(n_frames)
    np.savez(
        run_dir / "frame_times.npz",
        time_s=time_s, frame_index=np.arange(n_frames), daq_frame_s=time_s - FRAME_T0,
        tiff="71rtask_00001.tif", h5="d.h5", tsv=str(run_dir / "behaviour.tsv"),
        quality=json.dumps({"fraction_matched": 1.0}),
    )  # fmt: skip
    (run_dir / "without_destriping" / "source_tiff.json").write_text(
        json.dumps({"source_tiff": "/raw/ses-224/funcimg/71rtask_00001.tif"})
    )
    return {"run_dir": run_dir, "plane_dir": plane, "traces": traces, "n_frames": n_frames}
