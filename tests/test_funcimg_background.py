"""Background (non-ROI) F of imaging runs: mask, chunked trace, cache, alignment, figures."""

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("mini2p_preprocessing")  # the optional funcimg extra

from src.funcimg_analysis import background_io  # noqa: E402
from src.funcimg_analysis.background import background_mask, chunked_mask_mean  # noqa: E402
from src.funcimg_analysis.background_io import BackgroundUnavailableError, load_background  # noqa: E402
from src.funcimg_analysis.collect import (  # noqa: E402
    collect_background_responses,
    collect_pulse_responses,
)
from src.funcimg_analysis.frame_times import FRAME_TIMES_FILE, RunMismatchError  # noqa: E402
from src.funcimg_visualization.plot_funcimg_background import (  # noqa: E402
    plot_background_mask,
    plot_background_responses,
)

LY, LX, N_FRAMES, PERIOD, N_CELLS = 20, 30, 600, 0.133, 4
YRANGE, XRANGE = (1, 19), (2, 28)  # the registration's valid crop
EVENT_FRAMES = np.array([100, 200, 300, 400, 500])
PULSES = [0, 2, 2, 4, 1]


def roi_pixels(cell):
    ypix, xpix = np.meshgrid(np.arange(2, 5), np.arange(3 + 6 * cell, 6 + 6 * cell))
    return ypix.ravel(), xpix.ravel()


def make_stat(n_cells=N_CELLS):
    return [
        {"ypix": roi_pixels(c)[0], "xpix": roi_pixels(c)[1], "npix": 9, "med": [3, 4]}
        for c in range(n_cells)
    ]


def literal_background():
    """The expected background written out by hand: the crop minus each 3 x 3 ROI rectangle.

    Spelled out from the numbers (not from ``background_mask``) so the I/O and alignment tests
    do not depend on the function whose result they check.
    """
    mask = np.zeros((LY, LX), dtype=bool)
    mask[1:19, 2:28] = True  # rows 1-18, columns 2-27: the registration crop
    for cell in range(N_CELLS):
        mask[2:5, 3 + 6 * cell : 6 + 6 * cell] = False
    return mask


def edit_ops(plane_dir, **changes):
    """Rewrite a plane's ``ops.npy`` with *changes*; a value of None removes the key."""
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    for key, value in changes.items():
        if value is None:
            del ops[key]
        else:
            ops[key] = value
    np.save(plane_dir / "ops.npy", ops, allow_pickle=True)


def bg_trace(rng, n_frames=N_FRAMES):
    """An integer trace with a response at each event, scaled by its pulse count."""
    trace = 200 + rng.integers(-2, 3, n_frames)
    for frame, pulses in zip(EVENT_FRAMES, PULSES, strict=True):
        trace[frame : frame + 4] += 40 * (pulses + 1)
    return trace


def write_movie(path, trace, mask, roi_value=5000):
    """An int16 ``(frames, Ly, Lx)`` movie: *trace* on background pixels, *roi_value* elsewhere."""
    movie = np.full((len(trace), LY, LX), roi_value, dtype=np.int16)
    movie[:, mask] = trace[:, None].astype(np.int16)
    movie.tofile(path)


def write_plane(plane_dir, rng, trace_green, trace_red):
    """A plane in the layout mini2p_preprocessing writes, with registered binaries."""
    plane_dir.mkdir(parents=True)
    stat = make_stat()
    mask = literal_background()
    write_movie(plane_dir / "data.bin", trace_green, mask)
    write_movie(plane_dir / "data_chan2.bin", trace_red, mask)
    ops = {
        "fs": 1 / PERIOD,
        "Ly": LY,
        "Lx": LX,
        "nframes": len(trace_green),
        "nchannels": 2,
        "yrange": list(YRANGE),
        "xrange": list(XRANGE),
        "meanImg": rng.normal(100, 5, (LY, LX)).astype(np.float32),
        "meanImg_chan2": rng.normal(50, 5, (LY, LX)).astype(np.float32),
    }
    f = np.vstack([trace_green] + [trace_green + 10 * (c + 1) for c in range(N_CELLS - 1)])
    red = np.vstack([trace_red] + [trace_red + 10 * (c + 1) for c in range(N_CELLS - 1)])
    # cell 0 of each channel carries the background's trace, so the paths can be compared
    arrays = {"F": f, "Fneu": f * 0.2, "F_chan2": red, "Fneu_chan2": red * 0.1}
    arrays.update(iscell=np.ones((N_CELLS, 2)), spks=np.zeros_like(f))
    for name, array in arrays.items():
        np.save(plane_dir / f"{name}.npy", array.astype(float))
    np.save(plane_dir / "stat.npy", np.array(stat, dtype=object), allow_pickle=True)
    np.save(plane_dir / "ops.npy", ops, allow_pickle=True)
    settings = {"neuropil_coefficient": 0.7, "baseline": "prctile", "prctile_baseline": 8.0}
    (plane_dir / "cellpose_extraction.json").write_text(
        json.dumps({"fluorescence_settings": settings})
    )
    return mask


def write_frame_times(run_dir):
    time_s = 20.0 + PERIOD * np.arange(N_FRAMES)
    np.savez(
        run_dir / FRAME_TIMES_FILE,
        time_s=time_s,
        frame_index=np.arange(N_FRAMES),
        daq_frame_s=time_s - 20.0,
        tiff="71rtask_00001.tif",
        h5="d.h5",
        tsv=str(run_dir / "behaviour.tsv"),
        quality=json.dumps({"fraction_matched": 1.0}),
    )


def outcomes_for(tsv_path):
    times = 20.0 + PERIOD * EVENT_FRAMES + 0.01
    return pd.DataFrame({"outcome_time": times, "num_rew_pulses": PULSES, "phase": "task"})


@pytest.fixture
def derivatives(tmp_path):
    rng = np.random.default_rng(0)
    funcimg = tmp_path / "der" / "ses-224_date-20260809" / "funcimg"
    for run in ("run-01", "run-02"):
        plane = funcimg / run / "without_destriping" / "suite2p" / "plane0"
        write_plane(plane, rng, bg_trace(rng), bg_trace(rng) // 2)
        write_frame_times(funcimg / run)
        (funcimg / run / "without_destriping" / "source_tiff.json").write_text(
            json.dumps({"source_tiff": "/raw/ses-224/funcimg/71rtask_00001.tif"})
        )
    return tmp_path / "der"


def plane_of(derivatives, run="run-01"):
    funcimg = derivatives / "ses-224_date-20260809" / "funcimg"
    return funcimg / run / "without_destriping" / "suite2p" / "plane0"


# --- mask ---------------------------------------------------------------------------------------


def test_the_mask_leaves_out_every_roi_pixel_cells_and_non_cells_alike():
    stat = make_stat()
    mask = background_mask(stat, (LY, LX), (0, LY), (0, LX))
    for roi in stat:
        assert not mask[roi["ypix"], roi["xpix"]].any()
    assert mask.sum() == LY * LX - N_CELLS * 9


def test_overlapping_rois_are_excluded_in_full():
    stat = make_stat(2)
    stat[1]["ypix"], stat[1]["xpix"] = stat[0]["ypix"] + 1, stat[0]["xpix"] + 1  # overlaps ROI 0
    stat[0]["overlap"] = np.zeros(9, bool)
    stat[1]["overlap"] = np.ones(9, bool)  # Suite2p's flag; the pixels still belong to the ROI
    mask = background_mask(stat, (LY, LX), (0, LY), (0, LX))
    for roi in stat:
        assert not mask[roi["ypix"], roi["xpix"]].any()


def test_pixels_outside_the_registration_crop_are_not_background():
    mask = background_mask([], (LY, LX), YRANGE, XRANGE)
    assert mask.sum() == (YRANGE[1] - YRANGE[0]) * (XRANGE[1] - XRANGE[0])
    assert not mask[: YRANGE[0]].any() and not mask[YRANGE[1] :].any()
    assert not mask[:, : XRANGE[0]].any() and not mask[:, XRANGE[1] :].any()


def test_a_margin_removes_a_ring_around_each_roi():
    stat = make_stat(1)
    plain = background_mask(stat, (LY, LX), (0, LY), (0, LX))
    grown = background_mask(stat, (LY, LX), (0, LY), (0, LX), roi_margin_px=1)
    assert (~grown).sum() == 5 * 5  # the 3 x 3 ROI plus one pixel all round
    assert not (grown & ~plain).any()


@pytest.mark.parametrize("bad", [{"roi_margin_px": -1}, {"yrange": (5, 5)}, {"yrange": (0, 99)}])
def test_nonsense_settings_raise(bad):
    kwargs = {"yrange": (0, LY), "xrange": (0, LX), **bad}
    with pytest.raises(ValueError):
        background_mask(make_stat(), (LY, LX), **kwargs)


def test_a_roi_outside_the_image_raises():
    stat = make_stat(1)
    stat[0]["xpix"] = stat[0]["xpix"] + 100
    with pytest.raises(ValueError, match="outside"):
        background_mask(stat, (LY, LX), (0, LY), (0, LX))


def test_a_mask_with_no_pixels_raises():
    stat = [{"ypix": np.arange(LY).repeat(LX), "xpix": np.tile(np.arange(LX), LY)}]
    with pytest.raises(ValueError, match="no background pixels"):
        background_mask(stat, (LY, LX), (0, LY), (0, LX))


# --- chunked mean -------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 7, 100, 599, 600, 5000])
def test_the_chunked_mean_equals_the_mean_of_the_whole_movie(tmp_path, chunk):
    rng = np.random.default_rng(1)
    movie = rng.integers(0, 3000, (N_FRAMES, LY, LX)).astype(np.int16)
    mask = background_mask(make_stat(), (LY, LX), YRANGE, XRANGE)
    path = tmp_path / "data.bin"
    movie.tofile(path)
    mapped = np.memmap(path, dtype=np.int16, mode="r", shape=movie.shape)
    expected = movie[:, mask].astype(np.float64).mean(axis=1)
    np.testing.assert_allclose(chunked_mask_mean(mapped, mask, chunk), expected, rtol=1e-12)


def test_a_chunk_size_below_one_raises():
    with pytest.raises(ValueError, match="chunk_frames"):
        chunked_mask_mean(np.zeros((3, LY, LX), np.int16), np.ones((LY, LX), bool), 0)


def test_a_mask_of_the_wrong_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        chunked_mask_mean(np.zeros((3, LY, LX), np.int16), np.ones((LY, LX + 1), bool), 2)


# --- loading and the cache ----------------------------------------------------------------------


def test_load_background_reads_only_background_pixels_of_both_channels(derivatives, tmp_path):
    plane = plane_of(derivatives)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    bg = load_background(plane, cache_dir=tmp_path / "cache")
    expected = literal_background()
    assert bg["n_pixels"] == expected.sum()
    np.testing.assert_array_equal(bg["mask"], expected)
    green = np.fromfile(plane / "data.bin", np.int16).reshape(N_FRAMES, LY, LX)
    np.testing.assert_allclose(bg["F"][("green", "F")], green[:, expected].mean(axis=1))
    assert bg["F"][("red", "F")].shape == (N_FRAMES,)
    np.testing.assert_array_equal(bg["mean_images"]["green"], ops["meanImg"])
    assert bg["labels"].shape == (LY, LX) and bg["labels"].max() == N_CELLS


def test_a_binary_of_the_wrong_size_raises(derivatives, tmp_path):
    plane = plane_of(derivatives)
    with open(plane / "data.bin", "ab") as handle:
        handle.write(b"\0\0")
    with pytest.raises(ValueError, match="data.bin"):
        load_background(plane, cache_dir=tmp_path / "cache")


def test_a_missing_binary_raises(derivatives, tmp_path):
    plane = plane_of(derivatives)
    (plane / "data_chan2.bin").unlink()
    with pytest.raises(FileNotFoundError, match="data_chan2.bin"):
        load_background(plane, cache_dir=tmp_path / "cache")


def test_a_second_load_comes_from_the_cache_and_writes_only_there(
    derivatives, tmp_path, monkeypatch
):
    plane, cache = plane_of(derivatives), tmp_path / "cache"
    before = sorted(p.name for p in plane.iterdir())
    first = load_background(plane, cache_dir=cache)

    def forbidden(*args, **kwargs):
        raise AssertionError("the binaries were read again")

    monkeypatch.setattr(background_io, "chunked_mask_mean", forbidden)
    second = load_background(plane, cache_dir=cache)
    np.testing.assert_array_equal(first["F"][("green", "F")], second["F"][("green", "F")])
    assert sorted(p.name for p in plane.iterdir()) == before  # nothing written beside the data
    assert list(cache.glob("*.npz"))


def test_editing_a_binary_a_margin_or_the_code_recomputes(derivatives, tmp_path, monkeypatch):
    plane, cache = plane_of(derivatives), tmp_path / "cache"
    first = load_background(plane, cache_dir=cache)["F"][("green", "F")]
    calls = []
    real = background_io.chunked_mask_mean

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(background_io, "chunked_mask_mean", counting)

    load_background(plane, cache_dir=cache)
    assert not calls  # unchanged inputs: cached

    load_background(plane, roi_margin_px=1, cache_dir=cache)
    assert len(calls) == 2  # green and red

    movie = np.fromfile(plane / "data.bin", np.int16).reshape(N_FRAMES, LY, LX)
    movie[:, literal_background()] += 7
    movie.tofile(plane / "data.bin")  # same size, new mtime
    stat_time = os.stat(plane / "data.bin").st_mtime_ns
    os.utime(plane / "data.bin", ns=(stat_time + 10**9, stat_time + 10**9))
    calls.clear()
    changed = load_background(plane, cache_dir=cache)["F"][("green", "F")]
    assert len(calls) == 2
    np.testing.assert_allclose(changed, first + 7)

    calls.clear()
    monkeypatch.setattr(background_io, "CODE_VERSION", "another version")
    load_background(plane, cache_dir=cache)
    assert len(calls) == 2


@pytest.mark.parametrize("name", ["stat.npy", "ops.npy"])
def test_rewriting_the_rois_or_ops_recomputes(derivatives, tmp_path, monkeypatch, name):
    plane, cache = plane_of(derivatives), tmp_path / "cache"
    load_background(plane, cache_dir=cache)
    calls = []
    real = background_io.chunked_mask_mean
    monkeypatch.setattr(
        background_io, "chunked_mask_mean", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    stat = np.load(plane / "stat.npy", allow_pickle=True)
    if name == "stat.npy":
        np.save(plane / name, stat[:-1], allow_pickle=True)  # one ROI fewer
    else:
        edit_ops(plane, xrange=[3, 28])
    load_background(plane, cache_dir=cache)
    assert len(calls) == 2


def test_both_binaries_are_checked_before_either_is_read(derivatives, tmp_path, monkeypatch):
    plane = plane_of(derivatives)
    with open(plane / "data_chan2.bin", "ab") as handle:
        handle.write(b"\0\0")  # only the second binary is the wrong size

    def forbidden(*args, **kwargs):
        raise AssertionError("a binary was read before the other was checked")

    monkeypatch.setattr(background_io, "chunked_mask_mean", forbidden)
    with pytest.raises(RunMismatchError, match="data_chan2.bin"):
        load_background(plane, cache_dir=tmp_path / "cache")


def test_a_mean_image_of_the_wrong_shape_is_unavailable_with_the_reason(derivatives, tmp_path):
    plane = plane_of(derivatives)
    edit_ops(plane, meanImg_chan2=np.zeros((LY, LX + 1)))
    with pytest.raises(BackgroundUnavailableError, match=r"meanImg_chan2.*\(20, 31\)"):
        load_background(plane, cache_dir=tmp_path / "cache")


@pytest.mark.parametrize("key", ["meanImg", "yrange", "xrange"])
def test_an_ops_file_missing_a_needed_key_is_unavailable_with_the_reason(
    derivatives, tmp_path, key
):
    plane = plane_of(derivatives)
    edit_ops(plane, **{key: None})
    with pytest.raises(BackgroundUnavailableError, match=key):
        load_background(plane, cache_dir=tmp_path / "cache")


# --- reward-aligned background ------------------------------------------------------------------


def collect_background(derivatives, cache, **kwargs):
    return collect_background_responses(
        derivatives,
        224,
        224,
        "without_destriping",
        (-1.0, 2.0),
        ("task",),
        outcomes_for,
        cache_dir=cache,
        **kwargs,
    )


def test_the_background_is_aligned_exactly_like_a_cell_with_the_same_trace(derivatives, tmp_path):
    pooled, runs, skipped = collect_background(derivatives, tmp_path / "cache", runs=("run-01",))
    cells, cell_runs, _ = collect_pulse_responses(
        derivatives, 224, 224, "without_destriping", (-1.0, 2.0), ("task",), outcomes_for,
        runs=("run-01",),
    )  # fmt: skip
    assert skipped == []
    assert runs[0]["offsets"].tolist() == cell_runs[0]["offsets"].tolist()
    assert runs[0]["n_trials"] == cell_runs[0]["n_trials"]
    assert runs[0]["n_events_dropped"] == cell_runs[0]["n_events_dropped"]
    # cell 0's F is the background trace in each channel (see write_plane)
    for view in (("green", "F"), ("green", "F z"), ("red", "F"), ("red", "F z")):
        assert runs[0]["means"][view]
        for pulses, mean in runs[0]["means"][view].items():
            expected = cell_runs[0]["means"][view][pulses][0]
            np.testing.assert_allclose(mean[0], expected, atol=1e-9)
    assert set(pooled) == {("green", "F"), ("green", "F z"), ("red", "F"), ("red", "F z")}
    assert runs[0]["n_pixels"] == runs[0]["mask"].sum()


def test_runs_are_pooled_one_trace_each(derivatives, tmp_path):
    pooled, runs, _ = collect_background(derivatives, tmp_path / "cache")
    assert [r["run"] for r in runs] == ["run-01", "run-02"]
    stats = pooled[("green", "F z")][2]
    assert stats["n_runs"] == 2 and stats["n_cells"] == 2
    stack = np.vstack([r["means"][("green", "F z")][2] for r in runs])
    np.testing.assert_allclose(stats["mean"], stack.mean(axis=0))
    np.testing.assert_allclose(stats["sem"], stack.std(axis=0, ddof=1) / np.sqrt(2))


def test_a_run_without_its_binaries_is_left_out_with_the_reason(derivatives, tmp_path):
    (plane_of(derivatives, "run-02") / "data.bin").unlink()
    _, runs, skipped = collect_background(derivatives, tmp_path / "cache")
    assert [r["run"] for r in runs] == ["run-01"]
    assert skipped[0][1] == "run-02" and "data.bin" in skipped[0][2]


def test_a_run_with_the_wrong_frame_count_is_left_out(derivatives, tmp_path):
    run_dir = derivatives / "ses-224_date-20260809" / "funcimg" / "run-02"
    time_s = np.load(run_dir / FRAME_TIMES_FILE, allow_pickle=True)
    arrays = {k: time_s[k] for k in time_s.files}
    arrays["time_s"] = np.append(arrays["time_s"], arrays["time_s"][-1] + PERIOD)
    np.savez(run_dir / FRAME_TIMES_FILE, **arrays)
    _, runs, skipped = collect_background(derivatives, tmp_path / "cache")
    assert [r["run"] for r in runs] == ["run-01"]
    assert "frames" in skipped[0][2]


def test_raw_only_measures_give_no_z_views(derivatives, tmp_path):
    pooled, _, _ = collect_background(derivatives, tmp_path / "cache", measures=("F",))
    assert set(pooled) == {("green", "F"), ("red", "F")}


# --- figures ------------------------------------------------------------------------------------


def test_the_response_figure_has_both_channels_raw_z_and_the_cells_on_shared_z_limits(
    derivatives, tmp_path
):
    pooled, runs, _ = collect_background(derivatives, tmp_path / "cache")
    cells, _, _ = collect_pulse_responses(
        derivatives, 224, 224, "without_destriping", (-1.0, 2.0), ("task",), outcomes_for,
        measures=("F z",),
    )  # fmt: skip
    cells[("red", "F z")] = {
        pulses: {**stats, "mean": 3 * stats["mean"], "sem": 3 * stats["sem"]}
        for pulses, stats in cells[("red", "F z")].items()
    }
    fig = plot_background_responses(
        pooled, cells, runs[0]["offsets"], runs[0]["period_s"], per_run=[r["means"] for r in runs]
    )
    assert len(fig.axes) == 6  # raw F, background z, cells z; green and red
    raw, z_bg, z_cells = fig.axes[0], fig.axes[2], fig.axes[4]
    for col in (0, 1):
        z_bg_col, z_cells_col = fig.axes[2 + col], fig.axes[4 + col]
        assert z_bg_col.get_ylim() == z_cells_col.get_ylim()
        data = np.concatenate(
            [line.get_ydata() for ax in (z_bg_col, z_cells_col) for line in ax.lines[:4]]
        )
        low, high = z_bg_col.get_ylim()
        assert low <= np.nanmin(data) and np.nanmax(data) <= high
    background_top = max(np.nanmax(line.get_ydata()) for line in fig.axes[3].lines[:4])
    assert fig.axes[3].get_ylim()[1] > 2 * background_top  # the 3x larger cells set the limits
    assert raw.get_title().startswith("Background") and z_cells.get_title().startswith("Cells")
    labels = fig.axes[1].get_legend().get_texts()
    assert all("2 runs" in text.get_text() and "cells" not in text.get_text() for text in labels)
    thick = [line for line in z_bg.lines if line.get_linewidth() >= 1]
    thin = [line for line in z_bg.lines if line.get_linewidth() < 1 and len(line.get_xdata()) > 2]
    assert len(thick) >= 4 and len(thin) == 2 * len(pooled[("green", "F z")])
    plt.close(fig)


def test_the_response_figure_without_cells_or_run_lines_has_two_rows(derivatives, tmp_path):
    pooled, runs, _ = collect_background(derivatives, tmp_path / "cache")
    fig = plot_background_responses(pooled, None, runs[0]["offsets"], runs[0]["period_s"])
    assert len(fig.axes) == 4
    plt.close(fig)


def test_the_overlay_marks_the_background_pixels_and_reports_how_many(derivatives, tmp_path):
    _, runs, _ = collect_background(derivatives, tmp_path / "cache", runs=("run-01",))
    run = runs[0]
    fig = plot_background_mask(
        run["mean_images"], run["mask"], run["labels"], title="ses-224 run-01"
    )
    assert len(fig.axes) == 2
    n = int(run["mask"].sum())
    assert (
        f"{n}" in fig.axes[0].get_title()
        and f"{100 * n / (LY * LX):.1f}%" in fig.axes[0].get_title()
    )
    overlay = fig.axes[0].images[1].get_array()
    assert overlay.shape == (LY, LX, 4)
    assert np.array_equal(overlay[..., 3] > 0, run["mask"])  # exactly the included pixels
    assert 0 < overlay[..., 3].max() < 1  # semi-transparent
    plt.close(fig)


def test_the_overlay_draws_into_given_axes(derivatives, tmp_path):
    _, runs, _ = collect_background(derivatives, tmp_path / "cache", runs=("run-01",))
    run = runs[0]
    fig, axes = plt.subplots(1, 2)
    out = plot_background_mask(run["mean_images"], run["mask"], run["labels"], axes=axes)
    assert out is fig
    plt.close(fig)


def test_the_response_figure_draws_into_given_axes(derivatives, tmp_path):
    pooled, runs, _ = collect_background(derivatives, tmp_path / "cache")
    fig, axes = plt.subplots(2, 2)
    out = plot_background_responses(
        pooled, None, runs[0]["offsets"], runs[0]["period_s"], axes=axes
    )
    assert out is fig
    wrong, bad_axes = plt.subplots(3, 2)
    with pytest.raises(ValueError, match="2 x 2"):
        plot_background_responses(
            pooled, None, runs[0]["offsets"], runs[0]["period_s"], axes=bad_axes
        )
    plt.close("all")


# --- one run: the SEM is across trials ---------------------------------------------------------


def test_with_one_run_the_sem_is_across_trials_and_the_figure_says_so(derivatives, tmp_path):
    pooled, runs, _ = collect_background(derivatives, tmp_path / "cache", runs=("run-01",))
    stats = pooled[("green", "F")][2]  # two trials of 2 pulses
    assert stats["sem_kind"] == "trials" and np.isfinite(stats["sem"]).all()
    # the trials are two near-identical responses, so the SEM is small but defined
    assert stats["sem"].shape == stats["mean"].shape
    fig = plot_background_responses(pooled, None, runs[0]["offsets"], runs[0]["period_s"])
    legend = " ".join(t.get_text() for t in fig.axes[1].get_legend().get_texts())
    assert "SEM across trials" in legend and "1 run" in legend
    plt.close(fig)


def test_with_several_runs_the_sem_stays_across_runs(derivatives, tmp_path):
    pooled, _, _ = collect_background(derivatives, tmp_path / "cache")
    assert all(
        "sem_kind" not in stats or stats["sem_kind"] == "runs"
        for per_pulse in pooled.values()
        for stats in per_pulse.values()
    )


def test_a_pulse_count_with_one_trial_has_no_trial_sem_to_show(derivatives, tmp_path):
    pooled, _, _ = collect_background(derivatives, tmp_path / "cache", runs=("run-01",))
    assert np.isnan(pooled[("green", "F")][4]["sem"]).all()  # a single 4-pulse trial


# --- runs that cannot contribute ----------------------------------------------------------------


def test_a_run_whose_ops_lack_the_crop_is_left_out_with_the_reason(derivatives, tmp_path):
    edit_ops(plane_of(derivatives, "run-02"), yrange=None)
    _, runs, skipped = collect_background(derivatives, tmp_path / "cache")
    assert [r["run"] for r in runs] == ["run-01"]
    assert skipped[0][1] == "run-02" and "yrange" in skipped[0][2]


def test_a_constant_background_is_reported_as_having_no_z(derivatives, tmp_path):
    plane = plane_of(derivatives, "run-02")
    flat = np.full(N_FRAMES, 200)
    write_movie(plane / "data.bin", flat, literal_background())
    _, runs, _ = collect_background(derivatives, tmp_path / "cache")
    assert runs[0]["flat_cells"][("green", "F z")] == []
    assert runs[1]["flat_cells"][("green", "F z")] == [0]


def test_a_run_with_no_z_does_not_count_in_the_legend(derivatives, tmp_path):
    write_movie(
        plane_of(derivatives, "run-02") / "data.bin", np.full(N_FRAMES, 200), literal_background()
    )
    pooled, _, _ = collect_background(derivatives, tmp_path / "cache")
    assert pooled[("green", "F z")][2]["n_runs"] == 1
    assert pooled[("green", "F")][2]["n_runs"] == 2
