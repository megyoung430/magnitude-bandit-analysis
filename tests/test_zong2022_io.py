"""Reading a registered plane, extracting traces with the Cellpose masks, caching, outputs."""

import json
import os

import numpy as np
import pytest

pytest.importorskip("mini2p_preprocessing")  # the optional funcimg extra

from src.funcimg_analysis import zong2022_io  # noqa: E402
from src.funcimg_analysis.zong2022_io import (  # noqa: E402
    aligned_labels,
    cached,
    extract_channels,
    file_signature,
    read_frames,
    registration_offsets,
    seg_paths,
    stat_from_labels,
    write_derivatives,
)

LY, LX, T = 70, 80, 40


def plane_ops(n_frames=T, **extra):
    return {
        "Ly": LY,
        "Lx": LX,
        "fs": 7.52,
        "nframes": n_frames,
        "nchannels": 2,
        "yrange": [2, LY - 2],
        "xrange": [3, LX - 3],
        **extra,
    }


def pixel_values():
    """A static image with a different value in every pixel."""
    y, x = np.mgrid[:LY, :LX]
    return (100 + 3 * y + x).astype(np.float32)


def write_plane(plane_dir, n_frames=T, red_scale=2.0):
    """Registered binaries where frame t is the pixel image plus t (red: scaled)."""
    plane_dir.mkdir(parents=True, exist_ok=True)
    movie = pixel_values()[None] + np.arange(n_frames, dtype=np.float32)[:, None, None]
    movie.astype(np.int16).tofile(plane_dir / "data.bin")
    (movie * red_scale).astype(np.int16).tofile(plane_dir / "data_chan2.bin")
    np.save(plane_dir / "ops.npy", plane_ops(n_frames, meanImg=pixel_values()), allow_pickle=True)
    return movie


def three_square_labels():
    labels = np.zeros((LY, LX), int)
    labels[10:18, 10:18] = 1
    labels[30:40, 30:38] = 2
    labels[50:58, 55:63] = 3
    return labels


@pytest.fixture
def plane(tmp_path):
    plane_dir = tmp_path / "run-01" / "without_destriping" / "suite2p" / "plane0"
    movie = write_plane(plane_dir)
    return plane_dir, movie


# --- extraction -----------------------------------------------------------------------------------


def test_cell_trace_is_the_weighted_mean_of_the_roi_pixels(plane):
    plane_dir, movie = plane
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    stat = stat_from_labels(three_square_labels(), ops, diameter_px=12)
    out = extract_channels(stat, ops, plane_dir)
    assert out["f"].shape == (3, T) and out["f_chan2"].shape == (3, T)
    for roi, entry in zip(range(3), stat, strict=True):
        direct = movie[:, entry["ypix"], entry["xpix"]].mean(axis=1)  # lam is 1 for every pixel
        np.testing.assert_allclose(out["f"][roi], direct, rtol=1e-4)
        np.testing.assert_allclose(out["f_chan2"][roi], 2 * direct, rtol=1e-4)


def test_neuropil_never_includes_roi_pixels(plane):
    plane_dir, _ = plane
    flat = np.full((T, LY, LX), 100, np.int16)
    labels = three_square_labels()
    flat[:, labels > 0] = 5000
    flat.tofile(plane_dir / "data.bin")
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    out = extract_channels(stat_from_labels(labels, ops, 12), ops, plane_dir)
    np.testing.assert_allclose(out["f"], 5000)
    np.testing.assert_allclose(out["fneu"], 100)


def test_pixels_flagged_as_overlapping_other_rois_are_left_out(plane):
    plane_dir, movie = plane
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    stat = stat_from_labels(three_square_labels(), ops, diameter_px=12)
    stat[0]["overlap"] = np.zeros(len(stat[0]["ypix"]), bool)
    stat[0]["overlap"][:7] = True
    out = extract_channels(stat, ops, plane_dir)
    keep = ~stat[0]["overlap"]
    expected = movie[:, stat[0]["ypix"][keep], stat[0]["xpix"][keep]].mean(axis=1)
    np.testing.assert_allclose(out["f"][0], expected, rtol=1e-4)
    everything = movie[:, stat[0]["ypix"], stat[0]["xpix"]].mean(axis=1)
    assert not np.allclose(expected, everything)  # the test would otherwise prove nothing


def test_a_binary_whose_length_disagrees_with_ops_is_refused(plane):
    plane_dir, _ = plane
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    ops["nframes"] = T + 5
    stat = stat_from_labels(three_square_labels(), ops, 12)
    with pytest.raises(ValueError, match="frames"):
        extract_channels(stat, ops, plane_dir)


def test_a_single_channel_plane_gives_no_red_traces(plane):
    plane_dir, _ = plane
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    ops["nchannels"] = 1
    out = extract_channels(stat_from_labels(three_square_labels(), ops, 12), ops, plane_dir)
    assert out["f_chan2"] is None and out["fneu_chan2"] is None


# --- Cellpose masks -------------------------------------------------------------------------------


def test_seg_files_are_found_where_the_batch_extraction_reads_them(tmp_path):
    run_dir = tmp_path / "run-01"
    seg, image = seg_paths(run_dir)
    assert seg == run_dir / "suite2p/plane0/meanImg_green_functional_contrast_seg.npy"
    assert image == run_dir / "suite2p/plane0/meanImg_green_functional_contrast.png"


def test_masks_are_aligned_onto_the_variants_own_mean_image(tmp_path, plane):
    plane_dir, _ = plane
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    # a structured mean image, as it was drawn on in the run's own registration
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    image = gaussian_filter(rng.random((LY, LX)), 2.0)
    ops["meanImg"] = np.roll(image, (2, -3), axis=(0, 1)).astype(np.float32)  # the variant's view
    labels = three_square_labels()
    seg_file = tmp_path / "x_seg.npy"
    np.save(seg_file, {"masks": labels}, allow_pickle=True)
    mask_png = tmp_path / "x.png"
    from PIL import Image

    scaled = (255 * (image - image.min()) / np.ptp(image)).astype(np.uint8)
    Image.fromarray(scaled).save(mask_png)
    moved, report = aligned_labels(seg_file, mask_png, ops, diameter_px=12)
    assert (report["dy"], report["dx"]) == (2, -3)
    assert moved.shape == labels.shape and (moved > 0).sum() == (labels > 0).sum()


def test_masks_of_the_wrong_shape_are_refused(tmp_path, plane):
    plane_dir, _ = plane
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    seg_file = tmp_path / "x_seg.npy"
    np.save(seg_file, {"masks": np.ones((10, 10), int)}, allow_pickle=True)
    with pytest.raises(ValueError, match="shape"):
        aligned_labels(seg_file, tmp_path / "missing.png", ops, diameter_px=12)


# --- frames for the motion metrics ----------------------------------------------------------------


def test_read_frames_returns_the_requested_frames_cropped(plane):
    plane_dir, movie = plane
    indices = np.array([0, 7, 8, 30, 39])
    out = read_frames(plane_dir / "data.bin", (T, LY, LX), indices, (2, 60, 3, 70), chunk_frames=2)
    np.testing.assert_allclose(out, movie[indices][:, 2:60, 3:70])
    assert out.dtype == np.float32


def test_read_frames_refuses_a_file_that_does_not_match_the_shape(plane):
    plane_dir, _ = plane
    with pytest.raises(ValueError, match="size"):
        read_frames(plane_dir / "data.bin", (T + 1, LY, LX), np.array([0]), (0, LY, 0, LX))


def test_registration_offsets_returns_what_ops_has():
    ops = {"xoff": np.zeros(5), "yoff": np.ones(5), "xoff1": np.zeros((5, 4))}
    out = registration_offsets(ops)
    assert set(out) == {"xoff", "yoff", "xoff1"}


# --- cache and outputs ----------------------------------------------------------------------------


def test_cached_computes_once_per_key_and_reuses_the_result(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"value": np.arange(3)}

    first = cached(tmp_path, "extraction", {"a": 1}, compute)
    again = cached(tmp_path, "extraction", {"a": 1}, compute)
    other = cached(tmp_path, "extraction", {"a": 2}, compute)
    assert len(calls) == 2
    np.testing.assert_array_equal(first["value"], again["value"])
    assert other["value"].shape == (3,)
    assert sorted(p.parent.name for p in tmp_path.rglob("*.pkl")) == ["extraction", "extraction"]


def test_the_cache_key_changes_when_a_file_is_rewritten(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"0123")
    before = file_signature(path)
    path.write_bytes(b"0123")
    os.utime(path, ns=(10**18, 10**18))
    assert file_signature(path) != before
    assert file_signature(path)["size"] == 4


def test_derivatives_are_written_beside_but_never_into_suite2p(tmp_path):
    out = write_derivatives(tmp_path / "without_destriping", {"dff": np.zeros((2, 3))}, {"k": 1})
    assert out == tmp_path / "without_destriping" / "zong2022"
    assert (out / "traces.npz").is_file() and json.loads((out / "settings.json").read_text()) == {
        "k": 1
    }
    with pytest.raises(ValueError, match="suite2p"):
        write_derivatives(tmp_path / "x" / "suite2p" / "plane0", {"dff": np.zeros(1)}, {})


def test_extraction_cache_is_keyed_on_the_inputs_files(tmp_path, plane, monkeypatch):
    plane_dir, _ = plane
    calls = []
    real = zong2022_io.extract_channels

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(zong2022_io, "extract_channels", counting)
    cache = tmp_path / "cache"
    labels = three_square_labels()
    first = zong2022_io.extract_cached(labels, plane_dir, 12, cache)
    zong2022_io.extract_cached(labels, plane_dir, 12, cache)
    assert len(calls) == 1
    assert first["f"].shape == (3, T)
    stamp = os.stat(plane_dir / "data.bin").st_mtime_ns + 10**9
    os.utime(plane_dir / "data.bin", ns=(stamp, stamp))
    zong2022_io.extract_cached(labels, plane_dir, 12, cache)
    assert len(calls) == 2
